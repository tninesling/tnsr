//! Fixed bindings for generated, homogeneous subgraphs inside a typed island.
use super::{Binding, Buffer};
use crate::{
    graph::{NodeIndex, TensorGraph, TensorGraphNode},
    ptx::{CudaDType, PtxExecutor, PtxPlanAction, PtxReductionMode},
    tile::{MatMulPrecision, TileDType},
};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

mod contiguous_bias_reshape;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ProgramKernelOptions {
    pub indexed_loss_cooperative_ordered_backward: bool,
    pub indexed_loss_cooperative_ordered_forward: bool,
    pub shared_memory_reuse: bool,
    pub narrow_bf16_tile: bool,
    pub bf16_eight_compute_warps: bool,
    pub ordinary_bf16_four_warp_block: bool,
    pub plain_bf16_output: bool,
    pub contiguous_bias_reshape_aliases: bool,
    pub medium_bf16_tile_policy: crate::tile::MediumBf16TilePolicy,
    pub column_reduction_policy: crate::tile::ColumnReductionPolicy,
    pub large_bf16_tile_policy: crate::tile::LargeBf16TilePolicy,
    pub batched_attention_bf16_tile_policy: crate::tile::BatchedAttentionBf16TilePolicy,
}

#[derive(Clone)]
pub(crate) struct Kernel {
    /// Stable original semantic operation; preserved by clones and late rewrites.
    pub owner: Option<NodeIndex>,
    pub label: String,
    pub function: CudaFunction,
    pub config: LaunchConfig,
    pub pointers: Vec<u64>,
    pub zero_before: Option<Binding>,
}
impl Kernel {
    pub fn run(&self, stream: &Arc<CudaStream>) -> Result<()> {
        if let Some(target) = self.zero_before {
            stream.context().bind_to_thread()?;
            // SAFETY: the program retains the exact validated allocation;
            // zero and scatter are ordered on the same captured stream.
            unsafe {
                cudarc::driver::result::memset_d8_async(
                    target.pointer,
                    0,
                    target
                        .len
                        .checked_mul(4)
                        .context("scatter zero size overflow")?,
                    stream.cu_stream(),
                )
            }?;
        }
        let mut launch = stream.launch_builder(&self.function);
        for pointer in &self.pointers {
            launch.arg(pointer);
        }
        // SAFETY: constructors validate the compiled ABI's dtype/extents. Private
        // execution owners construct kernels. Each retains all buffers/modules,
        // uses one private ordered stream and synchronizes before destruction.
        unsafe { launch.launch(self.config) }.with_context(|| {
            format!(
                "linear island kernel {} (graph owner {:?})",
                self.label, self.owner
            )
        })?;
        Ok(())
    }
}

pub(crate) struct ProgramMetadata {
    pub plan: crate::ptx::PtxExecutionPlan,
    /// Original plan-step index for each emitted kernel, in order.
    pub kernel_steps: Vec<usize>,
}

pub(crate) struct Program<D: CudaDType> {
    pub contiguous_bias_reshape_alias_steps: usize,
    pub contiguous_bias_reshape_fallback_steps: usize,
    pub contiguous_bias_reshape_avoided_bytes: usize,
    pub indexed_loss_cooperative_ordered_backward_programs: usize,
    pub indexed_loss_cooperative_ordered_forward_programs: usize,
    pub metadata: Option<ProgramMetadata>,
    pub kernels: Vec<Kernel>,
    pub output: Binding,
    pub plan: String,
    pub ptx: String,
    pub values: HashMap<NodeIndex, Binding>,
    pub views: HashMap<NodeIndex, crate::tile::VirtualTensor>,
    pub root_fused_matmul: bool,
    pub ordinary_bf16_four_warp_count: usize,
    pub bf16_eight_compute_warps: bool,
    pub medium_bf16_tile_counts: (usize, usize),
    pub column_reduction_counts: (usize, usize),
    pub large_bf16_tile_counts: (usize, usize),
    /// Actual square kernels and requested square kernels restored to the
    /// complete baseline after compilation/resource qualification, respectively.
    pub large_square128_streamed_counts: (usize, usize),
    pub plain_bf16_output_counts: (usize, usize),
    pub batched_attention_bf16_tile_counts: crate::tile::BatchedAttentionTileCounts,
    // Inputs and optional root storage are owned by LinearTraining; other
    // outputs/intermediates must stay alive through all captures here.
    storage: Vec<CudaSlice<D::HostType>>,
}
impl<D: CudaDType> Program<D> {
    pub fn new(
        stream: &Arc<CudaStream>,
        graph: TensorGraph<D::HostType>,
        inputs: HashMap<&str, Binding>,
        root: Option<Binding>,
        precision: MatMulPrecision,
    ) -> Result<Self> {
        let bindings = graph
            .graph
            .node_indices()
            .filter_map(|node| {
                if let TensorGraphNode::Input { name, .. } = &graph[node] {
                    inputs.get(name).copied().map(|value| (node, value))
                } else {
                    None
                }
            })
            .collect();
        Self::new_bound(stream, graph, bindings, root, precision)
    }

    pub fn new_bound(
        stream: &Arc<CudaStream>,
        graph: TensorGraph<D::HostType>,
        inputs: HashMap<NodeIndex, Binding>,
        root: Option<Binding>,
        precision: MatMulPrecision,
    ) -> Result<Self> {
        Self::new_bound_with_shared_memory_reuse(stream, graph, inputs, root, precision, false)
    }

    /// Select the existing scratch-allocation experiment before compilation.
    /// Schedule selection and arithmetic remain those of the ordinary executor.
    pub fn new_bound_with_shared_memory_reuse(
        stream: &Arc<CudaStream>,
        graph: TensorGraph<D::HostType>,
        inputs: HashMap<NodeIndex, Binding>,
        root: Option<Binding>,
        precision: MatMulPrecision,
        shared_memory_reuse: bool,
    ) -> Result<Self> {
        Self::new_bound_with_kernel_options(
            stream,
            graph,
            inputs,
            root,
            precision,
            ProgramKernelOptions {
                shared_memory_reuse,
                ..ProgramKernelOptions::default()
            },
        )
    }

    pub fn new_bound_with_kernel_options(
        stream: &Arc<CudaStream>,
        graph: TensorGraph<D::HostType>,
        inputs: HashMap<NodeIndex, Binding>,
        root: Option<Binding>,
        precision: MatMulPrecision,
        options: ProgramKernelOptions,
    ) -> Result<Self> {
        Self::new_bound_impl(stream, graph, inputs, root, precision, options, false)
    }
    pub(crate) fn new_bound_with_plan_metadata(
        stream: &Arc<CudaStream>,
        graph: TensorGraph<D::HostType>,
        inputs: HashMap<NodeIndex, Binding>,
        root: Option<Binding>,
        precision: MatMulPrecision,
        options: ProgramKernelOptions,
    ) -> Result<Self> {
        Self::new_bound_impl(stream, graph, inputs, root, precision, options, true)
    }
    fn new_bound_impl(
        stream: &Arc<CudaStream>,
        graph: TensorGraph<D::HostType>,
        inputs: HashMap<NodeIndex, Binding>,
        root: Option<Binding>,
        precision: MatMulPrecision,
        options: ProgramKernelOptions,
        retain_metadata: bool,
    ) -> Result<Self> {
        let mut executor = PtxExecutor::<D>::with_stream(
            stream.clone(),
            PtxReductionMode::DeterministicTree,
            Arc::new(Mutex::new(())),
        )?;
        executor.set_matmul_precision(precision);
        executor.set_experimental_indexed_loss_cooperative_ordered_backward(
            options.indexed_loss_cooperative_ordered_backward,
        );

        executor.set_experimental_indexed_loss_cooperative_ordered_forward(
            options.indexed_loss_cooperative_ordered_forward,
        );
        executor.set_experimental_shared_memory_reuse(options.shared_memory_reuse);
        executor.set_experimental_plain_bf16_output(options.plain_bf16_output);
        executor.set_experimental_narrow_bf16_tile(options.narrow_bf16_tile);
        executor.set_experimental_bf16_eight_compute_warps(options.bf16_eight_compute_warps);
        executor
            .set_experimental_ordinary_bf16_four_warp_block(options.ordinary_bf16_four_warp_block);
        executor.set_experimental_medium_bf16_tile_policy(options.medium_bf16_tile_policy);
        executor.set_experimental_column_reduction_policy(options.column_reduction_policy);
        executor.set_experimental_large_bf16_tile_policy(options.large_bf16_tile_policy);
        executor.set_experimental_batched_attention_bf16_tile_policy(
            options.batched_attention_bf16_tile_policy,
        );
        let physical_storage = inputs
            .iter()
            .filter_map(|(&node, binding)| {
                (binding.dtype == crate::tile::DType::BF16
                    && D::HostType::TILE_DTYPE == crate::tile::DType::F32)
                    .then_some((node, binding.dtype))
            })
            .collect();
        executor.compile_bound_storage(&graph, &physical_storage)?;
        let mut plan_text = if physical_storage.is_empty() {
            executor.describe_plan(&graph)?
        } else {
            format!(
                "{:#?}",
                executor
                    .execution_plan()
                    .context("linear plan absent")?
                    .steps()
            )
        };
        let ptx = executor.module_source().context("linear PTX absent")?;
        let plan = executor.execution_plan().context("linear plan absent")?;
        let root_node = plan.graph_output().context("linear root absent")?;
        let dtype = D::HostType::TILE_DTYPE;
        let mut values: HashMap<NodeIndex, Binding> = HashMap::new();
        let mut storage = Vec::new();
        let mut kernels = Vec::new();
        let mut kernel_steps = Vec::new();
        let copy_requests = if dtype == crate::tile::DType::BF16 {
            contiguous_bias_reshape::requests(&graph, plan)
        } else {
            0
        };
        let audit_alias =
            cfg!(test) && std::env::var_os("TNSR_CONTIGUOUS_BIAS_RESHAPE_AUDIT").is_some();
        let (alias_candidate, mut alias_reason) =
            if copy_requests > 0 && (options.contiguous_bias_reshape_aliases || audit_alias) {
                match contiguous_bias_reshape::prove(&graph, plan, dtype, root.is_some()) {
                    Ok(candidate) => (
                        options.contiguous_bias_reshape_aliases.then_some(candidate),
                        if options.contiguous_bias_reshape_aliases {
                            "proof_pass"
                        } else {
                            "proof_pass_census_only"
                        }
                        .to_string(),
                    ),
                    Err(reason) => (None, format!("{reason:#}")),
                }
            } else {
                (None, String::new())
            };
        let mut contiguous_bias_reshape_alias_steps = 0;
        let mut contiguous_bias_reshape_fallback_steps = if options.contiguous_bias_reshape_aliases
        {
            copy_requests
        } else {
            0
        };
        let mut contiguous_bias_reshape_avoided_bytes = 0;
        let mut applied_alias = None;
        for (step_index, step) in plan.steps().iter().enumerate() {
            if matches!(
                graph[step.node],
                TensorGraphNode::IndexSource { .. }
                    | TensorGraphNode::Input { .. }
                    | TensorGraphNode::Constant { .. }
                    | TensorGraphNode::Parameter { .. }
            ) {
                let shape = graph[step.node].shape();
                let value = *inputs
                    .get(&step.node)
                    .context("program source binding missing")?;
                ensure!(
                    value.len == shape.iter().product::<usize>()
                        && value.dtype
                            == if matches!(graph[step.node], TensorGraphNode::IndexSource { .. }) {
                                crate::tile::DType::U32
                            } else if let Some(&physical) = physical_storage.get(&step.node) {
                                physical
                            } else {
                                dtype
                            },
                    "linear source extent/dtype mismatch"
                );
                values.insert(step.node, value);
                continue;
            }
            if step.action == PtxPlanAction::VirtualView {
                ensure!(
                    step.virtual_outputs.len() == 1,
                    "unexpected linear multiple virtual outputs"
                );
                let value = *values
                    .get(&step.virtual_outputs[0].source)
                    .context("linear virtual source absent")?;
                values.insert(step.node, value);
                continue;
            }
            // Decide before allocation. A failed typed binding proof retains
            // the original complete copy+reduction path and owner set.
            if let Some(candidate) = alias_candidate.filter(|c| c.step == step_index) {
                match candidate.binding(&values) {
                    Ok(binding) => {
                        values.insert(candidate.output, binding);
                        contiguous_bias_reshape_alias_steps = 1;
                        contiguous_bias_reshape_fallback_steps = 0;
                        contiguous_bias_reshape_avoided_bytes = candidate
                            .len
                            .checked_mul(2)
                            .context("contiguous alias avoided byte count overflow")?;
                        applied_alias = Some(candidate);
                        continue;
                    }
                    Err(reason) => alias_reason = format!("{reason:#}"),
                }
            }
            for (&node, shape) in step.outputs.iter().zip(&step.output_shapes) {
                let len = shape.iter().product::<usize>();
                let binding = if node == root_node && root.is_some() {
                    root.context("linear external root absent")?
                } else {
                    let allocation = stream.alloc_zeros::<D::HostType>(len)?;
                    let binding = Binding {
                        pointer: allocation.device_ptr(stream).0,
                        len,
                        dtype,
                    };
                    storage.push(allocation);
                    binding
                };
                ensure!(
                    binding.len == len && binding.dtype == dtype,
                    "linear root extent/dtype mismatch"
                );
                values.insert(node, binding);
            }
            // The homogeneous executor implements these planned steps with a
            // DtoD copy: their diagnostic Tile IR is deliberately empty. The
            // fixed-binding program needs a real copy in its captured kernels.
            if step.action == PtxPlanAction::Kernel
                && matches!(
                    graph[step.node],
                    TensorGraphNode::Reshape { .. } | TensorGraphNode::Flatten { .. }
                )
                && step.materialized_view.is_none()
            {
                let source = *values
                    .get(&step.inputs[0])
                    .context("contiguous copy source missing")?;
                let target = *values
                    .get(&step.node)
                    .context("contiguous copy output missing")?;
                kernels.push(super::kernels::convert(stream, source, target, false)?);
                if retain_metadata {
                    kernel_steps.push(step_index);
                }
                continue;
            }
            let pointer = |node: &NodeIndex| -> Result<u64> {
                Ok(values
                    .get(node)
                    .context("linear physical binding absent")?
                    .pointer)
            };
            let pointers = step
                .inputs
                .iter()
                .chain(&step.outputs)
                .map(pointer)
                .collect::<Result<Vec<_>>>()?;
            let function = executor.planned_kernel_function(step)?;
            let config = match step.action {
                PtxPlanAction::MatMulRegion(id) => {
                    let r = &plan.matmul_regions()[id];
                    LaunchConfig {
                        grid_dim: (
                            u32::try_from(r.n.div_ceil(r.schedule.block_tile.n))?,
                            u32::try_from(r.m.div_ceil(r.schedule.block_tile.m))?,
                            u32::try_from(r.batch_shape.iter().product::<usize>())?,
                        ),
                        block_dim: r.schedule.block_threads,
                        shared_mem_bytes: 0,
                    }
                }
                PtxPlanAction::ReductionRegion(id) => {
                    let len = u32::try_from(step.shape.iter().product::<usize>())?;
                    let region = &plan.reduction_regions()[id];
                    crate::ptx::target::region_reduction_launch_config(region, len, &function)?
                }
                PtxPlanAction::Kernel
                    if matches!(graph[step.node], TensorGraphNode::MatMul { .. }) =>
                {
                    ensure!(
                        step.shape.len() == 2,
                        "unfused batched linear matmul is unsupported"
                    );
                    let s = &plan.matmul_schedules()[&step.node];
                    LaunchConfig {
                        grid_dim: (
                            u32::try_from(step.shape[1].div_ceil(s.block_tile.n))?,
                            u32::try_from(step.shape[0].div_ceil(s.block_tile.m))?,
                            1,
                        ),
                        block_dim: s.block_threads,
                        shared_mem_bytes: 0,
                    }
                }
                PtxPlanAction::Kernel
                    if matches!(graph[step.node], TensorGraphNode::EmbeddingBackward { .. })
                        && dtype == crate::tile::DType::F32 =>
                {
                    let ins = graph.inputs(step.node);
                    LaunchConfig::for_num_elems(u32::try_from(
                        graph[ins[1]].shape().iter().product::<usize>(),
                    )?)
                }
                PtxPlanAction::Kernel
                    if matches!(
                        graph[step.node],
                        TensorGraphNode::Conv2dBackwardWeight { .. }
                    ) =>
                {
                    executor
                        .conv_weight_tree_launch_config(step.node, step.shape.iter().product())?
                        .unwrap_or(crate::ptx::target::pointwise_launch_config(u32::try_from(
                            step.shape.iter().product::<usize>(),
                        )?))
                }
                PtxPlanAction::Kernel
                    if matches!(
                        graph[step.node],
                        TensorGraphNode::IndexedCrossEntropyBackward { .. }
                            | TensorGraphNode::IndexedCrossEntropy { .. }
                    ) =>
                {
                    executor
                        .indexed_loss_row_launch_config(step.node, step.shape.iter().product())?
                        .unwrap_or(crate::ptx::target::pointwise_launch_config(u32::try_from(
                            step.shape.iter().product::<usize>(),
                        )?))
                }
                PtxPlanAction::Kernel | PtxPlanAction::PointwiseRegion(_) => {
                    crate::ptx::target::pointwise_launch_config(u32::try_from(
                        step.shape.iter().product::<usize>(),
                    )?)
                }
                _ => anyhow::bail!("unsupported linear plan action {:?}", step.action),
            };
            let large_label = match step.action {
                PtxPlanAction::MatMulRegion(id) => {
                    let schedule = &plan.matmul_regions()[id].schedule;
                    if schedule.block_tile.m == 128 || schedule.block_tile.n == 128 {
                        format!(
                            " large_bf16_tile={}x{} stages={} arena={}",
                            schedule.block_tile.m,
                            schedule.block_tile.n,
                            schedule.pipeline_stages,
                            schedule.reuse_shared_memory
                        )
                    } else {
                        String::new()
                    }
                }
                _ => String::new(),
            };
            let plain_output_label = match step.action {
                PtxPlanAction::MatMulRegion(id) if executor.plain_bf16_output_selected(id) => {
                    " plain_bf16_output=true"
                }
                _ => "",
            };
            let four_warp_label = match step.action {
                PtxPlanAction::MatMulRegion(id)
                    if plan.matmul_regions()[id].schedule.block_threads == (32, 4, 1) =>
                {
                    " ordinary_bf16_four_warp_block=true"
                }
                PtxPlanAction::Kernel
                    if plan
                        .matmul_schedules()
                        .get(&step.node)
                        .is_some_and(|s| s.block_threads == (32, 4, 1)) =>
                {
                    " ordinary_bf16_four_warp_block=true"
                }
                _ => "",
            };
            let column_label = match step.action {
                PtxPlanAction::ReductionRegion(id) => plan.reduction_regions()[id]
                    .column_tile()
                    .map(|tile| {
                        format!(
                            " column_tile={} column_threads={}",
                            tile.columns_per_block, tile.threads
                        )
                    })
                    .unwrap_or_default(),
                _ => String::new(),
            };
            let indexed_loss_label =
                if executor.indexed_loss_cooperative_ordered_backward_selected(step.node) {
                    " indexed_loss_cooperative_ordered_backward=true"
                } else if executor.indexed_loss_cooperative_ordered_forward_selected(step.node) {
                    " indexed_loss_cooperative_ordered_forward=true"
                } else {
                    ""
                };
            let square_label = match step.action {
                PtxPlanAction::MatMulRegion(id)
                    if plan.matmul_regions()[id].schedule.is_square128_streamed() =>
                {
                    " square128_streamed=true exchange_windows=4 fragment_load=StreamB"
                }
                PtxPlanAction::Kernel
                    if plan
                        .matmul_schedules()
                        .get(&step.node)
                        .is_some_and(|s| s.is_square128_streamed()) =>
                {
                    " square128_streamed=true exchange_windows=4 fragment_load=StreamB"
                }
                _ => "",
            };
            kernels.push(Kernel {
                owner: None,
                label: format!(
                    "{:?} {} {:?} {} inputs={:?} output={:?}{}{}{}{}{}{}",
                    dtype,
                    step.operation,
                    step.action,
                    graph[step.node].name(),
                    step.inputs
                        .iter()
                        .map(|n| graph[*n].shape())
                        .collect::<Vec<_>>(),
                    step.shape,
                    column_label,
                    large_label,
                    four_warp_label,
                    plain_output_label,
                    indexed_loss_label,
                    square_label
                ),
                function,
                config,
                pointers,
                zero_before: if matches!(
                    graph[step.node],
                    TensorGraphNode::EmbeddingBackward { .. }
                ) && dtype == crate::tile::DType::F32
                {
                    Some(*values.get(&step.node).context("scatter output missing")?)
                } else {
                    None
                },
            });
            if retain_metadata {
                kernel_steps.push(step_index);
            }
        }
        let output = *values.get(&root_node).context("linear output absent")?;
        let mut views: HashMap<NodeIndex, crate::tile::VirtualTensor> = plan
            .steps()
            .iter()
            .filter(|s| s.action == PtxPlanAction::VirtualView)
            .flat_map(|s| {
                s.outputs
                    .iter()
                    .copied()
                    .zip(s.virtual_outputs.iter().cloned().map(|mut view| {
                        view.dtype = values[&view.source].dtype;
                        view
                    }))
            })
            .collect();
        if let Some(alias) = applied_alias {
            views.insert(
                alias.output,
                crate::tile::VirtualTensor::identity(
                    alias.source,
                    graph[alias.source].shape().clone(),
                    dtype,
                )
                .reshape(graph[alias.output].shape().clone())?,
            );
        }
        if options.contiguous_bias_reshape_aliases && copy_requests > 0 {
            plan_text.push_str(&format!("\nProgram contiguous bias reshape aliases={contiguous_bias_reshape_alias_steps}, fallback={contiguous_bias_reshape_fallback_steps}, avoided_bytes={contiguous_bias_reshape_avoided_bytes}; {alias_reason}. Original plan is proof-only for omitted copy steps."));
        }
        #[cfg(test)]
        contiguous_bias_reshape::record(
            &graph,
            plan,
            &values,
            copy_requests,
            contiguous_bias_reshape_alias_steps,
            &alias_reason,
        )?;
        let root_fused_matmul = plan.steps().iter().any(|s| {
            s.outputs.contains(&root_node) && matches!(s.action, PtxPlanAction::MatMulRegion(_))
        });
        let metadata = retain_metadata.then(|| ProgramMetadata {
            plan: plan.clone(),
            kernel_steps,
        });
        Ok(Self {
            contiguous_bias_reshape_alias_steps,
            contiguous_bias_reshape_fallback_steps,
            contiguous_bias_reshape_avoided_bytes,
            indexed_loss_cooperative_ordered_backward_programs: executor
                .indexed_loss_cooperative_ordered_backward_programs(),
            indexed_loss_cooperative_ordered_forward_programs: executor
                .indexed_loss_cooperative_ordered_forward_programs(),
            metadata,
            root_fused_matmul,
            ordinary_bf16_four_warp_count: plan.ordinary_bf16_four_warp_count(),
            medium_bf16_tile_counts: plan.medium_bf16_tile_counts(),
            column_reduction_counts: plan.column_reduction_counts(),
            large_bf16_tile_counts: plan.large_bf16_tile_counts(),
            large_square128_streamed_counts: executor.large_square128_streamed_counts(),
            plain_bf16_output_counts: executor.plain_bf16_output_counts(),
            batched_attention_bf16_tile_counts: plan.batched_attention_bf16_tile_counts(),
            bf16_eight_compute_warps: plan
                .matmul_schedules()
                .values()
                .chain(plan.matmul_regions().iter().map(|r| &r.schedule))
                .any(|s| {
                    s.plan == crate::tile::MatMulPlan::TensorCoreBF16
                        && s.warp_topology == (2, 4)
                        && s.warp_fragments == (2, 1)
                }),
            kernels,
            output,
            plan: plan_text,
            ptx,
            values,
            views,
            storage,
        })
    }
    /// Live private storage only; logical Values may outlive reused allocations.
    pub(crate) fn storage_bindings(&self, stream: &Arc<CudaStream>) -> Vec<Binding> {
        self.storage
            .iter()
            .map(|v| Binding {
                pointer: v.device_ptr(stream).0,
                len: v.len(),
                dtype: D::HostType::TILE_DTYPE,
            })
            .collect()
    }

    pub fn bytes(&self) -> usize {
        self.storage
            .iter()
            .map(|v| v.len() * std::mem::size_of::<D::HostType>())
            .sum()
    }
}

pub(crate) fn binding(buffer: &Buffer, stream: &Arc<CudaStream>) -> Binding {
    match buffer {
        Buffer::F32(v) => Binding {
            pointer: v.device_ptr(stream).0,
            len: v.len(),
            dtype: crate::tile::DType::F32,
        },
        Buffer::BF16(v) => Binding {
            pointer: v.device_ptr(stream).0,
            len: v.len(),
            dtype: crate::tile::DType::BF16,
        },
    }
}

impl Program<crate::ptx::F32> {
    pub(crate) fn drain_storage(&mut self) -> Vec<Buffer> {
        self.storage.drain(..).map(Buffer::F32).collect()
    }
}
impl Program<crate::ptx::BF16> {
    pub(crate) fn drain_storage(&mut self) -> Vec<Buffer> {
        self.storage.drain(..).map(Buffer::BF16).collect()
    }
}
