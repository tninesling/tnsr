//! Fixed bindings for generated, homogeneous subgraphs inside a typed island.
use super::{Binding, Buffer};
use crate::{
    graph::{NodeIndex, TensorGraph, TensorGraphNode},
    ptx::{CudaDType, PtxExecutor, PtxPlanAction, PtxReductionMode},
    tile::{MatMulPrecision, ReductionSchedule, TileDType},
};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

#[derive(Clone)]
pub(crate) struct Kernel {
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
        // SAFETY: constructors validate the compiled ABI's dtype/extents. Only
        // the island constructs kernels; it retains all buffers/modules, uses
        // one private ordered stream and synchronizes before destruction.
        unsafe { launch.launch(self.config) }.context("linear island kernel")?;
        Ok(())
    }
}

pub(crate) struct Program<D: CudaDType> {
    pub kernels: Vec<Kernel>,
    pub output: Binding,
    pub plan: String,
    pub ptx: String,
    pub values: HashMap<NodeIndex, Binding>,
    pub views: HashMap<NodeIndex, crate::tile::VirtualTensor>,
    pub root_fused_matmul: bool,
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
        let mut executor = PtxExecutor::<D>::with_stream(
            stream.clone(),
            PtxReductionMode::DeterministicTree,
            Arc::new(Mutex::new(())),
        )?;
        executor.set_matmul_precision(precision);
        let physical_storage = inputs
            .iter()
            .filter_map(|(&node, binding)| {
                (binding.dtype == crate::tile::DType::BF16
                    && D::HostType::TILE_DTYPE == crate::tile::DType::F32)
                    .then_some((node, binding.dtype))
            })
            .collect();
        executor.compile_bound_storage(&graph, &physical_storage)?;
        let plan_text = if physical_storage.is_empty() {
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
        for step in plan.steps() {
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
                    match plan.reduction_regions()[id].schedule {
                        ReductionSchedule::Serial => LaunchConfig::for_num_elems(len),
                        ReductionSchedule::Subgroup { width } => LaunchConfig {
                            grid_dim: (len, 1, 1),
                            block_dim: (width, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        ReductionSchedule::Block { threads } => LaunchConfig {
                            grid_dim: (len, 1, 1),
                            block_dim: (threads, 1, 1),
                            shared_mem_bytes: 0,
                        },
                    }
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
                PtxPlanAction::Kernel | PtxPlanAction::PointwiseRegion(_) => {
                    LaunchConfig::for_num_elems(u32::try_from(
                        step.shape.iter().product::<usize>(),
                    )?)
                }
                _ => anyhow::bail!("unsupported linear plan action {:?}", step.action),
            };
            kernels.push(Kernel {
                label: format!(
                    "{:?} {} {:?} {} inputs={:?} output={:?}",
                    dtype,
                    step.operation,
                    step.action,
                    graph[step.node].name(),
                    step.inputs
                        .iter()
                        .map(|n| graph[*n].shape())
                        .collect::<Vec<_>>(),
                    step.shape
                ),
                function: executor.planned_kernel_function(step)?,
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
        }
        let output = *values.get(&root_node).context("linear output absent")?;
        let views = plan
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
        let root_fused_matmul = plan.steps().iter().any(|s| {
            s.outputs.contains(&root_node) && matches!(s.action, PtxPlanAction::MatMulRegion(_))
        });
        Ok(Self {
            root_fused_matmul,
            kernels,
            output,
            plan: plan_text,
            ptx,
            values,
            views,
            storage,
        })
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
