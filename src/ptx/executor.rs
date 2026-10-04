use super::graph::PtxGraph;
use super::plan::{PtxExecutionPlan, PtxFusionPolicy, PtxPlanAction, PtxReductionMode};
use super::target::PtxTarget;
use super::types::{CudaDType, F32};
use crate::Executor;
use crate::alloc::{AllocStats, CudaBufferPool};
use crate::graph::{TensorGraph, TensorGraphNode, WithGrad};
use crate::tile::{
    AnalyticalFusionScorer, IndexMap, MatMulPipeline, MatMulSharedLayout, TileGraph, VirtualTensor,
};
use anyhow::{Context as _, Result};
use cudarc::driver::{CudaContext, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use num_traits::ToPrimitive;
use petgraph::visit::EdgeRef;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;
use std::time::{Duration, Instant};

type RegionValues<T> = Vec<(petgraph::graph::NodeIndex, Arc<CudaSlice<T>>)>;

// This versions the in-memory compilation-key schema, not the crate release.
// Bump it whenever signature encoding or generated-code-affecting inputs change.
const PTX_GRAPH_SIGNATURE_MAGIC: &[u8] = b"tnsr-ptx-graph";
const PTX_GRAPH_SIGNATURE_VERSION: u8 = 11;

/// Measurements from the most recent successful PTX compilation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PtxCompileMetrics {
    pub graph_nodes: usize,
    pub generated_kernels: usize,
    pub ptx_source_bytes: usize,
    pub compile_time: Duration,
    pub index_optimization_time: Duration,
    pub fusion_selection_time: Duration,
}

/// Measurements from the most recent successful graph execution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PtxExecutionMetrics {
    pub kernel_launches: usize,
    pub materialized_values: usize,
    pub materialized_bytes: usize,
    pub intermediate_materialized_bytes: usize,
    pub device_copies: usize,
}

/// PTX executor that compiles TensorGraph → TileGraph → PtxGraph → PTX string
/// and executes kernels via cudarc
pub struct PtxExecutor<D: CudaDType = F32> {
    device: Arc<CudaContext>,
    module: Option<Arc<CudaModule>>,
    values: HashMap<petgraph::graph::NodeIndex, Arc<CudaSlice<D::HostType>>>,
    pool: RefCell<CudaBufferPool<D::HostType>>,
    stats: AllocStats,
    /// Stores the PtxGraph to access kernel names during execution
    ptx_graph: Option<Box<PtxGraph>>,
    execution_plan: Option<Box<PtxExecutionPlan>>,
    compilation_signature: Option<Vec<u8>>,
    compilation_count: usize,
    compile_metrics: PtxCompileMetrics,
    execution_metrics: PtxExecutionMetrics,
    kernel_launches: Cell<usize>,
    reduction_mode: PtxReductionMode,
    fusion_policy: PtxFusionPolicy,
    target: PtxTarget,
    matmul_precision: crate::tile::MatMulPrecision,
    matmul_shared_layout: MatMulSharedLayout,
    matmul_pipeline: MatMulPipeline,
}

impl<D: CudaDType> Default for PtxExecutor<D> {
    fn default() -> Self {
        Self::new_for_dtype()
    }
}

impl PtxExecutor<F32> {
    /// Create a new PTX executor, panicking if CUDA initialization fails.
    ///
    /// For fallible initialization, use [`PtxExecutor::try_new`].
    pub fn new() -> Self {
        Self::new_for_dtype()
    }

    /// Create a PTX executor with an explicit reduction ordering policy.
    pub fn new_with_reduction_mode(reduction_mode: PtxReductionMode) -> Self {
        Self::try_new_with_reduction_mode(reduction_mode)
            .expect("Failed to initialize PTX executor")
    }

    /// Try to create a new PTX executor, returning an error if initialization fails.
    pub fn try_new() -> Result<Self> {
        Self::try_new_for_dtype()
    }
}

impl<D: CudaDType> PtxExecutor<D> {
    /// Create an executor for a specific storage dtype.
    pub fn new_for_dtype() -> Self {
        Self::try_new_for_dtype().expect("Failed to initialize PTX executor")
    }

    /// Try to create an executor for a specific storage dtype.
    pub fn try_new_for_dtype() -> Result<Self> {
        Self::try_new_with_reduction_mode(PtxReductionMode::Strict)
    }

    /// Initialize this storage dtype with an explicit reduction policy.
    pub fn try_new_with_reduction_mode(reduction_mode: PtxReductionMode) -> Result<Self> {
        let device = CudaContext::new(0).context("Failed to initialize CUDA device 0")?;
        let target = PtxTarget::from_context(&device)?;
        anyhow::ensure!(
            <D::HostType as crate::tile::TileDType>::TILE_DTYPE != crate::tile::DType::BF16
                || target.compute_capability.0 >= 8,
            "Native BF16 PTX storage requires SM80 or newer; use the CUDA backend for f32 boundary conversion on older GPUs"
        );
        Ok(PtxExecutor {
            device,
            module: None,
            values: HashMap::new(),
            pool: RefCell::new(CudaBufferPool::default()),
            stats: AllocStats::default(),
            ptx_graph: None,
            execution_plan: None,
            compilation_signature: None,
            compilation_count: 0,
            compile_metrics: PtxCompileMetrics::default(),
            execution_metrics: PtxExecutionMetrics::default(),
            kernel_launches: Cell::new(0),
            reduction_mode,
            fusion_policy: PtxFusionPolicy::CostAware,
            target,
            matmul_precision: crate::tile::MatMulPrecision::AllowTf32,
            matmul_shared_layout: MatMulSharedLayout::Auto,
            matmul_pipeline: MatMulPipeline::Auto,
        })
    }

    /// Select matmul arithmetic for subsequent compilations. Changing the policy
    /// invalidates the compilation signature so execute recompiles cached graphs.
    pub fn set_matmul_precision(&mut self, precision: crate::tile::MatMulPrecision) {
        if self.matmul_precision != precision {
            self.matmul_precision = precision;
            self.compilation_signature = None;
        }
    }

    /// Choose shared operand layouts for subsequent compilations.
    pub fn set_matmul_shared_layout(&mut self, layout: MatMulSharedLayout) {
        if self.matmul_shared_layout != layout {
            self.matmul_shared_layout = layout;
            self.compilation_signature = None;
        }
    }

    /// Select synchronous staging or a bounded asynchronous pipeline.
    pub fn set_matmul_pipeline(&mut self, pipeline: MatMulPipeline) {
        if self.matmul_pipeline != pipeline {
            self.matmul_pipeline = pipeline;
            self.compilation_signature = None;
        }
    }

    /// Use cost-aware selection or the previous greedy fusion for ablations.
    pub fn set_fusion_policy(&mut self, policy: PtxFusionPolicy) {
        if self.fusion_policy != policy {
            self.fusion_policy = policy;
            self.compilation_signature = None;
        }
    }

    fn take_buffer(&self, len: usize) -> Result<CudaSlice<D::HostType>> {
        self.pool
            .borrow_mut()
            .take(&self.device.default_stream(), len)
    }

    fn validate_indices(
        &self,
        indices: &CudaSlice<D::HostType>,
        upper: usize,
        operation: &str,
    ) -> Result<()> {
        let mut host_indices = vec![D::HostType::default(); indices.len()];
        if !host_indices.is_empty() {
            self.device
                .default_stream()
                .memcpy_dtoh(indices, &mut host_indices)
                .with_context(|| format!("Failed to validate PTX {operation} indices"))?;
        }
        for (position, value) in host_indices.iter().enumerate() {
            let value = value
                .to_f32()
                .context("Index cannot be represented as f32")?;
            anyhow::ensure!(
                value.is_finite(),
                "{operation} index at position {position} must be finite, got {value}"
            );
            anyhow::ensure!(
                value >= 0.0,
                "{operation} index at position {position} must be non-negative, got {value}"
            );
            anyhow::ensure!(
                value.fract() == 0.0,
                "{operation} index at position {position} must be an integer, got {value}"
            );
            let index = value as usize;
            anyhow::ensure!(
                index as f32 == value,
                "{operation} index at position {position} cannot be represented as usize: {value}"
            );
            anyhow::ensure!(
                index < upper,
                "{operation} index at position {position} is out of range: {index} >= {upper}"
            );
        }
        Ok(())
    }

    /// Memory allocation statistics gathered during execution.
    pub fn stats(&self) -> &AllocStats {
        &self.stats
    }

    /// Number of PTX modules successfully compiled and loaded by this executor.
    pub fn compilation_count(&self) -> usize {
        self.compilation_count
    }

    pub fn compile_metrics(&self) -> &PtxCompileMetrics {
        &self.compile_metrics
    }

    pub fn execution_metrics(&self) -> &PtxExecutionMetrics {
        &self.execution_metrics
    }

    pub fn execution_plan(&self) -> Option<&PtxExecutionPlan> {
        self.execution_plan.as_deref()
    }

    /// Return the generated PTX module for diagnostics after compilation.
    pub fn module_source(&self) -> Option<String> {
        self.ptx_graph.as_ref().map(|graph| graph.module_source())
    }

    /// Describe the current physical PTX execution plan.
    pub fn describe_plan<G>(&self, graph: &TensorGraph<D::HostType, G>) -> Result<String> {
        let supplied_signature = graph_compilation_signature(
            graph,
            self.reduction_mode,
            self.target,
            self.matmul_precision,
            self.matmul_shared_layout,
            self.matmul_pipeline,
            self.fusion_policy,
        );
        let compiled_signature = self
            .compilation_signature
            .as_deref()
            .context("No compiled graph signature available. Call compile_owned() first.")?;
        anyhow::ensure!(
            compiled_signature == supplied_signature.as_slice(),
            "Supplied graph structure does not match the compiled PTX module"
        );
        let ptx_graph = self
            .ptx_graph
            .as_ref()
            .context("No PTX graph available. Call compile_owned() first.")?;
        let mut description = format!(
            "PTX plan: {} graph nodes, {} physical kernels\n",
            graph.graph.node_count(),
            self.compile_metrics.generated_kernels
        );
        let plan = self
            .execution_plan
            .as_ref()
            .context("No PTX execution plan available. Call compile_owned() first.")?;
        for (step, plan_step) in plan.steps().iter().enumerate() {
            let node_index = plan_step.node;
            let action = match plan_step.action {
                PtxPlanAction::Upload => "upload".to_string(),
                PtxPlanAction::DeviceCopy => "device-copy".to_string(),
                PtxPlanAction::VirtualView => "virtual-view".to_string(),
                PtxPlanAction::Kernel => format!(
                    "kernel {}",
                    ptx_graph
                        .kernel_name(node_index)
                        .context("Missing compiled kernel in unfused plan")?
                ),
                PtxPlanAction::PointwiseRegion(region_id) => format!(
                    "kernel {}",
                    ptx_graph
                        .region_kernel_name(region_id)
                        .context("Missing compiled pointwise region kernel")?
                ),
                PtxPlanAction::ReductionRegion(region_id) => {
                    let schedule = plan
                        .reduction_regions()
                        .get(region_id)
                        .context("Missing reduction region schedule")?
                        .schedule;
                    format!(
                        "kernel {} schedule={schedule:?}",
                        ptx_graph
                            .reduction_region_kernel_name(region_id)
                            .context("Missing compiled reduction region kernel")?
                    )
                }
                PtxPlanAction::OnlineRegion(id) => format!(
                    "kernel {} schedule=Online",
                    ptx_graph
                        .region_kernel_name(plan.regions().len() + id)
                        .context("Missing online region kernel")?
                ),
                PtxPlanAction::MatMulRegion(region_id) => format!(
                    "kernel {} schedule={:?}",
                    ptx_graph
                        .matmul_region_kernel_name(region_id)
                        .context("Missing compiled matmul region kernel")?,
                    plan.matmul_regions()[region_id].schedule,
                ),
            };
            writeln!(
                description,
                "{step:04}: node {:04} {:<32} shape={:?} outputs={:?} action={action}",
                node_index.index(),
                plan_step.operation,
                plan_step.shape,
                plan_step.outputs
            )?;
        }
        Ok(description)
    }

    fn record_kernel_launch(&self) {
        self.kernel_launches.set(self.kernel_launches.get() + 1);
    }

    fn recycle_value(&self, value: Arc<CudaSlice<D::HostType>>) -> usize {
        let bytes = value.len() * std::mem::size_of::<D::HostType>();
        if let Ok(value) = Arc::try_unwrap(value) {
            self.pool.borrow_mut().give(value);
            bytes
        } else {
            0
        }
    }

    /// Reset allocation counters without discarding pooled device buffers.
    pub fn reset_stats(&mut self) {
        self.stats.reset();
        self.pool.get_mut().reset_stats();
    }

    /// Discard cached device buffers to release VRAM held by the executor.
    pub fn clear_pool(&mut self) {
        self.pool.get_mut().clear();
    }

    /// Release pinned gradient buffers before the next execution.
    pub fn release_gradients(&mut self, graph: &TensorGraph<D::HostType, WithGrad>) {
        for grad_node in graph.gradient_metadata().param_to_grad.values() {
            if let Some(buf) = self.values.remove(grad_node) {
                self.recycle_value(buf);
            }
        }
    }

    /// Compile an owned TensorGraph to PTX, sharing its tensor buffers during lowering.
    pub fn compile_owned<G>(&mut self, graph: TensorGraph<D::HostType, G>) -> Result<()> {
        self.compile(&graph)
    }

    /// Compile a borrowed graph without cloning tensor data.
    pub fn compile<G>(&mut self, graph: &TensorGraph<D::HostType, G>) -> Result<()> {
        let started = Instant::now();
        let graph_nodes = graph.graph.node_count();
        let signature = graph_compilation_signature(
            graph,
            self.reduction_mode,
            self.target,
            self.matmul_precision,
            self.matmul_shared_layout,
            self.matmul_pipeline,
            self.fusion_policy,
        );
        let mut execution_plan =
            PtxExecutionPlan::build_with_reduction_mode(graph, self.reduction_mode)?;
        execution_plan.schedule_matmuls(
            self.target.matmul_capabilities(),
            self.matmul_precision,
            self.target.matmul_resources,
            self.matmul_shared_layout,
            self.matmul_pipeline,
            self.target.supports_async_copy(),
        )?;
        let fusion_started = Instant::now();
        if self.fusion_policy == PtxFusionPolicy::CostAware {
            execution_plan.select_fusion(
                graph,
                &AnalyticalFusionScorer {
                    multiprocessor_count: self.target.multiprocessor_count,
                    registers_per_sm: self.target.matmul_resources.registers_per_sm,
                    shared_bytes_per_sm: self.target.matmul_resources.shared_bytes_per_sm,
                    threads_per_sm: self.target.matmul_resources.threads_per_sm,
                    ..AnalyticalFusionScorer::default()
                },
            )?;
        }
        if self.reduction_mode == PtxReductionMode::Online {
            execution_plan.fuse_online(graph)?;
        }
        let fusion_selection_time = fusion_started.elapsed();
        let mut tile_graph = TileGraph::from_with_matmul_schedules(
            graph,
            self.target.supports_tf32()
                && self.matmul_precision == crate::tile::MatMulPrecision::AllowTf32,
            execution_plan.matmul_schedules(),
        );
        tile_graph.add_fusion_regions(execution_plan.regions())?;
        tile_graph.add_reduction_regions(execution_plan.reduction_regions())?;
        tile_graph.add_matmul_regions(execution_plan.matmul_regions())?;
        // Only emit kernels selected by the final plan; fused-away regions retain
        // diagnostic metadata but must not bloat the loaded module.
        tile_graph.region_kernels.retain(|k| {
            execution_plan
                .steps()
                .iter()
                .any(|s| s.action == PtxPlanAction::PointwiseRegion(k.region_id))
        });
        tile_graph.reduction_region_kernels.retain(|k| {
            execution_plan
                .steps()
                .iter()
                .any(|s| s.action == PtxPlanAction::ReductionRegion(k.region_id))
        });
        tile_graph.matmul_region_kernels.retain(|k| {
            execution_plan
                .steps()
                .iter()
                .any(|s| s.action == PtxPlanAction::MatMulRegion(k.region_id))
        });
        for (id, region) in execution_plan.online_regions().iter().enumerate() {
            tile_graph
                .region_kernels
                .push(crate::tile::graph::TileRegionKernel {
                    region_id: execution_plan.regions().len() + id,
                    ir: region.lower_to_tile_ir(id)?,
                });
        }
        tile_graph.set_physical_nodes(
            execution_plan
                .steps()
                .iter()
                .filter_map(|step| (step.action == PtxPlanAction::Kernel).then_some(step.node))
                .collect(),
        );
        let index_started = Instant::now();
        tile_graph.optimize_indices()?;
        let index_optimization_time = index_started.elapsed();
        let ptx_graph = PtxGraph::from(tile_graph).with_target(self.target);
        let ptx_src = ptx_graph.module_source();
        let generated_kernels = execution_plan
            .steps()
            .iter()
            .filter(|step| {
                matches!(
                    step.action,
                    PtxPlanAction::Kernel
                        | PtxPlanAction::PointwiseRegion(_)
                        | PtxPlanAction::ReductionRegion(_)
                        | PtxPlanAction::MatMulRegion(_)
                        | PtxPlanAction::OnlineRegion(_)
                )
            })
            .count();
        let ptx_source_bytes = ptx_src.len();

        let cuda_module = self
            .device
            .load_module(Ptx::from_src(&ptx_src))
            .context("Failed to load PTX module with cudarc")?;
        self.module = Some(cuda_module);
        self.ptx_graph = Some(Box::new(ptx_graph));
        self.execution_plan = Some(Box::new(execution_plan));
        self.compilation_signature = Some(signature);
        self.compilation_count += 1;
        self.compile_metrics = PtxCompileMetrics {
            graph_nodes,
            generated_kernels,
            ptx_source_bytes,
            compile_time: started.elapsed(),
            index_optimization_time,
            fusion_selection_time,
        };

        Ok(())
    }

    /// Get the value of a specific node from the executor's cache after execution
    pub fn get_value(&self, node_idx: petgraph::graph::NodeIndex) -> Option<Vec<D::HostType>> {
        self.values.get(&node_idx).and_then(|cuda_slice| {
            let mut storage = vec![D::HostType::default(); cuda_slice.len()];
            self.device
                .default_stream()
                .memcpy_dtoh(cuda_slice.as_ref(), &mut storage)
                .ok()?;
            let plan = self.execution_plan.as_ref()?;
            let view = plan.virtual_value(node_idx)?;
            let source_shape = plan.value_shape(view.source)?;
            materialize_host_view(storage, view, source_shape).ok()
        })
    }

    fn execute_matmul(
        &self,
        node: crate::graph::NodeIndex,
        kernel_name: &str,
        a: &CudaSlice<D::HostType>,
        b: &CudaSlice<D::HostType>,
        shapes: (&[usize], &[usize], &[usize]),
    ) -> Result<CudaSlice<D::HostType>> {
        let (a_shape, b_shape, output_shape) = shapes;
        anyhow::ensure!(
            a_shape.len() >= 2 && b_shape.len() >= 2 && output_shape.len() >= 2,
            "MatMul requires rank >= 2, got {a_shape:?} and {b_shape:?}"
        );
        let m = a_shape[a_shape.len() - 2];
        let k = a_shape[a_shape.len() - 1];
        let b_k = b_shape[b_shape.len() - 2];
        let n = b_shape[b_shape.len() - 1];
        anyhow::ensure!(k == b_k, "MatMul K mismatch: {k} != {b_k}");
        anyhow::ensure!(
            output_shape[output_shape.len() - 2..] == [m, n],
            "MatMul output shape {output_shape:?} does not end in [{m}, {n}]"
        );

        let output_batch_shape = &output_shape[..output_shape.len() - 2];
        validate_batch_broadcast(a_shape, output_batch_shape)?;
        validate_batch_broadcast(b_shape, output_batch_shape)?;
        let output_batches: usize = output_batch_shape.iter().product();
        let output_len: usize = output_shape.iter().product();
        let mut output = self.take_buffer(output_len)?;
        if output_batches == 0 || m == 0 || n == 0 {
            return Ok(output);
        }

        let schedule = self
            .execution_plan
            .as_ref()
            .and_then(|plan| plan.matmul_schedules().get(&node))
            .context("Matmul schedule is unavailable")?;
        let stream = self.device.default_stream();
        if k == 0 {
            stream
                .memset_zeros(&mut output)
                .context("Failed to zero MatMul output with K=0")?;
            return Ok(output);
        }
        let module = self.module.as_ref().context("PTX module is not compiled")?;
        let function = module.load_function(kernel_name)?;
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(n.div_ceil(schedule.block_tile.n))
                    .context("Matmul grid width exceeds u32")?,
                u32::try_from(m.div_ceil(schedule.block_tile.m))
                    .context("Matmul grid height exceeds u32")?,
                1,
            ),
            block_dim: schedule.block_threads,
            shared_mem_bytes: 0,
        };

        for batch in 0..output_batches {
            let a_batch = batch_matrix_offset(batch, output_batch_shape, a_shape);
            let b_batch = batch_matrix_offset(batch, output_batch_shape, b_shape);
            let a_start = a_batch * m * k;
            let b_start = b_batch * k * n;
            let output_start = batch * m * n;
            let a_view = a.slice(a_start..a_start + m * k);
            let b_view = b.slice(b_start..b_start + k * n);
            let mut output_view = output.slice_mut(output_start..output_start + m * n);
            let mut launcher = stream.launch_builder(&function);
            launcher.arg(&a_view);
            launcher.arg(&b_view);
            launcher.arg(&mut output_view);
            self.record_kernel_launch();
            unsafe { launcher.launch(config) }
                .with_context(|| format!("CUDA {kernel_name} kernel launch failed"))?;
        }
        Ok(output)
    }

    fn execute_matmul_region(
        &self,
        module: &Arc<CudaModule>,
        ptx_graph: &PtxGraph,
        region_id: usize,
        step: &super::plan::PtxPlanStep,
    ) -> Result<RegionValues<D::HostType>> {
        let kernel_name = ptx_graph
            .matmul_region_kernel_name(region_id)
            .context("Missing compiled matmul region kernel")?;
        let region = self
            .execution_plan
            .as_ref()
            .and_then(|plan| plan.matmul_regions().get(region_id))
            .context("Matmul region metadata is unavailable")?;
        anyhow::ensure!(
            step.outputs
                == region
                    .outputs
                    .iter()
                    .map(|output| output.node)
                    .collect::<Vec<_>>(),
            "Matmul region output bindings do not match the execution plan"
        );
        let input_values: Vec<_> = step
            .inputs
            .iter()
            .map(|input| {
                self.values.get(input).cloned().with_context(|| {
                    format!("Matmul region input node {} is unavailable", input.index())
                })
            })
            .collect::<Result<_>>()?;
        anyhow::ensure!(
            input_values.len() == region.inputs.len(),
            "Matmul region input binding count mismatch"
        );
        for (input, descriptor) in input_values.iter().zip(&region.inputs) {
            let expected =
                checked_element_count(&descriptor.source_shape, "Matmul region input storage")?;
            anyhow::ensure!(
                input.len() == expected,
                "Matmul region input length {} does not match storage length {expected}",
                input.len()
            );
        }
        let output_len = checked_element_count(&region.output_shape, "Matmul region output")?;
        let mut outputs = Vec::with_capacity(step.outputs.len());
        for _ in &step.outputs {
            outputs.push(self.take_buffer(output_len)?);
        }

        let function = module.load_function(kernel_name)?;
        let stream = self.device.default_stream();
        let mut launcher = stream.launch_builder(&function);
        for input in &input_values {
            launcher.arg(input.as_ref());
        }
        for output in &mut outputs {
            launcher.arg(output);
        }
        self.record_kernel_launch();
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(region.n.div_ceil(region.schedule.block_tile.n))
                    .context("Matmul region grid width exceeds u32")?,
                u32::try_from(region.m.div_ceil(region.schedule.block_tile.m))
                    .context("Matmul region grid height exceeds u32")?,
                u32::try_from(checked_element_count(
                    &region.batch_shape,
                    "Matmul batch count",
                )?)
                .context("Matmul batch grid exceeds u32")?,
            ),
            block_dim: region.schedule.block_threads,
            shared_mem_bytes: 0,
        };
        unsafe { launcher.launch(config) }
            .with_context(|| format!("CUDA {kernel_name} kernel launch failed"))?;

        Ok(step
            .outputs
            .iter()
            .copied()
            .zip(outputs.into_iter().map(Arc::new))
            .collect())
    }

    fn execute_scalar_region(
        &self,
        module: &Arc<CudaModule>,
        ptx_graph: &PtxGraph,
        step: &super::plan::PtxPlanStep,
    ) -> Result<RegionValues<D::HostType>> {
        let plan = self
            .execution_plan
            .as_ref()
            .context("Scalar region plan unavailable")?;
        let (kernel_id, inputs, threads) = match step.action {
            PtxPlanAction::PointwiseRegion(id) => (id, &plan.regions()[id].inputs, None),
            PtxPlanAction::OnlineRegion(id) => {
                let region = &plan.online_regions()[id];
                (
                    plan.regions().len() + id,
                    &region.inputs,
                    region.block_threads(),
                )
            }
            _ => anyhow::bail!("Expected a scalar region step"),
        };
        let kernel_name = ptx_graph
            .region_kernel_name(kernel_id)
            .context("Scalar region kernel unavailable")?;
        let input_values: Vec<_> = step
            .inputs
            .iter()
            .map(|input| {
                self.values.get(input).cloned().with_context(|| {
                    format!("Scalar region input node {} is unavailable", input.index())
                })
            })
            .collect::<Result<_>>()?;
        let len = checked_element_count(&step.shape, "Scalar region")?;
        anyhow::ensure!(
            inputs.len() == input_values.len(),
            "Scalar region input binding count mismatch"
        );
        for (input, descriptor) in input_values.iter().zip(inputs) {
            let expected =
                checked_element_count(&descriptor.source_shape, "Scalar region input storage")?;
            anyhow::ensure!(
                input.len() == expected,
                "Scalar region input length {} does not match storage length {expected}",
                input.len(),
            );
        }
        let mut outputs = Vec::with_capacity(step.outputs.len());
        for _ in &step.outputs {
            outputs.push(self.take_buffer(len)?);
        }
        if len != 0 {
            let config = if let Some(threads) = threads {
                LaunchConfig {
                    grid_dim: (
                        u32::try_from(len / step.shape[step.shape.len() - 1])
                            .context("Scalar grid too large")?,
                        1,
                        1,
                    ),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                }
            } else {
                LaunchConfig::for_num_elems(u32::try_from(len).context("Scalar output too large")?)
            };
            let function = module.load_function(kernel_name)?;
            let stream = self.device.default_stream();
            let mut launcher = stream.launch_builder(&function);
            for input in &input_values {
                launcher.arg(input.as_ref());
            }
            for output in &mut outputs {
                launcher.arg(output);
            }
            self.record_kernel_launch();
            // SAFETY: Storage lengths and output extent are checked; generated
            // kernels guard output lanes and traverse valid input coordinates.
            unsafe { launcher.launch(config) }
                .with_context(|| format!("CUDA {kernel_name} kernel launch failed"))?;
        }
        Ok(step
            .outputs
            .iter()
            .copied()
            .zip(outputs.into_iter().map(Arc::new))
            .collect())
    }

    fn execute_reduction_region(
        &self,
        module: &Arc<CudaModule>,
        ptx_graph: &PtxGraph,
        region_id: usize,
        step: &super::plan::PtxPlanStep,
    ) -> Result<RegionValues<D::HostType>> {
        let kernel_name = ptx_graph
            .reduction_region_kernel_name(region_id)
            .context("Missing compiled reduction region kernel")?;
        let region = self
            .execution_plan
            .as_ref()
            .and_then(|plan| plan.reduction_regions().get(region_id))
            .context("Reduction region metadata is unavailable")?;
        anyhow::ensure!(
            step.outputs
                == region
                    .outputs
                    .iter()
                    .map(|output| output.node)
                    .collect::<Vec<_>>(),
            "Reduction region output bindings do not match the execution plan"
        );
        anyhow::ensure!(
            step.output_shapes == region.output_shapes,
            "Reduction region output shapes do not match the execution plan"
        );
        anyhow::ensure!(
            step.shape == region.output_shape,
            "Reduction region launch shape does not match the execution plan"
        );
        let input_values: Vec<_> = step
            .inputs
            .iter()
            .map(|input| {
                self.values.get(input).cloned().with_context(|| {
                    format!(
                        "Reduction region input node {} is unavailable",
                        input.index()
                    )
                })
            })
            .collect::<Result<_>>()?;
        anyhow::ensure!(
            region.inputs.len() == input_values.len(),
            "Reduction region input binding count mismatch"
        );
        for (input, descriptor) in input_values.iter().zip(&region.inputs) {
            let expected =
                checked_element_count(&descriptor.source_shape, "Reduction region input storage")?;
            anyhow::ensure!(
                input.len() == expected,
                "Reduction region input length {} does not match storage length {expected}",
                input.len(),
            );
        }
        let len = checked_element_count(&step.shape, "Reduction region")?;
        anyhow::ensure!(
            step.output_shapes.len() == step.outputs.len(),
            "Reduction region output shape binding count mismatch"
        );
        let mut outputs = Vec::with_capacity(step.outputs.len());
        for shape in &step.output_shapes {
            outputs
                .push(self.take_buffer(checked_element_count(shape, "Reduction region output")?)?);
        }
        if len != 0 {
            let launch_len = u32::try_from(len)
                .context("Reduction region output is too large for a CUDA launch")?;
            let function = module.load_function(kernel_name)?;
            let stream = self.device.default_stream();
            let mut launcher = stream.launch_builder(&function);
            for input in &input_values {
                launcher.arg(input.as_ref());
            }
            for output in &mut outputs {
                launcher.arg(output);
            }
            self.record_kernel_launch();
            let config = match region.schedule {
                crate::tile::ReductionSchedule::Serial => LaunchConfig::for_num_elems(launch_len),
                crate::tile::ReductionSchedule::Subgroup { width } => LaunchConfig {
                    grid_dim: (launch_len, 1, 1),
                    block_dim: (width, 1, 1),
                    shared_mem_bytes: 0,
                },
                crate::tile::ReductionSchedule::Block { threads } => LaunchConfig {
                    grid_dim: (launch_len, 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                },
            };
            unsafe { launcher.launch(config) }
                .with_context(|| format!("CUDA {kernel_name} kernel launch failed"))?;
        }
        Ok(step
            .outputs
            .iter()
            .copied()
            .zip(outputs.into_iter().map(Arc::new))
            .collect())
    }

    /// Execute a compiled graph with the given inputs
    pub fn execute_compiled<G>(
        &mut self,
        graph: &TensorGraph<D::HostType, G>,
        inputs: HashMap<String, Vec<D::HostType>>,
    ) -> Result<Vec<D::HostType>> {
        self.execution_metrics = PtxExecutionMetrics::default();
        self.kernel_launches.set(0);
        let supplied_signature = graph_compilation_signature(
            graph,
            self.reduction_mode,
            self.target,
            self.matmul_precision,
            self.matmul_shared_layout,
            self.matmul_pipeline,
            self.fusion_policy,
        );
        let compiled_signature = self
            .compilation_signature
            .as_deref()
            .context("No compiled graph signature available. Call compile_owned() first.")?;
        anyhow::ensure!(
            compiled_signature == supplied_signature,
            "Supplied graph structure does not match the compiled PTX module; compile the graph before execution"
        );
        let module = self
            .module
            .as_ref()
            .context("No compiled module available. Call compile() first.")?;

        let ptx_graph = self
            .ptx_graph
            .as_ref()
            .context("No PTX graph available. Call compile() first.")?;

        let execution_plan = self
            .execution_plan
            .as_ref()
            .context("No PTX execution plan available. Call compile_owned() first.")?;
        let plan_steps = execution_plan.steps().to_vec();
        let graph_output = execution_plan.graph_output().context("Graph is empty")?;
        let old_values: Vec<_> = self.values.drain().map(|(_, value)| value).collect();
        for value in old_values {
            self.recycle_value(value);
        }
        let mut live_bytes = 0usize;

        for plan_step in &plan_steps {
            let node_idx = &plan_step.node;
            let planned_inputs = &plan_step.inputs;
            let node = &graph[*node_idx];
            if let PtxPlanAction::MatMulRegion(region_id) = plan_step.action {
                let region_outputs =
                    self.execute_matmul_region(module, ptx_graph, region_id, plan_step)?;
                for (output_node, output) in region_outputs {
                    let bytes = output.len() * std::mem::size_of::<D::HostType>();
                    self.execution_metrics.materialized_values += 1;
                    self.execution_metrics.materialized_bytes += bytes;
                    if output_node != graph_output {
                        self.execution_metrics.intermediate_materialized_bytes += bytes;
                    }
                    self.values.insert(output_node, output);
                    live_bytes += bytes;
                }
                self.stats.record_live(live_bytes);
                for &dead in &plan_step.release_after {
                    if let Some(value) = self.values.remove(&dead) {
                        live_bytes -= self.recycle_value(value);
                    }
                }
                continue;
            }
            if matches!(
                plan_step.action,
                PtxPlanAction::PointwiseRegion(_) | PtxPlanAction::OnlineRegion(_)
            ) {
                let region_outputs = self.execute_scalar_region(module, ptx_graph, plan_step)?;
                for (output_node, output) in region_outputs {
                    let bytes = output.len() * std::mem::size_of::<D::HostType>();
                    self.execution_metrics.materialized_values += 1;
                    self.execution_metrics.materialized_bytes += bytes;
                    if output_node != graph_output {
                        self.execution_metrics.intermediate_materialized_bytes += bytes;
                    }
                    self.values.insert(output_node, output);
                    live_bytes += bytes;
                }
                self.stats.record_live(live_bytes);
                for &dead in &plan_step.release_after {
                    if let Some(value) = self.values.remove(&dead) {
                        live_bytes -= self.recycle_value(value);
                    }
                }
                continue;
            }
            if let PtxPlanAction::ReductionRegion(region_id) = plan_step.action {
                let region_outputs =
                    self.execute_reduction_region(module, ptx_graph, region_id, plan_step)?;
                for (output_node, output) in region_outputs {
                    let bytes = output.len() * std::mem::size_of::<D::HostType>();
                    self.execution_metrics.materialized_values += 1;
                    self.execution_metrics.materialized_bytes += bytes;
                    if output_node != graph_output {
                        self.execution_metrics.intermediate_materialized_bytes += bytes;
                    }
                    self.values.insert(output_node, output);
                    live_bytes += bytes;
                }
                self.stats.record_live(live_bytes);
                for &dead in &plan_step.release_after {
                    if let Some(value) = self.values.remove(&dead) {
                        live_bytes -= self.recycle_value(value);
                    }
                }
                continue;
            }
            if plan_step.action == PtxPlanAction::VirtualView {
                let input = planned_inputs
                    .first()
                    .context("virtual view plan step has no input")?;
                let result = Arc::clone(
                    self.values
                        .get(input)
                        .context("virtual view input value is unavailable")?,
                );
                self.values.insert(*node_idx, result);
                self.stats.record_live(live_bytes);
                for &dead in &plan_step.release_after {
                    if let Some(value) = self.values.remove(&dead) {
                        live_bytes -= self.recycle_value(value);
                    }
                }
                continue;
            }
            let result = match node {
                TensorGraphNode::Constant { data, .. } => {
                    let stream = self.device.default_stream();
                    let mut device_data = self.take_buffer(data.len())?;
                    if !data.is_empty() {
                        self.execution_metrics.device_copies += 1;
                        stream
                            .memcpy_htod(data.as_slice(), &mut device_data)
                            .context("Failed to copy constant to CUDA device")?;
                    }
                    device_data
                }
                TensorGraphNode::Input { name, shape } => {
                    let val = inputs
                        .get::<str>(name)
                        .with_context(|| format!("Input '{}' not found", name))?;
                    let expected_len = checked_element_count(shape, &format!("Input '{name}'"))?;
                    anyhow::ensure!(
                        val.len() == expected_len,
                        "Input '{name}' has {} values but declared shape {shape:?} requires {expected_len}",
                        val.len()
                    );
                    let stream = self.device.default_stream();
                    let mut device_data = self.take_buffer(val.len())?;
                    if !val.is_empty() {
                        self.execution_metrics.device_copies += 1;
                        stream
                            .memcpy_htod(val.as_slice(), &mut device_data)
                            .context("Failed to copy input to CUDA device")?;
                    }
                    device_data
                }
                TensorGraphNode::Parameter { data, .. } => {
                    use anyhow::anyhow;
                    let v = data
                        .lock()
                        .map_err(|e| anyhow!("Failed to lock parameter data: {}", e))?;
                    let stream = self.device.default_stream();
                    let mut device_data = self.take_buffer(v.len())?;
                    if !v.is_empty() {
                        self.execution_metrics.device_copies += 1;
                        stream
                            .memcpy_htod(v.as_slice(), &mut device_data)
                            .context("Failed to copy parameter to CUDA device")?;
                    }
                    device_data
                }
                TensorGraphNode::Unary { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled unary kernel")?;

                    let ins = planned_inputs;
                    let input = self
                        .values
                        .get(&ins[0])
                        .context("Missing input value for unary operation")?;

                    let len = input.len();
                    let stream = self.device.default_stream();
                    let mut out = self.take_buffer(len)?;
                    if len != 0 {
                        let f = module.load_function(kernel_name)?;
                        let cfg = LaunchConfig::for_num_elems(len as u32);
                        let mut launcher = stream.launch_builder(&f);
                        launcher.arg(input.as_ref());
                        launcher.arg(&mut out);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(cfg) }.with_context(|| {
                            format!("CUDA {} kernel launch failed", kernel_name)
                        })?;
                    }

                    out
                }
                #[cfg(feature = "fusion")]
                TensorGraphNode::FusedUnary { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled fused unary kernel")?;

                    let ins = planned_inputs;
                    let input = self
                        .values
                        .get(&ins[0])
                        .context("Missing input value for fused unary operation")?;

                    let len = input.len();
                    let stream = self.device.default_stream();
                    let mut out = self.take_buffer(len)?;
                    if len != 0 {
                        let f = module.load_function(kernel_name)?;
                        let cfg = LaunchConfig::for_num_elems(len as u32);
                        let mut launcher = stream.launch_builder(&f);
                        launcher.arg(input.as_ref());
                        launcher.arg(&mut out);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(cfg) }.with_context(|| {
                            format!("CUDA {} kernel launch failed", kernel_name)
                        })?;
                    }

                    out
                }
                TensorGraphNode::Binary { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled binary kernel")?;

                    let ins = planned_inputs;
                    let lhs = self
                        .values
                        .get(&ins[0])
                        .context("Missing left operand for binary operation")?;
                    let rhs = self
                        .values
                        .get(&ins[1])
                        .context("Missing right operand for binary operation")?;

                    let len = lhs.len();
                    anyhow::ensure!(rhs.len() == len, "Binary operand length mismatch");
                    let stream = self.device.default_stream();
                    let mut out = self.take_buffer(len)?;
                    if len != 0 {
                        let f = module.load_function(kernel_name)?;
                        let cfg = LaunchConfig::for_num_elems(len as u32);
                        let mut launcher = stream.launch_builder(&f);
                        launcher.arg(lhs.as_ref());
                        launcher.arg(rhs.as_ref());
                        launcher.arg(&mut out);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(cfg) }.with_context(|| {
                            format!("CUDA {} kernel launch failed", kernel_name)
                        })?;
                    }

                    out
                }
                TensorGraphNode::Gt { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled Gt kernel")?;

                    let ins = planned_inputs;
                    let lhs = self
                        .values
                        .get(&ins[0])
                        .context("Missing left operand for Gt")?;
                    let rhs = self
                        .values
                        .get(&ins[1])
                        .context("Missing right operand for Gt")?;

                    let len = lhs.len();
                    anyhow::ensure!(rhs.len() == len, "Gt operand length mismatch");
                    let stream = self.device.default_stream();
                    let mut out = self.take_buffer(len)?;
                    if len != 0 {
                        let f = module.load_function(kernel_name)?;
                        let cfg = LaunchConfig::for_num_elems(len as u32);
                        let mut launcher = stream.launch_builder(&f);
                        launcher.arg(lhs.as_ref());
                        launcher.arg(rhs.as_ref());
                        launcher.arg(&mut out);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(cfg) }.with_context(|| {
                            format!("CUDA {} kernel launch failed", kernel_name)
                        })?;
                    }

                    out
                }
                TensorGraphNode::Mask { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled Mask kernel")?;

                    let ins = planned_inputs;
                    let values = self
                        .values
                        .get(&ins[0])
                        .context("Missing values for Mask")?;
                    let condition = self
                        .values
                        .get(&ins[1])
                        .context("Missing condition for Mask")?;

                    let len = values.len();
                    anyhow::ensure!(condition.len() == len, "Mask operand length mismatch");
                    let stream = self.device.default_stream();
                    let mut out = self.take_buffer(len)?;
                    if len != 0 {
                        let f = module.load_function(kernel_name)?;
                        let cfg = LaunchConfig::for_num_elems(len as u32);
                        let mut launcher = stream.launch_builder(&f);
                        launcher.arg(values.as_ref());
                        launcher.arg(condition.as_ref());
                        launcher.arg(&mut out);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(cfg) }
                            .context("CUDA mask kernel launch failed")?;
                    }

                    out
                }
                TensorGraphNode::MatMul { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled MatMul kernel")?;

                    let ins = planned_inputs;
                    let a = self
                        .values
                        .get(&ins[0])
                        .context("Missing A matrix for MatMul")?;
                    let b = self
                        .values
                        .get(&ins[1])
                        .context("Missing B matrix for MatMul")?;

                    let a_shape = graph.graph[ins[0]].shape();
                    let b_shape = graph.graph[ins[1]].shape();
                    self.execute_matmul(
                        *node_idx,
                        kernel_name,
                        a,
                        b,
                        (a_shape, b_shape, graph.graph[*node_idx].shape()),
                    )?
                }
                TensorGraphNode::Embedding { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled Embedding kernel")?;
                    let ins = planned_inputs;
                    anyhow::ensure!(ins.len() == 2, "Embedding requires 2 inputs");
                    let weight = self
                        .values
                        .get(&ins[0])
                        .context("Missing weight for Embedding")?;
                    let indices = self
                        .values
                        .get(&ins[1])
                        .context("Missing indices for Embedding")?;
                    let weight_shape = graph.graph[ins[0]].shape();
                    let indices_shape = graph.graph[ins[1]].shape();
                    anyhow::ensure!(
                        weight_shape.len() == 2,
                        "Embedding weight must be [V, C], got {weight_shape:?}"
                    );
                    let index_count: usize = indices_shape.iter().product();
                    let output_len: usize = node.shape().iter().product();
                    anyhow::ensure!(
                        indices.len() == index_count,
                        "Embedding indices length mismatch"
                    );
                    anyhow::ensure!(
                        weight.len() == weight_shape.iter().product::<usize>(),
                        "Embedding weight length mismatch"
                    );
                    self.validate_indices(indices, weight_shape[0], "Embedding")?;
                    let mut output = self.take_buffer(output_len)?;
                    if output_len != 0 {
                        let launch_len = u32::try_from(output_len)
                            .context("Embedding output is too large for a CUDA launch")?;
                        let function = module.load_function(kernel_name)?;
                        let stream = self.device.default_stream();
                        let mut launcher = stream.launch_builder(&function);
                        launcher.arg(weight.as_ref());
                        launcher.arg(indices.as_ref());
                        launcher.arg(&mut output);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(LaunchConfig::for_num_elems(launch_len)) }
                            .context("PTX embedding kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::EmbeddingBackward { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled EmbeddingBackward kernel")?;
                    let ins = planned_inputs;
                    anyhow::ensure!(ins.len() == 2, "EmbeddingBackward requires 2 inputs");
                    let indices = self
                        .values
                        .get(&ins[0])
                        .context("Missing indices for EmbeddingBackward")?;
                    let grad_output = self
                        .values
                        .get(&ins[1])
                        .context("Missing grad_output for EmbeddingBackward")?;
                    let output_shape = node.shape();
                    anyhow::ensure!(
                        output_shape.len() == 2,
                        "EmbeddingBackward output must be [V, C], got {output_shape:?}"
                    );
                    let index_count: usize = graph.graph[ins[0]].shape().iter().product();
                    let grad_output_len = index_count * output_shape[1];
                    anyhow::ensure!(
                        indices.len() == index_count,
                        "EmbeddingBackward indices length mismatch"
                    );
                    anyhow::ensure!(
                        grad_output.len() == grad_output_len,
                        "EmbeddingBackward grad_output length mismatch"
                    );
                    self.validate_indices(indices, output_shape[0], "EmbeddingBackward")?;
                    let output_len: usize = output_shape.iter().product();
                    let stream = self.device.default_stream();
                    let mut output = self.take_buffer(output_len)?;
                    if output_len != 0 {
                        stream
                            .memset_zeros(&mut output)
                            .context("Failed to zero PTX embedding gradient output")?;
                    }
                    if grad_output_len != 0 {
                        let launch_elements = if <D::HostType as crate::tile::TileDType>::TILE_DTYPE
                            == crate::tile::DType::F32
                        {
                            grad_output_len
                        } else {
                            output_len
                        };
                        let launch_len = u32::try_from(launch_elements).context(
                            "EmbeddingBackward grad_output is too large for a CUDA launch",
                        )?;
                        let function = module.load_function(kernel_name)?;
                        let mut launcher = stream.launch_builder(&function);
                        launcher.arg(indices.as_ref());
                        launcher.arg(grad_output.as_ref());
                        launcher.arg(&mut output);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(LaunchConfig::for_num_elems(launch_len)) }
                            .context("PTX embedding_backward kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::IndexedCrossEntropy { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled IndexedCrossEntropy kernel")?;
                    let ins = planned_inputs;
                    anyhow::ensure!(ins.len() == 2, "IndexedCrossEntropy requires 2 inputs");
                    let logits = self
                        .values
                        .get(&ins[0])
                        .context("Missing logits for IndexedCrossEntropy")?;
                    let targets = self
                        .values
                        .get(&ins[1])
                        .context("Missing targets for IndexedCrossEntropy")?;
                    let logits_shape = graph.graph[ins[0]].shape();
                    let vocabulary = *logits_shape
                        .last()
                        .context("IndexedCrossEntropy logits must have rank >= 1")?;
                    anyhow::ensure!(
                        vocabulary > 0,
                        "IndexedCrossEntropy vocabulary must be nonzero"
                    );
                    let row_count: usize = node.shape().iter().product();
                    anyhow::ensure!(
                        targets.len() == row_count,
                        "Cross entropy targets length mismatch"
                    );
                    anyhow::ensure!(
                        logits.len() == row_count * vocabulary,
                        "Cross entropy logits length mismatch"
                    );
                    self.validate_indices(targets, vocabulary, "IndexedCrossEntropy")?;
                    let mut output = self.take_buffer(row_count)?;
                    if row_count != 0 {
                        let launch_len = u32::try_from(row_count)
                            .context("IndexedCrossEntropy output is too large for a CUDA launch")?;
                        let function = module.load_function(kernel_name)?;
                        let stream = self.device.default_stream();
                        let mut launcher = stream.launch_builder(&function);
                        launcher.arg(logits.as_ref());
                        launcher.arg(targets.as_ref());
                        launcher.arg(&mut output);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(LaunchConfig::for_num_elems(launch_len)) }
                            .context("PTX indexed_cross_entropy kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::IndexedCrossEntropyBackward { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled IndexedCrossEntropyBackward kernel")?;
                    let ins = planned_inputs;
                    anyhow::ensure!(
                        ins.len() == 3,
                        "IndexedCrossEntropyBackward requires 3 inputs"
                    );
                    let logits = self
                        .values
                        .get(&ins[0])
                        .context("Missing logits for IndexedCrossEntropyBackward")?;
                    let targets = self
                        .values
                        .get(&ins[1])
                        .context("Missing targets for IndexedCrossEntropyBackward")?;
                    let grad_output = self
                        .values
                        .get(&ins[2])
                        .context("Missing grad_output for IndexedCrossEntropyBackward")?;
                    let output_shape = node.shape();
                    let vocabulary = *output_shape
                        .last()
                        .context("IndexedCrossEntropyBackward logits must have rank >= 1")?;
                    anyhow::ensure!(
                        vocabulary > 0,
                        "IndexedCrossEntropyBackward vocabulary must be nonzero"
                    );
                    let row_count = output_shape.iter().product::<usize>() / vocabulary;
                    anyhow::ensure!(
                        logits.len() == row_count * vocabulary,
                        "Cross entropy logits length mismatch"
                    );
                    anyhow::ensure!(
                        targets.len() == row_count,
                        "Cross entropy targets length mismatch"
                    );
                    anyhow::ensure!(
                        grad_output.len() == row_count,
                        "Cross entropy grad_output length mismatch"
                    );
                    self.validate_indices(targets, vocabulary, "IndexedCrossEntropyBackward")?;
                    let output_len = row_count * vocabulary;
                    let mut output = self.take_buffer(output_len)?;
                    if output_len != 0 {
                        let launch_len = u32::try_from(output_len).context(
                            "IndexedCrossEntropyBackward output is too large for a CUDA launch",
                        )?;
                        let function = module.load_function(kernel_name)?;
                        let stream = self.device.default_stream();
                        let mut launcher = stream.launch_builder(&function);
                        launcher.arg(logits.as_ref());
                        launcher.arg(targets.as_ref());
                        launcher.arg(grad_output.as_ref());
                        launcher.arg(&mut output);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(LaunchConfig::for_num_elems(launch_len)) }
                            .context("PTX indexed_cross_entropy_backward kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::BroadcastAxis { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled BroadcastAxis kernel")?;

                    let ins = planned_inputs;
                    let input = self
                        .values
                        .get(&ins[0])
                        .context("Missing input for BroadcastAxis")?;

                    // Get output shape
                    let out_shape = graph.graph[*node_idx].shape();
                    let out_len: usize = out_shape.iter().product();

                    let stream = self.device.default_stream();
                    let mut out = self.take_buffer(out_len)?;
                    if out_len != 0 {
                        let f = module.load_function(kernel_name)?;
                        let cfg = LaunchConfig::for_num_elems(out_len as u32);
                        let mut launcher = stream.launch_builder(&f);
                        launcher.arg(input.as_ref());
                        launcher.arg(&mut out);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(cfg) }.with_context(|| {
                            format!("CUDA {} kernel launch failed", kernel_name)
                        })?;
                    }

                    out
                }
                TensorGraphNode::ReduceAxis { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled ReduceAxis kernel")?;

                    let ins = planned_inputs;
                    let input = self
                        .values
                        .get(&ins[0])
                        .context("Missing input for ReduceAxis")?;

                    // Get output shape
                    let out_shape = graph.graph[*node_idx].shape();
                    let out_len: usize = out_shape.iter().product();

                    let stream = self.device.default_stream();
                    let mut out = self.take_buffer(out_len)?;
                    if out_len != 0 {
                        let f = module.load_function(kernel_name)?;
                        let cfg = LaunchConfig::for_num_elems(out_len as u32);
                        let mut launcher = stream.launch_builder(&f);
                        launcher.arg(input.as_ref());
                        launcher.arg(&mut out);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(cfg) }.with_context(|| {
                            format!("CUDA {} kernel launch failed", kernel_name)
                        })?;
                    }

                    out
                }
                TensorGraphNode::Transpose { .. } | TensorGraphNode::Permute { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled permutation kernel")?;

                    let ins = planned_inputs;
                    let input = self
                        .values
                        .get(&ins[0])
                        .context("Missing input for permutation")?;

                    let len = input.len();
                    let stream = self.device.default_stream();
                    let mut out = self.take_buffer(len)?;
                    if len != 0 {
                        let f = module.load_function(kernel_name)?;
                        let cfg = LaunchConfig::for_num_elems(len as u32);
                        let mut launcher = stream.launch_builder(&f);
                        launcher.arg(input.as_ref());
                        launcher.arg(&mut out);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(cfg) }.with_context(|| {
                            format!("CUDA {} kernel launch failed", kernel_name)
                        })?;
                    }

                    out
                }
                TensorGraphNode::Reshape { .. } | TensorGraphNode::Flatten { .. } => {
                    let input_index = planned_inputs[0];
                    let input = self
                        .values
                        .get(&input_index)
                        .context("Missing input for contiguous view")?;
                    let output_len: usize = node.shape().iter().product();
                    anyhow::ensure!(
                        output_len == input.len(),
                        "Contiguous view changes element count from {} to {output_len}",
                        input.len()
                    );
                    let mut output = self.take_buffer(output_len)?;
                    if output_len != 0 {
                        self.execution_metrics.device_copies += 1;
                        self.device
                            .default_stream()
                            .memcpy_dtod(input.as_ref(), &mut output)
                            .context("Failed to copy contiguous view")?;
                    }
                    output
                }
                TensorGraphNode::Conv2d { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled Conv2d kernel")?;
                    let inputs = planned_inputs;
                    let input = self
                        .values
                        .get(&inputs[0])
                        .context("Missing input for Conv2d")?;
                    let weight = self
                        .values
                        .get(&inputs[1])
                        .context("Missing weight for Conv2d")?;
                    let output_len: usize = node.shape().iter().product();
                    let mut output = self.take_buffer(output_len)?;
                    if output_len != 0 {
                        let function = module.load_function(kernel_name)?;
                        let stream = self.device.default_stream();
                        let mut launcher = stream.launch_builder(&function);
                        launcher
                            .arg(input.as_ref())
                            .arg(weight.as_ref())
                            .arg(&mut output);
                        self.record_kernel_launch();
                        unsafe {
                            launcher.launch(LaunchConfig::for_num_elems(
                                u32::try_from(output_len)
                                    .context("Conv2d output is too large for a CUDA launch")?,
                            ))
                        }
                        .context("PTX Conv2d kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::ConvTranspose2d { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled ConvTranspose2d kernel")?;
                    let inputs = planned_inputs;
                    let grad_output = self
                        .values
                        .get(&inputs[0])
                        .context("Missing grad_output for ConvTranspose2d")?;
                    let weight = self
                        .values
                        .get(&inputs[1])
                        .context("Missing weight for ConvTranspose2d")?;
                    let output_len: usize = node.shape().iter().product();
                    let mut output = self.take_buffer(output_len)?;
                    if output_len != 0 {
                        let function = module.load_function(kernel_name)?;
                        let stream = self.device.default_stream();
                        let mut launcher = stream.launch_builder(&function);
                        launcher
                            .arg(grad_output.as_ref())
                            .arg(weight.as_ref())
                            .arg(&mut output);
                        self.record_kernel_launch();
                        unsafe {
                            launcher.launch(LaunchConfig::for_num_elems(
                                u32::try_from(output_len).context(
                                    "ConvTranspose2d output is too large for a CUDA launch",
                                )?,
                            ))
                        }
                        .context("PTX ConvTranspose2d kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::Conv2dBackwardWeight { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled Conv2dBackwardWeight kernel")?;
                    let inputs = planned_inputs;
                    let input = self
                        .values
                        .get(&inputs[0])
                        .context("Missing input for Conv2dBackwardWeight")?;
                    let grad_output = self
                        .values
                        .get(&inputs[1])
                        .context("Missing grad_output for Conv2dBackwardWeight")?;
                    let output_len: usize = node.shape().iter().product();
                    let mut output = self.take_buffer(output_len)?;
                    if output_len != 0 {
                        let function = module.load_function(kernel_name)?;
                        let stream = self.device.default_stream();
                        let mut launcher = stream.launch_builder(&function);
                        launcher
                            .arg(input.as_ref())
                            .arg(grad_output.as_ref())
                            .arg(&mut output);
                        self.record_kernel_launch();
                        unsafe {
                            launcher.launch(LaunchConfig::for_num_elems(
                                u32::try_from(output_len).context(
                                    "Conv2dBackwardWeight output is too large for a CUDA launch",
                                )?,
                            ))
                        }
                        .context("PTX Conv2dBackwardWeight kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::MaxPool2d { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled MaxPool2d kernel")?;
                    let input_index = planned_inputs[0];
                    let input = self
                        .values
                        .get(&input_index)
                        .context("Missing input for MaxPool2d")?;
                    let output_len: usize = node.shape().iter().product();
                    let mut output = self.take_buffer(output_len)?;
                    if output_len != 0 {
                        let function = module.load_function(kernel_name)?;
                        let stream = self.device.default_stream();
                        let mut launcher = stream.launch_builder(&function);
                        launcher.arg(input.as_ref()).arg(&mut output);
                        self.record_kernel_launch();
                        unsafe {
                            launcher.launch(LaunchConfig::for_num_elems(
                                u32::try_from(output_len)
                                    .context("MaxPool2d output is too large for a CUDA launch")?,
                            ))
                        }
                        .context("PTX MaxPool2d kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::MaxPool2dBackward { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled MaxPool2dBackward kernel")?;
                    let inputs = planned_inputs;
                    let input = self
                        .values
                        .get(&inputs[0])
                        .context("Missing input for MaxPool2dBackward")?;
                    let pooled = self
                        .values
                        .get(&inputs[1])
                        .context("Missing pooled output for MaxPool2dBackward")?;
                    let grad_output = self
                        .values
                        .get(&inputs[2])
                        .context("Missing grad_output for MaxPool2dBackward")?;
                    let output_len: usize = node.shape().iter().product();
                    let mut output = self.take_buffer(output_len)?;
                    if output_len != 0 {
                        let function = module.load_function(kernel_name)?;
                        let stream = self.device.default_stream();
                        let mut launcher = stream.launch_builder(&function);
                        launcher
                            .arg(input.as_ref())
                            .arg(pooled.as_ref())
                            .arg(grad_output.as_ref())
                            .arg(&mut output);
                        self.record_kernel_launch();
                        unsafe {
                            launcher.launch(LaunchConfig::for_num_elems(
                                u32::try_from(output_len).context(
                                    "MaxPool2dBackward output is too large for a CUDA launch",
                                )?,
                            ))
                        }
                        .context("PTX MaxPool2dBackward kernel launch failed")?;
                    }
                    output
                }
            };
            let result = Arc::new(result);

            let bytes = result.len() * std::mem::size_of::<D::HostType>();
            self.execution_metrics.materialized_values += 1;
            self.execution_metrics.materialized_bytes += bytes;
            let is_source = matches!(
                node,
                TensorGraphNode::Constant { .. }
                    | TensorGraphNode::Input { .. }
                    | TensorGraphNode::Parameter { .. }
            );
            if !is_source && *node_idx != graph_output {
                self.execution_metrics.intermediate_materialized_bytes += bytes;
            }
            self.values.insert(*node_idx, result);
            live_bytes += bytes;
            self.stats.record_live(live_bytes);

            for &dead in &plan_step.release_after {
                if let Some(value) = self.values.remove(&dead) {
                    live_bytes -= self.recycle_value(value);
                }
            }
        }

        self.stats = self.stats.with_pool_stats(self.pool.get_mut().stats());
        self.execution_metrics.kernel_launches = self.kernel_launches.get();

        let last_node_idx = &graph_output;
        let out_device = self
            .values
            .get(last_node_idx)
            .context("Output value not found after execution")?;
        let mut storage_host = vec![D::HostType::default(); out_device.len()];
        if !storage_host.is_empty() {
            self.execution_metrics.device_copies += 1;
            self.device
                .default_stream()
                .memcpy_dtoh(out_device.as_ref(), &mut storage_host)
                .context("Failed to copy output from CUDA device to host")?;
        }
        let output_view = self
            .execution_plan
            .as_ref()
            .and_then(|plan| plan.virtual_value(graph_output))
            .context("Graph output has no virtual value")?;
        let source_shape = self
            .execution_plan
            .as_ref()
            .and_then(|plan| plan.value_shape(output_view.source))
            .context("Virtual output source is absent from the execution plan")?;
        materialize_host_view(storage_host, output_view, source_shape)
    }

    /// Compile and execute, reusing the module for the same graph structure.
    pub fn compile_and_execute<G>(
        &mut self,
        graph: &TensorGraph<D::HostType, G>,
        inputs: HashMap<String, Vec<D::HostType>>,
    ) -> Result<Vec<D::HostType>> {
        let signature = graph_compilation_signature(
            graph,
            self.reduction_mode,
            self.target,
            self.matmul_precision,
            self.matmul_shared_layout,
            self.matmul_pipeline,
            self.fusion_policy,
        );
        if self.module.is_none() || self.compilation_signature.as_deref() != Some(&signature) {
            self.compile(graph)?;
        }
        self.execute_compiled(graph, inputs)
    }
}

fn validate_batch_broadcast(input_shape: &[usize], output_batch_shape: &[usize]) -> Result<()> {
    let input_batch_shape = &input_shape[..input_shape.len() - 2];
    anyhow::ensure!(
        input_batch_shape.len() <= output_batch_shape.len(),
        "MatMul input batch shape {input_batch_shape:?} has higher rank than output batch shape {output_batch_shape:?}"
    );
    let leading = output_batch_shape.len() - input_batch_shape.len();
    for (dimension, &input_size) in input_batch_shape.iter().enumerate() {
        let output_size = output_batch_shape[leading + dimension];
        anyhow::ensure!(
            input_size == 1 || input_size == output_size,
            "Cannot broadcast MatMul batch dimension {input_size} to {output_size}"
        );
    }
    Ok(())
}

fn materialize_host_view<T: Copy>(
    storage: Vec<T>,
    view: &VirtualTensor,
    source_shape: &[usize],
) -> Result<Vec<T>> {
    anyhow::ensure!(
        view.predicate.is_none(),
        "Predicated virtual outputs are not yet supported at the host boundary"
    );
    if view.shape == source_shape && view.access == IndexMap::identity(view.shape.len()) {
        return Ok(storage);
    }
    let output_len = checked_element_count(&view.shape, "Virtual output")?;
    let mut output = Vec::with_capacity(output_len);
    for linear_output in 0..output_len {
        let mut remainder = linear_output;
        let mut output_index = vec![0; view.shape.len()];
        for dimension in (0..view.shape.len()).rev() {
            let extent = view.shape[dimension];
            anyhow::ensure!(extent > 0, "non-empty virtual output has a zero extent");
            output_index[dimension] = remainder % extent;
            remainder /= extent;
        }
        let source_index = view.source_index(&output_index)?;
        anyhow::ensure!(
            source_index.len() == source_shape.len(),
            "Virtual output source rank does not match its storage shape"
        );
        let mut linear_source = 0usize;
        for (&coordinate, &extent) in source_index.iter().zip(source_shape) {
            anyhow::ensure!(
                coordinate < extent,
                "Virtual output source coordinate {coordinate} exceeds extent {extent}"
            );
            linear_source = linear_source
                .checked_mul(extent)
                .and_then(|offset| offset.checked_add(coordinate))
                .context("Virtual output source index overflowed usize")?;
        }
        output.push(
            *storage
                .get(linear_source)
                .context("Virtual output source index exceeds its storage buffer")?,
        );
    }
    Ok(output)
}

fn checked_element_count(shape: &[usize], description: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .with_context(|| format!("{description} shape {shape:?} overflows usize"))
    })
}

fn signature_usize(signature: &mut Vec<u8>, value: usize) {
    signature.extend_from_slice(&(value as u64).to_le_bytes());
}

fn signature_slice(signature: &mut Vec<u8>, values: &[usize]) {
    signature_usize(signature, values.len());
    for &value in values {
        signature_usize(signature, value);
    }
}

fn signature_str(signature: &mut Vec<u8>, value: &str) {
    signature_usize(signature, value.len());
    signature.extend_from_slice(value.as_bytes());
}

fn graph_compilation_signature<D: crate::tile::TileDType, G>(
    graph: &TensorGraph<D, G>,
    reduction_mode: PtxReductionMode,
    target: PtxTarget,
    matmul_precision: crate::tile::MatMulPrecision,
    matmul_shared_layout: MatMulSharedLayout,
    matmul_pipeline: MatMulPipeline,
    fusion_policy: PtxFusionPolicy,
) -> Vec<u8> {
    let mut signature = Vec::new();
    signature.extend_from_slice(PTX_GRAPH_SIGNATURE_MAGIC);
    signature.push(PTX_GRAPH_SIGNATURE_VERSION);
    signature.push(match D::TILE_DTYPE {
        crate::tile::DType::F32 => 0,
        crate::tile::DType::F16 => 1,
        crate::tile::DType::BF16 => 2,
        _ => unreachable!("unsupported host storage dtype"),
    });
    signature.push(match reduction_mode {
        PtxReductionMode::Strict => 0,
        PtxReductionMode::DeterministicTree => 1,
        PtxReductionMode::Online => 2,
    });
    signature.push(match matmul_precision {
        crate::tile::MatMulPrecision::StrictF32 => 0,
        crate::tile::MatMulPrecision::AllowTf32 => 1,
    });
    signature.push(match matmul_shared_layout {
        MatMulSharedLayout::Auto => 0,
        MatMulSharedLayout::Contiguous => 1,
        MatMulSharedLayout::Padded => 2,
        MatMulSharedLayout::Swizzled => 3,
    });
    signature.push(match matmul_pipeline {
        MatMulPipeline::Auto => 0,
        MatMulPipeline::Synchronous => 1,
        MatMulPipeline::DoubleBuffered => 2,
    });
    signature.push(match fusion_policy {
        PtxFusionPolicy::CostAware => 0,
        PtxFusionPolicy::Greedy => 1,
    });
    signature.extend_from_slice(&target.compute_capability.0.to_le_bytes());
    signature.extend_from_slice(&target.compute_capability.1.to_le_bytes());
    signature_usize(&mut signature, target.multiprocessor_count);
    for budget in [
        target.matmul_resources.shared_bytes_per_block,
        target.matmul_resources.shared_bytes_per_sm,
        target.matmul_resources.registers_per_sm,
        target.matmul_resources.threads_per_sm,
    ] {
        signature_usize(&mut signature, budget);
    }
    signature_usize(&mut signature, graph.graph.node_count());
    for node_index in graph.graph.node_indices() {
        let node = &graph.graph[node_index];
        signature_usize(&mut signature, node_index.index());
        signature_str(&mut signature, node.name());
        match node {
            TensorGraphNode::Constant { .. } => signature.push(0),
            TensorGraphNode::Input { name, .. } => {
                signature.push(1);
                signature_str(&mut signature, name);
            }
            TensorGraphNode::Parameter { .. } => signature.push(2),
            TensorGraphNode::Unary { op, .. } => {
                signature.push(3);
                signature.push(match op {
                    crate::tensor::UnaryOp::Neg => 0,
                    crate::tensor::UnaryOp::Exp => 1,
                    crate::tensor::UnaryOp::Log => 2,
                    crate::tensor::UnaryOp::Relu => 3,
                });
            }
            #[cfg(feature = "fusion")]
            TensorGraphNode::FusedUnary { ops, .. } => {
                signature.push(4);
                signature_usize(&mut signature, ops.len());
                for op in ops {
                    signature.push(match op {
                        crate::tensor::UnaryOp::Neg => 0,
                        crate::tensor::UnaryOp::Exp => 1,
                        crate::tensor::UnaryOp::Log => 2,
                        crate::tensor::UnaryOp::Relu => 3,
                    });
                }
            }
            TensorGraphNode::Binary { op, .. } => {
                signature.push(5);
                signature.push(match op {
                    crate::tensor::BinaryOp::Add => 0,
                    crate::tensor::BinaryOp::Sub => 1,
                    crate::tensor::BinaryOp::Mul => 2,
                    crate::tensor::BinaryOp::Div => 3,
                });
            }
            TensorGraphNode::MatMul { .. } => signature.push(6),
            TensorGraphNode::Embedding { .. } => signature.push(7),
            TensorGraphNode::EmbeddingBackward { .. } => signature.push(8),
            TensorGraphNode::IndexedCrossEntropy { .. } => signature.push(9),
            TensorGraphNode::IndexedCrossEntropyBackward { .. } => signature.push(10),
            TensorGraphNode::Transpose { .. } => signature.push(11),
            TensorGraphNode::Reshape { .. } => signature.push(12),
            TensorGraphNode::Permute { axes, .. } => {
                signature.push(13);
                signature_slice(&mut signature, axes);
            }
            TensorGraphNode::BroadcastAxis { axis, .. } => {
                signature.push(14);
                signature_usize(&mut signature, *axis);
            }
            TensorGraphNode::ReduceAxis { op, axis, .. } => {
                signature.push(15);
                signature.push(match op {
                    crate::tensor::ReduceOp::Sum => 0,
                    crate::tensor::ReduceOp::Max => 1,
                    crate::tensor::ReduceOp::Mean => 2,
                });
                signature_usize(&mut signature, *axis);
            }
            TensorGraphNode::Gt { .. } => signature.push(16),
            TensorGraphNode::Mask { .. } => signature.push(17),
            TensorGraphNode::Conv2d {
                stride, padding, ..
            } => {
                signature.push(18);
                signature_usize(&mut signature, *stride);
                signature_usize(&mut signature, *padding);
            }
            TensorGraphNode::ConvTranspose2d {
                stride, padding, ..
            } => {
                signature.push(19);
                signature_usize(&mut signature, *stride);
                signature_usize(&mut signature, *padding);
            }
            TensorGraphNode::Conv2dBackwardWeight {
                stride, padding, ..
            } => {
                signature.push(20);
                signature_usize(&mut signature, *stride);
                signature_usize(&mut signature, *padding);
            }
            TensorGraphNode::MaxPool2d {
                kernel_size,
                stride,
                ..
            } => {
                signature.push(21);
                signature_usize(&mut signature, *kernel_size);
                signature_usize(&mut signature, *stride);
            }
            TensorGraphNode::MaxPool2dBackward {
                kernel_size,
                stride,
                ..
            } => {
                signature.push(22);
                signature_usize(&mut signature, *kernel_size);
                signature_usize(&mut signature, *stride);
            }
            TensorGraphNode::Flatten { .. } => signature.push(23),
        }
        signature_slice(&mut signature, node.shape());
    }

    let mut edges: Vec<(usize, usize, usize)> = graph
        .graph
        .edge_references()
        .map(|edge| (edge.source().index(), edge.target().index(), *edge.weight()))
        .collect();
    edges.sort_unstable();
    signature_usize(&mut signature, edges.len());
    for (source, target, weight) in edges {
        signature_usize(&mut signature, source);
        signature_usize(&mut signature, target);
        signature_usize(&mut signature, weight);
    }
    signature
}

fn batch_matrix_offset(
    output_batch: usize,
    output_batch_shape: &[usize],
    input_shape: &[usize],
) -> usize {
    let input_batch_shape = &input_shape[..input_shape.len() - 2];
    let leading = output_batch_shape.len() - input_batch_shape.len();
    let mut remaining = output_batch;
    let mut output_coordinates = vec![0; output_batch_shape.len()];
    for dimension in (0..output_batch_shape.len()).rev() {
        output_coordinates[dimension] = remaining % output_batch_shape[dimension];
        remaining /= output_batch_shape[dimension];
    }
    input_batch_shape
        .iter()
        .enumerate()
        .fold(0, |offset, (dimension, &size)| {
            let coordinate = if size == 1 {
                0
            } else {
                output_coordinates[leading + dimension]
            };
            offset * size + coordinate
        })
}

impl<D: CudaDType> Executor<D::HostType> for PtxExecutor<D> {
    fn execute<G>(
        &mut self,
        graph: &TensorGraph<D::HostType, G>,
        inputs: HashMap<String, Vec<D::HostType>>,
    ) -> Result<Vec<D::HostType>>
    where
        TensorGraph<D::HostType, G>: Clone,
    {
        let signature = graph_compilation_signature(
            graph,
            self.reduction_mode,
            self.target,
            self.matmul_precision,
            self.matmul_shared_layout,
            self.matmul_pipeline,
            self.fusion_policy,
        );
        let needs_compilation = self.module.is_none()
            || self.ptx_graph.is_none()
            || self.compilation_signature.as_deref() != Some(signature.as_slice());
        if needs_compilation {
            self.compile_owned(graph.clone())?;
        }
        self.execute_compiled(graph, inputs)
    }

    fn get_gradients(
        &self,
        graph: &TensorGraph<D::HostType, WithGrad>,
    ) -> HashMap<usize, Vec<D::HostType>> {
        let mut result = HashMap::new();

        // Iterate through all parameters in the gradient metadata
        for (param_id, grad_node_idx) in &graph.gradient_metadata().param_to_grad {
            // Look up the gradient value from our computed values
            if let Some(grad_value) = self.get_value(*grad_node_idx) {
                result.insert(*param_id, grad_value);
            }
        }

        result
    }
}
