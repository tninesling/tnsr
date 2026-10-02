use super::graph::PtxGraph;
use super::plan::{PtxExecutionPlan, PtxPlanAction};
use crate::Executor;
use crate::alloc::{AllocStats, CudaBufferPool};
use crate::graph::{TensorGraph, TensorGraphNode, WithGrad};
use crate::tile::{IndexMap, TileGraph, VirtualTensor};
use anyhow::{Context as _, Result};
use cudarc::driver::{CudaContext, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use petgraph::visit::EdgeRef;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Measurements from the most recent successful PTX compilation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PtxCompileMetrics {
    pub graph_nodes: usize,
    pub generated_kernels: usize,
    pub ptx_source_bytes: usize,
    pub compile_time: Duration,
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
pub struct PtxExecutor {
    device: Arc<CudaContext>,
    module: Option<Arc<CudaModule>>,
    values: HashMap<petgraph::graph::NodeIndex, Arc<CudaSlice<f32>>>,
    pool: RefCell<CudaBufferPool>,
    stats: AllocStats,
    /// Stores the PtxGraph to access kernel names during execution
    ptx_graph: Option<Box<PtxGraph>>,
    execution_plan: Option<Box<PtxExecutionPlan>>,
    compilation_signature: Option<Vec<u8>>,
    compilation_count: usize,
    compile_metrics: PtxCompileMetrics,
    execution_metrics: PtxExecutionMetrics,
    kernel_launches: Cell<usize>,
}

impl Default for PtxExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl PtxExecutor {
    /// Create a new PTX executor, panicking if CUDA initialization fails.
    ///
    /// For fallible initialization, use [`PtxExecutor::try_new`].
    pub fn new() -> Self {
        Self::try_new().expect("Failed to initialize PTX executor")
    }

    /// Try to create a new PTX executor, returning an error if initialization fails.
    pub fn try_new() -> Result<Self> {
        let device = CudaContext::new(0).context("Failed to initialize CUDA device 0")?;
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
        })
    }

    fn take_buffer(&self, len: usize) -> Result<CudaSlice<f32>> {
        self.pool
            .borrow_mut()
            .take(&self.device.default_stream(), len)
    }

    fn validate_indices(
        &self,
        indices: &CudaSlice<f32>,
        upper: usize,
        operation: &str,
    ) -> Result<()> {
        let mut host_indices = vec![0.0; indices.len()];
        if !host_indices.is_empty() {
            self.device
                .default_stream()
                .memcpy_dtoh(indices, &mut host_indices)
                .with_context(|| format!("Failed to validate PTX {operation} indices"))?;
        }
        for (position, &value) in host_indices.iter().enumerate() {
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

    /// Describe the current physical PTX execution plan.
    pub fn describe_plan<G>(&self, graph: &TensorGraph<f32, G>) -> Result<String> {
        let supplied_signature = graph_compilation_signature(graph);
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
                PtxPlanAction::ReductionRegion(region_id) => format!(
                    "kernel {}",
                    ptx_graph
                        .reduction_region_kernel_name(region_id)
                        .context("Missing compiled reduction region kernel")?
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

    fn recycle_value(&self, value: Arc<CudaSlice<f32>>) -> usize {
        let bytes = value.len() * std::mem::size_of::<f32>();
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
    pub fn release_gradients(&mut self, graph: &TensorGraph<f32, WithGrad>) {
        for grad_node in graph.gradient_metadata().param_to_grad.values() {
            if let Some(buf) = self.values.remove(grad_node) {
                self.recycle_value(buf);
            }
        }
    }

    /// Compile an owned TensorGraph to PTX without cloning during lowering.
    ///
    /// Prefer this over compiling from a reference when you have an owned graph.
    pub fn compile_owned<G>(&mut self, graph: TensorGraph<f32, G>) -> Result<()> {
        let started = Instant::now();
        let graph_nodes = graph.graph.node_count();
        let signature = graph_compilation_signature(&graph);
        let execution_plan = PtxExecutionPlan::build(&graph)?;
        let mut tile_graph: TileGraph = graph.into();
        tile_graph.add_fusion_regions(execution_plan.regions())?;
        tile_graph.add_reduction_regions(execution_plan.reduction_regions())?;
        tile_graph.set_physical_nodes(
            execution_plan
                .steps()
                .iter()
                .filter_map(|step| (step.action == PtxPlanAction::Kernel).then_some(step.node))
                .collect(),
        );
        let ptx_graph: PtxGraph = tile_graph.into();
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
        };

        Ok(())
    }

    /// Compile a TensorGraph to PTX via clone (for use with Executor trait).
    fn compile_via_clone<G>(&mut self, graph: &TensorGraph<f32, G>) -> Result<()>
    where
        TensorGraph<f32, G>: Clone,
    {
        self.compile_owned(graph.clone())
    }

    /// Get the value of a specific node from the executor's cache after execution
    pub fn get_value(&self, node_idx: petgraph::graph::NodeIndex) -> Option<Vec<f32>> {
        self.values.get(&node_idx).and_then(|cuda_slice| {
            let mut storage = vec![0.0f32; cuda_slice.len()];
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
        kernel_name: &str,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        a_shape: &[usize],
        b_shape: &[usize],
        output_shape: &[usize],
    ) -> Result<CudaSlice<f32>> {
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

        const TILE_SIZE: usize = 16;
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
                n.div_ceil(TILE_SIZE) as u32,
                m.div_ceil(TILE_SIZE) as u32,
                1,
            ),
            block_dim: (TILE_SIZE as u32, TILE_SIZE as u32, 1),
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

    fn execute_pointwise_region(
        &self,
        module: &Arc<CudaModule>,
        ptx_graph: &PtxGraph,
        region_id: usize,
        step: &super::plan::PtxPlanStep,
    ) -> Result<Vec<(petgraph::graph::NodeIndex, Arc<CudaSlice<f32>>)>> {
        let kernel_name = ptx_graph
            .region_kernel_name(region_id)
            .context("Missing compiled pointwise region kernel")?;
        let input_values: Vec<_> = step
            .inputs
            .iter()
            .map(|input| {
                self.values.get(input).cloned().with_context(|| {
                    format!(
                        "Pointwise region input node {} is unavailable",
                        input.index()
                    )
                })
            })
            .collect::<Result<_>>()?;
        let len = checked_element_count(&step.shape, "Pointwise region")?;
        let region = self
            .execution_plan
            .as_ref()
            .and_then(|plan| plan.regions().get(region_id))
            .context("Pointwise region metadata is unavailable")?;
        anyhow::ensure!(
            region.inputs.len() == input_values.len(),
            "Pointwise region input binding count mismatch"
        );
        for (input, descriptor) in input_values.iter().zip(&region.inputs) {
            let expected =
                checked_element_count(&descriptor.source_shape, "Pointwise region input storage")?;
            anyhow::ensure!(
                input.len() == expected,
                "Pointwise region input length {} does not match storage length {expected}",
                input.len(),
            );
        }
        let mut outputs = Vec::with_capacity(step.outputs.len());
        for _ in &step.outputs {
            outputs.push(self.take_buffer(len)?);
        }
        if len != 0 {
            let launch_len = u32::try_from(len)
                .context("Pointwise region output is too large for a CUDA launch")?;
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
            unsafe { launcher.launch(LaunchConfig::for_num_elems(launch_len)) }
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
    ) -> Result<Vec<(petgraph::graph::NodeIndex, Arc<CudaSlice<f32>>)>> {
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
            unsafe { launcher.launch(LaunchConfig::for_num_elems(launch_len)) }
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
        graph: &TensorGraph<f32, G>,
        inputs: HashMap<String, Vec<f32>>,
    ) -> Result<Vec<f32>> {
        self.execution_metrics = PtxExecutionMetrics::default();
        self.kernel_launches.set(0);
        let supplied_signature = graph_compilation_signature(graph);
        let compiled_signature = self
            .compilation_signature
            .as_deref()
            .context("No compiled graph signature available. Call compile_owned() first.")?;
        anyhow::ensure!(
            compiled_signature == supplied_signature.as_slice(),
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
            if let PtxPlanAction::PointwiseRegion(region_id) = plan_step.action {
                let region_outputs =
                    self.execute_pointwise_region(module, ptx_graph, region_id, plan_step)?;
                for (output_node, output) in region_outputs {
                    let bytes = output.len() * std::mem::size_of::<f32>();
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
                    let bytes = output.len() * std::mem::size_of::<f32>();
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
                        kernel_name,
                        a,
                        b,
                        a_shape,
                        b_shape,
                        graph.graph[*node_idx].shape(),
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
                        let launch_len = u32::try_from(grad_output_len).context(
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

            let bytes = result.len() * std::mem::size_of::<f32>();
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
        let mut storage_host = vec![0.0f32; out_device.len()];
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

    /// Compile and execute a TensorGraph in one call
    pub fn compile_and_execute<G>(
        &mut self,
        graph: &TensorGraph<f32, G>,
        inputs: HashMap<String, Vec<f32>>,
    ) -> Result<Vec<f32>>
    where
        TensorGraph<f32, G>: Clone,
    {
        self.compile_via_clone(graph)?;
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

fn materialize_host_view(
    storage: Vec<f32>,
    view: &VirtualTensor,
    source_shape: &[usize],
) -> Result<Vec<f32>> {
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

fn graph_compilation_signature<G>(graph: &TensorGraph<f32, G>) -> Vec<u8> {
    let mut signature = Vec::new();
    signature.extend_from_slice(b"tnsr-ptx-graph-v1");
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

impl Executor<f32> for PtxExecutor {
    fn execute<G>(
        &mut self,
        graph: &TensorGraph<f32, G>,
        inputs: HashMap<String, Vec<f32>>,
    ) -> Result<Vec<f32>>
    where
        TensorGraph<f32, G>: Clone,
    {
        let signature = graph_compilation_signature(graph);
        let needs_compilation = self.module.is_none()
            || self.ptx_graph.is_none()
            || self.compilation_signature.as_deref() != Some(signature.as_slice());
        if needs_compilation {
            self.compile_owned(graph.clone())?;
        }
        self.execute_compiled(graph, inputs)
    }

    fn get_gradients(&self, graph: &TensorGraph<f32, WithGrad>) -> HashMap<usize, Vec<f32>> {
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
