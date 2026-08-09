use super::graph::PtxGraph;
use crate::Executor;
use crate::alloc::{AllocStats, CudaBufferPool};
use crate::graph::{TensorGraph, TensorGraphNode, WithGrad, liveness};
use crate::tile::TileGraph;
use anyhow::{Context as _, Result};
use cudarc::driver::{CudaContext, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use petgraph::visit::EdgeRef;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// PTX executor that compiles TensorGraph → TileGraph → PtxGraph → PTX string
/// and executes kernels via cudarc
pub struct PtxExecutor {
    device: Arc<CudaContext>,
    module: Option<Arc<CudaModule>>,
    values: HashMap<petgraph::graph::NodeIndex, CudaSlice<f32>>,
    pool: RefCell<CudaBufferPool>,
    stats: AllocStats,
    /// Stores the PtxGraph to access kernel names during execution
    ptx_graph: Option<PtxGraph>,
    compilation_signature: Option<Vec<u8>>,
    compilation_count: usize,
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
            compilation_signature: None,
            compilation_count: 0,
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
                self.pool.get_mut().give(buf);
            }
        }
    }

    /// Compile an owned TensorGraph to PTX without cloning during lowering.
    ///
    /// Prefer this over compiling from a reference when you have an owned graph.
    pub fn compile_owned<G>(&mut self, graph: TensorGraph<f32, G>) -> Result<()> {
        validate_supported_graph(&graph)?;
        let signature = graph_compilation_signature(&graph);
        let tile_graph: TileGraph = graph.into();
        let ptx_graph: PtxGraph = tile_graph.into();
        let ptx_src = ptx_graph.module_source();

        let cuda_module = self
            .device
            .load_module(Ptx::from_src(&ptx_src))
            .context("Failed to load PTX module with cudarc")?;
        self.module = Some(cuda_module);
        self.ptx_graph = Some(ptx_graph);
        self.compilation_signature = Some(signature);
        self.compilation_count += 1;

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
            let mut host_vec = vec![0.0f32; cuda_slice.len()];
            self.device
                .default_stream()
                .memcpy_dtoh(cuda_slice, &mut host_vec)
                .ok()?;
            Some(host_vec)
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
            unsafe { launcher.launch(config) }
                .with_context(|| format!("CUDA {kernel_name} kernel launch failed"))?;
        }
        Ok(output)
    }

    /// Execute a compiled graph with the given inputs
    pub fn execute_compiled<G>(
        &mut self,
        graph: &TensorGraph<f32, G>,
        inputs: HashMap<String, Vec<f32>>,
    ) -> Result<Vec<f32>> {
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

        let order = graph.toposort();
        let liveness = liveness::analyze(graph, &order);
        for (_, value) in self.values.drain() {
            self.pool.get_mut().give(value);
        }
        let mut live_bytes = 0usize;

        for (pos, node_idx) in order.iter().enumerate() {
            let node = &graph[*node_idx];
            let result = match node {
                TensorGraphNode::Constant { data, .. } => {
                    let stream = self.device.default_stream();
                    let mut device_data = self.take_buffer(data.len())?;
                    if !data.is_empty() {
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

                    let ins = graph.inputs(*node_idx);
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
                        launcher.arg(input);
                        launcher.arg(&mut out);
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

                    let ins = graph.inputs(*node_idx);
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
                        launcher.arg(input);
                        launcher.arg(&mut out);
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

                    let ins = graph.inputs(*node_idx);
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
                        launcher.arg(lhs);
                        launcher.arg(rhs);
                        launcher.arg(&mut out);
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

                    let ins = graph.inputs(*node_idx);
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
                        launcher.arg(lhs);
                        launcher.arg(rhs);
                        launcher.arg(&mut out);
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

                    let ins = graph.inputs(*node_idx);
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
                        launcher.arg(values);
                        launcher.arg(condition);
                        launcher.arg(&mut out);
                        unsafe { launcher.launch(cfg) }
                            .context("CUDA mask kernel launch failed")?;
                    }

                    out
                }
                TensorGraphNode::MatMul { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled MatMul kernel")?;

                    let ins = graph.inputs(*node_idx);
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
                    let ins = graph.inputs(*node_idx);
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
                        launcher.arg(weight);
                        launcher.arg(indices);
                        launcher.arg(&mut output);
                        unsafe { launcher.launch(LaunchConfig::for_num_elems(launch_len)) }
                            .context("PTX embedding kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::EmbeddingBackward { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled EmbeddingBackward kernel")?;
                    let ins = graph.inputs(*node_idx);
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
                        launcher.arg(indices);
                        launcher.arg(grad_output);
                        launcher.arg(&mut output);
                        unsafe { launcher.launch(LaunchConfig::for_num_elems(launch_len)) }
                            .context("PTX embedding_backward kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::IndexedCrossEntropy { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled IndexedCrossEntropy kernel")?;
                    let ins = graph.inputs(*node_idx);
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
                        launcher.arg(logits);
                        launcher.arg(targets);
                        launcher.arg(&mut output);
                        unsafe { launcher.launch(LaunchConfig::for_num_elems(launch_len)) }
                            .context("PTX indexed_cross_entropy kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::IndexedCrossEntropyBackward { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled IndexedCrossEntropyBackward kernel")?;
                    let ins = graph.inputs(*node_idx);
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
                        launcher.arg(logits);
                        launcher.arg(targets);
                        launcher.arg(grad_output);
                        launcher.arg(&mut output);
                        unsafe { launcher.launch(LaunchConfig::for_num_elems(launch_len)) }
                            .context("PTX indexed_cross_entropy_backward kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::BroadcastAxis { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled BroadcastAxis kernel")?;

                    let ins = graph.inputs(*node_idx);
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
                        launcher.arg(input);
                        launcher.arg(&mut out);
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

                    let ins = graph.inputs(*node_idx);
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
                        launcher.arg(input);
                        launcher.arg(&mut out);
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

                    let ins = graph.inputs(*node_idx);
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
                        launcher.arg(input);
                        launcher.arg(&mut out);
                        unsafe { launcher.launch(cfg) }.with_context(|| {
                            format!("CUDA {} kernel launch failed", kernel_name)
                        })?;
                    }

                    out
                }
                TensorGraphNode::Reshape { .. } | TensorGraphNode::Flatten { .. } => {
                    let input_index = graph.inputs(*node_idx)[0];
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
                        self.device
                            .default_stream()
                            .memcpy_dtod(input, &mut output)
                            .context("Failed to copy contiguous view")?;
                    }
                    output
                }
                TensorGraphNode::Conv2d { .. }
                | TensorGraphNode::ConvTranspose2d { .. }
                | TensorGraphNode::Conv2dBackwardWeight { .. }
                | TensorGraphNode::MaxPool2d { .. }
                | TensorGraphNode::MaxPool2dBackward { .. } => {
                    anyhow::bail!(
                        "Operation is not yet supported by the PTX backend. Use the CUDA (static kernels) or CPU backend instead."
                    )
                }
            };

            let bytes = result.len() * std::mem::size_of::<f32>();
            self.values.insert(*node_idx, result);
            live_bytes += bytes;
            self.stats.record_live(live_bytes);

            for &dead in liveness.free_after(pos) {
                if let Some(value) = self.values.remove(&dead) {
                    live_bytes -= value.len() * std::mem::size_of::<f32>();
                    self.pool.get_mut().give(value);
                }
            }
        }

        self.stats = self.stats.with_pool_stats(self.pool.get_mut().stats());

        let last_node_idx = order.last().context("Graph is empty")?;
        let out_device = self
            .values
            .get(last_node_idx)
            .context("Output value not found after execution")?;
        let mut out_host = vec![0.0f32; out_device.len()];
        if !out_host.is_empty() {
            self.device
                .default_stream()
                .memcpy_dtoh(out_device, &mut out_host)
                .context("Failed to copy output from CUDA device to host")?;
        }
        Ok(out_host)
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

fn validate_supported_graph<G>(graph: &TensorGraph<f32, G>) -> Result<()> {
    for node_index in graph.graph.node_indices() {
        let node = &graph.graph[node_index];
        if matches!(
            node,
            TensorGraphNode::Conv2d { .. }
                | TensorGraphNode::ConvTranspose2d { .. }
                | TensorGraphNode::Conv2dBackwardWeight { .. }
                | TensorGraphNode::MaxPool2d { .. }
                | TensorGraphNode::MaxPool2dBackward { .. }
        ) {
            anyhow::bail!(
                "PTX backend does not support {} at node {}",
                node.name(),
                node_index.index()
            );
        }
    }
    Ok(())
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
