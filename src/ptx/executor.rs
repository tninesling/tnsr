use super::{Module, PtxGraph};
use crate::Executor;
use crate::graph::{TensorGraph, TensorGraphNode, WithGrad};
use crate::tile::TileGraph;
use anyhow::{Context as _, Result};
use cudarc::driver::{CudaContext, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use petgraph::visit::IntoNodeReferences;
use std::collections::HashMap;
use std::sync::Arc;

/// PTX executor that compiles TensorGraph → TileGraph → PtxGraph → PTX string
/// and executes kernels via cudarc
pub struct PtxExecutor {
    device: Arc<CudaContext>,
    module: Option<Arc<CudaModule>>,
    values: HashMap<petgraph::graph::NodeIndex, CudaSlice<f32>>,
    /// Stores the PtxGraph to access kernel names during execution
    ptx_graph: Option<PtxGraph>,
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
            ptx_graph: None,
        })
    }

    /// Compile an owned TensorGraph to PTX without cloning during lowering.
    ///
    /// Prefer this over compiling from a reference when you have an owned graph.
    pub fn compile_owned<G>(&mut self, graph: TensorGraph<f32, G>) -> Result<()> {
        let tile_graph: TileGraph = graph.into();
        let ptx_graph: PtxGraph = tile_graph.into();

        let mut module = Module::new();

        // Functions already have unique names from PtxGraph conversion
        for (_node_idx, function) in ptx_graph.graph.node_references() {
            module.add_function(function.clone());
        }

        let ptx_src = module.to_string();

        let cuda_module = self
            .device
            .load_module(Ptx::from_src(&ptx_src))
            .context("Failed to load PTX module with cudarc")?;
        self.module = Some(cuda_module);
        self.ptx_graph = Some(ptx_graph);

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
        self.values.get(&node_idx).map(|cuda_slice| {
            let mut host_vec = vec![0.0f32; cuda_slice.len()];
            self.device
                .default_stream()
                .memcpy_dtoh(cuda_slice, &mut host_vec)
                .unwrap();
            host_vec
        })
    }

    /// Execute a compiled graph with the given inputs
    pub fn execute_compiled<G>(
        &mut self,
        graph: &TensorGraph<f32, G>,
        inputs: HashMap<String, Vec<f32>>,
    ) -> Result<Vec<f32>> {
        let module = self
            .module
            .as_ref()
            .context("No compiled module available. Call compile() first.")?;

        let ptx_graph = self
            .ptx_graph
            .as_ref()
            .context("No PTX graph available. Call compile() first.")?;

        // Clear values from previous execution
        self.values.clear();

        let order = graph.toposort();

        for node_idx in order.iter() {
            let node = &graph[*node_idx];
            let result = match node {
                TensorGraphNode::Constant { data } => {
                    let stream = self.device.default_stream();
                    let mut device_data = stream
                        .alloc_zeros::<f32>(data.len())
                        .context("Failed to allocate CUDA memory for constant")?;
                    stream
                        .memcpy_htod(data.as_slice(), &mut device_data)
                        .context("Failed to copy constant to CUDA device")?;
                    device_data
                }
                TensorGraphNode::Input { name } => {
                    let val = inputs
                        .get::<str>(name)
                        .with_context(|| format!("Input '{}' not found", name))?
                        .clone();
                    let stream = self.device.default_stream();
                    let mut device_data = stream
                        .alloc_zeros::<f32>(val.len())
                        .context("Failed to allocate CUDA memory for input")?;
                    stream
                        .memcpy_htod(&val, &mut device_data)
                        .context("Failed to copy input to CUDA device")?;
                    device_data
                }
                TensorGraphNode::Parameter { data, .. } => {
                    use anyhow::anyhow;
                    let v = data
                        .lock()
                        .map_err(|e| anyhow!("Failed to lock parameter data: {}", e))?;
                    let stream = self.device.default_stream();
                    let mut device_data = stream
                        .alloc_zeros::<f32>(v.len())
                        .context("Failed to allocate CUDA memory for parameter")?;
                    stream
                        .memcpy_htod(v.as_slice(), &mut device_data)
                        .context("Failed to copy parameter to CUDA device")?;
                    device_data
                }
                TensorGraphNode::Unary { op: _ } => {
                    let kernel_name = &ptx_graph.graph[*node_idx].name;

                    let ins = graph.inputs(*node_idx);
                    let input = self
                        .values
                        .get(&ins[0])
                        .context("Missing input value for unary operation")?;

                    let len = input.len();
                    let stream = self.device.default_stream();
                    let mut out = stream.alloc_zeros::<f32>(len).unwrap();

                    let f = module.load_function(kernel_name)?;
                    let cfg = LaunchConfig::for_num_elems(len as u32);
                    let mut launcher = stream.launch_builder(&f);
                    launcher.arg(input);
                    launcher.arg(&mut out);
                    unsafe { launcher.launch(cfg) }
                        .with_context(|| format!("CUDA {} kernel launch failed", kernel_name))?;

                    out
                }
                TensorGraphNode::Binary { op: _ } => {
                    let kernel_name = &ptx_graph.graph[*node_idx].name;

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
                    let stream = self.device.default_stream();
                    let mut out = stream.alloc_zeros::<f32>(len).unwrap();

                    let f = module.load_function(kernel_name)?;
                    let cfg = LaunchConfig::for_num_elems(len as u32);
                    let mut launcher = stream.launch_builder(&f);
                    launcher.arg(lhs);
                    launcher.arg(rhs);
                    launcher.arg(&mut out);
                    unsafe { launcher.launch(cfg) }
                        .with_context(|| format!("CUDA {} kernel launch failed", kernel_name))?;

                    out
                }
                TensorGraphNode::Gt => {
                    let kernel_name = &ptx_graph.graph[*node_idx].name;

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
                    let stream = self.device.default_stream();
                    let mut out = stream.alloc_zeros::<f32>(len).unwrap();

                    let f = module.load_function(kernel_name)?;
                    let cfg = LaunchConfig::for_num_elems(len as u32);
                    let mut launcher = stream.launch_builder(&f);
                    launcher.arg(lhs);
                    launcher.arg(rhs);
                    launcher.arg(&mut out);
                    unsafe { launcher.launch(cfg) }
                        .with_context(|| format!("CUDA {} kernel launch failed", kernel_name))?;

                    out
                }
                TensorGraphNode::Mask => {
                    let kernel_name = &ptx_graph.graph[*node_idx].name;

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
                    let stream = self.device.default_stream();
                    let mut out = stream.alloc_zeros::<f32>(len).unwrap();

                    let f = module.load_function(kernel_name)?;
                    let cfg = LaunchConfig::for_num_elems(len as u32);
                    let mut launcher = stream.launch_builder(&f);
                    launcher.arg(values);
                    launcher.arg(condition);
                    launcher.arg(&mut out);
                    unsafe { launcher.launch(cfg) }.context("CUDA mask kernel launch failed")?;

                    out
                }
                TensorGraphNode::MatMul => {
                    let kernel_name = &ptx_graph.graph[*node_idx].name;

                    let ins = graph.inputs(*node_idx);
                    let a = self
                        .values
                        .get(&ins[0])
                        .context("Missing A matrix for MatMul")?;
                    let b = self
                        .values
                        .get(&ins[1])
                        .context("Missing B matrix for MatMul")?;

                    // Get matrix dimensions from shapes
                    // A is [M, K], B is [K, N], output is [M, N]
                    let a_shape = graph
                        .shapes
                        .get(&ins[0])
                        .context("Missing shape for A matrix")?;
                    let b_shape = graph
                        .shapes
                        .get(&ins[1])
                        .context("Missing shape for B matrix")?;

                    if a_shape.len() != 2 || b_shape.len() != 2 {
                        anyhow::bail!(
                            "MatMul requires 2D matrices, got shapes {:?} and {:?}",
                            a_shape,
                            b_shape
                        );
                    }

                    let m = a_shape[0];
                    let k_a = a_shape[1];
                    let k_b = b_shape[0];
                    let n = b_shape[1];

                    if k_a != k_b {
                        anyhow::bail!(
                            "MatMul dimension mismatch: A is [{}, {}] but B is [{}, {}]",
                            m,
                            k_a,
                            k_b,
                            n
                        );
                    }

                    const TILE_SIZE: usize = 16;

                    // Pad dimensions to multiples of TILE_SIZE
                    let m_padded = m.div_ceil(TILE_SIZE) * TILE_SIZE;
                    let n_padded = n.div_ceil(TILE_SIZE) * TILE_SIZE;
                    let k_padded = k_a.div_ceil(TILE_SIZE) * TILE_SIZE;

                    let stream = self.device.default_stream();

                    // Allocate padded matrices (zero-initialized)
                    let mut a_padded = stream.alloc_zeros::<f32>(m_padded * k_padded).unwrap();
                    stream.synchronize().unwrap();
                    let mut b_padded = stream.alloc_zeros::<f32>(k_padded * n_padded).unwrap();
                    stream.synchronize().unwrap();

                    // Copy original data into padded matrices (row by row to handle padding)
                    for i in 0..m {
                        let src_offset = i * k_a;
                        let dst_offset = i * k_padded;
                        stream
                            .memcpy_dtod(
                                &a.slice(src_offset..(src_offset + k_a)),
                                &mut a_padded.slice_mut(dst_offset..(dst_offset + k_a)),
                            )
                            .unwrap();
                    }

                    for i in 0..k_a {
                        let src_offset = i * n;
                        let dst_offset = i * n_padded;
                        stream
                            .memcpy_dtod(
                                &b.slice(src_offset..(src_offset + n)),
                                &mut b_padded.slice_mut(dst_offset..(dst_offset + n)),
                            )
                            .unwrap();
                    }

                    // Allocate padded output
                    let mut out_padded = stream.alloc_zeros::<f32>(m_padded * n_padded).unwrap();

                    let f = module.load_function(kernel_name)?;

                    // Calculate grid dimensions based on padded sizes
                    let grid_x = n_padded / TILE_SIZE;
                    let grid_y = m_padded / TILE_SIZE;

                    let cfg = LaunchConfig {
                        grid_dim: (grid_x as u32, grid_y as u32, 1),
                        block_dim: (TILE_SIZE as u32, TILE_SIZE as u32, 1), // 16x16 thread block
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&f);
                    launcher.arg(&a_padded);
                    launcher.arg(&b_padded);
                    launcher.arg(&mut out_padded);
                    unsafe { launcher.launch(cfg) }
                        .with_context(|| format!("CUDA {} kernel launch failed", kernel_name))?;

                    // Extract the unpadded result from the padded output
                    let out_len = m * n;

                    // IMPORTANT: Drop input padded buffers BEFORE allocating output
                    // This prevents CUDA's allocator from reusing their memory
                    drop(a_padded);
                    drop(b_padded);
                    stream.synchronize().unwrap();

                    // Allocate final output buffer
                    let mut out = stream.alloc_zeros::<f32>(out_len).unwrap();

                    // Copy unpadded rows from out_padded to out (device-to-device)
                    for i in 0..m {
                        let src_offset = i * n_padded;
                        let dst_offset = i * n;
                        stream
                            .memcpy_dtod(
                                &out_padded.slice(src_offset..(src_offset + n)),
                                &mut out.slice_mut(dst_offset..(dst_offset + n)),
                            )
                            .unwrap();
                    }

                    // Drop output padded buffer after copying
                    drop(out_padded);
                    stream.synchronize().unwrap();

                    out
                }
                TensorGraphNode::BroadcastAxis { .. } => {
                    let kernel_name = &ptx_graph.graph[*node_idx].name;

                    let ins = graph.inputs(*node_idx);
                    let input = self
                        .values
                        .get(&ins[0])
                        .context("Missing input for BroadcastAxis")?;

                    // Get output shape
                    let out_shape = graph
                        .shapes
                        .get(node_idx)
                        .context("Missing output shape for BroadcastAxis")?;
                    let out_len: usize = out_shape.iter().product();

                    let stream = self.device.default_stream();
                    let mut out = stream.alloc_zeros::<f32>(out_len).unwrap();

                    let f = module.load_function(kernel_name)?;
                    let cfg = LaunchConfig::for_num_elems(out_len as u32);
                    let mut launcher = stream.launch_builder(&f);
                    launcher.arg(input);
                    launcher.arg(&mut out);
                    unsafe { launcher.launch(cfg) }
                        .with_context(|| format!("CUDA {} kernel launch failed", kernel_name))?;

                    out
                }
                TensorGraphNode::ReduceAxis { .. } => {
                    let kernel_name = &ptx_graph.graph[*node_idx].name;

                    let ins = graph.inputs(*node_idx);
                    let input = self
                        .values
                        .get(&ins[0])
                        .context("Missing input for ReduceAxis")?;

                    // Get output shape
                    let out_shape = graph
                        .shapes
                        .get(node_idx)
                        .context("Missing output shape for ReduceAxis")?;
                    let out_len: usize = out_shape.iter().product();

                    let stream = self.device.default_stream();
                    let mut out = stream.alloc_zeros::<f32>(out_len).unwrap();

                    let f = module.load_function(kernel_name)?;
                    let cfg = LaunchConfig::for_num_elems(out_len as u32);
                    let mut launcher = stream.launch_builder(&f);
                    launcher.arg(input);
                    launcher.arg(&mut out);
                    unsafe { launcher.launch(cfg) }
                        .with_context(|| format!("CUDA {} kernel launch failed", kernel_name))?;

                    out
                }
                TensorGraphNode::Transpose => {
                    let kernel_name = &ptx_graph.graph[*node_idx].name;

                    let ins = graph.inputs(*node_idx);
                    let input = self
                        .values
                        .get(&ins[0])
                        .context("Missing input for Transpose")?;

                    let len = input.len();
                    let stream = self.device.default_stream();
                    let mut out = unsafe { stream.alloc::<f32>(len).unwrap() };

                    let f = module.load_function(kernel_name)?;
                    let cfg = LaunchConfig::for_num_elems(len as u32);
                    let mut launcher = stream.launch_builder(&f);
                    launcher.arg(input);
                    launcher.arg(&mut out);
                    unsafe { launcher.launch(cfg) }
                        .with_context(|| format!("CUDA {} kernel launch failed", kernel_name))?;

                    out
                }
            };

            self.values.insert(*node_idx, result);
        }

        let last_node_idx = order.last().context("Graph is empty")?;
        let out_device = self
            .values
            .get(last_node_idx)
            .context("Output value not found after execution")?;
        let mut out_host = vec![0.0f32; out_device.len()];
        self.device
            .default_stream()
            .memcpy_dtoh(out_device, &mut out_host)
            .context("Failed to copy output from CUDA device to host")?;
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

impl Executor<f32> for PtxExecutor {
    fn execute<G>(
        &mut self,
        graph: &TensorGraph<f32, G>,
        inputs: HashMap<String, Vec<f32>>,
    ) -> Result<Vec<f32>>
    where
        TensorGraph<f32, G>: Clone,
    {
        // Compile the graph on first execution
        if self.module.is_none() {
            // We need to clone here since compile_owned needs ownership
            // This only happens once per graph
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
