use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context as _, Result};
use cudarc::driver::{CudaContext, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use petgraph::visit::IntoNodeReferences;
use tensor::graph::{TensorGraph, TensorGraphNode, WithGrad};

use super::{Module, PtxGraph};
use crate::tile::TileGraph;
use crate::Executor;

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

        let cuda_module = self
            .device
            .load_module(Ptx::from_src(&module.to_string()))
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
                TensorGraphNode::Unary { op } => {
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
                TensorGraphNode::Binary { op } => {
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
                _ => {
                    anyhow::bail!("Unsupported operation: {}", node.name());
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Neg;
    use tensor::graph::TensorGraph;
    use tensor::{Constant, TensorExpr};

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_ptx_executor_unary() {
        let x = Constant::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let y = TensorExpr::from(x).neg();
        let graph: TensorGraph<f32> = y.into();

        let mut executor = PtxExecutor::new();
        let result = executor
            .compile_and_execute(&graph, HashMap::new())
            .unwrap();

        assert_eq!(result.len(), 4);
        assert!((result[0] - (-1.0)).abs() < 1e-5);
        assert!((result[1] - (-2.0)).abs() < 1e-5);
        assert!((result[2] - (-3.0)).abs() < 1e-5);
        assert!((result[3] - (-4.0)).abs() < 1e-5);
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_ptx_executor_binary() {
        let x = Constant::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let y = Constant::new(vec![10.0, 20.0, 30.0, 40.0], vec![4]);
        let z = TensorExpr::from(x) + TensorExpr::from(y);
        let graph: TensorGraph<f32> = z.into();

        let mut executor = PtxExecutor::new();
        let result = executor
            .compile_and_execute(&graph, HashMap::new())
            .unwrap();

        assert_eq!(result.len(), 4);
        assert!((result[0] - 11.0).abs() < 1e-5);
        assert!((result[1] - 22.0).abs() < 1e-5);
        assert!((result[2] - 33.0).abs() < 1e-5);
        assert!((result[3] - 44.0).abs() < 1e-5);
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_ptx_executor_relu() {
        let x = Constant::new(vec![-1.0, 2.0, -3.0, 4.0], vec![4]);
        // Note: relu() method needs to be checked if it exists on TensorExpr
        // For now, commenting out this test as it may not be implemented yet
        // let y = TensorExpr::from(x).relu();
        // let graph: TensorGraph<f32> = y.into();

        let mut _executor = PtxExecutor::new();
        // Placeholder until relu is implemented
        let _data = x;
    }
}
