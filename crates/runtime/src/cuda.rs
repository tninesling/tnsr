use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context as _, Result, anyhow};
use cudarc::driver::{CudaContext, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use tensor::graph::{TensorGraph, TensorGraphNode, WithGrad};

use crate::Executor;
use tracing::trace_span;

static PTX: &str = include_str!("cuda-kernels.ptx");

pub struct CudaExecutor {
    device: Arc<CudaContext>,
    module: Arc<CudaModule>,
    values: HashMap<petgraph::graph::NodeIndex, CudaSlice<f32>>,
    grads: HashMap<petgraph::graph::NodeIndex, CudaSlice<f32>>,
}

impl Default for CudaExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaExecutor {
    pub fn new() -> Self {
        let device = CudaContext::new(0).expect("Failed to initialize CUDA device 0");
        let module = device
            .load_module(Ptx::from_src(PTX))
            .expect("Failed to load PTX module with cudarc");
        CudaExecutor {
            device,
            module,
            values: HashMap::new(),
            grads: HashMap::new(),
        }
    }

    fn neg(&self, input: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("neg").entered();
        let len = input.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("neg").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(input);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA neg failed");
        out
    }

    fn exp(&self, input: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("exp").entered();
        let len = input.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("exp").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(input);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA exp failed");
        out
    }

    fn log(&self, input: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("log").entered();
        let len = input.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("log").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(input);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA log failed");
        out
    }

    fn relu(&self, input: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("relu").entered();
        let len = input.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("relu").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(input);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA relu failed");
        out
    }

    fn add(&self, lhs: &CudaSlice<f32>, rhs: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("add").entered();
        assert_eq!(lhs.len(), rhs.len(), "binary op input length mismatch");
        let len = lhs.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("add").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(lhs);
        launcher.arg(&len_u64);
        launcher.arg(rhs);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA add failed");
        out
    }

    fn sub(&self, lhs: &CudaSlice<f32>, rhs: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("sub").entered();
        assert_eq!(lhs.len(), rhs.len(), "binary op input length mismatch");
        let len = lhs.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("sub").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(lhs);
        launcher.arg(&len_u64);
        launcher.arg(rhs);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA sub failed");
        out
    }

    fn mul(&self, lhs: &CudaSlice<f32>, rhs: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("mul").entered();
        assert_eq!(lhs.len(), rhs.len(), "binary op input length mismatch");
        let len = lhs.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("mul").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(lhs);
        launcher.arg(&len_u64);
        launcher.arg(rhs);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA mul failed");
        out
    }

    fn div(&self, lhs: &CudaSlice<f32>, rhs: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("div").entered();
        assert_eq!(lhs.len(), rhs.len(), "binary op input length mismatch");
        let len = lhs.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("div").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(lhs);
        launcher.arg(&len_u64);
        launcher.arg(rhs);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA div failed");
        out
    }

    fn gt(&self, lhs: &CudaSlice<f32>, rhs: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("gt").entered();
        assert_eq!(lhs.len(), rhs.len(), "binary op input length mismatch");
        let len = lhs.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("gt").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(lhs);
        launcher.arg(&len_u64);
        launcher.arg(rhs);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA gt failed");
        out
    }

    fn mask(&self, values: &CudaSlice<f32>, condition: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("mask").entered();
        assert_eq!(
            values.len(),
            condition.len(),
            "binary op input length mismatch"
        );
        let len = values.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("mask").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(values);
        launcher.arg(&len_u64);
        launcher.arg(condition);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA mask failed");
        out
    }

    fn matmul(
        &self,
        lhs: &CudaSlice<f32>,
        rhs: &CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> CudaSlice<f32> {
        let _span = trace_span!("matmul").entered();
        let stream = self.device.default_stream();

        let out_len = m * n;
        let mut out = stream.alloc_zeros::<f32>(m * n).unwrap();

        let f = self
            .module
            .load_function("matmul")
            .expect("Failed to load matmul function");
        let cfg = LaunchConfig::for_num_elems(out_len as u32);
        let lhs_len_u64 = lhs.len() as u64;
        let rhs_len_u64 = rhs.len() as u64;
        let m_u64 = m as u64;
        let n_u64 = n as u64;
        let k_u64 = k as u64;
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(lhs);
        launcher.arg(&lhs_len_u64);
        launcher.arg(rhs);
        launcher.arg(&rhs_len_u64);
        launcher.arg(&mut out);
        launcher.arg(&m_u64);
        launcher.arg(&n_u64);
        launcher.arg(&k_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA matmul failed");

        out
    }

    fn transpose(&self, input: &CudaSlice<f32>, rows: usize, cols: usize) -> CudaSlice<f32> {
        let _span = trace_span!("transpose").entered();
        let len = input.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<f32>(len).unwrap();
        let f = self.module.load_function("transpose_2d").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let rows_u64 = rows as u64;
        let cols_u64 = cols as u64;
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(input);
        launcher.arg(&len_u64);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&rows_u64);
        launcher.arg(&cols_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA transpose failed");
        out
    }

    /// Get the value of a specific node from the executor's cache after a forward pass
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
}

impl Executor<f32> for CudaExecutor {
    fn forward<G>(
        &mut self,
        graph: &TensorGraph<f32, G>,
        inputs: HashMap<String, Vec<f32>>,
    ) -> Result<Vec<f32>> {
        let order = graph.toposort();
        let _fwd_span = trace_span!("forward", nodes = order.len()).entered();

        for node_idx in order.iter() {
            let node = &graph[*node_idx];
            let result = match node {
                TensorGraphNode::Constant { data } => {
                    let _span = trace_span!("constant", node = node_idx.index(), size = data.len())
                        .entered();
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
                    let _span =
                        trace_span!("input", node = node_idx.index(), name = name).entered();
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
                TensorGraphNode::Unary { op } => {
                    let inputs = graph.inputs(*node_idx);
                    let input = self
                        .values
                        .get(&inputs[0])
                        .context("Missing input value for unary operation")?;
                    match op {
                        tensor::UnaryOp::Neg => self.neg(input),
                        tensor::UnaryOp::Exp => self.exp(input),
                        tensor::UnaryOp::Log => self.log(input),
                        tensor::UnaryOp::Relu => self.relu(input),
                    }
                }
                TensorGraphNode::Binary { op } => {
                    let ins = graph.inputs(*node_idx);
                    let lhs = self
                        .values
                        .get(&ins[0])
                        .context("Missing left operand for binary operation")?;
                    let rhs = self
                        .values
                        .get(&ins[1])
                        .context("Missing right operand for binary operation")?;
                    match op {
                        tensor::BinaryOp::Add => self.add(lhs, rhs),
                        tensor::BinaryOp::Sub => self.sub(lhs, rhs),
                        tensor::BinaryOp::Mul => self.mul(lhs, rhs),
                        tensor::BinaryOp::Div => self.div(lhs, rhs),
                    }
                }
                TensorGraphNode::MatMul => {
                    let ins = graph.inputs(*node_idx);
                    let lhs = self
                        .values
                        .get(&ins[0])
                        .context("Missing left operand for matmul")?;
                    let rhs = self
                        .values
                        .get(&ins[1])
                        .context("Missing right operand for matmul")?;
                    let lhs_shape = graph
                        .shapes
                        .get(&ins[0])
                        .context("Missing shape for matmul left operand")?;
                    let rhs_shape = graph
                        .shapes
                        .get(&ins[1])
                        .context("Missing shape for matmul right operand")?;

                    anyhow::ensure!(
                        lhs_shape.len() == 2,
                        "MatMul expects 2D left operand, got {}D",
                        lhs_shape.len()
                    );
                    anyhow::ensure!(
                        rhs_shape.len() == 2,
                        "MatMul expects 2D right operand, got {}D",
                        rhs_shape.len()
                    );

                    let m = lhs_shape[0];
                    let k = lhs_shape[1];
                    let n = rhs_shape[1];

                    anyhow::ensure!(
                        k == rhs_shape[0],
                        "MatMul inner dimension mismatch: lhs has k={}, rhs has {}",
                        k,
                        rhs_shape[0]
                    );

                    self.matmul(lhs, rhs, m, n, k)
                }
                TensorGraphNode::Parameter { data, .. } => {
                    let v = data
                        .lock()
                        .map_err(|e| anyhow!("Failed to lock parameter data: {}", e))?;
                    let _span =
                        trace_span!("parameter", node = node_idx.index(), size = v.len()).entered();
                    let stream = self.device.default_stream();
                    let mut device_data = stream
                        .alloc_zeros::<f32>(v.len())
                        .context("Failed to allocate CUDA memory for parameter")?;
                    stream
                        .memcpy_htod(v.as_slice(), &mut device_data)
                        .context("Failed to copy parameter to CUDA device")?;
                    device_data
                }
                TensorGraphNode::BroadcastAxis { axis } => {
                    let stream = self.device.default_stream();
                    let in_idx = graph.inputs(*node_idx)[0];
                    let in_val = self
                        .values
                        .get(&in_idx)
                        .context("Missing input value for broadcast")?;
                    let in_shape = graph
                        .shapes
                        .get(&in_idx)
                        .context("Missing input shape for broadcast")?;
                    let out_shape = graph
                        .shapes
                        .get(node_idx)
                        .context("Missing output shape for broadcast")?;
                    let out_size: usize = out_shape.iter().product();

                    anyhow::ensure!(
                        in_shape.len() == out_shape.len(),
                        "BroadcastAxis requires matching rank, got input rank {} and output rank {}",
                        in_shape.len(),
                        out_shape.len()
                    );
                    anyhow::ensure!(
                        in_shape[*axis] == 1,
                        "BroadcastAxis requires input axis {} size to be 1, got {}",
                        axis,
                        in_shape[*axis]
                    );

                    if out_shape.len() == 1 {
                        // For 1D BroadcastAxis, we're broadcasting [1] to [n]
                        // This is just filling the output with the single input value
                        anyhow::ensure!(
                            *axis == 0,
                            "BroadcastAxis axis {} out of bounds for 1D tensor",
                            axis
                        );
                        
                        let n = out_shape[0];
                        let mut out = stream
                            .alloc_zeros::<f32>(out_size)
                            .context("Failed to allocate CUDA memory for broadcast output")?;
                        
                        // Copy the single value from device to host, then broadcast to all positions
                        let mut single_value = vec![0.0f32; 1];
                        stream
                            .memcpy_dtoh(in_val, &mut single_value)
                            .context("Failed to copy single value from device")?;
                        
                        // Fill output with the single value
                        let host_out = vec![single_value[0]; n];
                        stream
                            .memcpy_htod(&host_out, &mut out)
                            .context("Failed to copy broadcast result to device")?;
                        
                        out
                    } else if out_shape.len() == 2 {
                        let m = out_shape[0];
                        let n = out_shape[1];
                        let m_u64 = m as u64;
                        let n_u64 = n as u64;
                        let mut out = stream
                            .alloc_zeros::<f32>(out_size)
                            .context("Failed to allocate CUDA memory for broadcast output")?;

                        if *axis == 0 {
                            let f = self
                                .module
                                .load_function("broadcast_row")
                                .context("Failed to load broadcast_row kernel")?;
                            let cfg = LaunchConfig::for_num_elems(n as u32);
                            let mut launcher = stream.launch_builder(&f);
                            launcher.arg(in_val);
                            launcher.arg(&n_u64);
                            launcher.arg(&mut out);
                            launcher.arg(&m_u64);
                            launcher.arg(&n_u64);
                            unsafe { launcher.launch(cfg) }
                                .context("CUDA broadcast_row kernel launch failed")?;
                        } else if *axis == 1 {
                            let f = self
                                .module
                                .load_function("broadcast_col")
                                .context("Failed to load broadcast_col kernel")?;
                            let cfg = LaunchConfig::for_num_elems(m as u32);
                            let mut launcher = stream.launch_builder(&f);
                            launcher.arg(in_val);
                            launcher.arg(&m_u64);
                            launcher.arg(&mut out);
                            launcher.arg(&m_u64);
                            launcher.arg(&n_u64);
                            unsafe { launcher.launch(cfg) }
                                .context("CUDA broadcast_col kernel launch failed")?;
                        } else {
                            anyhow::bail!(
                                "BroadcastAxis axis {} out of bounds for 2D tensor",
                                axis
                            );
                        }
                        out
                    } else {
                        anyhow::bail!(
                            "BroadcastAxis currently only supports 1D and 2D tensors on GPU, got {}D",
                            out_shape.len()
                        );
                    }
                }
                TensorGraphNode::ReduceAxis { op, axis } => {
                    let in_idx = graph.inputs(*node_idx)[0];
                    let x_device = self
                        .values
                        .get(&in_idx)
                        .context("Missing input value for reduce")?;
                    let in_shape = graph
                        .shapes
                        .get(&in_idx)
                        .context("Missing input shape for reduce")?;
                    let out_shape = graph
                        .shapes
                        .get(node_idx)
                        .context("Missing output shape for reduce")?;
                    let out_len: usize = out_shape.iter().product();

                    anyhow::ensure!(
                        in_shape.len() == out_shape.len(),
                        "ReduceAxis preserves rank: input has {}D but output has {}D",
                        in_shape.len(),
                        out_shape.len()
                    );
                    anyhow::ensure!(
                        out_shape[*axis] == 1,
                        "ReduceAxis output axis {} must be 1, got {}",
                        axis,
                        out_shape[*axis]
                    );

                    let stream = self.device.default_stream();

                    if in_shape.len() == 1 {
                        // For 1D ReduceAxis, we're reducing [n] to [1]
                        anyhow::ensure!(
                            *axis == 0,
                            "ReduceAxis axis {} out of bounds for 1D tensor",
                            axis
                        );
                        
                        let n = in_shape[0];
                        
                        // Copy data from device to host for reduction
                        let mut host_data = vec![0.0f32; n];
                        stream
                            .memcpy_dtoh(x_device, &mut host_data)
                            .context("Failed to copy data from device for reduce")?;
                        
                        // Perform reduction on CPU
                        let result = match op {
                            tensor::ReduceOp::Sum => host_data.iter().sum::<f32>(),
                            tensor::ReduceOp::Max => host_data.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                            tensor::ReduceOp::Mean => {
                                let sum: f32 = host_data.iter().sum();
                                sum / n as f32
                            }
                        };
                        
                        // Copy result back to device
                        let mut out_device = stream
                            .alloc_zeros::<f32>(1)
                            .context("Failed to allocate CUDA memory for reduce output")?;
                        stream
                            .memcpy_htod(&[result], &mut out_device)
                            .context("Failed to copy reduce result to device")?;
                        
                        out_device
                    } else if in_shape.len() == 2 {
                        let m = in_shape[0];
                        let n = in_shape[1];
                        let m_u64 = m as u64;
                        let n_u64 = n as u64;
                        let in_len_u64 = x_device.len() as u64;
                        let mut out_device = stream
                            .alloc_zeros::<f32>(out_len)
                            .context("Failed to allocate CUDA memory for reduce output")?;

                        if *axis == 1 {
                            let kernel_name = match op {
                                tensor::ReduceOp::Sum => "reduce_sum_rows",
                                tensor::ReduceOp::Max => "reduce_max_rows",
                                tensor::ReduceOp::Mean => "reduce_mean_rows",
                            };
                            let f = self.module.load_function(kernel_name).with_context(|| {
                                format!("Failed to load {} kernel", kernel_name)
                            })?;
                            let cfg = LaunchConfig::for_num_elems(m as u32);
                            let out_len_u64 = m_u64;
                            let mut launcher = stream.launch_builder(&f);
                            launcher.arg(x_device);
                            launcher.arg(&in_len_u64);
                            launcher.arg(&mut out_device);
                            launcher.arg(&out_len_u64);
                            launcher.arg(&n_u64);
                            unsafe { launcher.launch(cfg) }.with_context(|| {
                                format!("CUDA {} kernel launch failed", kernel_name)
                            })?;
                        } else if *axis == 0 {
                            let kernel_name = match op {
                                tensor::ReduceOp::Sum => "reduce_sum_cols",
                                tensor::ReduceOp::Max => "reduce_max_cols",
                                tensor::ReduceOp::Mean => "reduce_mean_cols",
                            };
                            let f = self.module.load_function(kernel_name).with_context(|| {
                                format!("Failed to load {} kernel", kernel_name)
                            })?;
                            let cfg = LaunchConfig::for_num_elems(n as u32);
                            let out_len_u64 = n_u64;
                            let mut launcher = stream.launch_builder(&f);
                            launcher.arg(x_device);
                            launcher.arg(&in_len_u64);
                            launcher.arg(&mut out_device);
                            launcher.arg(&m_u64);
                            launcher.arg(&out_len_u64);
                            unsafe { launcher.launch(cfg) }.with_context(|| {
                                format!("CUDA {} kernel launch failed", kernel_name)
                            })?;
                        } else {
                            anyhow::bail!("ReduceAxis axis {} out of bounds for 2D tensor", axis);
                        }

                        out_device
                    } else {
                        anyhow::bail!(
                            "ReduceAxis currently only supports 1D and 2D tensors on GPU, got {}D",
                            in_shape.len()
                        );
                    }
                }
                TensorGraphNode::Transpose => {
                    let in_idx = graph.inputs(*node_idx)[0];
                    let input = self
                        .values
                        .get(&in_idx)
                        .context("Missing input value for transpose")?;
                    let in_shape = graph
                        .shapes
                        .get(&in_idx)
                        .context("Missing input shape for transpose")?;

                    anyhow::ensure!(
                        in_shape.len() == 2,
                        "Transpose currently only supports 2D tensors on GPU, got {}D",
                        in_shape.len()
                    );

                    let rows = in_shape[0];
                    let cols = in_shape[1];
                    self.transpose(input, rows, cols)
                }
                TensorGraphNode::Gt => {
                    let ins = graph.inputs(*node_idx);
                    let lhs = self
                        .values
                        .get(&ins[0])
                        .context("Missing left operand for Gt")?;
                    let rhs = self
                        .values
                        .get(&ins[1])
                        .context("Missing right operand for Gt")?;
                    self.gt(lhs, rhs)
                }
                TensorGraphNode::Mask => {
                    let ins = graph.inputs(*node_idx);
                    let values = self
                        .values
                        .get(&ins[0])
                        .context("Missing values for Mask")?;
                    let condition = self
                        .values
                        .get(&ins[1])
                        .context("Missing condition for Mask")?;
                    self.mask(values, condition)
                }
            };
            self.values.insert(*node_idx, result);
        }

        let last_node_idx = order.last().context("Graph is empty")?;
        let out_device = self
            .values
            .get(last_node_idx)
            .context("Output value not found after forward pass")?;
        let mut out_host = vec![0.0f32; out_device.len()];
        self.device
            .default_stream()
            .memcpy_dtoh(out_device, &mut out_host)
            .context("Failed to copy output from CUDA device to host")?;
        Ok(out_host)
    }

    fn get_gradients(&self, graph: &TensorGraph<f32, WithGrad>) -> HashMap<usize, Vec<f32>> {
        let metadata = graph.gradient_metadata();

        let mut grads = HashMap::new();
        for (param_id, grad_node) in &metadata.param_to_grad {
            let grad_value_gpu = self
                .values
                .get(grad_node)
                .expect("Gradient value not computed");

            // Copy gradient from GPU to host
            let mut grad_host = vec![0.0f32; grad_value_gpu.len()];
            self.device
                .default_stream()
                .memcpy_dtoh(grad_value_gpu, &mut grad_host)
                .expect("Failed to copy gradient from CUDA device to host");

            grads.insert(*param_id, grad_host);
        }
        grads
    }
}

