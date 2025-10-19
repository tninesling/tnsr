use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use tensor::graph::{TensorGraph, TensorGraphNode};

use crate::{BackwardResult, Executor};
use tracing::{Level, debug, span, trace};

static PTX: &str = include_str!("cuda-kernels.ptx");

pub struct CudaExecutor {
    device: Arc<CudaContext>,
    module: Arc<CudaModule>,
}

impl CudaExecutor {
    pub fn new() -> Self {
        let device = CudaContext::new(0).expect("Failed to initialize CUDA device 0");
        let module = device
            .load_module(Ptx::from_src(PTX))
            .expect("Failed to load PTX module with cudarc");
        CudaExecutor { device, module }
    }
}

impl Executor<f32> for CudaExecutor {
    fn execute(&self, graph: &TensorGraph<f32>, inputs: HashMap<String, Vec<f32>>) -> Vec<f32> {
        let order = graph.toposort();
        let exec_span = span!(Level::TRACE, "cuda_execute_graph", nodes = order.len());
        let _eg = exec_span.enter();
        let mut values: HashMap<_, CudaSlice<f32>> = HashMap::with_capacity(order.len());
        for node_idx in order.iter() {
            let node = &graph[*node_idx];
            let result = match node {
                TensorGraphNode::Constant { data } => {
                    trace!(idx = node_idx.index(), op = "Constant");
                    let stream = self.device.default_stream();
                    let mut device_data = stream.alloc_zeros::<f32>(data.len()).unwrap();
                    stream
                        .memcpy_htod(data.as_slice(), &mut device_data)
                        .unwrap();
                    device_data
                }
                TensorGraphNode::Input { name } => {
                    let val = inputs
                        .get(&name.to_string())
                        .expect("Input not found")
                        .clone();
                    trace!(idx = node_idx.index(), op = "Input", name = *name);
                    let stream = self.device.default_stream();
                    let mut device_data = stream.alloc_zeros::<f32>(val.len()).unwrap();
                    stream.memcpy_htod(&val, &mut device_data).unwrap();
                    device_data
                }
                TensorGraphNode::Unary { op } => {
                    let inputs = graph.inputs(*node_idx);
                    let input = values.get(&inputs[0]).unwrap();
                    let len = input.len();
                    let len_u64 = len as u64;

                    let stream = self.device.default_stream();
                    let mut out = stream.alloc_zeros::<f32>(len).unwrap();

                    match op {
                        tensor::UnaryOp::Neg => {
                            trace!(idx = node_idx.index(), op = "Neg");
                            let f = self.module.load_function("neg").unwrap();
                            let cfg = LaunchConfig::for_num_elems(len as u32);
                            let mut launcher = stream.launch_builder(&f);
                            launcher.arg(input);
                            launcher.arg(&len_u64);
                            launcher.arg(&mut out);
                            launcher.arg(&len_u64);
                            unsafe { launcher.launch(cfg) }.expect("CUDA neg failed");
                        }
                        tensor::UnaryOp::Exp => {
                            trace!(idx = node_idx.index(), op = "Exp");
                            let f = self.module.load_function("exp").unwrap();
                            let cfg = LaunchConfig::for_num_elems(len as u32);
                            let mut launcher = stream.launch_builder(&f);
                            launcher.arg(input);
                            launcher.arg(&len_u64);
                            launcher.arg(&mut out);
                            launcher.arg(&len_u64);
                            unsafe { launcher.launch(cfg) }.expect("CUDA exp failed");
                        }
                        tensor::UnaryOp::Log => {
                            trace!(idx = node_idx.index(), op = "Log");
                            let f = self.module.load_function("log").unwrap();
                            let cfg = LaunchConfig::for_num_elems(len as u32);
                            let mut launcher = stream.launch_builder(&f);
                            launcher.arg(input);
                            launcher.arg(&len_u64);
                            launcher.arg(&mut out);
                            launcher.arg(&len_u64);
                            unsafe { launcher.launch(cfg) }.expect("CUDA log failed");
                        }
                        tensor::UnaryOp::Relu => {
                            trace!(idx = node_idx.index(), op = "Relu");
                            let f = self.module.load_function("relu").unwrap();
                            let cfg = LaunchConfig::for_num_elems(len as u32);
                            let mut launcher = stream.launch_builder(&f);
                            launcher.arg(input);
                            launcher.arg(&len_u64);
                            launcher.arg(&mut out);
                            launcher.arg(&len_u64);
                            unsafe { launcher.launch(cfg) }.expect("CUDA relu failed");
                        }
                    }
                    out
                }
                TensorGraphNode::Binary { op } => {
                    let ins = graph.inputs(*node_idx);
                    let lhs = values.get(&ins[0]).unwrap();
                    let rhs = values.get(&ins[1]).unwrap();
                    assert_eq!(lhs.len(), rhs.len(), "binary op input length mismatch");
                    let len = lhs.len();
                    let len_u64 = len as u64;

                    let stream = self.device.default_stream();
                    let mut out = stream.alloc_zeros::<f32>(len).unwrap();

                    let (function_name, op_name) = match op {
                        tensor::BinaryOp::Add => ("add", "Add"),
                        tensor::BinaryOp::Sub => ("sub", "Sub"),
                        tensor::BinaryOp::Mul => ("mul", "Mul"),
                        tensor::BinaryOp::Div => ("div", "Div"),
                    };

                    trace!(idx = node_idx.index(), op = op_name);
                    let f = self.module.load_function(function_name).unwrap();
                    let cfg = LaunchConfig::for_num_elems(len as u32);
                    let mut launcher = stream.launch_builder(&f);
                    launcher.arg(lhs);
                    launcher.arg(&len_u64);
                    launcher.arg(rhs);
                    launcher.arg(&len_u64);
                    launcher.arg(&mut out);
                    launcher.arg(&len_u64);
                    unsafe { launcher.launch(cfg) }
                        .unwrap_or_else(|_| panic!("CUDA {} failed", function_name));
                    out
                }
                TensorGraphNode::MatMul => {
                    let ins = graph.inputs(*node_idx);
                    let lhs = values.get(&ins[0]).unwrap();
                    let rhs = values.get(&ins[1]).unwrap();
                    let lhs_shape = graph.shapes.get(&ins[0]).expect("shape missing for lhs");
                    let rhs_shape = graph.shapes.get(&ins[1]).expect("shape missing for rhs");
                    assert_eq!(lhs_shape.len(), 2, "MatMul expects 2D lhs");
                    assert_eq!(rhs_shape.len(), 2, "MatMul expects 2D rhs");
                    let m = lhs_shape[0];
                    let k = lhs_shape[1];
                    let n = rhs_shape[1];
                    assert_eq!(k, rhs_shape[0], "MatMul inner dim mismatch");

                    let out_len = m * n;
                    trace!(idx = node_idx.index(), op = "MatMul", m = m, n = n, k = k);
                    let lhs_len = lhs.len();
                    let rhs_len = rhs.len();

                    let stream = self.device.default_stream();
                    let mut out = stream.alloc_zeros::<f32>(out_len).unwrap();

                    let f = self.module.load_function("matmul").unwrap();
                    let cfg = LaunchConfig::for_num_elems(out_len as u32);
                    let lhs_len_u64 = lhs_len as u64;
                    let rhs_len_u64 = rhs_len as u64;
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
                TensorGraphNode::Reduce { op, axis } => {
                    let in_idx = graph.inputs(*node_idx)[0];
                    let x_device = values.get(&in_idx).unwrap();
                    let in_shape = graph.shapes.get(&in_idx).unwrap();
                    let out_shape = graph.shapes.get(node_idx).unwrap();
                    let in_len = x_device.len();
                    let out_len: usize = out_shape.iter().product();
                    trace!(idx = node_idx.index(), op = "Reduce", axis = *axis, in_len = in_len, out_len = out_len, kind = ?op, "launch");

                    let stream = self.device.default_stream();

                    if out_shape.is_empty() {
                        let total_threads_u64 = in_len as u64; // allocate one partial per element
                        let mut partials = stream
                            .alloc_zeros::<f32>(total_threads_u64 as usize)
                            .unwrap();
                        let mut out_device = stream.alloc_zeros::<f32>(1).unwrap();

                        // Pass 1: partials
                        let (partials_kernel, finalize_kernel) = match op {
                            tensor::ReduceOp::Sum => (
                                self.module.load_function("reduce_sum_partials").unwrap(),
                                self.module.load_function("reduce_sum_finalize").unwrap(),
                            ),
                            tensor::ReduceOp::Max => (
                                self.module.load_function("reduce_max_partials").unwrap(),
                                self.module.load_function("reduce_max_finalize").unwrap(),
                            ),
                            tensor::ReduceOp::Mean => (
                                self.module.load_function("reduce_mean_partials").unwrap(),
                                self.module.load_function("reduce_mean_finalize").unwrap(),
                            ),
                        };
                        let cfg_partials = LaunchConfig::for_num_elems(in_len as u32);
                        let in_len_u64 = in_len as u64;
                        let mut launcher = stream.launch_builder(&partials_kernel);
                        launcher.arg(x_device);
                        launcher.arg(&in_len_u64);
                        launcher.arg(&mut partials);
                        launcher.arg(&total_threads_u64);
                        unsafe { launcher.launch(cfg_partials) }
                            .expect("CUDA reduce partials failed");

                        // Pass 2: finalize
                        let cfg_finalize = LaunchConfig::for_num_elems(1);
                        let num_partials_u64 = total_threads_u64;
                        let partials_len_u64 = total_threads_u64;
                        let mut launcher2 = stream.launch_builder(&finalize_kernel);
                        launcher2.arg(&partials);
                        launcher2.arg(&partials_len_u64);
                        launcher2.arg(&mut out_device);
                        launcher2.arg(&num_partials_u64);

                        // Mean operation needs the original input length as 5th parameter for division
                        match op {
                            tensor::ReduceOp::Mean => {
                                launcher2.arg(&in_len_u64);
                            }
                            tensor::ReduceOp::Sum | tensor::ReduceOp::Max => {
                                // Sum and Max only need 4 parameters
                            }
                        }

                        unsafe { launcher2.launch(cfg_finalize) }
                            .expect("CUDA reduce finalize failed");

                        out_device
                    } else {
                        assert_eq!(
                            in_shape.len(),
                            2,
                            "Axis reduction currently supports only 2D tensors"
                        );
                        let m = in_shape[0];
                        let n = in_shape[1];
                        let mut out_device = stream.alloc_zeros::<f32>(out_len).unwrap();
                        let in_len_u64 = in_len as u64;
                        let m_u64 = m as u64;
                        let n_u64 = n as u64;
                        match axis {
                            // reduce across columns -> one per row
                            1 => {
                                let f = match op {
                                    tensor::ReduceOp::Sum => {
                                        self.module.load_function("reduce_sum_rows").unwrap()
                                    }
                                    tensor::ReduceOp::Max => {
                                        self.module.load_function("reduce_max_rows").unwrap()
                                    }
                                    tensor::ReduceOp::Mean => {
                                        self.module.load_function("reduce_mean_rows").unwrap()
                                    }
                                };
                                let cfg = LaunchConfig::for_num_elems(m as u32);
                                let out_len_u64 = m_u64;
                                let mut launcher = stream.launch_builder(&f);
                                launcher.arg(x_device);
                                launcher.arg(&in_len_u64);
                                launcher.arg(&mut out_device);
                                launcher.arg(&out_len_u64);
                                launcher.arg(&n_u64); // row_len
                                unsafe { launcher.launch(cfg) }.expect("CUDA reduce rows failed");
                            }
                            // reduce across rows -> one per column
                            0 => {
                                let f = match op {
                                    tensor::ReduceOp::Sum => {
                                        self.module.load_function("reduce_sum_cols").unwrap()
                                    }
                                    tensor::ReduceOp::Max => {
                                        self.module.load_function("reduce_max_cols").unwrap()
                                    }
                                    tensor::ReduceOp::Mean => {
                                        self.module.load_function("reduce_mean_cols").unwrap()
                                    }
                                };
                                let cfg = LaunchConfig::for_num_elems(n as u32);
                                let out_len_u64 = n_u64;
                                let mut launcher = stream.launch_builder(&f);
                                launcher.arg(x_device);
                                launcher.arg(&in_len_u64);
                                launcher.arg(&mut out_device);
                                launcher.arg(&m_u64); // rows
                                launcher.arg(&out_len_u64); // cols == out_len
                                unsafe { launcher.launch(cfg) }.expect("CUDA reduce cols failed");
                            }
                            _ => panic!("reduce axis out of bounds"),
                        }

                        out_device
                    }
                }
                TensorGraphNode::Parameter { data, .. } => {
                    let v = data.lock().unwrap();
                    trace!(idx = node_idx.index(), op = "Parameter");
                    let stream = self.device.default_stream();
                    let mut device_data = stream.alloc_zeros::<f32>(v.len()).unwrap();
                    stream.memcpy_htod(v.as_slice(), &mut device_data).unwrap();
                    device_data
                }
                TensorGraphNode::Broadcast => {
                    let stream = self.device.default_stream();
                    let in_idx = graph.inputs(*node_idx)[0];
                    let in_val = values.get(&in_idx).unwrap();
                    let in_shape = graph.shapes.get(&in_idx).unwrap();
                    let out_shape = graph.shapes.get(node_idx).unwrap();
                    let out_size: usize = out_shape.iter().product();
                    let mut out = stream.alloc_zeros::<f32>(out_size).unwrap();

                    let m = out_shape[0];
                    let n = out_shape[1];

                    // Scalar broadcast
                    if in_shape.is_empty() || *in_shape == vec![1] {
                        trace!(idx = node_idx.index(), op = "Broadcast", kind = "scalar");

                        // Copy scalar value from device to host to pass as parameter
                        let mut scalar_host = vec![0.0f32; 1];
                        stream.memcpy_dtoh(in_val, &mut scalar_host).unwrap();
                        let scalar_val = scalar_host[0];

                        let f = self.module.load_function("broadcast_scalar").unwrap();
                        let cfg = LaunchConfig::for_num_elems(out.len() as u32);
                        let out_len_u64 = out.len() as u64;
                        let mut launcher = stream.launch_builder(&f);
                        launcher.arg(&mut out);
                        launcher.arg(&scalar_val);
                        launcher.arg(&out_len_u64);
                        unsafe { launcher.launch(cfg) }.expect("CUDA broadcast scalar failed");
                    }
                    // Column broadcast: [m] or [m, 1] -> [m, n]
                    else if *in_shape == vec![m] || *in_shape == vec![m, 1] {
                        trace!(idx = node_idx.index(), op = "Broadcast", kind = "col");
                        let m_u64 = m as u64;
                        let n_u64 = n as u64;

                        let f = self.module.load_function("broadcast_col").unwrap();
                        let cfg = LaunchConfig::for_num_elems(m as u32);
                        let mut launcher = stream.launch_builder(&f);
                        launcher.arg(in_val);
                        launcher.arg(&m_u64);
                        launcher.arg(&mut out);
                        launcher.arg(&m_u64);
                        launcher.arg(&n_u64);
                        unsafe { launcher.launch(cfg) }.expect("CUDA broadcast col failed");
                    }
                    // Row broadcast: [n] or [1, n] -> [m, n]
                    else if *in_shape == vec![n] || *in_shape == vec![1, n] {
                        trace!(idx = node_idx.index(), op = "Broadcast", kind = "row");
                        let m_u64 = m as u64;
                        let n_u64 = n as u64;

                        let f = self.module.load_function("broadcast_row").unwrap();
                        let cfg = LaunchConfig::for_num_elems(n as u32);
                        let mut launcher = stream.launch_builder(&f);
                        launcher.arg(in_val);
                        launcher.arg(&n_u64);
                        launcher.arg(&mut out);
                        launcher.arg(&m_u64);
                        launcher.arg(&n_u64);
                        unsafe { launcher.launch(cfg) }.expect("CUDA broadcast row failed");
                    } else {
                        panic!(
                            "Unsupported broadcast from shape {:?} to {:?}",
                            in_shape, out_shape
                        );
                    }

                    out
                }
            };
            values.insert(*node_idx, result);
        }

        let last_node_idx = order.last().expect("Graph is empty");
        let out_device = values.remove(last_node_idx).expect("Output not found");
        let mut out_host = vec![0.0f32; out_device.len()];
        self.device
            .default_stream()
            .memcpy_dtoh(&out_device, &mut out_host)
            .unwrap();
        out_host
    }

    fn backward(
        &self,
        graph: &TensorGraph<f32>,
        inputs: HashMap<String, Vec<f32>>,
        loss_node: petgraph::graph::NodeIndex,
        seed_grad: Option<Vec<f32>>,
    ) -> BackwardResult<f32> {
        let order = graph.toposort();
        let exec_span = span!(Level::TRACE, "cuda_backward_pass", nodes = order.len());
        let _enter = exec_span.enter();

        // Forward pass to cache values needed for backward pass using GPU
        let mut values: HashMap<_, CudaSlice<f32>> = HashMap::with_capacity(order.len());
        for node_idx in order.iter() {
            let node = &graph[*node_idx];
            let result = match node {
                TensorGraphNode::Constant { data } => {
                    trace!(idx = node_idx.index(), op = "Constant");
                    let stream = self.device.default_stream();
                    let mut device_data = stream.alloc_zeros::<f32>(data.len()).unwrap();
                    stream
                        .memcpy_htod(data.as_slice(), &mut device_data)
                        .unwrap();
                    device_data
                }
                TensorGraphNode::Input { name } => {
                    let val = inputs
                        .get(&name.to_string())
                        .expect("Input not found")
                        .clone();
                    trace!(idx = node_idx.index(), op = "Input", name = *name);
                    let stream = self.device.default_stream();
                    let mut device_data = stream.alloc_zeros::<f32>(val.len()).unwrap();
                    stream.memcpy_htod(&val, &mut device_data).unwrap();
                    device_data
                }
                TensorGraphNode::Parameter { data, .. } => {
                    trace!(idx = node_idx.index(), op = "Parameter");
                    let host_data = data.lock().unwrap();
                    let stream = self.device.default_stream();
                    let mut device_data = stream.alloc_zeros::<f32>(host_data.len()).unwrap();
                    stream
                        .memcpy_htod(host_data.as_slice(), &mut device_data)
                        .unwrap();
                    device_data
                }
                TensorGraphNode::Unary { op } => {
                    let input_idx = graph.inputs(*node_idx)[0];
                    let x = values.get(&input_idx).unwrap();
                    let len = x.len();
                    let len_u64 = len as u64;
                    let stream = self.device.default_stream();
                    let mut out = stream.alloc_zeros::<f32>(len).unwrap();

                    let (function_name, op_name) = match op {
                        tensor::UnaryOp::Neg => ("neg", "Neg"),
                        tensor::UnaryOp::Exp => ("exp", "Exp"),
                        tensor::UnaryOp::Log => ("log", "Log"),
                        tensor::UnaryOp::Relu => ("relu", "Relu"),
                    };

                    trace!(idx = node_idx.index(), op = op_name);
                    let f = self.module.load_function(function_name).unwrap();
                    let cfg = LaunchConfig::for_num_elems(len as u32);
                    let mut launcher = stream.launch_builder(&f);
                    launcher.arg(x);
                    launcher.arg(&len_u64);
                    launcher.arg(&mut out);
                    launcher.arg(&len_u64);
                    unsafe { launcher.launch(cfg) }
                        .unwrap_or_else(|_| panic!("CUDA {} failed", function_name));
                    out
                }
                TensorGraphNode::Binary { op } => {
                    let ins = graph.inputs(*node_idx);
                    let a = values.get(&ins[0]).unwrap();
                    let b = values.get(&ins[1]).unwrap();
                    assert_eq!(a.len(), b.len(), "binary op input length mismatch");
                    let len = a.len();
                    let len_u64 = len as u64;

                    let stream = self.device.default_stream();
                    let mut out = stream.alloc_zeros::<f32>(len).unwrap();

                    let (function_name, op_name) = match op {
                        tensor::BinaryOp::Add => ("add", "Add"),
                        tensor::BinaryOp::Sub => ("sub", "Sub"),
                        tensor::BinaryOp::Mul => ("mul", "Mul"),
                        tensor::BinaryOp::Div => ("div", "Div"),
                    };

                    trace!(idx = node_idx.index(), op = op_name);
                    let f = self.module.load_function(function_name).unwrap();
                    let cfg = LaunchConfig::for_num_elems(len as u32);
                    let mut launcher = stream.launch_builder(&f);
                    launcher.arg(a);
                    launcher.arg(&len_u64);
                    launcher.arg(b);
                    launcher.arg(&len_u64);
                    launcher.arg(&mut out);
                    launcher.arg(&len_u64);
                    unsafe { launcher.launch(cfg) }
                        .unwrap_or_else(|_| panic!("CUDA {} failed", function_name));
                    out
                }
                TensorGraphNode::MatMul => {
                    // Use existing matmul implementation from forward pass
                    let ins = graph.inputs(*node_idx);
                    let lhs = values.get(&ins[0]).unwrap();
                    let rhs = values.get(&ins[1]).unwrap();
                    let lhs_shape = graph.shapes.get(&ins[0]).expect("shape missing for lhs");
                    let rhs_shape = graph.shapes.get(&ins[1]).expect("shape missing for rhs");
                    assert_eq!(lhs_shape.len(), 2, "MatMul expects 2D lhs");
                    assert_eq!(rhs_shape.len(), 2, "MatMul expects 2D rhs");
                    let m = lhs_shape[0];
                    let k = lhs_shape[1];
                    let n = rhs_shape[1];
                    assert_eq!(k, rhs_shape[0], "MatMul inner dim mismatch");

                    let out_len = m * n;
                    trace!(idx = node_idx.index(), op = "MatMul", m = m, n = n, k = k);
                    let lhs_len = lhs.len();
                    let rhs_len = rhs.len();

                    let stream = self.device.default_stream();
                    let mut out = stream.alloc_zeros::<f32>(out_len).unwrap();

                    let f = self.module.load_function("matmul").unwrap();
                    let cfg = LaunchConfig::for_num_elems(out_len as u32);
                    let lhs_len_u64 = lhs_len as u64;
                    let rhs_len_u64 = rhs_len as u64;
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
                TensorGraphNode::Broadcast => {
                    // TODO: Use existing broadcast implementation
                    // For now, fall back to CPU for this operation
                    let input_idx = graph.inputs(*node_idx)[0];
                    let x_gpu = values.get(&input_idx).unwrap();
                    let _input_shape = graph.shapes.get(&input_idx).unwrap();
                    let _output_shape = graph.shapes.get(node_idx).unwrap();

                    // Copy to host, use CPU, then copy back
                    let mut x_host = vec![0.0f32; x_gpu.len()];
                    self.device
                        .default_stream()
                        .memcpy_dtoh(x_gpu, &mut x_host)
                        .unwrap();
                    let result_host = crate::broadcast_forward(
                        graph,
                        &HashMap::from([(input_idx, x_host)]),
                        *node_idx,
                    );
                    let stream = self.device.default_stream();
                    let mut result_gpu = stream.alloc_zeros::<f32>(result_host.len()).unwrap();
                    stream.memcpy_htod(&result_host, &mut result_gpu).unwrap();
                    result_gpu
                }
                TensorGraphNode::Reduce { op, axis } => {
                    // TODO: Use existing reduce implementation
                    // For now, fall back to CPU for this operation
                    let input_idx = graph.inputs(*node_idx)[0];
                    let x_gpu = values.get(&input_idx).unwrap();

                    // Copy to host, use CPU, then copy back
                    let mut x_host = vec![0.0f32; x_gpu.len()];
                    self.device
                        .default_stream()
                        .memcpy_dtoh(x_gpu, &mut x_host)
                        .unwrap();
                    let result_host = crate::reduce_forward(
                        graph,
                        &HashMap::from([(input_idx, x_host)]),
                        *node_idx,
                        op,
                        *axis,
                    );
                    let stream = self.device.default_stream();
                    let mut result_gpu = stream.alloc_zeros::<f32>(result_host.len()).unwrap();
                    stream.memcpy_htod(&result_host, &mut result_gpu).unwrap();
                    result_gpu
                }
            };
            values.insert(*node_idx, result);
        }

        // Backward pass - gradients computation
        let mut grads: HashMap<petgraph::graph::NodeIndex, CudaSlice<f32>> = HashMap::new();
        let mut param_grads: HashMap<usize, Vec<f32>> = HashMap::new();

        // Initialize gradient for loss node
        let loss_shape = graph
            .shapes
            .get(&loss_node)
            .expect("Loss node shape missing");
        let seed = seed_grad.unwrap_or_else(|| vec![1.0f32; loss_shape.iter().product()]);
        let stream = self.device.default_stream();
        let mut seed_gpu = stream.alloc_zeros::<f32>(seed.len()).unwrap();
        stream.memcpy_htod(&seed, &mut seed_gpu).unwrap();
        grads.insert(loss_node, seed_gpu);

        // Backward pass in reverse topological order
        for &node_idx in order.iter().rev() {
            if let Some(dy) = grads.get(&node_idx) {
                match &graph[node_idx] {
                    TensorGraphNode::Constant { .. } => {}
                    TensorGraphNode::Input { .. } => {}
                    TensorGraphNode::Parameter { id, .. } => {
                        // Copy gradient to host for accumulation
                        let mut grad_host = vec![0.0f32; dy.len()];
                        self.device
                            .default_stream()
                            .memcpy_dtoh(dy, &mut grad_host)
                            .unwrap();
                        param_grads
                            .entry(*id)
                            .and_modify(|g| {
                                for (dst, src) in g.iter_mut().zip(grad_host.iter()) {
                                    *dst += *src;
                                }
                            })
                            .or_insert(grad_host);
                    }
                    TensorGraphNode::Unary { op } => {
                        let x_idx = graph.inputs(node_idx)[0];
                        let len = dy.len();
                        let len_u64 = len as u64;
                        let stream = self.device.default_stream();
                        let mut dx = stream.alloc_zeros::<f32>(len).unwrap();

                        match op {
                            tensor::UnaryOp::Neg => {
                                // d/dx(-x) = -1, so dx = -dy
                                trace!(idx = node_idx.index(), grad_op = "neg_grad");
                                let f = self.module.load_function("neg").unwrap();
                                let cfg = LaunchConfig::for_num_elems(len as u32);
                                let mut launcher = stream.launch_builder(&f);
                                launcher.arg(dy);
                                launcher.arg(&len_u64);
                                launcher.arg(&mut dx);
                                launcher.arg(&len_u64);
                                unsafe { launcher.launch(cfg) }.expect("CUDA neg grad failed");
                            }
                            tensor::UnaryOp::Exp => {
                                // d/dx(exp(x)) = exp(x), so dx = dy * exp(x) = dy * y
                                let y = values.get(&node_idx).unwrap();
                                trace!(idx = node_idx.index(), grad_op = "exp_grad");
                                let f = self.module.load_function("mul").unwrap();
                                let cfg = LaunchConfig::for_num_elems(len as u32);
                                let mut launcher = stream.launch_builder(&f);
                                launcher.arg(dy);
                                launcher.arg(&len_u64);
                                launcher.arg(y);
                                launcher.arg(&len_u64);
                                launcher.arg(&mut dx);
                                launcher.arg(&len_u64);
                                unsafe { launcher.launch(cfg) }.expect("CUDA exp grad failed");
                            }
                            tensor::UnaryOp::Log => {
                                // d/dx(log(x)) = 1/x, so dx = dy / x
                                let x = values.get(&x_idx).unwrap();
                                trace!(idx = node_idx.index(), grad_op = "log_grad");
                                let f = self.module.load_function("div").unwrap();
                                let cfg = LaunchConfig::for_num_elems(len as u32);
                                let mut launcher = stream.launch_builder(&f);
                                launcher.arg(dy);
                                launcher.arg(&len_u64);
                                launcher.arg(x);
                                launcher.arg(&len_u64);
                                launcher.arg(&mut dx);
                                launcher.arg(&len_u64);
                                unsafe { launcher.launch(cfg) }.expect("CUDA log grad failed");
                            }
                            tensor::UnaryOp::Relu => {
                                // TODO: Need relu_grad kernel that computes dy * (x > 0)
                                // For now, fall back to CPU for this specific operation
                                trace!(
                                    idx = node_idx.index(),
                                    grad_op = "relu_grad",
                                    fallback = "cpu"
                                );
                                let mut dy_host = vec![0.0f32; dy.len()];
                                let x = values.get(&x_idx).unwrap();
                                let mut x_host = vec![0.0f32; x.len()];
                                stream.memcpy_dtoh(dy, &mut dy_host).unwrap();
                                stream.memcpy_dtoh(x, &mut x_host).unwrap();
                                let dx_host: Vec<f32> = dy_host
                                    .iter()
                                    .zip(x_host.iter())
                                    .map(|(g, x)| if *x > 0.0 { *g } else { 0.0 })
                                    .collect();
                                stream.memcpy_htod(&dx_host, &mut dx).unwrap();
                            }
                        }
                        self.accumulate_grad_gpu(&mut grads, x_idx, dx);
                    }
                    TensorGraphNode::Binary { op } => {
                        let ins = graph.inputs(node_idx);
                        let a_idx = ins[0];
                        let b_idx = ins[1];
                        let a_shape = graph.shapes.get(&a_idx).unwrap();
                        let b_shape = graph.shapes.get(&b_idx).unwrap();
                        let output_shape = graph.shapes.get(&node_idx).unwrap();

                        match op {
                            tensor::BinaryOp::Add => {
                                // d/da(a + b) = 1, d/db(a + b) = 1
                                let da = self.reduce_like_cuda(dy, output_shape, a_shape);
                                let db = self.reduce_like_cuda(dy, output_shape, b_shape);
                                self.accumulate_grad_gpu(&mut grads, a_idx, da);
                                self.accumulate_grad_gpu(&mut grads, b_idx, db);
                            }
                            tensor::BinaryOp::Sub => {
                                // d/da(a - b) = 1, d/db(a - b) = -1
                                let da = self.reduce_like_cuda(dy, output_shape, a_shape);
                                let db_temp = self.reduce_like_cuda(dy, output_shape, b_shape);
                                let len = db_temp.len();
                                let len_u64 = len as u64;
                                let stream = self.device.default_stream();
                                let mut db = stream.alloc_zeros::<f32>(len).unwrap();
                                let f = self.module.load_function("neg").unwrap();
                                let cfg = LaunchConfig::for_num_elems(len as u32);
                                let mut launcher = stream.launch_builder(&f);
                                launcher.arg(&db_temp);
                                launcher.arg(&len_u64);
                                launcher.arg(&mut db);
                                launcher.arg(&len_u64);
                                unsafe { launcher.launch(cfg) }
                                    .expect("CUDA neg for sub grad failed");
                                self.accumulate_grad_gpu(&mut grads, a_idx, da);
                                self.accumulate_grad_gpu(&mut grads, b_idx, db);
                            }
                            tensor::BinaryOp::Mul => {
                                // d/da(a * b) = b, d/db(a * b) = a
                                let a_val = values.get(&a_idx).unwrap();
                                let b_val = values.get(&b_idx).unwrap();
                                let len = dy.len();
                                let len_u64 = len as u64;
                                let stream = self.device.default_stream();

                                let mut tmp_a = stream.alloc_zeros::<f32>(len).unwrap();
                                let mut tmp_b = stream.alloc_zeros::<f32>(len).unwrap();

                                let f = self.module.load_function("mul").unwrap();
                                let cfg = LaunchConfig::for_num_elems(len as u32);

                                // tmp_a = dy * b
                                let mut launcher = stream.launch_builder(&f);
                                launcher.arg(dy);
                                launcher.arg(&len_u64);
                                launcher.arg(b_val);
                                launcher.arg(&len_u64);
                                launcher.arg(&mut tmp_a);
                                launcher.arg(&len_u64);
                                unsafe { launcher.launch(cfg) }.expect("CUDA mul grad a failed");

                                // tmp_b = dy * a
                                let mut launcher = stream.launch_builder(&f);
                                launcher.arg(dy);
                                launcher.arg(&len_u64);
                                launcher.arg(a_val);
                                launcher.arg(&len_u64);
                                launcher.arg(&mut tmp_b);
                                launcher.arg(&len_u64);
                                unsafe { launcher.launch(cfg) }.expect("CUDA mul grad b failed");

                                let da = self.reduce_like_cuda(&tmp_a, output_shape, a_shape);
                                let db = self.reduce_like_cuda(&tmp_b, output_shape, b_shape);
                                self.accumulate_grad_gpu(&mut grads, a_idx, da);
                                self.accumulate_grad_gpu(&mut grads, b_idx, db);
                            }
                            tensor::BinaryOp::Div => {
                                // TODO: Need element-wise operations for db = -dy * a / (b * b)
                                // For now, fall back to CPU for this complex operation
                                trace!(
                                    idx = node_idx.index(),
                                    grad_op = "div_grad",
                                    fallback = "cpu"
                                );
                                let a_val = values.get(&a_idx).unwrap();
                                let b_val = values.get(&b_idx).unwrap();

                                let mut dy_host = vec![0.0f32; dy.len()];
                                let mut a_host = vec![0.0f32; a_val.len()];
                                let mut b_host = vec![0.0f32; b_val.len()];
                                let stream = self.device.default_stream();
                                stream.memcpy_dtoh(dy, &mut dy_host).unwrap();
                                stream.memcpy_dtoh(a_val, &mut a_host).unwrap();
                                stream.memcpy_dtoh(b_val, &mut b_host).unwrap();

                                let tmp_a_host: Vec<f32> = dy_host
                                    .iter()
                                    .zip(b_host.iter())
                                    .map(|(g, b)| g / b)
                                    .collect();
                                let tmp_b_host: Vec<f32> = dy_host
                                    .iter()
                                    .zip(a_host.iter())
                                    .map(|(g, a)| -g * a)
                                    .collect();
                                let b_sq_host: Vec<f32> = b_host.iter().map(|b| b * b).collect();
                                let tmp_b_final: Vec<f32> = tmp_b_host
                                    .iter()
                                    .zip(b_sq_host.iter())
                                    .map(|(t, bsq)| t / bsq)
                                    .collect();

                                // Use GPU reduce_like_cuda for the reductions
                                let mut tmp_a_gpu =
                                    stream.alloc_zeros::<f32>(tmp_a_host.len()).unwrap();
                                let mut tmp_b_gpu =
                                    stream.alloc_zeros::<f32>(tmp_b_final.len()).unwrap();
                                stream.memcpy_htod(&tmp_a_host, &mut tmp_a_gpu).unwrap();
                                stream.memcpy_htod(&tmp_b_final, &mut tmp_b_gpu).unwrap();

                                let da = self.reduce_like_cuda(&tmp_a_gpu, output_shape, a_shape);
                                let db = self.reduce_like_cuda(&tmp_b_gpu, output_shape, b_shape);

                                self.accumulate_grad_gpu(&mut grads, a_idx, da);
                                self.accumulate_grad_gpu(&mut grads, b_idx, db);
                            }
                        }
                    }
                    TensorGraphNode::MatMul => {
                        // TODO: Need specialized matmul gradient kernels
                        // For now, fall back to CPU for matmul gradients
                        trace!(
                            idx = node_idx.index(),
                            grad_op = "matmul_grad",
                            fallback = "cpu"
                        );
                        let ins = graph.inputs(node_idx);
                        let a_idx = ins[0];
                        let b_idx = ins[1];
                        let a_val = values.get(&a_idx).unwrap();
                        let b_val = values.get(&b_idx).unwrap();

                        let mut dy_host = vec![0.0f32; dy.len()];
                        let mut a_host = vec![0.0f32; a_val.len()];
                        let mut b_host = vec![0.0f32; b_val.len()];
                        let stream = self.device.default_stream();
                        stream.memcpy_dtoh(dy, &mut dy_host).unwrap();
                        stream.memcpy_dtoh(a_val, &mut a_host).unwrap();
                        stream.memcpy_dtoh(b_val, &mut b_host).unwrap();

                        let a_shape = graph.shapes.get(&a_idx).unwrap();
                        let b_shape = graph.shapes.get(&b_idx).unwrap();
                        let dy_shape = graph.shapes.get(&node_idx).unwrap();

                        let da_host = crate::matmul_grad_left(&dy_host, dy_shape, &b_host, b_shape);
                        let db_host =
                            crate::matmul_grad_right(&a_host, a_shape, &dy_host, dy_shape);

                        let mut da = stream.alloc_zeros::<f32>(da_host.len()).unwrap();
                        let mut db = stream.alloc_zeros::<f32>(db_host.len()).unwrap();
                        stream.memcpy_htod(&da_host, &mut da).unwrap();
                        stream.memcpy_htod(&db_host, &mut db).unwrap();

                        self.accumulate_grad_gpu(&mut grads, a_idx, da);
                        self.accumulate_grad_gpu(&mut grads, b_idx, db);
                    }
                    TensorGraphNode::Broadcast => {
                        let x_idx = graph.inputs(node_idx)[0];
                        let x_shape = graph.shapes.get(&x_idx).unwrap();
                        let y_shape = graph.shapes.get(&node_idx).unwrap();
                        let dx = self.reduce_like_cuda(dy, y_shape, x_shape);
                        self.accumulate_grad_gpu(&mut grads, x_idx, dx);
                    }
                    TensorGraphNode::Reduce { op, axis } => {
                        let x_idx = graph.inputs(node_idx)[0];
                        let x_shape = graph.shapes.get(&x_idx).unwrap();
                        match op {
                            tensor::ReduceOp::Sum => {
                                // Sum reduction gradient: expand dy to match original input shape
                                // For sum, gradient flows back unchanged to all positions
                                trace!(
                                    idx = node_idx.index(),
                                    grad_op = "reduce_sum_grad",
                                    axis = *axis
                                );
                                let dx = self.expand_to_gpu(dy, x_shape, *axis);
                                self.accumulate_grad_gpu(&mut grads, x_idx, dx);
                            }
                            tensor::ReduceOp::Mean => {
                                // TODO: Need expand_to and scalar division kernels
                                // For now, fall back to CPU
                                trace!(
                                    idx = node_idx.index(),
                                    grad_op = "reduce_mean_grad",
                                    fallback = "cpu"
                                );
                                let mut dy_host = vec![0.0f32; dy.len()];
                                self.device
                                    .default_stream()
                                    .memcpy_dtoh(dy, &mut dy_host)
                                    .unwrap();
                                let mut y_aligned_shape = x_shape.clone();
                                y_aligned_shape[*axis] = 1;
                                let mut dx_host =
                                    crate::expand_to(&dy_host, &y_aligned_shape, x_shape);
                                let axis_size = x_shape[*axis];
                                for v in dx_host.iter_mut() {
                                    *v /= axis_size as f32;
                                }
                                let stream = self.device.default_stream();
                                let mut dx = stream.alloc_zeros::<f32>(dx_host.len()).unwrap();
                                stream.memcpy_htod(&dx_host, &mut dx).unwrap();
                                self.accumulate_grad_gpu(&mut grads, x_idx, dx);
                            }
                            tensor::ReduceOp::Max => {
                                // TODO: Need argmax-based gradient computation kernel
                                // For now, zero gradient (correct but suboptimal)
                                trace!(
                                    idx = node_idx.index(),
                                    grad_op = "reduce_max_grad",
                                    note = "zero_grad"
                                );
                                let zeros = vec![0.0f32; x_shape.iter().product()];
                                let stream = self.device.default_stream();
                                let mut dx = stream.alloc_zeros::<f32>(zeros.len()).unwrap();
                                stream.memcpy_htod(&zeros, &mut dx).unwrap();
                                self.accumulate_grad_gpu(&mut grads, x_idx, dx);
                            }
                        }
                    }
                }
            }
        }

        // Get final loss value from GPU
        let loss_gpu = values.get(&loss_node).unwrap();
        let mut loss_value = vec![0.0f32; loss_gpu.len()];
        self.device
            .default_stream()
            .memcpy_dtoh(loss_gpu, &mut loss_value)
            .unwrap();

        debug!(
            params = param_grads.len(),
            nodes = grads.len(),
            "cuda_backward_pass_done"
        );

        BackwardResult {
            grads_by_node: HashMap::new(), // TODO: Convert GPU gradients to host if needed
            grads_by_param: param_grads,
            loss_value,
        }
    }
}

impl CudaExecutor {
    /// Accumulate gradient on GPU
    fn accumulate_grad_gpu(
        &self,
        grads: &mut HashMap<petgraph::graph::NodeIndex, CudaSlice<f32>>,
        node_idx: petgraph::graph::NodeIndex,
        new_grad: CudaSlice<f32>,
    ) {
        if let Some(existing_grad) = grads.get(&node_idx) {
            // Add the new gradient to the existing one
            let len = existing_grad.len();
            let len_u64 = len as u64;
            let stream = self.device.default_stream();
            let mut accumulated = stream.alloc_zeros::<f32>(len).unwrap();

            let f = self.module.load_function("add").unwrap();
            let cfg = LaunchConfig::for_num_elems(len as u32);
            let mut launcher = stream.launch_builder(&f);
            launcher.arg(existing_grad);
            launcher.arg(&len_u64);
            launcher.arg(&new_grad);
            launcher.arg(&len_u64);
            launcher.arg(&mut accumulated);
            launcher.arg(&len_u64);
            unsafe { launcher.launch(cfg) }.expect("CUDA grad accumulation failed");

            grads.insert(node_idx, accumulated);
        } else {
            grads.insert(node_idx, new_grad);
        }
    }

    /// Reduce a gradient array to match a smaller shape (for broadcast backward)
    /// TODO: This needs a proper CUDA kernel implementation
    fn reduce_like_cuda(
        &self,
        grad: &CudaSlice<f32>,
        grad_shape: &[usize],
        target_shape: &[usize],
    ) -> CudaSlice<f32> {
        // Handle common cases with existing GPU kernels
        if grad_shape == target_shape {
            // No reduction needed, return copy
            return grad.clone();
        }

        let stream = self.device.default_stream();

        // Optimize common broadcasting patterns in neural networks
        if grad_shape.len() == 2 && target_shape.len() == 2 {
            let (grad_rows, grad_cols) = (grad_shape[0], grad_shape[1]);
            let (target_rows, target_cols) = (target_shape[0], target_shape[1]);

            // Case 1: (batch_size, features) -> (1, features) - bias gradient
            if grad_rows > 1 && target_rows == 1 && grad_cols == target_cols {
                // Reduce along axis 0 (sum rows)
                let mut result = stream.alloc_zeros::<f32>(target_cols).unwrap();

                let f = self.module.load_function("reduce_sum_cols").unwrap();
                let cfg = LaunchConfig::for_num_elems(target_cols as u32);
                let grad_rows_u64 = grad_rows as u64;
                let grad_cols_u64 = grad_cols as u64;
                let mut launcher = stream.launch_builder(&f);
                launcher.arg(grad);
                launcher.arg(&mut result);
                launcher.arg(&grad_rows_u64);
                launcher.arg(&grad_cols_u64);
                unsafe { launcher.launch(cfg) }.expect("CUDA reduce_sum_cols failed");
                return result;
            }

            // Case 2: (batch_size, features) -> (batch_size, 1) - feature-wise sum
            if grad_rows == target_rows && grad_cols > 1 && target_cols == 1 {
                // Reduce along axis 1 (sum columns)
                let mut result = stream.alloc_zeros::<f32>(target_rows).unwrap();

                let f = self.module.load_function("reduce_sum_rows").unwrap();
                let cfg = LaunchConfig::for_num_elems(target_rows as u32);
                let grad_rows_u64 = grad_rows as u64;
                let grad_cols_u64 = grad_cols as u64;
                let mut launcher = stream.launch_builder(&f);
                launcher.arg(grad);
                launcher.arg(&mut result);
                launcher.arg(&grad_rows_u64);
                launcher.arg(&grad_cols_u64);
                unsafe { launcher.launch(cfg) }.expect("CUDA reduce_sum_rows failed");
                return result;
            }
        }

        // Case 3: Scalar reduction (any shape -> scalar)
        if target_shape.iter().all(|&x| x == 1) || target_shape.is_empty() {
            let mut result = stream.alloc_zeros::<f32>(1).unwrap();

            let f = self.module.load_function("reduce_sum_all").unwrap();
            let cfg = LaunchConfig::for_num_elems(grad.len() as u32);
            let len_u64 = grad.len() as u64;
            let mut launcher = stream.launch_builder(&f);
            launcher.arg(grad);
            launcher.arg(&mut result);
            launcher.arg(&len_u64);
            unsafe { launcher.launch(cfg) }.expect("CUDA reduce_sum_all failed");
            return result;
        }

        // For complex cases, fall back to CPU
        trace!(grad_shape = ?grad_shape, target_shape = ?target_shape, "reduce_like_cuda CPU fallback");
        let mut grad_host = vec![0.0f32; grad.len()];
        self.device
            .default_stream()
            .memcpy_dtoh(grad, &mut grad_host)
            .unwrap();
        let reduced_host = crate::reduce_like(&grad_host, grad_shape, target_shape);
        let stream = self.device.default_stream();
        let mut reduced_gpu = stream.alloc_zeros::<f32>(reduced_host.len()).unwrap();
        stream.memcpy_htod(&reduced_host, &mut reduced_gpu).unwrap();
        reduced_gpu
    }

    /// Expand a gradient to match a larger shape (for reduce backward)
    /// Implements the backward pass of reduce operations by broadcasting the gradient
    fn expand_to_gpu(
        &self,
        grad: &CudaSlice<f32>,
        target_shape: &[usize],
        reduced_axis: usize,
    ) -> CudaSlice<f32> {
        let target_size: usize = target_shape.iter().product();
        let stream = self.device.default_stream();
        let mut result = stream.alloc_zeros::<f32>(target_size).unwrap();

        // Handle common cases with existing broadcast kernels
        if target_shape.len() == 2 {
            let (rows, cols) = (target_shape[0], target_shape[1]);

            let rows_u64 = rows as u64;
            let cols_u64 = cols as u64;

            if reduced_axis == 0 {
                // Reduced along rows, broadcast to (rows, cols) from (1, cols)
                // Use broadcast_row_vector kernel
                let f = self.module.load_function("broadcast_row").unwrap();
                let cfg = LaunchConfig::for_num_elems(cols as u32);
                let grad_len_u64 = grad.len() as u64;
                let mut launcher = stream.launch_builder(&f);
                launcher.arg(grad);
                launcher.arg(&grad_len_u64);
                launcher.arg(&mut result);
                launcher.arg(&rows_u64);
                launcher.arg(&cols_u64);
                unsafe { launcher.launch(cfg) }.expect("CUDA broadcast_row failed");
            } else {
                // Reduced along cols, broadcast to (rows, cols) from (rows, 1)
                // Use broadcast_col_vector kernel
                let f = self.module.load_function("broadcast_col").unwrap();
                let cfg = LaunchConfig::for_num_elems(rows as u32);
                let grad_len_u64 = grad.len() as u64;
                let mut launcher = stream.launch_builder(&f);
                launcher.arg(grad);
                launcher.arg(&grad_len_u64);
                launcher.arg(&mut result);
                launcher.arg(&rows_u64);
                launcher.arg(&cols_u64);
                unsafe { launcher.launch(cfg) }.expect("CUDA broadcast_col failed");
            }
        } else {
            // For other cases, fall back to CPU implementation
            // TODO: Implement general expand_to kernel for arbitrary dimensions
            let mut grad_host = vec![0.0f32; grad.len()];
            stream.memcpy_dtoh(grad, &mut grad_host).unwrap();

            let mut grad_shape = target_shape.to_vec();
            grad_shape[reduced_axis] = 1;
            let expanded_host = crate::expand_to(&grad_host, &grad_shape, target_shape);

            stream.memcpy_htod(&expanded_host, &mut result).unwrap();
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use std::sync::{Arc, Mutex};
    use tensor::Constant;
    use tensor::TensorExpr;

    use super::*;
    use crate::SimpleExecutor;

    const EPSILON: f32 = 1e-5;

    macro_rules! assert_approx_eq {
        ($a:expr, $b:expr) => {
            if $a
                .iter()
                .zip($b.iter())
                .any(|(a, b)| (a - b).abs() > EPSILON)
            {
                panic!(
                    "assertion failed: `(left ~= right)` (left: `{:?}`, right: `{:?}`)",
                    $a, $b
                );
            }
        };
    }

    type UnaryApply = fn(Constant<f32>) -> TensorExpr<f32>;
    fn neg_node(a: Constant<f32>) -> TensorExpr<f32> {
        -TensorExpr::from(a)
    }
    fn exp_node(a: Constant<f32>) -> TensorExpr<f32> {
        TensorExpr::from(a).exp()
    }
    fn log_node(a: Constant<f32>) -> TensorExpr<f32> {
        TensorExpr::from(a).log()
    }
    fn relu_node(a: Constant<f32>) -> TensorExpr<f32> {
        TensorExpr::from(a).relu()
    }

    #[rstest]
    #[case::neg(
          vec![1.0f32, -2.0, 3.0, -4.0],
          vec![-1.0, 2.0, -3.0, 4.0],
          neg_node as UnaryApply
      )]
    #[case::exp(
          vec![0.0f32, 1.0, 2.0, 3.0],
          vec![1.0, std::f32::consts::E, std::f32::consts::E.powi(2), std::f32::consts::E.powi(3)],
          exp_node as UnaryApply
      )]
    #[case::log(
          vec![1.0f32, std::f32::consts::E, std::f32::consts::E.powi(2), std::f32::consts::E.powi(3)],
          vec![0.0, 1.0, 2.0, 3.0],
          log_node as UnaryApply
      )]
    #[case::relu(
          vec![-1.0f32, 2.0, -3.0, 4.0],
          vec![0.0, 2.0, 0.0, 4.0],
          relu_node as UnaryApply
      )]
    fn elementwise_unary_f32(
        #[case] input: Vec<f32>,
        #[case] expected: Vec<f32>,
        #[case] apply: UnaryApply,
    ) {
        let a = Constant::new(input, vec![2, 2]);
        let node = apply(a);

        let mut graph = TensorGraph::new();
        node.lower_to_graph(&mut graph);

        let exec = CudaExecutor::new();
        let result = exec.execute(&graph, Default::default());

        assert_approx_eq!(result, expected);
    }

    type BinaryApply = fn(Constant<f32>, Constant<f32>) -> TensorExpr<f32>;
    fn add_node(a: Constant<f32>, b: Constant<f32>) -> TensorExpr<f32> {
        TensorExpr::from(a) + TensorExpr::from(b)
    }
    fn sub_node(a: Constant<f32>, b: Constant<f32>) -> TensorExpr<f32> {
        TensorExpr::from(a) - TensorExpr::from(b)
    }
    fn mul_node(a: Constant<f32>, b: Constant<f32>) -> TensorExpr<f32> {
        TensorExpr::from(a) * TensorExpr::from(b)
    }
    fn div_node(a: Constant<f32>, b: Constant<f32>) -> TensorExpr<f32> {
        TensorExpr::from(a) / TensorExpr::from(b)
    }

    #[rstest]
    #[case::add(
          vec![1.0f32, 2.0, 3.0, 4.0],
          vec![5.0f32, 6.0, 7.0, 8.0],
          vec![6.0, 8.0, 10.0, 12.0],
          add_node as BinaryApply
      )]
    #[case::sub(
          vec![5.0f32, 6.0, 7.0, 8.0],
          vec![1.0f32, 2.0, 3.0, 4.0],
          vec![4.0, 4.0, 4.0, 4.0],
          sub_node as BinaryApply
      )]
    #[case::mul(
          vec![1.0f32, 2.0, 3.0, 4.0],
          vec![5.0f32, 6.0, 7.0, 8.0],
          vec![5.0, 12.0, 21.0, 32.0],
          mul_node as BinaryApply
      )]
    #[case::div(
          vec![5.0f32, 12.0, 21.0, 32.0],
          vec![1.0f32, 2.0, 3.0, 4.0],
          vec![5.0, 6.0, 7.0, 8.0],
          div_node as BinaryApply
      )]
    fn elementwise_binary_f32(
        #[case] a_in: Vec<f32>,
        #[case] b_in: Vec<f32>,
        #[case] expected: Vec<f32>,
        #[case] apply: BinaryApply,
    ) {
        let a = Constant::new(a_in, vec![2, 2]);
        let b = Constant::new(b_in, vec![2, 2]);
        let node = apply(a, b);

        let mut graph = TensorGraph::new();
        node.lower_to_graph(&mut graph);

        let exec = CudaExecutor::new();
        let result = exec.execute(&graph, Default::default());

        assert_approx_eq!(result, expected);
    }

    #[test]
    fn matmul_f32() {
        let a = Constant::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Constant::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let node = TensorExpr::from(a).matmul(b);

        let mut graph = TensorGraph::new();
        node.lower_to_graph(&mut graph);

        let cpu = SimpleExecutor {};
        let expected = cpu.execute(&graph, Default::default());

        let cuda = CudaExecutor::new();
        let result = cuda.execute(&graph, Default::default());

        assert_approx_eq!(result, expected);
    }

    #[test]
    fn reduce_sum_rows_axis1_2x3() {
        let a = Constant::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let node = TensorExpr::from(a).reduce_sum(1);

        let mut graph = TensorGraph::new();
        node.lower_to_graph(&mut graph);

        let cpu = SimpleExecutor {};
        let expected = cpu.execute(&graph, Default::default());

        let cuda = CudaExecutor::new();
        let result = cuda.execute(&graph, Default::default());

        assert_approx_eq!(result, expected);
    }

    #[test]
    fn reduce_sum_cols_axis0_2x3() {
        let a = Constant::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let node = TensorExpr::from(a).reduce_sum(0);

        let mut graph = TensorGraph::new();
        node.lower_to_graph(&mut graph);

        let cpu = SimpleExecutor {};
        let expected = cpu.execute(&graph, Default::default());

        let cuda = CudaExecutor::new();
        let result = cuda.execute(&graph, Default::default());

        assert_approx_eq!(result, expected);
    }

    #[test]
    fn reduce_sum_all_scalar() {
        let a = Constant::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let node = TensorExpr::from(a).reduce_sum(1).reduce_sum(0);

        let mut graph = TensorGraph::new();
        node.lower_to_graph(&mut graph);

        let cpu = SimpleExecutor {};
        let expected = cpu.execute(&graph, Default::default());

        let cuda = CudaExecutor::new();
        let result = cuda.execute(&graph, Default::default());

        assert_approx_eq!(result, expected);
    }

    #[test_log::test]
    fn reduce_max_and_mean_match_cpu() {
        let a = Constant::new(vec![1.0, -2.0, 3.5, 0.5, 10.0, -1.0], vec![2, 3]);

        // max axis 1
        let node_max_r = TensorExpr::from(a.clone()).reduce_max(1);
        let mut g1 = TensorGraph::new();
        node_max_r.lower_to_graph(&mut g1);
        let cpu = SimpleExecutor {};
        let exp1 = cpu.execute(&g1, Default::default());
        let cuda = CudaExecutor::new();
        let res1 = cuda.execute(&g1, Default::default());
        assert_approx_eq!(res1, exp1);

        // max axis 0
        let node_max_c = TensorExpr::from(a.clone()).reduce_max(0);
        let mut g2 = TensorGraph::new();
        node_max_c.lower_to_graph(&mut g2);
        let exp2 = cpu.execute(&g2, Default::default());
        let res2 = cuda.execute(&g2, Default::default());
        assert_approx_eq!(res2, exp2);

        // mean axis 1
        let a3 = Constant::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let node_mean_r = TensorExpr::from(a3.clone()).reduce_mean(1);
        let mut g3 = TensorGraph::new();
        node_mean_r.lower_to_graph(&mut g3);
        let exp3 = cpu.execute(&g3, Default::default());
        let res3 = cuda.execute(&g3, Default::default());
        assert_approx_eq!(res3, exp3);

        // mean all via mean_all()
        let a4 = Constant::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let node_mean_all = TensorExpr::from(a4).mean_all();
        let mut g4 = TensorGraph::new();
        node_mean_all.lower_to_graph(&mut g4);
        let exp4 = cpu.execute(&g4, Default::default());
        let res4 = cuda.execute(&g4, Default::default());
        assert_approx_eq!(res4, exp4);
    }

    #[test]
    fn parameter_passthrough_matches_cpu() {
        let p = TensorExpr::parameter(vec![1.0f32, -2.0, 3.5, 0.5], vec![2, 2]);
        let mut g = TensorGraph::new();
        p.lower_to_graph(&mut g);

        let cpu = SimpleExecutor {};
        let exp = cpu.execute(&g, Default::default());

        let cuda = CudaExecutor::new();
        let res = cuda.execute(&g, Default::default());

        assert_approx_eq!(res, exp);
    }

    #[test_log::test]
    fn broadcast_col_vector_to_matrix_matches_cpu() {
        // [2,1] -> [2,3]
        let a = Constant::new(vec![1.0f32, 2.0], vec![2, 1]);
        let node = TensorExpr::from(a).broadcast(vec![2, 3]);
        let mut g = TensorGraph::new();
        node.lower_to_graph(&mut g);

        let cpu = SimpleExecutor {};
        let exp = cpu.execute(&g, Default::default());

        let cuda = CudaExecutor::new();
        let res = cuda.execute(&g, Default::default());

        assert_approx_eq!(res, exp);
    }

    #[test]
    fn broadcast_row_vector_to_matrix_matches_cpu() {
        // [1,3] -> [2,3]
        let a = Constant::new(vec![1.0f32, 2.0, 3.0], vec![1, 3]);
        let node = TensorExpr::from(a).broadcast(vec![2, 3]);
        let mut g = TensorGraph::new();
        node.lower_to_graph(&mut g);

        let cpu = SimpleExecutor {};
        let exp = cpu.execute(&g, Default::default());

        let cuda = CudaExecutor::new();
        let res = cuda.execute(&g, Default::default());

        assert_approx_eq!(res, exp);
    }

    #[test_log::test]
    fn broadcast_scalar_to_matrix_matches_cpu() {
        // [] -> [2,3]
        let a = Constant::new(vec![3.0f32], vec![]);
        let node = TensorExpr::from(a).broadcast(vec![2, 3]);
        let mut g = TensorGraph::new();
        node.lower_to_graph(&mut g);

        let cpu = SimpleExecutor {};
        let exp = cpu.execute(&g, Default::default());

        let cuda = CudaExecutor::new();
        let res = cuda.execute(&g, Default::default());

        assert_approx_eq!(res, exp);
    }

    #[test]
    fn test_cuda_backward_simple_mul() {
        // Test CUDA backward pass with simple x^2 computation
        // This verifies GPU gradients match CPU gradients

        let mut graph = tensor::graph::TensorGraph::new();

        // Add parameter node with data = [2.0]
        let param_data = Arc::new(Mutex::new(vec![2.0f32]));
        let param_grad = Arc::new(Mutex::new(vec![0.0f32]));
        let param_node = graph
            .graph
            .add_node(tensor::graph::TensorGraphNode::Parameter {
                id: 0,
                data: param_data.clone(),
                grad: param_grad.clone(),
            });
        graph.shapes.insert(param_node, vec![1]);

        // Add multiplication node: x * x
        let mul_node = graph
            .graph
            .add_node(tensor::graph::TensorGraphNode::Binary {
                op: tensor::BinaryOp::Mul,
            });
        graph.shapes.insert(mul_node, vec![1]);

        // Connect parameter to both inputs of multiplication
        graph.graph.add_edge(param_node, mul_node, 0);
        graph.graph.add_edge(param_node, mul_node, 1);

        let inputs = std::collections::HashMap::new();

        // Test with CPU first
        let cpu_executor = SimpleExecutor {};
        let cpu_result =
            cpu_executor.backward(&graph, inputs.clone(), mul_node, Some(vec![1.0f32]));

        // Test with CUDA
        let cuda_executor = CudaExecutor::new();
        let cuda_result = cuda_executor.backward(&graph, inputs, mul_node, Some(vec![1.0f32]));

        // Check that gradients match
        assert_eq!(
            cpu_result.grads_by_param.len(),
            cuda_result.grads_by_param.len()
        );
        assert_eq!(cpu_result.grads_by_param.len(), 1);

        let cpu_grad = cpu_result.grads_by_param.get(&0).unwrap();
        let cuda_grad = cuda_result.grads_by_param.get(&0).unwrap();

        // Gradient of x^2 at x=2 should be 2*x = 4
        assert_approx_eq!(*cpu_grad, vec![4.0f32]);
        assert_approx_eq!(*cuda_grad, vec![4.0f32]);

        // Check loss values match
        assert_approx_eq!(cpu_result.loss_value, cuda_result.loss_value);
    }

    #[test]
    fn test_cuda_backward_binary_add() {
        // Test CUDA backward pass with binary addition
        // This should use GPU kernels for both forward and backward

        let a = Constant::new(vec![1.0f32, 2.0], vec![2]);
        let b = Constant::new(vec![3.0f32, 4.0], vec![2]);
        let node = TensorExpr::from(a) + TensorExpr::from(b); // Should produce [4.0, 6.0]

        let mut graph = tensor::graph::TensorGraph::new();
        let loss_node = node.lower_to_graph(&mut graph);

        let inputs = std::collections::HashMap::new();

        // Test with CPU first
        let cpu_executor = SimpleExecutor {};
        let cpu_result =
            cpu_executor.backward(&graph, inputs.clone(), loss_node, Some(vec![1.0f32, 1.0]));

        // Test with CUDA
        let cuda_executor = CudaExecutor::new();
        let cuda_result =
            cuda_executor.backward(&graph, inputs, loss_node, Some(vec![1.0f32, 1.0]));

        // Check that results match
        assert_approx_eq!(cpu_result.loss_value, cuda_result.loss_value);
        assert_eq!(
            cpu_result.grads_by_param.len(),
            cuda_result.grads_by_param.len()
        );
    }

    #[test]
    fn test_performance_comparison() {
        // Simple performance comparison between CPU and GPU backward pass
        // This test doesn't assert anything but provides timing information
        use std::time::Instant;

        // Create a moderately complex computation: (a * b + c) * (a + b)
        let a = Constant::new(vec![1.0f32; 1000], vec![1000]);
        let b = Constant::new(vec![2.0f32; 1000], vec![1000]);
        let c = Constant::new(vec![0.5f32; 1000], vec![1000]);

        let term1 = TensorExpr::from(a.clone()) * TensorExpr::from(b.clone()) + TensorExpr::from(c);
        let term2 = TensorExpr::from(a) + TensorExpr::from(b);
        let result = term1 * term2;

        let mut graph = tensor::graph::TensorGraph::new();
        let loss_node = result.lower_to_graph(&mut graph);

        let inputs = std::collections::HashMap::new();
        let seed = vec![1.0f32; 1000];

        // Time CPU backward pass
        let cpu_executor = SimpleExecutor {};
        let cpu_start = Instant::now();
        let cpu_result =
            cpu_executor.backward(&graph, inputs.clone(), loss_node, Some(seed.clone()));
        let cpu_duration = cpu_start.elapsed();

        // Time CUDA backward pass (including GPU memory transfers)
        let cuda_executor = CudaExecutor::new();
        let cuda_start = Instant::now();
        let cuda_result = cuda_executor.backward(&graph, inputs, loss_node, Some(seed));
        let cuda_duration = cuda_start.elapsed();

        // Print timing results (visible during test with --nocapture)
        println!("CPU backward pass: {:?}", cpu_duration);
        println!("CUDA backward pass: {:?}", cuda_duration);
        println!(
            "Speedup: {:.2}x",
            cpu_duration.as_secs_f64() / cuda_duration.as_secs_f64()
        );

        // Verify results match
        assert_approx_eq!(cpu_result.loss_value, cuda_result.loss_value);
        assert_eq!(
            cpu_result.grads_by_param.len(),
            cuda_result.grads_by_param.len()
        );
    }
}
