use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use tensor::graph::{TensorGraph, TensorGraphNode};

use crate::Executor;
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
                TensorGraphNode::Neg => {
                    let inputs = graph.inputs(*node_idx);
                    let input = values.get(&inputs[0]).unwrap();
                    let len_u64 = input.len() as u64;

                    trace!(idx = node_idx.index(), op = "Neg");
                    let stream = self.device.default_stream();
                    let mut out = stream.alloc_zeros::<f32>(input.len()).unwrap();

                    let f = self.module.load_function("neg").unwrap();
                    let cfg = LaunchConfig::for_num_elems(input.len() as u32);
                    let mut launcher = stream.launch_builder(&f);
                    launcher.arg(input);
                    launcher.arg(&len_u64);
                    launcher.arg(&mut out);
                    launcher.arg(&len_u64);
                    unsafe { launcher.launch(cfg) }.expect("CUDA neg failed");
                    out
                }
                TensorGraphNode::Exp => {
                    let inputs = graph.inputs(*node_idx);
                    let input = values.get(&inputs[0]).unwrap();
                    let len = input.len();
                    let len_u64 = len as u64;

                    trace!(idx = node_idx.index(), op = "Exp");
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
                TensorGraphNode::Log => {
                    let inputs = graph.inputs(*node_idx);
                    let input = values.get(&inputs[0]).unwrap();
                    let len = input.len();
                    let len_u64 = len as u64;

                    trace!(idx = node_idx.index(), op = "Log");
                    let stream = self.device.default_stream();
                    let mut out = stream.alloc_zeros::<f32>(len).unwrap();

                    let f = self.module.load_function("log").unwrap();
                    let cfg = LaunchConfig::for_num_elems(len as u32);
                    let mut launcher = stream.launch_builder(&f);
                    launcher.arg(input);
                    launcher.arg(&len_u64);
                    launcher.arg(&mut out);
                    launcher.arg(&len_u64);
                    unsafe { launcher.launch(cfg) }.expect("CUDA exp failed");
                    out
                }
                TensorGraphNode::Relu => {
                    let inputs = graph.inputs(*node_idx);
                    let input = values.get(&inputs[0]).unwrap();
                    let len = input.len();
                    let len_u64 = len as u64;

                    trace!(idx = node_idx.index(), op = "Relu");
                    let stream = self.device.default_stream();
                    let mut out = stream.alloc_zeros::<f32>(len).unwrap();

                    let f = self.module.load_function("relu").unwrap();
                    let cfg = LaunchConfig::for_num_elems(len as u32);
                    let mut launcher = stream.launch_builder(&f);
                    launcher.arg(input);
                    launcher.arg(&len_u64);
                    launcher.arg(&mut out);
                    launcher.arg(&len_u64);
                    unsafe { launcher.launch(cfg) }.expect("CUDA exp failed");
                    out
                }
                TensorGraphNode::Add => {
                    let ins = graph.inputs(*node_idx);
                    let lhs = values.get(&ins[0]).unwrap();
                    let rhs = values.get(&ins[1]).unwrap();
                    assert_eq!(lhs.len(), rhs.len(), "binary op input length mismatch");
                    let len = lhs.len();
                    let len_u64 = len as u64;

                    trace!(idx = node_idx.index(), op = "Add");
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
                TensorGraphNode::Sub => {
                    let ins = graph.inputs(*node_idx);
                    let lhs = values.get(&ins[0]).unwrap();
                    let rhs = values.get(&ins[1]).unwrap();
                    assert_eq!(lhs.len(), rhs.len(), "binary op input length mismatch");
                    let len = lhs.len();
                    let len_u64 = len as u64;

                    trace!(idx = node_idx.index(), op = "Sub");
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
                TensorGraphNode::Mul => {
                    let ins = graph.inputs(*node_idx);
                    let lhs = values.get(&ins[0]).unwrap();
                    let rhs = values.get(&ins[1]).unwrap();
                    assert_eq!(lhs.len(), rhs.len(), "binary op input length mismatch");
                    let len = lhs.len();
                    let len_u64 = len as u64;

                    trace!(idx = node_idx.index(), op = "Mul");
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
                TensorGraphNode::Div => {
                    let ins = graph.inputs(*node_idx);
                    let lhs = values.get(&ins[0]).unwrap();
                    let rhs = values.get(&ins[1]).unwrap();
                    assert_eq!(lhs.len(), rhs.len(), "binary op input length mismatch");
                    let len = lhs.len();
                    let len_u64 = len as u64;

                    trace!(idx = node_idx.index(), op = "Div");
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
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
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

    #[ignore = "need to fix kernel launch (bad params)"]
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
}
