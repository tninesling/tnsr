use std::collections::HashMap;

use crate::Executor;
use cust::memory::CopyDestination;
use cust::memory::DeviceBuffer;
use cust::stream::Stream;
use cust::stream::StreamFlags;
use cust::util::SliceExt;
use kernels_cuda::PTX;
use kernels_cuda::elementwise::{add, div, exp, log, mul, neg, relu, sub};
use tensor::graph::TensorGraph;
use tensor::graph::TensorGraphNode;

pub struct CudaExecutor {
    _context: cust::context::Context,
    module: cust::module::Module,
}

impl CudaExecutor {
    pub fn new() -> Self {
        let _context = cust::quick_init().expect("Failed to initialize CUDA");
        let module = cust::module::Module::from_ptx(PTX, &[]).expect("Failed to load PTX module");
        CudaExecutor { _context, module }
    }
}

impl Executor<f32> for CudaExecutor {
    fn execute(&self, graph: &TensorGraph<f32>, inputs: HashMap<String, Vec<f32>>) -> Vec<f32> {
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).expect("Failed to create stream");
        let order = graph.toposort();
        let mut values: HashMap<_, Vec<f32>> = HashMap::with_capacity(order.len());
        for node_idx in order.iter() {
            let node = &graph[*node_idx];
            let result = match node {
                TensorGraphNode::Constant { data } => data.as_ref().clone(),
                TensorGraphNode::Input { name } => inputs
                    .get(&name.to_string())
                    .expect("Input not found")
                    .clone(),
                TensorGraphNode::Neg => {
                    let inputs = graph.inputs(*node_idx);
                    let input = values
                        .get(&inputs[0])
                        .unwrap()
                        .as_slice()
                        .as_dbuf()
                        .expect("Copy to DeviceBuffer failed");
                    let mut output = unsafe {
                        DeviceBuffer::uninitialized(input.len()).expect("DeviceBuffer init")
                    };
                    neg(&self.module, &stream, &input, &mut output, input.len())
                        .expect("CUDA neg failed");

                    let mut value = vec![0.0f32; input.len()];
                    output
                        .copy_to(&mut value)
                        .expect("Copy from DeviceBuffer failed");
                    value
                }
                TensorGraphNode::Exp => {
                    let inputs = graph.inputs(*node_idx);
                    let input = values
                        .get(&inputs[0])
                        .unwrap()
                        .as_slice()
                        .as_dbuf()
                        .expect("Copy to DeviceBuffer failed");
                    let mut output = unsafe {
                        DeviceBuffer::uninitialized(input.len()).expect("DeviceBuffer init")
                    };
                    exp(&self.module, &stream, &input, &mut output, input.len())
                        .expect("CUDA exp failed");

                    let mut value = vec![0.0f32; input.len()];
                    output
                        .copy_to(&mut value)
                        .expect("Copy from DeviceBuffer failed");
                    value
                }
                TensorGraphNode::Log => {
                    let inputs = graph.inputs(*node_idx);
                    let input = values
                        .get(&inputs[0])
                        .unwrap()
                        .as_slice()
                        .as_dbuf()
                        .expect("Copy to DeviceBuffer failed");
                    let mut output = unsafe {
                        DeviceBuffer::uninitialized(input.len()).expect("DeviceBuffer init")
                    };
                    log(&self.module, &stream, &input, &mut output, input.len())
                        .expect("CUDA log failed");

                    let mut value = vec![0.0f32; input.len()];
                    output
                        .copy_to(&mut value)
                        .expect("Copy from DeviceBuffer failed");
                    value
                }
                TensorGraphNode::Relu => {
                    let inputs = graph.inputs(*node_idx);
                    let input = values
                        .get(&inputs[0])
                        .unwrap()
                        .as_slice()
                        .as_dbuf()
                        .expect("Copy to DeviceBuffer failed");
                    let mut output = unsafe {
                        DeviceBuffer::uninitialized(input.len()).expect("DeviceBuffer init")
                    };
                    relu(&self.module, &stream, &input, &mut output, input.len())
                        .expect("CUDA relu failed");

                    let mut value = vec![0.0f32; input.len()];
                    output
                        .copy_to(&mut value)
                        .expect("Copy from DeviceBuffer failed");
                    value
                }
                TensorGraphNode::Add => {
                    let ins = graph.inputs(*node_idx);
                    let a = values
                        .get(&ins[0])
                        .unwrap()
                        .as_slice()
                        .as_dbuf()
                        .expect("Copy to DeviceBuffer failed");
                    let b = values
                        .get(&ins[1])
                        .unwrap()
                        .as_slice()
                        .as_dbuf()
                        .expect("Copy to DeviceBuffer failed");
                    assert_eq!(a.len(), b.len(), "binary op input length mismatch");
                    let len = a.len();
                    let mut output =
                        unsafe { DeviceBuffer::uninitialized(len).expect("DeviceBuffer init") };
                    add(&self.module, &stream, &a, &b, &mut output, len).expect("CUDA add failed");

                    let mut value = vec![0.0f32; len];
                    output
                        .copy_to(&mut value)
                        .expect("Copy from DeviceBuffer failed");
                    value
                }
                TensorGraphNode::Sub => {
                    let ins = graph.inputs(*node_idx);
                    let a = values
                        .get(&ins[0])
                        .unwrap()
                        .as_slice()
                        .as_dbuf()
                        .expect("Copy to DeviceBuffer failed");
                    let b = values
                        .get(&ins[1])
                        .unwrap()
                        .as_slice()
                        .as_dbuf()
                        .expect("Copy to DeviceBuffer failed");
                    assert_eq!(a.len(), b.len(), "binary op input length mismatch");
                    let len = a.len();
                    let mut output =
                        unsafe { DeviceBuffer::uninitialized(len).expect("DeviceBuffer init") };
                    sub(&self.module, &stream, &a, &b, &mut output, len).expect("CUDA sub failed");

                    let mut value = vec![0.0f32; len];
                    output
                        .copy_to(&mut value)
                        .expect("Copy from DeviceBuffer failed");
                    value
                }
                TensorGraphNode::Mul => {
                    let ins = graph.inputs(*node_idx);
                    let a = values
                        .get(&ins[0])
                        .unwrap()
                        .as_slice()
                        .as_dbuf()
                        .expect("Copy to DeviceBuffer failed");
                    let b = values
                        .get(&ins[1])
                        .unwrap()
                        .as_slice()
                        .as_dbuf()
                        .expect("Copy to DeviceBuffer failed");
                    assert_eq!(a.len(), b.len(), "binary op input length mismatch");
                    let len = a.len();
                    let mut output =
                        unsafe { DeviceBuffer::uninitialized(len).expect("DeviceBuffer init") };
                    mul(&self.module, &stream, &a, &b, &mut output, len).expect("CUDA mul failed");

                    let mut value = vec![0.0f32; len];
                    output
                        .copy_to(&mut value)
                        .expect("Copy from DeviceBuffer failed");
                    value
                }
                TensorGraphNode::Div => {
                    let ins = graph.inputs(*node_idx);
                    let a = values
                        .get(&ins[0])
                        .unwrap()
                        .as_slice()
                        .as_dbuf()
                        .expect("Copy to DeviceBuffer failed");
                    let b = values
                        .get(&ins[1])
                        .unwrap()
                        .as_slice()
                        .as_dbuf()
                        .expect("Copy to DeviceBuffer failed");
                    assert_eq!(a.len(), b.len(), "binary op input length mismatch");
                    let len = a.len();
                    let mut output =
                        unsafe { DeviceBuffer::uninitialized(len).expect("DeviceBuffer init") };
                    div(&self.module, &stream, &a, &b, &mut output, len).expect("CUDA div failed");

                    let mut value = vec![0.0f32; len];
                    output
                        .copy_to(&mut value)
                        .expect("Copy from DeviceBuffer failed");
                    value
                }
                _ => unimplemented!(), // Implement other operations similarly
            };
            values.insert(*node_idx, result);
        }

        let last_node_idx = order.last().expect("Graph is empty");
        values.remove(last_node_idx).expect("Output not found")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use tensor::{Constant, Tensor};

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

    type UnaryApply = fn(Constant<f32>) -> Box<dyn Tensor<f32>>;
    fn neg_node(a: Constant<f32>) -> Box<dyn Tensor<f32>> {
        Box::new(-a)
    }
    fn exp_node(a: Constant<f32>) -> Box<dyn Tensor<f32>> {
        Box::new(a.exp())
    }
    fn log_node(a: Constant<f32>) -> Box<dyn Tensor<f32>> {
        Box::new(a.log())
    }
    fn relu_node(a: Constant<f32>) -> Box<dyn Tensor<f32>> {
        Box::new(a.relu())
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

    type BinaryApply = fn(Constant<f32>, Constant<f32>) -> Box<dyn Tensor<f32>>;
    fn add_node(a: Constant<f32>, b: Constant<f32>) -> Box<dyn Tensor<f32>> {
        Box::new(a + b)
    }
    fn sub_node(a: Constant<f32>, b: Constant<f32>) -> Box<dyn Tensor<f32>> {
        Box::new(a - b)
    }
    fn mul_node(a: Constant<f32>, b: Constant<f32>) -> Box<dyn Tensor<f32>> {
        Box::new(a * b)
    }
    fn div_node(a: Constant<f32>, b: Constant<f32>) -> Box<dyn Tensor<f32>> {
        Box::new(a / b)
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
}
