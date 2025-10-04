#![feature(portable_simd)]

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "simd")]
pub mod simd;

use std::collections::HashMap;

use tensor::graph::TensorGraph;
use tensor::graph::TensorGraphNode;

pub trait Executor<D> {
    fn execute(&self, graph: &TensorGraph<D>, inputs: HashMap<String, Vec<D>>) -> Vec<D>;
}

pub struct SimpleExecutor {}

impl Executor<f32> for SimpleExecutor {
    fn execute(&self, graph: &TensorGraph<f32>, inputs: HashMap<String, Vec<f32>>) -> Vec<f32> {
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
                    let input = values.get(&inputs[0]).unwrap();
                    input.iter().map(|x| -x).collect()
                }
                TensorGraphNode::Exp => {
                    let inputs = graph.inputs(*node_idx);
                    let input = values.get(&inputs[0]).unwrap();
                    input.iter().map(|x| x.exp()).collect()
                }
                TensorGraphNode::Log => {
                    let inputs = graph.inputs(*node_idx);
                    let input = values.get(&inputs[0]).unwrap();
                    input.iter().map(|x| x.ln()).collect()
                }
                TensorGraphNode::Relu => {
                    let inputs = graph.inputs(*node_idx);
                    let input = values.get(&inputs[0]).unwrap();
                    input.iter().map(|x| x.max(0.0)).collect()
                }
                TensorGraphNode::Add => {
                    let inputs = graph.inputs(*node_idx);
                    let a = values.get(&inputs[0]).unwrap();
                    let b = values.get(&inputs[1]).unwrap();
                    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
                }
                TensorGraphNode::Sub => {
                    let inputs = graph.inputs(*node_idx);
                    let a = values.get(&inputs[0]).unwrap();
                    let b = values.get(&inputs[1]).unwrap();
                    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
                }
                TensorGraphNode::Mul => {
                    let inputs = graph.inputs(*node_idx);
                    let a = values.get(&inputs[0]).unwrap();
                    let b = values.get(&inputs[1]).unwrap();
                    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
                }
                TensorGraphNode::Div => {
                    let inputs = graph.inputs(*node_idx);
                    let a = values.get(&inputs[0]).unwrap();
                    let b = values.get(&inputs[1]).unwrap();
                    a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()
                }
                // MatMul: expects two inputs, shapes available in graph.shapes
                TensorGraphNode::MatMul => {
                    let inputs_idx = graph.inputs(*node_idx);
                    let a_idx = inputs_idx[0];
                    let b_idx = inputs_idx[1];
                    let a = values.get(&a_idx).unwrap();
                    let b = values.get(&b_idx).unwrap();
                    let a_shape = graph.shapes.get(&a_idx).expect("shape missing for lhs");
                    let b_shape = graph.shapes.get(&b_idx).expect("shape missing for rhs");
                    if a_shape.len() != 2 || b_shape.len() != 2 {
                        panic!("MatMul only supports 2D tensors in executor");
                    }
                    let m = a_shape[0];
                    let k = a_shape[1];
                    let kb = b_shape[0];
                    let n = b_shape[1];
                    if k != kb {
                        panic!("MatMul inner dims mismatch at execute time");
                    }
                    let mut out = vec![0.0f32; m * n];
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0.0f32;
                            for p in 0..k {
                                let aval = a[i * k + p];
                                let bval = b[p * n + j];
                                sum += aval * bval;
                            }
                            out[i * n + j] = sum;
                        }
                    }
                    out
                }
                TensorGraphNode::Broadcast => {
                    // Single input; broadcast according to graph.shapes[node_idx]
                    let inputs_idx = graph.inputs(*node_idx);
                    let in_idx = inputs_idx[0];
                    let in_val = values.get(&in_idx).unwrap();
                    let in_shape = graph.shapes.get(&in_idx).expect("shape missing for broadcast input");
                    let out_shape = graph.shapes.get(node_idx).expect("shape missing for broadcast node");
                    // Support broadcasting where input rank <= out rank and trailing dims match or are 1
                    let mut out = vec![0.0f32; out_shape.iter().product()];
                    // We'll implement a simple indexing loop over output and map to input indices
                    let in_rank = in_shape.len();
                    let out_rank = out_shape.len();
                    let mut out_strides = vec![0usize; out_rank];
                    let mut in_strides = vec![0usize; in_rank];
                    // compute row-major strides
                    out_strides[out_rank - 1] = 1;
                    for i in (0..out_rank - 1).rev() {
                        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
                    }
                    if in_rank > 0 {
                        in_strides[in_rank - 1] = 1;
                        for i in (0..in_rank - 1).rev() {
                            in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
                        }
                    }
                    // iterate over output linear index and compute corresponding input index
                    for out_idx in 0..out.len() {
                        // decompose out_idx into coords
                        let mut rem = out_idx;
                        let mut in_linear = 0usize;
                        for dim in 0..out_rank {
                            let coord = rem / out_strides[dim];
                            rem = rem % out_strides[dim];
                            // corresponding input dim index (align right)
                            let in_dim_opt = if dim + in_rank >= out_rank {
                                Some(dim + in_rank - out_rank)
                            } else {
                                None
                            };
                            if let Some(in_dim) = in_dim_opt {
                                let in_dim_size = in_shape[in_dim];
                                let idx_in_dim = if in_dim_size == 1 { 0 } else { coord };
                                in_linear += idx_in_dim * in_strides[in_dim];
                            }
                        }
                        out[out_idx] = in_val[in_linear];
                    }
                    out
                }
                TensorGraphNode::Reduce { op: _, axis: _ } => todo!(),
            };
            values.insert(*node_idx, result);
        }

        let last_node = order.last().expect("Graph is empty");
        values.remove(last_node).expect("Output not found")
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::E;

    use rstest::rstest;
    use tensor::Constant;
    use tensor::Tensor;

    use super::*;

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
        vec![1.0, E, E.powi(2), E.powi(3)],
        exp_node as UnaryApply
    )]
    #[case::log(
        vec![1.0f32, E, E.powi(2), E.powi(3)],
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

        let executor = SimpleExecutor {};
        let result = executor.execute(&graph, Default::default());

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

        let executor = SimpleExecutor {};
        let result = executor.execute(&graph, Default::default());

        assert_approx_eq!(result, expected);
    }

    #[test]
    fn matmul_small_2x3_3x4() {
        let a_vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_vals = vec![
            1.0f32, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];
        let a = Constant::new(a_vals.clone(), vec![2, 3]);
        let b = Constant::new(b_vals.clone(), vec![3, 4]);
        let node = Box::new(a.matmul(b)) as Box<dyn Tensor<f32>>;

        let mut graph = TensorGraph::new();
        node.lower_to_graph(&mut graph);

        let executor = SimpleExecutor {};
        let result = executor.execute(&graph, Default::default());

        fn matmul_ref(a: &[f32], a_shape: &[usize], b: &[f32], b_shape: &[usize]) -> Vec<f32> {
            let m = a_shape[0];
            let k = a_shape[1];
            let n = b_shape[1];
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        sum += a[i * k + p] * b[p * n + j];
                    }
                    out[i * n + j] = sum;
                }
            }
            out
        }

        let expected = matmul_ref(&a_vals, &[2, 3], &b_vals, &[3, 4]);
        assert_approx_eq!(result, expected);
    }

    #[test]
    #[should_panic]
    fn matmul_mismatch_panics() {
        let a = Constant::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Constant::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![4, 2]);
        // constructor should panic due to inner-dimension mismatch
        let _ = a.matmul(b);
    }

    #[test]
    fn broadcast_vector_to_matrix() {
        let v = Constant::new(vec![10.0f32, 20.0f32], vec![2]);
        let node = Box::new(v.broadcast(vec![3, 2])) as Box<dyn Tensor<f32>>;
        let mut graph = TensorGraph::new();
        node.lower_to_graph(&mut graph);
        let exec = SimpleExecutor {};
        let result = exec.execute(&graph, Default::default());
        let expected = vec![10.0f32, 20.0f32, 10.0, 20.0, 10.0, 20.0];
        assert_approx_eq!(result, expected);
    }

    #[test]
    fn broadcast_scalar_to_matrix() {
        let s = Constant::new(vec![7.0f32], vec![]);
        let node = Box::new(s.broadcast(vec![2, 3])) as Box<dyn Tensor<f32>>;
        let mut graph = TensorGraph::new();
        node.lower_to_graph(&mut graph);
        let exec = SimpleExecutor {};
        let result = exec.execute(&graph, Default::default());
        let expected = vec![7.0f32; 6];
        assert_approx_eq!(result, expected);
    }

    #[test]
    fn broadcast_singleton_right_aligned() {
        let v = Constant::new(vec![10.0f32, 20.0f32, 30.0f32], vec![3, 1]);
        let node = Box::new(v.broadcast(vec![3, 4])) as Box<dyn Tensor<f32>>;
        let mut graph = TensorGraph::new();
        node.lower_to_graph(&mut graph);
        let exec = SimpleExecutor {};
        let result = exec.execute(&graph, Default::default());
        let expected = vec![
            10.0f32, 10.0, 10.0, 10.0,
            20.0, 20.0, 20.0, 20.0,
            30.0, 30.0, 30.0, 30.0,
        ];
        assert_approx_eq!(result, expected);
    }

    #[test]
    #[should_panic]
    fn broadcast_invalid_panics() {
        let v = Constant::new(vec![1.0f32, 2.0f32], vec![2]);
        // cannot broadcast [2] to [2,3]
        let _ = v.broadcast(vec![2, 3]);
    }

}
