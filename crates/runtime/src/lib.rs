#[cfg(feature = "cuda")]
pub mod cuda;

use tensor::graph::TensorGraph;
use tensor::graph::TensorGraphNode;

use std::collections::HashMap;

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
                // The next two require understanding the shape
                TensorGraphNode::MatMul => todo!(),
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
    use tensor::{Constant, Tensor};

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
}
