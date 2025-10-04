// TODO: Use `cust` deps to launch CUDA stream from runtime
// use cust::module::Module;
// use cust::stream::Stream;
// use cust::stream::StreamFlags;
use kernels_core::Gemm;
use tensor::graph::TensorGraph;
use tensor::graph::TensorGraphNode;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref GEMM_REGISTRY: Mutex<Vec<Arc<dyn Gemm + Send + Sync>>> = Mutex::new(Vec::new());
}

pub fn register_gemm(g: Arc<dyn Gemm + Send + Sync>) {
    GEMM_REGISTRY.lock().unwrap().push(g);
}

pub fn get_preferred_gemm() -> Arc<dyn Gemm + Send + Sync> {
    GEMM_REGISTRY
        .lock()
        .unwrap()
        .last()
        .expect("No GEMM registered")
        .clone()
}

pub trait Executor<D> {
    fn execute(&self, graph: &TensorGraph<D>, inputs: HashMap<String, Vec<D>>) -> Vec<D>;
}

struct SimpleExecutor {}

impl Executor<f32> for SimpleExecutor {
    fn execute(&self, graph: &TensorGraph<f32>, inputs: HashMap<String, Vec<f32>>) -> Vec<f32> {
        let mut values: HashMap<_, Vec<f32>> = HashMap::new();
        let order = graph.toposort();
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
                TensorGraphNode::Reduce { op, axis } => todo!(),
            };
            values.insert(*node_idx, result);
        }

        let last_node = order.last().unwrap();
        values.remove(last_node).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::E;
    use tensor::Constant;
    use tensor::Tensor;

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

    #[test]
    fn elementwise_neg_f32() {
        let a = Constant::new(vec![1.0f32, -2.0, 3.0, -4.0], vec![2, 2]);
        let b = -a;

        let mut graph = TensorGraph::new();
        b.lower_to_graph(&mut graph);

        let executor = SimpleExecutor {};
        let result = executor.execute(&graph, Default::default());

        assert_approx_eq!(result, vec![-1.0, 2.0, -3.0, 4.0]);
    }

    #[test]
    fn elementwise_exp_f32() {
        let a = Constant::new(vec![0.0f32, 1.0, 2.0, 3.0], vec![2, 2]);
        let b = a.exp();

        let mut graph = TensorGraph::new();
        b.lower_to_graph(&mut graph);

        let executor = SimpleExecutor {};
        let result = executor.execute(&graph, Default::default());

        assert_approx_eq!(result, vec![1.0, E, E.powi(2), E.powi(3)]);
    }

    #[test]
    fn elementwise_log_f32() {
        let a = Constant::new(vec![1.0f32, E, E.powi(2), E.powi(3)], vec![2, 2]);
        let b = a.log();

        let mut graph = TensorGraph::new();
        b.lower_to_graph(&mut graph);

        let executor = SimpleExecutor {};
        let result = executor.execute(&graph, Default::default());

        assert_approx_eq!(result, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn elementwise_relu_f32() {
        let a = Constant::new(vec![-1.0f32, 2.0, -3.0, 4.0], vec![2, 2]);
        let b = a.relu();

        let mut graph = TensorGraph::new();
        b.lower_to_graph(&mut graph);

        let executor = SimpleExecutor {};
        let result = executor.execute(&graph, Default::default());

        assert_approx_eq!(result, vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn elementwise_add_f32() {
        let a = Constant::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Constant::new(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = a + b;

        let mut graph = TensorGraph::new();
        c.lower_to_graph(&mut graph);

        let executor = SimpleExecutor {};
        let result = executor.execute(&graph, Default::default());

        assert_approx_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn elementwise_sub_f32() {
        let a = Constant::new(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
        let b = Constant::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = a - b;

        let mut graph = TensorGraph::new();
        c.lower_to_graph(&mut graph);

        let executor = SimpleExecutor {};
        let result = executor.execute(&graph, Default::default());

        assert_approx_eq!(result, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn elementwise_mul_f32() {
        let a = Constant::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Constant::new(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = a * b;

        let mut graph = TensorGraph::new();
        c.lower_to_graph(&mut graph);

        let executor = SimpleExecutor {};
        let result = executor.execute(&graph, Default::default());

        assert_approx_eq!(result, vec![5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn elementwise_div_f32() {
        let a = Constant::new(vec![5.0f32, 12.0, 21.0, 32.0], vec![2, 2]);
        let b = Constant::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = a / b;

        let mut graph = TensorGraph::new();
        c.lower_to_graph(&mut graph);

        let executor = SimpleExecutor {};
        let result = executor.execute(&graph, Default::default());

        assert_approx_eq!(result, vec![5.0, 6.0, 7.0, 8.0]);
    }
}
