#![cfg(feature = "simd")]

use std::collections::HashMap;
use std::simd::{num::SimdFloat, Simd};

use crate::Executor;
use tensor::graph::{TensorGraph, TensorGraphNode};

// Portable choice; rustc will split/merge as needed per target
pub const LANES: usize = 8;

#[inline]
fn unary_op(input: &[f32], f_vec: impl Fn(Simd<f32, LANES>) -> Simd<f32, LANES>, f_scalar: impl Fn(f32) -> f32) -> Vec<f32> {
    let len = input.len();
    let mut out = vec![0.0f32; len];
    let chunks = len / LANES;
    for c in 0..chunks {
        let base = c * LANES;
        let v = Simd::<f32, LANES>::from_slice(&input[base..base + LANES]);
        let r = f_vec(v);
        out[base..base + LANES].copy_from_slice(&r.to_array());
    }
    for i in (chunks * LANES)..len {
        out[i] = f_scalar(input[i]);
    }
    out
}

#[inline]
fn binary_op(a: &[f32], b: &[f32], f_vec: impl Fn(Simd<f32, LANES>, Simd<f32, LANES>) -> Simd<f32, LANES>, f_scalar: impl Fn(f32, f32) -> f32) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut out = vec![0.0f32; len];
    let chunks = len / LANES;
    for c in 0..chunks {
        let base = c * LANES;
        let va = Simd::<f32, LANES>::from_slice(&a[base..base + LANES]);
        let vb = Simd::<f32, LANES>::from_slice(&b[base..base + LANES]);
        let r = f_vec(va, vb);
        out[base..base + LANES].copy_from_slice(&r.to_array());
    }
    for i in (chunks * LANES)..len {
        out[i] = f_scalar(a[i], b[i]);
    }
    out
}

pub struct SimdExecutor;

impl Executor<f32> for SimdExecutor {
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
                    let x = values.get(&inputs[0]).unwrap();
                    unary_op(x, |v| -v, |x| -x)
                }
                TensorGraphNode::Exp => {
                    let inputs = graph.inputs(*node_idx);
                    let x = values.get(&inputs[0]).unwrap();
                    // scalar fallback for now
                    x.iter().copied().map(f32::exp).collect()
                }
                TensorGraphNode::Log => {
                    let inputs = graph.inputs(*node_idx);
                    let x = values.get(&inputs[0]).unwrap();
                    x.iter().copied().map(f32::ln).collect()
                }
                TensorGraphNode::Relu => {
                    let inputs = graph.inputs(*node_idx);
                    let x = values.get(&inputs[0]).unwrap();
                    unary_op(x, |v| v.simd_max(Simd::splat(0.0)), |x| x.max(0.0))
                }
                TensorGraphNode::Add => {
                    let inputs = graph.inputs(*node_idx);
                    let a = values.get(&inputs[0]).unwrap();
                    let b = values.get(&inputs[1]).unwrap();
                    binary_op(a, b, |x, y| x + y, |x, y| x + y)
                }
                TensorGraphNode::Sub => {
                    let inputs = graph.inputs(*node_idx);
                    let a = values.get(&inputs[0]).unwrap();
                    let b = values.get(&inputs[1]).unwrap();
                    binary_op(a, b, |x, y| x - y, |x, y| x - y)
                }
                TensorGraphNode::Mul => {
                    let inputs = graph.inputs(*node_idx);
                    let a = values.get(&inputs[0]).unwrap();
                    let b = values.get(&inputs[1]).unwrap();
                    binary_op(a, b, |x, y| x * y, |x, y| x * y)
                }
                TensorGraphNode::Div => {
                    let inputs = graph.inputs(*node_idx);
                    let a = values.get(&inputs[0]).unwrap();
                    let b = values.get(&inputs[1]).unwrap();
                    binary_op(a, b, |x, y| x / y, |x, y| x / y)
                }
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

    fn neg_node(a: Constant<f32>) -> Box<dyn Tensor<f32>> { Box::new(-a) }
    fn relu_node(a: Constant<f32>) -> Box<dyn Tensor<f32>> { Box::new(a.relu()) }
    fn exp_node(a: Constant<f32>) -> Box<dyn Tensor<f32>> { Box::new(a.exp()) }
    fn log_node(a: Constant<f32>) -> Box<dyn Tensor<f32>> { Box::new(a.log()) }

    fn sizes() -> Vec<usize> {
        vec![0usize, 1, LANES - 1, LANES, LANES + 1, 1024]
    }

    #[rstest]
    #[case::neg(neg_node as UnaryApply)]
    #[case::relu(relu_node as UnaryApply)]
    #[case::exp(exp_node as UnaryApply)]
    #[case::log(log_node as UnaryApply)]
    fn simd_unary_matches_simple(#[case] apply: UnaryApply) {
        for n in sizes() {
            let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 1.0).collect();
            let a = Constant::new(data.clone(), vec![n.max(1)]);
            let node = apply(a);

            let mut graph = TensorGraph::new();
            node.lower_to_graph(&mut graph);

            let simple = crate::SimpleExecutor {};
            let simd = SimdExecutor;

            let r_simple = simple.execute(&graph, Default::default());
            let r_simd = simd.execute(&graph, Default::default());

            assert_approx_eq!(r_simd, r_simple);
        }
    }

    type BinaryApply = fn(Constant<f32>, Constant<f32>) -> Box<dyn Tensor<f32>>;

    fn add_node(a: Constant<f32>, b: Constant<f32>) -> Box<dyn Tensor<f32>> { Box::new(a + b) }
    fn sub_node(a: Constant<f32>, b: Constant<f32>) -> Box<dyn Tensor<f32>> { Box::new(a - b) }
    fn mul_node(a: Constant<f32>, b: Constant<f32>) -> Box<dyn Tensor<f32>> { Box::new(a * b) }
    fn div_node(a: Constant<f32>, b: Constant<f32>) -> Box<dyn Tensor<f32>> { Box::new(a / b) }

    #[rstest]
    #[case::add(add_node as BinaryApply)]
    #[case::sub(sub_node as BinaryApply)]
    #[case::mul(mul_node as BinaryApply)]
    #[case::div(div_node as BinaryApply)]
    fn simd_binary_matches_simple(#[case] apply: BinaryApply) {
        for n in sizes() {
            let a_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 + 1.0).collect();
            let b_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 0.5).collect();
            let a = Constant::new(a_data.clone(), vec![n.max(1)]);
            let b = Constant::new(b_data.clone(), vec![n.max(1)]);
            let node = apply(a, b);

            let mut graph = TensorGraph::new();
            node.lower_to_graph(&mut graph);

            let simple = crate::SimpleExecutor {};
            let simd = SimdExecutor;

            let r_simple = simple.execute(&graph, Default::default());
            let r_simd = simd.execute(&graph, Default::default());

            assert_approx_eq!(r_simd, r_simple);
        }
    }
}
