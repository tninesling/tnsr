use std::collections::HashMap;

pub mod autograd;
pub mod optimizer;

use tensor::graph::TensorGraph;
use tensor::graph::TensorGraphNode;
use tracing::Level;
use tracing::debug;
use tracing::span;

pub trait Executor<D> {
    fn execute(&self, graph: &TensorGraph<D>, inputs: HashMap<String, Vec<D>>) -> Vec<D>;
}

pub struct SimpleExecutor {}

impl Executor<f32> for SimpleExecutor {
    fn execute(&self, graph: &TensorGraph<f32>, inputs: HashMap<String, Vec<f32>>) -> Vec<f32> {
        let order = graph.toposort();
        let exec_span = span!(Level::TRACE, "execute_graph", nodes = order.len());
        let _enter = exec_span.enter();

        let mut values: HashMap<_, Vec<f32>> = HashMap::with_capacity(order.len());
        for node_idx in order.iter() {
            let node = &graph[*node_idx];
            let result = match node {
                TensorGraphNode::Constant { data } => data.as_ref().clone(),
                TensorGraphNode::Input { name } => {
                    inputs.get::<str>(name).expect("Input not found").clone()
                }
                TensorGraphNode::Parameter { data, .. } => data.lock().unwrap().clone(),
                TensorGraphNode::Neg => {
                    let inputs = graph.inputs(*node_idx);
                    let x = values.get(&inputs[0]).unwrap();
                    x.iter().map(|v| -v).collect()
                }
                TensorGraphNode::Exp => {
                    let inputs = graph.inputs(*node_idx);
                    let x = values.get(&inputs[0]).unwrap();
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
                    x.iter().map(|v| v.max(0.0)).collect()
                }
                TensorGraphNode::Add => {
                    let ins = graph.inputs(*node_idx);
                    let a = values.get(&ins[0]).unwrap();
                    let b = values.get(&ins[1]).unwrap();
                    assert_eq!(a.len(), b.len());
                    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
                }
                TensorGraphNode::Sub => {
                    let ins = graph.inputs(*node_idx);
                    let a = values.get(&ins[0]).unwrap();
                    let b = values.get(&ins[1]).unwrap();
                    assert_eq!(a.len(), b.len());
                    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
                }
                TensorGraphNode::Mul => {
                    let ins = graph.inputs(*node_idx);
                    let a = values.get(&ins[0]).unwrap();
                    let b = values.get(&ins[1]).unwrap();
                    assert_eq!(a.len(), b.len());
                    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
                }
                TensorGraphNode::Div => {
                    let ins = graph.inputs(*node_idx);
                    let a = values.get(&ins[0]).unwrap();
                    let b = values.get(&ins[1]).unwrap();
                    assert_eq!(a.len(), b.len());
                    a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()
                }
                TensorGraphNode::MatMul => matmul_forward(graph, &values, *node_idx),
                TensorGraphNode::Broadcast => broadcast_forward(graph, &values, *node_idx),
                TensorGraphNode::Reduce { op, axis } => {
                    reduce_forward(graph, &values, *node_idx, op, *axis)
                }
            };
            values.insert(*node_idx, result);
        }
        let last_node = order.last().expect("Graph is empty");
        let out = values.remove(last_node).expect("Output not found");
        debug!(len = out.len(), "execute_graph_done");
        out
    }
}

fn rowmajor_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut s = vec![0usize; shape.len()];
    s[shape.len() - 1] = 1;
    for i in (0..shape.len() - 1).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn broadcast_forward(
    graph: &TensorGraph<f32>,
    values: &HashMap<petgraph::graph::NodeIndex, Vec<f32>>,
    node_idx: petgraph::graph::NodeIndex,
) -> Vec<f32> {
    let in_idx = graph.inputs(node_idx)[0];
    let in_val = values.get(&in_idx).unwrap();
    let in_shape = graph.shapes.get(&in_idx).unwrap();
    let out_shape = graph.shapes.get(&node_idx).unwrap();
    let out_size: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; out_size];
    let in_rank = in_shape.len();
    let out_rank = out_shape.len();
    let out_strides = rowmajor_strides(out_shape);
    let in_strides = rowmajor_strides(in_shape);
    for (out_idx, out_elem) in out.iter_mut().enumerate() {
        let mut rem = out_idx;
        let mut in_linear = 0usize;
        for (dim, stride) in out_strides.iter().enumerate().take(out_rank) {
            let coord = if out_rank == 0 { 0 } else { rem / *stride };
            if out_rank > 0 {
                rem %= *stride;
            }
            let in_dim_opt = if dim + in_rank >= out_rank {
                Some(dim + in_rank - out_rank)
            } else {
                None
            };
            if let Some(in_dim) = in_dim_opt {
                let in_dim_size = in_shape[in_dim];
                let idx_in_dim = if in_dim_size == 1 { 0 } else { coord };
                let stride = if in_strides.is_empty() {
                    0
                } else {
                    in_strides[in_dim]
                };
                in_linear += idx_in_dim * stride;
            }
        }
        *out_elem = in_val[in_linear];
    }
    out
}

fn reduce_forward(
    graph: &TensorGraph<f32>,
    values: &HashMap<petgraph::graph::NodeIndex, Vec<f32>>,
    node_idx: petgraph::graph::NodeIndex,
    op: &tensor::ReduceOp,
    axis: usize,
) -> Vec<f32> {
    let in_idx = graph.inputs(node_idx)[0];
    let x = values.get(&in_idx).unwrap();
    let in_shape = graph.shapes.get(&in_idx).unwrap();
    let out_shape = graph.shapes.get(&node_idx).unwrap();
    let rank = in_shape.len();
    let strides = rowmajor_strides(in_shape);
    let out_size: usize = out_shape.iter().product();
    let mut out = match op {
        tensor::ReduceOp::Max => vec![f32::NEG_INFINITY; out_size],
        _ => vec![0.0f32; out_size],
    };
    let axis_size = in_shape[axis];
    let out_rank = out_shape.len();
    let out_strides = if out_rank == 0 {
        vec![]
    } else {
        rowmajor_strides(out_shape)
    };
    let mut coords = vec![0usize; rank];
    let total: usize = in_shape.iter().product();
    for (idx, _) in x.iter().enumerate().take(total) {
        let mut rem = idx;
        for (d, stride) in strides.iter().enumerate().take(rank) {
            coords[d] = if rank == 0 { 0 } else { rem / *stride };
            if rank > 0 {
                rem %= *stride;
            }
        }
        let out_lin = if out_rank == 0 {
            0
        } else {
            let mut out_lin = 0usize;
            let mut out_dim = 0usize;
            for (d, _) in coords.iter().enumerate().take(rank) {
                if d == axis {
                    continue;
                }
                out_lin += coords[d] * out_strides[out_dim];
                out_dim += 1;
            }
            out_lin
        };
        match op {
            tensor::ReduceOp::Sum => {
                out[out_lin] += x[idx];
            }
            tensor::ReduceOp::Mean => {
                out[out_lin] += x[idx] / axis_size as f32;
            }
            tensor::ReduceOp::Max => {
                out[out_lin] = out[out_lin].max(x[idx]);
            }
        }
    }
    out
}

fn matmul_forward(
    graph: &TensorGraph<f32>,
    values: &HashMap<petgraph::graph::NodeIndex, Vec<f32>>,
    node_idx: petgraph::graph::NodeIndex,
) -> Vec<f32> {
    let inputs_idx = graph.inputs(node_idx);
    let a_idx = inputs_idx[0];
    let b_idx = inputs_idx[1];
    let a = values.get(&a_idx).unwrap();
    let b = values.get(&b_idx).unwrap();
    let a_shape = graph.shapes.get(&a_idx).expect("shape missing for lhs");
    let b_shape = graph.shapes.get(&b_idx).expect("shape missing for rhs");
    assert_eq!(a_shape.len(), 2);
    assert_eq!(b_shape.len(), 2);
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
