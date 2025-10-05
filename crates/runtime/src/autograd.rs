use std::collections::HashMap;

use petgraph::graph::NodeIndex;
use tensor::graph::TensorGraph;
use tensor::graph::TensorGraphNode;
use tracing::Level;
use tracing::span;
use tracing::trace;

pub struct BackwardResult {
    pub grads_by_node: HashMap<NodeIndex, Vec<f32>>, // for debugging
    pub grads_by_param: HashMap<usize, Vec<f32>>,    // parameter id -> grad
    pub loss_value: Vec<f32>,                        // cached loss forward value
}

pub fn forward_and_backward(
    graph: &TensorGraph<f32>,
    inputs: &HashMap<String, Vec<f32>>,
    loss_node: NodeIndex,
    seed: Option<Vec<f32>>,
) -> BackwardResult {
    let outer = span!(Level::TRACE, "autograd_fwd_bwd");
    let _outer_g = outer.enter();

    // Forward pass with value cache
    let order = graph.toposort();
    let fwd_span = span!(Level::TRACE, "forward", nodes = order.len());
    let _fwd_g = fwd_span.enter();
    let mut values: HashMap<NodeIndex, Vec<f32>> = HashMap::with_capacity(order.len());
    for &node_idx in order.iter() {
        let out = match &graph[node_idx] {
            TensorGraphNode::Constant { data } => data.as_ref().clone(),
            TensorGraphNode::Input { name } => {
                inputs.get::<str>(name).expect("Input not found").clone()
            }
            TensorGraphNode::Parameter { data, .. } => data.lock().unwrap().clone(),
            TensorGraphNode::Neg => {
                let x_idx = graph.inputs(node_idx)[0];
                let x = values.get(&x_idx).unwrap();
                x.iter().map(|v| -v).collect()
            }
            TensorGraphNode::Exp => {
                let x_idx = graph.inputs(node_idx)[0];
                let x = values.get(&x_idx).unwrap();
                x.iter().map(|v| v.exp()).collect()
            }
            TensorGraphNode::Log => {
                let x_idx = graph.inputs(node_idx)[0];
                let x = values.get(&x_idx).unwrap();
                x.iter().map(|v| v.ln()).collect()
            }
            TensorGraphNode::Relu => {
                let x_idx = graph.inputs(node_idx)[0];
                let x = values.get(&x_idx).unwrap();
                x.iter().map(|v| v.max(0.0)).collect()
            }
            TensorGraphNode::Add => elemwise2(graph, &values, node_idx, |a, b| a + b),
            TensorGraphNode::Sub => elemwise2(graph, &values, node_idx, |a, b| a - b),
            TensorGraphNode::Mul => elemwise2(graph, &values, node_idx, |a, b| a * b),
            TensorGraphNode::Div => elemwise2(graph, &values, node_idx, |a, b| a / b),
            TensorGraphNode::MatMul => matmul_forward(graph, &values, node_idx),
            TensorGraphNode::Broadcast => broadcast_forward(graph, &values, node_idx),
            TensorGraphNode::Reduce { op, axis } => {
                reduce_forward(graph, &values, node_idx, op, *axis)
            }
        };
        values.insert(node_idx, out);
    }
    let loss_value = values.get(&loss_node).cloned().unwrap_or_else(Vec::new);
    trace!(loss_len = loss_value.len(), "forward_done");

    // Backward pass
    let bwd_span = span!(Level::TRACE, "backward");
    let _bwd_g = bwd_span.enter();
    let mut grads: HashMap<NodeIndex, Vec<f32>> = HashMap::new();
    let mut grads_by_param: HashMap<usize, Vec<f32>> = HashMap::new();
    let loss_shape = graph.shapes.get(&loss_node).expect("loss shape missing");
    let seed_grad = seed.unwrap_or_else(|| vec![1.0f32; loss_shape.iter().product()]);
    grads.insert(loss_node, seed_grad);

    for &node_idx in order.iter().rev() {
        if let Some(dy) = grads.get(&node_idx).cloned() {
            match &graph[node_idx] {
                TensorGraphNode::Constant { .. } => {}
                TensorGraphNode::Input { .. } => {}
                TensorGraphNode::Parameter { id, .. } => {
                    grads_by_param
                        .entry(*id)
                        .and_modify(|g| add_inplace(g, &dy))
                        .or_insert(dy);
                }
                TensorGraphNode::Neg => {
                    let x_idx = graph.inputs(node_idx)[0];
                    accumulate(&mut grads, x_idx, dy.into_iter().map(|v| -v).collect());
                }
                TensorGraphNode::Exp => {
                    let x_idx = graph.inputs(node_idx)[0];
                    let y = values.get(&node_idx).unwrap();
                    let dx: Vec<f32> = dy.iter().zip(y.iter()).map(|(g, y)| g * y).collect();
                    accumulate(&mut grads, x_idx, dx);
                }
                TensorGraphNode::Log => {
                    let x_idx = graph.inputs(node_idx)[0];
                    let x = values.get(&x_idx).unwrap();
                    let dx: Vec<f32> = dy.iter().zip(x.iter()).map(|(g, x)| g / x).collect();
                    accumulate(&mut grads, x_idx, dx);
                }
                TensorGraphNode::Relu => {
                    let x_idx = graph.inputs(node_idx)[0];
                    let x = values.get(&x_idx).unwrap();
                    let dx: Vec<f32> = dy
                        .iter()
                        .zip(x.iter())
                        .map(|(g, x)| if *x > 0.0 { *g } else { 0.0 })
                        .collect();
                    accumulate(&mut grads, x_idx, dx);
                }
                TensorGraphNode::Add
                | TensorGraphNode::Sub
                | TensorGraphNode::Mul
                | TensorGraphNode::Div => {
                    let inputs_idx = graph.inputs(node_idx);
                    let a_idx = inputs_idx[0];
                    let b_idx = inputs_idx[1];
                    let a_shape = graph.shapes.get(&a_idx).unwrap();
                    let b_shape = graph.shapes.get(&b_idx).unwrap();
                    let a_val = values.get(&a_idx).unwrap();
                    let b_val = values.get(&b_idx).unwrap();
                    match &graph[node_idx] {
                        TensorGraphNode::Add => {
                            let da =
                                reduce_like(&dy, graph.shapes.get(&node_idx).unwrap(), a_shape);
                            let db =
                                reduce_like(&dy, graph.shapes.get(&node_idx).unwrap(), b_shape);
                            accumulate(&mut grads, a_idx, da);
                            accumulate(&mut grads, b_idx, db);
                        }
                        TensorGraphNode::Sub => {
                            let da =
                                reduce_like(&dy, graph.shapes.get(&node_idx).unwrap(), a_shape);
                            let mut db =
                                reduce_like(&dy, graph.shapes.get(&node_idx).unwrap(), b_shape);
                            for v in db.iter_mut() {
                                *v = -*v;
                            }
                            accumulate(&mut grads, a_idx, da);
                            accumulate(&mut grads, b_idx, db);
                        }
                        TensorGraphNode::Mul => {
                            let tmp_a: Vec<f32> =
                                dy.iter().zip(b_val.iter()).map(|(g, b)| g * b).collect();
                            let tmp_b: Vec<f32> =
                                dy.iter().zip(a_val.iter()).map(|(g, a)| g * a).collect();
                            let da =
                                reduce_like(&tmp_a, graph.shapes.get(&node_idx).unwrap(), a_shape);
                            let db =
                                reduce_like(&tmp_b, graph.shapes.get(&node_idx).unwrap(), b_shape);
                            accumulate(&mut grads, a_idx, da);
                            accumulate(&mut grads, b_idx, db);
                        }
                        TensorGraphNode::Div => {
                            let tmp_a: Vec<f32> =
                                dy.iter().zip(b_val.iter()).map(|(g, b)| g / b).collect();
                            let tmp_b: Vec<f32> =
                                dy.iter().zip(a_val.iter()).map(|(g, a)| -g * a).collect();
                            let mut b_sq = b_val.clone();
                            for v in b_sq.iter_mut() {
                                *v = *v * *v;
                            }
                            let tmp_b: Vec<f32> = tmp_b
                                .iter()
                                .zip(b_sq.iter())
                                .map(|(t, bsq)| t / bsq)
                                .collect();
                            let da =
                                reduce_like(&tmp_a, graph.shapes.get(&node_idx).unwrap(), a_shape);
                            let db =
                                reduce_like(&tmp_b, graph.shapes.get(&node_idx).unwrap(), b_shape);
                            accumulate(&mut grads, a_idx, da);
                            accumulate(&mut grads, b_idx, db);
                        }
                        _ => unreachable!(),
                    }
                }
                TensorGraphNode::MatMul => {
                    let inputs_idx = graph.inputs(node_idx);
                    let a_idx = inputs_idx[0];
                    let b_idx = inputs_idx[1];
                    let a_val = values.get(&a_idx).unwrap();
                    let b_val = values.get(&b_idx).unwrap();
                    let a_shape = graph.shapes.get(&a_idx).unwrap();
                    let b_shape = graph.shapes.get(&b_idx).unwrap();
                    let dy_shape = graph.shapes.get(&node_idx).unwrap();
                    // dA = dY * B^T
                    let da = matmul_grad_left(&dy, dy_shape, b_val, b_shape);
                    // dB = A^T * dY
                    let db = matmul_grad_right(a_val, a_shape, &dy, dy_shape);
                    accumulate(&mut grads, a_idx, da);
                    accumulate(&mut grads, b_idx, db);
                }
                TensorGraphNode::Broadcast => {
                    let x_idx = graph.inputs(node_idx)[0];
                    let x_shape = graph.shapes.get(&x_idx).unwrap();
                    let y_shape = graph.shapes.get(&node_idx).unwrap();
                    let dx = reduce_like(&dy, y_shape, x_shape);
                    accumulate(&mut grads, x_idx, dx);
                }
                TensorGraphNode::Reduce { op, axis } => {
                    let x_idx = graph.inputs(node_idx)[0];
                    let x_shape = graph.shapes.get(&x_idx).unwrap();
                    match op {
                        tensor::ReduceOp::Sum => {
                            let mut y_aligned_shape = x_shape.clone();
                            y_aligned_shape[*axis] = 1;
                            let dx = expand_to(&dy, &y_aligned_shape, x_shape);
                            accumulate(&mut grads, x_idx, dx);
                        }
                        tensor::ReduceOp::Mean => {
                            let mut y_aligned_shape = x_shape.clone();
                            y_aligned_shape[*axis] = 1;
                            let mut dx = expand_to(&dy, &y_aligned_shape, x_shape);
                            let axis_size = x_shape[*axis];
                            for v in dx.iter_mut() {
                                *v /= axis_size as f32;
                            }
                            accumulate(&mut grads, x_idx, dx);
                        }
                        tensor::ReduceOp::Max => {
                            // Not implemented yet: for correctness, return zeros
                            accumulate(&mut grads, x_idx, vec![0.0f32; x_shape.iter().product()]);
                        }
                    }
                }
            }
        }
    }
    trace!(
        params = grads_by_param.len(),
        nodes = grads.len(),
        "backward_done"
    );

    BackwardResult {
        grads_by_node: grads,
        grads_by_param,
        loss_value,
    }
}

fn add_inplace(acc: &mut [f32], src: &[f32]) {
    for (a, s) in acc.iter_mut().zip(src.iter()) {
        *a += *s;
    }
}

fn accumulate(map: &mut HashMap<NodeIndex, Vec<f32>>, idx: NodeIndex, add: Vec<f32>) {
    map.entry(idx)
        .and_modify(|g| add_inplace(g, &add))
        .or_insert(add);
}

fn elemwise2<F: Fn(f32, f32) -> f32>(
    graph: &TensorGraph<f32>,
    values: &HashMap<NodeIndex, Vec<f32>>,
    node_idx: NodeIndex,
    f: F,
) -> Vec<f32> {
    let inputs = graph.inputs(node_idx);
    let a = values.get(&inputs[0]).unwrap();
    let b = values.get(&inputs[1]).unwrap();
    a.iter().zip(b.iter()).map(|(x, y)| f(*x, *y)).collect()
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
    values: &HashMap<NodeIndex, Vec<f32>>,
    node_idx: NodeIndex,
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
    values: &HashMap<NodeIndex, Vec<f32>>,
    node_idx: NodeIndex,
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
    values: &HashMap<NodeIndex, Vec<f32>>,
    node_idx: NodeIndex,
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

fn matmul_grad_left(dy: &[f32], dy_shape: &[usize], b: &[f32], b_shape: &[usize]) -> Vec<f32> {
    let (m, n) = (dy_shape[0], dy_shape[1]);
    let (kb, nb) = (b_shape[0], b_shape[1]);
    assert_eq!(n, nb);
    let k = kb;
    let mut da = vec![0.0f32; m * k];
    for i in 0..m {
        for p in 0..k {
            let mut sum = 0.0f32;
            for j in 0..n {
                sum += dy[i * n + j] * b[p * n + j];
            }
            da[i * k + p] = sum;
        }
    }
    da
}

fn matmul_grad_right(a: &[f32], a_shape: &[usize], dy: &[f32], dy_shape: &[usize]) -> Vec<f32> {
    let (m, k) = (a_shape[0], a_shape[1]);
    let (mdy, n) = (dy_shape[0], dy_shape[1]);
    assert_eq!(m, mdy);
    let mut db = vec![0.0f32; k * n];
    for p in 0..k {
        for j in 0..n {
            let mut sum = 0.0f32;
            for i in 0..m {
                sum += a[i * k + p] * dy[i * n + j];
            }
            db[p * n + j] = sum;
        }
    }
    db
}

fn reduce_like(grad: &[f32], grad_shape: &[usize], target_shape: &[usize]) -> Vec<f32> {
    // Sum-reduce grad to match target_shape, in broadcasting semantics
    if grad_shape == target_shape {
        return grad.to_vec();
    }
    // align right
    let gr = grad_shape.len();
    let tr = target_shape.len();
    let mut out = vec![0.0f32; target_shape.iter().product()];
    let g_strides = rowmajor_strides(grad_shape);
    let t_strides = rowmajor_strides(target_shape);
    let total_g: usize = grad_shape.iter().product();
    // enumerate grad coords; map to target coords by collapsing dims where target dim == 1 or missing
    let mut g_coords = vec![0usize; gr.max(1)];
    for (g_idx, _) in grad.iter().enumerate().take(total_g) {
        let mut rem = g_idx;
        for (d, stride) in g_strides.iter().enumerate().take(gr) {
            g_coords[d] = if gr == 0 { 0 } else { rem / *stride };
            if gr > 0 {
                rem %= *stride;
            }
        }
        // build target linear index
        let mut t_lin = 0usize;
        if tr > 0 {
            let mut t_dim = tr as isize - 1;
            let mut g_dim = gr as isize - 1;
            while t_dim >= 0 {
                let t_size = target_shape[t_dim as usize];
                let coord = if g_dim >= 0 {
                    g_coords[g_dim as usize]
                } else {
                    0
                };
                let t_coord = if t_size == 1 { 0 } else { coord };
                t_lin += t_coord * t_strides[t_dim as usize];
                t_dim -= 1;
                g_dim -= 1;
            }
        }
        out[t_lin] += grad[g_idx];
    }
    out
}

fn expand_to(x: &[f32], x_shape: &[usize], target_shape: &[usize]) -> Vec<f32> {
    if x_shape == target_shape {
        return x.to_vec();
    }
    // reuse broadcasting forward mapping logic but ensure shapes compatible
    let in_shape = x_shape;
    let out_shape = target_shape;
    let out_size: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; out_size];
    let in_rank = in_shape.len();
    let out_rank = out_shape.len();
    let out_strides = rowmajor_strides(out_shape);
    let in_strides = rowmajor_strides(in_shape);
    for (out_idx, out_elem) in out.iter_mut().enumerate().take(out_size) {
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
        *out_elem = x[in_linear];
    }
    out
}

#[cfg(test)]
mod tests {
    use tensor::Input;
    use tensor::TensorExpr;

    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], eps: f32) {
        assert_eq!(a.len(), b.len(), "len mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            if (x - y).abs() > eps {
                panic!("mismatch at {i}: {x} vs {y}");
            }
        }
    }

    fn find_input(graph: &TensorGraph<f32>, name: &'static str) -> NodeIndex {
        for idx in graph.graph.node_indices() {
            if let TensorGraphNode::Input { name: n } = &graph[idx]
                && *n == name
            {
                return idx;
            }
        }
        panic!("input {name} not found");
    }

    #[test]
    fn unary_exp_backward_mean() {
        let x = Input::<f32>::new("x", vec![4]);
        let y: TensorExpr<f32> = TensorExpr::from(x.clone()).exp();
        let loss = y.reduce_mean(0);
        let mut g = TensorGraph::new();
        let loss_idx = loss.lower_to_graph(&mut g);
        let x_idx = find_input(&g, "x");
        let xval = vec![0.0f32, 1.0, -1.0, 2.0];
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), xval.clone());
        let res = forward_and_backward(&g, &inputs, loss_idx, None);
        assert_eq!(res.loss_value.len(), 1);
        let dx = res.grads_by_node.get(&x_idx).expect("dx missing");
        let expected: Vec<f32> = xval.iter().map(|&v| v.exp() / 4.0).collect();
        approx_eq(dx, &expected, 1e-6);
    }

    #[test]
    fn unary_log_backward_mean() {
        let x = Input::<f32>::new("x", vec![4]);
        let y: TensorExpr<f32> = TensorExpr::from(x.clone()).log();
        let loss = y.reduce_mean(0);
        let mut g = TensorGraph::new();
        let loss_idx = loss.lower_to_graph(&mut g);
        let x_idx = find_input(&g, "x");
        let xval = vec![0.5f32, 1.0, 2.0, 4.0];
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), xval.clone());
        let res = forward_and_backward(&g, &inputs, loss_idx, None);
        let dx = res.grads_by_node.get(&x_idx).expect("dx missing");
        let expected: Vec<f32> = xval.iter().map(|&v| 1.0 / (v * 4.0)).collect();
        approx_eq(dx, &expected, 1e-6);
    }

    #[test]
    fn unary_neg_backward_mean() {
        let x = Input::<f32>::new("x", vec![4]);
        let y: TensorExpr<f32> = -TensorExpr::from(x.clone());
        let loss = y.reduce_mean(0);
        let mut g = TensorGraph::new();
        let loss_idx = loss.lower_to_graph(&mut g);
        let x_idx = find_input(&g, "x");
        let xval = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), xval);
        let res = forward_and_backward(&g, &inputs, loss_idx, None);
        let dx = res.grads_by_node.get(&x_idx).expect("dx missing");
        let expected = vec![-0.25f32; 4];
        approx_eq(dx, &expected, 1e-6);
    }

    #[test]
    fn relu_backward_mean() {
        let x = Input::<f32>::new("x", vec![4]);
        let y: TensorExpr<f32> = TensorExpr::from(x.clone()).relu();
        let loss = y.reduce_mean(0);
        let mut g = TensorGraph::new();
        let loss_idx = loss.lower_to_graph(&mut g);
        let x_idx = find_input(&g, "x");
        let xval = vec![-1.0f32, 0.0, 0.5, 2.0];
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), xval);
        let res = forward_and_backward(&g, &inputs, loss_idx, None);
        let dx = res.grads_by_node.get(&x_idx).expect("dx missing");
        let expected = vec![0.0f32, 0.0, 0.25, 0.25];
        approx_eq(dx, &expected, 1e-6);
    }

    #[test]
    fn add_broadcast_backward_mean() {
        let a = Input::<f32>::new("a", vec![2, 3]);
        let b = Input::<f32>::new("b", vec![3]);
        let y: TensorExpr<f32> =
            TensorExpr::from(a.clone()) + TensorExpr::from(b.clone()).broadcast(vec![2, 3]);
        let loss = y.reduce_mean(1).reduce_mean(0);
        let mut g = TensorGraph::new();
        let loss_idx = loss.lower_to_graph(&mut g);
        let a_idx = find_input(&g, "a");
        let b_idx = find_input(&g, "b");
        let aval = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bval = vec![10.0f32, 20.0, 30.0];
        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), aval);
        inputs.insert("b".to_string(), bval);
        let res = forward_and_backward(&g, &inputs, loss_idx, None);
        let da = res.grads_by_node.get(&a_idx).expect("da missing");
        let db = res.grads_by_node.get(&b_idx).expect("db missing");
        // loss is mean over 6 elements => each dy element = 1/6
        approx_eq(da, &[1.0 / 6.0; 6], 1e-6);
        // db accumulates over 2 rows: 2 * (1/6) per class
        approx_eq(db, &[2.0 / 6.0; 3], 1e-6);
    }

    #[test]
    fn mul_div_backward_mean() {
        let a = Input::<f32>::new("a", vec![3]);
        let b = Input::<f32>::new("b", vec![3]);
        // y1 = mean(a*b)
        let y1: TensorExpr<f32> = TensorExpr::from(a.clone()) * TensorExpr::from(b.clone());
        let loss1 = y1.reduce_mean(0);
        let mut g1 = TensorGraph::new();
        let loss1_idx = loss1.lower_to_graph(&mut g1);
        let a_idx1 = find_input(&g1, "a");
        let b_idx1 = find_input(&g1, "b");
        let aval = vec![1.0f32, -2.0, 3.0];
        let bval = vec![2.0f32, 4.0, -1.0];
        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), aval.clone());
        inputs.insert("b".to_string(), bval.clone());
        let res1 = forward_and_backward(&g1, &inputs, loss1_idx, None);
        let da1 = res1.grads_by_node.get(&a_idx1).unwrap();
        let db1 = res1.grads_by_node.get(&b_idx1).unwrap();
        let exp_da1: Vec<f32> = bval.iter().map(|&v| v / 3.0).collect();
        let exp_db1: Vec<f32> = aval.iter().map(|&v| v / 3.0).collect();
        approx_eq(da1, &exp_da1, 1e-6);
        approx_eq(db1, &exp_db1, 1e-6);

        // y2 = mean(a/b)
        let y2: TensorExpr<f32> = TensorExpr::from(a.clone()) / TensorExpr::from(b.clone());
        let loss2 = y2.reduce_mean(0);
        let mut g2 = TensorGraph::new();
        let loss2_idx = loss2.lower_to_graph(&mut g2);
        let a_idx2 = find_input(&g2, "a");
        let b_idx2 = find_input(&g2, "b");
        let res2 = forward_and_backward(&g2, &inputs, loss2_idx, None);
        let da2 = res2.grads_by_node.get(&a_idx2).unwrap();
        let db2 = res2.grads_by_node.get(&b_idx2).unwrap();
        let exp_da2: Vec<f32> = bval.iter().map(|&v| 1.0 / (v * 3.0)).collect();
        let exp_db2: Vec<f32> = aval
            .iter()
            .zip(bval.iter())
            .map(|(&a, &b)| -(a) / (b * b) / 3.0)
            .collect();
        approx_eq(da2, &exp_da2, 1e-6);
        approx_eq(db2, &exp_db2, 1e-6);
    }

    #[test]
    fn picked_sum_backward_mean() {
        // Check: loss = mean(sum(labels * logits, axis=1)) => dlogits = labels / B
        let logits = Input::<f32>::new("logits", vec![2, 3]);
        let labels = Input::<f32>::new("labels", vec![2, 3]);
        let prod: TensorExpr<f32> =
            TensorExpr::from(logits.clone()) * TensorExpr::from(labels.clone());
        let picked = prod.reduce_sum(1);
        let loss = picked.reduce_mean(0);
        let mut g = TensorGraph::new();
        let loss_idx = loss.lower_to_graph(&mut g);
        let logits_idx = find_input(&g, "logits");
        let labels_idx = find_input(&g, "labels");
        let z = vec![1.0f32, 0.5, -1.0, -0.5, 2.0, 0.0];
        let y = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut inputs = HashMap::new();
        inputs.insert("logits".to_string(), z);
        inputs.insert("labels".to_string(), y.clone());
        let res = forward_and_backward(&g, &inputs, loss_idx, None);
        let dlogits = res.grads_by_node.get(&logits_idx).expect("dlogits missing");
        let dlabels = res.grads_by_node.get(&labels_idx).expect("dlabels missing");
        // Expect dlogits == labels / B
        let expected_dlogits: Vec<f32> = y.iter().map(|&v| v / 2.0).collect();
        approx_eq(dlogits, &expected_dlogits, 1e-6);
        // dlabels == logits / B (not used elsewhere, but check shape/finition)
        assert_eq!(dlabels.len(), 6);
    }

    #[test]
    fn reduce_sum_mean_max_backward() {
        let x = Input::<f32>::new("x", vec![2, 3]);
        // loss_sum = mean(sum(x, axis=1)) => scalar
        let sum: TensorExpr<f32> = TensorExpr::from(x.clone()).reduce_sum(1);
        let loss_sum = sum.reduce_mean(0);
        let mut gsum = TensorGraph::new();
        let loss_sum_idx = loss_sum.lower_to_graph(&mut gsum);
        let x_idx_sum = find_input(&gsum, "x");
        let xval = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), xval.clone());
        let res_sum = forward_and_backward(&gsum, &inputs, loss_sum_idx, None);
        let dx_sum = res_sum.grads_by_node.get(&x_idx_sum).unwrap();
        approx_eq(dx_sum, &[0.5f32; 6], 1e-6);

        // loss_mean = mean(mean(x, axis=1)) => scalar == global mean
        let mean: TensorExpr<f32> = TensorExpr::from(x.clone()).reduce_mean(1);
        let loss_mean = mean.reduce_mean(0);
        let mut gmean = TensorGraph::new();
        let loss_mean_idx = loss_mean.lower_to_graph(&mut gmean);
        let x_idx_mean = find_input(&gmean, "x");
        let res_mean = forward_and_backward(&gmean, &inputs, loss_mean_idx, None);
        let dx_mean = res_mean.grads_by_node.get(&x_idx_mean).unwrap();
        approx_eq(dx_mean, &[1.0f32 / 6.0; 6], 1e-6);

        // loss_max = mean(max(x, axis=1)) => grads are zeros (not implemented)
        let max: TensorExpr<f32> = TensorExpr::from(x).reduce_max(1);
        let loss_max = max.reduce_mean(0);
        let mut gmax = TensorGraph::new();
        let loss_max_idx = loss_max.lower_to_graph(&mut gmax);
        let x_idx_max = find_input(&gmax, "x");
        let res_max = forward_and_backward(&gmax, &inputs, loss_max_idx, None);
        let dx_max = res_max.grads_by_node.get(&x_idx_max).unwrap();
        approx_eq(dx_max, &[0.0f32; 6], 1e-6);
    }

    #[test]
    fn matmul_backward_mean() {
        let a = Input::<f32>::new("a", vec![2, 3]);
        let b = Input::<f32>::new("b", vec![3, 2]);
        let y: TensorExpr<f32> = TensorExpr::from(a.clone()).matmul(b.clone());
        // loss = mean over axis 1 then axis 0 (equiv to global mean)
        let loss = y.reduce_mean(1).reduce_mean(0);
        let mut g = TensorGraph::new();
        let loss_idx = loss.lower_to_graph(&mut g);
        let a_idx = find_input(&g, "a");
        let b_idx = find_input(&g, "b");
        let aval = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
        let bval = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]; // [3,2]
        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), aval.clone());
        inputs.insert("b".to_string(), bval.clone());
        let res = forward_and_backward(&g, &inputs, loss_idx, None);
        let da = res.grads_by_node.get(&a_idx).unwrap();
        let db = res.grads_by_node.get(&b_idx).unwrap();
        // dL/dY is ones of shape [2,2] scaled by 1/4 (global mean)
        let dy = [0.25f32; 4];
        // compute expected da = dy * B^T
        let mut exp_da = vec![0.0f32; 2 * 3];
        // dy shape [2,2], B shape [3,2]
        for i in 0..2 {
            for p in 0..3 {
                let mut sum = 0.0f32;
                for j in 0..2 {
                    sum += dy[i * 2 + j] * bval[p * 2 + j];
                }
                exp_da[i * 3 + p] = sum;
            }
        }
        // expected db = A^T * dy
        let mut exp_db = vec![0.0f32; 3 * 2];
        for p in 0..3 {
            for j in 0..2 {
                let mut sum = 0.0f32;
                for i in 0..2 {
                    sum += aval[i * 3 + p] * dy[i * 2 + j];
                }
                exp_db[p * 2 + j] = sum;
            }
        }
        approx_eq(da, &exp_da, 1e-6);
        approx_eq(db, &exp_db, 1e-6);
    }
}
