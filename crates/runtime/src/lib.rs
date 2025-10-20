use std::collections::HashMap;

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod optimizer;

use tensor::graph::TensorGraph;
use tensor::graph::TensorGraphNode;
use tracing::trace_span;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

pub struct TracingGuard {
    _guard: tracing_chrome::FlushGuard,
}

pub fn init_chrome_tracing(file_path: &str) -> Result<TracingGuard, Box<dyn std::error::Error>> {
    let (chrome_layer, guard) = ChromeLayerBuilder::new().file(file_path).build();

    tracing_subscriber::registry().with(chrome_layer).init();

    Ok(TracingGuard { _guard: guard })
}

pub struct BackwardResult<D> {
    pub grads_by_node: HashMap<petgraph::graph::NodeIndex, Vec<D>>,
    pub grads_by_param: HashMap<usize, Vec<D>>,
    pub loss_value: Vec<D>,
}

pub trait Executor<D> {
    fn forward(&mut self, graph: &TensorGraph<D>, inputs: HashMap<String, Vec<D>>) -> Vec<D>;

    fn backward(
        &mut self,
        graph: &TensorGraph<D>,
        loss_node: petgraph::graph::NodeIndex,
        seed_grad: Option<Vec<D>>,
    ) -> BackwardResult<D>;
}

pub struct SimpleExecutor {
    values: HashMap<petgraph::graph::NodeIndex, Vec<f32>>,
    grads: HashMap<petgraph::graph::NodeIndex, Vec<f32>>,
}

impl SimpleExecutor {
    pub fn new() -> Self {
        SimpleExecutor {
            values: HashMap::new(),
            grads: HashMap::new(),
        }
    }

    fn neg(&self, x: &[f32]) -> Vec<f32> {
        let _span = trace_span!("neg").entered();
        x.iter().map(|v| -v).collect()
    }

    fn exp(&self, x: &[f32]) -> Vec<f32> {
        let _span = trace_span!("exp").entered();
        x.iter().copied().map(f32::exp).collect()
    }

    fn log(&self, x: &[f32]) -> Vec<f32> {
        let _span = trace_span!("log").entered();
        x.iter().copied().map(f32::ln).collect()
    }

    fn relu(&self, x: &[f32]) -> Vec<f32> {
        let _span = trace_span!("relu").entered();
        x.iter().map(|v| v.max(0.0)).collect()
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let _span = trace_span!("add").entered();
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }

    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let _span = trace_span!("sub").entered();
        a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
    }

    fn mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let _span = trace_span!("mul").entered();
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }

    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let _span = trace_span!("div").entered();
        a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()
    }

    fn neg_grad(&self, dy: &[f32]) -> Vec<f32> {
        let _span = trace_span!("neg").entered();
        dy.iter().map(|g| -g).collect()
    }

    fn exp_grad(&self, dy: &[f32], y: &[f32]) -> Vec<f32> {
        let _span = trace_span!("exp").entered();
        dy.iter().zip(y.iter()).map(|(g, y)| g * y).collect()
    }

    fn log_grad(&self, dy: &[f32], x: &[f32]) -> Vec<f32> {
        let _span = trace_span!("log").entered();
        dy.iter().zip(x.iter()).map(|(g, x)| g / x).collect()
    }

    fn relu_grad(&self, dy: &[f32], x: &[f32]) -> Vec<f32> {
        let _span = trace_span!("relu").entered();
        dy.iter()
            .zip(x.iter())
            .map(|(g, x)| if *x > 0.0 { *g } else { 0.0 })
            .collect()
    }

    fn add_grad(&self, dy: &[f32], output_shape: &[usize], a_shape: &[usize], b_shape: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let _span = trace_span!("add").entered();
        let da = reduce_like(dy, output_shape, a_shape);
        let db = reduce_like(dy, output_shape, b_shape);
        (da, db)
    }

    fn sub_grad(&self, dy: &[f32], output_shape: &[usize], a_shape: &[usize], b_shape: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let _span = trace_span!("sub").entered();
        let da = reduce_like(dy, output_shape, a_shape);
        let mut db = reduce_like(dy, output_shape, b_shape);
        for v in db.iter_mut() {
            *v = -*v;
        }
        (da, db)
    }

    fn mul_grad(&self, dy: &[f32], output_shape: &[usize], a_shape: &[usize], b_shape: &[usize], a_val: &[f32], b_val: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let _span = trace_span!("mul").entered();
        let tmp_a: Vec<f32> = dy.iter().zip(b_val.iter()).map(|(g, b)| g * b).collect();
        let tmp_b: Vec<f32> = dy.iter().zip(a_val.iter()).map(|(g, a)| g * a).collect();
        let da = reduce_like(&tmp_a, output_shape, a_shape);
        let db = reduce_like(&tmp_b, output_shape, b_shape);
        (da, db)
    }

    fn div_grad(&self, dy: &[f32], output_shape: &[usize], a_shape: &[usize], b_shape: &[usize], a_val: &[f32], b_val: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let _span = trace_span!("div").entered();
        let tmp_a: Vec<f32> = dy.iter().zip(b_val.iter()).map(|(g, b)| g / b).collect();
        let tmp_b: Vec<f32> = dy.iter().zip(a_val.iter()).map(|(g, a)| -g * a).collect();
        let mut b_sq = b_val.to_vec();
        for v in b_sq.iter_mut() {
            *v = *v * *v;
        }
        let tmp_b: Vec<f32> = tmp_b
            .iter()
            .zip(b_sq.iter())
            .map(|(t, bsq)| t / bsq)
            .collect();
        let da = reduce_like(&tmp_a, output_shape, a_shape);
        let db = reduce_like(&tmp_b, output_shape, b_shape);
        (da, db)
    }
}

impl Executor<f32> for SimpleExecutor {
    fn forward(&mut self, graph: &TensorGraph<f32>, inputs: HashMap<String, Vec<f32>>) -> Vec<f32> {
        let order = graph.toposort();
        let _fwd_span = trace_span!("forward", nodes = order.len()).entered();

        for node_idx in order.iter() {
            let node = &graph[*node_idx];
            let result = match node {
                TensorGraphNode::Constant { data } => {
                    let _span = trace_span!("constant", node = node_idx.index()).entered();
                    data.as_ref().clone()
                }
                TensorGraphNode::Input { name } => {
                    let _span =
                        trace_span!("input", node = node_idx.index(), name = name).entered();
                    inputs.get::<str>(name).expect("Input not found").clone()
                }
                TensorGraphNode::Parameter { data, .. } => {
                    let _span = trace_span!("parameter", node = node_idx.index()).entered();
                    data.lock().unwrap().clone()
                }
                TensorGraphNode::Unary { op } => {
                    let inputs = graph.inputs(*node_idx);
                    let x = self.values.get(&inputs[0]).unwrap();
                    match op {
                        tensor::UnaryOp::Neg => self.neg(x),
                        tensor::UnaryOp::Exp => self.exp(x),
                        tensor::UnaryOp::Log => self.log(x),
                        tensor::UnaryOp::Relu => self.relu(x),
                    }
                }
                TensorGraphNode::Binary { op } => {
                    let ins = graph.inputs(*node_idx);
                    let a = self.values.get(&ins[0]).unwrap();
                    let b = self.values.get(&ins[1]).unwrap();
                    assert_eq!(a.len(), b.len());
                    match op {
                        tensor::BinaryOp::Add => self.add(a, b),
                        tensor::BinaryOp::Sub => self.sub(a, b),
                        tensor::BinaryOp::Mul => self.mul(a, b),
                        tensor::BinaryOp::Div => self.div(a, b),
                    }
                }
                TensorGraphNode::MatMul => {
                    let _span = trace_span!("matmul", node = node_idx.index()).entered();
                    matmul_forward(graph, &self.values, *node_idx)
                }
                TensorGraphNode::Broadcast => {
                    let _span = trace_span!("broadcast", node = node_idx.index()).entered();
                    broadcast_forward(graph, &self.values, *node_idx)
                }
                TensorGraphNode::Reduce { op, axis } => {
                    let _span = trace_span!(
                        "reduce",
                        op = node.name(),
                        axis = *axis,
                        node = node_idx.index()
                    )
                    .entered();
                    reduce_forward(graph, &self.values, *node_idx, op, *axis)
                }
            };
            self.values.insert(*node_idx, result);
        }
        let last_node = order.last().expect("Graph is empty");
        let out = self.values.get(last_node).expect("Output not found");
        out.clone()
    }

    fn backward(
        &mut self,
        graph: &TensorGraph<f32>,
        loss_node: petgraph::graph::NodeIndex,
        seed_grad: Option<Vec<f32>>,
    ) -> BackwardResult<f32> {
        let order = graph.toposort();
        let _bwd_span = trace_span!("backward", nodes = order.len()).entered();

        let mut param_grads: HashMap<usize, Vec<f32>> = HashMap::new();

        // Initialize gradient for loss node
        let loss_shape = graph
            .shapes
            .get(&loss_node)
            .expect("Loss node shape missing");
        let seed = seed_grad.unwrap_or_else(|| vec![1.0f32; loss_shape.iter().product()]);
        self.grads.insert(loss_node, seed);

        // Backward pass in reverse topological order
        for &node_idx in order.iter().rev() {
            if let Some(dy) = self.grads.get(&node_idx).cloned() {
                let node = &graph[node_idx];
                match node {
                    TensorGraphNode::Constant { .. } => {}
                    TensorGraphNode::Input { .. } => {}
                    TensorGraphNode::Parameter { id, .. } => {
                        param_grads
                            .entry(*id)
                            .and_modify(|g| add_inplace(g, &dy))
                            .or_insert(dy);
                    }
                    TensorGraphNode::Unary { op } => {
                        let x_idx = graph.inputs(node_idx)[0];
                        let dx = match op {
                            tensor::UnaryOp::Neg => self.neg_grad(&dy),
                            tensor::UnaryOp::Exp => {
                                let y = self.values.get(&node_idx).unwrap();
                                self.exp_grad(&dy, y)
                            }
                            tensor::UnaryOp::Log => {
                                let x = self.values.get(&x_idx).unwrap();
                                self.log_grad(&dy, x)
                            }
                            tensor::UnaryOp::Relu => {
                                let x = self.values.get(&x_idx).unwrap();
                                self.relu_grad(&dy, x)
                            }
                        };
                        accumulate_grad(&mut self.grads, x_idx, dx);
                    }
                    TensorGraphNode::Binary { op } => {
                        let ins = graph.inputs(node_idx);
                        let a_idx = ins[0];
                        let b_idx = ins[1];
                        let a_shape = graph.shapes.get(&a_idx).unwrap();
                        let b_shape = graph.shapes.get(&b_idx).unwrap();
                        let a_val = self.values.get(&a_idx).unwrap();
                        let b_val = self.values.get(&b_idx).unwrap();
                        let output_shape = graph.shapes.get(&node_idx).unwrap();

                        let (da, db) = match op {
                            tensor::BinaryOp::Add => self.add_grad(&dy, output_shape, a_shape, b_shape),
                            tensor::BinaryOp::Sub => self.sub_grad(&dy, output_shape, a_shape, b_shape),
                            tensor::BinaryOp::Mul => self.mul_grad(&dy, output_shape, a_shape, b_shape, a_val, b_val),
                            tensor::BinaryOp::Div => self.div_grad(&dy, output_shape, a_shape, b_shape, a_val, b_val),
                        };
                        accumulate_grad(&mut self.grads, a_idx, da);
                        accumulate_grad(&mut self.grads, b_idx, db);
                    }
                    TensorGraphNode::MatMul => {
                        let _span = trace_span!("matmul", node = node_idx.index()).entered();
                        let ins = graph.inputs(node_idx);
                        let a_idx = ins[0];
                        let b_idx = ins[1];
                        let a_val = self.values.get(&a_idx).unwrap();
                        let b_val = self.values.get(&b_idx).unwrap();
                        let a_shape = graph.shapes.get(&a_idx).unwrap();
                        let b_shape = graph.shapes.get(&b_idx).unwrap();
                        let dy_shape = graph.shapes.get(&node_idx).unwrap();

                        // dA = dY * B^T
                        let da = matmul_grad_left(&dy, dy_shape, b_val, b_shape);
                        // dB = A^T * dY
                        let db = matmul_grad_right(a_val, a_shape, &dy, dy_shape);
                        accumulate_grad(&mut self.grads, a_idx, da);
                        accumulate_grad(&mut self.grads, b_idx, db);
                    }
                    TensorGraphNode::Broadcast => {
                        let _span = trace_span!("broadcast", node = node_idx.index()).entered();
                        let x_idx = graph.inputs(node_idx)[0];
                        let x_shape = graph.shapes.get(&x_idx).unwrap();
                        let y_shape = graph.shapes.get(&node_idx).unwrap();
                        let dx = reduce_like(&dy, y_shape, x_shape);
                        accumulate_grad(&mut self.grads, x_idx, dx);
                    }
                    TensorGraphNode::Reduce { op, axis } => {
                        let _span = trace_span!(
                            "reduce",
                            op = node.name(),
                            axis = *axis,
                            node = node_idx.index()
                        )
                        .entered();
                        let x_idx = graph.inputs(node_idx)[0];
                        let x_shape = graph.shapes.get(&x_idx).unwrap();
                        match op {
                            tensor::ReduceOp::Sum => {
                                let mut y_aligned_shape = x_shape.clone();
                                y_aligned_shape[*axis] = 1;
                                let dx = expand_to(&dy, &y_aligned_shape, x_shape);
                                accumulate_grad(&mut self.grads, x_idx, dx);
                            }
                            tensor::ReduceOp::Mean => {
                                let mut y_aligned_shape = x_shape.clone();
                                y_aligned_shape[*axis] = 1;
                                let mut dx = expand_to(&dy, &y_aligned_shape, x_shape);
                                let axis_size = x_shape[*axis];
                                for v in dx.iter_mut() {
                                    *v /= axis_size as f32;
                                }
                                accumulate_grad(&mut self.grads, x_idx, dx);
                            }
                            tensor::ReduceOp::Max => {
                                // For max, gradient goes to the element that was the maximum
                                accumulate_grad(
                                    &mut self.grads,
                                    x_idx,
                                    vec![0.0f32; x_shape.iter().product()],
                                );
                            }
                        }
                    }
                }
            }
        }

        let loss_value = self.values.get(&loss_node).unwrap().clone();

        BackwardResult {
            grads_by_node: self.grads.drain().collect(),
            grads_by_param: param_grads,
            loss_value,
        }
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

fn add_inplace(acc: &mut [f32], src: &[f32]) {
    for (a, s) in acc.iter_mut().zip(src.iter()) {
        *a += *s;
    }
}

fn accumulate_grad(
    map: &mut HashMap<petgraph::graph::NodeIndex, Vec<f32>>,
    idx: petgraph::graph::NodeIndex,
    add: Vec<f32>,
) {
    map.entry(idx)
        .and_modify(|g| add_inplace(g, &add))
        .or_insert(add);
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
    if grad_shape == target_shape {
        return grad.to_vec();
    }

    let gr = grad_shape.len();
    let tr = target_shape.len();
    let mut out = vec![0.0f32; target_shape.iter().product()];
    let g_strides = rowmajor_strides(grad_shape);
    let t_strides = rowmajor_strides(target_shape);
    let total_g: usize = grad_shape.iter().product();

    let mut g_coords = vec![0usize; gr.max(1)];
    for (g_idx, _) in grad.iter().enumerate().take(total_g) {
        let mut rem = g_idx;
        for (d, stride) in g_strides.iter().enumerate().take(gr) {
            g_coords[d] = if gr == 0 { 0 } else { rem / *stride };
            if gr > 0 {
                rem %= *stride;
            }
        }

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
    use super::*;

    #[test]
    fn test_simple_backward_pass() {
        // Create a simple computation graph: x^2 where x is a parameter
        let x = tensor::Parameter::new(vec![2.0f32], vec![1]);
        let x_sq = x.clone() * x;
        let graph: TensorGraph<f32> = x_sq.into();

        let mut executor = SimpleExecutor::new();

        // Test backward pass - gradient of x^2 at x=2 should be 2*x = 4
        executor.forward(&graph, Default::default());
        let param_grads = executor.backward(&graph, 1.into(), Some(vec![1.0f32]));

        assert_eq!(param_grads.grads_by_param.len(), 1);
        let grad = param_grads
            .grads_by_param
            .get(&0)
            .expect("Parameter gradient missing");
        assert_eq!(grad.len(), 1);
        assert!(
            (grad[0] - 4.0).abs() < 1e-6,
            "Expected gradient 4.0, got {}",
            grad[0]
        );
    }

    #[test]
    fn test_unary_backward_pass() {
        // Test backward pass for exp(x) where x = [1.0]
        let x = tensor::Parameter::new(vec![1.0f32], vec![1]);
        let exp = x.exp();
        let graph: TensorGraph<f32> = exp.into();

        let mut executor = SimpleExecutor::new();

        // Gradient of exp(x) at x=1 should be exp(1) ≈ 2.718
        executor.forward(&graph, Default::default());
        let param_grads = executor.backward(&graph, 1.into(), Some(vec![1.0f32]));

        assert_eq!(param_grads.grads_by_param.len(), 1);
        let grad = param_grads
            .grads_by_param
            .get(&0)
            .expect("Parameter gradient missing");
        assert_eq!(grad.len(), 1);
        let expected = 1.0f32.exp();
        assert!(
            (grad[0] - expected).abs() < 1e-6,
            "Expected gradient {}, got {}",
            expected,
            grad[0]
        );
    }
}
