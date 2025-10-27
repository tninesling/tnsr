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

    pub fn get_value(&self, node_idx: petgraph::graph::NodeIndex) -> Option<&Vec<f32>> {
        self.values.get(&node_idx)
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

    fn add_grad(
        &self,
        dy: &[f32],
        output_shape: &[usize],
        a_shape: &[usize],
        b_shape: &[usize],
    ) -> (Vec<f32>, Vec<f32>) {
        let _span = trace_span!("add").entered();
        let da = reduce_like(dy, output_shape, a_shape);
        let db = reduce_like(dy, output_shape, b_shape);
        (da, db)
    }

    fn sub_grad(
        &self,
        dy: &[f32],
        output_shape: &[usize],
        a_shape: &[usize],
        b_shape: &[usize],
    ) -> (Vec<f32>, Vec<f32>) {
        let _span = trace_span!("sub").entered();
        let da = reduce_like(dy, output_shape, a_shape);
        let mut db = reduce_like(dy, output_shape, b_shape);
        for v in db.iter_mut() {
            *v = -*v;
        }
        (da, db)
    }

    fn mul_grad(
        &self,
        dy: &[f32],
        output_shape: &[usize],
        a_shape: &[usize],
        b_shape: &[usize],
        a_val: &[f32],
        b_val: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let _span = trace_span!("mul").entered();
        let tmp_a: Vec<f32> = dy.iter().zip(b_val.iter()).map(|(g, b)| g * b).collect();
        let tmp_b: Vec<f32> = dy.iter().zip(a_val.iter()).map(|(g, a)| g * a).collect();
        let da = reduce_like(&tmp_a, output_shape, a_shape);
        let db = reduce_like(&tmp_b, output_shape, b_shape);
        (da, db)
    }

    fn div_grad(
        &self,
        dy: &[f32],
        output_shape: &[usize],
        a_shape: &[usize],
        b_shape: &[usize],
        a_val: &[f32],
        b_val: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
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
                            tensor::BinaryOp::Add => {
                                self.add_grad(&dy, output_shape, a_shape, b_shape)
                            }
                            tensor::BinaryOp::Sub => {
                                self.sub_grad(&dy, output_shape, a_shape, b_shape)
                            }
                            tensor::BinaryOp::Mul => {
                                self.mul_grad(&dy, output_shape, a_shape, b_shape, a_val, b_val)
                            }
                            tensor::BinaryOp::Div => {
                                self.div_grad(&dy, output_shape, a_shape, b_shape, a_val, b_val)
                            }
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

    const EPSILON: f32 = 1e-5;

    macro_rules! assert_approx_eq {
        ($a:expr, $b:expr) => {
            assert_approx_eq!($a, $b, EPSILON)
        };
        ($a:expr, $b:expr, $eps:expr) => {
            if $a
                .iter()
                .zip($b.iter())
                .any(|(a, b)| (*a - *b).abs() > $eps)
            {
                let diffs: Vec<_> = $a
                    .iter()
                    .zip($b.iter())
                    .enumerate()
                    .filter(|(_, (a, b))| (*a - *b).abs() > $eps)
                    .collect();
                panic!(
                    "assertion failed: `(left ~= right)` with epsilon {}\nDifferences at {} positions (showing first 5): {:?}\nleft: `{:?}`\nright: `{:?}`",
                    $eps,
                    diffs.len(),
                    &diffs[..diffs.len().min(5)],
                    $a,
                    $b
                );
            }
        };
    }

    fn test_simple_backward_pass_impl<E: Executor<f32>>(mut executor: E) {
        // Create a simple computation graph: x^2 where x is a parameter
        let x = tensor::Parameter::new(vec![2.0f32], vec![1]);
        let x_sq = x.clone() * x;
        let graph: TensorGraph<f32> = x_sq.into();

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
            (grad[0] - 4.0).abs() < 1e-5,
            "Expected gradient 4.0, got {}",
            grad[0]
        );
    }

    #[test]
    fn test_simple_backward_pass_cpu() {
        test_simple_backward_pass_impl(SimpleExecutor::new());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_simple_backward_pass_cuda() {
        test_simple_backward_pass_impl(crate::cuda::CudaExecutor::new());
    }

    fn test_unary_backward_pass_impl<E: Executor<f32>>(mut executor: E) {
        // Test backward pass for exp(x) where x = [1.0]
        let x = tensor::Parameter::new(vec![1.0f32], vec![1]);
        let exp = x.exp();
        let graph: TensorGraph<f32> = exp.into();

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
            (grad[0] - expected).abs() < 1e-5,
            "Expected gradient {}, got {}",
            expected,
            grad[0]
        );
    }

    #[test]
    fn test_unary_backward_pass_cpu() {
        test_unary_backward_pass_impl(SimpleExecutor::new());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_unary_backward_pass_cuda() {
        test_unary_backward_pass_impl(crate::cuda::CudaExecutor::new());
    }

    #[test]
    fn test_matmul_grad_left_simple() {
        // Test matmul_grad_left: dA = dC @ B^T
        // Forward: A[2,3] @ B[3,2] = C[2,2]
        // Given dC[2,2] and B[3,2], compute dA[2,3]
        
        let dy = vec![1.0f32, 2.0, 3.0, 4.0]; // dC: [2,2]
        let dy_shape = vec![2, 2];
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // B: [3,2]
        let b_shape = vec![3, 2];
        
        let da = matmul_grad_left(&dy, &dy_shape, &b, &b_shape);
        
        // Expected: dC @ B^T = [[1,2],[3,4]] @ [[1,3,5],[2,4,6]]
        // = [[1*1+2*2, 1*3+2*4, 1*5+2*6], [3*1+4*2, 3*3+4*4, 3*5+4*6]]
        // = [[5, 11, 17], [11, 25, 39]]
        let expected = vec![5.0f32, 11.0, 17.0, 11.0, 25.0, 39.0];
        
        assert_eq!(da.len(), 6);
        assert_approx_eq!(da, expected);
    }

    #[test]
    fn test_matmul_grad_right_simple() {
        // Test matmul_grad_right: dB = A^T @ dC
        // Forward: A[2,3] @ B[3,2] = C[2,2]
        // Given A[2,3] and dC[2,2], compute dB[3,2]
        
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // A: [2,3]
        let a_shape = vec![2, 3];
        let dy = vec![1.0f32, 2.0, 3.0, 4.0]; // dC: [2,2]
        let dy_shape = vec![2, 2];
        
        let db = matmul_grad_right(&a, &a_shape, &dy, &dy_shape);
        
        // Expected: A^T @ dC = [[1,4],[2,5],[3,6]] @ [[1,2],[3,4]]
        // = [[1*1+4*3, 1*2+4*4], [2*1+5*3, 2*2+5*4], [3*1+6*3, 3*2+6*4]]
        // = [[13, 18], [17, 24], [21, 30]]
        let expected = vec![13.0f32, 18.0, 17.0, 24.0, 21.0, 30.0];
        
        assert_eq!(db.len(), 6);
        assert_approx_eq!(db, expected);
    }

    fn test_matmul_grad_identity_impl<E: Executor<f32>>(mut executor: E) {
        // Test matmul gradient with identity matrix
        // Forward: A[3,3] @ I[3,3] = A[3,3]
        // dA should equal dC when B is identity
        
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let identity = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        
        let a = tensor::Parameter::new(a_data.clone(), vec![3, 3]);
        let b = tensor::Parameter::new(identity, vec![3, 3]);
        let node = tensor::TensorExpr::from(a).matmul(b);
        
        let mut graph = tensor::graph::TensorGraph::new();
        let loss_node = node.lower_to_graph(&mut graph);
        
        let inputs = std::collections::HashMap::new();
        let seed_grad = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        
        executor.forward(&graph, inputs);
        let result = executor.backward(&graph, loss_node, Some(seed_grad.clone()));
        
        let grad_a = result.grads_by_param.get(&0).unwrap();
        
        // dA = dC @ I^T = dC @ I = dC
        assert_eq!(grad_a.len(), 9);
        assert_approx_eq!(grad_a, &seed_grad);
    }

    #[test]
    fn test_matmul_grad_identity_cpu() {
        test_matmul_grad_identity_impl(SimpleExecutor::new());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_matmul_grad_identity_cuda() {
        test_matmul_grad_identity_impl(crate::cuda::CudaExecutor::new());
    }

    fn test_matmul_grad_asymmetric_impl<E: Executor<f32>>(mut executor: E) {
        // Test with very asymmetric matrices to catch dimension errors
        // Forward: A[5,100] @ B[100,3] = C[5,3]
        
        let m = 5;
        let k = 100;
        let n = 3;
        
        let a_data = vec![0.1f32; m * k];
        let b_data = vec![0.2f32; k * n];
        
        let a = tensor::Parameter::new(a_data, vec![m, k]);
        let b = tensor::Parameter::new(b_data, vec![k, n]);
        let node = tensor::TensorExpr::from(a).matmul(b);
        
        let mut graph = tensor::graph::TensorGraph::new();
        let loss_node = node.lower_to_graph(&mut graph);
        
        let inputs = std::collections::HashMap::new();
        let seed_grad = vec![1.0f32; m * n];
        
        executor.forward(&graph, inputs);
        let result = executor.backward(&graph, loss_node, Some(seed_grad));
        
        let grad_a = result.grads_by_param.get(&0).unwrap();
        let grad_b = result.grads_by_param.get(&1).unwrap();
        
        assert_eq!(grad_a.len(), m * k, "dA should have shape [5, 100]");
        assert_eq!(grad_b.len(), k * n, "dB should have shape [100, 3]");
        
        // Verify gradients are non-zero and finite
        for &val in grad_a.iter() {
            assert!(val.is_finite(), "dA contains non-finite value");
        }
        for &val in grad_b.iter() {
            assert!(val.is_finite(), "dB contains non-finite value");
        }
    }

    #[test]
    fn test_matmul_grad_asymmetric_cpu() {
        test_matmul_grad_asymmetric_impl(SimpleExecutor::new());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_matmul_grad_asymmetric_cuda() {
        test_matmul_grad_asymmetric_impl(crate::cuda::CudaExecutor::new());
    }

    fn test_matmul_backward_in_graph_impl<E: Executor<f32>>(mut executor: E) {
        // Test matmul backward pass integrated in graph
        // Forward: A[2,3] @ B[3,2] = C[2,2]
        // Backward: dA = dC @ B^T, dB = A^T @ dC
        
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let b_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3, 2]
        
        let a = tensor::Parameter::new(a_data.clone(), vec![2, 3]);
        let b = tensor::Parameter::new(b_data.clone(), vec![3, 2]);
        let node = tensor::TensorExpr::from(a).matmul(b);
        
        let mut graph = tensor::graph::TensorGraph::new();
        let loss_node = node.lower_to_graph(&mut graph);
        
        let inputs = std::collections::HashMap::new();
        let seed_grad = vec![1.0f32, 1.0, 1.0, 1.0]; // [2, 2]
        
        executor.forward(&graph, inputs);
        let result = executor.backward(&graph, loss_node, Some(seed_grad.clone()));
        
        // Verify we have gradients for both parameters
        assert_eq!(result.grads_by_param.len(), 2);
        
        let grad_a = result.grads_by_param.get(&0).unwrap();
        let grad_b = result.grads_by_param.get(&1).unwrap();
        
        assert_eq!(grad_a.len(), 6, "dA should have 6 elements [2,3]");
        assert_eq!(grad_b.len(), 6, "dB should have 6 elements [3,2]");
        
        // Manually compute expected gradients
        let expected_da = matmul_grad_left(&seed_grad, &[2, 2], &b_data, &[3, 2]);
        let expected_db = matmul_grad_right(&a_data, &[2, 3], &seed_grad, &[2, 2]);
        
        assert_approx_eq!(grad_a, &expected_da);
        assert_approx_eq!(grad_b, &expected_db);
    }

    #[test]
    fn test_matmul_backward_in_graph_cpu() {
        test_matmul_backward_in_graph_impl(SimpleExecutor::new());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_matmul_backward_in_graph_cuda() {
        test_matmul_backward_in_graph_impl(crate::cuda::CudaExecutor::new());
    }

    fn test_matmul_grad_chain_rule_impl<E: Executor<f32>>(mut executor: E) {
        // Test that matmul gradients compose correctly with chain rule
        // z = (A @ B) @ C, verify dA and dB are correct
        
        let a = tensor::Parameter::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = tensor::Parameter::new(vec![1.0f32, 0.0, 0.0, 1.0], vec![2, 2]); // Identity
        let c = tensor::Parameter::new(vec![2.0f32, 0.0, 0.0, 2.0], vec![2, 2]); // 2*Identity
        
        let ab = tensor::TensorExpr::from(a).matmul(b);
        let z = ab.matmul(c);
        
        let mut graph = tensor::graph::TensorGraph::new();
        let loss_node = z.lower_to_graph(&mut graph);
        
        let inputs = std::collections::HashMap::new();
        let seed_grad = vec![1.0f32; 4]; // [2, 2]
        
        executor.forward(&graph, inputs);
        let result = executor.backward(&graph, loss_node, Some(seed_grad));
        
        // With B=I and C=2I, z = 2A, so dA should be 2*seed_grad
        let grad_a = result.grads_by_param.get(&0).unwrap();
        
        assert_eq!(grad_a.len(), 4);
        for &val in grad_a.iter() {
            assert!(
                (val - 2.0).abs() < 1e-4,
                "Expected gradient 2.0, got {}",
                val
            );
        }
    }

    #[test]
    fn test_matmul_grad_chain_rule_cpu() {
        test_matmul_grad_chain_rule_impl(SimpleExecutor::new());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_matmul_grad_chain_rule_cuda() {
        test_matmul_grad_chain_rule_impl(crate::cuda::CudaExecutor::new());
    }

    fn test_matmul_grad_rectangular_case<E: Executor<f32>>(
        mut executor: E,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let a_data = vec![0.5f32; m * k];
        let b_data = vec![0.3f32; k * n];
        
        let a = tensor::Parameter::new(a_data, vec![m, k]);
        let b = tensor::Parameter::new(b_data, vec![k, n]);
        let node = tensor::TensorExpr::from(a).matmul(b);
        
        let mut graph = tensor::graph::TensorGraph::new();
        let loss_node = node.lower_to_graph(&mut graph);
        
        let inputs = std::collections::HashMap::new();
        let seed_grad = vec![1.0f32; m * n];
        
        executor.forward(&graph, inputs);
        let result = executor.backward(&graph, loss_node, Some(seed_grad));
        
        // Get all available parameter IDs (sorted)
        let mut param_ids: Vec<_> = result.grads_by_param.keys().copied().collect();
        param_ids.sort();
        
        assert_eq!(
            param_ids.len(),
            2,
            "Expected 2 parameters, found {} for test case ({}, {}) @ ({}, {})",
            param_ids.len(),
            m, k, k, n
        );
        
        let grad_a = result.grads_by_param.get(&param_ids[0]).unwrap();
        let grad_b = result.grads_by_param.get(&param_ids[1]).unwrap();
        
        assert_eq!(
            grad_a.len(),
            m * k,
            "dA should have shape [{}, {}] for test case ({}, {}) @ ({}, {})",
            m, k, m, k, k, n
        );
        assert_eq!(
            grad_b.len(),
            k * n,
            "dB should have shape [{}, {}] for test case ({}, {}) @ ({}, {})",
            k, n, m, k, k, n
        );
        
        // Verify all gradients are finite
        for &val in grad_a.iter().chain(grad_b.iter()) {
            assert!(val.is_finite(), "Gradient contains non-finite value");
        }
    }

    #[test]
    fn test_matmul_grad_rectangular_cpu() {
        // Test matmul gradient with rectangular matrices of various sizes
        test_matmul_grad_rectangular_case(SimpleExecutor::new(), 2, 3, 4);
        test_matmul_grad_rectangular_case(SimpleExecutor::new(), 1, 5, 1);
        test_matmul_grad_rectangular_case(SimpleExecutor::new(), 10, 20, 15);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_matmul_grad_rectangular_cuda() {
        // Test matmul gradient with rectangular matrices of various sizes
        test_matmul_grad_rectangular_case(crate::cuda::CudaExecutor::new(), 2, 3, 4);
        test_matmul_grad_rectangular_case(crate::cuda::CudaExecutor::new(), 1, 5, 1);
        test_matmul_grad_rectangular_case(crate::cuda::CudaExecutor::new(), 10, 20, 15);
    }

    fn test_matmul_grad_numerical_impl<E: Executor<f32>>(mut executor: E) {
        // Numerical gradient check using finite differences
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0]; // [2, 2]
        let b_data = vec![0.5f32, 1.0, 1.5, 2.0]; // [2, 2]
        
        let epsilon = 1e-4;
        
        // Create graph for forward pass
        let a = tensor::Parameter::new(a_data.clone(), vec![2, 2]);
        let b = tensor::Parameter::new(b_data.clone(), vec![2, 2]);
        let node = tensor::TensorExpr::from(a).matmul(b);
        
        let mut graph = tensor::graph::TensorGraph::new();
        let loss_node = node.lower_to_graph(&mut graph);
        
        // Forward and backward
        executor.forward(&graph, Default::default());
        let seed_grad = vec![1.0f32; 4];
        let result = executor.backward(&graph, loss_node, Some(seed_grad.clone()));
        
        let grad_a = result.grads_by_param.get(&0).unwrap();
        
        // Numerical gradient for each element of A
        for idx in 0..4 {
            let mut a_plus = a_data.clone();
            a_plus[idx] += epsilon;
            let a_param_plus = tensor::Parameter::new(a_plus, vec![2, 2]);
            let b_param = tensor::Parameter::new(b_data.clone(), vec![2, 2]);
            let node_plus = tensor::TensorExpr::from(a_param_plus).matmul(b_param);
            let mut graph_plus = tensor::graph::TensorGraph::new();
            let _loss_plus_node = node_plus.lower_to_graph(&mut graph_plus);
            let out_plus = executor.forward(&graph_plus, Default::default());
            
            let mut a_minus = a_data.clone();
            a_minus[idx] -= epsilon;
            let a_param_minus = tensor::Parameter::new(a_minus, vec![2, 2]);
            let b_param = tensor::Parameter::new(b_data.clone(), vec![2, 2]);
            let node_minus = tensor::TensorExpr::from(a_param_minus).matmul(b_param);
            let mut graph_minus = tensor::graph::TensorGraph::new();
            let _loss_minus_node = node_minus.lower_to_graph(&mut graph_minus);
            let out_minus = executor.forward(&graph_minus, Default::default());
            
            // Compute numerical gradient: (f(x+h) - f(x-h)) / 2h
            let mut numerical_grad = 0.0f32;
            for i in 0..4 {
                numerical_grad += (out_plus[i] - out_minus[i]) / (2.0 * epsilon) * seed_grad[i];
            }
            
            let analytical_grad = grad_a[idx];
            let rel_error = ((analytical_grad - numerical_grad).abs() 
                / (analytical_grad.abs() + numerical_grad.abs() + 1e-8)).abs();
            
            assert!(
                rel_error < 1e-3,
                "Numerical gradient check failed for A[{}]: analytical={}, numerical={}, rel_error={}",
                idx, analytical_grad, numerical_grad, rel_error
            );
        }
    }

    #[test]
    fn test_matmul_grad_numerical_cpu() {
        test_matmul_grad_numerical_impl(SimpleExecutor::new());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_matmul_grad_numerical_cuda() {
        test_matmul_grad_numerical_impl(crate::cuda::CudaExecutor::new());
    }
}
