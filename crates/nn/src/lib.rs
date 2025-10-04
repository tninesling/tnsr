use tensor::Constant;
use tensor::DType;
use tensor::Shape;
use tensor::Tensor;

pub fn linear<D: DType + Default + 'static>(
    x: impl Tensor<D> + 'static,
    w: impl Tensor<D> + 'static,
    b: Option<impl Tensor<D> + 'static>,
) -> impl Tensor<D> {
    use tensor::TensorOps;
    let y = x.matmul(w);
    match b {
        Some(bias) => y + bias,
        None => y + Constant::new(vec![D::default()], vec![]),
    }
}

use std::sync::Arc;

pub fn mse_loss(
    pred: impl Tensor<f32> + 'static,
    target: impl Tensor<f32> + 'static,
) -> Arc<dyn Tensor<f32>> {
    // Build explicitly to avoid operator bounds on generic impl Trait params
    let diff = tensor::BinaryOpNode::new(pred, target, tensor::BinaryOp::Sub);
    let sq = tensor::BinaryOpNode::new(diff.clone(), diff, tensor::BinaryOp::Mul);

    let mut node: Arc<dyn Tensor<f32>> = Arc::new(sq);
    loop {
        let rank = node.shape().len();
        if rank == 0 {
            break;
        }
        node = Arc::new(tensor::ReduceOpNode::from_dyn(
            node,
            tensor::ReduceOp::Mean,
            rank - 1,
        ));
    }
    node
}

pub fn constant_f32(data: Vec<f32>, shape: Shape) -> Constant<f32> {
    Constant::new(data, shape)
}

// Helper to convert a generic tensor into a concrete BinaryOpNode by adding zero.
fn as_expr(x: impl Tensor<f32> + 'static) -> tensor::BinaryOpNode<f32> {
    let zero = Constant::new(vec![0.0f32], vec![]);
    tensor::BinaryOpNode::new(x, zero, tensor::BinaryOp::Add)
}

pub fn relu(x: impl Tensor<f32> + 'static) -> impl Tensor<f32> {
    as_expr(x).relu()
}

pub fn reduce_logsumexp_simple(x: impl Tensor<f32> + 'static, axis: usize) -> impl Tensor<f32> {
    // Numerically stable log-sum-exp without needing broadcast of per-row max
    let x_expr = as_expr(x);
    let m = tensor::ReduceOpNode::new(x_expr.clone(), tensor::ReduceOp::Max, axis);
    let ex = x_expr.exp();
    let sum = tensor::ReduceOpNode::new(ex, tensor::ReduceOp::Sum, axis);
    let denom = as_expr(m.clone()).exp();
    let l = as_expr(tensor::BinaryOpNode::new(sum, denom, tensor::BinaryOp::Div)).log();
    tensor::BinaryOpNode::new(l, m, tensor::BinaryOp::Add)
}

// Cross-entropy for one-hot labels with logits input (no softmax),
// computed as mean(logsumexp(logits) - sum(labels * logits, class_axis)).
pub fn cross_entropy_one_hot_logits(
    logits: impl Tensor<f32> + 'static,
    labels_one_hot: impl Tensor<f32> + 'static,
    class_axis: usize,
) -> Arc<dyn Tensor<f32>> {
    // logsumexp over class axis
    let logits_expr = as_expr(logits);
    let lse = reduce_logsumexp_simple(logits_expr.clone(), class_axis);

    // sum over classes of labels * logits -> per-example selected logit
    let prod = tensor::BinaryOpNode::new(logits_expr, labels_one_hot, tensor::BinaryOp::Mul);
    let picked = tensor::ReduceOpNode::new(prod, tensor::ReduceOp::Sum, class_axis);

    // per-example nll = lse - picked
    let nll = tensor::BinaryOpNode::new(lse, picked, tensor::BinaryOp::Sub);

    // mean over remaining axes (e.g., batch)
    let mut node: Arc<dyn Tensor<f32>> = Arc::new(nll);
    loop {
        let rank = node.shape().len();
        if rank == 0 {
            break;
        }
        node = Arc::new(tensor::ReduceOpNode::from_dyn(
            node,
            tensor::ReduceOp::Mean,
            rank - 1,
        ));
    }
    node
}

#[cfg(test)]
mod tests {
    use runtime::Executor;
    use runtime::SimpleExecutor;
    use runtime::autograd::forward_and_backward;
    use tensor::Input;
    use tensor::graph::TensorGraph;
    use tensor::graph::TensorGraphNode;

    use super::*;

    const EPSILON: f32 = 1e-5;

    fn assert_approx_eq(a: &[f32], b: &[f32]) {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            if (x - y).abs() > EPSILON {
                panic!("mismatch at {i}: {x} vs {y}");
            }
        }
    }

    use tensor::graph::NodeIndex;
    fn find_input(graph: &TensorGraph<f32>, name: &'static str) -> NodeIndex {
        for idx in graph.graph.node_indices() {
            if let TensorGraphNode::Input { name: n } = &graph[idx] {
                if *n == name {
                    return idx;
                }
            }
        }
        panic!("input {name} not found");
    }

    #[test]
    fn linear_no_bias() {
        let x = constant_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let w = constant_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

        let node = linear::<f32>(x, w, Option::<Constant<f32>>::None);

        let mut graph = TensorGraph::new();
        node.lower_to_graph(&mut graph);

        let exec = SimpleExecutor {};
        let out = exec.execute(&graph, Default::default());

        let expected = {
            let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            let (m, k, n) = (2usize, 3usize, 2usize);
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
        };

        assert_approx_eq(&out, &expected);
    }

    #[test]
    fn linear_with_bias_vector() {
        let x = constant_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let w = constant_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let b = constant_f32(vec![10.0, -1.0], vec![2]);

        let node = linear::<f32>(x, w, Some(b));

        let mut graph = TensorGraph::new();
        node.lower_to_graph(&mut graph);

        let exec = SimpleExecutor {};
        let out = exec.execute(&graph, Default::default());

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bmat = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias = [10.0f32, -1.0];
        let (m, k, n) = (2usize, 3usize, 2usize);
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * bmat[p * n + j];
                }
                expected[i * n + j] = sum + bias[j];
            }
        }

        assert_approx_eq(&out, &expected);
    }

    #[test]
    fn reduce_logsumexp_grad_matches_softmax() {
        use tensor::TensorOps;
        // x shape [2,3]
        let x = Input::<f32>::new("x", vec![2, 3]);
        let lse = reduce_logsumexp_simple(x.clone(), 1);
        // loss = mean over batch -> scalar
        let loss = tensor::ReduceOpNode::new(lse, tensor::ReduceOp::Mean, 0);
        let mut g = TensorGraph::new();
        let loss_idx = loss.lower_to_graph(&mut g);
        let x_idx = find_input(&g, "x");
        let xval = vec![1.0f32, 2.0, 3.0, -0.5, 2.0, 0.0]; // two rows
        let mut inputs = std::collections::HashMap::new();
        inputs.insert("x".to_string(), xval.clone());
        let res = forward_and_backward(&g, &inputs, loss_idx, None);
        // expected dx = softmax(x_row)/2 per row
        let mut expected = vec![0.0f32; 6];
        for row in 0..2 {
            let start = row * 3;
            let slice = &xval[start..start + 3];
            let maxv = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = slice.iter().map(|&v| (v - maxv).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for j in 0..3 {
                expected[start + j] = exps[j] / sum / 2.0;
            }
        }
        let dx = res.grads_by_node.get(&x_idx).expect("dx missing");
        assert_approx_eq(dx, &expected);
    }

    #[test]
    fn cross_entropy_forward_zero_logits() {
        // logits and labels constants, forward should be ln(C)
        let logits = constant_f32(vec![0.0; 2 * 3], vec![2, 3]);
        // one-hot for classes [0, 1]
        let labels = constant_f32(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]);
        let loss = cross_entropy_one_hot_logits(logits, labels, 1);
        let mut g = TensorGraph::new();
        loss.lower_to_graph(&mut g);
        let exec = SimpleExecutor {};
        let out = exec.execute(&g, Default::default());
        assert_eq!(out.len(), 1);
        let expected = (3.0f32).ln();
        assert!((out[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn cross_entropy_backward_matches_softmax_minus_one_hot() {
        // logits and labels as inputs
        let logits = Input::<f32>::new("logits", vec![2, 3]);
        let labels = Input::<f32>::new("labels", vec![2, 3]);
        let loss = cross_entropy_one_hot_logits(logits.clone(), labels.clone(), 1);
        let mut g = TensorGraph::new();
        let loss_idx = loss.lower_to_graph(&mut g);
        let logits_idx = find_input(&g, "logits");
        let labels_idx = find_input(&g, "labels");
        let z = vec![1.0f32, 0.5, -1.0, -0.5, 2.0, 0.0];
        let y = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut inputs = std::collections::HashMap::new();
        inputs.insert("logits".to_string(), z.clone());
        inputs.insert("labels".to_string(), y.clone());
        let res = forward_and_backward(&g, &inputs, loss_idx, None);
        let dz = res.grads_by_node.get(&logits_idx).expect("dz missing");
        // expected dz = (softmax(z) - y) / B
        let mut expected = vec![0.0f32; 6];
        for row in 0..2 {
            let start = row * 3;
            let slice = &z[start..start + 3];
            let maxv = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = slice.iter().map(|&v| (v - maxv).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for j in 0..3 {
                let sm = exps[j] / sum;
                expected[start + j] = (sm - y[start + j]) / 2.0;
            }
        }
        assert_approx_eq(dz, &expected);
        // labels gradient is not of interest here, but ensure it's finite-sized and present
        let _dl = res.grads_by_node.get(&labels_idx).unwrap();
    }
}
