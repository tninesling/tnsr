use tensor::Constant;
use tensor::DType;
use tensor::Shape;
use tensor::TensorExpr;

pub fn linear<D: DType + Default + 'static>(
    x: impl Into<TensorExpr<D>>,
    w: impl Into<TensorExpr<D>>,
    b: Option<impl Into<TensorExpr<D>>>,
) -> TensorExpr<D> {
    let y = x.into().matmul(w);
    match b {
        Some(bias) => y + bias,
        None => y + TensorExpr::<D>::constant(vec![D::default()], vec![]),
    }
}

pub fn mse_loss(
    pred: impl Into<TensorExpr<f32>>,
    target: impl Into<TensorExpr<f32>>,
) -> TensorExpr<f32> {
    let pred = pred.into();
    let target = target.into();
    let diff = pred.clone() - target;
    let sq = diff.clone() * diff;
    sq.mean_all()
}

pub fn constant_f32(data: Vec<f32>, shape: Shape) -> Constant<f32> {
    Constant::new(data, shape)
}

pub fn relu(x: impl Into<TensorExpr<f32>>) -> TensorExpr<f32> {
    x.into().relu()
}

pub fn reduce_logsumexp_simple(x: impl Into<TensorExpr<f32>>, axis: usize) -> TensorExpr<f32> {
    // Numerically stable-ish log-sum-exp without needing broadcast of per-row max
    let x = x.into();
    let m = x.clone().reduce_max(axis);
    let ex = x.exp();
    let sum = ex.reduce_sum(axis);
    let denom = m.clone().exp();
    let l = (sum / denom).log();
    l + m
}

// Cross-entropy for one-hot labels with logits input (no softmax),
// computed as mean(logsumexp(logits) - sum(labels * logits, class_axis)).
pub fn cross_entropy_one_hot_logits(
    logits: impl Into<TensorExpr<f32>>,
    labels_one_hot: impl Into<TensorExpr<f32>>,
    class_axis: usize,
) -> TensorExpr<f32> {
    // logsumexp over class axis
    let logits = logits.into();
    let lse = reduce_logsumexp_simple(logits.clone(), class_axis);

    // sum over classes of labels * logits -> per-example selected logit
    let picked = (logits.clone() * labels_one_hot).reduce_sum(class_axis);

    // per-example nll = lse - picked
    let nll = lse - picked;

    // mean over remaining axes (e.g., batch)
    nll.mean_all()
}

#[cfg(test)]
mod tests {
    use runtime::Executor;
    use runtime::SimpleExecutor;
    use runtime::autograd::forward_and_backward;
    use tensor::Input;
    use tensor::graph::TensorGraph;
    use tensor::graph::TensorGraphNode;

    use super::*; // bring trait methods like lower_to_graph into scope

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
            if let TensorGraphNode::Input { name: n } = &graph[idx]
                && *n == name
            {
                return idx;
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
            let b = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
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

        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
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
        // x shape [2,3]
        let x = Input::<f32>::new("x", vec![2, 3]);
        let lse = reduce_logsumexp_simple(x.clone(), 1);
        // loss = mean over batch -> scalar
        let loss = lse.reduce_mean(0);
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
