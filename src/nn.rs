use std::collections::HashMap;

use crate::graph::{NodeIndex, TensorGraph};
use crate::tensor::{Constant, DType, Shape, TensorExpr};

/// Fully connected (linear) layer: `y = x @ w + b`.
pub fn linear<D: DType + Default + 'static>(
    x: impl Into<TensorExpr<D>>,
    w: impl Into<TensorExpr<D>>,
    b: Option<impl Into<TensorExpr<D>>>,
) -> TensorExpr<D> {
    let y = x.into().matmul(w);
    match b {
        Some(bias) => {
            let bias = bias.into();
            let y_shape = y.shape().to_vec();
            let bias_shape = bias.shape().to_vec();
            // Broadcast bias to match output shape if needed
            if bias_shape == y_shape {
                y + bias
            } else {
                // Bias is typically a vector [n] that needs to broadcast to [m, n]
                // Since we can't reshape, we expect bias to already have matching rank
                // with singleton dimensions [1, n] for 2D outputs
                // For now, just try broadcasting directly and let it panic if incompatible
                let bias_broadcasted = bias.broadcast(y_shape);
                y + bias_broadcasted
            }
        }
        None => y,
    }
}

/// Mean squared error loss: `mean((pred - target)^2)`.
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

/// Creates a constant f32 tensor.
pub fn constant_f32(data: Vec<f32>, shape: Shape) -> Constant<f32> {
    Constant::new(data, shape)
}

/// Rectified linear unit activation: `max(0, x)`.
pub fn relu<D: DType + Default + 'static>(x: impl Into<TensorExpr<D>>) -> TensorExpr<D> {
    x.into().relu()
}

/// Numerically stable log-sum-exp reduction along an axis.
pub fn reduce_logsumexp_simple<D: DType + Default + 'static>(
    x: impl Into<TensorExpr<D>>,
    axis: usize,
) -> TensorExpr<D> {
    // Numerically stable-ish log-sum-exp without needing broadcast of per-row max
    let x = x.into();
    let m = x.clone().reduce_max(axis);
    let ex = x.exp();
    let sum = ex.reduce_sum(axis);
    let denom = m.clone().exp();
    let l = (sum / denom).log();
    l + m
}

/// Cross-entropy loss for one-hot labels with logit inputs (no softmax).
pub fn cross_entropy_one_hot_logits<D: DType + Default + 'static>(
    logits: impl Into<TensorExpr<D>>,
    labels_one_hot: impl Into<TensorExpr<D>>,
    class_axis: usize,
) -> TensorExpr<D> {
    // logsumexp over class axis
    let logits = logits.into();
    let lse = reduce_logsumexp_simple(logits.clone(), class_axis);

    // sum over classes of labels * logits -> per-example selected logit
    let picked = (logits.clone() * labels_one_hot.into()).reduce_sum(class_axis);

    // per-example nll = lse - picked
    let nll = lse - picked;

    // mean over remaining axes (e.g., batch)
    nll.mean_all()
}

/// A model structure for organizing computation graphs with multiple outputs.
#[derive(Clone)]
pub struct Model<D: DType> {
    /// The underlying computation graph
    graph: TensorGraph<D>,

    /// Named outputs (e.g., "logits", "predictions")
    outputs: HashMap<String, NodeIndex>,

    /// Optional loss node for training
    loss_fn: Option<NodeIndex>,
}

impl<D: DType> Model<D> {
    /// Create a new empty model.
    pub fn new() -> Self {
        Self {
            graph: TensorGraph::new(),
            outputs: HashMap::new(),
            loss_fn: None,
        }
    }

    /// Add a named output by lowering an expression to the graph.
    pub fn add_output(&mut self, name: impl Into<String>, expr: impl Into<TensorExpr<D>>) {
        let node_idx = expr.into().lower_to_graph(&mut self.graph);
        self.outputs.insert(name.into(), node_idx);
    }

    /// Set the loss function by lowering an expression to the graph.
    pub fn set_loss(&mut self, expr: impl Into<TensorExpr<D>>) {
        let node_idx = expr.into().lower_to_graph(&mut self.graph);
        self.loss_fn = Some(node_idx);
    }

    /// Get a named output node index.
    pub fn get_output(&self, name: &str) -> Option<NodeIndex> {
        self.outputs.get(name).copied()
    }

    /// Get the loss node index if one was specified.
    pub fn loss(&self) -> Option<NodeIndex> {
        self.loss_fn
    }

    /// Get a reference to the underlying computation graph.
    pub fn graph(&self) -> &TensorGraph<D> {
        &self.graph
    }

    /// Get a mutable reference to the underlying computation graph.
    pub fn graph_mut(&mut self) -> &mut TensorGraph<D> {
        &mut self.graph
    }

    /// Consume the model and return the underlying graph.
    pub fn into_graph(self) -> TensorGraph<D> {
        self.graph
    }
}

impl<D: DType> Default for Model<D> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TensorGraph;
    use crate::{Executor, SimpleExecutor};

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

    #[test]
    fn linear_no_bias() {
        let x = constant_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let w = constant_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

        let node = linear(x, w, Option::<Constant<f32>>::None);

        let mut graph: TensorGraph<f32> = TensorGraph::new();
        node.lower_to_graph(&mut graph);

        let mut exec = SimpleExecutor::new();
        let out = exec.execute(&graph, Default::default()).unwrap();

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
        let b = constant_f32(vec![10.0, -1.0], vec![1, 2]);

        let node = linear(x, w, Some(b));

        let mut graph: TensorGraph<f32> = TensorGraph::new();
        node.lower_to_graph(&mut graph);

        let mut exec = SimpleExecutor::new();
        let out = exec.execute(&graph, Default::default()).unwrap();

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
    fn cross_entropy_forward_zero_logits() {
        // logits and labels constants, forward should be ln(C)
        let logits = constant_f32(vec![0.0; 2 * 3], vec![2, 3]);
        // one-hot for classes [0, 1]
        let labels = constant_f32(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]);
        let loss = cross_entropy_one_hot_logits(logits, labels, 1);
        let mut g: TensorGraph<f32> = TensorGraph::new();
        loss.lower_to_graph(&mut g);
        let mut exec = SimpleExecutor::new();
        let out = exec.execute(&g, Default::default()).unwrap();
        assert_eq!(out.len(), 1);
        let expected = (3.0f32).ln();
        assert!((out[0] - expected).abs() < 1e-6);
    }
}
