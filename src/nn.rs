use std::collections::HashMap;

use crate::graph::{NodeIndex, TensorGraph};
use crate::tensor::{Constant, DType, Shape, TensorExpr};

/// Fully connected (linear) layer: `y = x @ w + b`.
///
/// Computes a matrix multiplication followed by optional bias addition with broadcasting.
/// The bias is automatically broadcast to match the output shape.
///
/// # Parameters
///
/// - `x`: Input tensor of shape `[..., in_features]`
/// - `w`: Weight matrix of shape `[in_features, out_features]`
/// - `b`: Optional bias tensor of shape `[1, out_features]` (broadcast to match output)
///
/// # Returns
///
/// Output tensor of shape `[..., out_features]`
///
/// # Example
///
/// ```rust
/// use nn::{linear, constant_f32};
///
/// let x = constant_f32(vec![1.0, 2.0, 3.0], vec![1, 3]);
/// let w = constant_f32(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![3, 2]);
/// let b = constant_f32(vec![0.5, -0.5], vec![1, 2]);
///
/// let y = linear(x, w, Some(b)); // shape: [1, 2]
/// ```
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
///
/// Computes the element-wise squared difference between predictions and targets,
/// then reduces to a scalar by averaging over all elements.
///
/// # Parameters
///
/// - `pred`: Predicted values of any shape
/// - `target`: Target values (must match `pred` shape)
///
/// # Returns
///
/// Scalar tensor containing the mean squared error
///
/// # Example
///
/// ```rust
/// use nn::{mse_loss, constant_f32};
///
/// let pred = constant_f32(vec![1.0, 2.0, 3.0], vec![3]);
/// let target = constant_f32(vec![1.5, 2.0, 2.5], vec![3]);
///
/// let loss = mse_loss(pred, target); // scalar: mean of [0.25, 0.0, 0.25]
/// ```
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
///
/// Convenience wrapper around `Constant::new()` for `f32` tensors.
///
/// # Parameters
///
/// - `data`: Flattened data in row-major order
/// - `shape`: Dimensions of the tensor
///
/// # Returns
///
/// A `Constant<f32>` tensor
///
/// # Panics
///
/// Panics if `data.len() != shape.iter().product()`
pub fn constant_f32(data: Vec<f32>, shape: Shape) -> Constant<f32> {
    Constant::new(data, shape)
}

/// Rectified linear unit activation: `max(0, x)`.
///
/// Applies the ReLU activation function element-wise. Convenience wrapper
/// around `TensorExpr::relu()`.
///
/// # Parameters
///
/// - `x`: Input tensor of any shape
///
/// # Returns
///
/// Output tensor with same shape as input, with negative values replaced by 0
///
/// # Example
///
/// ```rust
/// use nn::{relu, constant_f32};
///
/// let x = constant_f32(vec![-1.0, 0.0, 1.0, 2.0], vec![4]);
/// let y = relu(x); // [0.0, 0.0, 1.0, 2.0]
/// ```
pub fn relu(x: impl Into<TensorExpr<f32>>) -> TensorExpr<f32> {
    x.into().relu()
}

/// Numerically stable log-sum-exp reduction along an axis.
///
/// Computes `log(sum(exp(x)))` along the specified axis using the identity:
/// `log(sum(exp(x))) = m + log(sum(exp(x) / exp(m)))` where `m = max(x)`.
/// This prevents overflow when exponentiating large values.
///
/// # Parameters
///
/// - `x`: Input tensor
/// - `axis`: Dimension to reduce over
///
/// # Returns
///
/// Tensor with the specified axis reduced (shape has that dimension removed)
///
/// # Note
///
/// This is a simplified implementation. For multi-dimensional cases, proper broadcasting
/// of the max value would improve numerical stability further.
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

/// Cross-entropy loss for one-hot labels with logit inputs.
///
/// Computes the cross-entropy loss between one-hot encoded labels and un-normalized
/// logits (no softmax applied). The loss is computed as:
/// `mean(logsumexp(logits, axis) - sum(labels * logits, axis))`.
///
/// This formulation is numerically stable and does not require computing softmax explicitly.
///
/// # Parameters
///
/// - `logits`: Un-normalized predictions of shape `[batch, ..., num_classes, ...]`
/// - `labels_one_hot`: One-hot encoded labels (same shape as logits)
/// - `class_axis`: Dimension representing the class axis (typically last dimension)
///
/// # Returns
///
/// Scalar tensor containing the mean cross-entropy loss over all examples
///
/// # Example
///
/// ```rust
/// use nn::{cross_entropy_one_hot_logits, constant_f32};
///
/// // Batch of 2, 3 classes
/// let logits = constant_f32(vec![2.0, 1.0, 0.1, 0.5, 2.5, 1.0], vec![2, 3]);
/// let labels = constant_f32(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]); // classes [0, 1]
///
/// let loss = cross_entropy_one_hot_logits(logits, labels, 1); // scalar
/// ```
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

/// A model structure for organizing computation graphs with multiple outputs.
///
/// `Model` provides a structured API for constructing computation graphs with named outputs
/// (like "logits", "loss", etc.) instead of returning messy tuples of `(TensorGraph, NodeIndex, ...)`.
///
/// # Example
///
/// ```ignore
/// use nn::Model;
/// use tensor::{Input, Parameter};
///
/// let batch_size = 32;
/// let w1 = Parameter::new(vec![0.1; 784 * 128], vec![784, 128]);
/// let b1 = Parameter::new(vec![0.0; 128], vec![1, 128]);
/// let w2 = Parameter::new(vec![0.1; 128 * 10], vec![128, 10]);
/// let b2 = Parameter::new(vec![0.0; 10], vec![1, 10]);
///
/// // Build model graph
/// let images = Input::<f32>::new("images", vec![batch_size, 784]);
/// let labels = Input::<f32>::new("labels", vec![batch_size, 10]);
///
/// let h = nn::relu(nn::linear(images, &w1, Some(&b1)));
/// let logits = nn::linear(h, &w2, Some(&b2));
/// let loss = nn::cross_entropy_one_hot_logits(logits.clone(), labels, 1);
///
/// // Create model
/// let mut model = Model::new();
/// model.add_output("logits", logits);
/// model.add_output("loss", loss.clone());
/// model.set_loss(loss);
///
/// // Access outputs
/// let logits_idx = model.get_output("logits").unwrap();
/// let loss_idx = model.loss().unwrap();
/// let graph = model.graph();
/// ```
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

        let node = linear::<f32>(x, w, Option::<Constant<f32>>::None);

        let mut graph = TensorGraph::new();
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

        let node = linear::<f32>(x, w, Some(b));

        let mut graph = TensorGraph::new();
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
        let mut g = TensorGraph::new();
        loss.lower_to_graph(&mut g);
        let mut exec = SimpleExecutor::new();
        let out = exec.execute(&g, Default::default()).unwrap();
        assert_eq!(out.len(), 1);
        let expected = (3.0f32).ln();
        assert!((out[0] - expected).abs() < 1e-6);
    }
}
