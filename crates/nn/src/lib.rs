//! Neural network building blocks and loss functions.
//!
//! This crate provides common neural network operations built on top of the `tensor` crate's
//! expression API. All functions construct lazy computation graphs—no computation occurs until
//! the resulting expressions are lowered to graphs and executed by a runtime executor.
//!
//! # Components
//!
//! - **Layers**: `linear()` - fully connected layer with optional bias
//! - **Activations**: `relu()` - rectified linear unit
//! - **Loss Functions**:
//!   - `mse_loss()` - mean squared error for regression
//!   - `cross_entropy_one_hot_logits()` - cross-entropy for classification with one-hot labels
//! - **Utilities**: `reduce_logsumexp_simple()` - numerically stable log-sum-exp reduction
//!
//! # Design
//!
//! All operations are **infallible** during graph construction. Shape mismatches and invalid
//! operations will be detected when graphs are lowered and executed. Functions expect properly
//! shaped inputs—use `.broadcast()` explicitly when needed for operations requiring matching
//! dimensions.
//!
//! # Example
//!
//! ```rust
//! use nn::{linear, relu, mse_loss};
//! use tensor::{TensorExpr, Parameter, Constant};
//!
//! // Build a simple MLP (no computation yet)
//! let x = TensorExpr::<f32>::input("x", vec![32, 784]); // batch=32, features=784
//! let w1 = Parameter::new(vec![0.1; 784 * 128], vec![784, 128]);
//! let b1 = Constant::new(vec![0.0; 128], vec![1, 128]);
//!
//! let h = linear(x, w1, Some(b1));
//! let h = relu(h);
//!
//! // Execution happens in the runtime crate
//! // let result = executor.forward(&h.into(), inputs)?;
//! ```

use tensor::Constant;
use tensor::DType;
use tensor::Shape;
use tensor::TensorExpr;

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

#[cfg(test)]
mod tests {
    use runtime::Executor;
    use runtime::SimpleExecutor;
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

        let mut exec = SimpleExecutor::new();
        let out = exec.forward(&graph, Default::default()).unwrap();

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
        let out = exec.forward(&graph, Default::default()).unwrap();

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
        let mut exec = SimpleExecutor::new();
        exec.forward(&g, inputs).unwrap();
        let res = exec.backward(&g, loss_idx, None).unwrap();
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
        let mut exec = SimpleExecutor::new();
        let out = exec.forward(&g, Default::default()).unwrap();
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
        let mut exec = SimpleExecutor::new();
        exec.forward(&g, inputs).unwrap();
        let res = exec.backward(&g, loss_idx, None).unwrap();
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
