use std::collections::HashMap;

use crate::graph::{NodeIndex, TensorGraph};
use crate::tensor::{Constant, DType, Parameter, Shape, TensorExpr};

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

pub fn conv2d<D: DType + Default + 'static>(
    x: impl Into<TensorExpr<D>>,
    weight: impl Into<TensorExpr<D>>,
    stride: usize,
    padding: usize,
    bias: Option<impl Into<TensorExpr<D>>>,
) -> TensorExpr<D> {
    let conv = x.into().conv2d(weight, stride, padding);
    match bias {
        Some(b) => {
            let bias = b.into();
            let conv_shape = conv.shape().to_vec();
            conv + bias.broadcast(conv_shape)
        }
        None => conv,
    }
}

pub fn max_pool2d<D: DType + Default + 'static>(
    x: impl Into<TensorExpr<D>>,
    kernel_size: usize,
    stride: usize,
) -> TensorExpr<D> {
    x.into().max_pool2d(kernel_size, stride)
}

pub fn flatten<D: DType + Default + 'static>(x: impl Into<TensorExpr<D>>) -> TensorExpr<D> {
    x.into().flatten()
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

/// Numerically stable softmax along `axis`.
pub fn softmax<D: DType + 'static>(x: impl Into<TensorExpr<D>>, axis: usize) -> TensorExpr<D> {
    let x = x.into();
    let shape = x.shape().clone();
    assert!(
        axis < shape.len(),
        "softmax axis {axis} out of bounds for shape {shape:?}"
    );
    assert!(shape[axis] > 0, "softmax axis must be non-empty");

    let max = x.clone().reduce_max(axis).broadcast(shape.clone());
    let exp = (x - max).exp();
    let sum = exp.clone().reduce_sum(axis).broadcast(shape);
    exp / sum
}

/// Layer normalization over one axis with a learned scale and bias.
pub fn layer_norm(
    x: impl Into<TensorExpr<f32>>,
    axis: usize,
    weight: impl Into<TensorExpr<f32>>,
    bias: impl Into<TensorExpr<f32>>,
    epsilon: f32,
) -> TensorExpr<f32> {
    assert!(
        epsilon > 0.0 && epsilon.is_finite(),
        "layer_norm epsilon must be finite and positive"
    );

    let x = x.into();
    let shape = x.shape().clone();
    assert!(
        axis < shape.len(),
        "layer_norm axis {axis} out of bounds for shape {shape:?}"
    );

    let axis_size = shape[axis];
    assert!(axis_size > 0, "layer_norm axis must be non-empty");
    let weight = weight.into();
    let bias = bias.into();
    assert_eq!(
        weight.shape(),
        &vec![axis_size],
        "layer_norm weight must have shape [{axis_size}]"
    );
    assert_eq!(
        bias.shape(),
        &vec![axis_size],
        "layer_norm bias must have shape [{axis_size}]"
    );

    let mean = x.clone().reduce_mean(axis).broadcast(shape.clone());
    let centered = x - mean;
    let variance = (centered.clone() * centered.clone())
        .reduce_mean(axis)
        .broadcast(shape.clone());
    let inv_std = ((variance + epsilon).log() * -0.5).exp();

    let mut parameter_shape = vec![1; shape.len()];
    parameter_shape[axis] = axis_size;
    let weight = weight
        .reshape(parameter_shape.clone())
        .broadcast(shape.clone());
    let bias = bias.reshape(parameter_shape).broadcast(shape);
    centered * inv_std * weight + bias
}

/// Scaled dot-product attention over tensors shaped `[..., sequence, features]`.
pub fn scaled_dot_product_attention(
    query: impl Into<TensorExpr<f32>>,
    key: impl Into<TensorExpr<f32>>,
    value: impl Into<TensorExpr<f32>>,
    causal: bool,
) -> TensorExpr<f32> {
    let query = query.into();
    let key = key.into();
    let value = value.into();
    let q_shape = query.shape().clone();
    let k_shape = key.shape().clone();
    let v_shape = value.shape().clone();

    assert!(
        q_shape.len() >= 3,
        "attention query must have rank >= 3, got shape {q_shape:?}"
    );
    assert!(
        k_shape.len() >= 3,
        "attention key must have rank >= 3, got shape {k_shape:?}"
    );
    assert!(
        v_shape.len() >= 3,
        "attention value must have rank >= 3, got shape {v_shape:?}"
    );
    assert_eq!(
        q_shape.len(),
        k_shape.len(),
        "attention query and key ranks must match"
    );
    assert_eq!(
        q_shape.len(),
        v_shape.len(),
        "attention query and value ranks must match"
    );

    let rank = q_shape.len();
    assert_eq!(
        &q_shape[..rank - 2],
        &k_shape[..rank - 2],
        "attention query and key batch dimensions must match"
    );
    assert_eq!(
        &q_shape[..rank - 2],
        &v_shape[..rank - 2],
        "attention query and value batch dimensions must match"
    );
    assert_eq!(
        q_shape[rank - 1],
        k_shape[rank - 1],
        "attention query and key feature dimensions must match"
    );
    assert_eq!(
        k_shape[rank - 2],
        v_shape[rank - 2],
        "attention key and value sequence lengths must match"
    );
    assert!(
        q_shape[rank - 1] > 0,
        "attention feature dimension must be non-zero"
    );
    assert!(
        q_shape[rank - 2] > 0 && k_shape[rank - 2] > 0,
        "attention sequence lengths must be non-zero"
    );

    let key_transposed = key.swap_axes(rank - 2, rank - 1);
    let mut scores = query.matmul(key_transposed) * (q_shape[rank - 1] as f32).sqrt().recip();
    if causal {
        let query_len = q_shape[rank - 2];
        let key_len = k_shape[rank - 2];
        let mut mask = Vec::with_capacity(query_len * key_len);
        for query_index in 0..query_len {
            for key_index in 0..key_len {
                mask.push(if key_index <= query_index {
                    0.0
                } else {
                    f32::NEG_INFINITY
                });
            }
        }
        scores = scores + Constant::new(mask, vec![query_len, key_len]);
    }

    softmax(scores, rank - 1).matmul(value)
}

fn xavier_parameter(rows: usize, columns: usize, seed: u64) -> Parameter<f32> {
    let limit = (6.0 / (rows + columns) as f32).sqrt();
    let mut state = seed;
    let data = (0..rows * columns)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let unit = (state >> 40) as f32 / (1_u32 << 24) as f32;
            (unit * 2.0 - 1.0) * limit
        })
        .collect();
    Parameter::new(data, vec![rows, columns])
}

/// Trainable multi-head attention with independent Q, K, V, and output projections.
#[derive(Clone)]
pub struct MultiHeadAttention {
    pub query_weight: Parameter<f32>,
    pub query_bias: Parameter<f32>,
    pub key_weight: Parameter<f32>,
    pub key_bias: Parameter<f32>,
    pub value_weight: Parameter<f32>,
    pub value_bias: Parameter<f32>,
    pub output_weight: Parameter<f32>,
    pub output_bias: Parameter<f32>,
    embed_dim: usize,
    num_heads: usize,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert!(embed_dim > 0, "attention embed_dim must be non-zero");
        assert!(num_heads > 0, "attention num_heads must be non-zero");
        assert_eq!(
            embed_dim % num_heads,
            0,
            "attention embed_dim must be divisible by num_heads"
        );

        Self {
            query_weight: xavier_parameter(embed_dim, embed_dim, 1),
            query_bias: Parameter::new(vec![0.0; embed_dim], vec![embed_dim]),
            key_weight: xavier_parameter(embed_dim, embed_dim, 2),
            key_bias: Parameter::new(vec![0.0; embed_dim], vec![embed_dim]),
            value_weight: xavier_parameter(embed_dim, embed_dim, 3),
            value_bias: Parameter::new(vec![0.0; embed_dim], vec![embed_dim]),
            output_weight: xavier_parameter(embed_dim, embed_dim, 4),
            output_bias: Parameter::new(vec![0.0; embed_dim], vec![embed_dim]),
            embed_dim,
            num_heads,
        }
    }

    pub fn forward(
        &self,
        query: impl Into<TensorExpr<f32>>,
        key: impl Into<TensorExpr<f32>>,
        value: impl Into<TensorExpr<f32>>,
        causal: bool,
    ) -> TensorExpr<f32> {
        let query = query.into();
        let key = key.into();
        let value = value.into();
        let query_shape = query.shape().clone();
        let key_shape = key.shape().clone();
        let value_shape = value.shape().clone();
        assert!(
            query_shape.len() >= 3,
            "multi-head attention inputs must have rank >= 3"
        );
        assert!(
            key_shape.len() >= 3 && value_shape.len() >= 3,
            "multi-head attention inputs must have rank >= 3"
        );
        assert_eq!(
            query_shape.len(),
            key_shape.len(),
            "query and key ranks must match"
        );
        assert_eq!(
            query_shape.len(),
            value_shape.len(),
            "query and value ranks must match"
        );
        assert_eq!(
            query_shape.last(),
            Some(&self.embed_dim),
            "query's last dimension must equal embed_dim"
        );
        assert_eq!(
            key_shape.last(),
            Some(&self.embed_dim),
            "key's last dimension must equal embed_dim"
        );
        assert_eq!(
            value_shape.last(),
            Some(&self.embed_dim),
            "value's last dimension must equal embed_dim"
        );

        let query = linear(
            query,
            self.query_weight.clone(),
            Some(self.query_bias.clone()),
        );
        let key = linear(key, self.key_weight.clone(), Some(self.key_bias.clone()));
        let value = linear(
            value,
            self.value_weight.clone(),
            Some(self.value_bias.clone()),
        );
        let rank = query_shape.len();
        let head_dim = self.embed_dim / self.num_heads;

        let split_heads = |tensor: TensorExpr<f32>, shape: &Shape| {
            let mut split_shape = shape.clone();
            split_shape.pop();
            split_shape.extend([self.num_heads, head_dim]);
            tensor.reshape(split_shape).swap_axes(rank - 2, rank - 1)
        };
        let query = split_heads(query, &query_shape);
        let key = split_heads(key, &key_shape);
        let value = split_heads(value, &value_shape);
        let attended = scaled_dot_product_attention(query, key, value, causal);
        let merged = attended.swap_axes(rank - 2, rank - 1).reshape(query_shape);

        linear(
            merged,
            self.output_weight.clone(),
            Some(self.output_bias.clone()),
        )
    }
}

/// A pre-norm transformer block with a two-layer ReLU feed-forward network.
#[derive(Clone)]
pub struct TransformerBlock {
    pub attention: MultiHeadAttention,
    pub attention_norm_weight: Parameter<f32>,
    pub attention_norm_bias: Parameter<f32>,
    pub feed_forward_norm_weight: Parameter<f32>,
    pub feed_forward_norm_bias: Parameter<f32>,
    pub feed_forward_input_weight: Parameter<f32>,
    pub feed_forward_input_bias: Parameter<f32>,
    pub feed_forward_output_weight: Parameter<f32>,
    pub feed_forward_output_bias: Parameter<f32>,
    epsilon: f32,
}

impl TransformerBlock {
    pub fn new(embed_dim: usize, num_heads: usize, feed_forward_dim: usize, epsilon: f32) -> Self {
        assert!(
            feed_forward_dim > 0,
            "transformer feed_forward_dim must be non-zero"
        );
        assert!(
            epsilon > 0.0 && epsilon.is_finite(),
            "transformer epsilon must be finite and positive"
        );

        Self {
            attention: MultiHeadAttention::new(embed_dim, num_heads),
            attention_norm_weight: Parameter::new(vec![1.0; embed_dim], vec![embed_dim]),
            attention_norm_bias: Parameter::new(vec![0.0; embed_dim], vec![embed_dim]),
            feed_forward_norm_weight: Parameter::new(vec![1.0; embed_dim], vec![embed_dim]),
            feed_forward_norm_bias: Parameter::new(vec![0.0; embed_dim], vec![embed_dim]),
            feed_forward_input_weight: xavier_parameter(embed_dim, feed_forward_dim, 5),
            feed_forward_input_bias: Parameter::new(
                vec![0.0; feed_forward_dim],
                vec![feed_forward_dim],
            ),
            feed_forward_output_weight: xavier_parameter(feed_forward_dim, embed_dim, 6),
            feed_forward_output_bias: Parameter::new(vec![0.0; embed_dim], vec![embed_dim]),
            epsilon,
        }
    }

    pub fn forward(&self, x: impl Into<TensorExpr<f32>>, causal: bool) -> TensorExpr<f32> {
        let x = x.into();
        let rank = x.shape().len();
        assert!(rank >= 3, "transformer input must have rank >= 3");
        let feature_axis = rank - 1;

        let normalized = layer_norm(
            x.clone(),
            feature_axis,
            self.attention_norm_weight.clone(),
            self.attention_norm_bias.clone(),
            self.epsilon,
        );
        let attended =
            self.attention
                .forward(normalized.clone(), normalized.clone(), normalized, causal);
        let residual = x + attended;
        let normalized = layer_norm(
            residual.clone(),
            feature_axis,
            self.feed_forward_norm_weight.clone(),
            self.feed_forward_norm_bias.clone(),
            self.epsilon,
        );
        let hidden = linear(
            normalized,
            self.feed_forward_input_weight.clone(),
            Some(self.feed_forward_input_bias.clone()),
        )
        .relu();
        let output = linear(
            hidden,
            self.feed_forward_output_weight.clone(),
            Some(self.feed_forward_output_bias.clone()),
        );
        residual + output
    }
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

    #[test]
    fn transformer_expressions_have_expected_shapes() {
        let softmax_output: TensorExpr<f32> =
            softmax(constant_f32(vec![0.0; 24], vec![2, 3, 4]), 1);
        assert_eq!(softmax_output.shape(), &vec![2, 3, 4]);

        let norm_output = layer_norm(
            constant_f32(vec![0.0; 24], vec![2, 3, 4]),
            1,
            constant_f32(vec![1.0; 3], vec![3]),
            constant_f32(vec![0.0; 3], vec![3]),
            1e-5,
        );
        assert_eq!(norm_output.shape(), &vec![2, 3, 4]);

        let attention_output = scaled_dot_product_attention(
            constant_f32(vec![0.0; 2 * 4 * 3 * 8], vec![2, 4, 3, 8]),
            constant_f32(vec![0.0; 2 * 4 * 5 * 8], vec![2, 4, 5, 8]),
            constant_f32(vec![0.0; 2 * 4 * 5 * 6], vec![2, 4, 5, 6]),
            true,
        );
        assert_eq!(attention_output.shape(), &vec![2, 4, 3, 6]);

        let input = constant_f32(vec![0.0; 2 * 5 * 8], vec![2, 5, 8]);
        let attention = MultiHeadAttention::new(8, 2);
        assert_eq!(
            attention
                .forward(input.clone(), input.clone(), input.clone(), true)
                .shape(),
            &vec![2, 5, 8]
        );

        let block = TransformerBlock::new(8, 2, 16, 1e-5);
        assert_eq!(block.forward(input, true).shape(), &vec![2, 5, 8]);
    }
}
