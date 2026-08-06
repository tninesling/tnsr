use std::collections::{HashMap, HashSet};

use crate::graph::{TensorGraph, TensorGraphNode};

/// Stochastic Gradient Descent (SGD) optimizer.
///
/// Updates parameters using the gradient descent rule: `w = w - lr * grad`
/// where `lr` is the learning rate.
pub struct SGD {
    lr: f32,
}

impl SGD {
    /// Create a new SGD optimizer with the specified learning rate.
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }

    /// Perform a single optimization step, updating all parameters in the graph.
    ///
    /// Applies the update rule `w = w - lr * grad` to each parameter that has
    /// a corresponding gradient in `grads`.
    ///
    /// # Panics
    ///
    /// Panics if gradient size does not match parameter size.
    pub fn step<D, G>(&self, graph: &TensorGraph<D, G>, grads: &HashMap<usize, Vec<D>>)
    where
        D: num_traits::Float,
    {
        let mut updated: HashSet<usize> = HashSet::new();
        for node in graph.graph.node_weights() {
            if let TensorGraphNode::Parameter { id, data, .. } = node {
                if updated.contains(id) {
                    continue;
                }
                if let Some(grad) = grads.get(id) {
                    let mut w = data.lock().unwrap();
                    assert_eq!(
                        w.len(),
                        grad.len(),
                        "gradient size does not match parameter size"
                    );
                    let lr_d = D::from(self.lr).unwrap();
                    for (wi, &gi) in w.iter_mut().zip(grad.iter()) {
                        *wi = *wi - lr_d * gi;
                    }
                    updated.insert(*id);
                }
            }
        }
    }
}

/// Per-parameter optimizer state for Adam.
#[derive(Clone)]
struct AdamParamState<D> {
    /// First moment (mean) estimate.
    m: Vec<D>,
    /// Second moment (uncentered variance) estimate.
    v: Vec<D>,
}

/// Adam optimizer with bias-corrected first and second moment estimates.
///
/// Implements the Adam algorithm as described in [the paper](https://arxiv.org/pdf/1412.6980),
/// matching the defaults and behavior of
/// [PyTorch's `torch.optim.Adam`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html).
///
/// Adam maintains a per-parameter exponentially-decayed average of past gradients
/// (`m`) and past squared gradients (`v`). The update is bias-corrected to account
/// for initialization at zero:
///
/// ```text
/// m = β₁·m + (1-β₁)·g
/// v = β₂·v + (1-β₂)·g²
/// m̂ = m / (1-β₁ᵗ)
/// v̂ = v / (1-β₂ᵗ)
/// w = w - lr · m̂ / (√v̂ + ε)
/// ```
///
/// When `weight_decay > 0`, the gradient is adjusted as `g = g + weight_decay·w`
/// (L2 regularization), matching PyTorch's Adam.
pub struct Adam<D: num_traits::Float = f32> {
    /// Step size / learning rate.
    lr: f32,
    /// Exponential decay rate for the first moment estimate.
    beta1: f32,
    /// Exponential decay rate for the second moment estimate.
    beta2: f32,
    /// Constant added to the denominator for numerical stability.
    eps: f32,
    /// L2 regularization coefficient.
    weight_decay: f32,
    /// Global timestep, incremented on every `step`.
    t: usize,
    /// Per-parameter state, keyed by parameter ID.
    state: HashMap<usize, AdamParamState<D>>,
}

impl<D: num_traits::Float> Adam<D> {
    /// Create a new Adam optimizer with the specified learning rate and
    /// PyTorch defaults: `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0`.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            t: 0,
            state: HashMap::new(),
        }
    }

    /// Set the decay rates `(beta1, beta2)` for the first and second moments.
    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Set the epsilon constant for numerical stability.
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Set the L2 weight decay coefficient.
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Perform a single optimization step, updating all parameters in the graph.
    ///
    /// For each parameter with a gradient, updates the first/second moment
    /// estimates and applies the bias-corrected Adam update.
    ///
    /// # Panics
    ///
    /// Panics if gradient size does not match parameter size.
    pub fn step<G>(&mut self, graph: &TensorGraph<D, G>, grads: &HashMap<usize, Vec<D>>) {
        self.t += 1;
        let t = self.t as i32;

        let b1 = D::from(self.beta1).unwrap();
        let b2 = D::from(self.beta2).unwrap();
        let one = D::one();
        let zero = D::zero();
        let one_minus_b1 = one - b1;
        let one_minus_b2 = one - b2;
        let bias_correction1 = one - b1.powi(t);
        let bias_correction2 = one - b2.powi(t);
        let lr = D::from(self.lr).unwrap();
        let eps = D::from(self.eps).unwrap();
        let wd = D::from(self.weight_decay).unwrap();

        let mut updated: HashSet<usize> = HashSet::new();
        for node in graph.graph.node_weights() {
            if let TensorGraphNode::Parameter { id, data, .. } = node {
                if updated.contains(id) {
                    continue;
                }
                let id = *id;
                if let Some(grad) = grads.get(&id) {
                    let mut w = data.lock().unwrap();
                    assert_eq!(
                        w.len(),
                        grad.len(),
                        "gradient size does not match parameter size"
                    );

                    let st = self.state.entry(id).or_insert_with(|| AdamParamState {
                        m: vec![zero; w.len()],
                        v: vec![zero; w.len()],
                    });

                    for i in 0..w.len() {
                        // L2 weight decay: g = g + wd * w
                        let g = grad[i] + wd * w[i];

                        // Update biased first and second moment estimates
                        st.m[i] = b1 * st.m[i] + one_minus_b1 * g;
                        st.v[i] = b2 * st.v[i] + one_minus_b2 * g * g;

                        // Bias-corrected estimates
                        let m_hat = st.m[i] / bias_correction1;
                        let v_hat = st.v[i] / bias_correction2;

                        // w = w - lr * m̂ / (√v̂ + ε)
                        w[i] = w[i] - lr * m_hat / (v_hat.sqrt() + eps);
                    }
                    updated.insert(id);
                }
            }
        }
    }

    /// Reset all optimizer state (momentum buffers and timestep).
    pub fn reset(&mut self) {
        self.t = 0;
        self.state.clear();
    }

    /// Current timestep (number of `step` calls).
    pub fn timestep(&self) -> usize {
        self.t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{NoGrad, TensorGraph};
    use crate::tensor::{Parameter, TensorExpr};

    const EPSILON: f32 = 1e-6;

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

    /// Build a minimal graph with a single parameter so we can drive `step`.
    fn single_param_graph(data: Vec<f32>) -> (TensorGraph<f32, NoGrad>, usize) {
        let len = data.len();
        let p = Parameter::new(data, vec![len]);
        let id = p.id();
        let graph: TensorGraph<f32, NoGrad> = TensorExpr::from(p).into();
        (graph, id)
    }

    /// Read the first parameter's data out of a graph.
    fn read_param(graph: &TensorGraph<f32, NoGrad>) -> Vec<f32> {
        graph
            .graph
            .node_weights()
            .find_map(|n| match n {
                TensorGraphNode::Parameter { data, .. } => Some(data.lock().unwrap().clone()),
                _ => None,
            })
            .unwrap()
    }

    #[test]
    fn adam_first_step_matches_reference() {
        // w0 = [1.0, 2.0], grad = [0.1, 0.2], lr=0.1, defaults
        let (graph, id) = single_param_graph(vec![1.0, 2.0]);
        let grads = HashMap::from([(id, vec![0.1, 0.2])]);

        let mut opt = Adam::new(0.1);
        opt.step(&graph, &grads);

        // t=1: m = (1-b1)*g = 0.1*g, v = (1-b2)*g^2 = 0.001*g^2
        // m̂ = m / (1-b1^1) = g, v̂ = v / (1-b2^1) = g^2
        // w = w - lr * g / (|g| + eps)
        // For g=0.1: w[0] = 1.0 - 0.1*0.1/(0.1 + 1e-8) = 1.0 - 0.0999999...
        let expected = vec![
            1.0 - 0.1 * 0.1 / (0.1 + 1e-8),
            2.0 - 0.1 * 0.2 / (0.2 + 1e-8),
        ];
        assert_approx_eq(&read_param(&graph), &expected);
    }

    #[test]
    fn adam_timestep_increments() {
        let (graph, id) = single_param_graph(vec![1.0]);
        let grads = HashMap::from([(id, vec![1.0])]);

        let mut opt = Adam::new(0.01);
        assert_eq!(opt.timestep(), 0);
        opt.step(&graph, &grads);
        assert_eq!(opt.timestep(), 1);
        opt.step(&graph, &grads);
        assert_eq!(opt.timestep(), 2);
    }

    #[test]
    fn adam_converges_simple_quadratic() {
        // Minimize f(w) = 0.5 * (w - 5)^2, so grad = w - 5.
        // Adam should drive w toward 5.0.
        let (graph, id) = single_param_graph(vec![0.0]);
        let mut opt = Adam::new(0.1);

        for _ in 0..200 {
            let w = read_param(&graph)[0];
            let grads = HashMap::from([(id, vec![w - 5.0])]);
            opt.step(&graph, &grads);
        }

        let w = read_param(&graph)[0];
        assert!(
            (w - 5.0).abs() < 0.01,
            "Adam should converge to 5.0, got {w}"
        );
    }

    #[test]
    fn adam_weight_decay_applied() {
        // With weight_decay > 0 and zero gradient, the only force on w is
        // L2 decay: g = wd * w. This should shrink w toward zero.
        let (graph, id) = single_param_graph(vec![10.0]);
        let mut opt = Adam::new(0.1).weight_decay(0.01);

        let grads = HashMap::from([(id, vec![0.0])]);
        opt.step(&graph, &grads);

        // g = 0.01 * 10.0 = 0.1
        // m = 0.1 * 0.1 = 0.01, v = 0.001 * 0.01 = 0.0001
        // m̂ = 0.01, v̂ = 0.0001
        // w = 10.0 - 0.1 * 0.01 / (0.01 + 1e-8)
        let expected = 10.0 - 0.1 * 0.01 / (0.01 + 1e-8);
        assert_approx_eq(&[read_param(&graph)[0]], &[expected]);
    }

    #[test]
    fn adam_reset_clears_state() {
        let (graph, id) = single_param_graph(vec![1.0]);
        let grads = HashMap::from([(id, vec![1.0])]);

        let mut opt = Adam::new(0.1);
        opt.step(&graph, &grads);
        opt.step(&graph, &grads);
        assert_eq!(opt.timestep(), 2);

        opt.reset();
        assert_eq!(opt.timestep(), 0);
        assert!(opt.state.is_empty());

        // After reset, next step should behave as t=1 again
        let (graph2, id2) = single_param_graph(vec![1.0]);
        let grads2 = HashMap::from([(id2, vec![0.1])]);
        opt.step(&graph2, &grads2);

        let expected = 1.0 - 0.1 * 0.1 / (0.1 + 1e-8);
        assert_approx_eq(&[read_param(&graph2)[0]], &[expected]);
    }

    #[test]
    fn adam_custom_betas_and_eps() {
        let (graph, id) = single_param_graph(vec![1.0]);
        let grads = HashMap::from([(id, vec![0.5])]);

        let mut opt = Adam::new(0.1).betas(0.8, 0.95).eps(1e-6);
        opt.step(&graph, &grads);

        // t=1, b1=0.8, b2=0.95, eps=1e-6, g=0.5
        // m = 0.2 * 0.5 = 0.1, v = 0.05 * 0.25 = 0.0125
        // bc1 = 1-0.8 = 0.2, bc2 = 1-0.95 = 0.05
        // m̂ = 0.1/0.2 = 0.5, v̂ = 0.0125/0.05 = 0.25
        // w = 1.0 - 0.1 * 0.5 / (0.5 + 1e-6)
        let expected = 1.0 - 0.1 * 0.5 / (0.5 + 1e-6);
        assert_approx_eq(&[read_param(&graph)[0]], &[expected]);
    }

    #[test]
    fn adam_multiple_steps_accumulate_momentum() {
        // After 2 steps with the same gradient, the second update should be
        // larger than the first (momentum accumulation), even though the
        // bias correction partially offsets it.
        let g = 1.0f32;

        // Single step
        let (g1, id1) = single_param_graph(vec![0.0]);
        let grads = HashMap::from([(id1, vec![g])]);
        let mut opt1 = Adam::new(0.1);
        opt1.step(&g1, &grads);
        let update1 = read_param(&g1)[0].abs();

        // Two steps (fresh param)
        let (g2, id2) = single_param_graph(vec![0.0]);
        let grads = HashMap::from([(id2, vec![g])]);
        let mut opt2 = Adam::new(0.1);
        opt2.step(&g2, &grads);
        opt2.step(&g2, &grads);
        let update2 = read_param(&g2)[0].abs();

        // With constant gradient and these defaults, the second step's
        // per-step update magnitude should be larger than the first.
        assert!(
            update2 > update1,
            "second step ({update2}) should be larger than first ({update1})"
        );
    }
}
