//! Optimization algorithms for updating model parameters.
//!
//! Provides gradient descent optimizers that update trainable parameters
//! based on computed gradients from backward passes.

use std::collections::{HashMap, HashSet};

use tensor::graph::TensorGraph;
use tensor::graph::TensorGraphNode;

/// Stochastic Gradient Descent (SGD) optimizer.
///
/// Updates parameters using the gradient descent rule: `w = w - lr * grad`
/// where `lr` is the learning rate.
///
/// # Example
///
/// ```rust
/// use runtime::optimizer::SGD;
/// use std::collections::HashMap;
///
/// let optimizer = SGD::new(0.01); // learning rate = 0.01
/// // After computing gradients via backward pass:
/// // optimizer.step(&graph, &grads_by_param);
/// ```
pub struct SGD {
    lr: f32,
}

impl SGD {
    /// Create a new SGD optimizer with the specified learning rate.
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate (step size) for parameter updates
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }

    /// Perform a single optimization step, updating all parameters in the graph.
    ///
    /// Applies the update rule `w = w - lr * grad` to each parameter that has
    /// a corresponding gradient in `grads`.
    ///
    /// # Arguments
    ///
    /// * `graph` - The computation graph containing parameters to update
    /// * `grads` - Gradients indexed by parameter ID (from [`BackwardResult::grads_by_param`](crate::BackwardResult::grads_by_param))
    ///
    /// # Panics
    ///
    /// Panics if gradient size does not match parameter size.
    pub fn step(&self, graph: &TensorGraph<f32>, grads: &HashMap<usize, Vec<f32>>) {
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
                    for (wi, &gi) in w.iter_mut().zip(grad.iter()) {
                        *wi -= self.lr * gi;
                    }
                    updated.insert(*id);
                }
            }
        }
    }
}
