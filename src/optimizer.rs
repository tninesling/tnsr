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
