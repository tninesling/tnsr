use std::collections::{HashMap, HashSet};

use tensor::graph::TensorGraph;
use tensor::graph::TensorGraphNode;

pub struct SGD {
    lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }

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
