use anyhow::{Result, ensure};
use petgraph::visit::EdgeRef;

use super::{TensorGraph, TensorGraphNode, WithGrad};
use crate::tensor::TensorExpr;

impl TensorGraph<f32, WithGrad> {
    /// Reassociate long rank-two gradient contractions into four K partitions.
    ///
    /// This opt-in transformation preserves output/gradient node identities but
    /// changes floating-point summation order. Each partition is an ordinary
    /// batched matmul followed by an ordinary sum, so all backends retain their
    /// existing allocation, indexing, reduction and replay behavior.
    pub fn split_long_matmul_gradients(&mut self) -> Result<usize> {
        let candidates: Vec<_> = self
            .graph
            .node_indices()
            .filter(|node| {
                self.gradients.gradient_nodes.contains(node)
                    && matches!(self.graph[*node], TensorGraphNode::MatMul { .. })
            })
            .collect();
        let mut transformed = 0;
        for node in candidates {
            let inputs = self.inputs(node);
            ensure!(inputs.len() == 2, "gradient matmul must have two inputs");
            let a_shape = self.graph[inputs[0]].shape().clone();
            let b_shape = self.graph[inputs[1]].shape().clone();
            if a_shape.len() != 2 || b_shape.len() != 2 {
                continue;
            }
            let (m, k, n) = (a_shape[0], a_shape[1], b_shape[1]);
            if m < 128 || n < 128 || k < 2048 || k / m.max(n) < 4 || !k.is_multiple_of(4) {
                continue;
            }
            ensure!(
                b_shape[0] == k,
                "gradient matmul contraction dimensions disagree"
            );
            let a = TensorExpr::node_ref(inputs[0], a_shape)
                .reshape(vec![m, 4, k / 4])
                .permute(vec![1, 0, 2]);
            let b = TensorExpr::node_ref(inputs[1], b_shape).reshape(vec![4, k / 4, n]);
            let expression = a.matmul(b).reduce_sum(0).reshape(vec![m, n]);
            let first_new = self.graph.node_count();
            let replacement = self.add_expr(&expression, false);
            self.gradients.gradient_nodes.extend(
                self.graph
                    .node_indices()
                    .filter(|node| node.index() >= first_new),
            );
            let mut edges: Vec<_> = self
                .graph
                .edges_directed(node, petgraph::Direction::Incoming)
                .map(|edge| edge.id())
                .collect();
            edges.sort_by_key(|edge| std::cmp::Reverse(edge.index()));
            for edge in edges {
                self.graph.remove_edge(edge);
            }
            self.graph[node] = TensorGraphNode::Reshape { shape: vec![m, n] };
            self.graph.add_edge(replacement, node, 0);
            transformed += 1;
        }
        Ok(transformed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Parameter;
    use crate::{Executor, SimpleExecutor};
    use std::collections::HashMap;

    #[test]
    fn split_gradient_preserves_roots_forward_nodes_and_changed_feeds() {
        let (rows, width) = (2048, 128);
        let weight = Parameter::new(vec![0.125f32; width * width], vec![width, width]);
        let id = weight.id();
        let expression = TensorExpr::input("x", vec![rows, width])
            .matmul(TensorExpr::from(weight))
            .mean_all();
        let forward: TensorGraph<f32> = expression.into();
        let root = *forward.toposort().last().unwrap();
        let original = forward.with_gradients(root);
        let mut split = original.clone();
        let old_metadata = original.gradient_metadata();
        assert_eq!(split.split_long_matmul_gradients().unwrap(), 1);
        assert_eq!(
            split.gradient_metadata().param_to_grad,
            old_metadata.param_to_grad
        );
        assert_eq!(split.split_long_matmul_gradients().unwrap(), 0);
        for node in original
            .graph
            .node_indices()
            .filter(|node| !old_metadata.gradient_nodes.contains(node))
        {
            assert_eq!(original.graph[node].name(), split.graph[node].name());
            assert_eq!(original.inputs(node), split.inputs(node));
        }
        for sign in [1.0, -1.0] {
            let feeds = HashMap::from([(
                "x".to_owned(),
                (0..rows * width)
                    .map(|i| sign * ((i % 29) as f32 / 29.0 - 0.5))
                    .collect(),
            )]);
            let mut baseline = SimpleExecutor::new();
            let mut candidate = SimpleExecutor::new();
            baseline.execute(&original, feeds.clone()).unwrap();
            candidate.execute(&split, feeds).unwrap();
            let expected = baseline.get_gradients(&original);
            let actual = candidate.get_gradients(&split);
            assert_eq!(actual.len(), expected.len());
            for (a, b) in actual[&id].iter().zip(&expected[&id]) {
                assert!((a - b).abs() < 1e-7, "{a} vs {b}");
            }
        }
    }
}
