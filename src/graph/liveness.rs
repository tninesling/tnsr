//! Liveness analysis for graph execution.
//!
//! Determines when each node's value can be released during a single
//! execution pass, which is the basis for the memory allocator described in
//! `docs/memory-allocator-plan.md` (#37).
//!
//! Execution proceeds in topological order. A node's *last use* is the
//! topo-order position of its last consumer; once execution passes that
//! position, the node's value is dead and its buffer may be reused. Nodes in
//! the pin set are never released during execution.
//!
//! The pin set is every sink (node with no outgoing edges). Sinks exactly
//! cover the execution output and, for training graphs, every gradient
//! accumulation node read by [`Executor::get_gradients`], so no inspection of
//! the graph's typestate is required. Dead-branch ends are pinned too;
//! skipping dead nodes entirely is a separate follow-up.
//!
//! [`Executor::get_gradients`]: crate::Executor::get_gradients

use std::collections::{HashMap, HashSet};

use petgraph::Direction;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;

use super::TensorGraph;

/// Liveness analysis result for a single execution of a graph.
///
/// Computed fresh on every execution: graph rewrites and fusion change the
/// node set, so results must never be cached across executions.
#[derive(Debug, Clone)]
pub struct Liveness {
    /// Topo-order position of each node's last consumer (inclusive).
    last_use: HashMap<NodeIndex, usize>,
    /// Nodes releasable after each topo-order position, indexed by position.
    free_at: Vec<Vec<NodeIndex>>,
    /// Nodes that must remain available after execution completes.
    pinned: HashSet<NodeIndex>,
}

impl Liveness {
    /// Returns the topo-order position of the node's last consumer.
    ///
    /// Nodes with no consumers report their own position.
    pub fn last_use(&self, idx: NodeIndex) -> Option<usize> {
        self.last_use.get(&idx).copied()
    }

    /// Returns `true` if the node must remain available after execution.
    pub fn is_pinned(&self, idx: NodeIndex) -> bool {
        self.pinned.contains(&idx)
    }

    /// Returns the pinned node set.
    pub fn pinned(&self) -> &HashSet<NodeIndex> {
        &self.pinned
    }

    /// Returns the nodes whose values may be released after executing the
    /// node at topo-order position `pos`.
    pub fn free_after(&self, pos: usize) -> &[NodeIndex] {
        self.free_at.get(pos).map_or(&[], Vec::as_slice)
    }
}

/// Analyze a graph for a single execution in the given topological order.
///
/// Pins every sink node: the execution output and, for training graphs, all
/// gradient accumulation nodes.
pub fn analyze<D, G>(graph: &TensorGraph<D, G>, order: &[NodeIndex]) -> Liveness {
    let pinned = order
        .iter()
        .filter(|&&idx| {
            graph
                .graph
                .edges_directed(idx, Direction::Outgoing)
                .next()
                .is_none()
        })
        .copied()
        .collect();
    analyze_with_pins(graph, order, pinned)
}

fn analyze_with_pins<D, G>(
    graph: &TensorGraph<D, G>,
    order: &[NodeIndex],
    pinned: HashSet<NodeIndex>,
) -> Liveness {
    let position: HashMap<NodeIndex, usize> = order
        .iter()
        .enumerate()
        .map(|(pos, &idx)| (idx, pos))
        .collect();

    // A node's last use defaults to its own position (covers dead branches
    // and the output) and extends to the latest consumer position.
    let mut last_use: HashMap<NodeIndex, usize> = position.clone();
    for (pos, &idx) in order.iter().enumerate() {
        for edge in graph.graph.edges_directed(idx, Direction::Outgoing) {
            if let Some(&consumer_pos) = position.get(&edge.target()) {
                let entry = last_use.entry(idx).or_insert(pos);
                *entry = (*entry).max(consumer_pos);
            }
        }
    }

    let mut free_at = vec![Vec::new(); order.len()];
    for (&idx, &pos) in &last_use {
        if !pinned.contains(&idx) {
            free_at[pos].push(idx);
        }
    }

    Liveness {
        last_use,
        free_at,
        pinned,
    }
}

#[cfg(test)]
mod tests {
    use super::{Liveness, analyze};
    use crate::graph::{TensorGraph, TensorGraphNode};
    use crate::tensor::{Parameter, TensorExpr, UnaryOp};
    use petgraph::graph::NodeIndex;
    use std::collections::HashMap;

    fn positions(order: &[NodeIndex]) -> HashMap<NodeIndex, usize> {
        order
            .iter()
            .enumerate()
            .map(|(pos, &idx)| (idx, pos))
            .collect()
    }

    fn find_node<G>(
        graph: &TensorGraph<f32, G>,
        order: &[NodeIndex],
        pred: impl Fn(&TensorGraphNode<f32>) -> bool,
    ) -> NodeIndex {
        order
            .iter()
            .find(|&&idx| pred(&graph.graph[idx]))
            .copied()
            .expect("node not found in graph")
    }

    fn assert_pinned_never_freed(liveness: &Liveness, order_len: usize) {
        for &pin in liveness.pinned() {
            for pos in 0..order_len {
                assert!(
                    !liveness.free_after(pos).contains(&pin),
                    "pinned node {pin:?} must never be freed"
                );
            }
        }
    }

    #[test]
    fn chain_releases_each_node_after_its_consumer() {
        // x -> neg -> neg -> neg (output)
        let expr = -(-(-TensorExpr::<f32>::input("x", vec![4])));
        let graph: TensorGraph<f32> = expr.into();
        let order = graph.toposort();
        assert_eq!(order.len(), 4);

        let liveness = analyze(&graph, &order);

        let output = *order.last().unwrap();
        assert!(liveness.is_pinned(output));
        assert_eq!(liveness.last_use(output), Some(order.len() - 1));

        // Every other node dies right after its single consumer executes.
        for (pos, &idx) in order.iter().enumerate().take(order.len() - 1) {
            assert_eq!(liveness.last_use(idx), Some(pos + 1));
            assert!(liveness.free_after(pos + 1).contains(&idx));
        }
        assert_pinned_never_freed(&liveness, order.len());
    }

    #[test]
    fn shared_input_dies_at_its_last_consumer() {
        // x -> neg --+
        // x ---------+-> add (output)
        let x = TensorExpr::<f32>::input("x", vec![4]);
        let expr = -x.clone() + x;
        let graph: TensorGraph<f32> = expr.into();
        let order = graph.toposort();
        assert_eq!(order.len(), 3);

        let pos = positions(&order);
        let liveness = analyze(&graph, &order);

        let add = *order.last().unwrap();
        let input = find_node(&graph, &order, |n| {
            matches!(n, TensorGraphNode::Input { .. })
        });
        let neg = find_node(&graph, &order, |n| {
            matches!(n, TensorGraphNode::Unary { .. })
        });

        // The input feeds both neg and add, so it lives until the add.
        assert_eq!(liveness.last_use(input), Some(pos[&add]));
        assert_eq!(liveness.last_use(neg), Some(pos[&add]));
        assert!(liveness.is_pinned(add));
        assert_pinned_never_freed(&liveness, order.len());
    }

    #[test]
    fn diamond_uses_latest_consumer_regardless_of_topo_order() {
        // x -> neg --+
        // x -> exp --+-> add (output)
        let x = TensorExpr::<f32>::input("x", vec![4]);
        let expr = -x.clone() + x.exp();
        let graph: TensorGraph<f32> = expr.into();
        let order = graph.toposort();
        assert_eq!(order.len(), 4);

        let pos = positions(&order);
        let liveness = analyze(&graph, &order);

        let add = *order.last().unwrap();
        let input = find_node(&graph, &order, |n| {
            matches!(n, TensorGraphNode::Input { .. })
        });
        let neg = find_node(&graph, &order, |n| {
            matches!(
                n,
                TensorGraphNode::Unary {
                    op: UnaryOp::Neg,
                    ..
                }
            )
        });
        let exp = find_node(&graph, &order, |n| {
            matches!(
                n,
                TensorGraphNode::Unary {
                    op: UnaryOp::Exp,
                    ..
                }
            )
        });

        // neg and exp may appear in either order; the input dies at whichever
        // comes last, and both branches die at the add.
        assert_eq!(liveness.last_use(input), Some(pos[&neg].max(pos[&exp])));
        assert_eq!(liveness.last_use(neg), Some(pos[&add]));
        assert_eq!(liveness.last_use(exp), Some(pos[&add]));
        assert_pinned_never_freed(&liveness, order.len());
    }

    #[test]
    fn training_graph_pins_gradients_and_extends_activations() {
        // loss = exp(w * w); the backward pass for exp references the forward
        // exp output, so the activation must live into the backward sweep.
        let w = Parameter::new(vec![2.0f32], vec![1]);
        let w_id = w.id();
        let sq = TensorExpr::from(w.clone()) * TensorExpr::from(w);
        let loss = sq.exp();

        let graph: TensorGraph<f32> = loss.into();
        let loss_node = *graph.toposort().last().unwrap();
        let graph = graph.with_gradients(loss_node);

        let order = graph.toposort();
        let pos = positions(&order);
        let liveness = analyze(&graph, &order);

        // The gradient accumulation node for w is pinned.
        let grad_node = graph.gradient_metadata().param_to_grad[&w_id];
        assert!(liveness.is_pinned(grad_node));
        assert!(liveness.is_pinned(*order.last().unwrap()));

        // The forward exp node is consumed by a backward node, so its last
        // use is strictly after its own position.
        let is_forward_exp = |idx: NodeIndex| {
            matches!(
                graph.graph[idx],
                TensorGraphNode::Unary {
                    op: UnaryOp::Exp,
                    ..
                }
            ) && !graph.gradient_metadata().gradient_nodes.contains(&idx)
        };
        let exp_node = find_node(&graph, &order, |n| {
            matches!(
                n,
                TensorGraphNode::Unary {
                    op: UnaryOp::Exp,
                    ..
                }
            )
        });
        assert!(is_forward_exp(exp_node));
        assert!(
            liveness.last_use(exp_node).unwrap() > pos[&exp_node],
            "forward activation must live into the backward pass"
        );

        assert_pinned_never_freed(&liveness, order.len());
    }
}
