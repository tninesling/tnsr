//! Operator fusion analysis and transformation.
//!
//! This module provides analysis and transformation passes to fuse consecutive
//! element-wise operations into single kernels, reducing kernel launch overhead
//! and memory bandwidth requirements.
//!
//! # Fusion Strategy
//!
//! Currently supports fusing consecutive unary operations that:
//! - Operate on the same shape
//! - Have a single consumer (no branching in the middle of the chain)
//! - Are all element-wise operations (Neg, Exp, Log, Relu)
//!
//! # Example
//!
//! ```ignore
//! // Before fusion:
//! x -> Exp -> t1 -> Log -> t2 -> Relu -> y
//! // Requires 3 kernel launches and 2 intermediate memory reads/writes
//!
//! // After fusion:
//! x -> FusedUnary(Exp, Log, Relu) -> y
//! // Single kernel launch, no intermediate memory traffic
//! ```

use std::collections::HashSet;

use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;

use crate::UnaryOp;
use crate::graph::TensorGraph;
use crate::graph::TensorGraphNode;

/// Represents a chain of consecutive unary operations that can be fused.
#[derive(Debug, Clone)]
pub struct UnaryChain {
    /// The operations in the chain, in execution order.
    pub ops: Vec<UnaryOp>,
    /// The starting node of the chain (input).
    pub start_node: NodeIndex,
    /// The nodes in the chain (excluding start, including end).
    pub chain_nodes: Vec<NodeIndex>,
    /// The ending node of the chain (output).
    pub end_node: NodeIndex,
}

impl UnaryChain {
    /// Returns the length of the chain (number of operations).
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Returns true if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// Analyzes a TensorGraph to identify fusible operation chains.
pub struct FusionAnalyzer<'a, D, G> {
    graph: &'a TensorGraph<D, G>,
}

impl<'a, D, G> FusionAnalyzer<'a, D, G> {
    /// Create a new fusion analyzer for the given graph.
    pub fn new(graph: &'a TensorGraph<D, G>) -> Self {
        Self { graph }
    }

    /// Identify all fusible unary operation chains in the graph.
    ///
    /// Returns a vector of chains, sorted by topological order.
    /// Each chain contains at least 2 operations to be worth fusing.
    pub fn find_fusible_chains(&self) -> Vec<UnaryChain> {
        let mut chains = Vec::new();
        let mut visited = HashSet::new();

        // Process nodes in topological order
        for node_idx in self.graph.toposort() {
            // Skip if already part of a chain
            if visited.contains(&node_idx) {
                continue;
            }

            // Try to start a chain from this node
            if let Some(chain) = self.try_build_chain(node_idx, &visited) {
                // Only keep chains with 2+ operations
                if chain.len() >= 2 {
                    for &chain_node in &chain.chain_nodes {
                        visited.insert(chain_node);
                    }
                    chains.push(chain);
                }
            }
        }

        chains
    }

    /// Attempt to build a unary chain starting from the given node.
    fn try_build_chain(
        &self,
        start_idx: NodeIndex,
        visited: &HashSet<NodeIndex>,
    ) -> Option<UnaryChain> {
        let mut ops = Vec::new();
        let mut chain_nodes = Vec::new();
        let mut current = start_idx;

        // Check if this node is a unary operation
        let first_op = match &self.graph[current] {
            TensorGraphNode::Unary { op } => *op,
            _ => return None,
        };

        // Skip if already visited
        if visited.contains(&current) {
            return None;
        }

        // Get the input to this operation
        let inputs = self.graph.inputs(current);
        if inputs.len() != 1 {
            return None;
        }
        let start_node = inputs[0];

        // Check if start_node has exactly one consumer (the current node)
        // This prevents fusing operations whose input has multiple consumers
        let start_consumers: Vec<_> = self
            .graph
            .graph
            .edges_directed(start_node, petgraph::Direction::Outgoing)
            .map(|e| e.target())
            .collect();

        if start_consumers.len() != 1 || start_consumers[0] != current {
            return None;
        }

        // Build the chain by following single consumers
        ops.push(first_op);
        chain_nodes.push(current);

        loop {
            // Check if current node has exactly one consumer
            let consumers: Vec<_> = self
                .graph
                .graph
                .edges_directed(current, petgraph::Direction::Outgoing)
                .map(|e| e.target())
                .collect();

            if consumers.len() != 1 {
                break;
            }

            let next = consumers[0];

            // Check if next node is a unary operation
            let next_op = match &self.graph[next] {
                TensorGraphNode::Unary { op } => *op,
                _ => break,
            };

            // Check if next node is already visited
            if visited.contains(&next) {
                break;
            }

            // Check if shapes match (required for element-wise fusion)
            let current_shape = self.graph.shapes.get(&current);
            let next_shape = self.graph.shapes.get(&next);
            if current_shape != next_shape {
                break;
            }

            // Check if next node has exactly one input (the current node)
            let next_inputs = self.graph.inputs(next);
            if next_inputs.len() != 1 || next_inputs[0] != current {
                break;
            }

            // Add to chain
            ops.push(next_op);
            chain_nodes.push(next);
            current = next;
        }

        let end_node = current;

        Some(UnaryChain {
            ops,
            start_node,
            chain_nodes,
            end_node,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorExpr;

    #[test]
    fn test_find_simple_chain() {
        // Create graph: x -> exp -> log
        let x = TensorExpr::<f32>::input("x", vec![4]);
        let y = x.exp().log();
        let graph: TensorGraph<f32> = y.into();

        let analyzer = FusionAnalyzer::new(&graph);
        let chains = analyzer.find_fusible_chains();

        assert_eq!(chains.len(), 1, "Should find one chain");
        assert_eq!(chains[0].len(), 2, "Chain should have 2 ops");
        assert_eq!(chains[0].ops[0], UnaryOp::Exp);
        assert_eq!(chains[0].ops[1], UnaryOp::Log);
    }

    #[test]
    fn test_find_longer_chain() {
        // Create graph: x -> neg -> exp -> log -> relu
        let x = TensorExpr::<f32>::input("x", vec![4]);
        let y = (-x).exp().log().relu();
        let graph: TensorGraph<f32> = y.into();

        let analyzer = FusionAnalyzer::new(&graph);
        let chains = analyzer.find_fusible_chains();

        assert_eq!(chains.len(), 1, "Should find one chain");
        assert_eq!(chains[0].len(), 4, "Chain should have 4 ops");
        assert_eq!(chains[0].ops[0], UnaryOp::Neg);
        assert_eq!(chains[0].ops[1], UnaryOp::Exp);
        assert_eq!(chains[0].ops[2], UnaryOp::Log);
        assert_eq!(chains[0].ops[3], UnaryOp::Relu);
    }

    #[test]
    fn test_no_chain_for_single_op() {
        // Create graph: x -> exp
        let x = TensorExpr::<f32>::input("x", vec![4]);
        let y = x.exp();
        let graph: TensorGraph<f32> = y.into();

        let analyzer = FusionAnalyzer::new(&graph);
        let chains = analyzer.find_fusible_chains();

        assert_eq!(chains.len(), 0, "Should find no chains (single op)");
    }

    #[test]
    fn test_no_chain_with_branching() {
        // Create graph with branching: x -> exp -> (log, relu)
        // The exp node has two consumers, so it should not be part of any chain
        let x = TensorExpr::<f32>::input("x", vec![4]);
        let exp_x = x.exp();
        let log_exp = exp_x.clone().log();
        let relu_exp = exp_x.relu();
        let y = log_exp + relu_exp;
        let graph: TensorGraph<f32> = y.into();

        let analyzer = FusionAnalyzer::new(&graph);
        let chains = analyzer.find_fusible_chains();

        // Since exp has 2 consumers, it cannot be fused with either log or relu
        // So we should find no chains (both log and relu are single ops after exp)
        assert_eq!(
            chains.len(),
            0,
            "Should find no chains (exp has multiple consumers, log and relu are single ops)"
        );
    }

    #[test]
    fn test_multiple_independent_chains() {
        // Create two independent chains: x -> exp -> log, y -> neg -> relu
        let x = TensorExpr::<f32>::input("x", vec![4]);
        let y = TensorExpr::<f32>::input("y", vec![4]);
        let a = x.exp().log();
        let b = (-y).relu();
        let result = a + b;
        let graph: TensorGraph<f32> = result.into();

        let analyzer = FusionAnalyzer::new(&graph);
        let chains = analyzer.find_fusible_chains();

        assert_eq!(chains.len(), 2, "Should find two independent chains");
        assert!(
            chains
                .iter()
                .any(|c| c.len() == 2 && c.ops[0] == UnaryOp::Exp)
        );
        assert!(
            chains
                .iter()
                .any(|c| c.len() == 2 && c.ops[0] == UnaryOp::Neg)
        );
    }

    #[test]
    fn test_chain_stops_at_binary_op() {
        // Create graph: x -> exp -> log -> (add with y)
        let x = TensorExpr::<f32>::input("x", vec![4]);
        let y = TensorExpr::<f32>::input("y", vec![4]);
        let a = x.exp().log();
        let result = a + y;
        let graph: TensorGraph<f32> = result.into();

        let analyzer = FusionAnalyzer::new(&graph);
        let chains = analyzer.find_fusible_chains();

        assert_eq!(chains.len(), 1, "Should find one chain");
        assert_eq!(chains[0].len(), 2, "Chain should stop at binary op");
        assert_eq!(chains[0].ops[0], UnaryOp::Exp);
        assert_eq!(chains[0].ops[1], UnaryOp::Log);
    }

    #[test]
    fn test_fusion_preserves_edges() {
        use crate::Constant;

        let x = Constant::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let node = TensorExpr::from(x).relu().log().exp();

        let mut graph: TensorGraph<f32> = node.into();

        eprintln!("Before fusion:");
        eprintln!("  Nodes: {}", graph.graph.node_count());
        for idx in graph.graph.node_indices() {
            let node = &graph.graph[idx];
            let inputs: Vec<_> = graph.inputs(idx).into_iter().map(|n| n.index()).collect();
            eprintln!(
                "  Node {}: {:?}, inputs: {:?}",
                idx.index(),
                node.name(),
                inputs
            );
        }

        // Debug: check what chains are found
        let analyzer = FusionAnalyzer::new(&graph);
        let chains = analyzer.find_fusible_chains();
        for chain in &chains {
            eprintln!(
                "Chain: start={}, end={}, nodes={:?}, ops={:?}",
                chain.start_node.index(),
                chain.end_node.index(),
                chain
                    .chain_nodes
                    .iter()
                    .map(|n| n.index())
                    .collect::<Vec<_>>(),
                chain.ops
            );
        }

        let num_fused = graph.apply_fusion();
        assert_eq!(num_fused, 1, "Should have fused one chain");

        eprintln!("\nAfter fusion:");
        eprintln!("  Nodes: {}", graph.graph.node_count());
        for idx in graph.graph.node_indices() {
            let node = &graph.graph[idx];
            let inputs: Vec<_> = graph.inputs(idx).into_iter().map(|n| n.index()).collect();
            eprintln!(
                "  Node {}: {:?}, inputs: {:?}",
                idx.index(),
                node.name(),
                inputs
            );
        }

        // Find the FusedUnary node
        let fused_node = graph
            .graph
            .node_indices()
            .find(|&idx| matches!(graph.graph[idx], TensorGraphNode::FusedUnary { .. }))
            .expect("Should have a FusedUnary node");

        let fused_inputs = graph.inputs(fused_node);
        assert_eq!(
            fused_inputs.len(),
            1,
            "FusedUnary should have exactly one input"
        );
    }
}
