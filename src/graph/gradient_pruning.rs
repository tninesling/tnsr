//! Retain the original forward graph and only the generated derivatives needed
//! by parameter-gradient roots. Frozen input/constant gradients are not outputs.
use super::{NodeIndex, TensorGraphNode};
use petgraph::{Direction, Graph};
use std::collections::{HashMap, HashSet};

pub(super) fn retain_parameter_gradients<D>(
    graph: Graph<TensorGraphNode<D>, usize>,
    forward_nodes: usize,
    parameter_gradients: &mut HashMap<usize, NodeIndex>,
    gradient_nodes: &mut HashSet<NodeIndex>,
) -> Graph<TensorGraphNode<D>, usize> {
    let mut needed = HashSet::new();
    let mut stack: Vec<_> = parameter_gradients.values().copied().collect();
    while let Some(node) = stack.pop() {
        if needed.insert(node) {
            stack.extend(graph.neighbors_directed(node, Direction::Incoming));
        }
    }
    let keep: Vec<_> = graph
        .node_indices()
        .map(|node| node.index() < forward_nodes || needed.contains(&node))
        .collect();
    let mut remap = vec![NodeIndex::end(); graph.node_count()];
    let mut next = 0;
    for (old, &retained) in keep.iter().enumerate() {
        if retained {
            remap[old] = NodeIndex::new(next);
            next += 1;
        }
    }
    // filter_map_owned visits nodes in index order. Keeping the entire original
    // prefix preserves every public forward NodeIndex, including the loss and
    // multiple forward outputs. Every gradient root is in its own closure.
    for node in parameter_gradients.values_mut() {
        *node = remap[node.index()];
    }
    *gradient_nodes = gradient_nodes
        .iter()
        .filter(|node| keep[node.index()])
        .map(|node| remap[node.index()])
        .collect();
    graph.filter_map_owned(
        |node, value| keep[node.index()].then_some(value),
        |_, edge| Some(edge),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TensorGraph;
    use crate::tensor::{Input, Parameter, TensorExpr};
    use crate::{Executor, SimpleExecutor};
    use petgraph::visit::EdgeRef;

    #[test]
    fn frozen_matmul_inputs_and_constants_do_not_generate_executable_derivatives() {
        let weight = Parameter::new(vec![0.5f32; 12], vec![3, 4]);
        let unused = Parameter::new(vec![9.0f32], vec![1]);
        let x: TensorExpr<f32> = Input::new("x", vec![2, 3]).into();
        let loss = (x.matmul(weight.clone()) + 1.0).reduce_sum(1).reduce_sum(0);
        let mut forward: TensorGraph<f32> = loss.into();
        let output = *forward.toposort().last().unwrap();
        let detached = TensorExpr::from(unused.clone()).lower_to_graph(&mut forward);
        let prefix: Vec<_> = forward
            .graph
            .node_indices()
            .map(|node| {
                (
                    node,
                    std::mem::discriminant(&forward[node]),
                    forward[node].shape().clone(),
                )
            })
            .collect();
        let original_edges: HashSet<_> = forward
            .graph
            .edge_references()
            .map(|edge| (edge.source(), edge.target(), *edge.weight()))
            .collect();
        let count = forward.graph.node_count();
        let graph = forward.with_gradients(output);
        for (node, kind, shape) in prefix {
            assert_eq!(std::mem::discriminant(&graph[node]), kind);
            assert_eq!(graph[node].shape(), &shape);
        }
        assert!(
            matches!(graph[detached], TensorGraphNode::Parameter { id, .. } if id == unused.id())
        );
        assert_eq!(graph.gradient_metadata().param_to_grad.len(), 1);
        assert!(
            !graph
                .gradient_metadata()
                .param_to_grad
                .contains_key(&unused.id())
        );
        // Forward GEMM and weight-gradient GEMM remain; frozen-input GEMM and
        // broadcast reductions for the frozen scalar never reach execution.
        assert_eq!(
            graph
                .graph
                .node_weights()
                .filter(|node| matches!(node, TensorGraphNode::MatMul { .. }))
                .count(),
            2
        );
        assert_eq!(
            graph
                .graph
                .node_weights()
                .filter(|node| matches!(node, TensorGraphNode::ReduceAxis { .. }))
                .count(),
            2
        );
        let inputs = HashMap::from([("x".into(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
        let mut cpu = SimpleExecutor::new();
        cpu.execute(&graph, inputs.clone()).unwrap();
        assert_eq!(cpu.get_value(output).unwrap(), vec![50.0]);
        assert_eq!(
            cpu.get_gradients(&graph)[&weight.id()],
            vec![5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0, 9.0, 9.0, 9.0, 9.0]
        );
        let restored = graph.without_gradients();
        assert_eq!(restored.graph.node_count(), count);
        let restored_edges: HashSet<_> = restored
            .graph
            .edge_references()
            .map(|edge| (edge.source(), edge.target(), *edge.weight()))
            .collect();
        assert_eq!(restored_edges, original_edges);
        cpu.execute(&restored, inputs).unwrap();
        assert_eq!(cpu.get_value(output).unwrap(), vec![50.0]);
    }

    #[test]
    fn trainable_inputs_and_shared_producers_retain_both_parameter_derivatives() {
        let input = Parameter::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let weight = Parameter::new(vec![0.5f32; 12], vec![3, 4]);
        let y = TensorExpr::from(input.clone()).matmul(weight.clone());
        let graph: TensorGraph<f32> = (y.clone() + y).reduce_sum(1).reduce_sum(0).into();
        let output = *graph.toposort().last().unwrap();
        let graph = graph.with_gradients(output);
        let mut cpu = SimpleExecutor::new();
        cpu.execute(&graph, HashMap::new()).unwrap();
        let gradients = cpu.get_gradients(&graph);
        assert_eq!(gradients[&input.id()], vec![4.0; 6]);
        assert_eq!(
            gradients[&weight.id()],
            vec![
                10.0, 10.0, 10.0, 10.0, 14.0, 14.0, 14.0, 14.0, 18.0, 18.0, 18.0, 18.0
            ]
        );
        assert_eq!(
            graph
                .graph
                .node_weights()
                .filter(|node| matches!(node, TensorGraphNode::MatMul { .. }))
                .count(),
            3
        );
    }

    #[test]
    fn shared_seed_gradient_still_gets_a_dedicated_parameter_sink() {
        let p = Parameter::new(vec![2.0f32], vec![1]);
        let q = Parameter::new(vec![3.0f32], vec![1]);
        let q_value = TensorExpr::from(q.clone());
        let graph: TensorGraph<f32> =
            (TensorExpr::from(p.clone()) + q_value.clone() * q_value).into();
        let output = *graph.toposort().last().unwrap();
        let graph = graph.with_gradients(output);
        let mut cpu = SimpleExecutor::new();
        cpu.execute(&graph, HashMap::new()).unwrap();
        let gradients = cpu.get_gradients(&graph);
        assert_eq!(gradients[&p.id()], vec![1.0]);
        assert_eq!(gradients[&q.id()], vec![6.0]);
        for &node in graph.gradient_metadata().param_to_grad.values() {
            assert!(
                graph
                    .graph
                    .neighbors_directed(node, Direction::Outgoing)
                    .next()
                    .is_none()
            );
        }
    }

    #[test]
    fn graph_without_trainable_parameters_keeps_only_its_forward_computation() {
        let x: TensorExpr<f32> = Input::new("x", vec![2]).into();
        let graph: TensorGraph<f32> = (x + 3.0).reduce_sum(0).into();
        let count = graph.graph.node_count();
        let output = *graph.toposort().last().unwrap();
        let graph = graph.with_gradients(output);
        assert!(graph.gradient_metadata().param_to_grad.is_empty());
        assert!(graph.gradient_metadata().gradient_nodes.is_empty());
        assert_eq!(graph.graph.node_count(), count);
        let mut cpu = SimpleExecutor::new();
        cpu.execute(&graph, HashMap::from([("x".into(), vec![1.0, 2.0])]))
            .unwrap();
        assert_eq!(cpu.get_value(output).unwrap(), vec![9.0]);
    }
}
