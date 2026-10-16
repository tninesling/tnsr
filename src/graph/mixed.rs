//! Storage semantics for explicitly mixed f32 graphs.
//!
//! Sources are f32 masters. Casts introduce BF16 storage; views preserve it,
//! and arithmetic promotes to f32 if any floating operand is f32. Every BF16
//! node is a rounding boundary, including gradient accumulation nodes.
use super::{NodeIndex, TensorGraph, TensorGraphNode};
use crate::tensor::ValueDType;
use std::collections::HashMap;

/// Empty for homogeneous graphs, preserving their existing execution rules.
pub fn value_dtypes<D, G>(graph: &TensorGraph<D, G>) -> HashMap<NodeIndex, ValueDType> {
    if !graph
        .graph
        .node_weights()
        .any(|n| matches!(n, TensorGraphNode::Cast { .. }))
    {
        return HashMap::new();
    }
    let mut dtypes = HashMap::new();
    for node in graph.toposort() {
        if matches!(graph[node], TensorGraphNode::IndexSource { .. }) {
            continue;
        }
        let dtype = match &graph[node] {
            TensorGraphNode::Cast { dtype, .. } => *dtype,
            TensorGraphNode::Constant { .. }
            | TensorGraphNode::Input { .. }
            | TensorGraphNode::Parameter { .. } => ValueDType::F32,
            _ => {
                let inputs = graph.inputs(node);
                if inputs
                    .iter()
                    .filter_map(|n| dtypes.get(n))
                    .all(|d| *d == ValueDType::BF16)
                {
                    ValueDType::BF16
                } else {
                    ValueDType::F32
                }
            }
        };
        dtypes.insert(node, dtype);
    }
    dtypes
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{
        Executor, SimpleExecutor,
        tensor::{Parameter, TensorExpr},
    };

    pub(crate) fn fanout(shared: bool) -> (TensorGraph<f32, super::super::WithGrad>, usize) {
        let master = Parameter::new(vec![1.0078125_f32], vec![1]);
        let id = master.id();
        let x: TensorExpr<f32> = master.into();
        let a = x.clone().cast(ValueDType::BF16);
        let b = if shared {
            a.clone()
        } else {
            x.cast(ValueDType::BF16)
        };
        let a = a.cast(ValueDType::F32) * TensorExpr::constant(vec![1. + 1. / 256.], vec![1]);
        let b = b.cast(ValueDType::F32) * TensorExpr::constant(vec![1. / 256.], vec![1]);
        let mut graph = TensorGraph::new();
        let loss = graph.add_expr(&(a + b), true);
        (graph.with_gradients(loss), id)
    }

    #[test]
    fn cast_autograd_preserves_shared_and_distinct_rounding_boundaries() {
        for shared in [true, false] {
            let (graph, id) = fanout(shared);
            let mut executor = SimpleExecutor::new();
            executor.execute(&graph, HashMap::new()).unwrap();
            // 1.00390625 narrows to 1; the BF16 sum of 1 and 1/256
            // ties back to 1. Distinct casts sum the promoted values in f32.
            assert_eq!(
                executor.get_gradients(&graph)[&id],
                vec![if shared { 1. } else { 1. + 1. / 256. }]
            );
            let dtypes = value_dtypes(&graph);
            assert_eq!(
                dtypes[&graph.gradient_metadata().param_to_grad[&id]],
                ValueDType::F32
            );
            let forward = graph.clone().without_gradients();
            assert!(!value_dtypes(&forward).is_empty());
        }
    }

    #[test]
    fn casts_survive_rewrites_and_repeated_narrowing() {
        let x = TensorExpr::input("x", vec![3])
            .cast(ValueDType::BF16)
            .cast(ValueDType::F32);
        let mut graph = TensorGraph::new();
        graph.add_expr(&x, true);
        assert_eq!(
            graph
                .graph
                .node_weights()
                .filter(|n| matches!(n, TensorGraphNode::Cast { .. }))
                .count(),
            2
        );
        let mut executor = SimpleExecutor::new();
        let values = vec![1. + 1. / 256., -1. - 1. / 256., 1e-43];
        assert_eq!(
            executor
                .execute(&graph, HashMap::from([("x".into(), values)]))
                .unwrap(),
            vec![1., -1., 0.]
        );
    }

    #[test]
    fn bf16_scalar_loss_seeds_bf16_backward_operands() {
        let x: TensorExpr<f32> = Parameter::new(vec![2_f32], vec![1, 1]).into();
        let y = x
            .cast(ValueDType::BF16)
            .matmul(TensorExpr::input("w", vec![1, 1]).cast(ValueDType::BF16));
        let mut graph = TensorGraph::new();
        let loss = graph.add_expr(&y, false);
        let graph = graph.with_gradients(loss);
        let dtypes = value_dtypes(&graph);
        for node in graph
            .graph
            .node_indices()
            .filter(|n| matches!(graph[*n], TensorGraphNode::MatMul { .. }))
        {
            assert_eq!(dtypes[&node], ValueDType::BF16);
        }
    }
}
