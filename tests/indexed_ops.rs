use std::collections::HashMap;

use tnsr::graph::{TensorGraph, TensorGraphNode};
use tnsr::tensor::{ExprKind, TensorExpr};
use tnsr::{Executor, SimpleExecutor};

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "element {index}: expected {expected}, got {actual}"
        );
    }
}

fn parameter_id(expression: &TensorExpr<f32>) -> usize {
    match expression.kind() {
        ExprKind::Parameter { id, .. } => *id,
        _ => panic!("expected parameter expression"),
    }
}

#[test]
fn embedding_gathers_rows_for_arbitrary_index_shape() {
    let weight = TensorExpr::constant(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![4, 2]);
    let indices = TensorExpr::constant(vec![2.0, 0.0, 3.0, 1.0], vec![2, 2]);
    let embedding = weight.embedding(indices);
    assert_eq!(embedding.shape(), &vec![2, 2, 2]);

    let graph: TensorGraph<f32> = embedding.into();
    let result = SimpleExecutor::default()
        .execute(&graph, HashMap::new())
        .unwrap();
    assert_close(&result, &[5.0, 6.0, 1.0, 2.0, 7.0, 8.0, 3.0, 4.0]);
}

#[test]
fn embedding_gradient_scatter_adds_repeated_ids() {
    let weight = TensorExpr::parameter(vec![0.0; 6], vec![3, 2]);
    let weight_id = parameter_id(&weight);
    let indices = TensorExpr::constant(vec![1.0, 1.0, 2.0], vec![3]);
    let loss = weight.embedding(indices).mean_all();
    let graph: TensorGraph<f32> = loss.into();
    let loss_node = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(loss_node);

    let mut executor = SimpleExecutor::default();
    executor.execute(&graph, HashMap::new()).unwrap();
    let gradients = executor.get_gradients(&graph);
    assert_close(
        &gradients[&weight_id],
        &[0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
    );
}

#[test]
fn indexed_cross_entropy_is_stable_and_has_closed_form_gradient() {
    let logits_data = vec![1000.0, 1001.0, 999.0, -1000.0, -999.0, -1001.0];
    let targets = TensorExpr::constant(vec![1.0, 0.0], vec![2]);
    let losses = TensorExpr::constant(logits_data.clone(), vec![2, 3])
        .indexed_cross_entropy(targets.clone());
    let graph: TensorGraph<f32> = losses.into();
    let result = SimpleExecutor::default()
        .execute(&graph, HashMap::new())
        .unwrap();
    let log_partition = (1.0f32 + (-1.0f32).exp() + (-2.0f32).exp()).ln();
    assert_close(&result, &[log_partition, 1.0 + log_partition]);

    let logits = TensorExpr::parameter(logits_data, vec![2, 3]);
    let logits_id = parameter_id(&logits);
    let loss = logits.indexed_cross_entropy(targets).mean_all();
    let graph: TensorGraph<f32> = loss.into();
    let loss_node = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(loss_node);
    let mut executor = SimpleExecutor::default();
    executor.execute(&graph, HashMap::new()).unwrap();
    let gradients = executor.get_gradients(&graph);

    let denominator = 1.0 + (-1.0f32).exp() + (-2.0f32).exp();
    let high = 1.0 / denominator;
    let middle = (-1.0f32).exp() / denominator;
    let low = (-2.0f32).exp() / denominator;
    assert_close(
        &gradients[&logits_id],
        &[
            middle / 2.0,
            (high - 1.0) / 2.0,
            low / 2.0,
            (middle - 1.0) / 2.0,
            high / 2.0,
            low / 2.0,
        ],
    );
}

#[test]
fn indexed_operations_reject_invalid_indices() {
    let invalid = [
        (f32::NAN, "finite"),
        (f32::INFINITY, "finite"),
        (-1.0, "non-negative"),
        (1.5, "integer"),
        (f32::MAX, "usize"),
        (3.0, "out of range"),
    ];

    for (index, expected_message) in invalid {
        let expression = TensorExpr::constant(vec![1.0; 6], vec![3, 2])
            .embedding(TensorExpr::constant(vec![index], vec![1]));
        let graph: TensorGraph<f32> = expression.into();
        let error = SimpleExecutor::default()
            .execute(&graph, HashMap::new())
            .unwrap_err();
        assert!(
            error.to_string().contains(expected_message),
            "expected {expected_message:?} in {error:#}"
        );
    }

    let expression = TensorExpr::constant(vec![1.0, 2.0, 3.0], vec![1, 3])
        .indexed_cross_entropy(TensorExpr::constant(vec![-1.0], vec![1]));
    let graph: TensorGraph<f32> = expression.into();
    assert!(
        SimpleExecutor::default()
            .execute(&graph, HashMap::new())
            .unwrap_err()
            .to_string()
            .contains("non-negative")
    );
}

#[test]
fn optimization_bypasses_and_lowering_preserves_indexed_operations() {
    let embedding = TensorExpr::constant(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
        .embedding(TensorExpr::constant(vec![1.0], vec![1]));
    let expression = embedding + TensorExpr::constant(vec![0.0, 0.0], vec![1, 2]);
    let optimized = expression.optimize();
    assert!(matches!(optimized.kind(), ExprKind::Binary { .. }));

    let mut graph = TensorGraph::new();
    graph.add_expr(&expression, true);
    assert!(
        graph
            .graph
            .node_weights()
            .any(|node| matches!(node, TensorGraphNode::Embedding { .. }))
    );
    let result = SimpleExecutor::default()
        .execute(&graph, HashMap::new())
        .unwrap();
    assert_close(&result, &[3.0, 4.0]);
}

#[test]
fn lowering_memoizes_shared_expression_nodes() {
    let input = TensorExpr::<f32>::input("x", vec![2]);
    let shared = -input;
    let expression = shared.clone() + shared;
    let graph: TensorGraph<f32> = expression.into();

    assert_eq!(graph.graph.node_count(), 3);
    let unary_count = graph
        .graph
        .node_weights()
        .filter(|node| matches!(node, TensorGraphNode::Unary { .. }))
        .count();
    assert_eq!(unary_count, 1);
}
