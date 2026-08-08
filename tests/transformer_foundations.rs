use std::collections::HashMap;

use tnsr::graph::TensorGraph;
use tnsr::tensor::TensorExpr;
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

#[test]
fn reshape_and_permute_preserve_row_major_values() {
    let expression =
        TensorExpr::constant((0..12).map(|value| value as f32).collect(), vec![2, 2, 3])
            .reshape(vec![2, 3, 2])
            .permute(vec![0, 2, 1]);
    assert_eq!(expression.shape(), &vec![2, 2, 3]);

    let graph: TensorGraph<f32> = expression.into();
    let mut executor = SimpleExecutor::default();
    let result = executor.execute(&graph, HashMap::new()).unwrap();
    assert_close(
        &result,
        &[0.0, 2.0, 4.0, 1.0, 3.0, 5.0, 6.0, 8.0, 10.0, 7.0, 9.0, 11.0],
    );
}

#[test]
fn batched_matmul_broadcasts_batch_prefixes() {
    let a = TensorExpr::constant(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![2, 2, 3],
    );
    let b = TensorExpr::constant(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![1, 3, 2]);
    let expression = a.matmul(b);
    assert_eq!(expression.shape(), &vec![2, 2, 2]);

    let graph: TensorGraph<f32> = expression.into();
    let mut executor = SimpleExecutor::default();
    let result = executor.execute(&graph, HashMap::new()).unwrap();
    assert_close(&result, &[4.0, 5.0, 10.0, 11.0, 16.0, 17.0, 22.0, 23.0]);
}

#[test]
fn batched_matmul_preserves_zero_batch_dimensions() {
    let a = TensorExpr::constant(vec![], vec![0, 2, 3]);
    let b = TensorExpr::constant(vec![1.0; 12], vec![1, 3, 4]);
    let expression = a.matmul(b);
    assert_eq!(expression.shape(), &vec![0, 2, 4]);

    let graph: TensorGraph<f32> = expression.into();
    let mut executor = SimpleExecutor::default();
    assert!(executor.execute(&graph, HashMap::new()).unwrap().is_empty());
}

#[test]
fn batched_matmul_reduces_broadcast_parameter_gradient() {
    let a = TensorExpr::parameter(vec![1.0, 2.0, 3.0, 4.0], vec![2, 1, 2]);
    let a_id = match a.kind() {
        tnsr::tensor::ExprKind::Parameter { id, .. } => *id,
        _ => unreachable!(),
    };
    let b = TensorExpr::parameter(vec![5.0, 6.0], vec![1, 2, 1]);
    let b_id = match b.kind() {
        tnsr::tensor::ExprKind::Parameter { id, .. } => *id,
        _ => unreachable!(),
    };
    let loss = a.matmul(b).mean_all();
    let graph: TensorGraph<f32> = loss.into();
    let loss_node = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(loss_node);

    let mut executor = SimpleExecutor::default();
    executor.execute(&graph, HashMap::new()).unwrap();
    let gradients = executor.get_gradients(&graph);
    assert_close(&gradients[&a_id], &[2.5, 3.0, 2.5, 3.0]);
    assert_close(&gradients[&b_id], &[2.0, 3.0]);
}

#[test]
fn reshape_and_permute_autograd_restore_input_layout() {
    let values = TensorExpr::parameter(vec![1.0; 6], vec![2, 3]);
    let values_id = match values.kind() {
        tnsr::tensor::ExprKind::Parameter { id, .. } => *id,
        _ => unreachable!(),
    };
    let loss = values
        .reshape(vec![1, 2, 3])
        .permute(vec![2, 0, 1])
        .mean_all();
    let graph: TensorGraph<f32> = loss.into();
    let loss_node = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(loss_node);

    let mut executor = SimpleExecutor::default();
    executor.execute(&graph, HashMap::new()).unwrap();
    let gradients = executor.get_gradients(&graph);
    assert_close(&gradients[&values_id], &[1.0 / 6.0; 6]);
}

#[test]
fn implicit_binary_broadcast_has_reduced_gradients() {
    let values = TensorExpr::parameter(vec![1.0; 6], vec![2, 3]);
    let values_id = match values.kind() {
        tnsr::tensor::ExprKind::Parameter { id, .. } => *id,
        _ => unreachable!(),
    };
    let bias = TensorExpr::parameter(vec![2.0; 3], vec![3]);
    let bias_id = match bias.kind() {
        tnsr::tensor::ExprKind::Parameter { id, .. } => *id,
        _ => unreachable!(),
    };
    let loss = (values + bias).mean_all();
    let graph: TensorGraph<f32> = loss.into();
    let loss_node = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(loss_node);

    let mut executor = SimpleExecutor::default();
    executor.execute(&graph, HashMap::new()).unwrap();
    let gradients = executor.get_gradients(&graph);
    assert_close(
        gradients
            .get(&values_id)
            .unwrap_or_else(|| panic!("missing values gradient; keys: {:?}", gradients.keys())),
        &[1.0 / 6.0; 6],
    );
    assert_close(
        gradients
            .get(&bias_id)
            .unwrap_or_else(|| panic!("missing bias gradient; keys: {:?}", gradients.keys())),
        &[1.0 / 3.0; 3],
    );
}
