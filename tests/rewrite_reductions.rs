use std::collections::HashMap;
use tnsr::graph::TensorGraph;
use tnsr::graph::rewrite::{RewriteConfig, RewriteNumerics, optimize_expr_with_numerics};
use tnsr::tensor::{ExprKind, ReduceOp, TensorExpr};
use tnsr::{Executor, SimpleExecutor};

fn values(shape: &[usize], seed: usize) -> TensorExpr<f32> {
    TensorExpr::constant(
        (0..shape.iter().product())
            .map(|i| ((i + seed) % 19) as f32 / 19.0 - 0.5)
            .collect(),
        shape.to_vec(),
    )
}

fn execute(expr: TensorExpr<f32>) -> Vec<f32> {
    SimpleExecutor::new()
        .execute(&TensorGraph::from(expr), HashMap::new())
        .unwrap()
}

fn optimized(expr: &TensorExpr<f32>, numerics: RewriteNumerics) -> TensorExpr<f32> {
    let result = optimize_expr_with_numerics(expr, RewriteConfig::default(), numerics);
    assert_eq!(result.shape(), expr.shape());
    let expected = execute(expr.clone());
    let actual = execute(result.clone());
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        if b.is_nan() {
            assert!(a.is_nan());
        } else if numerics == RewriteNumerics::PreserveOperations {
            assert_eq!(a.to_bits(), b.to_bits());
        } else {
            assert!((a - b).abs() < 2e-4 * (1.0 + b.abs()), "{a} vs {b}");
        }
    }
    result
}

#[test]
fn mean_of_nonempty_broadcast_is_opt_in_and_eliminates_materialization() {
    let expr = values(&[1, 17], 0).broadcast_axis(0, 64).reduce_mean(0);
    assert!(matches!(
        optimized(&expr, RewriteNumerics::PreserveOperations).kind(),
        ExprKind::ReduceAxis { .. }
    ));
    assert!(matches!(
        optimized(&expr, RewriteNumerics::RealAlgebra).kind(),
        ExprKind::Constant { .. }
    ));
}

#[test]
fn empty_broadcast_mean_is_not_cancelled() {
    let expr = values(&[1, 3], 0).broadcast_axis(0, 0).reduce_mean(0);
    let result = optimize_expr_with_numerics(
        &expr,
        RewriteConfig::default(),
        RewriteNumerics::RealAlgebra,
    );
    assert!(matches!(
        result.kind(),
        ExprKind::ReduceAxis {
            op: ReduceOp::Mean,
            ..
        }
    ));
    assert_eq!(result.shape(), &[1, 3]);
}

#[test]
fn sum_and_mean_move_before_rank_two_matmul_in_both_dimensions() {
    for axis in [0, 1] {
        for mean in [false, true] {
            let product = values(&[32, 8], 0).matmul(values(&[8, 24], 3));
            let expr = if mean {
                product.reduce_mean(axis)
            } else {
                product.reduce_sum(axis)
            };
            assert!(matches!(
                optimized(&expr, RewriteNumerics::PreserveOperations).kind(),
                ExprKind::ReduceAxis { .. }
            ));
            let result = optimized(&expr, RewriteNumerics::RealAlgebra);
            let ExprKind::MatMul { a, b } = result.kind() else {
                panic!("expected reduction pushed through matmul");
            };
            assert_eq!(a.shape()[0], if axis == 0 { 1 } else { 32 });
            assert_eq!(b.shape()[1], if axis == 1 { 1 } else { 24 });
        }
    }
}

#[test]
fn batched_matmul_reduction_is_not_moved() {
    let expr = values(&[2, 8, 4], 0)
        .matmul(values(&[2, 4, 6], 2))
        .reduce_sum(1);
    assert!(matches!(
        optimized(&expr, RewriteNumerics::RealAlgebra).kind(),
        ExprKind::ReduceAxis { .. }
    ));
}

#[test]
fn multiplicative_reduction_factors_must_be_axis_invariant() {
    for mean in [false, true] {
        let product = values(&[64, 16], 0) * values(&[1, 16], 2);
        let expr = if mean {
            product.reduce_mean(0)
        } else {
            product.reduce_sum(0)
        };
        let result = optimized(&expr, RewriteNumerics::RealAlgebra);
        assert!(matches!(result.kind(), ExprKind::Binary { .. }));
        assert!(matches!(
            optimized(&expr, RewriteNumerics::PreserveOperations).kind(),
            ExprKind::ReduceAxis { .. }
        ));
    }
    let expr = (values(&[64, 16], 0) * values(&[64, 16], 2)).reduce_sum(0);
    assert!(matches!(
        optimized(&expr, RewriteNumerics::RealAlgebra).kind(),
        ExprKind::ReduceAxis { .. }
    ));
}

#[test]
fn broadcasting_unaries_preserve_special_values_and_signed_zero() {
    let x = TensorExpr::constant(
        vec![
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            0.0,
            -2.0,
            3.0,
        ],
        vec![1, 7],
    );
    let expanded = x.broadcast_axis(0, 16);
    for expr in [
        -expanded.clone(),
        expanded.clone().exp(),
        expanded.clone().log(),
        expanded.relu(),
    ] {
        let result = optimized(&expr, RewriteNumerics::PreserveOperations);
        assert!(matches!(result.kind(), ExprKind::BroadcastAxis { .. }));
    }
}

#[test]
fn reduction_pushdown_preserves_parameter_gradients() {
    use tnsr::tensor::Parameter;
    let a = Parameter::new((0..48).map(|i| (i % 7) as f32 / 7.0).collect(), vec![8, 6]);
    let b = Parameter::new(
        (0..42).map(|i| (i % 11) as f32 / 11.0).collect(),
        vec![6, 7],
    );
    let expr = TensorExpr::from(a.clone()).matmul(b.clone()).reduce_sum(0);
    let result = optimized(&expr, RewriteNumerics::RealAlgebra);
    let gradients = |expr: TensorExpr<f32>| {
        let graph = TensorGraph::from((expr.clone() * expr).reduce_sum(1).reduce_sum(0));
        let root = *graph.toposort().last().unwrap();
        let graph = graph.with_gradients(root);
        let mut executor = SimpleExecutor::new();
        executor.execute(&graph, HashMap::new()).unwrap();
        executor.get_gradients(&graph)
    };
    let actual = gradients(result);
    let expected = gradients(expr);
    for id in [a.id(), b.id()] {
        assert_eq!(actual[&id].len(), expected[&id].len());
        for (a, b) in actual[&id].iter().zip(&expected[&id]) {
            assert!((a - b).abs() < 1e-3 * (1.0 + b.abs()), "{a} vs {b}");
        }
    }
}
