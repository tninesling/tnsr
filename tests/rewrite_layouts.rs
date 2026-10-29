use std::collections::HashMap;
use tnsr::graph::TensorGraph;
use tnsr::graph::rewrite::{RewriteConfig, RewriteNumerics, optimize_expr_with_numerics};
use tnsr::tensor::{ExprKind, TensorExpr};
use tnsr::{Executor, SimpleExecutor};

fn tensor(shape: &[usize]) -> TensorExpr<f32> {
    let size = shape.iter().product();
    TensorExpr::constant(
        (0..size).map(|i| (i % 13) as f32 - 6.0).collect(),
        shape.to_vec(),
    )
}

fn layouts(expr: &TensorExpr<f32>) -> usize {
    usize::from(matches!(
        expr.kind(),
        ExprKind::Transpose { .. } | ExprKind::Permute { .. }
    )) + expr
        .children()
        .iter()
        .map(|child| layouts(child))
        .sum::<usize>()
}

fn verify(expr: &TensorExpr<f32>) -> TensorExpr<f32> {
    let optimized = optimize_expr_with_numerics(
        expr,
        RewriteConfig::default(),
        RewriteNumerics::PreserveOperations,
    );
    assert_eq!(optimized.shape(), expr.shape());
    let expected = SimpleExecutor::new()
        .execute(&TensorGraph::from(expr.clone()), HashMap::new())
        .unwrap();
    let actual = SimpleExecutor::new()
        .execute(&TensorGraph::from(optimized.clone()), HashMap::new())
        .unwrap();
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        if b.is_nan() {
            assert!(a.is_nan());
        } else {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
    optimized
}

#[test]
fn mixed_transpose_and_permutation_round_trips_disappear() {
    let x = tensor(&[2, 3, 4]);
    for expr in [
        x.clone().transpose().permute(vec![0, 2, 1]),
        x.permute(vec![0, 2, 1]).transpose(),
    ] {
        assert!(matches!(verify(&expr).kind(), ExprKind::Constant { .. }));
    }
}

#[test]
fn mixed_layout_composition_preserves_nonidentity_mapping() {
    let expr = tensor(&[2, 3, 4]).transpose().permute(vec![2, 0, 1]);
    assert_eq!(layouts(&verify(&expr)), 1);
}

#[test]
fn moving_singleton_axes_uses_a_reshape() {
    let expr = tensor(&[2, 1, 3, 1, 4]).permute(vec![1, 0, 3, 2, 4]);
    let optimized = verify(&expr);
    assert_eq!(layouts(&optimized), 0);
    assert!(matches!(optimized.kind(), ExprKind::Reshape { .. }));
    let expr = tensor(&[1, 24]).transpose();
    assert!(matches!(verify(&expr).kind(), ExprKind::Reshape { .. }));
    let expr = tensor(&[1, 1, 1]).permute(vec![2, 0, 1]);
    assert!(matches!(verify(&expr).kind(), ExprKind::Constant { .. }));
}

#[test]
fn equal_sized_nonsingleton_axes_still_reorder_values() {
    let expr = tensor(&[2, 2]).transpose();
    assert_eq!(layouts(&verify(&expr)), 1);
}

#[test]
fn inverse_layouts_through_unary_preserve_special_values() {
    let x = TensorExpr::constant(
        vec![-0.0, 0.0, 1.0, -2.0, f32::INFINITY, f32::NAN],
        vec![2, 3],
    );
    let inner = x.transpose();
    for expr in [
        (-inner.clone()).transpose(),
        inner.clone().exp().transpose(),
        inner.clone().log().transpose(),
        inner.relu().transpose(),
    ] {
        let optimized = verify(&expr);
        assert_eq!(layouts(&optimized), 0);
        assert!(matches!(optimized.kind(), ExprKind::Unary { .. }));
    }
}

#[test]
fn inverse_permutations_through_unary_remove_both_layouts() {
    let expr = tensor(&[2, 3, 4])
        .permute(vec![1, 2, 0])
        .relu()
        .permute(vec![2, 0, 1]);
    assert_eq!(layouts(&verify(&expr)), 0);
}

#[test]
fn reshape_round_trip_through_unary_disappears() {
    let expr = tensor(&[4, 6])
        .reshape(vec![2, 12])
        .relu()
        .reshape(vec![4, 6]);
    let optimized = verify(&expr);
    assert!(
        matches!(optimized.kind(), ExprKind::Unary { x, .. } if matches!(x.kind(), ExprKind::Constant { .. }))
    );
}

#[test]
fn noninverse_layouts_do_not_cancel_through_unary() {
    let expr = tensor(&[2, 3, 4])
        .permute(vec![1, 2, 0])
        .relu()
        .permute(vec![1, 0, 2]);
    assert_eq!(layouts(&verify(&expr)), 2);
}
