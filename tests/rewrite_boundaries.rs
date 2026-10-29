//! Supported parents may optimize while opaque children keep their identities.
use std::collections::HashMap;
use tnsr::graph::TensorGraph;
use tnsr::graph::rewrite::{RewriteConfig, RewriteNumerics, optimize_expr_with_numerics};
use tnsr::tensor::{ExprKind, IndexTensor, TensorExpr, ValueDType};
use tnsr::{Executor, SimpleExecutor};

fn retains_identity(expr: &TensorExpr<f32>, expected: &TensorExpr<f32>) -> bool {
    std::ptr::eq(expr.kind(), expected.kind())
        || expr
            .children()
            .iter()
            .any(|child| retains_identity(child, expected))
}

fn count(expr: &TensorExpr<f32>, predicate: fn(&ExprKind<f32>) -> bool) -> usize {
    usize::from(predicate(expr.kind()))
        + expr
            .children()
            .iter()
            .map(|child| count(child, predicate))
            .sum::<usize>()
}

fn verify(expr: &TensorExpr<f32>, expected_leaves: &[&TensorExpr<f32>]) -> TensorExpr<f32> {
    let expected = SimpleExecutor::new()
        .execute(&TensorGraph::from(expr.clone()), HashMap::new())
        .unwrap();
    let mut conservative = None;
    for numerics in [
        RewriteNumerics::PreserveOperations,
        RewriteNumerics::RealAlgebra,
    ] {
        let optimized = optimize_expr_with_numerics(expr, RewriteConfig::default(), numerics);
        assert_eq!(optimized.shape(), expr.shape());
        for leaf in expected_leaves {
            assert!(
                retains_identity(&optimized, leaf),
                "opaque identity was replaced"
            );
        }
        let actual = SimpleExecutor::new()
            .execute(&TensorGraph::from(optimized.clone()), HashMap::new())
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(&expected) {
            if b.is_nan() {
                assert!(a.is_nan());
            } else {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
        if numerics == RewriteNumerics::PreserveOperations {
            conservative = Some(optimized);
        }
    }
    conservative.unwrap()
}

#[test]
fn cast_boundaries_survive_parent_layout_elimination() {
    let input = TensorExpr::constant(
        vec![1.003_906_3f32, 1.011_718_8, -1.003_906_3, -0.0, 2.0, 3.0],
        vec![2, 3],
    );
    let narrowed = input.clone().cast(ValueDType::BF16).cast(ValueDType::F32);
    let promoted = input.cast(ValueDType::F32);
    let expr = (narrowed.clone() + promoted.clone())
        .transpose()
        .transpose();
    let optimized = verify(&expr, &[&narrowed, &promoted]);
    assert_eq!(
        count(&optimized, |node| matches!(node, ExprKind::Cast { .. })),
        3
    );
    assert_eq!(
        count(&optimized, |node| matches!(
            node,
            ExprKind::Transpose { .. }
        )),
        0
    );
}

#[test]
fn distinct_identical_cast_nodes_do_not_merge() {
    let input = TensorExpr::constant(vec![1.001f32; 6], vec![2, 3]);
    let a = input.clone().cast(ValueDType::BF16);
    let b = input.cast(ValueDType::BF16);
    assert!(!std::ptr::eq(a.kind(), b.kind()));
    let optimized = verify(&(a.clone() + b.clone()).transpose().transpose(), &[&a, &b]);
    assert_eq!(
        count(&optimized, |node| matches!(node, ExprKind::Cast { .. })),
        2
    );
}

#[test]
fn distinct_max_reduction_axes_survive_parent_layout_elimination() {
    let a = TensorExpr::constant(vec![-1.0f32, 2.0, 3.0, -4.0], vec![2, 2]).reduce_max(0);
    let b = TensorExpr::constant(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2]).reduce_max(1);
    let optimized = verify(&(a.clone() + b.clone()).transpose().transpose(), &[&a, &b]);
    assert_eq!(
        count(&optimized, |node| matches!(
            node,
            ExprKind::Transpose { .. }
        )),
        0
    );
    assert_eq!(
        count(&optimized, |node| matches!(
            node,
            ExprKind::ReduceAxis { .. }
        )),
        2
    );
}

#[test]
fn exact_index_sources_stay_inside_distinct_embedding_leaves() {
    let table = TensorExpr::constant(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let a = table
        .clone()
        .embedding_indices(IndexTensor::constant(vec![0, 1], vec![2]).unwrap());
    let b = table.embedding_indices(IndexTensor::constant(vec![2, 0], vec![2]).unwrap());
    let a_indices = a.children()[1].clone();
    let b_indices = b.children()[1].clone();
    let optimized = verify(
        &(a.clone() + b.clone()).transpose().transpose(),
        &[&a, &b, &a_indices, &b_indices],
    );
    assert_eq!(
        count(&optimized, |node| matches!(node, ExprKind::Index { .. })),
        2
    );
    assert_eq!(
        count(&optimized, |node| matches!(
            node,
            ExprKind::Transpose { .. }
        )),
        0
    );
}

#[test]
fn convolution_leaves_keep_distinct_weights_and_inputs() {
    let input = TensorExpr::constant((1_u8..=9).map(f32::from).collect(), vec![1, 1, 3, 3]);
    let a = input.clone().conv2d(
        TensorExpr::constant(vec![1.0f32, -1.0, 2.0, 0.0], vec![1, 1, 2, 2]),
        1,
        0,
    );
    let b = input.conv2d(
        TensorExpr::constant(vec![0.0f32, 1.0, -2.0, 3.0], vec![1, 1, 2, 2]),
        1,
        0,
    );
    let optimized = verify(&(a.clone() + b.clone()).transpose().transpose(), &[&a, &b]);
    assert_eq!(
        count(&optimized, |node| matches!(node, ExprKind::Conv2d { .. })),
        2
    );
    assert_eq!(
        count(&optimized, |node| matches!(
            node,
            ExprKind::Transpose { .. }
        )),
        0
    );
}

#[test]
fn pooling_flatten_narrow_and_concat_leaves_keep_their_metadata() {
    let input = TensorExpr::constant((1_u8..=9).map(f32::from).collect(), vec![1, 1, 3, 3]);
    let pooled = input.max_pool2d(2, 1);
    let flattened = pooled.clone().flatten();
    let narrow_a = flattened.clone().narrow(1, 0, 2);
    let narrow_b = flattened.clone().narrow(1, 1, 2);
    let concat = TensorExpr::concat(&[narrow_a.clone(), narrow_b.clone()], 1);
    let expr = concat.clone().transpose().transpose();
    let optimized = verify(&expr, &[&pooled, &flattened, &narrow_a, &narrow_b, &concat]);
    assert!(std::ptr::eq(optimized.kind(), concat.kind()));
}

#[test]
fn wide_opaque_input_can_have_supported_reduction_parents() {
    let wide = usize::try_from(i32::MAX).unwrap() + 1;
    let input = TensorExpr::<f32>::input("wide", vec![wide]);
    let reduced = input.clone().reduce_sum(0);
    let expr = -(-reduced);
    let optimized = expr.optimize();
    assert_eq!(optimized.shape(), &[1]);
    assert!(
        matches!(optimized.kind(), ExprKind::ReduceAxis { x, .. } if std::ptr::eq(x.kind(), input.kind()))
    );
}
