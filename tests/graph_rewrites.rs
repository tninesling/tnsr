use std::collections::HashMap;
use tnsr::graph::TensorGraph;
use tnsr::graph::rewrite::{RewriteConfig, RewriteNumerics, optimize_expr_with_numerics};
use tnsr::tensor::{BinaryOp, ExprKind, TensorExpr};
use tnsr::{Executor, SimpleExecutor};

fn tensor(shape: &[usize], seed: usize) -> TensorExpr<f32> {
    TensorExpr::constant(
        (0..shape.iter().product())
            .map(|i| ((i + seed) % 13) as f32 / 13.0 - 0.5)
            .collect(),
        shape.to_vec(),
    )
}

fn count(expr: &TensorExpr<f32>, predicate: fn(&ExprKind<f32>) -> bool) -> usize {
    usize::from(predicate(expr.kind()))
        + expr
            .children()
            .iter()
            .map(|x| count(x, predicate))
            .sum::<usize>()
}

fn matmuls(expr: &TensorExpr<f32>) -> usize {
    count(expr, |kind| matches!(kind, ExprKind::MatMul { .. }))
}

fn optimize(expr: &TensorExpr<f32>, policy: RewriteNumerics) -> TensorExpr<f32> {
    let optimized = optimize_expr_with_numerics(expr, RewriteConfig::default(), policy);
    assert_eq!(optimized.shape(), expr.shape());
    compare(&mut SimpleExecutor::new(), expr, &optimized, policy);
    optimized
}

fn compare<E: Executor<f32>>(
    executor: &mut E,
    expr: &TensorExpr<f32>,
    optimized: &TensorExpr<f32>,
    policy: RewriteNumerics,
) {
    let expected = executor
        .execute(&TensorGraph::from(expr.clone()), HashMap::new())
        .unwrap();
    let actual = executor
        .execute(&TensorGraph::from(optimized.clone()), HashMap::new())
        .unwrap();
    assert_eq!(expected.len(), actual.len());
    for (a, b) in actual.iter().zip(expected) {
        if b.is_nan() {
            assert!(a.is_nan());
        } else if policy == RewriteNumerics::PreserveOperations {
            assert_eq!(a.to_bits(), b.to_bits());
        } else {
            assert!((a - b).abs() < 2e-4 * (1.0 + b.abs()), "{a} vs {b}");
        }
    }
}

#[test]
fn layout_composition_preserves_nonidentity_mapping() {
    let expr = tensor(&[2, 3, 4], 0)
        .permute(vec![1, 2, 0])
        .permute(vec![1, 0, 2]);
    let optimized = optimize(&expr, RewriteNumerics::PreserveOperations);
    assert_eq!(
        count(&optimized, |k| matches!(k, ExprKind::Permute { .. })),
        1
    );
    let expr = tensor(&[2, 3, 4], 0)
        .reshape(vec![6, 4])
        .reshape(vec![4, 6]);
    let optimized = optimize(&expr, RewriteNumerics::PreserveOperations);
    assert_eq!(
        count(&optimized, |k| matches!(k, ExprKind::Reshape { .. })),
        1
    );
}

#[test]
fn inverse_layouts_disappear_for_both_policies() {
    for policy in [
        RewriteNumerics::PreserveOperations,
        RewriteNumerics::RealAlgebra,
    ] {
        let x = tensor(&[2, 3, 4], 0);
        for expr in [
            x.clone().permute(vec![1, 2, 0]).permute(vec![2, 0, 1]),
            x.clone().reshape(vec![6, 4]).reshape(vec![2, 3, 4]),
            x.permute(vec![0, 1, 2]),
        ] {
            assert!(matches!(
                optimize(&expr, policy).kind(),
                ExprKind::Constant { .. }
            ));
        }
    }
}

#[test]
fn transpose_pointwise_control_preserves_broadcasting_and_special_values() {
    let a = TensorExpr::constant(
        vec![f32::NAN, -0.0, 0.0, f32::INFINITY, -3.0, 2.0],
        vec![2, 3],
    );
    let b = tensor(&[1, 2, 1], 3);
    let a = a.transpose();
    let b = b.transpose();
    for expr in [
        a.clone() + b.clone(),
        a.clone() - b.clone(),
        a.clone() * b.clone(),
        a / b,
    ] {
        let optimized = optimize(&expr, RewriteNumerics::PreserveOperations);
        // Singleton-axis transposes may now become reshapes, but pointwise
        // work must not be sunk into an outer transpose (the CUDA regression).
        assert!(matches!(optimized.kind(), ExprKind::Binary { .. }));
        assert!(count(&optimized, |k| matches!(k, ExprKind::Transpose { .. })) <= 2);
    }
}

#[test]
fn factoring_shared_left_and_right_requires_opt_in() {
    let a = tensor(&[16, 8], 0);
    let b = tensor(&[8, 16], 1);
    let c = tensor(&[8, 16], 2);
    for expr in [
        a.clone().matmul(b.clone()) + a.clone().matmul(c.clone()),
        b.matmul(a.clone()) + c.matmul(a),
    ] {
        assert_eq!(
            matmuls(&optimize(&expr, RewriteNumerics::PreserveOperations)),
            2
        );
        assert_eq!(matmuls(&optimize(&expr, RewriteNumerics::RealAlgebra)), 1);
    }
}

#[test]
fn extraction_rejects_expensive_factoring() {
    let a = tensor(&[1, 256], 0);
    let expr = a.clone().matmul(tensor(&[256, 1], 1)) + a.matmul(tensor(&[256, 1], 2));
    assert_eq!(matmuls(&optimize(&expr, RewriteNumerics::RealAlgebra)), 2);
}

#[test]
fn matrix_chain_selects_lower_work_in_either_direction() {
    for (m, k, n, p, right) in [(32, 2, 32, 2, true), (2, 32, 2, 32, false)] {
        let a = tensor(&[m, k], 0);
        let b = tensor(&[k, n], 1);
        let c = tensor(&[n, p], 2);
        let expr = if right {
            a.matmul(b).matmul(c)
        } else {
            a.matmul(b.matmul(c))
        };
        let optimized = optimize(&expr, RewriteNumerics::RealAlgebra);
        let ExprKind::MatMul { a, b } = optimized.kind() else {
            panic!("expected matmul");
        };
        assert_eq!(matches!(b.kind(), ExprKind::MatMul { .. }), right);
        assert_eq!(matches!(a.kind(), ExprKind::MatMul { .. }), !right);
    }
}

#[test]
fn factoring_does_not_move_broadcast_adds_through_contractions() {
    let a = tensor(&[4, 8], 0);
    let expr = a.clone().matmul(tensor(&[8, 4], 1)) + a.matmul(tensor(&[8, 1], 2));
    assert_eq!(matmuls(&optimize(&expr, RewriteNumerics::RealAlgebra)), 2);
}

#[test]
fn default_retains_arithmetic_order_in_matrix_chains() {
    let expr = tensor(&[32, 2], 0)
        .matmul(tensor(&[2, 32], 1))
        .matmul(tensor(&[32, 2], 2));
    let optimized = optimize(&expr, RewriteNumerics::PreserveOperations);
    assert!(
        matches!(optimized.kind(), ExprKind::MatMul { a, .. } if matches!(a.kind(), ExprKind::MatMul { .. }))
    );
}

#[test]
fn zero_sized_layouts_remain_well_shaped() {
    let expr = TensorExpr::<f32>::input("empty", vec![0, 2, 3])
        .permute(vec![1, 2, 0])
        .permute(vec![2, 0, 1]);
    let optimized = expr.optimize();
    assert_eq!(optimized.shape(), &[0, 2, 3]);
    assert!(matches!(optimized.kind(), ExprKind::Input { .. }));
}

#[test]
fn shape_cost_handles_large_metadata_without_overflow() {
    let dim = usize::try_from(i32::MAX).unwrap();
    let x = TensorExpr::<f32>::input("large", vec![dim, dim, dim]);
    let optimized = (x.clone() + x).optimize();
    assert_eq!(optimized.shape(), &[dim, dim, dim]);
    assert!(matches!(
        optimized.kind(),
        ExprKind::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rewrites_match_cpu_reference() {
    // Fail rather than silently skipping when CUDA validation is requested.
    let mut executor = tnsr::ptx::PtxExecutor::<tnsr::ptx::F32>::try_new().unwrap();
    executor.set_matmul_precision(tnsr::tile::MatMulPrecision::StrictF32);
    let a = tensor(&[32, 8], 0);
    let b = tensor(&[8, 32], 1);
    for (expr, policy) in [
        (
            tensor(&[4, 8, 16], 0)
                .permute(vec![1, 2, 0])
                .permute(vec![2, 0, 1]),
            RewriteNumerics::PreserveOperations,
        ),
        (
            tensor(&[32, 32], 0).transpose() + tensor(&[32, 32], 1).transpose(),
            RewriteNumerics::PreserveOperations,
        ),
        (
            a.clone().matmul(b.clone()) + a.clone().matmul(tensor(&[8, 32], 2)),
            RewriteNumerics::RealAlgebra,
        ),
        (
            a.matmul(b).matmul(tensor(&[32, 8], 2)),
            RewriteNumerics::RealAlgebra,
        ),
    ] {
        let optimized = optimize(&expr, policy);
        let expected = SimpleExecutor::new()
            .execute(&TensorGraph::from(expr.clone()), HashMap::new())
            .unwrap();
        let actual = executor
            .execute(&TensorGraph::from(optimized), HashMap::new())
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() <= 2e-4 * (1.0 + b.abs()), "{a} vs {b}");
        }
    }
}

#[test]
fn matrix_rewrites_preserve_parameter_gradients_within_tolerance() {
    use tnsr::tensor::Parameter;
    let a = Parameter::new(vec![0.2f32; 32], vec![8, 4]);
    let b = Parameter::new(vec![0.3f32; 32], vec![4, 8]);
    let c = Parameter::new(vec![-0.1f32; 32], vec![4, 8]);
    let d = Parameter::new(vec![0.4f32; 32], vec![8, 4]);
    let x = TensorExpr::from(a.clone());
    for (expr, parameters) in [
        (
            x.clone().matmul(b.clone()) + x.clone().matmul(c.clone()),
            vec![a.id(), b.id(), c.id()],
        ),
        (
            x.matmul(b.clone()).matmul(d.clone()),
            vec![a.id(), b.id(), d.id()],
        ),
    ] {
        let optimized = optimize(&expr, RewriteNumerics::RealAlgebra);
        let gradients = |expr: TensorExpr<f32>| {
            let loss = (expr.clone() * expr).reduce_sum(1).reduce_sum(0);
            let graph = TensorGraph::from(loss);
            let root = *graph.toposort().last().unwrap();
            let graph = graph.with_gradients(root);
            let mut executor = SimpleExecutor::new();
            executor.execute(&graph, HashMap::new()).unwrap();
            executor.get_gradients(&graph)
        };
        let expected = gradients(expr);
        let actual = gradients(optimized);
        for parameter in parameters {
            assert_eq!(actual[&parameter].len(), expected[&parameter].len());
            for (a, b) in actual[&parameter].iter().zip(&expected[&parameter]) {
                assert!((a - b).abs() <= 1e-4 * (1.0 + b.abs()), "{a} vs {b}");
            }
        }
    }
}
