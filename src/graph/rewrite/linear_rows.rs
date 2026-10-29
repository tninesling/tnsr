//! Shape-local, opt-in row regrouping before differentiation.
use crate::tensor::{BinaryOp, DType, ExprKind, TensorExpr};

pub(super) fn rewrite<D: DType + Default + 'static>(expr: &TensorExpr<D>) -> Option<TensorExpr<D>> {
    let (matrix, bias) = match expr.kind() {
        ExprKind::MatMul { .. } => (expr, None),
        ExprKind::Binary {
            op: BinaryOp::Add,
            a,
            b,
        } => {
            if matches!(a.kind(), ExprKind::MatMul { .. }) {
                (a, Some(b))
            } else if matches!(b.kind(), ExprKind::MatMul { .. }) {
                (b, Some(a))
            } else {
                return None;
            }
        }
        _ => return None,
    };
    if expr.shape() != matrix.shape() {
        return None;
    }
    let ExprKind::MatMul { a, b } = matrix.kind() else {
        return None;
    };
    let shape = a.shape();
    if shape.len() <= 2 || b.shape().len() != 2 || shape.contains(&0) || b.shape().contains(&0) {
        return None;
    }
    let k = *shape.last()?;
    if b.shape()[0] != k {
        return None;
    }
    let n = b.shape()[1];
    let rows = shape[..shape.len() - 1]
        .iter()
        .try_fold(1usize, |v, &d| v.checked_mul(d))?;
    rows.checked_mul(n)?;
    let bias = match bias {
        Some(bias) => Some(shared_bias(bias, n)?),
        None => None,
    };
    let output = a.clone().reshape(vec![rows, k]).matmul(b.clone());
    let output = if let Some(bias) = bias {
        output + bias.broadcast(vec![rows, n])
    } else {
        output
    };
    Some(output.reshape(expr.shape().clone()))
}

fn shared_bias<D: DType + Default + 'static>(
    expr: &TensorExpr<D>,
    n: usize,
) -> Option<TensorExpr<D>> {
    if expr.shape().last() != Some(&n) {
        return None;
    }
    match expr.kind() {
        ExprKind::BroadcastAxis { x, axis } if axis + 1 < expr.shape().len() => shared_bias(x, n),
        ExprKind::Reshape { x }
            if expr
                .shape()
                .iter()
                .try_fold(1usize, |v, &d| v.checked_mul(d))?
                == n =>
        {
            shared_bias(x, n)
        }
        _ if expr
            .shape()
            .iter()
            .try_fold(1usize, |v, &d| v.checked_mul(d))?
            == n =>
        {
            Some(if expr.shape().len() == 1 {
                expr.clone()
            } else {
                expr.clone().reshape(vec![n])
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        RewriteConfig, RewriteNumerics, optimize_expr, optimize_expr_with_numerics,
    };
    use super::*;
    use crate::{
        Executor, SimpleExecutor,
        graph::{TensorGraph, TensorGraphNode},
        nn,
        tensor::Parameter,
    };

    #[test]
    fn row_regrouping_is_opt_in_and_uses_one_matrix_gradient() {
        let weight = Parameter::new(
            (0..15).map(|i| (i as f32 - 7.0) / 13.0).collect(),
            vec![3, 5],
        );
        let bias = Parameter::new(vec![0.03f32; 5], vec![5]);
        let input = TensorExpr::constant(
            (0..24).map(|i| (i as f32 - 12.0) / 17.0).collect(),
            vec![2, 4, 3],
        );
        let original = nn::linear(input, weight.clone(), Some(bias.clone()));
        let kept = optimize_expr(&original, RewriteConfig::default());
        let rewritten = optimize_expr_with_numerics(
            &original,
            RewriteConfig::default(),
            RewriteNumerics::RealAlgebra,
        );
        assert_eq!(rewritten.shape(), original.shape());
        let kept_graph: TensorGraph<f32> = kept.into();
        assert!(
            kept_graph
                .graph
                .node_weights()
                .any(|n| matches!(n,TensorGraphNode::MatMul {shape}if shape.len()==3))
        );
        let mut expected = None;
        for expr in [original, rewritten] {
            let forward: TensorGraph<f32> = expr.clone().into();
            let values = SimpleExecutor::new()
                .execute(&forward, HashMap::new())
                .unwrap();
            let loss = expr.reduce_sum(2).reduce_sum(1).reduce_sum(0);
            let graph: TensorGraph<f32> = loss.into();
            let root = *graph.toposort().last().unwrap();
            let graph = graph.with_gradients(root);
            let mut cpu = SimpleExecutor::new();
            cpu.execute(&graph, HashMap::new()).unwrap();
            let gradients = cpu.get_gradients(&graph);
            if let Some((ref old_values, ref old_gradients)) = expected {
                let old_values: &Vec<f32> = old_values;
                let old_gradients: &std::collections::HashMap<usize, Vec<f32>> = old_gradients;
                for (a, b) in values.iter().zip(old_values) {
                    assert!((a - b).abs() < 1e-5);
                }
                for parameter in [weight.id(), bias.id()] {
                    for (a, b) in gradients[&parameter].iter().zip(&old_gradients[&parameter]) {
                        assert!((a - b).abs() < 1e-5);
                    }
                }
                assert!(
                    graph
                        .graph
                        .node_weights()
                        .all(|n| !matches!(n,TensorGraphNode::MatMul {shape}if shape.len()>2))
                );
            } else {
                expected = Some((values, gradients));
            }
        }
    }

    #[test]
    fn row_specific_bias_batched_weights_and_zero_extents_are_not_regrouped() {
        let x = TensorExpr::<f32>::input("x", vec![2, 4, 3]);
        let w = TensorExpr::input("w", vec![3, 5]);
        let expr = x.clone().matmul(w) + TensorExpr::input("row-bias", vec![2, 4, 5]);
        assert!(rewrite(&expr).is_none());
        assert!(rewrite(&x.matmul(TensorExpr::input("batched-w", vec![2, 3, 5]))).is_none());
        assert!(
            rewrite(
                &TensorExpr::<f32>::input("zero", vec![0, 4, 3])
                    .matmul(TensorExpr::input("w", vec![3, 5]))
            )
            .is_none()
        );
    }

    #[test]
    fn opaque_cast_input_keeps_its_rounding_boundary() {
        let x = TensorExpr::<f32>::input("x", vec![2, 4, 3])
            .cast(crate::tensor::ValueDType::BF16)
            .cast(crate::tensor::ValueDType::F32);
        let expr = x.matmul(TensorExpr::input("w", vec![3, 5]));
        let optimized = optimize_expr_with_numerics(
            &expr,
            RewriteConfig::default(),
            RewriteNumerics::RealAlgebra,
        );
        let graph: TensorGraph<f32> = optimized.into();
        assert_eq!(
            graph
                .graph
                .node_weights()
                .filter(|n| matches!(n, TensorGraphNode::Cast { .. }))
                .count(),
            2
        );
        assert!(
            graph
                .graph
                .node_weights()
                .any(|n| matches!(n,TensorGraphNode::MatMul {shape}if shape==&[8,5]))
        );
    }
    use std::collections::HashMap;
}
