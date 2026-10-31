//! Optimize supported islands without moving operations across opaque boundaries.
use super::{RewriteConfig, RewriteNumerics, is_opaque, optimize_supported, segments};
use crate::tensor::{DType, TensorExpr};
use std::collections::HashMap;

pub(super) fn optimize<D: DType + Clone + Default + 'static>(
    expr: &TensorExpr<D>,
    config: &RewriteConfig,
    numerics: RewriteNumerics,
) -> TensorExpr<D> {
    let mut pass = Islands {
        config,
        numerics,
        prepared: HashMap::new(),
        optimized: HashMap::new(),
    };
    pass.optimize(expr)
}

struct Islands<'a, D: DType> {
    config: &'a RewriteConfig,
    numerics: RewriteNumerics,
    prepared: HashMap<usize, TensorExpr<D>>,
    optimized: HashMap<usize, TensorExpr<D>>,
}

fn identity<D: DType>(expr: &TensorExpr<D>) -> usize {
    std::ptr::from_ref(expr.kind()).addr()
}

impl<D: DType + Clone + Default + 'static> Islands<'_, D> {
    fn optimize(&mut self, expr: &TensorExpr<D>) -> TensorExpr<D> {
        let key = identity(expr);
        if let Some(previous) = self.optimized.get(&key) {
            return previous.clone();
        }
        let prepared = self.prepare(expr);
        let result = if prepared.children().is_empty() {
            prepared
        } else {
            optimize_supported(&prepared, self.config, self.numerics)
        };
        self.optimized.insert(key, result.clone());
        result
    }

    fn prepare(&mut self, expr: &TensorExpr<D>) -> TensorExpr<D> {
        let key = identity(expr);
        if let Some(previous) = self.prepared.get(&key) {
            return previous.clone();
        }
        // Saturate once per supported island, not once per interior node. Shared
        // opaque children are optimized once and remain the same leaf in every
        // enclosing island. All original expressions stay alive throughout the
        // pass, so these pointer keys cannot be recycled.
        let rebuilt = if is_opaque(expr) {
            expr.map_children(|child| self.optimize(child))
        } else {
            expr.map_children(|child| self.prepare(child))
        };
        // Segment copies are forward-bit-exact, but removing them before
        // autograd can remove zero-filled gradient scaffolding and its floating
        // arithmetic. Keep that change under the explicit algebraic policy.
        let result = if self.numerics == RewriteNumerics::RealAlgebra {
            segments::simplify(&rebuilt).unwrap_or(rebuilt)
        } else {
            rebuilt
        };
        self.prepared.insert(key, result.clone());
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Executor, SimpleExecutor,
        graph::{TensorGraph, TensorGraphNode},
        tensor::{ExprKind, TensorExpr, ValueDType},
    };
    use std::collections::HashMap;

    fn count(expr: &TensorExpr<f32>, predicate: fn(&ExprKind<f32>) -> bool) -> usize {
        usize::from(predicate(expr.kind()))
            + expr
                .children()
                .iter()
                .map(|child| count(child, predicate))
                .sum::<usize>()
    }

    fn run(expr: TensorExpr<f32>) -> Vec<u32> {
        SimpleExecutor::new()
            .execute(&TensorGraph::from(expr), HashMap::new())
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    #[test]
    fn layout_round_trips_optimize_inside_casts_without_erasing_rounding() {
        let source = TensorExpr::constant(
            vec![
                1.003_906_3f32,
                -0.0,
                1.011_718_8,
                f32::INFINITY,
                f32::from_bits(0x7fc0_0042),
                -2.0,
            ],
            vec![2, 3],
        );
        let expr = source
            .transpose()
            .transpose()
            .cast(ValueDType::BF16)
            .cast(ValueDType::F32);
        let optimized = expr.optimize();
        assert_eq!(
            count(&optimized, |kind| matches!(
                kind,
                ExprKind::Transpose { .. }
            )),
            0
        );
        assert_eq!(
            count(&optimized, |kind| matches!(kind, ExprKind::Cast { .. })),
            2
        );
        assert_eq!(run(expr), run(optimized));
    }

    #[test]
    fn shared_changed_cast_keeps_one_identity_and_distinct_casts_remain_distinct() {
        let source = TensorExpr::constant(vec![1.003_906_3f32; 6], vec![2, 3]);
        let inner = source.transpose().transpose();
        let shared = inner.clone().cast(ValueDType::BF16);
        let other = inner.cast(ValueDType::BF16);
        let expr = TensorExpr::concat(&[shared.clone(), shared, other], 0);
        let optimized = expr.optimize();
        let ExprKind::Concat { inputs, .. } = optimized.kind() else {
            panic!("concat disappeared")
        };
        assert!(std::ptr::eq(inputs[0].kind(), inputs[1].kind()));
        assert!(!std::ptr::eq(inputs[0].kind(), inputs[2].kind()));
        let graph: TensorGraph<f32> = optimized.clone().into();
        assert_eq!(
            graph
                .graph
                .node_weights()
                .filter(|node| matches!(node, TensorGraphNode::Cast { .. }))
                .count(),
            2
        );
        assert_eq!(run(expr), run(optimized));
    }

    #[test]
    fn unchanged_opaque_operator_and_leaves_retain_identity() {
        let source = TensorExpr::constant(vec![1.0f32; 6], vec![2, 3]);
        let cast = source.cast(ValueDType::BF16);
        let optimized = cast.optimize();
        assert!(std::ptr::eq(cast.kind(), optimized.kind()));
        let wide = TensorExpr::<f32>::input("wide", vec![usize::try_from(i32::MAX).unwrap() + 1]);
        let optimized = wide.optimize();
        assert!(std::ptr::eq(wide.kind(), optimized.kind()));
    }

    #[test]
    fn optimize_supported_children_under_max_flatten_and_segment_roots() {
        let source = TensorExpr::constant(vec![1.0f32, 2.0, -3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        for expr in [
            source.clone().transpose().transpose().reduce_max(1),
            source.clone().transpose().transpose().flatten(),
            source.transpose().transpose().narrow(1, 0, 1),
        ] {
            let optimized = expr.optimize();
            assert_eq!(
                count(&optimized, |kind| matches!(
                    kind,
                    ExprKind::Transpose { .. }
                )),
                0
            );
            assert_eq!(run(expr), run(optimized));
        }
    }
}
