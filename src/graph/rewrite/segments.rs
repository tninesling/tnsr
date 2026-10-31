//! Forward-exact copy identities, enabled only by RealAlgebra because their
//! zero-filled autograd scaffolding can affect signed zero and nonfinite gradients.
use crate::tensor::{DType, ExprKind, TensorExpr};

fn same<D: DType>(a: &TensorExpr<D>, b: &TensorExpr<D>) -> bool {
    std::ptr::eq(a.kind(), b.kind())
}

pub(super) fn simplify<D: DType + Clone + Default + 'static>(
    expr: &TensorExpr<D>,
) -> Option<TensorExpr<D>> {
    match expr.kind() {
        ExprKind::Narrow { x, axis, start } => {
            let length = *expr.shape().get(*axis)?;
            match x.kind() {
                ExprKind::Narrow {
                    x: source,
                    axis: inner_axis,
                    start: inner_start,
                } if axis == inner_axis => {
                    let offset = inner_start.checked_add(*start)?;
                    let extent = *source.shape().get(*axis)?;
                    (offset <= extent && length <= extent - offset)
                        .then(|| source.clone().narrow(*axis, offset, length))
                }
                ExprKind::Concat {
                    inputs,
                    axis: concat_axis,
                } if axis == concat_axis => {
                    let mut offset = 0usize;
                    for input in inputs {
                        let extent = *input.shape().get(*axis)?;
                        let end = offset.checked_add(extent)?;
                        // A slice spanning multiple segments stays unchanged.
                        if *start >= offset && *start <= end && length <= end - start {
                            return Some(input.clone().narrow(*axis, start - offset, length));
                        }
                        offset = end;
                    }
                    None
                }
                _ => None,
            }
        }
        ExprKind::Concat { inputs, axis } => {
            let mut source: Option<&TensorExpr<D>> = None;
            let mut offset = 0usize;
            for input in inputs {
                let ExprKind::Narrow {
                    x,
                    axis: input_axis,
                    start,
                } = input.kind()
                else {
                    return None;
                };
                if input_axis != axis || *start != offset || source.is_some_and(|s| !same(s, x)) {
                    return None;
                }
                source = Some(x);
                offset = offset.checked_add(*input.shape().get(*axis)?)?;
            }
            let source = source?;
            (source.shape() == expr.shape() && source.shape().get(*axis) == Some(&offset))
                .then(|| source.clone())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::rewrite::{RewriteConfig, RewriteNumerics, optimize_expr_with_numerics};
    use crate::{
        Executor, SimpleExecutor,
        graph::TensorGraph,
        tensor::{ExprKind, Parameter, TensorExpr},
    };
    use std::collections::HashMap;

    fn run(expr: TensorExpr<f32>) -> Vec<u32> {
        SimpleExecutor::new()
            .execute(&TensorGraph::from(expr), HashMap::new())
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    fn check(expr: &TensorExpr<f32>) -> TensorExpr<f32> {
        let optimized = optimize(expr);
        assert_eq!(expr.shape(), optimized.shape());
        assert_eq!(run(expr.clone()), run(optimized.clone()));
        optimized
    }

    fn optimize(expr: &TensorExpr<f32>) -> TensorExpr<f32> {
        optimize_expr_with_numerics(expr, RewriteConfig::default(), RewriteNumerics::RealAlgebra)
    }

    #[test]
    fn exact_segments_round_trip_stored_bits_on_outer_and_inner_axes() {
        let bits = [
            0x8000_0000,
            0,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0042,
            0x3f80_0000,
        ];
        let source =
            TensorExpr::constant(bits.into_iter().map(f32::from_bits).collect(), vec![2, 3]);
        for (axis, split) in [(0, 1), (1, 1), (1, 2)] {
            let first = source.clone().narrow(axis, 0, split);
            let last = source
                .clone()
                .narrow(axis, split, source.shape()[axis] - split);
            let joined = TensorExpr::concat(&[first.clone(), last.clone()], axis);
            let optimized = check(&joined);
            assert!(std::ptr::eq(optimized.kind(), source.kind()));
            for (start, length, expected) in [
                (0, split, first),
                (split, source.shape()[axis] - split, last),
            ] {
                // Add a different input so cancellation is exercised directly,
                // rather than only after reconstructing a common source.
                let other = TensorExpr::constant(vec![3.0f32; 6], vec![2, 3]);
                let pieces = if start == 0 {
                    vec![
                        expected.clone(),
                        other.narrow(axis, start + length, source.shape()[axis] - start - length),
                    ]
                } else {
                    vec![other.narrow(axis, 0, start), expected.clone()]
                };
                let output = check(&TensorExpr::concat(&pieces, axis).narrow(axis, start, length));
                assert!(std::ptr::eq(output.kind(), expected.kind()));
            }
        }
    }

    #[test]
    fn nested_narrow_composes_only_the_same_axis() {
        let source = TensorExpr::constant((0_u8..30).map(f32::from).collect(), vec![5, 6]);
        let expr = source.clone().narrow(1, 1, 4).narrow(1, 2, 1);
        let optimized = check(&expr);
        assert!(
            matches!(optimized.kind(), ExprKind::Narrow { x, axis: 1, start: 3 } if std::ptr::eq(x.kind(), source.kind()))
        );
        let expr = source.narrow(1, 1, 4).narrow(0, 1, 2);
        let optimized = check(&expr);
        assert!(
            matches!(optimized.kind(), ExprKind::Narrow { x, axis: 0, start: 1 } if matches!(x.kind(), ExprKind::Narrow {axis: 1, ..}))
        );
    }

    #[test]
    fn every_in_bounds_slice_of_concat_preserves_bits_and_empty_extents() {
        for axis in 0..2 {
            let a = TensorExpr::constant(vec![-0.0f32, 1.0, 2.0, f32::NAN, 3.0, 4.0], vec![2, 3]);
            let b =
                TensorExpr::constant(vec![5.0f32, -0.0, 6.0, 7.0, f32::INFINITY, 8.0], vec![2, 3]);
            let concat = TensorExpr::concat(&[a, b], axis);
            let extent = concat.shape()[axis];
            for start in 0..=extent {
                for length in 0..=extent - start {
                    check(&concat.clone().narrow(axis, start, length));
                }
            }
        }
    }

    #[test]
    fn reordered_overlapping_gapped_or_distinct_sources_do_not_cancel() {
        let x = TensorExpr::constant((0_u8..6).map(f32::from).collect(), vec![1, 6]);
        let y = TensorExpr::constant((0_u8..6).map(f32::from).collect(), vec![1, 6]);
        for pieces in [
            vec![x.clone().narrow(1, 3, 3), x.clone().narrow(1, 0, 3)],
            vec![x.clone().narrow(1, 0, 3), x.clone().narrow(1, 2, 3)],
            vec![x.clone().narrow(1, 0, 2), x.clone().narrow(1, 3, 3)],
            vec![x.clone().narrow(1, 0, 3), y.narrow(1, 3, 3)],
        ] {
            assert!(matches!(
                check(&TensorExpr::concat(&pieces, 1)).kind(),
                ExprKind::Concat { .. }
            ));
        }
    }

    #[test]
    fn segment_rewrites_preserve_parameter_gradients_with_fanout() {
        let parameter = Parameter::new(
            (0_u8..12).map(|i| f32::from(i) / 13.0).collect(),
            vec![3, 4],
        );
        let source: TensorExpr<f32> = parameter.clone().into();
        let joined = TensorExpr::concat(
            &[
                source.clone().narrow(1, 0, 1),
                source.clone().narrow(1, 1, 3),
            ],
            1,
        );
        let selected = joined.narrow(0, 1, 2).narrow(0, 0, 1);
        let expr = ((selected.clone() * selected).reduce_sum(1).reduce_sum(0)
            + source.reduce_sum(1).reduce_sum(0))
        .reduce_sum(0);
        let mut expected = None;
        for candidate in [expr.clone(), optimize(&expr)] {
            let graph: TensorGraph<f32> = candidate.into();
            let root = *graph.toposort().last().unwrap();
            let graph = graph.with_gradients(root);
            let mut executor = SimpleExecutor::new();
            executor.execute(&graph, HashMap::new()).unwrap();
            let bits: Vec<_> = executor.get_gradients(&graph)[&parameter.id()]
                .iter()
                .copied()
                .map(f32::to_bits)
                .collect();
            if let Some(expected) = &expected {
                assert_eq!(&bits, expected);
            } else {
                expected = Some(bits);
            }
        }
    }

    #[test]
    fn segment_gradient_signed_zero_changes_require_explicit_opt_in() {
        let parameter = Parameter::new(vec![1.0f32, 2.0], vec![2]);
        let x: TensorExpr<f32> = parameter.clone().into();
        let expr = TensorExpr::concat(&[x.clone().narrow(0, 0, 1), x.narrow(0, 1, 1)], 0);
        let mut results = Vec::new();
        for candidate in [expr.clone(), expr.optimize(), optimize(&expr)] {
            let loss = (candidate * TensorExpr::constant(vec![-0.0f32; 2], vec![2])).reduce_sum(0);
            let graph: TensorGraph<f32> = loss.into();
            let root = *graph.toposort().last().unwrap();
            let graph = graph.with_gradients(root);
            let mut cpu = SimpleExecutor::new();
            cpu.execute(&graph, HashMap::new()).unwrap();
            results.push(
                cpu.get_gradients(&graph)[&parameter.id()]
                    .iter()
                    .copied()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(results[0], vec![0, 0]);
        assert_eq!(results[0], results[1]);
        assert_eq!(results[2], vec![(-0.0f32).to_bits(); 2]);
    }

    #[test]
    fn removing_unused_concat_branch_nonfinite_gradients_requires_explicit_opt_in() {
        let parameter = Parameter::new(vec![0.0f32], vec![1]);
        let x: TensorExpr<f32> = parameter.clone().into();
        let expr = TensorExpr::concat(&[x.log(), TensorExpr::constant(vec![2.0f32], vec![1])], 0)
            .narrow(0, 1, 1);
        assert_eq!(run(expr.clone()), run(optimize(&expr)));
        let mut results = Vec::new();
        for candidate in [expr.clone(), expr.optimize(), optimize(&expr)] {
            let graph: TensorGraph<f32> = candidate.into();
            let root = *graph.toposort().last().unwrap();
            let graph = graph.with_gradients(root);
            let mut cpu = SimpleExecutor::new();
            cpu.execute(&graph, HashMap::new()).unwrap();
            results.push(cpu.get_gradients(&graph));
        }
        assert!(results[0][&parameter.id()][0].is_nan());
        assert_eq!(
            results[0][&parameter.id()][0].to_bits(),
            results[1][&parameter.id()][0].to_bits()
        );
        assert!(!results[2].contains_key(&parameter.id()));
    }
}
