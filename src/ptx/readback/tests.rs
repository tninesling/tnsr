use super::*;
use crate::tile::{DType, IndexMap, IndexPredicate};
use petgraph::graph::NodeIndex;

fn identity(shape: &[usize]) -> VirtualTensor {
    VirtualTensor::identity(NodeIndex::new(0), shape.to_vec(), DType::F32)
}
fn compare(view: &VirtualTensor, source: &[usize], affine: bool) {
    let storage: Vec<usize> = (0..count(source).unwrap()).collect();
    assert_eq!(
        StridedCopy::compile(view, source).unwrap().is_some(),
        affine,
        "{view:?}"
    );
    assert_eq!(
        materialize_host_view(storage.clone(), view, source).unwrap(),
        reference(storage, view, source).unwrap()
    );
}
#[test]
fn common_gradient_layouts_and_all_permutations_match_reference() {
    for shape in [
        vec![1, 3, 7],
        vec![2, 3, 5],
        vec![3, 1, 7],
        vec![1, 1, 4],
        vec![2, 3, 1],
    ] {
        for axes in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let view = identity(&shape).permute(&axes).unwrap();
            compare(&view, &shape, true);
            let squeezed: Vec<usize> = view.shape.iter().copied().filter(|&x| x != 1).collect();
            compare(&view.reshape(squeezed).unwrap(), &shape, true);
        }
    }
}
#[test]
fn shape_changing_reshape_and_broadcast_maps_compile_exactly() {
    let source = [2, 3, 4];
    let view = identity(&source)
        .reshape(vec![1, 4, 6])
        .unwrap()
        .permute(&[0, 2, 1])
        .unwrap()
        .broadcast_axis(0, 5)
        .unwrap();
    // Individual decoded source coordinates are non-affine, but their flattened
    // address is affine and can be copied without interpreting each element.
    compare(&view, &source, true);
    for rows in 2..6 {
        for cols in 2..7 {
            let source = [rows * cols];
            let view = identity(&source)
                .reshape(vec![rows, cols])
                .unwrap()
                .permute(&[1, 0])
                .unwrap();
            compare(&view, &source, true);
        }
    }
}
#[test]
fn slices_reverse_broadcast_and_floor_mod_compile_safely() {
    use IndexExpr::*;
    let mut view = identity(&[9]);
    view.shape = vec![4];
    view.access = IndexMap {
        results: vec![Add(
            Box::new(Const(1)),
            Box::new(Mul(Box::new(IterDim(0)), Box::new(Const(2)))),
        )],
    };
    compare(&view, &[9], true);
    view.access = IndexMap {
        results: vec![Sub(
            Box::new(Const(8)),
            Box::new(Mul(Box::new(IterDim(0)), Box::new(Const(2)))),
        )],
    };
    compare(&view, &[9], true);
    let mut view = identity(&[1, 5]).broadcast_axis(0, 4).unwrap();
    compare(&view, &[1, 5], true);
    view.access.results[1] = FloorDiv(
        Box::new(Add(
            Box::new(Mul(Box::new(IterDim(1)), Box::new(Const(2)))),
            Box::new(Const(1)),
        )),
        2,
    );
    compare(&view, &[1, 5], true);
    view.access.results[1] = Mod(Box::new(Add(Box::new(IterDim(1)), Box::new(Const(5)))), 5);
    compare(&view, &[1, 5], true);
    compare(
        &identity(&[])
            .reshape(vec![1, 1])
            .unwrap()
            .broadcast_axis(0, 4)
            .unwrap(),
        &[],
        true,
    );
}
#[test]
fn rank_only_reshape_reuses_storage_and_copy_preserves_float_bits() {
    let view = identity(&[1, 2, 3]).reshape(vec![2, 3]).unwrap();
    let data = vec![0u32, 1, 2, 3, 4, 5];
    let pointer = data.as_ptr();
    let output = materialize_host_view(data, &view, &[1, 2, 3]).unwrap();
    assert_eq!(pointer, output.as_ptr());
    let bits = [
        0x80000000, 0x7fc00123, 0x7f800000, 0xff800000, 0x7fa00001, 0,
    ];
    let values = bits.iter().copied().map(f32::from_bits).collect();
    let output = materialize_host_view(
        values,
        &identity(&[2, 3]).permute(&[1, 0]).unwrap(),
        &[2, 3],
    )
    .unwrap();
    assert_eq!(
        output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        vec![bits[0], bits[3], bits[1], bits[4], bits[2], bits[5]]
    );
}
#[test]
fn empty_scalar_and_invalid_maps() {
    assert_eq!(
        materialize_host_view(vec![7], &identity(&[]), &[]).unwrap(),
        vec![7]
    );
    assert!(
        materialize_host_view(
            Vec::<u32>::new(),
            &identity(&[0, 3]).permute(&[1, 0]).unwrap(),
            &[0, 3]
        )
        .unwrap()
        .is_empty()
    );
    assert!(materialize_host_view(vec![1], &identity(&[2]), &[2]).is_err());
    let mut view = identity(&[2]);
    view.access.results[0] = IndexExpr::Const(-1);
    assert!(materialize_host_view(vec![1, 2], &view, &[2]).is_err());
    view.access.results[0] = IndexExpr::Const(2);
    assert!(materialize_host_view(vec![1, 2], &view, &[2]).is_err());
    view.access.results[0] = IndexExpr::IterDim(5);
    assert!(materialize_host_view(vec![1, 2], &view, &[2]).is_err());
    view.access.results[0] = IndexExpr::FloorDiv(Box::new(IndexExpr::IterDim(0)), 0);
    assert!(materialize_host_view(vec![1, 2], &view, &[2]).is_err());
    view.access.results.clear();
    assert!(materialize_host_view(vec![1, 2], &view, &[2]).is_err());
    view = identity(&[2]);
    view.predicate = Some(IndexPredicate::And(Vec::new()));
    assert!(materialize_host_view(vec![1, 2], &view, &[2]).is_err());
}
#[test]
fn affine_compilation_does_not_hide_intermediate_overflow() {
    use IndexExpr::*;
    let expression = Sub(
        Box::new(Add(Box::new(IterDim(0)), Box::new(Const(i64::MAX)))),
        Box::new(Const(i64::MAX)),
    );
    let mut view = identity(&[2]);
    view.access.results[0] = expression;
    assert!(Affine::compile(&view.access.results[0], &view.shape).is_none());
    assert!(materialize_host_view(vec![1, 2], &view, &[2]).is_err());
    // Constant-domain floor/mod handling must retain Euclidean negative behavior.
    for offset in -8..9 {
        for divisor in 1..6 {
            for operation in [
                FloorDiv(Box::new(Const(offset)), divisor),
                Mod(Box::new(Const(offset)), divisor),
            ] {
                let affine = Affine::compile(&operation, &[3]).unwrap();
                for index in 0..3 {
                    assert_eq!(affine.offset, operation.evaluate(&[index], &[]).unwrap());
                }
            }
        }
    }
}

#[test]
fn affine_strided_maps_cover_positive_negative_and_zero_strides() {
    use IndexExpr::*;
    for rows in 1..=4 {
        for cols in 1..=4 {
            for a in -3i64..=3 {
                for b in -3i64..=3 {
                    let lo = (a * (rows - 1)).min(0) + (b * (cols - 1)).min(0);
                    let hi = (a * (rows - 1)).max(0) + (b * (cols - 1)).max(0);
                    let source = [usize::try_from(hi - lo + 5).unwrap()];
                    let mut view = identity(&source);
                    view.shape = vec![
                        usize::try_from(rows).unwrap(),
                        usize::try_from(cols).unwrap(),
                    ];
                    view.access.results[0] = Add(
                        Box::new(Const(2 - lo)),
                        Box::new(Add(
                            Box::new(Mul(Box::new(IterDim(0)), Box::new(Const(a)))),
                            Box::new(Mul(Box::new(IterDim(1)), Box::new(Const(b)))),
                        )),
                    );
                    compare(&view, &source, true);
                }
            }
        }
    }
}

#[test]
fn shape_changing_row_major_reshapes_reuse_storage() {
    for source in [vec![2, 3, 4], vec![1, 2, 3, 4], vec![4, 6], vec![24]] {
        for shape in [vec![4, 6], vec![3, 8], vec![2, 2, 6], vec![1, 24, 1]] {
            let view = identity(&source).reshape(shape).unwrap();
            compare(&view, &source, true);
            let data: Vec<u32> = (0..24).collect();
            let pointer = data.as_ptr();
            let output = materialize_host_view(data, &view, &source).unwrap();
            assert_eq!(pointer, output.as_ptr(), "{view:?}");
        }
    }
    let view = identity(&[128, 512]).reshape(vec![256, 256]).unwrap();
    let bits = [
        0x8000_0000,
        0x7fc0_0123,
        0x7f80_0000,
        0xff80_0000,
        0x7fa0_0001,
        0,
    ];
    let data: Vec<f32> = (0..65536)
        .map(|i| f32::from_bits(bits[i % bits.len()]))
        .collect();
    let pointer = data.as_ptr();
    let output = materialize_host_view(data, &view, &[128, 512]).unwrap();
    assert_eq!(pointer, output.as_ptr());
    assert!(
        output
            .iter()
            .enumerate()
            .all(|(i, value)| value.to_bits() == bits[i % bits.len()])
    );
}

#[test]
fn composed_reshapes_and_permuted_decode_match_reference() {
    for rows in 2..=5 {
        for cols in 2..=5 {
            let source = [rows, cols, 2];
            let view = identity(&source)
                .reshape(vec![2, rows * cols])
                .unwrap()
                .reshape(vec![cols, rows * 2])
                .unwrap();
            compare(&view, &source, true);
            let permuted = view.permute(&[1, 0]).unwrap();
            compare(&permuted, &source, true);
        }
    }
}

#[test]
fn nonlinear_modulo_gathers_still_use_checked_reference() {
    use IndexExpr::*;
    let mut view = identity(&[6]);
    view.shape = vec![4, 6];
    view.access.results[0] = Mod(Box::new(Add(Box::new(IterDim(0)), Box::new(IterDim(1)))), 6);
    compare(&view, &[6], false);
}

#[test]
fn linearized_address_does_not_hide_invalid_coordinate_or_overflow() {
    use IndexExpr::*;
    let mut view = identity(&[2, 3]);
    // The flattened address would be the identity, but coordinate 1 exceeds 2.
    view.access.results = vec![
        Const(0),
        Add(
            Box::new(Mul(Box::new(IterDim(0)), Box::new(Const(3)))),
            Box::new(IterDim(1)),
        ),
    ];
    assert!(materialize_host_view((0..6).collect::<Vec<_>>(), &view, &[2, 3]).is_err());
    let mut view = identity(&[2]);
    view.access.results[0] = Mod(
        Box::new(Add(Box::new(IterDim(0)), Box::new(Const(i64::MAX)))),
        2,
    );
    assert!(StridedCopy::compile(&view, &[2]).unwrap().is_none());
    assert!(materialize_host_view(vec![0, 1], &view, &[2]).is_err());
}

#[test]
fn conservative_modulo_bounds_do_not_reject_reachable_subsets() {
    use IndexExpr::*;
    let mut view = identity(&[3]);
    view.shape = vec![2, 3];
    view.access.results[0] = Mod(
        Box::new(Add(
            Box::new(Mul(Box::new(IterDim(0)), Box::new(Const(100)))),
            Box::new(IterDim(1)),
        )),
        100,
    );
    // The divisible leading digit does not change the bounded low digit.
    // Mixed-radix compilation proves the reachable subset instead of using
    // the overly conservative interval 0..100.
    compare(&view, &[3], true);
    let mut constant = identity(&[2]);
    constant.access.results[0] = Mod(
        Box::new(Add(
            Box::new(Mul(Box::new(IterDim(0)), Box::new(Const(100)))),
            Box::new(Const(1)),
        )),
        100,
    );
    compare(&constant, &[2], true);
}

#[test]
fn decoded_affine_addresses_preserve_negative_and_zero_strides() {
    use IndexExpr::*;
    for rows in 1..=3 {
        for cols in 1..=3 {
            for a in -2i64..=2 {
                for b in -2i64..=2 {
                    let lo = (a * (rows - 1)).min(0) + (b * (cols - 1)).min(0);
                    let hi = (a * (rows - 1)).max(0) + (b * (cols - 1)).max(0);
                    let size = usize::try_from(hi - lo + 5).unwrap();
                    let source = [size.div_ceil(7), 7];
                    let mut view = identity(&source);
                    view.shape = vec![
                        usize::try_from(rows).unwrap(),
                        usize::try_from(cols).unwrap(),
                    ];
                    let linear = Add(
                        Box::new(Const(2 - lo)),
                        Box::new(Add(
                            Box::new(Mul(Box::new(IterDim(0)), Box::new(Const(a)))),
                            Box::new(Mul(Box::new(IterDim(1)), Box::new(Const(b)))),
                        )),
                    );
                    view.access.results = vec![
                        FloorDiv(Box::new(linear.clone()), 7),
                        Mod(Box::new(linear), 7),
                    ];
                    compare(&view, &source, true);
                }
            }
        }
    }
}

#[test]
fn negative_i64_remainder_reconstruction_preserves_checked_semantics() {
    use IndexExpr::*;
    let mut view = identity(&[2, 3]);
    view.shape = vec![2];
    let linear = Add(Box::new(Const(i64::MIN)), Box::new(IterDim(0)));
    // The original modulo operations are valid. Their expanded q*d values can
    // be below i64::MIN, which must remain compile-time i128 arithmetic only.
    view.access.results = vec![
        FloorDiv(Box::new(Mod(Box::new(linear.clone()), 6)), 3),
        Mod(Box::new(linear), 3),
    ];
    compare(&view, &[2, 3], true);
    assert_eq!(
        materialize_host_view((0..6).collect::<Vec<_>>(), &view, &[2, 3]).unwrap(),
        vec![4, 5]
    );
    let mut invalid = identity(&[3]);
    invalid.shape = vec![2];
    invalid.access.results[0] = Mod(
        Box::new(Sub(Box::new(Const(i64::MIN)), Box::new(IterDim(0)))),
        3,
    );
    assert!(StridedCopy::compile(&invalid, &[3]).unwrap().is_none());
    assert!(materialize_host_view(vec![0, 1, 2], &invalid, &[3]).is_err());
}

fn affine_value(affine: &Affine, coordinates: &[i64]) -> i64 {
    i64::try_from(
        affine
            .strides
            .iter()
            .zip(coordinates)
            .fold(i128::from(affine.offset), |value, (&stride, &index)| {
                value + i128::from(stride) * i128::from(index)
            }),
    )
    .unwrap()
}

#[test]
fn mixed_radix_floor_and_modulo_compile_rectangular_transposes_and_reversals() {
    use IndexExpr::*;
    for rows in [2, 3, 17] {
        for cols in [2, 5, 16] {
            let mut transposed = identity(&[cols, rows]);
            transposed.shape = vec![rows, cols];
            let linear = Add(
                Box::new(Mul(
                    Box::new(IterDim(0)),
                    Box::new(Const(i64::try_from(cols).unwrap())),
                )),
                Box::new(IterDim(1)),
            );
            transposed.access.results = vec![
                Mod(Box::new(linear.clone()), i64::try_from(cols).unwrap()),
                FloorDiv(Box::new(linear), i64::try_from(cols).unwrap()),
            ];
            compare(&transposed, &[cols, rows], true);
            let mut reversed = identity(&[rows, cols]);
            let d = i64::try_from(cols).unwrap();
            let linear = Sub(
                Box::new(Add(
                    Box::new(Mul(Box::new(IterDim(0)), Box::new(Const(d)))),
                    Box::new(Const(d - 1)),
                )),
                Box::new(IterDim(1)),
            );
            reversed.access.results = vec![
                FloorDiv(Box::new(linear.clone()), d),
                Mod(Box::new(linear), d),
            ];
            compare(&reversed, &[rows, cols], true);
        }
    }
}

#[test]
fn mixed_radix_affine_results_match_euclidean_evaluation_for_signed_coefficients() {
    use IndexExpr::*;
    let shape = [3, 4];
    let mut compiled = 0usize;
    for divisor in 1..=7 {
        for offset in -8..=8 {
            for a in -8..=8 {
                for b in -8..=8 {
                    let value = Add(
                        Box::new(Const(offset)),
                        Box::new(Add(
                            Box::new(Mul(Box::new(IterDim(0)), Box::new(Const(a)))),
                            Box::new(Mul(Box::new(IterDim(1)), Box::new(Const(b)))),
                        )),
                    );
                    for expression in [
                        FloorDiv(Box::new(value.clone()), divisor),
                        Mod(Box::new(value), divisor),
                    ] {
                        if let Some(affine) = Affine::compile(&expression, &shape) {
                            compiled += 1;
                            for i in 0..3 {
                                for j in 0..4 {
                                    assert_eq!(
                                        affine_value(&affine, &[i, j]),
                                        expression.evaluate(&[i, j], &[]).unwrap(),
                                        "{expression:?}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(compiled > 10_000);
}

#[test]
fn mixed_radix_carry_boundaries_and_original_overflows_remain_checked() {
    use IndexExpr::*;
    // The low digit can reach d, so its quotient contributes a varying carry.
    let value = Add(
        Box::new(Mul(Box::new(IterDim(0)), Box::new(Const(4)))),
        Box::new(IterDim(1)),
    );
    for expression in [
        FloorDiv(Box::new(value.clone()), 4),
        Mod(Box::new(value), 4),
    ] {
        assert!(Affine::compile(&expression, &[2, 5]).is_none());
    }
    // A flattened mathematical cancellation may not hide an overflowing
    // multiply in the original checked expression before division/modulo.
    let term = Mul(Box::new(IterDim(0)), Box::new(Const(i64::MAX)));
    let value = Sub(Box::new(term.clone()), Box::new(term));
    for expression in [
        FloorDiv(Box::new(value.clone()), 4),
        Mod(Box::new(value), 4),
    ] {
        assert!(Affine::compile(&expression, &[3]).is_none());
        let mut domain = CheckedDomain {
            shape: &[3],
            bounds: HashMap::new(),
        };
        assert!(LinearAddress::compile(&expression, &mut domain).is_none());
        assert!(expression.evaluate(&[2], &[]).is_err());
    }
}

#[test]
fn positive_two_dimensional_copy_fast_path_matches_reference() {
    use IndexExpr::*;
    for rows in [2, 3, 17] {
        for cols in [2, 5, 16] {
            let source = [cols, rows];
            let view = identity(&source).permute(&[1, 0]).unwrap();
            let storage: Vec<usize> = (0..rows * cols).collect();
            let plan = StridedCopy::compile(&view, &source).unwrap().unwrap();
            let copied = plan
                .copy_positive_matrix(&storage, rows * cols)
                .unwrap()
                .unwrap();
            assert_eq!(copied, reference(storage, &view, &source).unwrap());
        }
    }
    // Positive non-unit strides and a nonzero offset, plus either-axis
    // broadcasting, remain valid matrix maps without scalar copying overhead.
    for (a, b, offset) in [(8, 2, 1), (0, 2, 1), (2, 0, 1), (0, 0, 0)] {
        let mut view = identity(&[32]);
        view.shape = vec![3, 4];
        view.access.results[0] = Add(
            Box::new(Const(offset)),
            Box::new(Add(
                Box::new(Mul(Box::new(IterDim(0)), Box::new(Const(a)))),
                Box::new(Mul(Box::new(IterDim(1)), Box::new(Const(b)))),
            )),
        );
        let storage: Vec<usize> = (0..32).collect();
        let plan = StridedCopy::compile(&view, &[32]).unwrap().unwrap();
        assert_eq!(plan.block, 1);
        assert_eq!(
            plan.copy_positive_matrix(&storage, 12).unwrap().unwrap(),
            reference(storage, &view, &[32]).unwrap()
        );
    }
}

#[test]
fn positive_matrix_copy_validates_bounds_before_indexing() {
    let plan = |offset, strides, extents| StridedCopy {
        offset,
        strides,
        extents,
        block: 1,
    };
    assert!(
        plan(0, vec![3, 1], vec![2, 3])
            .copy_positive_matrix(&[0, 1, 2], 6)
            .is_err()
    );
    assert!(
        plan(0, vec![3, 1], vec![2, 3])
            .copy_positive_matrix(&[0; 6], 5)
            .is_err()
    );
    assert!(
        plan(0, vec![i64::MAX, i64::MAX], vec![3, 3])
            .copy_positive_matrix(&[0; 6], 9)
            .is_err()
    );
    assert!(
        plan(0, vec![0, 0], vec![0, 3])
            .copy_positive_matrix(&[0], 0)
            .is_err()
    );
    // General or reversed maps keep the existing checked strided iterator.
    assert!(
        plan(3, vec![-1, 0], vec![2, 3])
            .copy_positive_matrix(&[0; 6], 6)
            .unwrap()
            .is_none()
    );
    assert!(
        plan(0, vec![1], vec![6])
            .copy_positive_matrix(&[0; 6], 6)
            .unwrap()
            .is_none()
    );
}

#[test]
fn physical_identity_readback_reuses_storage_but_rejects_prefix_maps() {
    let bits = [
        0x8000_0000,
        0x7fc0_0123,
        0,
        0x7f80_0000,
        0xff80_0000,
        0x7fa0_0001,
    ];
    let data: Vec<f32> = bits.iter().copied().map(f32::from_bits).collect();
    let pointer = data.as_ptr();
    let output = materialize_host_view(data, &identity(&[2, 3]), &[2, 3]).unwrap();
    assert_eq!(pointer, output.as_ptr());
    assert_eq!(output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits);
    let mut view = identity(&[2, 3]);
    view.access.results.truncate(1);
    assert!(view.access.is_identity());
    assert!(materialize_host_view((0..6).collect::<Vec<_>>(), &view, &[2, 3]).is_err());
    view.access.results.clear();
    assert!(view.access.is_identity());
    assert!(materialize_host_view((0..6).collect::<Vec<_>>(), &view, &[2, 3]).is_err());
    let mut predicated = identity(&[2, 3]);
    predicated.predicate = Some(IndexPredicate::And(Vec::new()));
    assert!(materialize_host_view((0..6).collect::<Vec<_>>(), &predicated, &[2, 3]).is_err());
    assert!(materialize_host_view(vec![0], &identity(&[2, 3]), &[2, 3]).is_err());
}
