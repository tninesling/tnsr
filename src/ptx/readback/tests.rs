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
fn reshape_and_broadcast_fallbacks_remain_exact() {
    let source = [2, 3, 4];
    let view = identity(&source)
        .reshape(vec![1, 4, 6])
        .unwrap()
        .permute(&[0, 2, 1])
        .unwrap()
        .broadcast_axis(0, 5)
        .unwrap();
    // Flattening/decoding across the 3/4 source axes is non-affine.
    compare(&view, &source, false);
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
