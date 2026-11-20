use super::*;
use crate::tile::{CompareOp, IndexExpr, IndexPredicate, ScalarValue};

fn fixture(batch: usize, heads: usize, source_shape: Vec<usize>, sequence: bool) -> Value {
    let source = NodeIndex::new(7);
    let mut view = VirtualTensor::identity(source, source_shape.clone(), DType::BF16);
    if sequence {
        view = view
            .reshape(vec![batch, SEQUENCE, heads, WIDTH])
            .unwrap()
            .permute(&[0, 2, 1, 3])
            .unwrap();
    } else {
        view = view.reshape(vec![batch, heads, SEQUENCE, WIDTH]).unwrap();
    }
    Value {
        binding: Binding {
            pointer: 0x1000,
            len: batch * heads * SEQUENCE * WIDTH,
            dtype: DType::BF16,
        },
        source_shape,
        view,
        chain: Vec::new(),
    }
}

#[test]
fn training_attention_union_input_layout_proves_complete_source_maps_and_key_transpose() {
    for (shape, sequence, expected) in [
        (
            vec![2, 512, 3 * 64],
            true,
            InputLayout::SequenceMajor { heads: 3 },
        ),
        (
            vec![2, 512, 3, 64],
            true,
            InputLayout::SequenceMajor { heads: 3 },
        ),
        (vec![2, 3, 512, 64], false, InputLayout::DenseHeadMajor),
        (vec![6, 512, 64], false, InputLayout::DenseHeadMajor),
    ] {
        let query = fixture(2, 3, shape, sequence);
        let mut key = query.clone();
        key.view = key.view.permute(&[0, 1, 3, 2]).unwrap();
        assert_eq!(prove(&query, 2, 3, false), Some(expected));
        assert_eq!(prove(&key, 2, 3, true), Some(expected));
        assert_eq!(prove_triplet(&query, &key, &query, 2, 3), Some(expected));
        assert_eq!(prove(&key, 2, 3, false), None);
        assert_eq!(prove(&query, 2, 3, true), None);
    }
}

#[test]
fn training_attention_union_input_layout_rejects_offsets_masks_broadcast_and_wrong_storage() {
    let value = fixture(2, 3, vec![2, 512, 192], true);
    let mut failures = Vec::new();
    let mut bad = value.clone();
    bad.binding.len += 1;
    failures.push(bad);
    let mut bad = value.clone();
    bad.binding.dtype = DType::F32;
    failures.push(bad);
    let mut bad = value.clone();
    bad.view.dtype = DType::F32;
    failures.push(bad);
    let mut bad = value.clone();
    bad.binding.pointer += 1;
    failures.push(bad);
    let mut bad = value.clone();
    bad.binding.pointer = u64::MAX - 1;
    failures.push(bad);
    let mut bad = value.clone();
    bad.source_shape = vec![1024, 192];
    failures.push(bad);
    let mut bad = value.clone();
    bad.source_shape[2] *= 3;
    failures.push(bad);
    let mut bad = value.clone();
    bad.view.access.results[2] = IndexExpr::Add(
        Box::new(bad.view.access.results[2].clone()),
        Box::new(IndexExpr::Const(1)),
    );
    failures.push(bad);
    let mut bad = value.clone();
    bad.view.access.results[0] = IndexExpr::Const(0);
    failures.push(bad);
    let mut bad = value.clone();
    bad.view.predicate = Some(IndexPredicate::Compare {
        op: CompareOp::Lt,
        lhs: IndexExpr::IterDim(2),
        rhs: IndexExpr::Const(511),
    });
    failures.push(bad);
    let mut bad = value.clone();
    bad.view.out_of_bounds = Some(ScalarValue::BF16(0));
    failures.push(bad);
    for bad in failures {
        assert_eq!(prove(&bad, 2, 3, false), None);
    }
    let dense = fixture(2, 3, vec![6, 512, 64], false);
    let mut key = value.clone();
    key.view = key.view.permute(&[0, 1, 3, 2]).unwrap();
    assert_eq!(prove_triplet(&value, &key, &dense, 2, 3), None);
}

fn evaluate(expr: &Expr, head: usize, sequence: usize, feature: usize) -> usize {
    match expr {
        Expr::Const(value) => usize::try_from(*value).unwrap(),
        Expr::Var(name) if name == "head" => head,
        Expr::Var(name) if name == "sequence" => sequence,
        Expr::Var(name) if name == "feature" => feature,
        Expr::Add(a, b) => {
            evaluate(a, head, sequence, feature) + evaluate(b, head, sequence, feature)
        }
        Expr::Mul(a, b) => {
            evaluate(a, head, sequence, feature) * evaluate(b, head, sequence, feature)
        }
        Expr::FloorDiv(a, n) => evaluate(a, head, sequence, feature) / n,
        Expr::Mod(a, n) => evaluate(a, head, sequence, feature) % n,
        _ => panic!("unexpected physical-index expression"),
    }
}

#[test]
fn training_attention_union_sequence_index_matches_virtual_map_bijectively() -> Result<()> {
    let (batch, heads) = (2, 3);
    let source = fixture(batch, heads, vec![batch, 512, heads * 64], true);
    let expr = InputLayout::SequenceMajor { heads }.index(
        Expr::Var("head".into()),
        Expr::Var("sequence".into()),
        Expr::Var("feature".into()),
    );
    let mut visits = vec![0_u8; source.binding.len];
    for b in 0..batch {
        for h in 0..heads {
            for s in 0..SEQUENCE {
                for d in 0..WIDTH {
                    let coordinate = source
                        .view
                        .access
                        .evaluate(&[b as i64, h as i64, s as i64, d as i64], &[])?;
                    let expected = coordinate
                        .iter()
                        .zip(&source.source_shape)
                        .fold(0_i64, |n, (&i, &extent)| n * extent as i64 + i);
                    let actual = evaluate(&expr, b * heads + h, s, d);
                    assert_eq!(actual, usize::try_from(expected)?);
                    visits[actual] += 1;
                }
            }
        }
    }
    assert!(visits.into_iter().all(|count| count == 1));
    assert!(
        InputLayout::SequenceMajor { heads: 0 }
            .validate(Geometry::new(6, 32)?)
            .is_err()
    );
    assert!(
        InputLayout::SequenceMajor { heads: 4 }
            .validate(Geometry::new(6, 32)?)
            .is_err()
    );
    assert!(
        ir_with_input_layout(
            Geometry::new(6, 32)?,
            false,
            InputLayout::SequenceMajor { heads: 3 }
        )
        .is_err()
    );
    assert!(
        ir_with_input_layout(
            Geometry::new(6, 16)?,
            true,
            InputLayout::SequenceMajor { heads: 3 }
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn training_attention_union_dense_ptx_stays_byte_identical_to_retained_golden() -> Result<()> {
    let Some(directory) = std::env::var_os("TNSR_ATTENTION_DENSE_GOLDEN") else {
        return Ok(());
    };
    for query_tile in [16, 32] {
        for dump in [false, true] {
            let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
            tiles
                .region_kernels
                .push(crate::tile::graph::TileRegionKernel {
                    region_id: 0,
                    ir: ir(Geometry::new(2, query_tile)?, dump)?,
                });
            tiles.optimize_indices()?;
            let mut target = PtxTarget::sm80();
            target.compute_capability = (8, 9);
            let actual = PtxGraph::from(tiles).with_target(target).to_ptx();
            let file = std::path::Path::new(&directory)
                .join(format!("union-q{query_tile}-dump{dump}.ptx"));
            let expected = std::fs::read_to_string(&file)?;
            ensure!(
                actual == expected,
                "dense PTX changed for {} (first differing byte {:?})",
                file.display(),
                actual
                    .bytes()
                    .zip(expected.bytes())
                    .position(|(a, b)| a != b)
            );
        }
    }
    Ok(())
}
