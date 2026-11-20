use super::*;
#[path = "gpu.rs"]
mod gpu;
fn spans(len: usize, members: usize) -> Vec<[Span; 10]> {
    let sizes = [
        len * 4,
        len * 2,
        len * 4,
        len * 4,
        40,
        len * 4,
        len * 4,
        len * 4,
        len / 64,
        len * 2,
    ];
    (0..members)
        .map(|member| {
            std::array::from_fn(|index| Span {
                address: (member * 10 + index + 1) * 0x04000000,
                bytes: sizes[index],
            })
        })
        .collect()
}
fn eval(index: &Expr, block: usize, thread: usize) -> usize {
    match index {
        Expr::Const(v) => usize::try_from(*v).unwrap(),
        Expr::BlockIdx(Dim::X) => block,
        Expr::ThreadIdx(Dim::X) => thread,
        Expr::BlockDim(Dim::X) => 256,
        Expr::Add(a, b) => eval(a, block, thread) + eval(b, block, thread),
        Expr::Sub(a, b) => eval(a, block, thread) - eval(b, block, thread),
        Expr::Mul(a, b) => eval(a, block, thread) * eval(b, block, thread),
        Expr::FloorDiv(a, n) => eval(a, block, thread) / n,
        Expr::Mod(a, n) => eval(a, block, thread) % n,
        Expr::ShiftRight(a, n) => eval(a, block, thread) >> n,
        Expr::BitAnd(a, n) => eval(a, block, thread) & usize::try_from(*n).unwrap(),
        other => panic!("unexpected batch coverage index {other:?}"),
    }
}
fn selected(statement: &Stmt, block: usize) -> bool {
    let Stmt::If {
        condition: ScalarPredicate::IndexLt { lhs, rhs },
        body,
    } = statement
    else {
        panic!("missing upper region");
    };
    let [
        Stmt::If {
            condition:
                ScalarPredicate::IndexLt {
                    lhs: lower,
                    rhs: upper,
                },
            ..
        },
    ] = body.stmts.as_slice()
    else {
        panic!("missing lower region");
    };
    eval(lhs, block, 0) < eval(rhs, block, 0) && eval(lower, block, 0) < eval(upper, block, 0)
}
#[test]
fn paired_vector_batch_inverse_keeps_every_arithmetic_rounding_and_health_statement() -> Result<()>
{
    for len in [1024usize, 65536, 1_048_576, 2_818_048, 3_145_728] {
        let old = paired_ir(len, AdamKind::Vector4)?;
        let stride = stride(&old.body)?;
        for members in [1usize, 4] {
            let batch = ir(len, members)?;
            assert_eq!(batch.params.len(), 10 * members);
            assert_eq!(batch.shared_mem_bytes, 32);
            assert_eq!(batch.body.stmts.len(), members + 1);
            for member in 0..members {
                let inverse = Rewrite {
                    member,
                    lower: member * (len / 1024),
                    offset: 1 + member * stride,
                    inverse: true,
                };
                let restored = inverse.block(region_body(&batch.body.stmts[member + 1])?)?;
                assert_eq!(format!("{restored:?}"), format!("{:?}", old.body));
                for (actual, original) in batch.params[member * 10..(member + 1) * 10]
                    .iter()
                    .zip(&old.params)
                {
                    assert_eq!(inverse.param(&actual.name)?, original.name);
                    assert_eq!(actual.dtype, original.dtype);
                    assert_eq!(actual.is_input, original.is_input);
                }
            }
        }
    }
    for (len, members) in [
        (0, 1),
        (1023, 4),
        (1025, 1),
        (1024, 0),
        (1024, 5),
        (usize::MAX, 4),
    ] {
        assert!(ir(len, members).is_err());
    }
    Ok(())
}
#[test]
fn paired_vector_batch_cta_regions_cover_each_member_and_health_group_once() -> Result<()> {
    for len in [1024usize, 65536, 1_048_576, 2_818_048, 3_145_728] {
        let batch = ir(len, 4)?;
        let mut seen = vec![vec![0u8; len]; 4];
        let mut groups = vec![vec![0u8; len / 256]; 4];
        let regions = &batch.body.stmts[1..];
        for block in 0..(len / 1024) * 4 {
            let owners = regions
                .iter()
                .enumerate()
                .filter(|(_, s)| selected(s, block))
                .collect::<Vec<_>>();
            assert_eq!(owners.len(), 1);
            let (member, region) = owners[0];
            let body = region_body(region)?;
            let group_index = body
                .stmts
                .iter()
                .find_map(|s| {
                    if let Stmt::StoreGlobalF32x4 {
                        param, group_index, ..
                    } = s
                        && param == &format!("next_weight_{member}")
                    {
                        Some(group_index)
                    } else {
                        None
                    }
                })
                .unwrap();
            for thread in 0..256 {
                let base = eval(group_index, block, thread) * 4;
                for lane in 0..4 {
                    let element = base + lane;
                    assert!(element < len);
                    seen[member][element] += 1;
                    assert_eq!(
                        element / 256,
                        (block - member * (len / 1024)) * 4 + thread / 64
                    );
                }
                if thread % 64 == 0 {
                    groups[member][base / 256] += 1;
                }
            }
        }
        assert!(seen.into_iter().flatten().all(|n| n == 1));
        assert!(groups.into_iter().flatten().all(|n| n == 1));
        assert!(!regions.iter().any(|s| selected(s, (len / 1024) * 4)));
    }
    Ok(())
}
#[test]
fn paired_vector_batch_full_abi_alignment_and_cross_parameter_write_disjointness() {
    let len = 65536;
    for count in [1usize, 4] {
        let good = spans(len, count);
        assert!(admitted(len, &good));
        for member in 0..count {
            for operand in 0..10 {
                let mut bad = good.clone();
                bad[member][operand].bytes -= 1;
                assert!(!admitted(len, &bad));
                bad = good.clone();
                bad[member][operand].address += 1;
                assert!(!admitted(len, &bad));
            }
            for output in 5..10 {
                for other in 0..count {
                    for operand in 0..10 {
                        if member == other && output == operand {
                            continue;
                        }
                        let mut bad = good.clone();
                        bad[member][output].address = bad[other][operand].address;
                        assert!(!admitted(len, &bad));
                    }
                }
            }
        }
        // Same-group settings can be shared readonly. Writes cannot overlap it.
        let mut shared = good.clone();
        for member in 1..count {
            shared[member][4] = shared[0][4];
        }
        assert!(admitted(len, &shared));
    }
    assert!(!admitted(len, &[]));
    assert!(!admitted(len, &spans(len, 5)));
}
#[test]
fn paired_vector_batch_ptx_has_root_40_pointer_abi_and_one_shared_scratch() -> Result<()> {
    use crate::{
        graph::TensorGraph,
        ptx::{PtxGraph, PtxTarget},
        tile::TileGraph,
    };
    let batch = ir(2_818_048, 4)?;
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles
        .region_kernels
        .push(crate::tile::graph::TileRegionKernel {
            region_id: 0,
            ir: batch,
        });
    tiles.optimize_indices()?;
    let source = PtxGraph::from(tiles)
        .with_target(PtxTarget::sm80())
        .module_source();
    assert_eq!(source.matches(".param .u64").count(), 40);
    assert_eq!(source.matches("ld.param.u64").count(), 40);
    assert_eq!(source.matches("st.global.v4.f32").count(), 12);
    assert_eq!(source.matches("cvt.rn.bf16.f32").count(), 16);
    assert!(!source.contains("call."));
    assert!(!source.contains(".local"));
    Ok(())
}
