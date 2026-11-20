use super::*;
#[path = "gpu.rs"]
mod gpu;
use crate::{
    graph::TensorGraph,
    ptx::{PtxGraph, PtxTarget},
};
fn source(ir: TileIR) -> Result<String> {
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles
        .region_kernels
        .push(crate::tile::graph::TileRegionKernel { region_id: 0, ir });
    tiles.optimize_indices()?;
    Ok(PtxGraph::from(tiles)
        .with_target(PtxTarget::sm80())
        .to_ptx())
}
fn walk(block: &Block, visit: &mut impl FnMut(&Stmt)) {
    for s in &block.stmts {
        visit(s);
        if let Stmt::If { body, .. } | Stmt::ForLoop { body, .. } = s {
            walk(body, visit)
        }
    }
}
fn restore_row(p: &RowProof) -> Result<Block> {
    let mut b = p.rewritten.clone();
    fn undo(b: &mut Block, p: &RowProof) -> Result<()> {
        for s in &mut b.stmts {
            if let Stmt::If { body, .. } = s {
                undo(body, p)?;
                continue;
            }
            for j in 0..2 {
                let Stmt::LoadScalar { dest, .. } = &p.reads[j] else {
                    bail!("old read missing")
                };
                let replacement = Stmt::SetScalar {
                    dest: p.remap.var(*dest),
                    value: ScalarExpr::Var(p.carry[j]),
                };
                if format!("{s:?}") == format!("{replacement:?}") {
                    *s = p.remap.statement(&p.reads[j])?;
                    break;
                }
                let Stmt::StoreScalar { value, .. } = &p.writes[j] else {
                    bail!("old write missing")
                };
                let replacement = Stmt::SetScalar {
                    dest: p.carry[j],
                    value: ScalarExpr::Var(p.remap.var(*value)),
                };
                if format!("{s:?}") == format!("{replacement:?}") {
                    *s = p.remap.statement(&p.writes[j])?;
                    break;
                }
            }
        }
        Ok(())
    }
    undo(&mut b, p)?;
    p.remap.inverse().block(&b)
}
#[test]
fn online_register_stats_global_entry_keeps_exact_default_ir_and_ptx() -> Result<()> {
    for shape in [Shape::new(8, 8)?, Shape::new(32, 8)?] {
        for layout in [ReadLayout::HeadMajor, ReadLayout::SequenceMajor] {
            for transport in [Transport::Scalar, Transport::Async16] {
                let old = ir_with_transport(shape, layout, transport)?;
                let explicit =
                    ir_with_statistics(shape, layout, transport, StatisticsStorage::Global)?;
                assert_eq!(format!("{old:?}"), format!("{explicit:?}"));
                assert_eq!(source(old)?, source(explicit)?);
            }
        }
    }
    Ok(())
}
#[test]
fn online_register_stats_inverse_preserves_every_row_math_rounding_and_uniform_barrier()
-> Result<()> {
    for layout in [ReadLayout::HeadMajor, ReadLayout::SequenceMajor] {
        for transport in [Transport::Scalar, Transport::Async16] {
            let old = ir_with_transport(Shape::new(8, 8)?, layout, transport)?;
            let (candidate, proof) = build(old.clone())?;
            candidate.validate_layouts()?;
            let entry = ir_with_statistics(
                Shape::new(8, 8)?,
                layout,
                transport,
                StatisticsStorage::Registers,
            )?;
            assert_eq!(format!("{candidate:?}"), format!("{entry:?}"));

            assert_eq!(candidate.shared_mem_bytes, old.shared_mem_bytes);
            assert_eq!(
                format!("{:?}", candidate.params),
                format!("{:?}", old.params)
            );
            assert_eq!(proof.rows.len(), 8);
            assert_eq!(proof.outputs.len(), 16);
            assert_eq!(proof.carries.len(), 16);
            // Bind inverse proofs to the actual complete candidate, not just
            // detached copies of row/output blocks returned as metadata.
            let old_keys = old
                .body
                .stmts
                .iter()
                .find_map(|s| {
                    if let Stmt::ForLoop { loop_var, body, .. } = s {
                        loop_var.starts_with("causal_keys").then_some(body)
                    } else {
                        None
                    }
                })
                .unwrap();
            let row_start=old_keys.stmts.iter().position(|s|matches!(s,Stmt::ForLoop {loop_var,..}if loop_var.starts_with("online_rows"))).unwrap();
            let key_position=candidate.body.stmts.iter().position(|s|matches!(s,Stmt::ForLoop {loop_var,..}if loop_var.starts_with("causal_keys"))).unwrap();
            let Stmt::ForLoop {
                body: actual_keys, ..
            } = &candidate.body.stmts[key_position]
            else {
                panic!("key loop missing")
            };
            let joined = proof
                .rows
                .iter()
                .flat_map(|p| p.rewritten.stmts.clone())
                .collect::<Vec<_>>();
            assert_eq!(
                format!(
                    "{:?}",
                    &actual_keys.stmts[row_start..row_start + joined.len()]
                ),
                format!("{joined:?}")
            );
            let mut cursor = key_position + 1;
            for round in 0..8 {
                let Stmt::If { condition, body } = &candidate.body.stmts[cursor] else {
                    panic!("statistics leader export missing")
                };
                assert_eq!(
                    *condition,
                    ScalarPredicate::IndexLt {
                        lhs: Expr::ThreadIdx(Dim::X),
                        rhs: ix(1)
                    }
                );
                assert_eq!(body.stmts.len(), 2);
                for (slot, s) in body.stmts.iter().enumerate() {
                    let Stmt::StoreScalar {
                        target: ScalarMemory::Global(name),
                        value,
                        ..
                    } = s
                    else {
                        panic!("statistics export changed")
                    };
                    assert_eq!(name, "statistics");
                    assert_eq!(*value, proof.carries[round * 2 + slot].var);
                }
                cursor += 1;
                let Stmt::WarpBroadcastLeader { dest, src } = &candidate.body.stmts[cursor] else {
                    panic!("denominator broadcast missing")
                };
                assert_eq!(*src, proof.carries[round * 2 + 1].var);
                assert_eq!(*dest, proof.outputs[round * 2].broadcast);
                assert_eq!(*dest, proof.outputs[round * 2 + 1].broadcast);
                cursor += 1;
                for half in 0..2 {
                    let block = &proof.outputs[round * 2 + half].rewritten;
                    assert_eq!(
                        format!(
                            "{:?}",
                            &candidate.body.stmts[cursor..cursor + block.stmts.len()]
                        ),
                        format!("{:?}", block.stmts)
                    );
                    cursor += block.stmts.len();
                }
            }
            assert_eq!(cursor, candidate.body.stmts.len());
            for p in &proof.rows {
                assert_eq!(
                    format!("{:?}", restore_row(p)?),
                    format!("{:?}", p.original)
                );
                assert_eq!(
                    p.rewritten
                        .stmts
                        .iter()
                        .filter(|s| matches!(s, Stmt::Barrier))
                        .count(),
                    1
                );
                assert_eq!(
                    p.rewritten
                        .stmts
                        .iter()
                        .filter(|s| matches!(s, Stmt::WarpBroadcastLeader { .. }))
                        .count(),
                    4
                );
            }
            for p in &proof.outputs {
                let mut output = p.rewritten.clone();
                let old = p.remap.statement(&p.denominator)?;
                let Stmt::LoadScalar { dest, .. } = old else {
                    bail!("old denominator absent")
                };
                let replacement = Stmt::SetScalar {
                    dest,
                    value: ScalarExpr::Var(p.broadcast),
                };
                let mut count = 0;
                for s in &mut output.stmts {
                    if format!("{s:?}") == format!("{replacement:?}") {
                        *s = old.clone();
                        count += 1;
                    }
                }
                assert_eq!(count, 1);
                assert_eq!(
                    format!("{:?}", p.remap.inverse().block(&output)?),
                    format!("{:?}", p.original)
                );
            }
            let mut reads = 0;
            let mut writes = 0;
            let mut online_loops = 0;
            walk(&candidate.body, &mut |s| match s {
                Stmt::LoadScalar {
                    source: ScalarMemory::Global(p),
                    ..
                } if p == "statistics" => reads += 1,
                Stmt::StoreScalar {
                    target: ScalarMemory::Global(p),
                    ..
                } if p == "statistics" => writes += 1,
                Stmt::ForLoop {
                    loop_var, carries, ..
                } if loop_var.starts_with("causal_keys") => {
                    assert_eq!(carries.len(), 16);
                }
                Stmt::ForLoop { loop_var, .. }
                    if loop_var.starts_with("online_rows")
                        || loop_var.starts_with("store_output")
                        || loop_var.starts_with("initialize_statistics") =>
                {
                    online_loops += 1
                }
                _ => {}
            });
            assert_eq!(reads, 0);
            assert_eq!(writes, 16);
            assert_eq!(online_loops, 0);
            for pair in proof.carries.chunks_exact(2) {
                assert_eq!(pair[0].initial, ScalarExpr::Constant(f32::NEG_INFINITY));
                assert_eq!(pair[1].initial, ScalarExpr::Constant(0.));
            }
        }
    }
    Ok(())
}
fn eval(e: &Expr, block: usize, head: usize, warp: usize, lane: usize) -> usize {
    match e {
        Expr::Const(v) => usize::try_from(*v).unwrap(),
        Expr::BlockIdx(Dim::X) => block,
        Expr::BlockIdx(Dim::Y) => head,
        Expr::ThreadIdx(Dim::Y) => warp,
        Expr::ThreadIdx(Dim::X) => lane,
        Expr::Add(a, b) => eval(a, block, head, warp, lane) + eval(b, block, head, warp, lane),
        Expr::Sub(a, b) => eval(a, block, head, warp, lane) - eval(b, block, head, warp, lane),
        Expr::Mul(a, b) => eval(a, block, head, warp, lane) * eval(b, block, head, warp, lane),
        Expr::FloorDiv(a, n) => eval(a, block, head, warp, lane) / n,
        Expr::Mod(a, n) => eval(a, block, head, warp, lane) % n,
        other => panic!("unexpected coverage expression {other:?}"),
    }
}
#[test]
fn online_register_stats_complete_owned_output_indices_and_slots_zero_one_once() -> Result<()> {
    let (ir, proof) = build(ir_with_transport(
        Shape::new(8, 8)?,
        ReadLayout::SequenceMajor,
        Transport::Async16,
    )?)?;
    let mut exports = Vec::new();
    walk(&ir.body, &mut |s| {
        if let Stmt::StoreScalar {
            target: ScalarMemory::Global(p),
            index,
            ..
        } = s
            && p == "statistics"
        {
            exports.push(index.clone());
        }
    });
    assert_eq!(exports.len(), 16);
    for bh in [0, 63] {
        for block in 0..8 {
            let mut outputs = vec![0u8; 4096];
            let mut stats = [0u8; 64 * 3];
            for p in &proof.outputs {
                let Stmt::StoreScalar {
                    target: ScalarMemory::Global(name),
                    index,
                    ..
                } = p.rewritten.stmts.last().unwrap()
                else {
                    panic!("output store absent")
                };
                assert_eq!(name, "output_0");
                for warp in 0..8 {
                    for lane in 0..32 {
                        let index = eval(index, block, bh, warp, lane);
                        let local = index - (bh * 512 + block * 64) * 64;
                        assert!(local < 4096);
                        outputs[local] += 1;
                    }
                }
            }
            for index in &exports {
                for warp in 0..8 {
                    let local = eval(index, block, bh, warp, 0) - (bh * 512 + block * 64) * 3;
                    assert!(local < 192);
                    stats[local] += 1;
                }
            }
            assert!(outputs.iter().all(|x| *x == 1));
            assert!(
                stats
                    .iter()
                    .enumerate()
                    .all(|(i, x)| *x == u8::from(i % 3 != 2))
            );
        }
    }
    Ok(())
}

#[test]
fn online_register_stats_rejects_changed_causal_extent_and_statistics_contract() -> Result<()> {
    let old = ir_with_transport(
        Shape::new(8, 8)?,
        ReadLayout::SequenceMajor,
        Transport::Async16,
    )?;
    let mut wrong_dtype = old.clone();
    wrong_dtype.params[4].dtype = DType::BF16;
    assert!(build(wrong_dtype).is_err());
    let mut changed = old.clone();
    changed.shared_mem_bytes += 512;
    assert!(build(changed).is_err());
    let mut changed = old.clone();
    for s in &mut changed.body.stmts {
        if let Stmt::ForLoop { loop_var, end, .. } = s
            && loop_var.starts_with("causal_keys")
        {
            *end = Expr::BlockIdx(Dim::X) + ix(2);
        }
    }
    assert!(build(changed).is_err());
    let mut changed = old;
    fn move_first_statistics_read(b: &mut Block) -> bool {
        for s in &mut b.stmts {
            match s {
                Stmt::LoadScalar {
                    source: ScalarMemory::Global(p),
                    index,
                    ..
                } if p == "statistics" => {
                    *index = index.clone() + ix(3);
                    return true;
                }
                Stmt::If { body, .. } | Stmt::ForLoop { body, .. } => {
                    if move_first_statistics_read(body) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }
    assert!(move_first_statistics_read(&mut changed.body));
    assert!(build(changed).is_err());
    Ok(())
}
