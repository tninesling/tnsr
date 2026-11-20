use super::*;
use crate::tile::{Dim, Expr, IndexExpr, LoopCarry, MemorySpace, TileLayout};
fn target() -> PtxTarget {
    let mut t = PtxTarget::sm80();
    t.compute_capability = (8, 9);
    t.matmul_resources = MatMulResources {
        shared_bytes_per_sm: 100 * 1024,
        threads_per_sm: 1536,
        ..MatMulResources::default()
    };
    t
}
fn eval(e: &Expr, thread: usize, bx: usize, by: usize, round: usize) -> usize {
    let dim = |d: &Dim, x, y, z| match d {
        Dim::X => x,
        Dim::Y => y,
        Dim::Z => z,
    };
    match e {
        Expr::Const(v) => *v as usize,
        Expr::Var(_) => round,
        Expr::ThreadIdx(d) => dim(d, thread % 32, thread / 32, 0),
        Expr::BlockIdx(d) => dim(d, bx, by, 0),
        Expr::BlockDim(d) => dim(d, 32, 4, 1),
        Expr::Add(a, b) => {
            eval(a, thread, bx, by, round).wrapping_add(eval(b, thread, bx, by, round))
        }
        Expr::Sub(a, b) => {
            eval(a, thread, bx, by, round).wrapping_sub(eval(b, thread, bx, by, round))
        }
        Expr::Mul(a, b) => {
            eval(a, thread, bx, by, round).wrapping_mul(eval(b, thread, bx, by, round))
        }
        Expr::FloorDiv(a, b) => eval(a, thread, bx, by, round) / b,
        Expr::Mod(a, b) => eval(a, thread, bx, by, round) % b,
        Expr::ShiftRight(a, b) => eval(a, thread, bx, by, round) >> b,
        Expr::BitAnd(a, b) => eval(a, thread, bx, by, round) & (*b as usize),
    }
}
fn unpatch(b: &mut Block) {
    b.stmts.retain(|s| {
        !matches!(s,Stmt::LoadScalar{source:ScalarMemory::Global(n),..}if n=="input_2")
            && !matches!(
                s,
                Stmt::SetScalar {
                    value: ScalarExpr::Binary {
                        op: ScalarBinaryOp::AddRn,
                        ..
                    },
                    ..
                }
            )
    });
    for s in &mut b.stmts {
        if let Stmt::ForLoop { body, .. } = s {
            unpatch(body);
        }
    }
}
#[test]
fn fused_nt128_four_wide_exact_contract_full_coverage_bias_and_retirement() -> Result<()> {
    for c in CASES.into_iter().filter(|c| !c.split) {
        let r = region(c, target())?;
        let before = format!("{r:?}");
        assert_eq!(ir::bias_slot(&r)?, 2);
        let (candidate, s, plain) = ir::candidate(&r, target())?;
        assert_eq!(before, format!("{r:?}"));
        assert_eq!(
            s.block_tile,
            MatMulTile {
                m: 128,
                n: 128,
                k: 32
            }
        );
        assert_eq!(s.block_threads, (32, 4, 1));
        assert_eq!(s.warp_topology, (2, 2));
        assert_eq!(s.warp_fragments, (4, 4));
        assert_eq!(s.fragment_load, MatMulFragmentLoad::StreamB);
        assert_eq!(
            s.accumulator_exchange,
            MatMulAccumulatorExchange::FragmentWindows {
                fragment_rows: 2,
                fragment_cols: 4
            }
        );
        assert_eq!(candidate.shared_mem_bytes, 40960);
        assert_eq!(s.copy_bytes, 16);
        assert_eq!(s.async_operands, (true, true));
        assert_eq!(s.pipeline_stages, 2);
        let mut erased = candidate.body.clone();
        unpatch(&mut erased);
        assert_eq!(format!("{erased:?}"), format!("{:?}", plain.body));
        let first = candidate
            .body
            .stmts
            .iter()
            .position(|s| matches!(s, Stmt::StoreAccumulatorWindow { .. }))
            .context("no windows")?;
        let wait = candidate.body.stmts[..first]
            .iter()
            .rposition(|s| matches!(s, Stmt::AsyncWait))
            .context("no drain")?;
        assert!(matches!(candidate.body.stmts[wait + 1], Stmt::Barrier));
        assert_eq!(
            candidate.body.stmts[first..]
                .iter()
                .filter(|s| matches!(s, Stmt::Barrier))
                .count(),
            3
        );
        let mut seen = vec![false; 128 * 128];
        let mut loops = 0;
        for stmt in &candidate.body.stmts {
            if let Stmt::ForLoop {
                loop_var,
                start,
                end,
                body,
                ..
            } = stmt
            {
                if !loop_var.starts_with("square_output_round_") {
                    continue;
                }
                loops += 1;
                assert_eq!(*start, Expr::Const(0));
                assert_eq!(*end, Expr::Const(64));
                let store = body
                    .stmts
                    .iter()
                    .position(|s| matches!(s, Stmt::StoreGlobalPredicated { .. }))
                    .unwrap();
                let Stmt::StoreGlobalPredicated { src, row, col, .. } = &body.stmts[store] else {
                    unreachable!()
                };
                let Stmt::SetScalar {
                    dest,
                    value:
                        ScalarExpr::Binary {
                            op: ScalarBinaryOp::AddRn,
                            lhs,
                            rhs,
                        },
                } = &body.stmts[store - 1]
                else {
                    panic!("single explicit RN bias addition missing")
                };
                assert_eq!(dest, src);
                assert_eq!(**lhs, ScalarExpr::Var(*src));
                let Stmt::LoadScalar {
                    dest: bias,
                    source: ScalarMemory::Global(name),
                    index,
                } = &body.stmts[store - 2]
                else {
                    panic!("typed bias load missing")
                };
                assert_eq!(name, "input_2");
                assert_eq!(**rhs, ScalarExpr::Var(*bias));
                assert_eq!(index, col);
                assert_eq!(
                    body.stmts
                        .iter()
                        .filter(|s| matches!(s, Stmt::SetScalar { .. }))
                        .count(),
                    1
                );
                for round in 0..64 {
                    for thread in 0..128 {
                        let rr = eval(row, thread, 0, 0, round);
                        let cc = eval(col, thread, 0, 0, round);
                        assert!(rr < 128 && cc < 128);
                        assert!(!seen[rr * 128 + cc]);
                        seen[rr * 128 + cc] = true;
                        let last_col = eval(index, thread, c.n / 128 - 1, c.m / 128 - 1, round);
                        assert!(last_col < c.n);
                        assert_eq!(
                            last_col,
                            eval(col, thread, c.n / 128 - 1, c.m / 128 - 1, round)
                        );
                    }
                }
            }
        }
        assert_eq!(loops, 2);
        assert!(seen.into_iter().all(|x| x));
    }
    Ok(())
}
#[test]
fn fused_nt128_four_wide_rejects_wrong_epilogue_dtype_offsets_layouts_and_tails() -> Result<()> {
    let r = region(CASES[0], target())?;
    let mut bad = r.clone();
    if let RegionOpKind::Binary { lhs, rhs, .. } = &mut bad.epilogue_operations[0].kind {
        std::mem::swap(lhs, rhs);
    }
    assert!(ir::bias_slot(&bad).is_err());
    let mut bad = r.clone();
    bad.inputs[2].tensor.dtype = DType::F32;
    assert!(ir::bias_slot(&bad).is_err());
    let mut bad = r.clone();
    bad.inputs[2].tensor.access.results = vec![IndexExpr::IterDim(0)];
    assert!(ir::bias_slot(&bad).is_err());
    let mut bad = r.clone();
    bad.inputs[2].tensor.access.results = vec![IndexExpr::Add(
        Box::new(IndexExpr::IterDim(1)),
        Box::new(IndexExpr::Const(1)),
    )];
    assert!(ir::bias_slot(&bad).is_err());
    let mut bad = r.clone();
    bad.inputs[1].source_shape = vec![bad.k, bad.n];
    assert!(ir::bias_slot(&bad).is_err());
    let mut bad = r.clone();
    bad.epilogue_operations
        .push(bad.epilogue_operations[0].clone());
    assert!(ir::bias_slot(&bad).is_err());
    let mut bad = r.clone();
    bad.m += 1;
    assert!(ir::bias_slot(&bad).is_err());
    assert!(
        crate::tile::large_tile_policy::square128_streamed::region_candidate(
            &r,
            MatMulSharedLayout::Padded,
            MatMulPipeline::Auto,
            true,
            target().matmul_resources
        )
        .is_none()
    );
    Ok(())
}
#[test]
fn fused_nt128_four_wide_fresh_ids_cover_recursive_defs_uses_and_reject_unknown() -> Result<()> {
    let schedule = region(CASES[0], target())?.schedule;
    let body = Block {
        stmts: vec![
            Stmt::AllocTile {
                var: TileVar(1),
                space: MemorySpace::Register,
                layout: TileLayout::ThreadScalar,
                dtype: DType::F32,
                rows: 1,
                cols: 1,
            },
            Stmt::ForLoop {
                loop_var: "x".into(),
                start: Expr::Const(0),
                end: Expr::Const(1),
                carries: vec![LoopCarry {
                    var: TileVar(400),
                    initial: ScalarExpr::Var(TileVar(600)),
                }],
                body: Block {
                    stmts: vec![
                        Stmt::MatMul {
                            dest: TileVar(1000),
                            a: TileVar(2000),
                            b: TileVar(3000),
                            layout: crate::tile::MatMulLayout::NN,
                            schedule,
                        },
                        Stmt::If {
                            condition: ScalarPredicate::Equal {
                                lhs: ScalarExpr::Var(TileVar(4000)),
                                rhs: ScalarExpr::Constant(0.),
                            },
                            body: Block {
                                stmts: vec![Stmt::StoreScalar {
                                    target: ScalarMemory::Shared(TileVar(5000)),
                                    index: Expr::Const(0),
                                    value: TileVar(9000),
                                }],
                            },
                        },
                    ],
                },
            },
        ],
    };
    assert_eq!(ir::fresh_variable(&body)?, TileVar(9001));
    assert!(
        ir::fresh_variable(&Block {
            stmts: vec![Stmt::Embedding {
                vocabulary: 1,
                width: 1,
                index_count: 1
            }]
        })
        .is_err()
    );
    assert!(
        ir::fresh_variable(&Block {
            stmts: vec![Stmt::Zero {
                tile: TileVar(usize::MAX)
            }]
        })
        .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "manual CPU proof requires the retained qualified eight-warp loaded PTX directory"]
fn fused_nt128_four_wide_preserves_qualified_eight_warp_loaded_ptx() -> Result<()> {
    let golden = PathBuf::from(
        std::env::var_os("TNSR_FUSED_NT128_EIGHT_GOLDEN")
            .context("set PRECHANGE qualified eight-warp loaded PTX directory")?,
    );
    for c in CASES.into_iter().filter(|c| !c.split) {
        let r = region(c, target())?;
        let (ir, _, _) = ir::qualified_eight_warp_candidate(&r, target())?;
        let mut graph = TileGraph::from_graph(&TensorGraph::<f32>::new());
        graph.add_matmul_regions(&[r])?;
        graph.matmul_region_kernels[0].ir = ir;
        graph.optimize_indices()?;
        let actual = PtxGraph::from(graph).with_target(target()).module_source();
        let expected = std::fs::read_to_string(golden.join(c.name).join("candidate.ptx"))?;
        assert_eq!(
            actual, expected,
            "qualified eight-warp PTX changed for {}",
            c.name
        );
    }
    Ok(())
}
