use super::*;
fn target() -> PtxTarget {
    let mut t = PtxTarget::sm80();
    t.compute_capability = (8, 9);
    t.matmul_resources = crate::tile::MatMulResources {
        shared_bytes_per_sm: 100 * 1024,
        threads_per_sm: 1536,
        ..crate::tile::MatMulResources::default()
    };
    t
}
fn eval(e: &Expr, x: usize, y: usize, bx: usize, by: usize, kt: usize) -> usize {
    match e {
        Expr::Const(v) => usize::try_from(*v).unwrap(),
        Expr::ThreadIdx(Dim::X) => x,
        Expr::ThreadIdx(Dim::Y) => y,
        Expr::BlockIdx(Dim::X) => bx,
        Expr::BlockIdx(Dim::Y) => by,
        Expr::Var(n) if n == "k_tile" => kt,
        Expr::Add(a, b) => eval(a, x, y, bx, by, kt) + eval(b, x, y, bx, by, kt),
        Expr::Sub(a, b) => eval(a, x, y, bx, by, kt) - eval(b, x, y, bx, by, kt),
        Expr::Mul(a, b) => eval(a, x, y, bx, by, kt) * eval(b, x, y, bx, by, kt),
        Expr::FloorDiv(a, b) => eval(a, x, y, bx, by, kt) / *b,
        Expr::Mod(a, b) => eval(a, x, y, bx, by, kt) % *b,
        Expr::ShiftRight(a, b) => eval(a, x, y, bx, by, kt) >> b,
        Expr::BitAnd(a, b) => eval(a, x, y, bx, by, kt) & usize::try_from(*b).unwrap(),
        _ => panic!("unexpected bounded NN expression {e:?}"),
    }
}
#[test]
fn nn_rectangular_actual_rank3_whole_volume_flatten_is_exact_and_rejects_other_maps() -> Result<()>
{
    let r = fixture(target())?;
    assert_eq!(r.inputs[0].source_shape, [32, 512, K]);
    let a = &r.inputs[0].tensor;
    // Symbolic normalized-map equality in guard proves every coordinate; row
    // boundary enumeration independently checks the actual plan's rank3 map.
    for row in 0..M {
        for k in [0, K - 1] {
            assert_eq!(a.source_index(&[row, k])?, [row / 512, row % 512, k]);
        }
    }
    let mut wrong = r.clone();
    wrong.inputs[0].tensor.access.results[0] = crate::tile::IndexExpr::Const(0);
    assert!(guard(&wrong).is_err());
    let mut wrong = r.clone();
    wrong.inputs[0].tensor.dtype = DType::F32;
    assert!(guard(&wrong).is_err());
    let mut wrong = r.clone();
    wrong.inputs[0].source_shape = vec![16, 1024, K];
    assert!(guard(&wrong).is_err());
    let mut wrong = r.clone();
    wrong.k = K - 1;
    assert!(guard(&wrong).is_err());
    assert_eq!(r.schedule.shared_mem_bytes(), 19456);
    Ok(())
}
#[test]
fn nn_rectangular_x4_trans_provider_register_identity_alignment_and_shared_bounds() -> Result<()> {
    let r = fixture(target())?;
    let (_, r) = template(&r, target())?;
    for lhs in [true, false] {
        let layout = if lhs {
            r.schedule.operand_layouts.0
        } else {
            r.schedule.operand_layouts.1
        };
        assert!(!layout.column_major);
        assert_eq!(layout.row_stride, if lhs { 40 } else { 72 });
        let bytes = if lhs { 10240 } else { 4608 };
        for warp in 0..8 {
            for k in [0, 16] {
                for fragment in 0..2 {
                    let (row, col) = if lhs {
                        crate::ptx::graph::bf16_ldmatrix_provider_expr(true, fragment, k)
                    } else {
                        crate::ptx::graph::bf16_ldmatrix_nn_rect_provider_expr(fragment, k)
                    };
                    let p = std::array::from_fn::<_, 32, _>(|l| {
                        (eval(&row, l, warp, 0, 0, 0), eval(&col, l, warp, 0, 0, 0))
                    });
                    for &(rr, cc) in &p {
                        assert!(rr < if lhs { 128 } else { 32 });
                        assert!(cc + 7 < if lhs { 32 } else { 64 });
                        let offset = layout.offset(rr, cc) * 2;
                        assert_eq!(offset % 16, 0);
                        assert!(offset + 16 <= bytes);
                        for stage in [0, 14848] {
                            let start = stage + if lhs { 0 } else { 10240 };
                            assert_eq!((start + offset) % 16, 0);
                            assert!(start + offset + 16 <= SHARED_RECT);
                        }
                    }
                    let mut seen = [false; 256];
                    for lane in 0..32 {
                        for reg in 0..4 {
                            for half in 0..2 {
                                let actual = if lhs {
                                    let p = p[reg * 8 + lane / 4];
                                    (p.0, p.1 + 2 * (lane & 3) + half)
                                } else {
                                    let p = p[reg * 8 + 2 * (lane & 3) + half];
                                    (p.0, p.1 + lane / 4)
                                };
                                let expected = if lhs {
                                    (
                                        (warp / 2) * 32 + fragment * 16 + 8 * (reg % 2) + lane / 4,
                                        k + 8 * (reg / 2) + 2 * (lane & 3) + half,
                                    )
                                } else {
                                    (
                                        k + 8 * (reg % 2) + 2 * (lane & 3) + half,
                                        (warp % 2) * 32 + fragment * 16 + 8 * (reg / 2) + lane / 4,
                                    )
                                };
                                assert_eq!(
                                    actual, expected,
                                    "documented transposed B matches exact mma B element"
                                );
                                let local = if lhs {
                                    (actual.0 - (warp / 2) * 32 - fragment * 16, actual.1 - k)
                                } else {
                                    (actual.0 - k, actual.1 - (warp % 2) * 32 - fragment * 16)
                                };
                                assert!(!seen[local.0 * 16 + local.1]);
                                seen[local.0 * 16 + local.1] = true;
                            }
                        }
                    }
                    assert!(seen.into_iter().all(|v| v));
                }
            }
        }
    }
    Ok(())
}
#[test]
fn nn_rectangular_actual_copy16_coverage_sentinel_k16_order_and_final_drain() -> Result<()> {
    let r = fixture(target())?;
    let (old, selected) = template(&r, target())?;
    let ir = candidate(&r, target())?;
    let end = ir.body.stmts.len() - 1;
    let mut restored = ir.body.stmts[..end].to_vec();
    fn restore(stmts: &mut [Stmt], selected: &MatMulRegion) {
        for s in stmts {
            match s {
                Stmt::AllocSharedArena { bytes, .. } => *bytes = 32768,
                Stmt::AllocTile { layout, .. }
                    if *layout == TileLayout::Bf16MmaAccumulator128x64LdmatrixNn =>
                {
                    *layout = selected.schedule.accumulator_tile_layout()
                }
                Stmt::MatMul { schedule, .. } => {
                    schedule.fragment_load = MatMulFragmentLoad::PreloadAll
                }
                Stmt::ForLoop { body, .. } => restore(&mut body.stmts, selected),
                _ => {}
            }
        }
    }
    restore(&mut restored, &selected);
    assert_eq!(
        format!("{:?}", &old.body.stmts[..end]),
        format!("{restored:?}"),
        "transport/control prefix unchanged except typed fragments/streamed descriptor and dead exchange arena bytes"
    );
    let loop_index = ir
        .body
        .stmts
        .iter()
        .position(|s| matches!(s,Stmt::ForLoop{loop_var,..}if loop_var=="k_tile"))
        .unwrap();
    let Stmt::ForLoop {
        start,
        end,
        body: main,
        ..
    } = &ir.body.stmts[loop_index]
    else {
        unreachable!()
    };
    assert_eq!(start, &Expr::Const(0));
    assert_eq!(end, &Expr::Const(44));
    assert!(matches!(ir.body.stmts[loop_index + 1], Stmt::AsyncWait));
    assert!(matches!(ir.body.stmts[loop_index + 2], Stmt::Barrier));
    assert!(
        main.stmts
            .iter()
            .position(|s| matches!(s, Stmt::AsyncWait))
            .unwrap()
            < main
                .stmts
                .iter()
                .position(|s| matches!(s, Stmt::MatMul { .. }))
                .unwrap()
    );
    for (statements, is_loop) in [
        (&ir.body.stmts[..loop_index], false),
        (main.stmts.as_slice(), true),
    ] {
        for (param, rows, cols) in [("input_0", 128, 32), ("input_1", 32, 64)] {
            let copies = statements
                .iter()
                .filter_map(|s| {
                    if let Stmt::AsyncCopy {
                        src_param,
                        element_index,
                        row,
                        col,
                        tile_row,
                        tile_col,
                        layout,
                        copy_bytes,
                        ..
                    } = s
                    {
                        if src_param == param {
                            Some((
                                element_index,
                                row,
                                col,
                                tile_row,
                                tile_col,
                                layout,
                                *copy_bytes,
                            ))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(
                copies.len(),
                if param == "input_0" { 2 } else { 1 },
                "exact copy16 groups/thread/operand"
            );
            let mut hits = vec![0; rows * cols];
            for (index, row, col, tr, tc, layout, bytes) in copies {
                assert_eq!(bytes, 16);
                for y in 0..8 {
                    for x in 0..32 {
                        let rr = eval(tr, x, y, 0, 0, 0);
                        let cc = eval(tc, x, y, 0, 0, 0);
                        for h in 0..8 {
                            assert!(rr < rows && cc + h < cols);
                            hits[rr * cols + cc + h] += 1;
                        }
                        for (bx, by) in [(0, 0), (7, 127)] {
                            for kt in [0, 42, 43] {
                                let global_r = eval(row, x, y, bx, by, kt);
                                let global_c = eval(col, x, y, bx, by, kt);
                                let offset = eval(index, x, y, bx, by, kt);
                                assert_eq!(offset % 8, 0);
                                let valid = global_r < layout.rows && global_c + 7 < layout.cols;
                                let sentinel = is_loop && kt == 43;
                                assert_eq!(
                                    valid, !sentinel,
                                    "last speculative tile must zero-fill"
                                );
                                if valid {
                                    assert_eq!(offset, global_r * layout.cols + global_c);
                                    assert!(offset + 7 < layout.rows * layout.cols);
                                }
                            }
                        }
                    }
                }
            }
            assert!(hits.into_iter().all(|n| n == 1));
        }
    }
    let order = (0..44)
        .flat_map(|kt| [kt * 32, kt * 32 + 16])
        .collect::<Vec<_>>();
    assert_eq!(order, (0..K).step_by(16).collect::<Vec<_>>());
    let ptx = source(&r, target(), ir)?;
    assert_eq!(
        ptx.matches("ldmatrix.sync.aligned.m8n8.x4.shared.b16")
            .count(),
        4
    );
    assert_eq!(
        ptx.matches("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16")
            .count(),
        4
    );
    assert_eq!(
        ptx.matches("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
            .count(),
        16
    );
    assert_eq!(ptx.matches("cvt.rn.bf16.f32").count(), 32);
    assert_eq!(ptx.matches("st.global.b32").count(), 16);
    assert!(!ptx.contains("input_2"));
    assert!(!ptx.contains("add.rn.f32"));
    Ok(())
}

#[test]
fn nn_rectangular_actual_register_network_and_output_bijection() {
    for (bx, by) in [(0, 0), (7, 127)] {
        let mut hits = vec![false; 128 * 64];
        for warp in 0..8 {
            for rb in 0..4 {
                let mut words = std::array::from_fn::<_, 32, _>(|lane| {
                    std::array::from_fn::<_, 4, _>(|j| (rb * 8 + lane / 4, j * 8 + (lane & 3) * 2))
                });
                for bit in 0..2 {
                    let mask = 4usize << bit;
                    let q = 1usize << bit;
                    let old = words;
                    for lane in 0..32 {
                        for j0 in (0..4).filter(|j| j & q == 0) {
                            let j1 = j0 ^ q;
                            let p = lane & mask != 0;
                            let peer = if (lane ^ mask) & mask != 0 {
                                old[lane ^ mask][j0]
                            } else {
                                old[lane ^ mask][j1]
                            };
                            words[lane][j0] = if p { peer } else { old[lane][j0] };
                            words[lane][j1] = if p { old[lane][j1] } else { peer };
                        }
                    }
                }
                for (lane, word) in words.iter().enumerate() {
                    for (g, &(rr, cc)) in word.iter().enumerate() {
                        assert_eq!((rr, cc), (rb * 8 + (lane >> 4) * 4 + g, (lane & 15) * 2));
                        let row = by * 128 + (warp / 2) * 32 + rr;
                        let col = bx * 64 + (warp % 2) * 32 + cc;
                        assert!(row < M && col + 1 < N);
                        let byte = (row * N + col) * 2;
                        assert_eq!(byte % 4, 0);
                        assert_eq!((byte - (lane & 15) * 4) % 64, 0);
                        for h in 0..2 {
                            let index = (row - by * 128) * 64 + col + h - bx * 64;
                            assert!(!hits[index]);
                            hits[index] = true;
                        }
                    }
                }
            }
        }
        assert!(hits.into_iter().all(|v| v));
    }
}
