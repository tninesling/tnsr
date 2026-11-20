use super::*;
use std::collections::HashMap;

fn find_loop<'a>(block: &'a Block, prefix: &str) -> Option<(&'a str, &'a Expr, &'a Block)> {
    for statement in &block.stmts {
        if let Stmt::ForLoop {
            loop_var,
            end,
            body,
            ..
        } = statement
        {
            if loop_var.starts_with(prefix) {
                return Some((loop_var, end, body));
            }
            if let Some(found) = find_loop(body, prefix) {
                return Some(found);
            }
        }
    }
    None
}
fn all_statements<'a>(block: &'a Block, output: &mut Vec<&'a Stmt>) {
    for stmt in &block.stmts {
        output.push(stmt);
        if let Stmt::ForLoop { body, .. } = stmt {
            all_statements(body, output);
        }
    }
}

#[test]
fn online_training_backward_arena_aliases_and_compression_have_explicit_cta_retirement()
-> Result<()> {
    let shape = Shape::new(8, 8)?;
    for joint in [false, true] {
        let ir = if joint {
            d_key_value_ir(shape, ReadLayout::SequenceMajor, ReadLayout::SequenceMajor)?
        } else {
            d_query_ir(shape, ReadLayout::SequenceMajor, ReadLayout::SequenceMajor)?
        };
        ir.validate_layouts()?;
        assert_eq!(ir.shared_mem_bytes, 49152);
        let mut all = Vec::new();
        all_statements(&ir.body, &mut all);
        for stmt in &all {
            if let Stmt::ViewSharedArena {
                byte_offset,
                dtype,
                rows,
                cols,
                layout,
                ..
            } = stmt
            {
                assert!(byte_offset.is_multiple_of(32));
                assert!(*byte_offset + layout.allocation_bytes(*rows, *cols, *dtype) <= 49152);
                if *dtype == DType::BF16 && layout.row_stride == 128 {
                    assert_eq!(layout.allocation_bytes(*rows, *cols, *dtype), 16384);
                }
            }
            if let Stmt::MatMul { schedule, .. } = stmt {
                assert_eq!(schedule.warp_topology, (4, 2));
                assert_eq!(schedule.warp_fragments, (1, 2));
                assert_eq!(schedule.instruction_tile.k, 16);
                assert_eq!(schedule.block_tile.k, 64);
                assert_eq!(schedule.operand_dtype, DType::BF16);
            }
        }
        let (_, end, rows) = find_loop(&ir.body, "backward_rows").unwrap();
        assert!(matches!(end, Expr::Const(8)));
        let fence = rows
            .stmts
            .iter()
            .position(|s| matches!(s, Stmt::Barrier))
            .unwrap();
        assert_eq!(
            rows.stmts[..fence]
                .iter()
                .filter(|s| matches!(
                    s,
                    Stmt::LoadScalar {
                        source: ScalarMemory::Shared(_),
                        ..
                    }
                ))
                .count(),
            4
        );
        assert!(!rows.stmts[..fence].iter().any(|s| matches!(
            s,
            Stmt::StoreScalar {
                target: ScalarMemory::Shared(_),
                ..
            }
        )));
        assert_eq!(
            rows.stmts[fence + 1..]
                .iter()
                .filter(|s| matches!(
                    s,
                    Stmt::StoreScalar {
                        target: ScalarMemory::Shared(_),
                        ..
                    }
                ))
                .count(),
            if joint { 4 } else { 2 }
        );
        let (_, _, outer) = find_loop(
            &ir.body,
            if joint {
                "causal_queries"
            } else {
                "causal_keys"
            },
        )
        .unwrap();
        let second_mma = outer
            .stmts
            .iter()
            .enumerate()
            .filter_map(|(i, s)| matches!(s, Stmt::MatMul { .. }).then_some(i))
            .nth(1)
            .unwrap();
        assert!(matches!(
            outer.stmts[second_mma + 1],
            Stmt::ConvertLayout { .. }
        ));
        assert!(matches!(outer.stmts[second_mma + 2], Stmt::Barrier));
        assert!(matches!(
            outer.stmts[second_mma + 3],
            Stmt::ConvertLayout { .. }
        ));
        assert!(matches!(outer.stmts[second_mma + 4], Stmt::Barrier));
        assert!(matches!(outer.stmts.last(), Some(Stmt::Barrier)));
        let fragments = all
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    Stmt::AllocTile {
                        space: MemorySpace::Fragment,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(fragments, if joint { 4 } else { 3 });
        let names: Vec<_> = ir.params.iter().map(|p| p.name.as_str()).collect();
        assert!(!names.contains(&"probability") && !names.contains(&"qk"));
        assert_eq!(names.len(), 7);
    }
    Ok(())
}

fn eval(
    expr: &Expr,
    vars: &HashMap<String, usize>,
    x: usize,
    y: usize,
    bx: usize,
    bh: usize,
) -> usize {
    match expr {
        Expr::Const(v) => usize::try_from(*v).unwrap(),
        Expr::Var(v) => vars[v],
        Expr::ThreadIdx(Dim::X) => x,
        Expr::ThreadIdx(Dim::Y) => y,
        Expr::ThreadIdx(Dim::Z) => 0,
        Expr::BlockIdx(Dim::X) => bx,
        Expr::BlockIdx(Dim::Y) => bh,
        Expr::BlockIdx(Dim::Z) => 0,
        Expr::Add(a, b) => eval(a, vars, x, y, bx, bh) + eval(b, vars, x, y, bx, bh),
        Expr::Sub(a, b) => eval(a, vars, x, y, bx, bh) - eval(b, vars, x, y, bx, bh),
        Expr::Mul(a, b) => eval(a, vars, x, y, bx, bh) * eval(b, vars, x, y, bx, bh),
        Expr::FloorDiv(a, n) => eval(a, vars, x, y, bx, bh) / n,
        Expr::Mod(a, n) => eval(a, vars, x, y, bx, bh) % n,
        _ => panic!("unexpected output index expression"),
    }
}

#[test]
fn online_training_backward_transposed_key_and_value_stores_cover_disjoint_coalesced_tiles()
-> Result<()> {
    let shape = Shape::new(8, 8)?;
    let ir = d_key_value_ir(shape, ReadLayout::SequenceMajor, ReadLayout::SequenceMajor)?;
    for (prefix, name, transposed) in [
        ("store_dkey_transposed", "d_key_transposed", true),
        ("store_dvalue", "d_value", false),
    ] {
        let (variable, end, body) = find_loop(&ir.body, prefix).unwrap();
        assert!(matches!(end, Expr::Const(16)));
        let index = body
            .stmts
            .iter()
            .find_map(|s| match s {
                Stmt::StoreScalar {
                    target: ScalarMemory::Global(target),
                    index,
                    ..
                } if target == name => Some(index),
                _ => None,
            })
            .unwrap();
        let mut visits = vec![0_u8; shape.elements()];
        for head in 0..shape.bh() {
            for block in 0..8 {
                for round in 0..16 {
                    let vars = HashMap::from([(variable.to_string(), round)]);
                    for warp in 0..8 {
                        for lane in 0..32 {
                            let actual = eval(index, &vars, lane, warp, block, head);
                            let tile_index = round * 256 + warp * 32 + lane;
                            let expected = if transposed {
                                (head * 64 + tile_index / 64) * 512 + block * 64 + tile_index % 64
                            } else {
                                (head * 512 + block * 64 + tile_index / 64) * 64 + tile_index % 64
                            };
                            assert_eq!(actual, expected);
                            visits[actual] += 1;
                            if lane < 31 {
                                assert_eq!(
                                    eval(index, &vars, lane + 1, warp, block, head),
                                    actual + 1
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(visits.into_iter().all(|v| v == 1));
    }
    Ok(())
}
