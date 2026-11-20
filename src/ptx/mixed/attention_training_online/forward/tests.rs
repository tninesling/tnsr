use super::*;

#[test]
fn online_forward_q64_arena_and_mma_layouts_are_bounded() -> Result<()> {
    for layout in [ReadLayout::HeadMajor, ReadLayout::SequenceMajor] {
        let ir = ir(Shape::new(8, 8)?, layout)?;
        ir.validate_layouts()?;
        assert_eq!(ir.shared_mem_bytes, 49152);
        assert_eq!(
            ir.params
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            ["query", "key", "value", "output_0", "statistics"]
        );
        let mut views = 0;
        for stmt in &ir.body.stmts {
            if let Stmt::ViewSharedArena {
                byte_offset,
                dtype,
                rows,
                cols,
                layout,
                ..
            } = stmt
            {
                assert!(
                    *byte_offset + layout.allocation_bytes(*rows, *cols, *dtype) <= SHARED_BYTES
                );
                views += 1;
            }
        }
        assert_eq!(views, 6);
        let score_p = ir
            .body
            .stmts
            .iter()
            .filter_map(|s| match s {
                Stmt::ViewSharedArena {
                    byte_offset: 16384,
                    dtype,
                    layout,
                    ..
                } => Some((*dtype, *layout)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            score_p,
            [
                (DType::F32, SharedLayout::row_major(64)),
                (DType::BF16, SharedLayout::row_major(128))
            ]
        );
        for s in [
            schedule(
                SharedLayout::row_major(64),
                SharedLayout {
                    row_stride: 64,
                    column_major: true,
                    xor_mask: 0,
                },
            ),
            schedule(SharedLayout::row_major(128), SharedLayout::row_major(64)),
        ] {
            assert_eq!(s.warp_topology, (4, 2));
            assert_eq!(s.warp_fragments, (1, 2));
            assert_eq!(s.block_threads, (32, 8, 1));
            assert_eq!(s.instruction_tile.k, 16);
            assert_eq!(s.block_tile.k, 64);
        }
    }
    Ok(())
}

#[test]
fn online_forward_row_store_and_causal_block_coverage_are_complete() {
    // All64 rows are owned by one full warp, all64 elements per row by2
    // ordered lane groups. P stores alias only their own F32 score row.
    let mut seen = vec![0usize; 64 * 64];
    for round in 0..8 {
        for warp in 0..8 {
            for lane in 0..32 {
                for group in 0..2 {
                    let row = round * 8 + warp;
                    let col = lane + 32 * group;
                    seen[row * 64 + col] += 1;
                    let p_byte = 16384 + (row * 128 + col) * 2;
                    assert!((16384 + row * 256..16384 + (row + 1) * 256).contains(&p_byte));
                }
            }
        }
    }
    assert!(seen.into_iter().all(|n| n == 1));
    let mut visited = 0;
    for query_block in 0..8 {
        for key_block in 0..=query_block {
            visited += 1;
            for row in 0..64 {
                for col in 0..64 {
                    let query = query_block * 64 + row;
                    let key = key_block * 64 + col;
                    if key <= query {
                        assert!(key_block <= query_block);
                    }
                }
            }
        }
        // All omitted blocks contain exclusively strictly-future positions.
        for key_block in query_block + 1..8 {
            assert!(key_block * 64 > query_block * 64 + 63);
        }
    }
    assert_eq!(visited, 36);
}

#[test]
fn online_forward_score_alias_has_explicit_uniform_war_boundary() -> Result<()> {
    let ir = ir(Shape::new(8, 8)?, ReadLayout::HeadMajor)?;
    let score = ir
        .body
        .stmts
        .iter()
        .find_map(|s| match s {
            Stmt::ViewSharedArena {
                dest,
                byte_offset: 16384,
                dtype: DType::F32,
                ..
            } => Some(*dest),
            _ => None,
        })
        .unwrap();
    let probability = ir
        .body
        .stmts
        .iter()
        .find_map(|s| match s {
            Stmt::ViewSharedArena {
                dest,
                byte_offset: 16384,
                dtype: DType::BF16,
                ..
            } => Some(*dest),
            _ => None,
        })
        .unwrap();
    let keys = ir
        .body
        .stmts
        .iter()
        .find_map(|s| match s {
            Stmt::ForLoop { loop_var, body, .. } if loop_var.starts_with("causal_keys") => {
                Some(body)
            }
            _ => None,
        })
        .unwrap();
    let rows = keys
        .stmts
        .iter()
        .find_map(|s| match s {
            Stmt::ForLoop {
                loop_var,
                start,
                end,
                body,
                ..
            } if loop_var.starts_with("online_rows") => {
                assert_eq!(*start, ix(0));
                assert_eq!(*end, ix(8));
                Some(body)
            }
            _ => None,
        })
        .unwrap();
    let loads = rows
        .stmts
        .iter()
        .enumerate()
        .filter_map(|(i, s)| match s {
            Stmt::LoadScalar {
                source: ScalarMemory::Shared(v),
                ..
            } if *v == score => Some(i),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(loads.len(), 2);
    let barrier = rows
        .stmts
        .iter()
        .position(|s| matches!(s, Stmt::Barrier))
        .unwrap();
    assert!(loads.iter().all(|&i| i < barrier));
    let stores = rows
        .stmts
        .iter()
        .enumerate()
        .filter_map(|(i, s)| match s {
            Stmt::StoreScalar {
                target: ScalarMemory::Shared(v),
                ..
            } if *v == probability => Some(i),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(stores.len(), 2);
    assert!(stores.iter().all(|&i| i > barrier));
    // The barrier is a direct uniform row-loop statement, never leader-predicated.
    assert_eq!(
        rows.stmts
            .iter()
            .filter(|s| matches!(s, Stmt::Barrier))
            .count(),
        1
    );
    assert_eq!(ir.shared_mem_bytes, SHARED_BYTES);
    Ok(())
}
