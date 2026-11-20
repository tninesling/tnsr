//! Isolated cfg(test) rectangle. No production policy or baseline mutation.
use super::*;
use crate::tile::{Stmt, TileIR, TileLayout};
use anyhow::Result;

pub(super) fn candidate(
    r: &MatMulRegion,
    budget: MatMulResources,
    asynchronous: bool,
) -> Result<(MatMulSchedule, TileIR)> {
    let c = SplitKBf16TilePolicy::Square64
        .candidate(
            r,
            4,
            r.k.checked_mul(4)
                .ok_or_else(|| anyhow::anyhow!("K overflow"))?,
            budget,
            true,
        )
        .ok_or_else(|| anyhow::anyhow!("outside exact current split physical family"))?;
    c.rectangle128x64_partial_ir(r, budget, asynchronous)
}

#[test]
fn split128_geometry_bijection_source_bounds_and_per_output_k16_order() {
    for (m, n, k) in [
        (512, 1408, 4096),
        (1408, 512, 4096),
        (512, 1536, 4096),
        (512, 4096, 4096),
    ] {
        let r = fixture(m, n, k);
        let saved = r.clone();
        let (s, ir) = candidate(&r, resources(), true).unwrap();
        assert_eq!(r, saved);
        assert_eq!(s.shared_mem_bytes(), 32768);
        assert_eq!(ir.shared_mem_bytes, 32768);
        assert_eq!(
            ir.params
                .iter()
                .map(|p| (p.dtype, p.is_input))
                .collect::<Vec<_>>(),
            [
                (DType::BF16, true),
                (DType::BF16, true),
                (DType::F32, false)
            ]
        );
        let mut seen = vec![0u8; 128 * 64];
        for warp in 0..8 {
            for fm in 0..2 {
                for fn_ in 0..2 {
                    for row in 0..16 {
                        for col in 0..16 {
                            let row = (warp / 2) * 32 + fm * 16 + row;
                            let col = (warp % 2) * 32 + fn_ * 16 + col;
                            seen[row * 64 + col] += 1;
                        }
                    }
                }
            }
        }
        assert!(seen.iter().all(|&v| v == 1));
        // Full CTA rectangles are disjoint, covering all four output planes.
        assert_eq!(4 * (m / 128) * (n / 64) * seen.len(), 4 * m * n);
        for part in 0..4 {
            for kk in [0, k - 1] {
                for mm in [0, m - 1] {
                    for nn in [0, n - 1] {
                        let lhs = (part * k + kk) * m + mm;
                        let rhs = (part * k + kk) * n + nn;
                        assert!(lhs < 4 * k * m && rhs < 4 * k * n);
                        assert_eq!(lhs / m, part * k + kk);
                        assert_eq!(rhs / n, part * k + kk);
                    }
                }
            }
        }
        let c = SplitKBf16TilePolicy::Square64
            .candidate(&r, 4, 4 * k, resources(), true)
            .unwrap();
        let (_, old) = c.partial_ir(&r, resources(), true).unwrap();
        assert_eq!(
            old.params
                .iter()
                .map(|p| (&p.name, p.dtype, p.is_input))
                .collect::<Vec<_>>(),
            ir.params
                .iter()
                .map(|p| (&p.name, p.dtype, p.is_input))
                .collect::<Vec<_>>(),
            "same typed root bindings in identical order"
        );
        // All WarpAccumulator representations retain identical2x2 per-warp
        // fragment/K32 mechanics; only topology and full tile extents differ.
        fn ownership(block: &crate::tile::Block, out: &mut Vec<(usize, usize)>) {
            for stmt in &block.stmts {
                match stmt {
                    Stmt::AllocTile {
                        layout: TileLayout::WarpAccumulator { warp_fragments, .. },
                        ..
                    } => out.push(*warp_fragments),
                    Stmt::If { body, .. } | Stmt::ForLoop { body, .. } => ownership(body, out),
                    _ => {}
                }
            }
        }
        let (mut a, mut b) = (vec![], vec![]);
        ownership(&old.body, &mut a);
        ownership(&ir.body, &mut b);
        assert_eq!(a, b);
        assert!(!a.is_empty());
        assert!(a.iter().all(|x| *x == (2, 2)));
        fn math_contract(
            block: &crate::tile::Block,
            loops: &mut Vec<(String, crate::tile::Expr, crate::tile::Expr)>,
            operations: &mut Vec<MatMulSchedule>,
        ) {
            for stmt in &block.stmts {
                match stmt {
                    Stmt::ForLoop {
                        loop_var,
                        start,
                        end,
                        body,
                        ..
                    } => {
                        loops.push((loop_var.clone(), start.clone(), end.clone()));
                        math_contract(body, loops, operations);
                    }
                    Stmt::If { body, .. } => math_contract(body, loops, operations),
                    Stmt::MatMul { schedule, .. } => operations.push(*schedule),
                    _ => {}
                }
            }
        }
        let (mut oldloops, mut newloops, mut oldops, mut newops) = (vec![], vec![], vec![], vec![]);
        math_contract(&old.body, &mut oldloops, &mut oldops);
        math_contract(&ir.body, &mut newloops, &mut newops);
        assert_eq!(oldloops, newloops);
        assert_eq!(
            oldloops,
            vec![(
                "k_tile".into(),
                crate::tile::Expr::Const(0),
                crate::tile::Expr::Const((k / 32) as i64)
            )]
        );
        assert_eq!(oldops.len(), 1);
        assert_eq!(newops.len(), 1);
        let mut normalized = newops[0];
        normalized.block_tile = oldops[0].block_tile;
        normalized.warp_topology = oldops[0].warp_topology;
        normalized.operand_layouts = oldops[0].operand_layouts;
        assert_eq!(
            normalized, oldops[0],
            "all arithmetic/pipeline/fragment schedule fields unchanged"
        );
        for part in 0..4 {
            let sequence = |schedule: MatMulSchedule| {
                (0..k / schedule.block_tile.k)
                    .flat_map(move |tile| {
                        (0..schedule.block_tile.k / 16)
                            .map(move |step| part * k + tile * schedule.block_tile.k + step * 16)
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(sequence(oldops[0]), sequence(newops[0]));
            assert_eq!(
                sequence(newops[0]),
                (part * k..(part + 1) * k).step_by(16).collect::<Vec<_>>()
            );
        }
    }
}
#[test]
fn split128_rejects_unproven_maps_partial_tiles_and_resource_failures() {
    let original = fixture(512, 1408, 4096);
    for change in 0..5 {
        let mut r = original.clone();
        match change {
            0 => r.inputs[0].tensor.access.results[0] = IndexExpr::Const(0),
            1 => r.inputs[1].tensor.dtype = DType::F32,
            2 => r.inputs[0].source_shape[0] += 1,
            3 => r.output_shape[0] = 2,
            _ => {
                r.inputs[0].tensor.access.results[0] = IndexExpr::Add(
                    Box::new(r.inputs[0].tensor.access.results[0].clone()),
                    Box::new(IndexExpr::Const(1)),
                )
            }
        }
        assert!(candidate(&r, resources(), true).is_err());
    }
    assert!(candidate(&fixture(576, 1408, 4096), resources(), true).is_err());
    assert!(candidate(&original, resources(), false).is_err());
    for budget in [
        MatMulResources {
            shared_bytes_per_block: 32767,
            ..resources()
        },
        MatMulResources {
            shared_bytes_per_sm: 65535,
            ..resources()
        },
        MatMulResources {
            registers_per_sm: 65535,
            ..resources()
        },
    ] {
        assert!(candidate(&original, budget, true).is_err());
    }
}

#[test]
fn split128_forced_new_resource_failure_preserves_complete_square64_17_plus_4_census() {
    let mut square_fallbacks = 0;
    let mut ordinary_fallbacks = 0;
    for (m, n, count) in [
        (512, 4096, 1),
        (1408, 512, 4),
        (512, 1408, 8),
        (512, 1536, 4),
        (512, 512, 4),
    ] {
        let mut r = fixture(m, n, 4096);
        if (m, n) == (512, 512) {
            // The four retained profile fallthroughs already have64x64 grids.
            // This is an explicitly out-of-scope schedule descriptor, not an
            // inference that geometry alone proves actual physical provenance.
            r.schedule.block_tile = MatMulTile {
                m: 64,
                n: 64,
                k: 32,
            };
            r.schedule.warp_fragments = (2, 2);
        }
        for _ in 0..count {
            if let Some(c) =
                SplitKBf16TilePolicy::Square64.candidate(&r, 4, 16384, resources(), true)
            {
                let (old_schedule, old_ir) = c.partial_ir(&r, resources(), true).unwrap();
                let too_small = MatMulResources {
                    shared_bytes_per_block: 32767,
                    ..resources()
                };
                assert!(c.rectangle128x64_partial_ir(&r, too_small, true).is_err());
                let (fallback_schedule, fallback_ir) = c.partial_ir(&r, resources(), true).unwrap();
                assert_eq!(old_schedule, fallback_schedule);
                assert_eq!(format!("{old_ir:?}"), format!("{fallback_ir:?}"));
                assert_eq!(
                    fallback_schedule.block_tile,
                    MatMulTile {
                        m: 64,
                        n: 64,
                        k: 32
                    }
                );
                assert_eq!(fallback_schedule.shared_mem_bytes(), 18432);
                square_fallbacks += 1;
            } else {
                ordinary_fallbacks += 1;
            }
        }
    }
    assert_eq!((square_fallbacks, ordinary_fallbacks), (17, 4));
    let r = fixture(256, 512, 512);
    let c = SplitKBf16TilePolicy::Square64
        .candidate(&r, 4, 2048, resources(), true)
        .unwrap();
    assert!(c.partial_ir(&r, resources(), true).is_some());
    assert!(
        c.rectangle128x64_partial_ir(&r, resources(), true).is_err(),
        "unqualified partK must retain Square64"
    );
    assert_ne!(
        SplitKBf16TilePolicy::Rectangle128x64.codegen_tag(),
        SplitKBf16TilePolicy::Square64.codegen_tag()
    );
    assert_eq!(SplitKBf16TilePolicy::default(), SplitKBf16TilePolicy::Off);
}
