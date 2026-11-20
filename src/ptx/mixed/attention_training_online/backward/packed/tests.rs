use super::*;
use half::bf16;
use std::collections::HashSet;

fn address(shape: Shape, bh: usize, s: usize, d: usize, w: usize) -> usize {
    ((bh / shape.heads * SEQUENCE + s) * (3 * shape.heads * TILE))
        + w * shape.heads * TILE
        + (bh % shape.heads) * TILE
        + d
}
fn eval(expr: &Expr, bh: usize, sequence: usize, feature: usize) -> i64 {
    match expr {
        Expr::Const(x) => *x,
        Expr::Var(x) if x == "sequence" => sequence as i64,
        Expr::Var(x) if x == "feature" => feature as i64,
        Expr::BlockIdx(Dim::Y) => bh as i64,
        Expr::Add(a, b) => eval(a, bh, sequence, feature) + eval(b, bh, sequence, feature),
        Expr::Mul(a, b) => eval(a, bh, sequence, feature) * eval(b, bh, sequence, feature),
        Expr::FloorDiv(a, n) => eval(a, bh, sequence, feature) / (*n as i64),
        Expr::Mod(a, n) => eval(a, bh, sequence, feature) % (*n as i64),
        _ => panic!("unexpected packed address expression {expr:?}"),
    }
}
fn tree_value(tree: &Join, region: &QkvGradPack, window: usize, active: f32, root: bool) -> f32 {
    match tree {
        Join::Input(i) => {
            if region.inputs[*i].window == window {
                active
            } else {
                0.0
            }
        }
        Join::Add(a, b) => {
            let sum = tree_value(a, region, window, active, false)
                + tree_value(b, region, window, active, false);
            if root {
                sum
            } else {
                bf16::from_f32(sum).to_f32()
            }
        }
    }
}
#[test]
fn packed_backward_windows_and_transpose_have_complete_disjoint_ownership() -> Result<()> {
    // Exhaustive mixed-radix coverage, not sampled addresses. The inverse is
    // (b,s,window,h,d), so the same proof applies to every admitted B/H.
    let shape = Shape { batch: 2, heads: 3 };
    let mut seen = HashSet::new();
    for bh in 0..shape.bh() {
        for s in 0..SEQUENCE {
            for w in 0..3 {
                for d in 0..TILE {
                    let index = address(shape, bh, s, d, w);
                    assert_eq!(
                        eval(
                            &packed_index(
                                shape,
                                Expr::Var("sequence".into()),
                                Expr::Var("feature".into()),
                                w
                            ),
                            bh,
                            s,
                            d
                        ),
                        index as i64
                    );
                    assert!(index < 3 * shape.elements() && seen.insert(index));
                    assert_eq!(index % TILE, d);
                    assert_eq!((index / TILE) % shape.heads, bh % shape.heads);
                    assert_eq!((index / (TILE * shape.heads)) % 3, w);
                    assert_eq!(
                        index / (3 * TILE * shape.heads),
                        bh / shape.heads * SEQUENCE + s
                    );
                }
            }
        }
    }
    assert_eq!(seen.len(), 3 * shape.elements());
    let mut transpose = HashSet::new();
    for feature in 0..TILE {
        for key in 0..TILE {
            let row_exchange = feature * TILE + key;
            let padded = key * TRANSPOSE_PITCH + feature;
            assert!(transpose.insert(padded));
            assert_eq!(row_exchange / TILE, feature);
            assert_eq!(
                (padded / TRANSPOSE_PITCH, padded % TRANSPOSE_PITCH),
                (key, feature)
            );
            assert!(
                TRANSPOSE_OFFSET + padded * 4 + 4 <= TRANSPOSE_OFFSET + TILE * TRANSPOSE_PITCH * 4
            );
        }
    }
    assert_eq!(transpose.len(), TILE * TILE);
    const {
        assert!(TRANSPOSE_OFFSET + TILE * TRANSPOSE_PITCH * 4 <= ROUND_OFFSET);
        assert!(ROUND_OFFSET + ROUND_ELEMENTS * 2 <= SHARED_BYTES);
    }
    // The lower 16KiB exchange and padded transpose never overlap. Scratch
    // overlays dead Q/dO/score storage only after the last causal-loop barrier.
    const {
        assert!(TILE * TILE * 4 <= TRANSPOSE_OFFSET);
    }
    Ok(())
}

#[test]
fn packed_backward_keeps_mma_async_prefix_and_explicit_transpose_phases() -> Result<()> {
    for batch in [8, 16, 32] {
        let shape = Shape::new(batch, 8)?;
        let (region, raw) = fixture(shape)?;
        validate_region(&region, shape)?;
        let pack = qkv_grad_pack::diagnostic_ir(&region, &raw)?;
        assert_eq!(pack.params.len(), 4);
        for layout in [ReadLayout::HeadMajor, ReadLayout::SequenceMajor] {
            let old_q =
                super::super::d_query_ir_with_transport(shape, layout, layout, Transport::Async16)?;
            let q = dquery_ir(shape, layout, &region)?;
            assert_eq!(
                format!("{:?}", &old_q.body.stmts[1..old_q.body.stmts.len() - 1]),
                format!("{:?}", &q.body.stmts[2..q.body.stmts.len() - 1])
            );
            let old_k = super::super::d_key_value_ir_with_transport(
                shape,
                layout,
                layout,
                Transport::Async16,
            )?;
            let kv = dkeyvalue_ir(shape, layout, &region)?;
            let before = old_k.body.stmts.len() - 7;
            assert_eq!(
                format!("{:?}", &old_k.body.stmts[1..before]),
                format!("{:?}", &kv.body.stmts[3..before + 2])
            );
            let tail = &kv.body.stmts[before + 2..];
            assert_eq!(tail.len(), 9);
            assert!(matches!(tail[0], Stmt::ConvertLayout { .. }));
            for i in [1, 3, 5, 7] {
                assert!(matches!(tail[i], Stmt::Barrier));
            }
            assert!(matches!(tail[6], Stmt::ConvertLayout { .. }));
            assert_eq!(
                kv.params
                    .iter()
                    .filter(|p| !p.is_input)
                    .map(|p| p.name.as_str())
                    .collect::<Vec<_>>(),
                vec!["packed_gradient"]
            );
            assert_eq!(q.shared_mem_bytes, SHARED_BYTES);
            assert_eq!(kv.shared_mem_bytes, SHARED_BYTES);
            q.validate_layouts()?;
            kv.validate_layouts()?;
        }
    }
    Ok(())
}
#[test]
fn packed_backward_preserves_pad_join_rounds_and_rejects_wrong_windows() -> Result<()> {
    let shape = Shape::new(8, 8)?;
    let (mut region, _) = fixture(shape)?;
    for bits in 0..=u16::MAX {
        let raw = bf16::from_bits(bits);
        let checkpoint = raw.to_f32();
        for window in 0..3 {
            // Baseline raw BF16 load, then the pack's BF16 pad cast.
            let old_pad = bf16::from_f32(checkpoint).to_f32();
            let old = bf16::from_f32(tree_value(&region.tree, &region, window, old_pad, true));
            // Candidate F32 accumulator is first rounded to the removed raw
            // checkpoint, then rounded AGAIN at the original padded cast.
            let new_pad = bf16::from_f32(bf16::from_f32(checkpoint).to_f32()).to_f32();
            let new = bf16::from_f32(tree_value(&region.tree, &region, window, new_pad, true));
            assert!((old.is_nan() && new.is_nan()) || old.to_bits() == new.to_bits());
            if bits == 0x8000 {
                assert_eq!(old.to_bits(), 0, "zero Add boundary must canonicalize -0");
            }
        }
    }
    region.inputs[0].window = region.inputs[1].window;
    assert!(validate_region(&region, shape).is_err());
    assert!(dquery_ir(shape, ReadLayout::HeadMajor, &region).is_err());
    assert!(dkeyvalue_ir(shape, ReadLayout::HeadMajor, &region).is_err());
    Ok(())
}

#[test]
fn packed_backward_all_three_bh_scopes_cover_each_owned_row_once() -> Result<()> {
    for batch in [8, 16, 32] {
        let shape = Shape::new(batch, 8)?;
        let n = shape.elements();
        let mut rows = HashSet::new();
        for bh in 0..shape.bh() {
            for sequence in 0..SEQUENCE {
                for window in 0..3 {
                    // packed_index has feature only as its final additive coordinate,
                    // so each row writes the entire contiguous interval [base,base+64).
                    // Exhaust every row at every actual BH, not sampled heads/tiles.
                    let base = address(shape, bh, sequence, 0, window);
                    let last = address(shape, bh, sequence, TILE - 1, window);
                    assert_eq!(base % TILE, 0);
                    assert_eq!(last, base + TILE - 1);
                    assert!(last < 3 * n && rows.insert(base / TILE));
                    for feature in [0, TILE - 1] {
                        assert_eq!(
                            eval(
                                &packed_index(
                                    shape,
                                    Expr::Var("sequence".into()),
                                    Expr::Var("feature".into()),
                                    window
                                ),
                                bh,
                                sequence,
                                feature
                            ),
                            address(shape, bh, sequence, feature, window) as i64
                        );
                    }
                    // Exact inverse tuple separates Q/K/V, heads, sequences and batches.
                    assert_eq!(base / TILE % shape.heads, bh % shape.heads);
                    assert_eq!(base / (TILE * shape.heads) % 3, window);
                    assert_eq!(
                        base / (3 * TILE * shape.heads),
                        bh / shape.heads * SEQUENCE + sequence
                    );
                }
            }
        }
        assert_eq!(rows.len(), 3 * n / TILE);
        assert!(rows.iter().all(|&row| row < 3 * n / TILE));
        // Every original 64x64 per-head producer is a complete 16x256-thread
        // domain. Its local (row,column) and transposed (column,row) are bijective.
        let mut local = HashSet::new();
        for iteration in 0..16 {
            for thread in 0..THREADS {
                let index = iteration * THREADS + thread;
                let (row, column) = (index / TILE, index % TILE);
                assert!(row < TILE && column < TILE && local.insert((row, column)));
                assert_eq!((column * TILE + row) / TILE, column);
                assert_eq!((column * TILE + row) % TILE, row);
            }
        }
        assert_eq!(local.len(), TILE * TILE);
    }
    Ok(())
}

#[test]
#[ignore = "manual CPU-only five-module PTX identities; requires retained leaf goldens"]
fn packed_backward_all_bh_complete_qualified_ptx_goldens_cpu_only() -> Result<()> {
    use crate::{
        ptx::{PtxGraph, PtxTarget},
        tile::{TileGraph, TileIR},
    };
    fn source(ir: TileIR) -> Result<String> {
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles
            .region_kernels
            .push(crate::tile::graph::TileRegionKernel { region_id: 0, ir });
        tiles.optimize_indices()?;
        Ok(PtxGraph::from(tiles)
            .with_target(PtxTarget {
                compute_capability: (8, 9),
                ..PtxTarget::sm80()
            })
            .to_ptx())
    }
    let goldens = std::path::PathBuf::from(
        std::env::var_os("TNSR_ONLINE_PACKED_GOLDENS")
            .context("retained leaf-v1 PTX golden directory required")?,
    );
    for batch in [8, 16, 32] {
        let shape = Shape::new(batch, 8)?;
        let (region, _) = fixture(shape)?;
        let layout = ReadLayout::SequenceMajor;
        let modules = [
            (
                "packed-forward.ptx",
                super::super::super::forward::ir_with_transport(shape, layout, Transport::Async16)?,
            ),
            (
                "packed-old-dquery.ptx",
                super::super::d_query_ir_with_transport(shape, layout, layout, Transport::Async16)?,
            ),
            (
                "packed-old-dkeyvalue.ptx",
                super::super::d_key_value_ir_with_transport(
                    shape,
                    layout,
                    layout,
                    Transport::Async16,
                )?,
            ),
            (
                "packed-packed-dquery.ptx",
                dquery_ir(shape, layout, &region)?,
            ),
            (
                "packed-packed-dkeyvalue.ptx",
                dkeyvalue_ir(shape, layout, &region)?,
            ),
        ];
        for (name, ir) in modules {
            assert_eq!(
                source(ir)?,
                std::fs::read_to_string(goldens.join(name))?,
                "qualified complete PTX changed for BH{} / {name}",
                shape.bh()
            );
        }
    }
    Ok(())
}
