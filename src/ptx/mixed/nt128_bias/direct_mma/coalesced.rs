//! Separate test-only epilogue diagnostic; prior scalar artifacts stay immutable.
use super::*;
mod gpu_tests;
pub(super) fn candidate(r: &MatMulRegion, target: PtxTarget, bias: bool) -> Result<TileIR> {
    let mut scalar = super::candidate(r, target, bias)?;
    let first = scalar
        .body
        .stmts
        .iter()
        .position(|s| matches!(s, Stmt::ExtractRegister { .. }))
        .context("no direct scalar suffix")?;
    ensure!(
        scalar.body.stmts[first..]
            .iter()
            .filter(|s| matches!(s, Stmt::StoreGlobalPredicated { .. }))
            .count()
            == 64,
        "scalar output invariant changed"
    );
    let src = match scalar.body.stmts[first] {
        Stmt::ExtractRegister { src, .. } => src,
        _ => unreachable!(),
    };
    scalar.body.stmts.truncate(first);
    scalar.body.stmts.push(Stmt::StoreBf16MmaCoalesced {
        src,
        output_param: "output_0".into(),
        bias_param: bias.then(|| "input_2".into()),
        m: r.m,
        n: r.n,
    });
    scalar.kernel_name = if bias {
        "test_explicit_mma_nt128_coalesced_bias"
    } else {
        "test_explicit_mma_nt128_coalesced_plain"
    }
    .into();
    scalar.validate_layouts()?;
    Ok(scalar)
}
fn pack(lo: u16, hi: u16) -> u32 {
    u32::from(lo) | (u32::from(hi) << 16)
}
fn tag(row: usize, col: usize) -> u16 {
    u16::try_from(row * 64 + col).unwrap()
}
#[test]
fn coalesced_mma_actual_three_stage_network_byte_coverage_and_sector_geometry() -> Result<()> {
    let pairs = crate::ptx::graph::explicit_bf16_mma_transpose_pairs();
    assert_eq!(pairs.len(), 12);
    for mask in [4, 8, 16] {
        assert_eq!(pairs.iter().filter(|p| p.0 == mask).count(), 4);
    }
    let mut bytes = vec![false; 128 * 128 * 2];
    for warp in 0usize..8 {
        for rb in 0..4 {
            let mut words = [[0u32; 8]; 32];
            for (lane, regs) in words.iter_mut().enumerate() {
                for (j, v) in regs.iter_mut().enumerate() {
                    *v = pack(
                        tag(rb * 8 + lane / 4, j * 8 + (lane % 4) * 2),
                        tag(rb * 8 + lane / 4, j * 8 + (lane % 4) * 2 + 1),
                    );
                }
            }
            for &(mask, j0, j1) in &pairs {
                let mask = usize::try_from(mask)?;
                let previous = words;
                let selected = std::array::from_fn::<_, 32, _>(|lane| {
                    if lane & mask != 0 {
                        previous[lane][j0]
                    } else {
                        previous[lane][j1]
                    }
                });
                for lane in 0..32 {
                    assert_eq!(lane & 3, (lane ^ mask) & 3);
                    let peer = selected[lane ^ mask];
                    words[lane][j0] = if lane & mask != 0 {
                        peer
                    } else {
                        previous[lane][j0]
                    };
                    words[lane][j1] = if lane & mask != 0 {
                        previous[lane][j1]
                    } else {
                        peer
                    };
                }
            }
            for (lane, regs) in words.iter().enumerate() {
                for (g, &word) in regs.iter().enumerate() {
                    let localrow = rb * 8 + g;
                    let localcol = lane * 2;
                    assert_eq!(
                        word,
                        pack(tag(localrow, localcol), tag(localrow, localcol + 1))
                    );
                    assert_eq!(u16::try_from(word & 0xffff)?, tag(localrow, localcol));
                    assert_eq!(u16::try_from(word >> 16)?, tag(localrow, localcol + 1));
                    let row = (warp / 2) * 32 + localrow;
                    let col = (warp % 2) * 64 + localcol;
                    let index = (row * 128 + col) * 2;
                    assert_eq!(index % 4, 0);
                    for b in &mut bytes[index..index + 4] {
                        assert!(!*b);
                        *b = true;
                    }
                    for n in [1408usize, 1536] {
                        for (bx, by) in [(0, 0), (n / 128 - 1, 16384 / 128 - 1)] {
                            let rr = 128 * by + row;
                            let cc = 128 * bx + col;
                            assert!(rr < 16384 && cc + 1 < n);
                            let address = 0x1000 + (rr * n + cc) * 2;
                            assert_eq!(address % 4, 0);
                            assert_eq!((address - 4 * lane) % 128, 0);
                        }
                    }
                }
            }
        }
    }
    assert!(bytes.into_iter().all(|b| b));
    for parity in 0..2 {
        let sectors = (0usize..32)
            .map(|l| ((l / 4) * 1408 * 2 + (2 * (l % 4) + parity) * 2) / 32)
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(sectors.len(), 8); //64 useful /256 sector bytes.
    }
    assert_eq!(
        (0usize..32)
            .map(|l| l * 2 / 32)
            .collect::<std::collections::HashSet<_>>()
            .len(),
        2
    );
    assert_eq!(
        (0usize..32)
            .map(|l| l * 4 / 32)
            .collect::<std::collections::HashSet<_>>()
            .len(),
        4
    );
    Ok(())
}
#[test]
fn coalesced_mma_prefix_scalar_ptx_identity_and_instruction_contract() -> Result<()> {
    let golden = std::path::Path::new(
        "target/paper-kernels/explicit-bf16-mma-direct-qualification/device-v1",
    );
    let (_, target) = fixture()?;
    for (name, n) in [("ffn_forward_nt", 1408usize), ("qkv_forward_nt", 1536)] {
        let r = fixture_for(16384, n, target)?;
        let scalar = super::candidate(&r, target, true)?;
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.add_matmul_regions(std::slice::from_ref(&r))?;
        tiles.matmul_region_kernels[0].ir = scalar.clone();
        tiles.optimize_indices()?;
        assert_eq!(
            PtxGraph::from(tiles).with_target(target).module_source(),
            std::fs::read_to_string(golden.join(name).join("candidate.ptx"))?
        );
        for bias in [false, true] {
            let old = super::candidate(&r, target, bias)?;
            let next = candidate(&r, target, bias)?;
            let index = old
                .body
                .stmts
                .iter()
                .position(|s| matches!(s, Stmt::ExtractRegister { .. }))
                .unwrap();
            assert_eq!(
                format!("{:?}", &old.body.stmts[..index]),
                format!("{:?}", &next.body.stmts[..index])
            );
            assert_eq!(next.body.stmts.len(), index + 1);
            let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
            tiles.add_matmul_regions(std::slice::from_ref(&r))?;
            tiles.matmul_region_kernels[0].ir = next;
            tiles.optimize_indices()?;
            let ptx = PtxGraph::from(tiles).with_target(target).module_source();
            assert_eq!(
                ptx.matches("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
                    .count(),
                32
            );
            assert_eq!(ptx.matches("ld.shared.b32").count(), 48);
            assert_eq!(ptx.matches("st.global.b32").count(), 32);
            assert_eq!(ptx.matches("shfl.sync.bfly.b32").count(), 48);
            assert_eq!(ptx.matches("selp.b32").count(), 144);
            assert_eq!(ptx.matches("cvt.rn.bf16.f32").count(), 64);
            assert_eq!(ptx.matches("add.rn.f32").count(), if bias { 64 } else { 0 });
            assert!(!ptx.contains("wmma.") && !ptx.contains("st.global.b16"));
        }
    }
    Ok(())
}

/// Output preflight is part of this isolated caller, never inferred from a tag.
/// A valid16-byte ABI with weaker128-byte alignment retains the entire WMMA IR.
fn prepare(
    r: &MatMulRegion,
    target: PtxTarget,
    output: u64,
    len: usize,
    bias: bool,
) -> Result<(TileIR, bool)> {
    let expected = r.m.checked_mul(r.n).context("output extent overflow")?;
    ensure!(
        output != 0
            && output.is_multiple_of(16)
            && len == expected
            && len
                .checked_mul(2)
                .and_then(|b| u64::try_from(b).ok())
                .and_then(|b| output.checked_add(b))
                .is_some(),
        "invalid original output ABI"
    );
    if !output.is_multiple_of(128) {
        let (old, _, plain) = ir::candidate(r, target)?;
        return Ok((if bias { old } else { plain }, false));
    }
    Ok((candidate(r, target, bias)?, true))
}
#[test]
fn coalesced_mma_unaligned_output_keeps_whole_wmma_and_rejects_partial_warps() -> Result<()> {
    let (r, target) = fixture()?;
    for bias in [false, true] {
        let (old, _, plain) = ir::candidate(&r, target)?;
        let expected = if bias { old } else { plain };
        let (body, selected) = prepare(&r, target, 0x1010, r.m * r.n, bias)?;
        assert!(!selected);
        assert_eq!(format!("{body:?}"), format!("{expected:?}"));
        assert!(prepare(&r, target, 0, r.m * r.n, bias).is_err());
        assert!(prepare(&r, target, 0x1000, r.m * r.n - 1, bias).is_err());
        assert!(prepare(&r, target, u64::MAX - 127, r.m * r.n, bias).is_err());
        let (mut body, selected) = prepare(&r, target, 0x1000, r.m * r.n, bias)?;
        assert!(selected);
        let epilogue = body.body.stmts.pop().unwrap();
        let mut opaque = epilogue.clone();
        let Stmt::StoreBf16MmaCoalesced { src, .. } = &mut opaque else {
            unreachable!()
        };
        let original = *src;
        for s in &mut body.body.stmts {
            if let Stmt::AllocTile { var, layout, .. } = s
                && *var == original
            {
                *layout = r.schedule.accumulator_tile_layout();
            }
        }
        body.body.stmts.push(opaque);
        assert!(body.validate_layouts().is_err());
        let mut body = candidate(&r, target, bias)?;
        body.body.stmts.insert(0, Stmt::BoundsCheck { extent: 1 });
        assert!(body.validate_layouts().is_err());
        let mut body = candidate(&r, target, bias)?;
        let epilogue = body.body.stmts.pop().unwrap();
        body.body.stmts.push(Stmt::ForLoop {
            loop_var: "invalid_collective".into(),
            start: Expr::Const(0),
            end: Expr::Const(1),
            body: Block {
                stmts: vec![epilogue],
            },
            carries: Vec::new(),
        });
        assert!(body.validate_layouts().is_err());
    }
    Ok(())
}
