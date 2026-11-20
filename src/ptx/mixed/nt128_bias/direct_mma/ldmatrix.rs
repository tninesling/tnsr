//! Separate test-only x4 operand transport. No geometry/math/epilogue change.
use super::*;
mod gpu_tests;
use crate::ptx::graph::bf16_ldmatrix_provider_expr;
fn candidate(r: &MatMulRegion, target: PtxTarget, coalesced: bool, bias: bool) -> Result<TileIR> {
    let mut result = super::normalized::candidate(r, target, coalesced, bias)?;
    let mut count = 0;
    for s in &mut result.body.stmts {
        if let Stmt::AllocTile {
            space: MemorySpace::Fragment,
            layout,
            ..
        } = s
        {
            ensure!(
                *layout == TileLayout::Bf16MmaAccumulator128Normalized,
                "only normalized explicit MMA source"
            );
            *layout = TileLayout::Bf16MmaAccumulator128Ldmatrix;
            count += 1;
        }
    }
    ensure!(count == 1, "one accumulator required");
    result.kernel_name = format!(
        "test_ldmatrix_mma_nt128_{}_{}",
        if coalesced { "coalesced" } else { "scalar" },
        if bias { "bias" } else { "plain" }
    );
    result.validate_layouts()?;
    Ok(result)
}
fn eval(e: &Expr, lane: usize, warp: usize) -> usize {
    match e {
        Expr::Const(v) => usize::try_from(*v).unwrap(),
        Expr::ThreadIdx(Dim::X) => lane,
        Expr::ThreadIdx(Dim::Y) => warp,
        Expr::Add(a, b) => eval(a, lane, warp) + eval(b, lane, warp),
        Expr::Mul(a, b) => eval(a, lane, warp) * eval(b, lane, warp),
        Expr::ShiftRight(a, s) => eval(a, lane, warp) >> s,
        Expr::BitAnd(a, m) => eval(a, lane, warp) & usize::try_from(*m).unwrap(),
        _ => panic!("provider must use only explicit shift/mask addresses"),
    }
}
#[test]
fn ldmatrix_x4_provider_register_element_identity_alignment_and_bounds() -> Result<()> {
    let (r, target) = fixture()?;
    let (_, schedule, _) = ir::candidate(&r, target)?;
    for lhs in [true, false] {
        let layout = if lhs {
            schedule.operand_layouts.0
        } else {
            schedule.operand_layouts.1
        };
        assert_eq!(layout.row_stride, 40);
        assert_eq!(layout.column_major, !lhs);
        assert_eq!(layout.xor_mask, 0);
        for warp in 0..8 {
            for k in [0, 16] {
                for fragment in 0..if lhs { 2 } else { 4 } {
                    let (row, col) = bf16_ldmatrix_provider_expr(lhs, fragment, k);
                    let providers = std::array::from_fn::<_, 32, _>(|lane| {
                        (eval(&row, lane, warp), eval(&col, lane, warp))
                    });
                    for &(rr, cc) in &providers {
                        let (pr, pc) = if lhs { (rr, cc) } else { (cc, rr) };
                        assert!(pr < 128 && pc + 7 < 32);
                        let bytes = (pr * 40 + pc) * 2;
                        assert_eq!(bytes % 16, 0);
                        assert!(bytes + 16 <= 10240);
                        for base in [0, 10240, 20480, 30720] {
                            assert_eq!((base + bytes) % 16, 0);
                            assert!(base + bytes + 16 <= base + 10240);
                        }
                    }
                    let mut seen = [false; 256];
                    for lane in 0..32 {
                        for reg in 0..4 {
                            for half in 0..2 {
                                // Documented ldmatrix: register reg receives matrix reg,
                                // row lane>>2, pair2*(lane&3), preserving halfword order.
                                let provider = providers[reg * 8 + lane / 4];
                                let actual = if lhs {
                                    (provider.0, provider.1 + 2 * (lane & 3) + half)
                                } else {
                                    (provider.0 + 2 * (lane & 3) + half, provider.1)
                                };
                                let expected = if lhs {
                                    (
                                        (warp / 2) * 32 + fragment * 16 + lane / 4 + 8 * (reg % 2),
                                        k + 8 * (reg / 2) + 2 * (lane & 3) + half,
                                    )
                                } else {
                                    (
                                        k + 8 * (reg % 2) + 2 * (lane & 3) + half,
                                        (warp % 2) * 64 + fragment * 16 + 8 * (reg / 2) + lane / 4,
                                    )
                                };
                                assert_eq!(
                                    actual, expected,
                                    "must equal qualified packed scalar LDS BF16 element"
                                );
                                let local = if lhs {
                                    ((actual.0 - (warp / 2) * 32 - fragment * 16), actual.1 - k)
                                } else {
                                    (actual.0 - k, actual.1 - (warp % 2) * 64 - fragment * 16)
                                };
                                assert!(local.0 < 16 && local.1 < 16);
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
fn source(r: &MatMulRegion, target: PtxTarget, ir: TileIR) -> Result<String> {
    let mut graph = TileGraph::from_graph(&TensorGraph::<f32>::new());
    graph.add_matmul_regions(std::slice::from_ref(r))?;
    graph.matmul_region_kernels[0].ir = ir;
    graph.optimize_indices()?;
    Ok(PtxGraph::from(graph).with_target(target).module_source())
}
#[test]
fn ldmatrix_only_transport_changes_all_prior_explicit_and_normalized_ptx_identical() -> Result<()> {
    let (_, target) = fixture()?;
    let root = std::path::Path::new("target/paper-kernels/explicit-bf16-mma-direct-qualification");
    for (name, n) in [("ffn", 1408), ("qkv", 1536)] {
        let r = fixture_for(16384, n, target)?;
        for coalesced in [false, true] {
            let old = super::normalized::candidate(&r, target, coalesced, true)?;
            let result = candidate(&r, target, coalesced, true)?;
            let mut restored = result.clone();
            for s in &mut restored.body.stmts {
                if let Stmt::AllocTile { layout, .. } = s
                    && *layout == TileLayout::Bf16MmaAccumulator128Ldmatrix
                {
                    *layout = TileLayout::Bf16MmaAccumulator128Normalized;
                }
            }
            assert_eq!(format!("{:?}", old.body), format!("{:?}", restored.body));
            assert_eq!(format!("{:?}", old.params), format!("{:?}", result.params));
            assert_eq!(old.shared_mem_bytes, result.shared_mem_bytes);
            let prior = root
                .join("normalized-backend-v1/device-v1")
                .join(format!(
                    "{name}_{}",
                    if coalesced { "coalesced" } else { "scalar" }
                ))
                .join("candidate.ptx");
            let previous = source(&r, target, old)?;
            assert_eq!(previous, std::fs::read_to_string(prior)?);
            let ptx = source(&r, target, result)?;
            let math = |text: &str| {
                text.lines()
                    .map(str::trim)
                    .filter(|l| {
                        l.starts_with("mma.sync.")
                            || l.starts_with("add.rn.f32")
                            || l.starts_with("cvt.rn.bf16.f32")
                    })
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                math(&previous),
                math(&ptx),
                "ordered MMA/accumulator operands/bias/RN dataflow must remain identical"
            );
            assert_eq!(
                ptx.matches("ldmatrix.sync.aligned.m8n8.x4.shared.b16")
                    .count(),
                12
            );
            assert_eq!(
                ptx.matches("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
                    .count(),
                32
            );
            assert!(
                !ptx.contains("ld.shared.b32")
                    && !ptx.contains("div.u64")
                    && !ptx.contains("rem.u64")
                    && !ptx.contains(".trans.shared")
            );
            assert_eq!(ptx.matches("cp.async.ca.shared.global").count(), 8);
            assert_eq!(ptx.matches("cp.async.wait_group").count(), 2);
            assert_eq!(ptx.matches("cvt.rn.bf16.f32").count(), 64);
            assert_eq!(ptx.matches("add.rn.f32").count(), 64);
            assert_eq!(
                ptx.matches(if coalesced {
                    "st.global.b32"
                } else {
                    "st.global.b16"
                })
                .count(),
                if coalesced { 32 } else { 64 }
            );
        }
    }
    Ok(())
}

fn prepare(
    r: &MatMulRegion,
    target: PtxTarget,
    output: u64,
    len: usize,
    coalesced: bool,
) -> Result<(TileIR, bool)> {
    ensure!(
        output != 0
            && output.is_multiple_of(16)
            && r.m.checked_mul(r.n) == Some(len)
            && len
                .checked_mul(2)
                .and_then(|v| u64::try_from(v).ok())
                .and_then(|v| output.checked_add(v))
                .is_some(),
        "original output ABI invalid"
    );
    if !output.is_multiple_of(128) {
        return Ok((ir::candidate(r, target)?.0, false));
    }
    Ok((candidate(r, target, coalesced, true)?, true))
}
