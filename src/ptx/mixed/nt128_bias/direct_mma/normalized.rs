//! Separate test-only mode: normalize backend-created unsigned addresses only.
//! No new tensor operation, operand transport, FP arithmetic or epilogue.
use super::*;
mod gpu_tests;
use crate::ptx::graph::normalized_mma_backend_expr;

pub(super) fn candidate(
    r: &MatMulRegion,
    target: PtxTarget,
    coalesced: bool,
    bias: bool,
) -> Result<TileIR> {
    let mut body = if coalesced {
        super::coalesced::candidate(r, target, bias)?
    } else {
        super::candidate(r, target, bias)?
    };
    let mut changed = 0;
    for statement in &mut body.body.stmts {
        if let Stmt::AllocTile {
            space: MemorySpace::Fragment,
            layout,
            ..
        } = statement
        {
            ensure!(
                *layout == TileLayout::Bf16MmaAccumulator128,
                "only explicit old MMA may be normalized"
            );
            *layout = TileLayout::Bf16MmaAccumulator128Normalized;
            changed += 1;
        }
    }
    ensure!(changed == 1, "one explicit accumulator required");
    body.kernel_name = format!(
        "test_normalized_backend_mma_nt128_{}_{}",
        if coalesced { "coalesced" } else { "scalar" },
        if bias { "bias" } else { "plain" }
    );
    body.validate_layouts()?;
    Ok(body)
}
fn eval(e: &Expr, lane: u64, warp: u64, bx: u64, by: u64) -> u64 {
    match e {
        Expr::Const(v) => u64::try_from(*v).unwrap(),
        Expr::ThreadIdx(Dim::X) => lane,
        Expr::ThreadIdx(Dim::Y) => warp,
        Expr::BlockIdx(Dim::X) => bx,
        Expr::BlockIdx(Dim::Y) => by,
        Expr::Add(a, b) => eval(a, lane, warp, bx, by).wrapping_add(eval(b, lane, warp, bx, by)),
        Expr::Sub(a, b) => eval(a, lane, warp, bx, by).wrapping_sub(eval(b, lane, warp, bx, by)),
        Expr::Mul(a, b) => eval(a, lane, warp, bx, by).wrapping_mul(eval(b, lane, warp, bx, by)),
        Expr::FloorDiv(v, d) => eval(v, lane, warp, bx, by) / u64::try_from(*d).unwrap(),
        Expr::Mod(v, d) => eval(v, lane, warp, bx, by) % u64::try_from(*d).unwrap(),
        Expr::ShiftRight(v, d) => eval(v, lane, warp, bx, by) >> d,
        Expr::BitAnd(v, d) => eval(v, lane, warp, bx, by) & d,
        _ => panic!("unexpected bounded backend expression"),
    }
}
#[test]
fn normalized_backend_unsigned_lane_warp_and_address_equivalence() {
    let lane = Expr::ThreadIdx(Dim::X);
    let warp = Expr::ThreadIdx(Dim::Y);
    let group = Expr::FloorDiv(Box::new(lane.clone()), 4);
    let pair = Expr::Mod(Box::new(lane.clone()), 4) * 2usize;
    let rowbase =
        Expr::BlockIdx(Dim::Y) * 128usize + Expr::FloorDiv(Box::new(warp.clone()), 2) * 32usize;
    let colbase =
        Expr::BlockIdx(Dim::X) * 128usize + Expr::Mod(Box::new(warp.clone()), 2) * 64usize;
    let expressions = [
        group.clone(),
        pair.clone(),
        rowbase.clone(),
        colbase.clone(),
        rowbase.clone() * 1408usize + colbase.clone() + pair.clone(),
        (group.clone() + Expr::Const(16)) * 40usize + pair.clone() + Expr::Const(16),
        (colbase.clone() + group.clone()) * 40usize + pair.clone() + Expr::Const(8),
    ];
    for e in expressions {
        let transformed = normalized_mma_backend_expr(&e);
        for lane in 0..32 {
            for warp in 0..8 {
                for (bx, by) in [(0, 0), (10, 127), (11, 127)] {
                    assert_eq!(
                        eval(&e, lane, warp, bx, by),
                        eval(&transformed, lane, warp, bx, by)
                    );
                }
            }
        }
    }
    for value in [0, 1, 2, 3, 4, 31, 32, 127, 511, i64::MAX] {
        for divisor in [2, 4] {
            for e in [
                Expr::FloorDiv(Box::new(Expr::Const(value)), divisor),
                Expr::Mod(Box::new(Expr::Const(value)), divisor),
            ] {
                assert_eq!(
                    eval(&e, 0, 0, 0, 0),
                    eval(&normalized_mma_backend_expr(&e), 0, 0, 0, 0)
                );
            }
        }
    }
    // No global/general reciprocal transform: non-power-of-two stays division.
    assert!(matches!(
        normalized_mma_backend_expr(&Expr::FloorDiv(Box::new(lane), 3)),
        Expr::FloorDiv(_, 3)
    ));
}
fn source(r: &MatMulRegion, target: PtxTarget, ir: TileIR) -> Result<String> {
    let mut graph = TileGraph::from_graph(&TensorGraph::<f32>::new());
    graph.add_matmul_regions(std::slice::from_ref(r))?;
    graph.matmul_region_kernels[0].ir = ir;
    graph.optimize_indices()?;
    Ok(PtxGraph::from(graph).with_target(target).module_source())
}
#[test]
fn normalized_backend_only_tag_changes_and_no_division_with_same_tensor_work() -> Result<()> {
    let (_, target) = fixture()?;
    for n in [1408, 1536] {
        let r = fixture_for(16384, n, target)?;
        for coalesced in [false, true] {
            for bias in [false, true] {
                let original = if coalesced {
                    super::coalesced::candidate(&r, target, bias)?
                } else {
                    super::candidate(&r, target, bias)?
                };
                let transformed = candidate(&r, target, coalesced, bias)?;
                let mut restored = transformed.clone();
                for s in &mut restored.body.stmts {
                    if let Stmt::AllocTile { layout: tag, .. } = s
                        && *tag == TileLayout::Bf16MmaAccumulator128Normalized
                    {
                        *tag = TileLayout::Bf16MmaAccumulator128;
                    }
                }
                assert_eq!(
                    format!("{:?}", original.body),
                    format!("{:?}", restored.body)
                );
                assert_eq!(
                    format!("{:?}", original.params),
                    format!("{:?}", transformed.params)
                );
                assert_eq!(original.shared_mem_bytes, transformed.shared_mem_bytes);
                let ptx = source(&r, target, transformed)?;
                assert!(!ptx.contains("div.u64") && !ptx.contains("rem.u64"));
                assert_eq!(ptx.matches("ld.shared.b32").count(), 48);
                assert_eq!(
                    ptx.matches("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
                        .count(),
                    32
                );
                assert_eq!(ptx.matches("cvt.rn.bf16.f32").count(), 64);
                assert_eq!(ptx.matches("add.rn.f32").count(), if bias { 64 } else { 0 });
                assert_eq!(ptx.matches("cp.async.ca.shared.global").count(), 8);
                assert_eq!(ptx.matches("cp.async.wait_group").count(), 2);
                if coalesced {
                    assert_eq!(ptx.matches("st.global.b32").count(), 32);
                    assert_eq!(ptx.matches("shfl.sync.bfly.b32").count(), 48);
                    assert_eq!(ptx.matches("selp.b32").count(), 144);
                } else {
                    assert_eq!(ptx.matches("st.global.b16").count(), 64);
                }
            }
        }
    }
    Ok(())
}
#[test]
fn normalized_backend_leaves_all_four_prior_explicit_modules_byte_identical() -> Result<()> {
    let (_, target) = fixture()?;
    let root = std::path::Path::new("target/paper-kernels/explicit-bf16-mma-direct-qualification");
    for (name, n) in [("ffn_forward_nt", 1408), ("qkv_forward_nt", 1536)] {
        let r = fixture_for(16384, n, target)?;
        for (coalesced, path) in [
            (false, root.join("device-v1")),
            (true, root.join("coalesced-v1/device-v1")),
        ] {
            let old = if coalesced {
                super::coalesced::candidate(&r, target, true)?
            } else {
                super::candidate(&r, target, true)?
            };
            assert_eq!(
                source(&r, target, old)?,
                std::fs::read_to_string(path.join(name).join("candidate.ptx"))?
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
