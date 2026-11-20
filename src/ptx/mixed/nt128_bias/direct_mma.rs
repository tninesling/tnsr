//! Isolated explicit BF16 MMA/direct-scalar-output experiment, never selected.
//! Reference maps: PTX ISA, mma.m16n8k16 floating-point A/B/C/D fragments.
use super::*;
mod coalesced;
mod gpu_tests;
mod ldmatrix;
mod normalized;
use crate::tile::ir::MatrixLayout;
fn constant(value: usize) -> Expr {
    Expr::Const(i64::try_from(value).expect("bounded output coordinate"))
}
use crate::tile::{Dim, Expr, MemorySpace, TileLayout};

/// Register order: existing (m16,n16) fragment, then low/high n8, then c0..c3.
fn coordinates(lane: usize, warp: usize, index: usize) -> (usize, usize) {
    let fragment = index / 8;
    let c = index % 4;
    (
        (warp / 2) * 32 + (fragment / 4) * 16 + lane / 4 + (c / 2) * 8,
        (warp % 2) * 64 + (fragment % 4) * 16 + ((index % 8) / 4) * 8 + (lane % 4) * 2 + c % 2,
    )
}
fn count_windows(block: &Block) -> usize {
    block
        .stmts
        .iter()
        .map(|s| match s {
            Stmt::StoreAccumulatorWindow { .. } => 1,
            Stmt::ForLoop { body, .. } | Stmt::If { body, .. } => count_windows(body),
            _ => 0,
        })
        .sum()
}
/// Staging/copy/K loops and final async wait/barrier are retained verbatim.
/// A fresh IR is returned only after all original four windows are proved.
fn candidate(r: &MatMulRegion, target: PtxTarget, with_bias: bool) -> Result<TileIR> {
    let (old, schedule, plain) = ir::candidate(r, target)?;
    ensure!(
        schedule.effective_grid_group_m() <= 1,
        "ungrouped original coordinates required"
    );
    let mut result = if with_bias { old } else { plain };
    ensure!(
        count_windows(&result.body) == 4,
        "exact four-window original required"
    );
    let first = result
        .body
        .stmts
        .iter()
        .position(|s| matches!(s, Stmt::StoreAccumulatorWindow { .. }))
        .context("no original output windows")?;
    ensure!(
        matches!(
            result.body.stmts[..first]
                .iter()
                .rev()
                .find(|s| !matches!(s, Stmt::ViewSharedArena { .. })),
            Some(Stmt::Barrier)
        ),
        "final shared operand retirement missing"
    );
    let suffix = &result.body.stmts[first..];
    ensure!(suffix.len() == 15, "original window suffix changed");
    for pass in 0..4 {
        let offset = pass * 4;
        ensure!(
            matches!(suffix[offset], Stmt::StoreAccumulatorWindow { fragment_origin, fragment_shape: (1, 2), .. } if fragment_origin == (pass / 2, (pass % 2) * 2)),
            "original window mapping changed"
        );
        ensure!(
            matches!(suffix[offset + 1], Stmt::Barrier),
            "original publication barrier changed"
        );
        let Stmt::ForLoop {
            start, end, body, ..
        } = &suffix[offset + 2]
        else {
            anyhow::bail!("original output loop changed");
        };
        ensure!(
            *start == Expr::Const(0)
                && *end == Expr::Const(16)
                && body.stmts.len() == (if with_bias { 5 } else { 3 })
                && matches!(body.stmts.last(), Some(Stmt::StoreGlobalPredicated { .. })),
            "original scalar store sequence changed"
        );
        if pass != 3 {
            ensure!(
                matches!(suffix[offset + 3], Stmt::Barrier),
                "original reader retirement changed"
            );
        }
    }
    let mut accumulator = None;
    for s in &mut result.body.stmts[..first] {
        if let Stmt::AllocTile {
            var,
            space: MemorySpace::Fragment,
            layout,
            ..
        } = s
        {
            ensure!(accumulator.is_none(), "one accumulator required");
            ensure!(
                *layout == schedule.accumulator_tile_layout(),
                "original WMMA allocation changed"
            );
            accumulator = Some(*var);
            *layout = TileLayout::Bf16MmaAccumulator128;
        }
    }
    let accumulator = accumulator.context("missing accumulator")?;
    // The deleted suffix is specifically the qualified shared-window epilogue,
    // not arbitrary arithmetic. Its builder is independently guarded by ir::candidate.
    result.body.stmts.truncate(first);
    let scalar = ir::fresh_variable(&result.body)?;
    let bias = TileVar(scalar.0.checked_add(1).context("variable overflow")?);
    for var in [scalar, bias] {
        result.body.stmts.push(Stmt::AllocTile {
            var,
            space: MemorySpace::Register,
            layout: TileLayout::ThreadScalar,
            dtype: DType::F32,
            rows: 1,
            cols: 1,
        });
    }
    let lane = Expr::ThreadIdx(Dim::X);
    let warp = Expr::ThreadIdx(Dim::Y);
    for index in 0..64 {
        let fragment = index / 8;
        let c = index % 4;
        let row = Expr::BlockIdx(Dim::Y) * 128usize
            + Expr::FloorDiv(Box::new(warp.clone()), 2) * 32usize
            + constant((fragment / 4) * 16)
            + Expr::FloorDiv(Box::new(lane.clone()), 4)
            + constant((c / 2) * 8);
        let col = Expr::BlockIdx(Dim::X) * 128usize
            + Expr::Mod(Box::new(warp.clone()), 2) * 64usize
            + constant((fragment % 4) * 16)
            + constant(((index % 8) / 4) * 8)
            + Expr::Mod(Box::new(lane.clone()), 4) * 2usize
            + constant(c % 2);
        result.body.stmts.push(Stmt::ExtractRegister {
            dest: scalar,
            src: accumulator,
            index,
        });
        if with_bias {
            result.body.stmts.push(Stmt::LoadScalar {
                dest: bias,
                source: ScalarMemory::Global("input_2".into()),
                index: col.clone(),
            });
            result.body.stmts.push(Stmt::SetScalar {
                dest: scalar,
                value: ScalarExpr::binary(ScalarBinaryOp::AddRn, scalar, bias),
            });
        }
        result.body.stmts.push(Stmt::StoreGlobalPredicated {
            dest_param: "output_0".into(),
            src: scalar,
            element_index: row.clone() * r.n + col.clone(),
            row,
            col,
            layout: MatrixLayout {
                rows: r.m,
                cols: r.n,
                row_stride: r.n,
            },
        });
    }
    result.kernel_name = if with_bias {
        "test_explicit_bf16_mma_nt128_bias"
    } else {
        "test_explicit_bf16_mma_nt128_plain"
    }
    .into();
    result.validate_layouts()?;
    Ok(result)
}

#[test]
fn explicit_bf16_mma_output_and_operand_maps_are_bijective() {
    let mut output = vec![false; 128 * 128];
    for warp in 0..8 {
        for lane in 0usize..32 {
            for index in 0..64 {
                let (r, c) = coordinates(lane, warp, index);
                assert!(r < 128 && c < 128);
                assert!(!output[r * 128 + c]);
                output[r * 128 + c] = true;
            }
        }
    }
    assert!(output.into_iter().all(|v| v));
    // Packed pair loads cover a complete16x16 A and16x8 B fragment.
    let mut a = vec![false; 256];
    let mut b = vec![false; 128];
    for lane in 0usize..32 {
        for q in 0..4 {
            let row = lane / 4 + (q % 2) * 8;
            let col = 2 * (lane % 4) + (q / 2) * 8;
            assert!(col.is_multiple_of(2));
            for i in 0..2 {
                assert!(!a[row * 16 + col + i]);
                a[row * 16 + col + i] = true;
            }
        }
        for q in 0..2 {
            let row = 2 * (lane % 4) + q * 8;
            let col = lane / 4;
            assert!(row.is_multiple_of(2));
            for i in 0..2 {
                assert!(!b[(row + i) * 8 + col]);
                b[(row + i) * 8 + col] = true;
            }
        }
    }
    assert!(a.into_iter().all(|v| v));
    assert!(b.into_iter().all(|v| v));
}

fn fixture() -> Result<(MatMulRegion, PtxTarget)> {
    use crate::tile::MatMulResources;
    let mut target = PtxTarget::sm80();
    target.compute_capability = (8, 9);
    target.matmul_resources = MatMulResources {
        shared_bytes_per_sm: 100 * 1024,
        threads_per_sm: 1536,
        ..MatMulResources::default()
    };
    Ok((fixture_for(4096, 1408, target)?, target))
}
fn fixture_for(m: usize, n: usize, target: PtxTarget) -> Result<MatMulRegion> {
    use crate::{tensor::TensorExpr, tile::MatMulPrecision};
    let a = TensorExpr::<half::bf16>::input("a", vec![m, 512]);
    let w = TensorExpr::<half::bf16>::input("w", vec![n, 512]).transpose();
    let bias = TensorExpr::<half::bf16>::input("bias", vec![n])
        .reshape(vec![1, n])
        .broadcast(vec![m, n]);
    let graph: TensorGraph<half::bf16> = (a.matmul(w) + bias).into();
    let plan = crate::ptx::PtxExecutionPlan::build_with_reduction_mode(
        &graph,
        crate::ptx::PtxReductionMode::DeterministicTree,
    )?;
    ensure!(
        plan.matmul_regions().len() == 1,
        "one actual rebased linear required"
    );
    let mut r = plan.matmul_regions()[0].clone();
    r.schedule = MatMulSchedule::select_with_resources(
        m,
        n,
        512,
        DType::BF16,
        target.matmul_capabilities(),
        MatMulPrecision::Tf32x3,
        target.matmul_resources,
    )
    .with_shared_memory_reuse(true);
    r.configure_pipeline(
        MatMulSharedLayout::Padded,
        MatMulPipeline::Auto,
        target.supports_async_copy(),
        target.matmul_resources,
    )?;
    Ok(r)
}
fn evaluate(e: &Expr, lane: usize, warp: usize, bx: usize, by: usize) -> usize {
    match e {
        Expr::Const(v) => usize::try_from(*v).unwrap(),
        Expr::ThreadIdx(Dim::X) => lane,
        Expr::ThreadIdx(Dim::Y) => warp,
        Expr::BlockIdx(Dim::X) => bx,
        Expr::BlockIdx(Dim::Y) => by,
        Expr::Add(a, b) => evaluate(a, lane, warp, bx, by) + evaluate(b, lane, warp, bx, by),
        Expr::Mul(a, b) => evaluate(a, lane, warp, bx, by) * evaluate(b, lane, warp, bx, by),
        Expr::FloorDiv(a, b) => evaluate(a, lane, warp, bx, by) / b,
        Expr::Mod(a, b) => evaluate(a, lane, warp, bx, by) % b,
        _ => panic!("unexpected direct-store index expression"),
    }
}
#[test]
fn explicit_bf16_mma_preserves_operand_prefix_drain_and_has_no_exchange() -> Result<()> {
    let (r, target) = fixture()?;
    let before = format!("{r:?}");
    for bias in [false, true] {
        let (old, _, plain) = ir::candidate(&r, target)?;
        let reference = if bias { old } else { plain };
        let first = reference
            .body
            .stmts
            .iter()
            .position(|s| matches!(s, Stmt::StoreAccumulatorWindow { .. }))
            .unwrap();
        let result = candidate(&r, target, bias)?;
        let mut restored = result.body.stmts[..first].to_vec();
        for s in &mut restored {
            if let Stmt::AllocTile {
                layout,
                space: MemorySpace::Fragment,
                ..
            } = s
            {
                *layout = reference
                    .body
                    .stmts
                    .iter()
                    .find_map(|s| match s {
                        Stmt::AllocTile {
                            layout,
                            space: MemorySpace::Fragment,
                            ..
                        } => Some(*layout),
                        _ => None,
                    })
                    .unwrap();
            }
        }
        assert_eq!(
            format!("{restored:?}"),
            format!("{:?}", &reference.body.stmts[..first])
        );
        assert_eq!(count_windows(&result.body), 0);
        assert_eq!(result.shared_mem_bytes, 40960);
        assert_eq!(
            result
                .body
                .stmts
                .iter()
                .filter(|s| matches!(s, Stmt::StoreGlobalPredicated { .. }))
                .count(),
            64
        );
        assert_eq!(
            result
                .body
                .stmts
                .iter()
                .filter(|s| matches!(
                    s,
                    Stmt::SetScalar {
                        value: ScalarExpr::Binary {
                            op: ScalarBinaryOp::AddRn,
                            ..
                        },
                        ..
                    }
                ))
                .count(),
            if bias { 64 } else { 0 }
        );
        for (index, (row, col, element)) in result
            .body
            .stmts
            .iter()
            .filter_map(|s| match s {
                Stmt::StoreGlobalPredicated {
                    row,
                    col,
                    element_index,
                    ..
                } => Some((row, col, element_index)),
                _ => None,
            })
            .enumerate()
        {
            for warp in 0..8 {
                for lane in 0..32 {
                    let (local_row, local_col) = coordinates(lane, warp, index);
                    for (bx, by) in [(0, 0), (r.n / 128 - 1, r.m / 128 - 1)] {
                        let rr = evaluate(row, lane, warp, bx, by);
                        let cc = evaluate(col, lane, warp, bx, by);
                        assert_eq!((rr, cc), (128 * by + local_row, 128 * bx + local_col));
                        assert!(rr < r.m && cc < r.n);
                        assert_eq!(evaluate(element, lane, warp, bx, by), rr * r.n + cc);
                    }
                }
            }
        }
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.add_matmul_regions(std::slice::from_ref(&r))?;
        tiles.matmul_region_kernels[0].ir = result;
        tiles.optimize_indices()?;
        let ptx = PtxGraph::from(tiles).with_target(target).module_source();
        assert_eq!(
            ptx.matches("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
                .count(),
            32
        );
        assert!(!ptx.contains("wmma."));
        assert!(!ptx.contains("ld.shared.f32"));
        // Shared operand pairs remain explicit 32-bit packed loads, while
        // every global output uses the existing scalar BF16 RN store helper.
        assert_eq!(ptx.matches("ld.shared.b32").count(), 48);
        assert!(ptx.contains("cvt.rn.bf16.f32"));
    }
    assert_eq!(before, format!("{r:?}"));
    Ok(())
}
