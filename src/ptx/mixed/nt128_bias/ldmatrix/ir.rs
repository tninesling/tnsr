//! Qualified coalesced ldmatrix IR: exact original operand pipeline and K16 order.
use super::*;
use crate::tile::{Expr, MemorySpace, TileLayout};
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
pub(super) fn candidate(r: &MatMulRegion, target: PtxTarget) -> Result<TileIR> {
    let (old, schedule, plain) = super::super::ir::candidate(r, target)?;
    ensure!(
        schedule.effective_grid_group_m() <= 1,
        "ungrouped original coordinates required"
    );
    let mut result = old;
    let _ = plain;
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
                && body.stmts.len() == 5
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
            *layout = TileLayout::Bf16MmaAccumulator128Ldmatrix;
        }
    }
    let accumulator = accumulator.context("missing accumulator")?;
    // The deleted suffix is specifically the qualified shared-window epilogue,
    // not arbitrary arithmetic. Its builder is independently guarded by ir::candidate.
    result.body.stmts.truncate(first);
    let scalar = super::super::ir::fresh_variable(&result.body)?;
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
    result.body.stmts.push(Stmt::StoreBf16MmaCoalesced {
        src: accumulator,
        output_param: "output_0".into(),
        bias_param: Some("input_2".into()),
        m: r.m,
        n: r.n,
    });
    result.kernel_name = "test_ldmatrix_mma_nt128_coalesced_bias".into();
    result.validate_layouts()?;
    Ok(result)
}
