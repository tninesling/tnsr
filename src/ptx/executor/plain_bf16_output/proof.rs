//! Narrow exact output-interface proof, independent of GPU allocations.
use crate::tile::{
    DType, Dim, Expr, MatMulAccumulatorExchange, MatMulFragmentLoad, MatMulRegion, Stmt, TileIR,
};
use anyhow::{Context, Result, ensure};
#[derive(Clone, Copy)]
pub(super) struct Scope {
    pub m: usize,
    pub n: usize,
}
const BASE: &str = "plain_output_base";
pub(super) fn guard(r: &MatMulRegion) -> Result<()> {
    let s = r.schedule;
    ensure!(
        r.m.checked_mul(r.n).is_some_and(|v| v <= i64::MAX as usize)
            && r.n.checked_mul(64).is_some_and(|v| v <= i64::MAX as usize),
        "output index extent overflow"
    );
    ensure!(
        r.batch_shape.is_empty()
            && r.epilogue_operations.is_empty()
            && r.outputs.len() == 1
            && r.outputs[0].value == r.matmul_value
            && r.inputs.len() == 2
            && r.output_shape == [r.m, r.n]
            && r.lhs_shape == [r.m, r.k]
            && r.rhs_shape == [r.k, r.n]
            && r.m > 0
            && r.n > 0
            && r.m.is_multiple_of(64)
            && r.n.is_multiple_of(64)
            && r.k >= 512
            && r.k.is_multiple_of(32)
            && s.logical_shape
                == (crate::tile::MatMulTile {
                    m: r.m,
                    n: r.n,
                    k: r.k
                })
            && s.plan == crate::tile::MatMulPlan::TensorCoreBF16
            && s.block_tile
                == (crate::tile::MatMulTile {
                    m: 64,
                    n: 64,
                    k: 32
                })
            && s.block_threads == (32, 8, 1)
            && s.warp_topology == (2, 2)
            && s.warp_fragments == (2, 2)
            && s.accumulator_exchange == MatMulAccumulatorExchange::FullTile
            && s.fragment_load == MatMulFragmentLoad::PreloadAll
            && s.storage_dtype == DType::BF16
            && s.operand_dtype == DType::BF16
            && s.pipeline_stages == 2
            && s.async_operands == (true, true)
            && s.copy_bytes == 16
            && s.reuse_shared_memory
            && s.effective_grid_group_m() <= 1
            && s.operand_layouts
                == (
                    crate::tile::SharedLayout::padded(32, 8)?,
                    crate::tile::SharedLayout::padded(64, 8)?
                ),
        "probe requires accepted full-MN plain rank2 BF16 geometry"
    );
    for (v, shape) in [(r.lhs_value, &r.lhs_shape), (r.rhs_value, &r.rhs_shape)] {
        let i = r
            .inputs
            .iter()
            .find(|i| i.value == v)
            .context("missing operand")?;
        ensure!(
            crate::tile::ordinary_bf16_block::aligned_operand(i, shape, false).is_some(),
            "unproved contiguous NN source"
        );
    }
    Ok(())
}
fn base_expr(n: usize) -> Expr {
    let tid = Expr::ThreadIdx(Dim::Y) * 32usize + Expr::ThreadIdx(Dim::X);
    Expr::BlockIdx(Dim::Y) * (64 * n)
        + Expr::BlockIdx(Dim::X) * 64usize
        + Expr::FloorDiv(Box::new(tid.clone()), 64) * n
        + Expr::Mod(Box::new(tid), 64)
}
// Exact affine coefficients in (1,blockY,blockX), after fixing one
// thread. Reject a non-affine/unresolved expression rather than infer equality
// from corner samples. Nonnegative checked u128 coefficients prove no wrapping
// for every admitted grid coordinate when the checked maximum fits u64.
fn affine_inner(
    e: &Expr,
    tid: usize,
    defs: &std::collections::HashMap<String, Expr>,
    depth: usize,
) -> Option<[u128; 3]> {
    if depth == 0 {
        return None;
    }
    let f = |x: &Expr| affine_inner(x, tid, defs, depth - 1);
    let constant = |v: u128| Some([v, 0, 0]);
    let divide = |a: [u128; 3], d: u128, rem: bool| {
        if d == 0 || !a[1].is_multiple_of(d) || !a[2].is_multiple_of(d) {
            return None;
        }
        Some(if rem {
            [a[0] % d, 0, 0]
        } else {
            [a[0] / d, a[1] / d, a[2] / d]
        })
    };
    match e {
        Expr::Const(v) => constant(u128::try_from(*v).ok()?),
        Expr::Var(v) => f(defs.get(v)?),
        Expr::ThreadIdx(Dim::X) => constant((tid % 32) as u128),
        Expr::ThreadIdx(Dim::Y) => constant((tid / 32) as u128),
        Expr::ThreadIdx(Dim::Z) | Expr::BlockIdx(Dim::Z) => constant(0),
        Expr::BlockIdx(Dim::Y) => Some([0, 1, 0]),
        Expr::BlockIdx(Dim::X) => Some([0, 0, 1]),
        Expr::BlockDim(Dim::X) => constant(32),
        Expr::BlockDim(Dim::Y) => constant(8),
        Expr::BlockDim(Dim::Z) => constant(1),
        Expr::Add(a, b) => {
            let (a, b) = (f(a)?, f(b)?);
            Some([
                a[0].checked_add(b[0])?,
                a[1].checked_add(b[1])?,
                a[2].checked_add(b[2])?,
            ])
        }
        Expr::Sub(a, b) => {
            let (a, b) = (f(a)?, f(b)?);
            Some([
                a[0].checked_sub(b[0])?,
                a[1].checked_sub(b[1])?,
                a[2].checked_sub(b[2])?,
            ])
        }
        Expr::Mul(a, b) => {
            let (mut a, mut b) = (f(a)?, f(b)?);
            if a[1] == 0 && a[2] == 0 {
                std::mem::swap(&mut a, &mut b);
            }
            if b[1] != 0 || b[2] != 0 {
                return None;
            }
            Some([
                a[0].checked_mul(b[0])?,
                a[1].checked_mul(b[0])?,
                a[2].checked_mul(b[0])?,
            ])
        }
        Expr::FloorDiv(a, d) => divide(f(a)?, *d as u128, false),
        Expr::Mod(a, d) => divide(f(a)?, *d as u128, true),
        Expr::ShiftRight(a, n) => divide(f(a)?, 1u128.checked_shl(u32::try_from(*n).ok()?)?, false),
        Expr::BitAnd(a, m) => {
            let d = u128::from(*m) + 1;
            if !d.is_power_of_two() {
                return None;
            }
            divide(f(a)?, d, true)
        }
    }
}
fn prove_stores(ir: &TileIR, case: Scope) -> Result<()> {
    ensure!(
        case.m
            .checked_mul(case.n)
            .is_some_and(|v| v <= i64::MAX as usize)
            && case
                .n
                .checked_mul(64)
                .is_some_and(|v| v <= i64::MAX as usize),
        "output index extent overflow"
    );
    ensure!(
        case.m > 0 && case.n > 0 && case.m.is_multiple_of(64) && case.n.is_multiple_of(64),
        "partial output grid"
    );
    ensure!(
        ir.params.iter().filter(|p| !p.is_input).count() == 1
            && ir
                .params
                .iter()
                .any(|p| p.name == "output_0" && !p.is_input && p.dtype == DType::BF16),
        "non-BF16 output interface"
    );
    fn nested_write(block: &crate::tile::Block) -> bool {
        block.stmts.iter().any(|s| match s {
            Stmt::Store { .. } | Stmt::StoreGlobalPredicated { .. } => true,
            Stmt::ForLoop { body, .. } | Stmt::If { body, .. } => nested_write(body),
            _ => false,
        })
    }
    let mut defs = std::collections::HashMap::new();
    let mut stores = Vec::new();
    for stmt in &ir.body.stmts {
        match stmt {
            Stmt::LetIndex { name, value } => {
                ensure!(
                    defs.insert(name.clone(), value.clone()).is_none(),
                    "shadowed output index binding"
                );
            }
            Stmt::StoreGlobalPredicated {
                row,
                col,
                element_index,
                ..
            } => stores.push((row, col, element_index, defs.clone())),
            Stmt::Store { .. } => anyhow::bail!("unexpected existing unpredicated global store"),
            Stmt::ForLoop { body, .. } | Stmt::If { body, .. } => ensure!(
                !nested_write(body),
                "nested global stores outside scalar output contract"
            ),
            _ => {}
        }
    }
    ensure!(stores.len() == 16, "expected sixteen output stores");
    for (round, (row, col, index, defs)) in stores.iter().enumerate() {
        for tid in 0..256 {
            let t = round * 256 + tid;
            let (lr, lc) = (t / 64, t % 64);
            let expected = [
                lr as u128 * case.n as u128 + lc as u128,
                64 * case.n as u128,
                64,
            ];
            ensure!(
                affine_inner(row, tid, defs, 128) == Some([lr as u128, 64, 0])
                    && affine_inner(col, tid, defs, 128) == Some([lc as u128, 0, 64])
                    && affine_inner(index, tid, defs, 128) == Some(expected),
                "original output SSA is not the exact full-MN affine mapping"
            );
            let max = expected[0]
                + expected[1] * (case.m / 64 - 1) as u128
                + expected[2] * (case.n / 64 - 1) as u128;
            ensure!(
                max < (case.m as u128) * (case.n as u128) && max <= u64::MAX as u128,
                "output index overflow/bounds"
            );
        }
    }
    Ok(())
}
pub(super) fn rewrite(ir: &mut TileIR, case: Scope) -> Result<()> {
    prove_stores(ir, case)?;
    let count = ir
        .body
        .stmts
        .iter()
        .filter(|s| matches!(s, Stmt::StoreGlobalPredicated { .. }))
        .count();
    ensure!(
        count == 16,
        "expected exactly sixteen original scalar stores"
    );
    ensure!(
        !ir.body
            .stmts
            .iter()
            .any(|s| matches!(s,Stmt::LetIndex{name,..} if name==BASE)),
        "duplicate probe binding"
    );
    for s in &ir.body.stmts {
        if let Stmt::StoreGlobalPredicated {
            dest_param, layout, ..
        } = s
        {
            ensure!(
                dest_param == "output_0"
                    && layout.rows == case.m
                    && layout.cols == case.n
                    && layout.row_stride == case.n,
                "unexpected output interface"
            );
        }
    }
    // Validate the complete rewrite before committing any statement. Retain
    // original LetIndex definitions and never rerun placement: all instructions
    // through the exchange/read phase stay identical to the original program.
    let mut result = Vec::with_capacity(ir.body.stmts.len() + 1);
    let mut round = 0;
    for s in ir.body.stmts.clone() {
        if let Stmt::StoreGlobalPredicated {
            dest_param, src, ..
        } = s
        {
            if round == 0 {
                result.push(Stmt::LetIndex {
                    name: BASE.into(),
                    value: base_expr(case.n),
                });
            }
            result.push(Stmt::Store {
                dest_param,
                src,
                row_offset: Expr::Var(BASE.into()) + Expr::Const((round * 4 * case.n) as i64),
                col_offset: Expr::Const(0),
            });
            round += 1;
        } else {
            result.push(s);
        }
    }
    let mut proposed = ir.clone();
    proposed.body.stmts = result;
    proposed.validate_layouts()?;
    *ir = proposed;
    Ok(())
}
