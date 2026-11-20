//! Local-test/IR-builder warp-row S512/D64 attention normalization.
//!
//! Uses explicit bitwise `Stmt::WarpBroadcastLeader`. Existing
//! WarpReduce{broadcast:true} is a butterfly and MUST remain unchanged.
//!
//! A block is (32,8,1): each whole warp owns one row and every lane is active.
//! Physical lane l holds virtual128 contributors l+32*g, g=0..3. Each virtual
//! contributor retains four ordered keys l+32*g+128*p, p=0..3. The first two
//! old tree stages are exactly (v0 op v2) op (v1 op v3). Canonical shuffle-down
//! 16,8,4,2,1 then computes lane0 using precisely the remaining old DAG. Only
//! lane0 is broadcast, bitwise, with shfl.sync.idx lane0/fullmask. Nonleader
//! shuffle values are NOT an all-reduction result and must never be stored.
//!
//! Keep existing recognizers, identity-binding proofs, global mask argument,
//! output allocation/lifetimes and all BF16 boundaries. No masked-key skip,
//! zero/NaN shortcut, denominator rewrite, Flash delta or changed dot tree.
//! Final integration must check actual compiled256-thread resource feasibility
//! and fall back to the complete old row kernel when rejected. This module has
//! no Mixed runtime hooks, options, counters or compilation-signature changes.
#[cfg(test)]
use super::*;
use crate::{
    tensor::{BinaryOp, ReduceOp, UnaryOp},
    tile::{Dim, Expr, ScalarBinaryOp, ScalarBuilder, ScalarExpr, ScalarMemory, TileVar},
};

#[cfg(test)]
use crate::tile::{Block, KernelParam, TileIR};

const COLUMNS: usize = 512;
#[cfg(test)]
const ROWS_PER_BLOCK: usize = 8;

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct WarpRowsLaunch {
    pub rows: usize,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
}

#[cfg(test)]
pub(super) fn geometry(shape: &[usize], head_width: usize) -> Option<WarpRowsLaunch> {
    let &[batch, heads, queries, keys] = shape else {
        return None;
    };
    if head_width != 64 || queries != COLUMNS || keys != COLUMNS || batch == 0 || heads == 0 {
        return None;
    }
    let bh = batch.checked_mul(heads)?;
    if bh < 64 {
        return None;
    }
    let rows = bh.checked_mul(COLUMNS)?;
    rows.checked_mul(COLUMNS)?;
    if !rows.is_multiple_of(ROWS_PER_BLOCK) {
        return None;
    }
    let blocks = u32::try_from(rows / ROWS_PER_BLOCK).ok()?;
    if blocks > i32::MAX as u32 {
        return None;
    }
    Some(WarpRowsLaunch {
        rows,
        grid: (blocks, 1, 1),
        block: (32, 8, 1),
    })
}

fn arithmetic(op: BinaryOp, a: impl Into<ScalarExpr>, b: impl Into<ScalarExpr>) -> ScalarExpr {
    ScalarExpr::binary(ScalarBinaryOp::Arithmetic(op), a, b)
}
#[cfg(test)]
fn row() -> Expr {
    Expr::BlockIdx(Dim::X) * ROWS_PER_BLOCK + Expr::ThreadIdx(Dim::Y)
}
fn column(group: usize, part: usize) -> Expr {
    Expr::ThreadIdx(Dim::X) + Expr::Const((32 * group + 128 * part) as i64)
}
fn merge_and_broadcast(b: &mut ScalarBuilder, partial: [TileVar; 4], op: ReduceOp) -> TileVar {
    let combine = if op == ReduceOp::Max {
        ScalarBinaryOp::Max
    } else {
        ScalarBinaryOp::Arithmetic(BinaryOp::Add)
    };
    let a02 = b.value(ScalarExpr::binary(combine, partial[0], partial[2]));
    let a13 = b.value(ScalarExpr::binary(combine, partial[1], partial[3]));
    let combined = b.value(ScalarExpr::binary(combine, a02, a13));
    let leader = b.warp_reduce(combined, op); // canonical descending shuffle, NOT butterfly
    b.warp_broadcast_leader(leader) // explicit bitwise lane0/fullmask primitive
}

/// Each memory uses its declared physical dtype. Row expressions are row
/// indices, not byte offsets. The caller must provide BF16 scores, an F32 mask,
/// F32 probability storage and BF16 narrowed storage/checkpoint.
/// All sixteen score values per lane are retained
/// before any output store, permitting an in-place shared BF16 score/P16 tile.
pub(super) struct ForwardRowBindings {
    pub scores: ScalarMemory,
    pub score_row: Expr,
    pub mask: ScalarMemory,
    pub mask_row: Expr,
    pub probability: ScalarMemory,
    pub probability_row: Expr,
    pub narrowed: ScalarMemory,
    pub narrowed_row: Expr,
    pub narrowed_checkpoint: Option<(ScalarMemory, Expr)>,
}

/// Emit one D64/S512 row per whole active warp, with canonical virtual128
/// arithmetic. The caller owns shared publication/reuse barriers outside this
/// function; every lane must execute each shuffle under the same warp control.
pub(super) fn forward_row(b: &mut ScalarBuilder, bindings: ForwardRowBindings) {
    let maximum = std::array::from_fn(|_| b.value(ScalarExpr::Constant(f32::NEG_INFINITY)));
    let mut scores = Vec::with_capacity(16);
    for (group, &maximum_var) in maximum.iter().enumerate() {
        for part in 0..4 {
            let col = column(group, part);
            let qk = b.load(
                bindings.scores.clone(),
                bindings.score_row.clone() * COLUMNS + col.clone(),
            );
            let mask = b.load(
                bindings.mask.clone(),
                bindings.mask_row.clone() * COLUMNS + col,
            );
            let scaled = b.value(arithmetic(BinaryOp::Mul, qk, ScalarExpr::Constant(0.125)));
            let score = b.value(arithmetic(BinaryOp::Add, scaled, mask));
            b.set(
                maximum_var,
                ScalarExpr::binary(ScalarBinaryOp::Max, maximum_var, score),
            );
            scores.push(score);
        }
    }
    let maximum = merge_and_broadcast(b, maximum, ReduceOp::Max);
    let denominator = std::array::from_fn(|_| b.value(ScalarExpr::Constant(0.)));
    let mut exponentials = Vec::with_capacity(16);
    for (index, score) in scores.into_iter().enumerate() {
        let shifted = b.value(arithmetic(BinaryOp::Sub, score, maximum));
        let exp = b.value(ScalarExpr::unary(UnaryOp::Exp, shifted));
        let group = index / 4;
        b.set(
            denominator[group],
            arithmetic(BinaryOp::Add, denominator[group], exp),
        );
        exponentials.push(exp);
    }
    let denominator = merge_and_broadcast(b, denominator, ReduceOp::Sum);
    for (index, exp) in exponentials.into_iter().enumerate() {
        let p = b.value(arithmetic(BinaryOp::Div, exp, denominator));
        let col = column(index / 4, index % 4);
        b.store(
            bindings.probability.clone(),
            bindings.probability_row.clone() * COLUMNS + col.clone(),
            p,
        );
        b.store(
            bindings.narrowed.clone(),
            bindings.narrowed_row.clone() * COLUMNS + col.clone(),
            p,
        );
        if let Some((memory, row)) = &bindings.narrowed_checkpoint {
            b.store(memory.clone(), row.clone() * COLUMNS + col, p);
        }
    }
}

#[cfg(test)]
pub(super) fn forward_ir(scale: f32) -> Result<TileIR> {
    ensure!(
        scale.to_bits() == 0.125f32.to_bits(),
        "warp attention requires the proven D64 scale"
    );
    let mut b = ScalarBuilder::default();
    let row = row();
    forward_row(
        &mut b,
        ForwardRowBindings {
            scores: ScalarMemory::Global("qk".into()),
            score_row: row.clone(),
            mask: ScalarMemory::Global("mask".into()),
            mask_row: Expr::Mod(Box::new(row.clone()), COLUMNS),
            probability: ScalarMemory::Global("probability".into()),
            probability_row: row.clone(),
            narrowed: ScalarMemory::Global("narrowed".into()),
            narrowed_row: row,
            narrowed_checkpoint: None,
        },
    );
    Ok(TileIR {
        kernel_name: "attention_softmax_warp512".into(),
        params: vec![
            KernelParam {
                name: "qk".into(),
                dtype: DType::BF16,
                is_input: true,
            },
            KernelParam {
                name: "mask".into(),
                dtype: DType::F32,
                is_input: true,
            },
            KernelParam {
                name: "probability".into(),
                dtype: DType::F32,
                is_input: false,
            },
            KernelParam {
                name: "narrowed".into(),
                dtype: DType::BF16,
                is_input: false,
            },
        ],
        body: Block { stmts: b.stmts },
        shared_mem_bytes: 0,
    })
}

#[cfg(test)]
pub(super) fn backward_ir() -> Result<TileIR> {
    let mut b = ScalarBuilder::default();
    let row = row();
    let dot = std::array::from_fn(|_| b.value(ScalarExpr::Constant(0.)));
    let mut pairs = Vec::with_capacity(16);
    for (group, &dot_var) in dot.iter().enumerate() {
        for part in 0..4 {
            let index = row.clone() * COLUMNS + column(group, part);
            let p = b.load(ScalarMemory::Global("probability".into()), index.clone());
            let g = b.load(ScalarMemory::Global("incoming".into()), index);
            let product = b.value(arithmetic(BinaryOp::Mul, p, g));
            // Retain exactly the old Mul+Add adjacency and contraction
            // opportunity. Each of the four independent dots visits p0..3.
            b.set(dot_var, arithmetic(BinaryOp::Add, dot_var, product));
            pairs.push((p, g));
        }
    }
    let dot = merge_and_broadcast(&mut b, dot, ReduceOp::Sum);
    for (index, (p, g)) in pairs.into_iter().enumerate() {
        let centered = b.value(arithmetic(BinaryOp::Sub, g, dot));
        let derivative = b.value(arithmetic(BinaryOp::Mul, p, centered));
        let scaled = b.value(arithmetic(
            BinaryOp::Mul,
            derivative,
            ScalarExpr::Constant(0.125),
        ));
        b.store(
            ScalarMemory::Global("narrowed".into()),
            row.clone() * COLUMNS + column(index / 4, index % 4),
            scaled,
        );
    }
    Ok(TileIR {
        kernel_name: "attention_softmax_backward_warp512".into(),
        params: vec![
            KernelParam {
                name: "probability".into(),
                dtype: DType::F32,
                is_input: true,
            },
            KernelParam {
                name: "incoming".into(),
                dtype: DType::BF16,
                is_input: true,
            },
            KernelParam {
                name: "narrowed".into(),
                dtype: DType::BF16,
                is_input: false,
            },
        ],
        body: Block { stmts: b.stmts },
        shared_mem_bytes: 0,
    })
}

#[cfg(test)]
mod tests;
