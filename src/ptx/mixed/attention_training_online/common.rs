//! Shared source-only helpers for the disconnected online backward kernels.
use super::leaf::{ReadLayout, Shape, Transport};
use crate::{
    tensor::{ReduceOp, UnaryOp},
    tile::*,
};
use anyhow::Result;
mod async_stage;
pub(super) use async_stage::{Stage, finish_staging, stage_with_transport};
pub(super) const TILE: usize = 64;
pub(super) const SEQUENCE: usize = 512;
pub(super) const THREADS: usize = 256;
pub(super) const SHARED_BYTES: usize = 48 * 1024;
pub(super) fn ix(n: usize) -> Expr {
    Expr::Const(n as i64)
}
pub(super) fn div(x: Expr, n: usize) -> Expr {
    Expr::FloorDiv(Box::new(x), n)
}
pub(super) fn rem(x: Expr, n: usize) -> Expr {
    Expr::Mod(Box::new(x), n)
}
pub(super) fn global(name: &str) -> ScalarMemory {
    ScalarMemory::Global(name.into())
}
pub(super) fn sm(tile: TileVar) -> ScalarMemory {
    ScalarMemory::Shared(tile)
}
pub(super) fn column(stride: usize) -> SharedLayout {
    SharedLayout {
        row_stride: stride,
        column_major: true,
        xor_mask: 0,
    }
}
pub(super) fn rn(
    b: &mut ScalarBuilder,
    op: ScalarBinaryOp,
    a: impl Into<ScalarExpr>,
    rhs: impl Into<ScalarExpr>,
) -> TileVar {
    b.value(ScalarExpr::binary(op, a, rhs))
}
pub(super) fn exp_difference(b: &mut ScalarBuilder, a: TileVar, rhs: TileVar) -> TileVar {
    let difference = rn(b, ScalarBinaryOp::SubRn, a, rhs);
    b.value(ScalarExpr::unary(UnaryOp::Exp, difference))
}
pub(super) fn view(
    b: &mut ScalarBuilder,
    arena: TileVar,
    offset: usize,
    dtype: DType,
    layout: SharedLayout,
) -> TileVar {
    let dest = b.var();
    b.stmts.push(Stmt::ViewSharedArena {
        dest,
        arena,
        byte_offset: offset,
        dtype,
        rows: TILE,
        cols: TILE,
        layout,
    });
    dest
}
pub(super) fn schedule(a: SharedLayout, rhs: SharedLayout) -> MatMulSchedule {
    let tile = MatMulTile {
        m: TILE,
        n: TILE,
        k: TILE,
    };
    MatMulSchedule {
        plan: MatMulPlan::TensorCoreBF16,
        logical_shape: tile,
        grid_group_m: 1,
        block_tile: tile,
        instruction_tile: MatMulTile {
            m: 16,
            n: 16,
            k: 16,
        },
        block_threads: (32, 8, 1),
        warp_topology: (4, 2),
        warp_fragments: (1, 2),
        accumulator_exchange: MatMulAccumulatorExchange::FullTile,
        fragment_load: MatMulFragmentLoad::PreloadAll,
        operand_layouts: (a, rhs),
        pipeline_stages: 1,
        reuse_shared_memory: false,
        copy_bytes: 0,
        async_operands: (false, false),
        operand_dtype: DType::BF16,
        storage_dtype: DType::BF16,
        accumulator_layout: MatMulAccumulatorLayout::WarpFragment,
    }
}
pub(super) fn fragment(b: &mut ScalarBuilder, s: MatMulSchedule) -> TileVar {
    let var = b.var();
    b.stmts.push(Stmt::AllocTile {
        var,
        space: MemorySpace::Fragment,
        layout: s.accumulator_tile_layout(),
        dtype: DType::F32,
        rows: TILE,
        cols: TILE,
    });
    var
}
pub(super) fn mma(
    b: &mut ScalarBuilder,
    dest: TileVar,
    a: TileVar,
    rhs: TileVar,
    s: MatMulSchedule,
) {
    b.stmts.push(Stmt::MatMul {
        dest,
        a,
        b: rhs,
        layout: MatMulLayout::NN,
        schedule: s,
    });
}
pub(super) fn convert(b: &mut ScalarBuilder, dest: TileVar, src: TileVar) {
    b.stmts.push(Stmt::ConvertLayout {
        dest,
        src,
        coordinates: None,
    });
}
pub(super) fn distribute(
    b: &mut ScalarBuilder,
    prefix: &str,
    f: impl FnOnce(&mut ScalarBuilder, Expr) -> Result<()>,
) -> Result<()> {
    let name = b.loop_name(prefix);
    b.fold(name, ix(TILE * TILE / THREADS), vec![], |b, round| {
        f(
            b,
            round * THREADS + Expr::ThreadIdx(Dim::Y) * 32usize + Expr::ThreadIdx(Dim::X),
        )
    })
}
pub(super) fn input_index(shape: Shape, layout: ReadLayout, row: Expr, feature: Expr) -> Expr {
    let bh = Expr::BlockIdx(Dim::Y);
    match layout {
        ReadLayout::HeadMajor => (bh * SEQUENCE + row) * TILE + feature,
        ReadLayout::SequenceMajor => {
            (div(bh.clone(), shape.heads) * SEQUENCE + row) * (shape.heads * TILE)
                + rem(bh, shape.heads) * TILE
                + feature
        }
    }
}
pub(super) fn dense_index(row: Expr, feature: Expr) -> Expr {
    (Expr::BlockIdx(Dim::Y) * SEQUENCE + row) * TILE + feature
}
pub(super) fn statistics_index(row: Expr) -> Expr {
    (Expr::BlockIdx(Dim::Y) * SEQUENCE + row) * 3usize
}
pub(super) fn leader_statistic(b: &mut ScalarBuilder, row: Expr, field: usize) -> Result<TileVar> {
    let value = b.value(ScalarExpr::Constant(0.));
    b.when(
        ScalarPredicate::IndexLt {
            lhs: Expr::ThreadIdx(Dim::X),
            rhs: ix(1),
        },
        |b| {
            let loaded = b.load(global("statistics"), statistics_index(row) + ix(field));
            b.set(value, loaded);
            Ok(())
        },
    )?;
    Ok(b.warp_broadcast_leader(value))
}
pub(super) fn reduce_leader(b: &mut ScalarBuilder, value: TileVar) -> TileVar {
    let value = b.warp_reduce(value, ReduceOp::Sum);
    b.warp_broadcast_leader(value)
}
pub(super) fn stage(
    b: &mut ScalarBuilder,
    prefix: &str,
    name: &str,
    shape: Shape,
    layout: ReadLayout,
    row_base: Expr,
    dest: TileVar,
) -> Result<()> {
    distribute(b, prefix, |b, index| {
        let row = row_base.clone() + div(index.clone(), TILE);
        let feature = rem(index.clone(), TILE);
        let loaded = b.load(global(name), input_index(shape, layout, row, feature));
        b.store(sm(dest), index, loaded);
        Ok(())
    })
}
pub(super) fn masked_score(b: &mut ScalarBuilder, raw: TileVar, query: Expr, key: Expr) -> TileVar {
    let scaled = rn(b, ScalarBinaryOp::MulRn, raw, ScalarExpr::Constant(0.125));
    let mask = ScalarExpr::select(
        ScalarPredicate::IndexLt {
            lhs: key,
            rhs: query + ix(1),
        },
        ScalarExpr::Constant(0.),
        ScalarExpr::Constant(f32::NEG_INFINITY),
    );
    rn(b, ScalarBinaryOp::AddRn, scaled, mask)
}
pub(super) fn param(name: &str, dtype: DType, is_input: bool) -> KernelParam {
    KernelParam {
        name: name.into(),
        dtype,
        is_input,
    }
}
