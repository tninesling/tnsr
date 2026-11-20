//! Disconnected paired-online forward leaf, S512/D64, Q64/K64.
//!
//! New numerical policy: F32 QK, BF16 unnormalized probabilities, explicit
//! RN numerator rescale and online m/l. Causal block pruning deliberately omits
//! future-key blocks (including their nonfinite propagation). Within a visited
//! block the additive mask is evaluated even for nonfinite scores. No zero,
//! empty-row or nonfinite shortcut is taken. This must only execute as part of
//! the separately qualified complete forward/backward pair.
use super::{
    common::{Stage, finish_staging, stage_with_transport},
    leaf::{ReadLayout, Shape, Transport},
};
use crate::{
    tensor::{BinaryOp, ReduceOp, UnaryOp},
    tile::*,
};
use anyhow::Result;

const TILE: usize = 64;
const SEQUENCE: usize = 512;
const THREADS: usize = 256;
pub(super) const SHARED_BYTES: usize = 48 * 1024;
fn ix(n: usize) -> Expr {
    Expr::Const(n as i64)
}
fn div(x: Expr, n: usize) -> Expr {
    Expr::FloorDiv(Box::new(x), n)
}
fn rem(x: Expr, n: usize) -> Expr {
    Expr::Mod(Box::new(x), n)
}
fn global(n: &str) -> ScalarMemory {
    ScalarMemory::Global(n.into())
}
fn sm(v: TileVar) -> ScalarMemory {
    ScalarMemory::Shared(v)
}
fn expr(op: ScalarBinaryOp, a: impl Into<ScalarExpr>, b: impl Into<ScalarExpr>) -> ScalarExpr {
    ScalarExpr::binary(op, a, b)
}
fn rn(
    b: &mut ScalarBuilder,
    op: ScalarBinaryOp,
    a: impl Into<ScalarExpr>,
    c: impl Into<ScalarExpr>,
) -> TileVar {
    b.value(expr(op, a, c))
}
fn exp(b: &mut ScalarBuilder, a: TileVar, c: TileVar) -> TileVar {
    let d = rn(b, ScalarBinaryOp::SubRn, a, c);
    b.value(ScalarExpr::unary(UnaryOp::Exp, d))
}
fn view(
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
fn schedule(a: SharedLayout, rhs: SharedLayout) -> MatMulSchedule {
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
fn fragment(b: &mut ScalarBuilder, s: MatMulSchedule) -> TileVar {
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
fn convert(b: &mut ScalarBuilder, dest: TileVar, src: TileVar) {
    b.stmts.push(Stmt::ConvertLayout {
        dest,
        src,
        coordinates: None,
    });
}
fn mma(b: &mut ScalarBuilder, dest: TileVar, a: TileVar, rhs: TileVar, schedule: MatMulSchedule) {
    b.stmts.push(Stmt::MatMul {
        dest,
        a,
        b: rhs,
        layout: MatMulLayout::NN,
        schedule,
    });
}
fn distribute(
    b: &mut ScalarBuilder,
    prefix: &str,
    f: impl FnOnce(&mut ScalarBuilder, Expr) -> Result<()>,
) -> Result<()> {
    let name = b.loop_name(prefix);
    b.fold(name, ix(TILE * TILE / THREADS), vec![], |b, r| {
        f(
            b,
            r * THREADS + Expr::ThreadIdx(Dim::Y) * 32usize + Expr::ThreadIdx(Dim::X),
        )
    })
}
/// Read canonical Q/K/V storage; K's logical transpose was normalized by leaf::prove.
fn input_index(shape: Shape, layout: ReadLayout, row: Expr, col: Expr) -> Expr {
    let bh = Expr::BlockIdx(Dim::Y);
    match layout {
        ReadLayout::HeadMajor => (bh * SEQUENCE + row) * TILE + col,
        ReadLayout::SequenceMajor => {
            (div(bh.clone(), shape.heads) * SEQUENCE + row) * (shape.heads * TILE)
                + rem(bh, shape.heads) * TILE
                + col
        }
    }
}
fn stats_index(query: Expr) -> Expr {
    (Expr::BlockIdx(Dim::Y) * SEQUENCE + query) * 3usize
}
fn reduce(b: &mut ScalarBuilder, v: TileVar, op: ReduceOp) -> TileVar {
    let leader = b.warp_reduce(v, op);
    b.warp_broadcast_leader(leader)
}

/// Parameters: Q16,K16,V16,dense BHSD O16, interleaved F32 [BH,S,3].
/// Forward defines statistics slots0/1 only; the paired dQ kernel defines2.
/// All8 warps execute every CTA barrier; no conditional masked-memory shortcut.
pub(super) fn ir(shape: Shape, layout: ReadLayout) -> Result<TileIR> {
    ir_with_transport(shape, layout, Transport::Scalar)
}

pub(super) fn ir_with_transport(
    shape: Shape,
    layout: ReadLayout,
    transport: Transport,
) -> Result<TileIR> {
    Shape::new(shape.batch, shape.heads)?;
    let mut b = ScalarBuilder::default();
    let arena = b.var();
    b.stmts.push(Stmt::AllocSharedArena {
        arena,
        bytes: SHARED_BYTES,
    });
    let row = SharedLayout::row_major(TILE);
    let key = SharedLayout {
        row_stride: TILE,
        column_major: true,
        xor_mask: 0,
    };
    let q = view(&mut b, arena, 0, DType::BF16, row);
    let k = view(&mut b, arena, 8192, DType::BF16, key);
    let v = view(&mut b, arena, 8192, DType::BF16, row);
    let score = view(&mut b, arena, 16384, DType::F32, row);
    // Each P16 row occupies only the first128B of its F32 score row256B.
    let p = view(
        &mut b,
        arena,
        16384,
        DType::BF16,
        SharedLayout::row_major(128),
    );
    let numerator = view(&mut b, arena, 32768, DType::F32, row);
    let qs = schedule(row, key);
    let ps = schedule(SharedLayout::row_major(128), row);
    let qk = fragment(&mut b, qs);
    let pv = fragment(&mut b, ps);
    let query_base = Expr::BlockIdx(Dim::X) * TILE;
    if transport == Transport::Scalar {
        distribute(&mut b, "initialize", |b, i| {
            let query = query_base.clone() + div(i.clone(), TILE);
            let value = b.load(
                global("query"),
                input_index(shape, layout, query, rem(i.clone(), TILE)),
            );
            b.store(sm(q), i.clone(), value);
            let zero = b.value(ScalarExpr::Constant(0.));
            b.store(sm(numerator), i, zero);
            Ok(())
        })?;
    } else {
        stage_with_transport(
            &mut b,
            shape,
            layout,
            Stage {
                prefix: "initialize_query",
                source: "query",
                row_base: query_base.clone(),
                dest: q,
                column_major: false,
            },
            transport,
        )?;
        distribute(&mut b, "initialize_numerator", |b, i| {
            let zero = b.value(ScalarExpr::Constant(0.));
            b.store(sm(numerator), i, zero);
            Ok(())
        })?;
    }
    let rows = b.loop_name("initialize_statistics");
    b.fold(rows, ix(TILE / 8), vec![], |b, r| {
        let row = r * 8usize + Expr::ThreadIdx(Dim::Y);
        b.when(
            ScalarPredicate::IndexLt {
                lhs: Expr::ThreadIdx(Dim::X),
                rhs: ix(1),
            },
            |b| {
                let index = stats_index(query_base.clone() + row);
                let neg = b.value(ScalarExpr::Constant(f32::NEG_INFINITY));
                let zero = b.value(ScalarExpr::Constant(0.));
                b.store(global("statistics"), index.clone(), neg);
                b.store(global("statistics"), index + ix(1), zero);
                Ok(())
            },
        )
    })?;
    finish_staging(&mut b, transport);
    b.barrier();
    let keys = b.loop_name("causal_keys");
    // Query CTA n visits blocks0..=n, including full diagonal with additive mask.
    b.fold(keys, Expr::BlockIdx(Dim::X) + ix(1), vec![], |b, tile| {
        let key_base = tile * TILE;
        if transport == Transport::Scalar {
            distribute(b, "stage_key", |b, i| {
                let key_row = key_base.clone() + div(i.clone(), TILE);
                let value = b.load(
                    global("key"),
                    input_index(shape, layout, key_row, rem(i.clone(), TILE)),
                );
                b.store(sm(k), i, value);
                Ok(())
            })?;
        } else {
            stage_with_transport(
                b,
                shape,
                layout,
                Stage {
                    prefix: "stage_key",
                    source: "key",
                    row_base: key_base.clone(),
                    dest: k,
                    column_major: true,
                },
                transport,
            )?;
        }
        finish_staging(b, transport);
        b.barrier();
        b.stmts.push(Stmt::Zero { tile: qk });
        mma(b, qk, q, k, qs);
        convert(b, score, qk);
        b.barrier(); // QK readers retire before K arena becomes V.
        let rows = b.loop_name("online_rows");
        b.fold(rows, ix(TILE / 8), vec![], |b, r| {
            let row = r * 8usize + Expr::ThreadIdx(Dim::Y);
            let query = query_base.clone() + row.clone();
            let lane = Expr::ThreadIdx(Dim::X);
            let mut scores = Vec::with_capacity(2);
            let maximum = b.value(ScalarExpr::Constant(f32::NEG_INFINITY));
            for group in 0..2 {
                let col = lane.clone() + ix(group * 32);
                let raw = b.load(sm(score), row.clone() * TILE + col.clone());
                let scaled = rn(b, ScalarBinaryOp::MulRn, raw, ScalarExpr::Constant(0.125));
                let mask = ScalarExpr::select(
                    ScalarPredicate::IndexLt {
                        lhs: key_base.clone() + col,
                        rhs: query.clone() + ix(1),
                    },
                    ScalarExpr::Constant(0.),
                    ScalarExpr::Constant(f32::NEG_INFINITY),
                );
                let s = rn(b, ScalarBinaryOp::AddRn, scaled, mask);
                b.set(maximum, expr(ScalarBinaryOp::Max, maximum, s));
                scores.push(s);
            }
            // Explicit WAR phase boundary: every warp loads both F32 score
            // halves before any lane may overwrite another lane's score bytes
            // through the padded P16 alias. Shuffle is not a memory fence.
            // All8 warps execute all8 row rounds and this unconditional join.
            b.barrier();
            let block_max = reduce(b, maximum, ReduceOp::Max);
            let old_m = b.value(ScalarExpr::Constant(0.));
            let old_l = b.value(ScalarExpr::Constant(0.));
            b.when(
                ScalarPredicate::IndexLt {
                    lhs: lane.clone(),
                    rhs: ix(1),
                },
                |b| {
                    let index = stats_index(query.clone());
                    let m = b.load(global("statistics"), index.clone());
                    let l = b.load(global("statistics"), index + ix(1));
                    b.set(old_m, m);
                    b.set(old_l, l);
                    Ok(())
                },
            )?;
            let old_m = b.warp_broadcast_leader(old_m);
            let old_l = b.warp_broadcast_leader(old_l);
            let next = b.value(expr(ScalarBinaryOp::Max, old_m, block_max));
            let alpha = exp(b, old_m, next);
            let sum = b.value(ScalarExpr::Constant(0.));
            for (group, s) in scores.into_iter().enumerate() {
                let weight = exp(b, s, next);
                b.set(sum, expr(ScalarBinaryOp::AddRn, sum, weight));
                b.store(
                    sm(p),
                    row.clone() * 128usize + lane.clone() + ix(group * 32),
                    weight,
                );
                let col = lane.clone() + ix(group * 32);
                let old = b.load(sm(numerator), row.clone() * TILE + col.clone());
                let rescaled = rn(b, ScalarBinaryOp::MulRn, alpha, old);
                b.store(sm(numerator), row.clone() * TILE + col, rescaled);
            }
            let block_sum = reduce(b, sum, ReduceOp::Sum);
            let scaled_l = rn(b, ScalarBinaryOp::MulRn, alpha, old_l);
            let next_l = rn(b, ScalarBinaryOp::AddRn, scaled_l, block_sum);
            b.when(
                ScalarPredicate::IndexLt {
                    lhs: lane,
                    rhs: ix(1),
                },
                |b| {
                    let index = stats_index(query);
                    b.store(global("statistics"), index.clone(), next);
                    b.store(global("statistics"), index + ix(1), next_l);
                    Ok(())
                },
            )?;
            Ok(())
        })?;
        if transport == Transport::Scalar {
            distribute(b, "stage_value", |b, i| {
                let key_row = key_base.clone() + div(i.clone(), TILE);
                let value = b.load(
                    global("value"),
                    input_index(shape, layout, key_row, rem(i.clone(), TILE)),
                );
                b.store(sm(v), i, value);
                Ok(())
            })?;
        } else {
            stage_with_transport(
                b,
                shape,
                layout,
                Stage {
                    prefix: "stage_value",
                    source: "value",
                    row_base: key_base.clone(),
                    dest: v,
                    column_major: false,
                },
                transport,
            )?;
        }
        finish_staging(b, transport);
        b.barrier(); // publishes P, V and rescaled numerator to all MMA warps.
        convert(b, pv, numerator);
        mma(b, pv, p, v, ps);
        convert(b, numerator, pv);
        b.barrier(); // numerator and V readers retire before next block.
        Ok(())
    })?;
    distribute(&mut b, "store_output", |b, i| {
        let query = query_base.clone() + div(i.clone(), TILE);
        let num = b.load(sm(numerator), i.clone());
        let den = b.load(global("statistics"), stats_index(query.clone()) + ix(1));
        let output = b.value(expr(ScalarBinaryOp::Arithmetic(BinaryOp::Div), num, den));
        b.store(
            global("output_0"),
            (Expr::BlockIdx(Dim::Y) * SEQUENCE + query) * TILE + rem(i, TILE),
            output,
        );
        Ok(())
    })?;
    Ok(TileIR {
        kernel_name: transport.kernel_name(format!(
            "training_attention_online_forward_q64_k64_{}",
            match layout {
                ReadLayout::HeadMajor => "bhsd",
                ReadLayout::SequenceMajor => "bshd",
            }
        )),
        params: [
            ("query", DType::BF16, true),
            ("key", DType::BF16, true),
            ("value", DType::BF16, true),
            ("output_0", DType::BF16, false),
            ("statistics", DType::F32, false),
        ]
        .into_iter()
        .map(|(name, dtype, is_input)| KernelParam {
            name: name.into(),
            dtype,
            is_input,
        })
        .collect(),
        body: Block { stmts: b.stmts },
        shared_mem_bytes: SHARED_BYTES,
    })
}

#[cfg(test)]
mod tests;

/// Experimental statistics storage. Global delegates to the unchanged accepted
/// generator; Registers is available to the opt-in, resource-qualified pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StatisticsStorage {
    Global,
    Registers,
}
pub(super) fn ir_with_statistics(
    shape: Shape,
    layout: ReadLayout,
    transport: Transport,
    statistics: StatisticsStorage,
) -> Result<TileIR> {
    let global = match transport {
        Transport::Scalar => ir(shape, layout)?,
        Transport::Async16 => ir_with_transport(shape, layout, transport)?,
    };
    match statistics {
        StatisticsStorage::Global => Ok(global),
        StatisticsStorage::Registers => register_stats::transform(global),
    }
}
mod register_stats;
