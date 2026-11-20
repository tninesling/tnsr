//! Complete paired-online backward leaves, not old checkpoint VJPs.
//! dQ owns queries and publishes O16*dO16 delta; dKT/dV owns keys. No atomics.
//! Add/sub/mul are explicit RN. Normalization uses the existing approximate F32
//! division primitive, matching forward; no RN-division claim or new IR.
use super::{
    common::*,
    leaf::{ReadLayout, Shape, Transport},
};
use crate::tile::*;
use anyhow::Result;

fn arena(b: &mut ScalarBuilder) -> TileVar {
    let a = b.var();
    b.stmts.push(Stmt::AllocSharedArena {
        arena: a,
        bytes: SHARED_BYTES,
    });
    a
}

/// All two score and dP values/lane are consumed before an explicit uniform CTA
/// barrier. Only then may either first-half-row P16/DS16 overwrite F32 storage.
fn weights(
    b: &mut ScalarBuilder,
    score: TileVar,
    dp: TileVar,
    ds: TileVar,
    p: Option<TileVar>,
    query_base: Expr,
    key_base: Expr,
) -> Result<()> {
    let rows = b.loop_name("backward_rows");
    b.fold(rows, ix(TILE / 8), vec![], |b, round| {
        let row = round * 8usize + Expr::ThreadIdx(Dim::Y);
        let query = query_base.clone() + row.clone();
        let maximum = leader_statistic(b, query.clone(), 0)?;
        let denominator = leader_statistic(b, query.clone(), 1)?;
        let delta = leader_statistic(b, query.clone(), 2)?;
        let mut values = Vec::with_capacity(2);
        for group in 0..2 {
            let col = Expr::ThreadIdx(Dim::X) + ix(group * 32);
            let raw_score = b.load(sm(score), row.clone() * TILE + col.clone());
            let raw_dp = b.load(sm(dp), row.clone() * TILE + col.clone());
            let masked = masked_score(b, raw_score, query.clone(), key_base.clone() + col.clone());
            let exponential = exp_difference(b, masked, maximum);
            let probability = rn(
                b,
                ScalarBinaryOp::Arithmetic(crate::tensor::BinaryOp::Div),
                exponential,
                denominator,
            );
            let centered = rn(b, ScalarBinaryOp::SubRn, raw_dp, delta);
            let product = rn(b, ScalarBinaryOp::MulRn, probability, centered);
            let derivative = rn(
                b,
                ScalarBinaryOp::MulRn,
                product,
                ScalarExpr::Constant(0.125),
            );
            values.push((col, probability, derivative));
        }
        // WAR retirement, not a shuffle memory-fence assumption. All8warps run
        // all8 rounds, including diagonal rows; the barrier is never predicated.
        b.barrier();
        for (col, probability, derivative) in values {
            let index = row.clone() * 128usize + col;
            if let Some(p) = p {
                b.store(sm(p), index.clone(), probability);
            }
            b.store(sm(ds), index, derivative);
        }
        Ok(())
    })?;
    b.barrier(); // Publish every padded P/DS row before WMMA readers.
    Ok(())
}

fn delta(
    b: &mut ScalarBuilder,
    shape: Shape,
    incoming_layout: ReadLayout,
    query_base: Expr,
) -> Result<()> {
    let rows = b.loop_name("delta_rows");
    b.fold(rows, ix(TILE / 8), vec![], |b, round| {
        let query = query_base.clone() + round * 8usize + Expr::ThreadIdx(Dim::Y);
        let sum = b.value(ScalarExpr::Constant(0.));
        for group in 0..2 {
            let feature = Expr::ThreadIdx(Dim::X) + ix(group * 32);
            let output = b.load(
                global("output_0"),
                dense_index(query.clone(), feature.clone()),
            );
            let incoming = b.load(
                global("incoming"),
                input_index(shape, incoming_layout, query.clone(), feature),
            );
            let product = rn(b, ScalarBinaryOp::MulRn, output, incoming);
            let next = rn(b, ScalarBinaryOp::AddRn, sum, product);
            b.set(sum, next);
        }
        let sum = reduce_leader(b, sum);
        b.when(
            ScalarPredicate::IndexLt {
                lhs: Expr::ThreadIdx(Dim::X),
                rhs: ix(1),
            },
            |b| {
                b.store(global("statistics"), statistics_index(query) + ix(2), sum);
                Ok(())
            },
        )
    })?;
    b.barrier();
    Ok(())
}

/// Q,K,V,O16,dO16,statistics,dQ16. Statistics slots0/1 are read-only; this
/// kernel owns slot2 for its64 query rows. Launch before joint dKT/dV.
pub(super) fn d_query_ir(
    shape: Shape,
    qkv_layout: ReadLayout,
    incoming_layout: ReadLayout,
) -> Result<TileIR> {
    d_query_ir_with_transport(shape, qkv_layout, incoming_layout, Transport::Scalar)
}

pub(super) fn d_query_ir_with_transport(
    shape: Shape,
    qkv_layout: ReadLayout,
    incoming_layout: ReadLayout,
    transport: Transport,
) -> Result<TileIR> {
    Shape::new(shape.batch, shape.heads)?;
    let mut b = ScalarBuilder::default();
    let arena = arena(&mut b);
    let row = SharedLayout::row_major(TILE);
    let col = column(TILE);
    let q = view(&mut b, arena, 0, DType::BF16, row);
    let incoming = view(&mut b, arena, 8192, DType::BF16, row);
    let k = view(&mut b, arena, 16384, DType::BF16, row);
    let kt = view(&mut b, arena, 16384, DType::BF16, col);
    let vt = view(&mut b, arena, 24576, DType::BF16, col);
    let dp_shared = view(&mut b, arena, 32768, DType::F32, row);
    let score = view(&mut b, arena, 0, DType::F32, row); // Q+dO retire before this alias.
    let ds = view(
        &mut b,
        arena,
        32768,
        DType::BF16,
        SharedLayout::row_major(128),
    );
    let qks = schedule(row, col);
    let dps = schedule(row, col);
    let dqs = schedule(SharedLayout::row_major(128), row);
    let qk = fragment(&mut b, qks);
    let dp = fragment(&mut b, dps);
    let dq = fragment(&mut b, dqs);
    b.stmts.push(Stmt::Zero { tile: dq });
    let query_base = Expr::BlockIdx(Dim::X) * TILE;
    delta(&mut b, shape, incoming_layout, query_base.clone())?;
    let keys = b.loop_name("causal_keys");
    b.fold(keys, Expr::BlockIdx(Dim::X) + ix(1), vec![], |b, tile| {
        let key_base = tile * TILE;
        stage_with_transport(
            b,
            shape,
            qkv_layout,
            Stage {
                prefix: "stage_query",
                source: "query",
                row_base: query_base.clone(),
                dest: q,
                column_major: false,
            },
            transport,
        )?;
        stage_with_transport(
            b,
            shape,
            incoming_layout,
            Stage {
                prefix: "stage_incoming",
                source: "incoming",
                row_base: query_base.clone(),
                dest: incoming,
                column_major: false,
            },
            transport,
        )?;
        stage_with_transport(
            b,
            shape,
            qkv_layout,
            Stage {
                prefix: "stage_key",
                source: "key",
                row_base: key_base.clone(),
                dest: k,
                column_major: false,
            },
            transport,
        )?;
        stage_with_transport(
            b,
            shape,
            qkv_layout,
            Stage {
                prefix: "stage_value",
                source: "value",
                row_base: key_base.clone(),
                dest: vt,
                column_major: true,
            },
            transport,
        )?;
        finish_staging(b, transport);
        b.barrier();
        b.stmts.push(Stmt::Zero { tile: qk });
        b.stmts.push(Stmt::Zero { tile: dp });
        mma(b, qk, q, kt, qks);
        mma(b, dp, incoming, vt, dps);
        convert(b, dp_shared, dp);
        b.barrier(); // Retire all Q/dO readers before score overwrites them.
        convert(b, score, qk);
        b.barrier();
        weights(b, score, dp_shared, ds, None, query_base.clone(), key_base)?;
        mma(b, dq, ds, k, dqs);
        b.barrier(); // Retire K/DS before all four inputs are reloaded.
        Ok(())
    })?;
    // All input readers retired; Q+dO is the final16KiB F32 exchange.
    convert(&mut b, score, dq);
    b.barrier();
    distribute(&mut b, "store_dquery", |b, index| {
        let value = b.load(sm(score), index.clone());
        b.store(
            global("d_query"),
            dense_index(
                query_base.clone() + div(index.clone(), TILE),
                rem(index, TILE),
            ),
            value,
        );
        Ok(())
    })?;
    let result = TileIR {
        kernel_name: transport.kernel_name(format!(
            "online_training_dquery_h{}_q64_{qkv_layout:?}_{incoming_layout:?}",
            shape.heads
        )),
        params: vec![
            param("query", DType::BF16, true),
            param("key", DType::BF16, true),
            param("value", DType::BF16, true),
            param("output_0", DType::BF16, true),
            param("incoming", DType::BF16, true),
            param("statistics", DType::F32, false),
            param("d_query", DType::BF16, false),
        ],
        body: Block { stmts: b.stmts },
        shared_mem_bytes: SHARED_BYTES,
    };
    result.validate_layouts()?;
    Ok(result)
}

/// Q,K,V,dO16,statistics,dKT16,dV16. One CTA owns64 keys, reduces every
/// causally eligible query block in ascending K16 order, and writes once.
pub(super) fn d_key_value_ir(
    shape: Shape,
    qkv_layout: ReadLayout,
    incoming_layout: ReadLayout,
) -> Result<TileIR> {
    d_key_value_ir_with_transport(shape, qkv_layout, incoming_layout, Transport::Scalar)
}

pub(super) fn d_key_value_ir_with_transport(
    shape: Shape,
    qkv_layout: ReadLayout,
    incoming_layout: ReadLayout,
    transport: Transport,
) -> Result<TileIR> {
    Shape::new(shape.batch, shape.heads)?;
    let mut b = ScalarBuilder::default();
    let arena = arena(&mut b);
    let row = SharedLayout::row_major(TILE);
    let col = column(TILE);
    let q = view(&mut b, arena, 0, DType::BF16, row);
    let qt = view(&mut b, arena, 0, DType::BF16, col);
    let incoming = view(&mut b, arena, 8192, DType::BF16, row);
    let kt = view(&mut b, arena, 16384, DType::BF16, col);
    let vt = view(&mut b, arena, 24576, DType::BF16, col);
    let dp_shared = view(&mut b, arena, 32768, DType::F32, row);
    let score = view(&mut b, arena, 16384, DType::F32, row); // Dead K+V.
    let p = view(
        &mut b,
        arena,
        16384,
        DType::BF16,
        SharedLayout::row_major(128),
    );
    let pt = view(&mut b, arena, 16384, DType::BF16, column(128));
    let ds = view(
        &mut b,
        arena,
        32768,
        DType::BF16,
        SharedLayout::row_major(128),
    );
    let exchange = view(&mut b, arena, 0, DType::F32, row); // Dead Q+dO at final store.
    let qks = schedule(row, col);
    let dps = schedule(row, col);
    let dks = schedule(col, SharedLayout::row_major(128));
    let dvs = schedule(column(128), row);
    let qk = fragment(&mut b, qks);
    let dp = fragment(&mut b, dps);
    let dk = fragment(&mut b, dks);
    let dv = fragment(&mut b, dvs);
    b.stmts.push(Stmt::Zero { tile: dk });
    b.stmts.push(Stmt::Zero { tile: dv });
    let key_base = Expr::BlockIdx(Dim::X) * TILE;
    let queries = b.loop_name("causal_queries");
    b.fold(
        queries,
        Expr::Sub(
            Box::new(ix(SEQUENCE / TILE)),
            Box::new(Expr::BlockIdx(Dim::X)),
        ),
        vec![],
        |b, tile| {
            let query_base = (tile + Expr::BlockIdx(Dim::X)) * TILE;
            stage_with_transport(
                b,
                shape,
                qkv_layout,
                Stage {
                    prefix: "stage_query",
                    source: "query",
                    row_base: query_base.clone(),
                    dest: q,
                    column_major: false,
                },
                transport,
            )?;
            stage_with_transport(
                b,
                shape,
                incoming_layout,
                Stage {
                    prefix: "stage_incoming",
                    source: "incoming",
                    row_base: query_base.clone(),
                    dest: incoming,
                    column_major: false,
                },
                transport,
            )?;
            stage_with_transport(
                b,
                shape,
                qkv_layout,
                Stage {
                    prefix: "stage_key",
                    source: "key",
                    row_base: key_base.clone(),
                    dest: kt,
                    column_major: true,
                },
                transport,
            )?;
            stage_with_transport(
                b,
                shape,
                qkv_layout,
                Stage {
                    prefix: "stage_value",
                    source: "value",
                    row_base: key_base.clone(),
                    dest: vt,
                    column_major: true,
                },
                transport,
            )?;
            finish_staging(b, transport);
            b.barrier();
            b.stmts.push(Stmt::Zero { tile: qk });
            b.stmts.push(Stmt::Zero { tile: dp });
            mma(b, qk, q, kt, qks);
            mma(b, dp, incoming, vt, dps);
            convert(b, dp_shared, dp);
            b.barrier(); // Retire every K/V read before alias.
            convert(b, score, qk);
            b.barrier();
            weights(
                b,
                score,
                dp_shared,
                ds,
                Some(p),
                query_base,
                key_base.clone(),
            )?;
            mma(b, dk, qt, ds, dks);
            mma(b, dv, pt, incoming, dvs);
            b.barrier(); // Retire Q/dO and P/DS before the next query block.
            Ok(())
        },
    )?;
    convert(&mut b, exchange, dk);
    b.barrier();
    distribute(&mut b, "store_dkey_transposed", |b, index| {
        let feature = div(index.clone(), TILE);
        let key = key_base.clone() + rem(index.clone(), TILE);
        let value = b.load(sm(exchange), index);
        b.store(
            global("d_key_transposed"),
            (Expr::BlockIdx(Dim::Y) * TILE + feature) * SEQUENCE + key,
            value,
        );
        Ok(())
    })?;
    b.barrier(); // Retire all dKT exchange readers before dV overwrites it.
    convert(&mut b, exchange, dv);
    b.barrier();
    distribute(&mut b, "store_dvalue", |b, index| {
        let value = b.load(sm(exchange), index.clone());
        b.store(
            global("d_value"),
            dense_index(
                key_base.clone() + div(index.clone(), TILE),
                rem(index, TILE),
            ),
            value,
        );
        Ok(())
    })?;
    let result = TileIR {
        kernel_name: transport.kernel_name(format!(
            "online_training_dkeyvalue_h{}_k64_{qkv_layout:?}_{incoming_layout:?}",
            shape.heads
        )),
        params: vec![
            param("query", DType::BF16, true),
            param("key", DType::BF16, true),
            param("value", DType::BF16, true),
            param("incoming", DType::BF16, true),
            param("statistics", DType::F32, true),
            param("d_key_transposed", DType::BF16, false),
            param("d_value", DType::BF16, false),
        ],
        body: Block { stmts: b.stmts },
        shared_mem_bytes: SHARED_BYTES,
    };
    result.validate_layouts()?;
    Ok(result)
}

#[cfg(test)]
mod tests;

pub(super) mod packed;
