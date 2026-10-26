//! BF16 tiled causal attention, expressed entirely with ordinary Tile IR.
//! A CTA owns sixteen or thirty-two queries and streams thirty-two keys at a time.
//! Four/eight key partitions expose parallelism, then merge raw f32 numerators. Scores,
//! online softmax statistics and the running numerator stay f32; MMA operands
//! (including unnormalized probabilities) and the final output are BF16.
use super::attention::AttentionScan;
use crate::tensor::{BinaryOp, UnaryOp};
use crate::tile::*;
use anyhow::{Result, ensure};

const WARPS: usize = 8;
pub(super) const STATISTICS_WIDTH: usize = 2;

pub(super) fn partitions(chunk: usize) -> usize {
    if chunk >= 256 { 8 } else { 4 }
}

fn queries(scan: &AttentionScan) -> usize {
    if scan.chunk >= 1024 && scan.width / scan.heads <= 64 {
        32
    } else {
        16
    }
}

fn ix(n: usize) -> Expr {
    Expr::Const(n as i64)
}
fn div(x: Expr, n: usize) -> Expr {
    Expr::FloorDiv(Box::new(x), n)
}
fn rem(x: Expr, n: usize) -> Expr {
    Expr::Mod(Box::new(x), n)
}
fn op(kind: BinaryOp, a: impl Into<ScalarExpr>, b: impl Into<ScalarExpr>) -> ScalarExpr {
    ScalarExpr::binary(ScalarBinaryOp::Arithmetic(kind), a, b)
}
fn expdiff(a: impl Into<ScalarExpr>, b: impl Into<ScalarExpr>) -> ScalarExpr {
    ScalarExpr::unary(UnaryOp::Exp, op(BinaryOp::Sub, a, b))
}
fn lt(a: Expr, b: Expr) -> ScalarPredicate {
    ScalarPredicate::IndexLt { lhs: a, rhs: b }
}
fn sm(v: TileVar) -> ScalarMemory {
    ScalarMemory::Shared(v)
}
fn constant(x: f32) -> ScalarExpr {
    ScalarExpr::Constant(x)
}
fn shared(b: &mut ScalarBuilder, rows: usize, cols: usize, dtype: DType) -> TileVar {
    shared_layout(b, rows, cols, dtype, SharedLayout::row_major(cols))
}
fn shared_layout(
    b: &mut ScalarBuilder,
    rows: usize,
    cols: usize,
    dtype: DType,
    physical: SharedLayout,
) -> TileVar {
    let var = b.var();
    b.stmts.push(Stmt::AllocTile {
        var,
        space: MemorySpace::Shared,
        layout: TileLayout::Shared(physical),
        dtype,
        rows,
        cols,
    });
    var
}
fn schedule(m: usize, n: usize, k: usize, warps: usize) -> MatMulSchedule {
    let tile = MatMulTile { m, n, k };
    MatMulSchedule {
        plan: MatMulPlan::TensorCoreBF16,
        logical_shape: tile,
        block_tile: tile,
        instruction_tile: MatMulTile {
            m: 16,
            n: 16,
            k: 16,
        },
        block_threads: (32, warps as u32, 1),
        warp_topology: (m / 16, n / 16),
        warp_fragments: (1, 1),
        operand_layouts: (SharedLayout::row_major(k), SharedLayout::row_major(n)),
        pipeline_stages: 1,
        copy_bytes: 0,
        async_operands: (false, false),
        operand_dtype: DType::BF16,
        storage_dtype: DType::BF16,
        accumulator_layout: MatMulAccumulatorLayout::WarpFragment,
    }
}
fn fragment(b: &mut ScalarBuilder, schedule: MatMulSchedule) -> TileVar {
    let var = b.var();
    b.stmts.push(Stmt::AllocTile {
        var,
        space: MemorySpace::Fragment,
        layout: schedule.accumulator_tile_layout(),
        dtype: DType::F32,
        rows: schedule.block_tile.m,
        cols: schedule.block_tile.n,
    });
    var
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
fn to_shared(b: &mut ScalarBuilder, dest: TileVar, src: TileVar) {
    b.stmts.push(Stmt::ConvertLayout {
        dest,
        src,
        coordinates: None,
    });
    b.barrier();
}
// Every thread participates in barriers; predicates only surround scalar memory.
fn distribute(
    b: &mut ScalarBuilder,
    count: usize,
    threads: usize,
    f: impl FnOnce(&mut ScalarBuilder, Expr) -> Result<()>,
) -> Result<()> {
    let name = b.loop_name("stage");
    b.fold(name, ix(count.div_ceil(threads)), vec![], |b, i| {
        let offset = i * threads + Expr::ThreadIdx(Dim::Y) * 32usize + Expr::ThreadIdx(Dim::X);
        b.when(lt(offset.clone(), ix(count)), |b| f(b, offset))
    })
}

// Four contiguous rounds per loop amortize synchronous operand staging.
fn stage_operand(
    b: &mut ScalarBuilder,
    count: usize,
    threads: usize,
    mut f: impl FnMut(&mut ScalarBuilder, Expr) -> Result<()>,
) -> Result<()> {
    let name = b.loop_name("operand");
    b.fold(name, ix(count.div_ceil(threads * 4)), vec![], |b, i| {
        for round in 0..4 {
            let offset = i.clone() * (threads * 4)
                + Expr::ThreadIdx(Dim::Y) * 32usize
                + Expr::ThreadIdx(Dim::X)
                + ix(round * threads);
            b.when(lt(offset.clone(), ix(count)), |b| f(b, offset))?;
        }
        Ok(())
    })
}

pub(super) fn launch(scan: &AttentionScan) -> Result<cudarc::driver::LaunchConfig> {
    Ok(cudarc::driver::LaunchConfig {
        grid_dim: (
            u32::try_from(scan.chunk.div_ceil(queries(scan)))?,
            u32::try_from(scan.batch * scan.heads)?,
            u32::try_from(scan.partitions)?,
        ),
        block_dim: (32, u32::try_from(WARPS)?, 1),
        shared_mem_bytes: 0,
    })
}

pub(super) fn ir(scan: &AttentionScan) -> Result<TileIR> {
    let columns = scan.width / scan.heads;
    ensure!(
        scan.dtype == DType::BF16 && columns > 0 && columns <= 128 && scan.partitions > 0,
        "unsupported BF16 tiled prefill shape"
    );
    let padded = columns.max(16).next_power_of_two();
    let queries = queries(scan);
    let keys: usize = 32;
    let threads = WARPS * 32;
    let mut b = ScalarBuilder::default();
    b.stmts.push(Stmt::LoadIndex {
        name: "prefix".into(),
        source: "length".into(),
        index: ix(0),
    });
    let prefix = Expr::Var("prefix".into());
    let batch = div(Expr::BlockIdx(Dim::Y), scan.heads);
    let head = rem(Expr::BlockIdx(Dim::Y), scan.heads);
    let query_base = Expr::BlockIdx(Dim::X) * queries;
    let tid = Expr::ThreadIdx(Dim::Y) * 32usize + Expr::ThreadIdx(Dim::X);
    let q = shared(&mut b, queries, padded, DType::BF16);
    let key_layout = SharedLayout {
        row_stride: padded,
        column_major: true,
        xor_mask: 0,
    };
    let k = shared_layout(&mut b, padded, keys, DType::BF16, key_layout);
    let scores = shared(&mut b, queries, keys, DType::F32);
    let p = shared(&mut b, queries, keys, DType::BF16);
    let v = b.var();
    b.stmts.push(Stmt::AliasShared {
        dest: v,
        src: k,
        layout: SharedLayout::row_major(padded),
        rows: keys,
        cols: padded,
    });
    // Score staging is dead after softmax. The persistent accumulator can
    // reuse it for rescaling/finalization when the score allocation is large enough.
    let numerator = if keys >= padded {
        let var = b.var();
        b.stmts.push(Stmt::AliasShared {
            dest: var,
            src: scores,
            layout: SharedLayout::row_major(padded),
            rows: queries,
            cols: padded,
        });
        var
    } else {
        shared(&mut b, queries, padded, DType::F32)
    };
    let maximum = shared(&mut b, 1, queries, DType::F32);
    let denominator = shared(&mut b, 1, queries, DType::F32);
    let rescale = shared(&mut b, 1, queries, DType::F32);
    let changed = shared(&mut b, 1, 1, DType::F32);
    let mut qk_schedule = schedule(queries, keys, padded, WARPS);
    qk_schedule.operand_layouts.1 = key_layout;
    let pv_schedule = schedule(queries, padded, keys, WARPS);
    let qk = fragment(&mut b, qk_schedule);
    let pv = fragment(&mut b, pv_schedule);
    let body_start = b.stmts.len();
    b.when(lt(tid.clone(), ix(queries)), |b| {
        let neg = b.value(constant(f32::NEG_INFINITY));
        let zero = b.value(constant(0.));
        b.store(sm(maximum), tid.clone(), neg);
        b.store(sm(denominator), tid.clone(), zero);
        Ok(())
    })?;
    distribute(&mut b, queries * padded, threads, |b, i| {
        let query = query_base.clone() + div(i.clone(), padded);
        let feature = rem(i.clone(), padded);
        let value = b.value(constant(0.));
        b.when(lt(query.clone(), ix(scan.chunk)), |b| {
            b.when(lt(feature.clone(), ix(columns)), |b| {
                let offset = batch.clone() * (scan.chunk * scan.width)
                    + query * scan.width
                    + head.clone() * columns
                    + feature;
                let x = b.load(ScalarMemory::Global("query".into()), offset);
                b.set(value, x);
                Ok(())
            })
        })?;
        b.store(sm(q), i, value);
        Ok(())
    })?;
    b.barrier();
    b.stmts.push(Stmt::Zero { tile: pv });
    let name = b.loop_name("keys");
    let causal_tiles = div(
        prefix.clone() + query_base.clone() + ix(queries + keys - 1),
        keys,
    );
    let tiles = if scan.partitions > 1 {
        div(
            prefix.clone() + ix(scan.chunk + keys * scan.partitions - 1),
            keys * scan.partitions,
        )
    } else {
        causal_tiles.clone()
    };
    let part = Expr::BlockIdx(Dim::Z);
    b.fold(name, tiles.clone(), vec![], |b, tile| {
        let local_tile = tile.clone();
        let tile = if scan.partitions > 1 {
            tile + Expr::Mul(Box::new(part.clone()), Box::new(tiles.clone()))
        } else {
            tile
        };
        b.when(lt(tile.clone(), causal_tiles.clone()), |b| {
            let key_base = tile * keys;
            b.stmts.push(Stmt::Zero { tile: qk });
            stage_operand(b, keys * padded, threads, |b, i| {
                let key_column = div(i.clone(), padded);
                let feature = rem(i, padded);
                let value = b.value(constant(0.));
                b.when(lt(feature.clone(), ix(columns)), |b| {
                    let x = scan.load_kv(
                        b,
                        "key",
                        batch.clone(),
                        head.clone(),
                        key_base.clone() + key_column.clone(),
                        feature.clone(),
                    )?;
                    b.set(value, x);
                    Ok(())
                })?;
                b.store(sm(k), key_column * padded + feature, value);
                Ok(())
            })?;
            b.barrier();
            mma(b, qk, q, k, qk_schedule);
            to_shared(b, scores, qk);
            let rows = b.loop_name("softmax_rows");
            b.fold(rows, ix(queries / WARPS), vec![], |b, i| {
                let row = i * WARPS + Expr::ThreadIdx(Dim::Y);
                let lane = Expr::ThreadIdx(Dim::X);
                let query = query_base.clone() + row.clone();
                let mut scores_for_lane = Vec::new();
                let local_max = b.value(constant(f32::NEG_INFINITY));
                for group in 0..keys / 32 {
                    let key_column = lane.clone() + ix(group * 32);
                    let score = b.value(constant(f32::NEG_INFINITY));
                    b.when(
                        lt(
                            key_base.clone() + key_column.clone(),
                            prefix.clone() + query.clone() + ix(1),
                        ),
                        |b| {
                            let x = b.load(sm(scores), row.clone() * keys + key_column);
                            b.set(
                                score,
                                op(BinaryOp::Mul, x, constant((columns as f32).sqrt().recip())),
                            );
                            Ok(())
                        },
                    )?;
                    b.set(
                        local_max,
                        ScalarExpr::binary(ScalarBinaryOp::Max, local_max, score),
                    );
                    scores_for_lane.push(score);
                }
                let tile_max = b.warp_all_reduce(local_max, crate::tensor::ReduceOp::Max);
                let old = b.load(sm(maximum), row.clone());
                let next = b.value(ScalarExpr::binary(ScalarBinaryOp::Max, old, tile_max));
                let empty = ScalarPredicate::Equal {
                    lhs: next.into(),
                    rhs: constant(f32::NEG_INFINITY),
                };
                let alpha = b.value(ScalarExpr::select(
                    empty.clone(),
                    constant(1.),
                    expdiff(old, next),
                ));
                let local_sum = b.value(constant(0.));
                for (group, score) in scores_for_lane.into_iter().enumerate() {
                    let weight = b.value(ScalarExpr::select(
                        empty.clone(),
                        constant(0.),
                        expdiff(score, next),
                    ));
                    b.set(local_sum, op(BinaryOp::Add, local_sum, weight));
                    b.store(
                        sm(p),
                        row.clone() * keys + lane.clone() + ix(group * 32),
                        weight,
                    );
                }
                let sum = b.warp_all_reduce(local_sum, crate::tensor::ReduceOp::Sum);
                b.when(lt(lane, ix(1)), |b| {
                    let old_den = b.load(sm(denominator), row.clone());
                    let den = b.value(op(BinaryOp::Add, op(BinaryOp::Mul, old_den, alpha), sum));
                    b.store(sm(maximum), row.clone(), next);
                    b.store(sm(denominator), row.clone(), den);
                    b.store(sm(rescale), row, alpha);
                    Ok(())
                })?;
                Ok(())
            })?;
            stage_operand(b, keys * padded, threads, |b, i| {
                let key = key_base.clone() + div(i.clone(), padded);
                let col = rem(i.clone(), padded);
                let value = b.value(constant(0.));
                b.when(lt(col.clone(), ix(columns)), |b| {
                    let x = scan.load_kv(b, "value", batch.clone(), head.clone(), key, col)?;
                    b.set(value, x);
                    Ok(())
                })?;
                b.store(sm(v), i, value);
                Ok(())
            })?;
            b.barrier();
            b.when(lt(ix(0), local_tile), |b| {
                b.when(lt(Expr::ThreadIdx(Dim::Y), ix(1)), |b| {
                    let delta = b.value(constant(0.));
                    for group in 0..queries.div_ceil(32) {
                        let row = Expr::ThreadIdx(Dim::X) + ix(group * 32);
                        b.when(lt(row.clone(), ix(queries)), |b| {
                            let alpha = b.load(sm(rescale), row);
                            b.set(
                                delta,
                                ScalarExpr::binary(
                                    ScalarBinaryOp::Max,
                                    delta,
                                    op(BinaryOp::Sub, constant(1.), alpha),
                                ),
                            );
                            Ok(())
                        })?;
                    }
                    let delta = b.warp_all_reduce(delta, crate::tensor::ReduceOp::Max);
                    b.when(lt(Expr::ThreadIdx(Dim::X), ix(1)), |b| {
                        b.store(sm(changed), ix(0), delta);
                        Ok(())
                    })
                })?;
                b.barrier();
                let flag = b.load(sm(changed), ix(0));
                let flag = b.value(ScalarExpr::select(
                    ScalarPredicate::Equal {
                        lhs: flag.into(),
                        rhs: constant(0.),
                    },
                    constant(0.),
                    constant(1.),
                ));
                b.when(
                    ScalarPredicate::Equal {
                        lhs: flag.into(),
                        rhs: constant(1.),
                    },
                    |b| {
                        to_shared(b, numerator, pv);
                        distribute(b, queries * padded, threads, |b, i| {
                            let alpha = b.load(sm(rescale), div(i.clone(), padded));
                            let old = b.load(sm(numerator), i.clone());
                            let value = b.value(op(BinaryOp::Mul, old, alpha));
                            b.store(sm(numerator), i, value);
                            Ok(())
                        })?;
                        b.barrier();
                        b.stmts.push(Stmt::ConvertLayout {
                            dest: pv,
                            src: numerator,
                            coordinates: None,
                        });
                        Ok(())
                    },
                )?;
                Ok(())
            })?;
            mma(b, pv, p, v, pv_schedule);
            // The next key stage reuses the value operand's shared allocation.
            b.barrier();
            Ok(())
        })
    })?;
    to_shared(&mut b, numerator, pv);
    distribute(&mut b, queries * padded, threads, |b, i| {
        let row = div(i.clone(), padded);
        let col = rem(i.clone(), padded);
        let query = query_base.clone() + row.clone();
        b.when(lt(query.clone(), ix(scan.chunk)), |b| {
            b.when(lt(col.clone(), ix(columns)), |b| {
                let num = b.load(sm(numerator), i.clone());
                let result = if scan.partitions > 1 {
                    num
                } else {
                    let den = b.load(sm(denominator), row.clone());
                    let empty = ScalarPredicate::Equal {
                        lhs: den.into(),
                        rhs: constant(0.),
                    };
                    b.value(ScalarExpr::select(
                        empty,
                        constant(0.),
                        op(BinaryOp::Div, num, den),
                    ))
                };
                let output_row = (batch.clone() * scan.heads + head.clone()) * scan.chunk + query;
                let offset =
                    (output_row.clone() * scan.partitions + part.clone()) * columns + col.clone();
                if scan.partitions > 1 {
                    b.when(lt(col, ix(1)), |b| {
                        let max = b.load(sm(maximum), row.clone());
                        let den = b.load(sm(denominator), row);
                        let index =
                            (output_row * scan.partitions + part.clone()) * STATISTICS_WIDTH;
                        b.store(
                            ScalarMemory::Global("statistics".into()),
                            index.clone(),
                            max,
                        );
                        b.store(
                            ScalarMemory::Global("statistics".into()),
                            index + ix(1),
                            den,
                        );
                        Ok(())
                    })?;
                }
                b.store(ScalarMemory::Global("output_0".into()), offset, result);
                Ok(())
            })
        })
    })?;
    if scan.partitions > 1 {
        let span = div(
            prefix.clone() + ix(scan.chunk + keys * scan.partitions - 1),
            keys * scan.partitions,
        ) * keys;
        let start = Expr::Mul(Box::new(part.clone()), Box::new(span));
        let live = lt(start, prefix.clone() + query_base.clone() + ix(queries));
        let live_body = Block {
            stmts: b.stmts.split_off(body_start),
        };
        b.stmts.push(Stmt::If {
            condition: live.clone(),
            body: live_body,
        });
        let empty = b.value(ScalarExpr::select(live, constant(0.), constant(1.)));
        b.when(
            ScalarPredicate::Equal {
                lhs: empty.into(),
                rhs: constant(1.),
            },
            |b| {
                let zero = b.value(constant(0.));
                let neg = b.value(constant(f32::NEG_INFINITY));
                distribute(b, queries * padded, threads, |b, i| {
                    let query = query_base.clone() + div(i.clone(), padded);
                    let col = rem(i, padded);
                    b.when(lt(query.clone(), ix(scan.chunk)), |b| {
                        b.when(lt(col.clone(), ix(columns)), |b| {
                            let row =
                                (batch.clone() * scan.heads + head.clone()) * scan.chunk + query;
                            b.store(
                                ScalarMemory::Global("output_0".into()),
                                (row.clone() * scan.partitions + part.clone()) * columns
                                    + col.clone(),
                                zero,
                            );
                            b.when(lt(col, ix(1)), |b| {
                                b.store(
                                    ScalarMemory::Global("statistics".into()),
                                    (row.clone() * scan.partitions + part.clone())
                                        * STATISTICS_WIDTH,
                                    neg,
                                );
                                b.store(
                                    ScalarMemory::Global("statistics".into()),
                                    (row * scan.partitions + part.clone()) * STATISTICS_WIDTH
                                        + ix(1),
                                    zero,
                                );
                                Ok(())
                            })
                        })
                    })
                })
            },
        )?;
    }
    let mut params: Vec<_> = [
        "query",
        "key",
        "value",
        "cached_key",
        "cached_value",
        "length",
        "output_0",
    ]
    .into_iter()
    .map(|name| KernelParam {
        name: name.into(),
        dtype: if name == "length" {
            DType::U32
        } else if name == "output_0" && scan.partitions > 1 {
            DType::F32
        } else {
            DType::BF16
        },
        is_input: name != "output_0",
    })
    .collect();
    if scan.partitions > 1 {
        params.push(KernelParam {
            name: "statistics".into(),
            dtype: DType::F32,
            is_input: false,
        });
    }
    let shared_mem_bytes = b
        .stmts
        .iter()
        .map(|stmt| match stmt {
            Stmt::AllocTile {
                space: MemorySpace::Shared,
                layout: TileLayout::Shared(layout),
                rows,
                cols,
                dtype,
                ..
            } => layout.allocation_bytes(*rows, *cols, *dtype),
            _ => 0,
        })
        .sum();
    Ok(TileIR {
        kernel_name: "decoder_prefill_bf16".into(),
        params,
        body: Block { stmts: b.stmts },
        shared_mem_bytes,
    })
}
