//! Build normalized reductions from explicit scalar arithmetic and loop state.
use super::indexed_sum::SumAccumulator;
use super::region::input_element_index;
use super::*;
use crate::tensor::{BinaryOp, ReduceOp, UnaryOp};
use anyhow::{Context, Result};

fn constant(n: usize) -> Result<Expr> {
    Ok(Expr::Const(
        i64::try_from(n).context("schedule extent exceeds i64")?,
    ))
}
fn arithmetic(op: BinaryOp, a: impl Into<ScalarExpr>, b: impl Into<ScalarExpr>) -> ScalarExpr {
    ScalarExpr::binary(ScalarBinaryOp::Arithmetic(op), a, b)
}
fn exp_difference(a: impl Into<ScalarExpr>, b: impl Into<ScalarExpr>) -> ScalarExpr {
    ScalarExpr::unary(UnaryOp::Exp, arithmetic(BinaryOp::Sub, a, b))
}
fn equal(a: impl Into<ScalarExpr>, b: f32) -> ScalarPredicate {
    ScalarPredicate::Equal {
        lhs: a.into(),
        rhs: ScalarExpr::Constant(b),
    }
}
fn less(a: Expr, n: usize) -> Result<ScalarPredicate> {
    Ok(ScalarPredicate::IndexLt {
        lhs: a,
        rhs: constant(n)?,
    })
}
fn load(
    builder: &mut ScalarBuilder,
    region: &OnlineRegion,
    input: usize,
    offset: Expr,
) -> Result<TileVar> {
    let index = input_element_index(
        &region.inputs[input],
        &region.inputs[input].tensor.shape,
        &offset,
    )?;
    Ok(builder.load(ScalarMemory::Global(format!("input_{input}")), index))
}
fn coordinates(shape: &[usize], offset: &Expr) -> Vec<Expr> {
    let mut point = Vec::new();
    let mut stride = 1;
    for &dim in shape.iter().rev() {
        point.push(Expr::Mod(
            Box::new(Expr::FloorDiv(Box::new(offset.clone()), stride)),
            dim,
        ));
        stride *= dim;
    }
    point.reverse();
    point
}
fn score(
    builder: &mut ScalarBuilder,
    region: &OnlineRegion,
    expression: &OnlineExpr,
    offset: Expr,
) -> Result<TileVar> {
    Ok(match expression {
        OnlineExpr::Input(input) => load(builder, region, *input, offset)?,
        OnlineExpr::Unary(op, value) => {
            let value = score(builder, region, value, offset)?;
            builder.value(ScalarExpr::unary(*op, value))
        }
        OnlineExpr::Binary(op, a, b) => {
            let a = score(builder, region, a, offset.clone())?;
            let b = score(builder, region, b, offset)?;
            builder.value(arithmetic(*op, a, b))
        }
        OnlineExpr::Sum(sum) => {
            let coordinates = coordinates(&region.score_shape, &offset);
            sum.lower(builder, &region.inputs, &coordinates)?
        }
    })
}

/// Merge a new reference into the normalizer, and return the scale that any
/// homogeneous sum consumer must apply to its own carried state.
fn merge(
    builder: &mut ScalarBuilder,
    maximum: TileVar,
    denominator: TileVar,
    reference: TileVar,
) -> TileVar {
    let next = builder.value(ScalarExpr::binary(ScalarBinaryOp::Max, maximum, reference));
    let alpha = builder.value(ScalarExpr::select(
        equal(next, f32::NEG_INFINITY),
        ScalarExpr::Constant(1.0),
        exp_difference(maximum, next),
    ));
    builder.set(maximum, next);
    builder.set(denominator, arithmetic(BinaryOp::Mul, denominator, alpha));
    alpha
}

pub(super) fn lower(region: &OnlineRegion, extent: usize) -> Result<(Block, usize)> {
    let rank = region.score_shape.len();
    let keys = region.score_shape[rank - 1];
    let columns = region.output_shape[rank - 1];
    let cooperative = region.block_threads().is_some();
    let lane = Expr::ThreadIdx(Dim::X);
    let tid = Expr::Mul(
        Box::new(Expr::BlockIdx(Dim::X)),
        Box::new(Expr::BlockDim(Dim::X)),
    ) + lane.clone();
    let (fiber, column, output) = if cooperative {
        (
            Expr::BlockIdx(Dim::X),
            lane.clone(),
            Expr::BlockIdx(Dim::X) * columns + lane.clone(),
        )
    } else {
        (
            Expr::FloorDiv(Box::new(tid.clone()), columns),
            Expr::Mod(Box::new(tid.clone()), columns),
            tid,
        )
    };
    let score_base = fiber.clone() * keys;
    let mut builder = ScalarBuilder::default();
    if !cooperative {
        builder.stmts.push(Stmt::BoundsCheck { extent });
    }
    let shared = if cooperative {
        let var = builder.var();
        builder.stmts.push(Stmt::AllocTile {
            var,
            space: MemorySpace::Shared,
            layout: TileLayout::Shared(SharedLayout::row_major(33)),
            dtype: DType::F32,
            rows: 1,
            cols: 33,
        });
        Some(var)
    } else {
        None
    };
    let maximum = builder.var();
    let denominator = builder.var();
    let numerator = SumAccumulator::new(&mut builder);
    let tile_size = if cooperative { 32 } else { 1 };
    let name = builder.loop_name("scan");
    builder.fold(
        name,
        constant(keys.div_ceil(tile_size))?,
        vec![
            LoopCarry {
                var: maximum,
                initial: ScalarExpr::Constant(f32::NEG_INFINITY),
            },
            LoopCarry {
                var: denominator,
                initial: ScalarExpr::Constant(0.0),
            },
            numerator.carry(),
        ],
        |builder, tile| {
            let base = tile * tile_size;
            if let Some(shared) = shared {
                builder.when(less(lane.clone(), 32)?, |builder| {
                    let key = base.clone() + lane.clone();
                    let value = builder.value(ScalarExpr::Constant(f32::NEG_INFINITY));
                    builder.when(less(key.clone(), keys)?, |builder| {
                        let computed =
                            score(builder, region, &region.score, score_base.clone() + key)?;
                        builder.set(value, computed);
                        Ok(())
                    })?;
                    builder.store(ScalarMemory::Shared(shared), lane.clone(), value);
                    let reference = builder.warp_reduce(value, ReduceOp::Max);
                    builder.when(less(lane.clone(), 1)?, |builder| {
                        builder.store(ScalarMemory::Shared(shared), Expr::Const(32), reference);
                        Ok(())
                    })?;
                    Ok(())
                })?;
                builder.barrier();
                let reference = builder.load(ScalarMemory::Shared(shared), Expr::Const(32));
                let alpha = merge(builder, maximum, denominator, reference);
                numerator.rescale(builder, alpha);
            }
            let name = builder.loop_name("consume");
            builder.fold(name, constant(tile_size)?, Vec::new(), |builder, local| {
                let key = base.clone() + local.clone();
                builder.when(less(key.clone(), keys)?, |builder| {
                    let value = if let Some(shared) = shared {
                        builder.load(ScalarMemory::Shared(shared), local)
                    } else {
                        score(
                            builder,
                            region,
                            &region.score,
                            score_base.clone() + key.clone(),
                        )?
                    };
                    if !cooperative {
                        let alpha = merge(builder, maximum, denominator, value);
                        numerator.rescale(builder, alpha);
                    }
                    let probability = builder.value(ScalarExpr::select(
                        equal(value, f32::NEG_INFINITY),
                        ScalarExpr::Constant(0.0),
                        exp_difference(value, maximum),
                    ));
                    builder.set(
                        denominator,
                        arithmetic(BinaryOp::Add, denominator, probability),
                    );
                    if let OnlineConsumer::Sum(sum) = &region.consumer {
                        builder.when(less(column.clone(), columns)?, |builder| {
                            let mut point = coordinates(&region.output_shape, &output);
                            point.push(key);
                            sum.accumulate(
                                builder,
                                &region.inputs,
                                &numerator,
                                &point,
                                Some(probability),
                            )
                        })?;
                    }
                    Ok(())
                })?;
                Ok(())
            })?;
            if cooperative {
                builder.barrier();
            }
            Ok(())
        },
    )?;
    builder.when(less(column, columns)?, |builder| {
        let result = builder.value(match region.consumer {
            OnlineConsumer::Normalizer => ScalarExpr::select(
                equal(maximum, f32::NEG_INFINITY),
                ScalarExpr::Constant(f32::NAN),
                denominator,
            ),
            OnlineConsumer::Sum(_) => arithmetic(BinaryOp::Div, numerator.0, denominator),
        });
        builder.store(ScalarMemory::Global("output_0".into()), output, result);
        Ok(())
    })?;
    Ok((
        Block {
            stmts: builder.stmts,
        },
        if cooperative { 132 } else { 0 },
    ))
}
