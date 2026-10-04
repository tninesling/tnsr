//! Lower coupled normalization state using ordinary scalar PTX operations.
use super::*;
use crate::tile::{
    OnlineExpr, OnlineRegion, ReductionInput, ReductionInputDomain, RegionOp, RegionOpKind,
    RegionValue,
};

pub(super) fn lower_online_region<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    region: &OnlineRegion,
) {
    let cooperative = region.block_threads().is_some();
    let rank = region.score_shape.len();
    let rows = region.score_shape[rank - 2];
    let keys = region.score_shape[rank - 1];
    let columns = region.output_shape[rank - 1];
    let (tid, fiber, column) = if cooperative {
        let fiber = func.add_u64_register();
        let column = func.add_u64_register();
        func.add_inst(Inst::convert_u64_u32(
            fiber.clone(),
            Operand::symbol("%ctaid.x"),
        ));
        func.add_inst(Inst::convert_u64_u32(
            column.clone(),
            Operand::symbol("%tid.x"),
        ));
        let base = multiply_u64(func, fiber.clone(), columns);
        let tid = add_u64(func, base, column.clone());
        (tid, fiber, column)
    } else {
        let tid = global_linear_tid(func);
        let fiber = divide_u64(func, tid.clone(), columns);
        let column = decode_coordinate(func, tid.clone(), 1, columns);
        (tid, fiber, column)
    };
    let batch = divide_u64(func, fiber.clone(), rows);
    let score_base = multiply_u64(func, fiber, keys);
    let weighted_base = multiply_u64(func, batch, keys * columns);
    let maximum = func.add_f32_register();
    let denominator = func.add_f32_register();
    let numerator = func.add_f32_register();
    func.add_inst(Inst::mov_f32(
        maximum.clone(),
        Operand::imm_f32(f32::NEG_INFINITY),
    ));
    func.add_inst(Inst::mov_f32(denominator.clone(), Operand::imm_f32(0.0)));
    func.add_inst(Inst::mov_f32(numerator.clone(), Operand::imm_f32(0.0)));
    let tile_size = if cooperative { 32 } else { 1 };
    let shared = if cooperative {
        let name = ctx.arena.alloc_str("online_scores");
        func.add_shared_memory(name, 33 * 4);
        let base = func.add_u64_register();
        func.add_inst(Inst::mov_u64(base.clone(), Operand::symbol(name)));
        Some(base)
    } else {
        None
    };
    let scan = begin_counted_loop(func, ctx, "online_scan", keys.div_ceil(tile_size));
    let tile_base = multiply_u64(func, scan.counter.clone(), tile_size);
    if let Some(shared) = &shared {
        let skip = next_label(ctx, "online_score_tile_done");
        let other_warp = func.add_predicate_register();
        func.add_inst(Inst::setp_ge_u64(
            other_warp.clone(),
            column.clone(),
            Operand::imm_u64(32),
        ));
        func.add_inst(Inst::Bra {
            condition: other_warp,
            target: skip,
        });
        let key = add_u64(func, tile_base.clone(), column.clone());
        let outside = func.add_predicate_register();
        let masked = next_label(ctx, "online_score_tile_edge");
        let score = func.add_f32_register();
        func.add_inst(Inst::mov_f32(
            score.clone(),
            Operand::imm_f32(f32::NEG_INFINITY),
        ));
        func.add_inst(Inst::setp_ge_u64(
            outside.clone(),
            key.clone(),
            Operand::imm_u64(keys as u64),
        ));
        func.add_inst(Inst::Bra {
            condition: outside,
            target: masked,
        });
        let offset = add_u64(func, score_base.clone(), key);
        let value = emit_score(func, ctx, region, &region.score, offset);
        func.add_inst(Inst::mov_f32(score.clone(), value));
        func.add_inst(Inst::Label(masked));
        let bytes = multiply_u64(func, column.clone(), 4);
        let address = add_u64(func, shared.clone(), bytes);
        store_shared_from_f32(func, DType::F32, address, score.clone());
        // Reduce the tile reference once, so the coupled state is rescaled once
        // per tile rather than introducing a recurrence at every key.
        for distance in [16, 8, 4, 2, 1] {
            let bits = func.add_b32_register();
            func.add_inst(Inst::mov_b32(bits.clone(), score.clone()));
            let shuffled = func.add_b32_register();
            let valid = func.add_predicate_register();
            func.add_inst(Inst::shfl_sync_down_b32(
                shuffled.clone(),
                valid.clone(),
                bits,
                distance,
            ));
            let peer = func.add_f32_register();
            func.add_inst(Inst::mov_f32_b32(peer.clone(), shuffled));
            func.add_inst(Inst::selp_f32(
                peer.clone(),
                peer.clone(),
                Operand::imm_f32(f32::NEG_INFINITY),
                valid,
            ));
            func.add_inst(Inst::max_f32(score.clone(), score.clone(), peer));
        }
        let leader_done = next_label(ctx, "online_tile_max_done");
        let not_leader = func.add_predicate_register();
        func.add_inst(Inst::setp_ne_u64(
            not_leader.clone(),
            column.clone(),
            Operand::imm_u64(0),
        ));
        func.add_inst(Inst::Bra {
            condition: not_leader,
            target: leader_done,
        });
        let max_address = add_u64(func, shared.clone(), Operand::imm_u64(128));
        store_shared_from_f32(func, DType::F32, max_address, score);
        func.add_inst(Inst::Label(leader_done));
        func.add_inst(Inst::Label(skip));
        func.add_inst(Inst::BarSync { barrier_id: 0 });
    }
    if let Some(shared) = &shared {
        let address = add_u64(func, shared.clone(), Operand::imm_u64(128));
        let tile_maximum = func.add_f32_register();
        load_shared_as_f32(func, DType::F32, tile_maximum.clone(), address);
        let next_maximum = func.add_f32_register();
        func.add_inst(Inst::max_f32(
            next_maximum.clone(),
            maximum.clone(),
            tile_maximum,
        ));
        let alpha = exp_difference(func, maximum.clone(), next_maximum.clone());
        let empty = func.add_predicate_register();
        func.add_inst(Inst::setp_eq_f32(
            empty.clone(),
            next_maximum.clone(),
            Operand::imm_f32(f32::NEG_INFINITY),
        ));
        func.add_inst(Inst::selp_f32(
            alpha.clone(),
            Operand::imm_f32(1.0),
            alpha.clone(),
            empty,
        ));
        func.add_inst(Inst::mov_f32(maximum.clone(), next_maximum));
        func.add_inst(Inst::mul_f32(
            denominator.clone(),
            denominator.clone(),
            alpha.clone(),
        ));
        func.add_inst(Inst::mul_f32(numerator.clone(), numerator.clone(), alpha));
    }
    let consume = begin_counted_loop(func, ctx, "online_consume", tile_size);
    let key = add_u64(func, tile_base, consume.counter.clone());
    let next = next_label(ctx, "online_next_key");
    let outside = func.add_predicate_register();
    func.add_inst(Inst::setp_ge_u64(
        outside.clone(),
        key.clone(),
        Operand::imm_u64(keys as u64),
    ));
    func.add_inst(Inst::Bra {
        condition: outside,
        target: next,
    });
    let score = if let Some(shared) = &shared {
        let bytes = multiply_u64(func, consume.counter.clone(), 4);
        let address = add_u64(func, shared.clone(), bytes);
        let value = func.add_f32_register();
        load_shared_as_f32(func, DType::F32, value.clone(), address);
        value
    } else {
        let offset = add_u64(func, score_base, key.clone());
        emit_score(func, ctx, region, &region.score, offset)
    };
    if !cooperative {
        let next_maximum = func.add_f32_register();
        func.add_inst(Inst::max_f32(
            next_maximum.clone(),
            maximum.clone(),
            score.clone(),
        ));
        let alpha = exp_difference(func, maximum.clone(), next_maximum.clone());
        let empty = func.add_predicate_register();
        func.add_inst(Inst::setp_eq_f32(
            empty.clone(),
            next_maximum.clone(),
            Operand::imm_f32(f32::NEG_INFINITY),
        ));
        func.add_inst(Inst::selp_f32(
            alpha.clone(),
            Operand::imm_f32(1.0),
            alpha.clone(),
            empty,
        ));
        func.add_inst(Inst::mov_f32(maximum.clone(), next_maximum));
        func.add_inst(Inst::mul_f32(
            denominator.clone(),
            denominator.clone(),
            alpha.clone(),
        ));
        func.add_inst(Inst::mul_f32(numerator.clone(), numerator.clone(), alpha));
    }
    let probability = exp_difference(func, score.clone(), maximum);
    // Leading masked values preserve the identity. NaN scores remain NaN, and
    // zero probabilities still multiply V to preserve nonfinite propagation.
    let masked = func.add_predicate_register();
    func.add_inst(Inst::setp_eq_f32(
        masked.clone(),
        score,
        Operand::imm_f32(f32::NEG_INFINITY),
    ));
    func.add_inst(Inst::selp_f32(
        probability.clone(),
        Operand::imm_f32(0.0),
        probability.clone(),
        masked,
    ));
    func.add_inst(Inst::add_f32(
        denominator.clone(),
        denominator.clone(),
        probability.clone(),
    ));
    let inactive_end = next_label(ctx, "online_inactive_column");
    if cooperative {
        let inactive = func.add_predicate_register();
        func.add_inst(Inst::setp_ge_u64(
            inactive.clone(),
            column.clone(),
            Operand::imm_u64(columns as u64),
        ));
        func.add_inst(Inst::Bra {
            condition: inactive,
            target: inactive_end,
        });
    }
    let weight_row = multiply_u64(func, key, columns);
    let weight_offset = add_u64(func, weighted_base, weight_row);
    let weight_offset = add_u64(func, weight_offset, column.clone());
    let weight = load_input(func, ctx, region, region.weights, weight_offset);
    let contribution = func.add_f32_register();
    func.add_inst(Inst::mul_f32(contribution.clone(), probability, weight));
    func.add_inst(Inst::add_f32(
        numerator.clone(),
        numerator.clone(),
        contribution,
    ));
    func.add_inst(Inst::Label(inactive_end));
    func.add_inst(Inst::Label(next));
    end_counted_loop(func, consume);
    if cooperative {
        func.add_inst(Inst::BarSync { barrier_id: 0 });
    }
    end_counted_loop(func, scan);
    let result = func.add_f32_register();
    func.add_inst(Inst::div_f32(result.clone(), numerator, denominator));
    let done = next_label(ctx, "online_store_done");
    if cooperative {
        let inactive = func.add_predicate_register();
        func.add_inst(Inst::setp_ge_u64(
            inactive.clone(),
            column,
            Operand::imm_u64(columns as u64),
        ));
        func.add_inst(Inst::Bra {
            condition: inactive,
            target: done,
        });
    }
    store_param_f32_at(func, ctx, "output_0", tid, result);
    func.add_inst(Inst::Label(done));
}

fn load_input<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    region: &OnlineRegion,
    index: usize,
    logical_offset: Operand<'a, U64>,
) -> Operand<'a, F32> {
    let input = &region.inputs[index];
    let descriptor = ReductionInput {
        node: input.node,
        value: input.value,
        domain: ReductionInputDomain::Full,
        tensor: input.tensor.clone(),
        source_shape: input.source_shape.clone(),
    };
    let offset =
        reduction_region_input_offset(func, &descriptor, &input.tensor.shape, logical_offset);
    load_param_f32_at(func, ctx, &format!("input_{index}"), offset)
}

fn emit_score<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    region: &OnlineRegion,
    expression: &OnlineExpr,
    offset: Operand<'a, U64>,
) -> Operand<'a, F32> {
    match expression {
        OnlineExpr::Input(index) => load_input(func, ctx, region, *index, offset),
        OnlineExpr::Dot { lhs, rhs, width } => {
            let rank = region.score_shape.len();
            let rows = region.score_shape[rank - 2];
            let keys = region.score_shape[rank - 1];
            let row = divide_u64(func, offset.clone(), keys);
            let key = decode_coordinate(func, offset, 1, keys);
            let batch = divide_u64(func, row.clone(), rows);
            let lhs_base = multiply_u64(func, row, *width);
            let rhs_base = multiply_u64(func, batch, width * keys);
            let rhs_base = add_u64(func, rhs_base, key);
            // For affine views, decode the fixed coordinates once before the
            // contraction and advance the physical address with the axis stride.
            let lhs_access = affine_axis(func, region, *lhs, lhs_base.clone(), rank - 1);
            let rhs_access = affine_axis(func, region, *rhs, rhs_base.clone(), rank - 2);
            let result = func.add_f32_register();
            func.add_inst(Inst::mov_f32(result.clone(), Operand::imm_f32(0.0)));
            let dot = begin_counted_loop(func, ctx, "online_dot", *width);
            let a = if let Some((base, stride)) = lhs_access {
                let step = multiply_u64(func, dot.counter.clone(), stride);
                let offset = add_u64(func, base, step);
                load_param_f32_at(func, ctx, &format!("input_{lhs}"), offset)
            } else {
                let offset = add_u64(func, lhs_base, dot.counter.clone());
                load_input(func, ctx, region, *lhs, offset)
            };
            let b = if let Some((base, stride)) = rhs_access {
                let step = multiply_u64(func, dot.counter.clone(), stride);
                let offset = add_u64(func, base, step);
                load_param_f32_at(func, ctx, &format!("input_{rhs}"), offset)
            } else {
                let step = multiply_u64(func, dot.counter.clone(), keys);
                let offset = add_u64(func, rhs_base, step);
                load_input(func, ctx, region, *rhs, offset)
            };
            let product = func.add_f32_register();
            func.add_inst(Inst::mul_f32(product.clone(), a, b));
            func.add_inst(Inst::add_f32(result.clone(), result.clone(), product));
            end_counted_loop(func, dot);
            result
        }
        OnlineExpr::Unary(op, value) => {
            let value = emit_score(func, ctx, region, value, offset);
            scalar_operation(
                func,
                region.output,
                RegionOpKind::Unary {
                    op: *op,
                    input: RegionValue(0),
                },
                vec![value],
            )
        }
        OnlineExpr::Binary(op, lhs, rhs) => {
            let a = emit_score(func, ctx, region, lhs, offset.clone());
            let b = emit_score(func, ctx, region, rhs, offset);
            scalar_operation(
                func,
                region.output,
                RegionOpKind::Binary {
                    op: *op,
                    lhs: RegionValue(0),
                    rhs: RegionValue(1),
                },
                vec![a, b],
            )
        }
    }
}
fn scalar_operation<'a>(
    func: &mut Function<'a>,
    node: NodeIndex,
    kind: RegionOpKind,
    operands: Vec<Operand<'a, F32>>,
) -> Operand<'a, F32> {
    let mut values: HashMap<_, _> = operands
        .into_iter()
        .enumerate()
        .map(|(i, value)| (RegionValue(i), value))
        .collect();
    let output = RegionValue(2);
    emit_region_op(func, &RegionOp { node, output, kind }, &mut values);
    values[&output].clone()
}
fn exp_difference<'a>(
    func: &mut Function<'a>,
    a: Operand<'a, F32>,
    b: Operand<'a, F32>,
) -> Operand<'a, F32> {
    let difference = func.add_f32_register();
    func.add_inst(Inst::sub_f32(difference.clone(), a, b));
    func.add_inst(Inst::mul_f32(
        difference.clone(),
        difference.clone(),
        Operand::imm_f32(std::f32::consts::LOG2_E),
    ));
    let result = func.add_f32_register();
    func.add_inst(Inst::ex2_f32(result.clone(), difference));
    result
}
fn add_u64<'a>(
    func: &mut Function<'a>,
    a: Operand<'a, U64>,
    b: Operand<'a, U64>,
) -> Operand<'a, U64> {
    let result = func.add_u64_register();
    func.add_inst(Inst::add_u64(result.clone(), a, b));
    result
}
fn divide_u64<'a>(
    func: &mut Function<'a>,
    value: Operand<'a, U64>,
    divisor: usize,
) -> Operand<'a, U64> {
    if divisor == 1 {
        return value;
    }
    let result = func.add_u64_register();
    func.add_inst(Inst::div_u64(
        result.clone(),
        value,
        Operand::imm_u64(divisor as u64),
    ));
    result
}

fn affine_axis<'a>(
    func: &mut Function<'a>,
    region: &OnlineRegion,
    index: usize,
    logical_base: Operand<'a, U64>,
    axis: usize,
) -> Option<(Operand<'a, U64>, usize)> {
    let input = &region.inputs[index];
    let strides = contiguous_strides(&input.source_shape);
    let mut coefficient = 0usize;
    for (expr, stride) in input.tensor.access.results.iter().zip(strides) {
        match expr {
            crate::tile::IndexExpr::IterDim(dimension) => {
                if *dimension == axis {
                    coefficient += stride;
                }
            }
            crate::tile::IndexExpr::Const(_) => {}
            _ => return None,
        }
    }
    let descriptor = ReductionInput {
        node: input.node,
        value: input.value,
        domain: ReductionInputDomain::Full,
        tensor: input.tensor.clone(),
        source_shape: input.source_shape.clone(),
    };
    let base = reduction_region_input_offset(func, &descriptor, &input.tensor.shape, logical_base);
    Some((base, coefficient))
}
