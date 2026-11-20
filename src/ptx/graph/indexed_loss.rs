//! Schedule ordered f32 indexed losses by row, including promoted BF16 logits.
use super::*;
mod cooperative_ordered;
mod cooperative_ordered_forward;

pub(super) const ROW_THREADS: u32 = 256;

#[derive(Clone, Copy)]
pub(super) struct RowSchedule {
    pub rows: u32,
    pub vocabulary: usize,
    pub output_len: usize,
    pub threads: u32,
    pub cooperative_ordered_backward: bool,
    pub cooperative_ordered_forward: bool,
}

pub(super) fn schedule<'a>(
    vocabulary: usize,
    rows: usize,
    parameters: impl Iterator<Item = (&'a str, DType)>,
) -> Option<RowSchedule> {
    if !(256..=65536).contains(&vocabulary)
        || rows == 0
        || rows.checked_mul(vocabulary)? > u32::MAX as usize
        || parameters.into_iter().any(|(name, dtype)| {
            dtype != DType::F32
                && !(name == "targets" && dtype == DType::U32)
                && !(name == "logits" && dtype == DType::BF16)
        })
    {
        return None;
    }
    Some(RowSchedule {
        rows: u32::try_from(rows).ok()?,
        vocabulary,
        output_len: rows * vocabulary,
        threads: ROW_THREADS,
        cooperative_ordered_backward: false,
        cooperative_ordered_forward: false,
    })
}

pub(super) fn schedule_for_ir(ir: &TileIR) -> Option<RowSchedule> {
    if ir
        .body
        .stmts
        .iter()
        .filter(|s| !matches!(s, Stmt::BoundsCheck { .. }))
        .count()
        != 1
    {
        return None;
    }
    ir.body.stmts.iter().find_map(|s| match s {
        Stmt::IndexedCrossEntropyCooperativeOrdered {
            vocabulary,
            row_count,
        } => {
            let mut row = schedule(
                *vocabulary,
                *row_count,
                ir.params.iter().map(|p| (p.name.as_str(), p.dtype)),
            )?;
            if *vocabulary != 4096 || !forward_parameters(ir) {
                return None;
            }
            row.output_len = *row_count;
            row.cooperative_ordered_forward = true;
            Some(row)
        }
        Stmt::IndexedCrossEntropyBackwardCooperativeOrdered {
            vocabulary,
            row_count,
        } => {
            let mut schedule = schedule(
                *vocabulary,
                *row_count,
                ir.params.iter().map(|p| (p.name.as_str(), p.dtype)),
            )?;
            if *vocabulary != 4096 || !cooperative_parameters(ir) {
                return None;
            }
            schedule.cooperative_ordered_backward = true;
            Some(schedule)
        }
        Stmt::IndexedCrossEntropyBackward {
            vocabulary,
            row_count,
        } => schedule(
            *vocabulary,
            *row_count,
            ir.params.iter().map(|p| (p.name.as_str(), p.dtype)),
        ),
        Stmt::IndexedCrossEntropy {
            vocabulary,
            row_count,
        } => {
            let mut schedule = schedule(
                *vocabulary,
                *row_count,
                ir.params.iter().map(|p| (p.name.as_str(), p.dtype)),
            )?;
            schedule.output_len = *row_count;
            schedule.threads = 32;
            Some(schedule)
        }
        _ => None,
    })
}

pub(super) fn forward_row<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
) -> Operand<'a, U64> {
    // Each row owns one warp-sized block. Only its first lane executes the
    // original ordered scalar loops; independent rows can occupy different SMs.
    let lane = lower_expr(func, ctx, &Expr::ThreadIdx(Dim::X));
    let follower = func.add_predicate_register();
    func.add_inst(Inst::setp_ne_u64(
        follower.clone(),
        lane,
        Operand::imm_u64(0),
    ));
    func.add_inst(Inst::Bra {
        condition: follower,
        target: ctx.return_label,
    });
    lower_expr(func, ctx, &Expr::BlockIdx(Dim::X))
}

pub(super) fn lower_backward<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    vocabulary: usize,
) {
    let row = lower_expr(func, ctx, &Expr::BlockIdx(Dim::X));
    let lane = lower_expr(func, ctx, &Expr::ThreadIdx(Dim::X));
    let row_start = multiply_u64(func, row.clone(), vocabulary);
    let shared_name = ctx.arena.alloc_str("ce_row_stats");
    func.add_shared_memory(shared_name, 2 * std::mem::size_of::<f32>());
    let base = func.add_u64_register();
    func.add_inst(Inst::mov_u64(base.clone(), Operand::symbol(shared_name)));
    let sum_address = func.add_u64_register();
    func.add_inst(Inst::add_u64(
        sum_address.clone(),
        base.clone(),
        Operand::imm_u64(4),
    ));
    let broadcast = next_label(ctx, "ce_row_broadcast");
    let follower = func.add_predicate_register();
    func.add_inst(Inst::setp_ne_u64(
        follower.clone(),
        lane.clone(),
        Operand::imm_u64(0),
    ));
    func.add_inst(Inst::Bra {
        condition: follower,
        target: broadcast,
    });
    // Exactly the original increasing-column loops, initial values, exp
    // approximation and f32 sum. Only redundant executions are removed.
    let (maximum, sum) = emit_row_softmax_stats(func, ctx, row_start.clone(), vocabulary);
    func.add_inst(Inst::StSharedF32 {
        addr: base.clone(),
        src: vec![maximum],
        vec: super::super::instructions::VecWidth::Scalar,
    });
    func.add_inst(Inst::StSharedF32 {
        addr: sum_address.clone(),
        src: vec![sum],
        vec: super::super::instructions::VecWidth::Scalar,
    });
    func.add_inst(Inst::Label(broadcast));
    func.add_inst(Inst::BarSync { barrier_id: 0 });
    let maximum = func.add_f32_register();
    let sum = func.add_f32_register();
    func.add_inst(Inst::LdSharedF32 {
        dst: vec![maximum.clone()],
        addr: base,
        vec: super::super::instructions::VecWidth::Scalar,
    });
    func.add_inst(Inst::LdSharedF32 {
        dst: vec![sum.clone()],
        addr: sum_address,
        vec: super::super::instructions::VecWidth::Scalar,
    });
    lower_backward_columns(func, ctx, vocabulary, row, lane, row_start, maximum, sum);
}

pub(super) fn lower_cooperative_ordered_backward<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    vocabulary: usize,
) {
    debug_assert_eq!(vocabulary, 4096);
    let row = lower_expr(func, ctx, &Expr::BlockIdx(Dim::X));
    let lane = lower_expr(func, ctx, &Expr::ThreadIdx(Dim::X));
    let row_start = multiply_u64(func, row.clone(), vocabulary);
    let stats = cooperative_ordered::row_stats_ordered_sum(func, ctx, row_start.clone());
    lower_backward_columns(
        func,
        ctx,
        vocabulary,
        row,
        lane,
        row_start,
        stats.maximum,
        stats.sum,
    );
}

#[allow(clippy::too_many_arguments)]
fn lower_backward_columns<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    vocabulary: usize,
    row: Operand<'a, U64>,
    lane: Operand<'a, U64>,
    row_start: Operand<'a, U64>,
    maximum: Operand<'a, F32>,
    sum: Operand<'a, F32>,
) {
    let target = load_index_at(func, ctx, "targets", row.clone());
    let upstream = load_param_f32_at(func, ctx, "grad_output", row);
    let columns = begin_strided_counted_loop(
        func,
        ctx,
        "ce_row_column",
        vocabulary,
        lane,
        u64::from(ROW_THREADS),
    );
    let offset = func.add_u64_register();
    func.add_inst(Inst::add_u64(
        offset.clone(),
        row_start,
        columns.counter.clone(),
    ));
    let logit = load_param_f32_at(func, ctx, "logits", offset.clone());
    let shifted = func.add_f32_register();
    func.add_inst(Inst::sub_f32(shifted.clone(), logit, maximum));
    let exponential = emit_exp(func, shifted);
    let probability = func.add_f32_register();
    func.add_inst(Inst::div_f32(probability.clone(), exponential, sum));
    let is_target = func.add_predicate_register();
    func.add_inst(Inst::setp_eq_u64(
        is_target.clone(),
        columns.counter.clone(),
        target,
    ));
    let indicator = func.add_f32_register();
    func.add_inst(Inst::selp_f32(
        indicator.clone(),
        Operand::imm_f32(1.0),
        Operand::imm_f32(0.0),
        is_target,
    ));
    let difference = func.add_f32_register();
    func.add_inst(Inst::sub_f32(difference.clone(), probability, indicator));
    let gradient = func.add_f32_register();
    func.add_inst(Inst::mul_f32(gradient.clone(), difference, upstream));
    store_param_f32_at(func, ctx, "output", offset, gradient);
    end_counted_loop(func, columns);
}

#[cfg(test)]
#[path = "indexed_loss_tests.rs"]
mod tests;

fn cooperative_parameters(ir: &TileIR) -> bool {
    ir.shared_mem_bytes == 0
        && ir.params.len() == 4
        && [
            ("logits", true),
            ("targets", true),
            ("grad_output", true),
            ("output", false),
        ]
        .iter()
        .all(|(name, input)| {
            ir.params
                .iter()
                .filter(|p| {
                    p.name == *name
                        && p.is_input == *input
                        && match *name {
                            "logits" => matches!(p.dtype, DType::F32 | DType::BF16),
                            "targets" => matches!(p.dtype, DType::F32 | DType::U32),
                            _ => p.dtype == DType::F32,
                        }
                })
                .count()
                == 1
        })
}

pub(super) fn cooperative_ordered_eligible(ir: &TileIR, shared_limit: usize) -> bool {
    let Some(schedule) = schedule_for_ir(ir) else {
        return false;
    };
    schedule.vocabulary == 4096
        && shared_limit >= cooperative_ordered::SHARED_BYTES
        && cooperative_parameters(ir)
        && ir
            .body
            .stmts
            .iter()
            .any(|s| matches!(s, Stmt::IndexedCrossEntropyBackward { .. }))
}

pub(super) fn enable_cooperative_ordered_backward(ir: &mut TileIR, shared_limit: usize) -> bool {
    if !cooperative_ordered_eligible(ir, shared_limit) {
        return false;
    }
    let Some(stmt) = ir
        .body
        .stmts
        .iter_mut()
        .find(|s| matches!(s, Stmt::IndexedCrossEntropyBackward { .. }))
    else {
        return false;
    };
    let (vocabulary, row_count) = match stmt {
        Stmt::IndexedCrossEntropyBackward {
            vocabulary,
            row_count,
        } => (*vocabulary, *row_count),
        _ => return false,
    };
    *stmt = Stmt::IndexedCrossEntropyBackwardCooperativeOrdered {
        vocabulary,
        row_count,
    };
    ir.kernel_name.push_str("_cooperative_ordered_backward");
    true
}

pub(super) fn cooperative_ordered_resources_fit(
    shared: usize,
    local: usize,
    max_threads: i32,
    active_blocks: u32,
    shared_limit: usize,
) -> bool {
    shared >= cooperative_ordered::SHARED_BYTES
        && shared <= shared_limit
        && local == 0
        && max_threads >= 256
        && active_blocks >= 1
}

#[cfg(test)]
mod cooperative_tests;

pub(super) fn forward_parameters(ir: &TileIR) -> bool {
    ir.params.len() == 3
        && [("logits", true), ("targets", true), ("output", false)]
            .iter()
            .all(|(name, input)| {
                ir.params
                    .iter()
                    .filter(|p| {
                        p.name == *name
                            && p.is_input == *input
                            && match *name {
                                "logits" => matches!(p.dtype, DType::F32 | DType::BF16),
                                "targets" => matches!(p.dtype, DType::F32 | DType::U32),
                                _ => p.dtype == DType::F32,
                            }
                    })
                    .count()
                    == 1
            })
}
pub(super) fn forward_eligible(ir: &TileIR, limit: usize) -> bool {
    schedule_for_ir(ir).is_some_and(|s| s.vocabulary == 4096 && s.threads == 32 && s.rows > 0)
        && ir.shared_mem_bytes == 0
        && limit >= cooperative_ordered::SHARED_BYTES
        && forward_parameters(ir)
        && ir
            .body
            .stmts
            .iter()
            .any(|s| matches!(s, Stmt::IndexedCrossEntropy { .. }))
}
pub(super) fn enable_cooperative_ordered_forward(ir: &mut TileIR, limit: usize) -> bool {
    if !forward_eligible(ir, limit) {
        return false;
    }
    for stmt in &mut ir.body.stmts {
        if let Stmt::IndexedCrossEntropy {
            vocabulary,
            row_count,
        } = stmt
        {
            *stmt = Stmt::IndexedCrossEntropyCooperativeOrdered {
                vocabulary: *vocabulary,
                row_count: *row_count,
            };
            ir.kernel_name.push_str("_cooperative_ordered_forward");
            return true;
        }
    }
    false
}
pub(super) use cooperative_ordered_forward::lower_forward as lower_cooperative_ordered_forward;

#[cfg(test)]
mod forward_tests;
