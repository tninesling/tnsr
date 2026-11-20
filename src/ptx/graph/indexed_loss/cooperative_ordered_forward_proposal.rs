//! UNDECLARED forward preparation: no production hook, option, IR variant or
//! launch change. The current 32-thread ordered forward remains untouched.
//! Reuse only the exact cooperative-max/exp plus ORIGINAL serial-sum contract.
//! The numerical parallel-sum-tree proposal remains disconnected.
use super::*;

pub(super) const FORWARD_THREADS: u32 = 256;
pub(super) const VOCABULARY: usize = 4096;

/// Pure future eligibility proof over the ORIGINAL standalone forward IR.
/// This neither mutates IR nor selects a launch. Future integration must derive
/// selected/fallback launch metadata from the actual final compiled graph.
pub(super) fn eligible_original_forward(ir: &TileIR, shared_limit: usize) -> bool {
    let Some(schedule) = schedule_for_ir(ir) else { return false; };
    if schedule.vocabulary != VOCABULARY || schedule.rows == 0
        || schedule.output_len != schedule.rows as usize
        || schedule.threads != 32
        || shared_limit < cooperative_ordered::SHARED_BYTES
        || ir.shared_mem_bytes != 0 || ir.params.len() != 3
        || !ir.body.stmts.iter().any(|s| matches!(s, Stmt::IndexedCrossEntropy { vocabulary: 4096, .. }))
    { return false; }
    [("logits", true), ("targets", true), ("output", false)].iter().all(|(name, input)| {
        ir.params.iter().filter(|p| p.name == *name && p.is_input == *input && match *name {
            "logits" => matches!(p.dtype, DType::F32 | DType::BF16),
            "targets" => matches!(p.dtype, DType::F32 | DType::U32),
            _ => p.dtype == DType::F32,
        }).count() == 1
    })
}

/// Future caller contract: one valid row per full 256-thread block, V4096,
/// compatible typed three-parameter ABI, and no return before this function.
/// This must NOT be entered through existing forward_row(), which returns the
/// follower lanes before statistics and would deadlock a cooperative kernel.
pub(super) fn lower_forward<'a>(func: &mut Function<'a>, ctx: &mut LoweringContext<'a>) {
    let row = lower_expr(func, ctx, &Expr::BlockIdx(Dim::X));
    let row_start = multiply_u64(func, row.clone(), VOCABULARY);
    let stats = cooperative_ordered::row_stats_ordered_sum(func, ctx, row_start.clone());
    // All MAX, EXP-publication and SUM-broadcast barriers have completed.
    // From here, only lane0 computes/stores the original scalar loss.
    let lane = lower_expr(func, ctx, &Expr::ThreadIdx(Dim::X));
    let follower = func.add_predicate_register();
    func.add_inst(Inst::setp_ne_u64(follower.clone(), lane, Operand::imm_u64(0)));
    func.add_inst(Inst::Bra { condition: follower, target: ctx.return_label });
    let target = load_index_at(func, ctx, "targets", row.clone());
    let target_offset = func.add_u64_register();
    func.add_inst(Inst::add_u64(target_offset.clone(), row_start, target));
    let target_logit = load_param_f32_at(func, ctx, "logits", target_offset);
    let shifted = func.add_f32_register();
    func.add_inst(Inst::sub_f32(shifted.clone(), stats.maximum, target_logit));
    let log_sum = emit_log(func, stats.sum);
    let loss = func.add_f32_register();
    func.add_inst(Inst::add_f32(loss.clone(), shifted, log_sum));
    store_param_f32_at(func, ctx, "output", row, loss);
}

#[cfg(test)]
mod tests;
