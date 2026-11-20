//! Exact cooperative forward, all barriers precede follower return.
use super::*;
const VOCABULARY: usize = 4096;
pub(in crate::ptx::graph) fn lower_forward<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
) {
    let row = lower_expr(func, ctx, &Expr::BlockIdx(Dim::X));
    let row_start = multiply_u64(func, row.clone(), VOCABULARY);
    let stats = cooperative_ordered::row_stats_ordered_sum(func, ctx, row_start.clone());
    // All MAX, EXP-publication and SUM-broadcast barriers have completed.
    // From here, only lane0 computes/stores the original scalar loss.
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
