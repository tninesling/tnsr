//! Undeclared preparation for an explicit parallel cross-entropy experiment.
//!
//! This changes the F32 sum order. It is not an exact replacement for the
//! ordered row statistics and must remain disconnected until independently
//! qualified. V1 is restricted to 4096 columns and 256 participating threads.
//! Every thread must execute both statistics and all barriers; forward's
//! follower return belongs AFTER this helper. MAX and SUM have distinct storage.
//! Small positive exponential terms lost by the ordered sum can survive this
//! tree, producing a nonzero target gradient where the old path produced zero.
//! Adam can amplify that difference; unchanged full update gates are mandatory.
use super::*;

const CONTRIBUTORS: usize = 256;
const VOCABULARY: usize = 4096;

fn shared_address<'a>(
    func: &mut Function<'a>,
    base: Operand<'a, U64>,
    index: Operand<'a, U64>,
) -> Operand<'a, U64> {
    let offset = multiply_u64(func, index, std::mem::size_of::<f32>());
    let address = func.add_u64_register();
    func.add_inst(Inst::add_u64(address.clone(), base, offset));
    address
}

fn tree<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    lane: Operand<'a, U64>,
    base: Operand<'a, U64>,
    partial: Operand<'a, F32>,
    is_maximum: bool,
) -> Operand<'a, F32> {
    let own_address = shared_address(func, base.clone(), lane.clone());
    func.add_inst(Inst::StSharedF32 {
        addr: own_address.clone(),
        src: vec![partial.clone()],
        vec: super::super::super::instructions::VecWidth::Scalar,
    });
    func.add_inst(Inst::BarSync { barrier_id: 0 });
    for stride in [128_u64, 64, 32, 16, 8, 4, 2, 1] {
        let done = next_label(ctx, "ce_parallel_tree_join");
        let inactive = func.add_predicate_register();
        func.add_inst(Inst::setp_ge_u64(
            inactive.clone(),
            lane.clone(),
            Operand::imm_u64(stride),
        ));
        func.add_inst(Inst::Bra {
            condition: inactive,
            target: done,
        });
        let peer_index = func.add_u64_register();
        func.add_inst(Inst::add_u64(
            peer_index.clone(),
            lane.clone(),
            Operand::imm_u64(stride),
        ));
        let peer_address = shared_address(func, base.clone(), peer_index);
        let peer = func.add_f32_register();
        func.add_inst(Inst::LdSharedF32 {
            dst: vec![peer.clone()],
            addr: peer_address,
            vec: super::super::super::instructions::VecWidth::Scalar,
        });
        if is_maximum {
            func.add_inst(Inst::max_f32(partial.clone(), partial.clone(), peer));
        } else {
            func.add_inst(Inst::add_f32(partial.clone(), partial.clone(), peer));
        }
        func.add_inst(Inst::StSharedF32 {
            addr: own_address.clone(),
            src: vec![partial.clone()],
            vec: super::super::super::instructions::VecWidth::Scalar,
        });
        func.add_inst(Inst::Label(done));
        // Inactive lanes skip arithmetic, never the whole-block rendezvous.
        func.add_inst(Inst::BarSync { barrier_id: 0 });
    }
    let result = func.add_f32_register();
    func.add_inst(Inst::LdSharedF32 {
        dst: vec![result.clone()],
        addr: base,
        vec: super::super::super::instructions::VecWidth::Scalar,
    });
    result
}

pub(super) fn row_stats<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    row_start: Operand<'a, U64>,
) -> (Operand<'a, F32>, Operand<'a, F32>) {
    let lane = lower_expr(func, ctx, &Expr::ThreadIdx(Dim::X));
    let name = next_label(ctx, "ce_parallel_workspace");
    func.add_shared_memory(name, 2 * CONTRIBUTORS * std::mem::size_of::<f32>());
    let max_base = func.add_u64_register();
    func.add_inst(Inst::mov_u64(max_base.clone(), Operand::symbol(name)));
    let sum_base = func.add_u64_register();
    func.add_inst(Inst::add_u64(
        sum_base.clone(),
        max_base.clone(),
        Operand::imm_u64(1024),
    ));
    let maximum = func.add_f32_register();
    func.add_inst(Inst::mov_f32(
        maximum.clone(),
        Operand::imm_f32(f32::NEG_INFINITY),
    ));
    let columns = begin_strided_counted_loop(
        func,
        ctx,
        "ce_parallel_max_column",
        VOCABULARY,
        lane.clone(),
        256,
    );
    let offset = func.add_u64_register();
    func.add_inst(Inst::add_u64(
        offset.clone(),
        row_start.clone(),
        columns.counter.clone(),
    ));
    let logit = load_param_f32_at(func, ctx, "logits", offset);
    func.add_inst(Inst::max_f32(maximum.clone(), maximum.clone(), logit));
    end_counted_loop(func, columns);
    let maximum = tree(func, ctx, lane.clone(), max_base, maximum, true);

    let sum = func.add_f32_register();
    func.add_inst(Inst::mov_f32(sum.clone(), Operand::imm_f32(0.0)));
    let columns = begin_strided_counted_loop(
        func,
        ctx,
        "ce_parallel_sum_column",
        VOCABULARY,
        lane.clone(),
        256,
    );
    let offset = func.add_u64_register();
    func.add_inst(Inst::add_u64(
        offset.clone(),
        row_start,
        columns.counter.clone(),
    ));
    let logit = load_param_f32_at(func, ctx, "logits", offset);
    let shifted = func.add_f32_register();
    func.add_inst(Inst::sub_f32(shifted.clone(), logit, maximum.clone()));
    let exponential = emit_exp(func, shifted);
    func.add_inst(Inst::add_f32(sum.clone(), sum.clone(), exponential));
    end_counted_loop(func, columns);
    let sum = tree(func, ctx, lane, sum_base, sum, false);
    (maximum, sum)
}

/// Separate, unqualified preparation for cooperative exp with an ORIGINAL
/// ordered sum. Intended first for backward, whose existing block is 256 threads.
/// This has MAX[256], EXP[4096] and SUM[1] in distinct shared storage. It retains
/// the 4096-add chain; it makes no speed or exact-equivalence claim before tests.
pub(super) fn row_stats_ordered_sum<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    row_start: Operand<'a, U64>,
) -> (Operand<'a, F32>, Operand<'a, F32>) {
    let lane = lower_expr(func, ctx, &Expr::ThreadIdx(Dim::X));
    let name = next_label(ctx, "ce_cooperative_ordered_workspace");
    func.add_shared_memory(
        name,
        (CONTRIBUTORS + VOCABULARY + 1) * std::mem::size_of::<f32>(),
    );
    let max_base = func.add_u64_register();
    func.add_inst(Inst::mov_u64(max_base.clone(), Operand::symbol(name)));
    let exp_base = func.add_u64_register();
    func.add_inst(Inst::add_u64(
        exp_base.clone(),
        max_base.clone(),
        Operand::imm_u64(1024),
    ));
    let sum_address = func.add_u64_register();
    func.add_inst(Inst::add_u64(
        sum_address.clone(),
        exp_base.clone(),
        Operand::imm_u64(16384),
    ));
    let maximum = func.add_f32_register();
    func.add_inst(Inst::mov_f32(
        maximum.clone(),
        Operand::imm_f32(f32::NEG_INFINITY),
    ));
    let columns = begin_strided_counted_loop(
        func,
        ctx,
        "ce_cooperative_max_column",
        VOCABULARY,
        lane.clone(),
        256,
    );
    let offset = func.add_u64_register();
    func.add_inst(Inst::add_u64(
        offset.clone(),
        row_start.clone(),
        columns.counter.clone(),
    ));
    let logit = load_param_f32_at(func, ctx, "logits", offset);
    func.add_inst(Inst::max_f32(maximum.clone(), maximum.clone(), logit));
    end_counted_loop(func, columns);
    let maximum = tree(func, ctx, lane.clone(), max_base, maximum, true);

    let columns = begin_strided_counted_loop(
        func,
        ctx,
        "ce_cooperative_exp_column",
        VOCABULARY,
        lane.clone(),
        256,
    );
    let offset = func.add_u64_register();
    func.add_inst(Inst::add_u64(
        offset.clone(),
        row_start,
        columns.counter.clone(),
    ));
    let logit = load_param_f32_at(func, ctx, "logits", offset);
    let shifted = func.add_f32_register();
    func.add_inst(Inst::sub_f32(shifted.clone(), logit, maximum.clone()));
    let exponential = emit_exp(func, shifted);
    let address = shared_address(func, exp_base.clone(), columns.counter.clone());
    func.add_inst(Inst::StSharedF32 {
        addr: address,
        src: vec![exponential],
        vec: super::super::super::instructions::VecWidth::Scalar,
    });
    end_counted_loop(func, columns);
    func.add_inst(Inst::BarSync { barrier_id: 0 });

    let broadcast = next_label(ctx, "ce_cooperative_sum_broadcast");
    let follower = func.add_predicate_register();
    func.add_inst(Inst::setp_ne_u64(
        follower.clone(),
        lane,
        Operand::imm_u64(0),
    ));
    func.add_inst(Inst::Bra {
        condition: follower,
        target: broadcast,
    });
    let sum = func.add_f32_register();
    func.add_inst(Inst::mov_f32(sum.clone(), Operand::imm_f32(0.0)));
    let columns = begin_strided_counted_loop(
        func,
        ctx,
        "ce_cooperative_ordered_column",
        VOCABULARY,
        Operand::imm_u64(0),
        16,
    );
    // Prefetch one contiguous group, then retain its exact increasing-column
    // left fold. Never introduce independent sums or a horizontal tree here.
    let mut prefetched = Vec::with_capacity(16);
    for offset in 0..16_u64 {
        let index = func.add_u64_register();
        func.add_inst(Inst::add_u64(
            index.clone(),
            columns.counter.clone(),
            Operand::imm_u64(offset),
        ));
        let address = shared_address(func, exp_base.clone(), index);
        let term = func.add_f32_register();
        func.add_inst(Inst::LdSharedF32 {
            dst: vec![term.clone()],
            addr: address,
            vec: super::super::super::instructions::VecWidth::Scalar,
        });
        prefetched.push(term);
    }
    for term in prefetched {
        func.add_inst(Inst::add_f32(sum.clone(), sum.clone(), term));
    }
    end_counted_loop(func, columns);
    func.add_inst(Inst::StSharedF32 {
        addr: sum_address.clone(),
        src: vec![sum],
        vec: super::super::super::instructions::VecWidth::Scalar,
    });
    func.add_inst(Inst::Label(broadcast));
    func.add_inst(Inst::BarSync { barrier_id: 0 });
    let sum = func.add_f32_register();
    func.add_inst(Inst::LdSharedF32 {
        dst: vec![sum.clone()],
        addr: sum_address,
        vec: super::super::super::instructions::VecWidth::Scalar,
    });
    (maximum, sum)
}
