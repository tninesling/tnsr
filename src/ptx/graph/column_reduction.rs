//! Adjacent-column mapping with the original per-column Block128 arithmetic.
use super::*;

pub(super) fn lower<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    region: &crate::tile::ReductionRegion,
    tile: crate::tile::ColumnReductionTile,
) {
    let legacy = next_label(ctx, "column_reduction_legacy");
    let block_threads = lower_expr(func, ctx, &Expr::BlockDim(Dim::X));
    let use_legacy = func.add_predicate_register();
    func.add_inst(Inst::setp_ne_u64(
        use_legacy.clone(),
        block_threads,
        Operand::imm_u64(u64::from(tile.threads)),
    ));
    // The launch dimension is uniform across the entire CTA. Each path has
    // its own complete set of barriers; no thread can enter the other path.
    func.add_inst(Inst::Bra {
        condition: use_legacy,
        target: legacy,
    });
    lower_columns(func, ctx, region, tile);
    func.add_inst(Inst::BraUni {
        target: ctx.return_label,
    });
    func.add_inst(Inst::Label(legacy));
    // Both mutually exclusive paths use one static allocation. The baseline
    // accesses its first128 entries and retains precisely its old mapping.
    ctx.column_reduction_shared_bytes = Some(tile.shared_bytes);
    lower_cooperative_reduction_region(func, ctx, region, 128, true);
}

fn lower_columns<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    region: &crate::tile::ReductionRegion,
    tile: crate::tile::ColumnReductionTile,
) {
    let q = tile.columns_per_block as u64;
    let threads = u64::from(tile.threads);
    let global_tid = global_linear_tid(func);
    let block = func.add_u64_register();
    func.add_inst(Inst::div_u64(
        block.clone(),
        global_tid.clone(),
        Operand::imm_u64(threads),
    ));
    let consumed = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        consumed.clone(),
        block.clone(),
        Operand::imm_u64(threads),
    ));
    let local = func.add_u64_register();
    func.add_inst(Inst::sub_u64(local.clone(), global_tid, consumed));
    let row_lane = func.add_u64_register();
    func.add_inst(Inst::div_u64(
        row_lane.clone(),
        local.clone(),
        Operand::imm_u64(q),
    ));
    let row_start = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        row_start.clone(),
        row_lane.clone(),
        Operand::imm_u64(q),
    ));
    let column_lane = func.add_u64_register();
    func.add_inst(Inst::sub_u64(column_lane.clone(), local.clone(), row_start));
    let tiles = tile.columns.div_ceil(tile.columns_per_block) as u64;
    let batch = func.add_u64_register();
    func.add_inst(Inst::div_u64(
        batch.clone(),
        block.clone(),
        Operand::imm_u64(tiles),
    ));
    let used_tiles = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        used_tiles.clone(),
        batch.clone(),
        Operand::imm_u64(tiles),
    ));
    let column_tile = func.add_u64_register();
    func.add_inst(Inst::sub_u64(column_tile.clone(), block, used_tiles));
    let column_base = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        column_base.clone(),
        column_tile,
        Operand::imm_u64(q),
    ));
    let column = func.add_u64_register();
    func.add_inst(Inst::add_u64(column.clone(), column_base, column_lane));
    let invalid = func.add_predicate_register();
    func.add_inst(Inst::setp_ge_u64(
        invalid.clone(),
        column.clone(),
        Operand::imm_u64(tile.columns as u64),
    ));
    let fiber_base = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        fiber_base.clone(),
        batch.clone(),
        Operand::imm_u64(tile.columns as u64),
    ));
    let fiber = func.add_u64_register();
    func.add_inst(Inst::add_u64(fiber.clone(), fiber_base, column.clone()));
    let batch_base = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        batch_base.clone(),
        batch,
        Operand::imm_u64((tile.rows * tile.columns) as u64),
    ));
    let base = func.add_u64_register();
    func.add_inst(Inst::add_u64(base.clone(), batch_base, column));
    let accumulator = func.add_f32_register();
    func.add_inst(Inst::mov_f32(accumulator.clone(), Operand::imm_f32(0.0)));
    let counter = func.add_u64_register();
    func.add_inst(Inst::mov_u64(counter.clone(), row_lane.clone()));
    let start = next_label(ctx, "column_reduction_loop");
    let end = next_label(ctx, "column_reduction_end");
    // Tail columns skip memory and arithmetic, but join ALL subsequent barriers.
    func.add_inst(Inst::Bra {
        condition: invalid.clone(),
        target: end,
    });
    func.add_inst(Inst::Label(start));
    let done = func.add_predicate_register();
    func.add_inst(Inst::setp_ge_u64(
        done.clone(),
        counter.clone(),
        Operand::imm_u64(tile.rows as u64),
    ));
    func.add_inst(Inst::Bra {
        condition: done,
        target: end,
    });
    let contribution = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        contribution.clone(),
        counter.clone(),
        Operand::imm_u64(tile.columns as u64),
    ));
    let offset = func.add_u64_register();
    func.add_inst(Inst::add_u64(offset.clone(), base, contribution));
    let mut values = HashMap::new();
    for (index, input) in region.inputs.iter().enumerate() {
        if input.domain == crate::tile::ReductionInputDomain::Full {
            let value = load_param_f32_at(func, ctx, &format!("input_{index}"), offset.clone());
            values.insert(input.value, value);
        }
    }
    for operation in &region.producer_operations {
        emit_region_op(func, operation, &mut values);
    }
    // Keep producer instructions adjacent to the same accumulation Add. A JIT
    // may contract Mul+Add exactly as it does in the existing generic lowering.
    func.add_inst(Inst::add_f32(
        accumulator.clone(),
        accumulator.clone(),
        values[&region.reduction_input].clone(),
    ));
    func.add_inst(Inst::add_u64(
        counter.clone(),
        counter,
        Operand::imm_u64(128),
    ));
    func.add_inst(Inst::BraUni { target: start });
    func.add_inst(Inst::Label(end));
    let shared_base = func.add_u64_register();
    func.add_inst(Inst::mov_u64(
        shared_base.clone(),
        Operand::symbol("reduction_partials"),
    ));
    let address = shared_address_for_lane(func, shared_base.clone(), local.clone());
    func.add_inst(Inst::StSharedF32 {
        addr: address,
        src: vec![accumulator.clone()],
        vec: super::super::instructions::VecWidth::Scalar,
    });
    func.add_inst(Inst::BarSync { barrier_id: 0 });
    for stride in [64u64, 32, 16, 8, 4, 2, 1] {
        let skip = next_label(ctx, "column_reduction_tree_skip");
        let inactive = func.add_predicate_register();
        func.add_inst(Inst::setp_ge_u64(
            inactive.clone(),
            row_lane.clone(),
            Operand::imm_u64(stride),
        ));
        func.add_inst(Inst::Bra {
            condition: inactive,
            target: skip,
        });
        let peer = func.add_u64_register();
        func.add_inst(Inst::add_u64(
            peer.clone(),
            local.clone(),
            Operand::imm_u64(stride * q),
        ));
        let peer_address = shared_address_for_lane(func, shared_base.clone(), peer);
        let peer_value = func.add_f32_register();
        func.add_inst(Inst::LdSharedF32 {
            dst: vec![peer_value.clone()],
            addr: peer_address,
            vec: super::super::instructions::VecWidth::Scalar,
        });
        func.add_inst(Inst::add_f32(
            accumulator.clone(),
            accumulator.clone(),
            peer_value,
        ));
        let address = shared_address_for_lane(func, shared_base.clone(), local.clone());
        func.add_inst(Inst::StSharedF32 {
            addr: address,
            src: vec![accumulator.clone()],
            vec: super::super::instructions::VecWidth::Scalar,
        });
        func.add_inst(Inst::Label(skip));
        func.add_inst(Inst::BarSync { barrier_id: 0 });
    }
    // No barriers remain. Only a valid column's contributor0 evaluates the
    // original pure reduced epilogue and stores its unchanged typed result.
    let not_leader = func.add_predicate_register();
    func.add_inst(Inst::setp_ne_u64(
        not_leader.clone(),
        row_lane,
        Operand::imm_u64(0),
    ));
    func.add_inst(Inst::Bra {
        condition: not_leader,
        target: ctx.return_label,
    });
    func.add_inst(Inst::Bra {
        condition: invalid,
        target: ctx.return_label,
    });
    values.clear();
    values.insert(region.reduced_value, accumulator);
    for (index, input) in region.inputs.iter().enumerate() {
        if input.domain == crate::tile::ReductionInputDomain::Reduced {
            let value = load_param_f32_at(func, ctx, &format!("input_{index}"), fiber.clone());
            values.insert(input.value, value);
        }
    }
    for operation in &region.epilogue_operations {
        emit_region_op(func, operation, &mut values);
    }
    for (index, output) in region.outputs.iter().enumerate() {
        store_param_f32_at(
            func,
            ctx,
            &format!("output_{index}"),
            fiber.clone(),
            values[&output.value].clone(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{graph::TensorGraph, tensor::TensorExpr, tile::ColumnReductionPolicy as Policy};

    fn module<D: crate::tile::TileDType, G>(
        graph: &TensorGraph<D, G>,
        policy: Policy,
    ) -> anyhow::Result<(String, super::super::super::plan::PtxExecutionPlan)> {
        let mut plan = crate::ptx::PtxExecutionPlan::build_with_reduction_mode(
            graph,
            crate::ptx::PtxReductionMode::DeterministicTree,
        )?;
        plan.set_column_reduction_policy(policy, 49152);
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.add_reduction_regions(plan.reduction_regions())?;
        tiles.optimize_indices()?;
        Ok((PtxGraph::from(tiles).module_source(), plan))
    }

    #[test]
    fn column_codegen_keeps_uniform_fallback_scratch_tree_and_ssa() -> anyhow::Result<()> {
        let x = TensorExpr::<f32>::input("x", vec![2, 257, 9]);
        let y = TensorExpr::<f32>::input("y", vec![2, 257, 9]);
        let graph: TensorGraph<f32> = (x * y).reduce_sum(1).into();
        let (baseline, old_plan) = module(&graph, Policy::Off)?;
        for (policy, q) in [(Policy::Four, 4), (Policy::Eight, 8)] {
            let (source, mut plan) = module(&graph, policy)?;
            assert_eq!(
                plan.column_reduction_counts(),
                if q == 4 { (1, 0) } else { (0, 1) }
            );
            assert!(source.contains(&format!("reduction_partials[{}]", 128 * q * 4)));
            assert_eq!(
                source
                    .matches(".shared .align 32 .b8 reduction_partials[")
                    .count(),
                1
            );
            assert!(source.contains("column_reduction_legacy"));
            assert!(source.contains("%ntid.x"));
            assert!(source.contains("column_reduction_loop"));
            assert_eq!(source.matches("bar.sync 0;").count(), 16);
            assert_eq!(
                source.matches("mul.f32").count(),
                2 * baseline.matches("mul.f32").count()
            );
            let a = &old_plan.reduction_regions()[0];
            let b = &plan.reduction_regions()[0];
            assert_eq!(a.producer_operations, b.producer_operations);
            assert_eq!(a.epilogue_operations, b.epilogue_operations);
            assert_eq!(a.inputs, b.inputs);
            assert_eq!(a.outputs, b.outputs);
            // Simulated compiled-resource rejection retains graph bindings and
            // selects the emitted original128-thread body, without claiming q.
            plan.retain_column_reductions(|_, _| Ok(false))?;
            assert_eq!(plan.column_reduction_counts(), (0, 0));
        }
        Ok(())
    }

    #[test]
    fn bf16_column_codegen_retains_typed_widen_and_final_rn() -> anyhow::Result<()> {
        let graph: TensorGraph<half::bf16> =
            TensorExpr::input("x", vec![513, 9]).reduce_sum(0).into();
        for policy in [Policy::Four, Policy::Eight] {
            let (source, plan) = module(&graph, policy)?;
            assert_ne!(plan.column_reduction_counts(), (0, 0));
            assert_eq!(
                source.matches("cvt.rn.bf16.f32").count(),
                2,
                "one final conversion per mutually exclusive body"
            );
            assert_eq!(source.matches("cvt.f32.bf16").count(), 2);
            assert!(!source.contains("shfl.sync"));
        }
        Ok(())
    }
}
