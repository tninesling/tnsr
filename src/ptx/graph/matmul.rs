//! Register-tiled Tensor Core lowering with operand-fragment reuse.
use super::*;

pub(super) fn lower_tiled_tensor_core_matmul<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    dest: TileVar,
    a: TileVar,
    b: TileVar,
    schedule: &crate::tile::MatMulSchedule,
) {
    let accumulators = ctx.fragment_regs[&dest].clone();
    let a_shared = ctx.reg_to_shared.get(&a).copied().unwrap_or(a);
    let b_shared = ctx.reg_to_shared.get(&b).copied().unwrap_or(b);
    // A warp can reuse each operand fragment across multiple output fragments.
    // Every block thread still participates in staging and surrounding barriers.
    let done =
        bumpalo::format!(in ctx.arena, "tf32_matmul_done_{}", ctx.label_counter).into_bump_str();
    ctx.label_counter += 1;
    skip_noncomputing_warps(
        func,
        schedule.block_threads.0,
        schedule.warp_topology.0 * schedule.warp_topology.1,
        done,
    );
    let (warp_row, warp_col) = warp_coordinates(schedule.block_threads.0, schedule.warp_topology);
    let (fm_count, fn_count) = schedule.warp_fragments;
    for k_offset in (0..schedule.block_tile.k).step_by(schedule.instruction_tile.k) {
        let mut left = Vec::new();
        let mut right = Vec::new();
        for (is_left, count, shared) in [(true, fm_count, a_shared), (false, fn_count, b_shared)] {
            for fragment in 0..count {
                let (row, col) = if is_left {
                    (
                        warp_row.clone() * fm_count + Expr::Const((fragment * 16) as i64),
                        Expr::Const(k_offset as i64),
                    )
                } else {
                    (
                        Expr::Const(k_offset as i64),
                        warp_col.clone() * fn_count + Expr::Const((fragment * 16) as i64),
                    )
                };
                let address = shared_coordinate_address(func, ctx, shared, &row, &col);
                let fragments: Vec<Operand<'a, B32>> = (0
                    ..if schedule.operand_dtype == DType::F16 {
                        8
                    } else {
                        4
                    })
                    .map(|_| func.add_b32_register())
                    .collect();
                let stride = Operand::imm_i32(shared_row_stride(ctx, shared) as i32);
                func.add_inst(if is_left {
                    Inst::WmmaLoadA {
                        column_major: shared_layout(ctx, shared).column_major,
                        dtype: schedule.operand_dtype,
                        frags: fragments.clone(),
                        addr: address,
                        stride,
                    }
                } else {
                    Inst::WmmaLoadB {
                        column_major: shared_layout(ctx, shared).column_major,
                        dtype: schedule.operand_dtype,
                        frags: fragments.clone(),
                        addr: address,
                        stride,
                    }
                });
                if schedule.pipeline_stages == 2
                    && schedule.operand_dtype == DType::TF32
                    && schedule.plan != crate::tile::MatMulPlan::TensorCoreTf32x3
                {
                    // Async copies retain f32 storage bits until fragment loading.
                    for fragment in &fragments {
                        func.add_inst(Inst::ConvertTf32Bits {
                            dst: fragment.clone(),
                            src: fragment.clone(),
                        });
                    }
                }
                if is_left {
                    left.push(fragments);
                } else {
                    right.push(fragments);
                }
            }
        }
        for (fm, a_fragments) in left.iter().enumerate() {
            for (fn_, b_fragments) in right.iter().enumerate() {
                let offset = 8 * (fm * fn_count + fn_);
                let accumulators = accumulators[offset..offset + 8].to_vec();
                if schedule.plan == crate::tile::MatMulPlan::TensorCoreTf32x3 {
                    let (a_hi, a_lo) = split_tf32_fragments(func, a_fragments);
                    let (b_hi, b_lo) = split_tf32_fragments(func, b_fragments);
                    // Preserve the original cross-product accumulation order.
                    for (left, right) in [(a_lo, b_hi.clone()), (a_hi.clone(), b_lo), (a_hi, b_hi)]
                    {
                        func.add_inst(Inst::WmmaMma {
                            column_major: (
                                schedule.operand_layouts.0.column_major,
                                schedule.operand_layouts.1.column_major,
                            ),
                            dtype: DType::TF32,
                            d_frags: accumulators.clone(),
                            a_frags: left,
                            b_frags: right,
                            c_frags: accumulators.clone(),
                        });
                    }
                } else {
                    func.add_inst(Inst::WmmaMma {
                        column_major: (
                            schedule.operand_layouts.0.column_major,
                            schedule.operand_layouts.1.column_major,
                        ),
                        dtype: schedule.operand_dtype,
                        d_frags: accumulators.clone(),
                        a_frags: a_fragments.clone(),
                        b_frags: b_fragments.clone(),
                        c_frags: accumulators,
                    });
                }
            }
        }
    }
    func.add_inst(Inst::Label(done));
}

/// Four independent increasing-K sums per thread, with a shared RHS load.
/// The surrounding IR owns staging, barriers, tail predicates and epilogues.
pub(super) fn lower_register_tiled_scalar_matmul<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    dest: TileVar,
    a: TileVar,
    b: TileVar,
    schedule: &crate::tile::MatMulSchedule,
) {
    let accumulators = ctx.fragment_regs[&dest].clone();
    let a = ctx.reg_to_shared.get(&a).copied().unwrap_or(a);
    let b = ctx.reg_to_shared.get(&b).copied().unwrap_or(b);
    let linear =
        Expr::ThreadIdx(Dim::Y) * schedule.block_threads.0 as usize + Expr::ThreadIdx(Dim::X);
    let row = lower_expr(
        func,
        ctx,
        &Expr::FloorDiv(Box::new(linear.clone()), schedule.block_tile.n),
    );
    let col = lower_expr(
        func,
        ctx,
        &Expr::Mod(Box::new(linear), schedule.block_tile.n),
    );
    let row_step = schedule.thread_count() / schedule.block_tile.n;
    let mut rows = Vec::new();
    let mut a_bases = Vec::new();
    for index in 0..accumulators.len() {
        let coordinate = func.add_u64_register();
        func.add_inst(Inst::add_u64(
            coordinate.clone(),
            row.clone(),
            Operand::imm_u64((index * row_step) as u64),
        ));
        let base = shared_address(func, ctx, a, coordinate.clone(), Operand::imm_u64(0));
        rows.push(coordinate);
        a_bases.push(base);
    }
    let b_base = shared_address(func, ctx, b, Operand::imm_u64(0), col.clone());
    let a_layout = shared_layout(ctx, a);
    let b_layout = shared_layout(ctx, b);
    let reduction = begin_counted_loop(func, ctx, "register_matmul", schedule.block_tile.k);
    let k = reduction.counter.clone();
    let b_address = if b_layout.xor_mask == 0 && !b_layout.column_major {
        let offset = multiply_u64(func, k.clone(), b_layout.row_stride * 4);
        let address = func.add_u64_register();
        func.add_inst(Inst::add_u64(address.clone(), b_base, offset));
        address
    } else {
        shared_address(func, ctx, b, k.clone(), col)
    };
    let b_value = func.add_f32_register();
    load_shared_as_f32(func, DType::F32, b_value.clone(), b_address);
    let a_offset = multiply_u64(func, k.clone(), 4);
    let a_value = func.add_f32_register();
    let product = func.add_f32_register();
    for (index, accumulator) in accumulators.into_iter().enumerate() {
        let address = if a_layout.xor_mask == 0 && !a_layout.column_major {
            let address = func.add_u64_register();
            func.add_inst(Inst::add_u64(
                address.clone(),
                a_bases[index].clone(),
                a_offset.clone(),
            ));
            address
        } else {
            shared_address(func, ctx, a, rows[index].clone(), k.clone())
        };
        load_shared_as_f32(func, DType::F32, a_value.clone(), address);
        // Keep the original multiply followed by add for each output. No
        // reassociation, split-K or precision conversion is introduced.
        func.add_inst(Inst::mul_f32(
            product.clone(),
            a_value.clone(),
            b_value.clone(),
        ));
        func.add_inst(Inst::add_f32(
            accumulator.clone(),
            accumulator,
            product.clone(),
        ));
    }
    end_counted_loop(func, reduction);
}
