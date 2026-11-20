//! Aligned vector scalar memory lowering; admission remains the caller's obligation.
use super::*;
use crate::ptx::instructions::VecWidth;

pub(super) fn lower<'a>(func: &mut Function<'a>, ctx: &mut LoweringContext<'a>, stmt: &Stmt) {
    let (param, group, extent, variables, load) = match stmt {
        Stmt::LoadGlobalF32x4 {
            dests,
            param,
            group_index,
            extent,
        } => (param, group_index, *extent, dests, true),
        Stmt::StoreGlobalF32x4 {
            values,
            param,
            group_index,
            extent,
        } => (param, group_index, *extent, values, false),
        _ => unreachable!("isolated vector memory dispatch"),
    };
    let values = variables
        .iter()
        .map(|v| ctx.get_or_alloc_reg(func, *v))
        .collect::<Vec<_>>();
    if load {
        for value in &values {
            func.add_inst(Inst::mov_f32(value.clone(), Operand::imm_f32(0.0)));
        }
    }
    let group = lower_expr(func, ctx, group);
    let skip =
        bumpalo::format!(in ctx.arena,"test_vector4_skip_{}",ctx.label_counter).into_bump_str();
    ctx.label_counter += 1;
    let outside = func.add_predicate_register();
    func.add_inst(Inst::setp_ge_u64(
        outside.clone(),
        group.clone(),
        Operand::imm_u64((extent / 4) as u64),
    ));
    func.add_inst(Inst::Bra {
        condition: outside,
        target: skip,
    });
    let offset = multiply_u64(func, group, 4);
    let address = element_address(func, ctx.param_ptrs[param].clone(), offset, DType::F32);
    if load {
        func.add_inst(Inst::LdGlobalF32 {
            dst: values,
            addr: address,
            vec: VecWidth::V4,
        });
    } else {
        func.add_inst(Inst::StGlobalF32 {
            src: values,
            addr: address,
            vec: VecWidth::V4,
        });
    }
    func.add_inst(Inst::Label(skip));
}
