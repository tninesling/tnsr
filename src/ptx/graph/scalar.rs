//! Backend lowering for general scalar arithmetic, memory, predicates, and folds.
//! No normalization or attention semantics are present in this module.
use super::*;
use crate::ptx::types::Pred;
use crate::tensor::{BinaryOp, ReduceOp, UnaryOp};
use crate::tile::{ScalarBinaryOp, ScalarExpr, ScalarMemory, ScalarPredicate};

pub(super) fn lower_scalar_stmt<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    stmt: &Stmt,
) {
    match stmt {
        Stmt::SetScalar { dest, value } => {
            let value = expression(func, ctx, value);
            let dest = ctx.get_or_alloc_reg(func, *dest);
            func.add_inst(Inst::mov_f32(dest, value));
        }
        Stmt::LoadScalar {
            dest,
            source,
            index,
        } => {
            let offset = lower_expr(func, ctx, index);
            let dest = ctx.get_or_alloc_reg(func, *dest);
            match source {
                ScalarMemory::Global(param) => {
                    let value = load_param_f32_at(func, ctx, param, offset);
                    func.add_inst(Inst::mov_f32(dest, value));
                }
                ScalarMemory::Shared(tile) => {
                    let bytes = multiply_u64(func, offset, 4);
                    let addr = func.add_u64_register();
                    func.add_inst(Inst::add_u64(
                        addr.clone(),
                        ctx.shared_mem_ptrs[tile].clone(),
                        bytes,
                    ));
                    load_shared_as_f32(func, DType::F32, dest, addr);
                }
            }
        }
        Stmt::StoreScalar {
            target,
            index,
            value,
        } => {
            let offset = lower_expr(func, ctx, index);
            let value = ctx.get_or_alloc_reg(func, *value);
            match target {
                ScalarMemory::Global(param) => store_param_f32_at(func, ctx, param, offset, value),
                ScalarMemory::Shared(tile) => {
                    let bytes = multiply_u64(func, offset, 4);
                    let addr = func.add_u64_register();
                    func.add_inst(Inst::add_u64(
                        addr.clone(),
                        ctx.shared_mem_ptrs[tile].clone(),
                        bytes,
                    ));
                    store_shared_from_f32(func, DType::F32, addr, value);
                }
            }
        }
        Stmt::If { condition, body } => {
            let pred = predicate(func, ctx, condition);
            let yes = next_label(ctx, "scalar_if_body");
            let end = next_label(ctx, "scalar_if_end");
            func.add_inst(Inst::Bra {
                condition: pred,
                target: yes,
            });
            func.add_inst(Inst::BraUni { target: end });
            func.add_inst(Inst::Label(yes));
            lower_block(func, ctx, body);
            func.add_inst(Inst::Label(end));
        }
        Stmt::ScalarLoop {
            loop_var,
            start,
            end,
            carries,
            body,
        } => {
            for carry in carries {
                let initial = expression(func, ctx, &carry.initial);
                let dest = ctx.get_or_alloc_reg(func, carry.var);
                func.add_inst(Inst::mov_f32(dest, initial));
            }
            let counter = func.add_u64_register();
            let initial = lower_expr(func, ctx, start);
            func.add_inst(Inst::mov_u64(counter.clone(), initial));
            let outer = ctx.index_vars.insert(loop_var.clone(), counter.clone());
            let begin = next_label(ctx, &format!("{loop_var}_loop"));
            let done = next_label(ctx, &format!("{loop_var}_end"));
            func.add_inst(Inst::Label(begin));
            let limit = lower_expr(func, ctx, end);
            let finished = func.add_predicate_register();
            func.add_inst(Inst::setp_ge_u64(finished.clone(), counter.clone(), limit));
            func.add_inst(Inst::Bra {
                condition: finished,
                target: done,
            });
            lower_block(func, ctx, body);
            func.add_inst(Inst::add_u64(counter.clone(), counter, Operand::imm_u64(1)));
            func.add_inst(Inst::BraUni { target: begin });
            func.add_inst(Inst::Label(done));
            if let Some(value) = outer {
                ctx.index_vars.insert(loop_var.clone(), value);
            } else {
                ctx.index_vars.remove(loop_var);
            }
        }
        Stmt::WarpReduce { dest, src, op } => {
            let src = ctx.get_or_alloc_reg(func, *src);
            let dest = ctx.get_or_alloc_reg(func, *dest);
            func.add_inst(Inst::mov_f32(dest.clone(), src));
            let identity = if *op == ReduceOp::Max {
                f32::NEG_INFINITY
            } else {
                0.0
            };
            for distance in [16, 8, 4, 2, 1] {
                let bits = func.add_b32_register();
                func.add_inst(Inst::mov_b32(bits.clone(), dest.clone()));
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
                    Operand::imm_f32(identity),
                    valid,
                ));
                if *op == ReduceOp::Max {
                    func.add_inst(Inst::max_f32(dest.clone(), dest.clone(), peer));
                } else {
                    func.add_inst(Inst::add_f32(dest.clone(), dest.clone(), peer));
                }
            }
            if *op == ReduceOp::Mean {
                func.add_inst(Inst::div_f32(dest.clone(), dest, Operand::imm_f32(32.0)));
            }
        }
        _ => unreachable!("scalar dispatch only accepts scalar statements"),
    }
}

fn expression<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    value: &ScalarExpr,
) -> Operand<'a, F32> {
    match value {
        ScalarExpr::Constant(value) => Operand::imm_f32(*value),
        ScalarExpr::Var(var) => ctx.get_or_alloc_reg(func, *var),
        ScalarExpr::Unary { op, value } => {
            let value = expression(func, ctx, value);
            let dest = func.add_f32_register();
            match op {
                UnaryOp::Neg => func.add_inst(Inst::neg_f32(dest.clone(), value)),
                UnaryOp::Relu => {
                    func.add_inst(Inst::max_f32(dest.clone(), value, Operand::imm_f32(0.0)))
                }
                UnaryOp::Exp => {
                    let scaled = func.add_f32_register();
                    func.add_inst(Inst::mul_f32(
                        scaled.clone(),
                        value,
                        Operand::imm_f32(std::f32::consts::LOG2_E),
                    ));
                    func.add_inst(Inst::ex2_f32(dest.clone(), scaled));
                }
                UnaryOp::Log => {
                    let logarithm = func.add_f32_register();
                    func.add_inst(Inst::lg2_f32(logarithm.clone(), value));
                    func.add_inst(Inst::mul_f32(
                        dest.clone(),
                        logarithm,
                        Operand::imm_f32(std::f32::consts::LN_2),
                    ));
                }
            }
            dest
        }
        ScalarExpr::Binary { op, lhs, rhs } => {
            let lhs = expression(func, ctx, lhs);
            let rhs = expression(func, ctx, rhs);
            let dest = func.add_f32_register();
            let inst = match op {
                ScalarBinaryOp::Arithmetic(BinaryOp::Add) => Inst::add_f32(dest.clone(), lhs, rhs),
                ScalarBinaryOp::Arithmetic(BinaryOp::Sub) => Inst::sub_f32(dest.clone(), lhs, rhs),
                ScalarBinaryOp::Arithmetic(BinaryOp::Mul) => Inst::mul_f32(dest.clone(), lhs, rhs),
                ScalarBinaryOp::Arithmetic(BinaryOp::Div) => Inst::div_f32(dest.clone(), lhs, rhs),
                ScalarBinaryOp::Max => Inst::max_f32(dest.clone(), lhs, rhs),
            };
            func.add_inst(inst);
            dest
        }
        ScalarExpr::Select {
            condition,
            then_value,
            else_value,
        } => {
            let condition = predicate(func, ctx, condition);
            let a = expression(func, ctx, then_value);
            let b = expression(func, ctx, else_value);
            let dest = func.add_f32_register();
            func.add_inst(Inst::selp_f32(dest.clone(), a, b, condition));
            dest
        }
    }
}
fn predicate<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    condition: &ScalarPredicate,
) -> Operand<'a, Pred> {
    let pred = func.add_predicate_register();
    match condition {
        ScalarPredicate::IndexLt { lhs, rhs } => {
            let lhs = lower_expr(func, ctx, lhs);
            let rhs = lower_expr(func, ctx, rhs);
            func.add_inst(Inst::setp_lt_u64(pred.clone(), lhs, rhs));
        }
        ScalarPredicate::Equal { lhs, rhs } => {
            let lhs = expression(func, ctx, lhs);
            let rhs = expression(func, ctx, rhs);
            func.add_inst(Inst::setp_eq_f32(pred.clone(), lhs, rhs));
        }
    }
    pred
}
