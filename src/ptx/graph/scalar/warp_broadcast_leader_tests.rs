//! Compatibility fixture copied before adding leader-broadcast lowering.
//! Keep the preexisting reduction lowering fixed here: canonical down reductions
//! and butterfly all-reductions must generate byte-identical PTX.
use super::*;
use crate::tile::TileVar;
fn legacy_reduce<'a>(func: &mut Function<'a>, ctx: &mut LoweringContext<'a>, stmt: &Stmt) {
    match stmt {
        Stmt::WarpReduce {
            dest,
            src,
            op,
            broadcast,
        } => {
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
                let peer = func.add_f32_register();
                if *broadcast {
                    // A full-warp butterfly visits every lane without edge
                    // predicates and leaves the result in all lanes.
                    func.add_inst(Inst::ShflSyncBflyB32 {
                        dst: shuffled.clone(),
                        src: bits,
                        mask: distance,
                    });
                    func.add_inst(Inst::mov_f32_b32(peer.clone(), shuffled));
                } else {
                    let valid = func.add_predicate_register();
                    func.add_inst(Inst::shfl_sync_down_b32(
                        shuffled.clone(),
                        valid.clone(),
                        bits,
                        distance,
                    ));
                    func.add_inst(Inst::mov_f32_b32(peer.clone(), shuffled));
                    func.add_inst(Inst::selp_f32(
                        peer.clone(),
                        peer.clone(),
                        Operand::imm_f32(identity),
                        valid,
                    ));
                }
                if *op == ReduceOp::Max {
                    func.add_inst(Inst::max_f32(dest.clone(), dest.clone(), peer));
                } else {
                    func.add_inst(Inst::add_f32(dest.clone(), dest.clone(), peer));
                }
            }
            if *op == ReduceOp::Mean {
                func.add_inst(Inst::div_f32(
                    dest.clone(),
                    dest.clone(),
                    Operand::imm_f32(32.0),
                ));
            }
        }
        _ => panic!("only legacy WarpReduce fixture"),
    }
}
fn legacy_source(reference: bool, op: ReduceOp, broadcast: bool) -> String {
    let arena = bumpalo::Bump::new();
    let mut func = Function::new("legacy_warp_reduce", &arena);
    let mut ctx = LoweringContext::new(&arena);
    let src = ctx.get_or_alloc_reg(&mut func, TileVar(0));
    func.add_inst(Inst::mov_f32(src, Operand::imm_f32(-0.)));
    let stmt = Stmt::WarpReduce {
        dest: TileVar(1),
        src: TileVar(0),
        op,
        broadcast,
    };
    if reference {
        legacy_reduce(&mut func, &mut ctx, &stmt);
    } else {
        lower_scalar_stmt(&mut func, &mut ctx, &stmt);
    }
    func.add_inst(Inst::Ret);
    let mut module = crate::ptx::Module::new_with_target((8, 9));
    module.add_function(func);
    module.to_string()
}
#[test]
fn warp_broadcast_leader_keeps_all_legacy_reduce_ptx_byte_identical() {
    for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max] {
        for broadcast in [false, true] {
            let old = legacy_source(true, op, broadcast);
            let current = legacy_source(false, op, broadcast);
            assert_eq!(old, current);
            assert!(!current.contains("shfl.sync.idx"));
            assert_eq!(
                current
                    .matches(if broadcast {
                        "shfl.sync.bfly.b32"
                    } else {
                        "shfl.sync.down.b32"
                    })
                    .count(),
                5
            );
        }
    }
}
