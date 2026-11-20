//! UNDECLARED PTX proposal for Stmt::WarpBroadcastLeader. The existing
//! ShflSyncIdxB32 formatter already emits lane clamp0x1f/member mask0xffffffff.
//! This helper deliberately performs no f32 arithmetic, preserving signed zero
//! and arbitrary NaN payload bits across the broadcast itself.
use super::*;

pub(super) fn lower<'a>(func:&mut Function<'a>,ctx:&mut LoweringContext<'a>,dest:TileVar,src:TileVar) {
    let src=ctx.get_or_alloc_reg(func,src);
    let dest=ctx.get_or_alloc_reg(func,dest);
    let bits=func.add_b32_register();
    func.add_inst(Inst::mov_b32(bits.clone(),src));
    let result=func.add_b32_register();
    func.add_inst(Inst::ShflSyncIdxB32 {dst:result.clone(),src:bits,lane:0});
    func.add_inst(Inst::mov_f32_b32(dest,result));
}
