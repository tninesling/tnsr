//! Documented nontransposed x4 providers, NOT opaque WMMA fragments.
use super::*;
fn load<'a>(
    func: &mut Function<'a>,
    ctx: &LoweringContext<'a>,
    tile: TileVar,
    lhs: bool,
    fragment: usize,
    k: usize,
) -> Vec<Operand<'a, B32>> {
    let (row, col) = crate::ptx::graph::bf16_ldmatrix_provider_expr(lhs, fragment, k);
    let addr = shared_coordinate_address(func, ctx, tile, &row, &col);
    let frags = (0..4).map(|_| func.add_b32_register()).collect::<Vec<_>>();
    func.add_inst(Inst::LdMatrixX4B16 {
        frags: frags.clone(),
        addr,
    });
    frags
}
pub(super) fn lower<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    dest: TileVar,
    a: TileVar,
    b: TileVar,
    schedule: &crate::tile::MatMulSchedule,
) {
    let accum = ctx.fragment_regs[&dest].clone();
    let a = ctx.reg_to_shared.get(&a).copied().unwrap_or(a);
    let b = ctx.reg_to_shared.get(&b).copied().unwrap_or(b);
    for k in (0..schedule.block_tile.k).step_by(16) {
        let left = (0..2)
            .map(|m| load(func, ctx, a, true, m, k))
            .collect::<Vec<_>>();
        for n in 0..4 {
            let right = load(func, ctx, b, false, n, k);
            for half in 0..2 {
                for (m, lhs) in left.iter().enumerate() {
                    let offset = 8 * (m * 4 + n) + 4 * half;
                    let c = accum[offset..offset + 4].to_vec();
                    func.add_inst(Inst::MmaBf16M16N8K16 {
                        d_frags: c.clone(),
                        a_frags: lhs.clone(),
                        b_frags: right[2 * half..2 * half + 2].to_vec(),
                        c_frags: c,
                    });
                }
            }
        }
    }
}
