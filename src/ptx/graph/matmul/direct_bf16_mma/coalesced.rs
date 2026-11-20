//! Bits-only 8x8 register transpose after scalar BF16 conversion.
use super::*;
use crate::ptx::instructions::SelpInst;
fn select<'a>(
    func: &mut Function<'a>,
    dst: Operand<'a, B32>,
    a: Operand<'a, B32>,
    b: Operand<'a, B32>,
    p: Operand<'a, crate::ptx::types::Pred>,
) {
    func.add_inst(Inst::SelpB32(SelpInst::new(dst, a, b, p)));
}
pub(in crate::ptx::graph) fn lower_store<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    src: TileVar,
    output: &str,
    bias: Option<&str>,
    n: usize,
) {
    let normalized = ctx.tile_layouts[&src].normalizes_bf16_mma_addresses();
    let accum = ctx.fragment_regs[&src].clone();
    let lane = Expr::ThreadIdx(Dim::X);
    let warp = Expr::ThreadIdx(Dim::Y);
    let row_base =
        Expr::BlockIdx(Dim::Y) * 128usize + Expr::FloorDiv(Box::new(warp.clone()), 2) * 32usize;
    let col_base = Expr::BlockIdx(Dim::X) * 128usize + Expr::Mod(Box::new(warp), 2) * 64usize;
    let predicates = (0..3)
        .map(|bit| {
            let flag = lower_expr(
                func,
                ctx,
                &Expr::BitAnd(Box::new(lane.clone()), 4u64 << bit),
            );
            let p = func.add_predicate_register();
            func.add_inst(Inst::setp_ge_u64(p.clone(), flag, Operand::imm_u64(1)));
            p
        })
        .collect::<Vec<_>>();
    for rb in 0..4 {
        let mut packed = Vec::new();
        for j in 0..8 {
            let mut halves = Vec::new();
            for parity in 0..2 {
                let index = 8 * ((rb / 2) * 4 + j / 2) + 4 * (j % 2) + 2 * (rb % 2) + parity;
                let mut value = accum[index].clone();
                if let Some(bias) = bias {
                    let col = col_base.clone()
                        + constant(j * 8 + parity)
                        + Expr::Mod(Box::new(lane.clone()), 4) * 2usize;
                    let col = if normalized {
                        crate::ptx::graph::normalized_mma_backend_expr(&col)
                    } else {
                        col
                    };
                    let offset = lower_expr(func, ctx, &col);
                    let address =
                        element_address(func, ctx.param_ptrs[bias].clone(), offset, DType::BF16);
                    let b = func.add_f32_register();
                    load_global_as_f32(func, DType::BF16, b.clone(), address);
                    let result = func.add_f32_register();
                    func.add_inst(Inst::add_rn_f32(result.clone(), value, b));
                    value = result;
                }
                let rounded = func.add_bf16_register();
                func.add_inst(Inst::convert_bf16_f32(rounded.clone(), value));
                halves.push(rounded);
            }
            let word = func.add_b32_register();
            func.add_inst(Inst::PackBf16Pair {
                dst: word.clone(),
                lo: halves[0].clone(),
                hi: halves[1].clone(),
            });
            packed.push(word);
        }
        for (mask, j0, j1) in crate::ptx::graph::explicit_bf16_mma_transpose_pairs() {
            let bit = usize::try_from(mask.trailing_zeros() - 2).expect("three stage masks");
            let p = predicates[bit].clone();
            let selected = func.add_b32_register();
            select(
                func,
                selected.clone(),
                packed[j0].clone(),
                packed[j1].clone(),
                p.clone(),
            );
            let peer = func.add_b32_register();
            func.add_inst(Inst::ShflSyncBflyB32 {
                dst: peer.clone(),
                src: selected,
                mask,
            });
            select(
                func,
                packed[j0].clone(),
                peer.clone(),
                packed[j0].clone(),
                p.clone(),
            );
            select(func, packed[j1].clone(), packed[j1].clone(), peer, p);
        }
        for (g, word) in packed.into_iter().enumerate() {
            let row = row_base.clone() + constant(rb * 8 + g);
            let col = col_base.clone() + lane.clone() * 2usize;
            let index = row * n + col;
            let index = if normalized {
                crate::ptx::graph::normalized_mma_backend_expr(&index)
            } else {
                index
            };
            let offset = lower_expr(func, ctx, &index);
            let address =
                element_address(func, ctx.param_ptrs[output].clone(), offset, DType::BF16);
            func.add_inst(Inst::StGlobalB32 {
                addr: address,
                src: word,
            });
        }
    }
}
