//! Documented PTX mma.m16n8k16 BF16 register mapping.
//! No WMMA fragment is consumed here. A/B each load adjacent BF16 pairs.
use super::*;
pub(super) mod coalesced;
pub(super) mod coalesced_nn_rectangular;
mod ldmatrix;
#[cfg(test)]
mod ldmatrix_nn;
pub(super) mod ldmatrix_nn_rectangular;
fn constant(value: usize) -> Expr {
    Expr::Const(i64::try_from(value).expect("bounded fragment coordinate"))
}

pub(super) fn lower<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    dest: TileVar,
    a: TileVar,
    b: TileVar,
    schedule: &crate::tile::MatMulSchedule,
) {
    if ctx.tile_layouts[&dest] == TileLayout::Bf16MmaAccumulator128x64LdmatrixNn {
        ldmatrix_nn_rectangular::lower(func, ctx, dest, a, b, schedule);
        return;
    }
    #[cfg(test)]
    if ctx.tile_layouts[&dest] == TileLayout::Bf16MmaAccumulator128LdmatrixNn {
        ldmatrix_nn::lower(func, ctx, dest, a, b, schedule);
        return;
    }
    if ctx.tile_layouts[&dest] == TileLayout::Bf16MmaAccumulator128Ldmatrix {
        ldmatrix::lower(func, ctx, dest, a, b, schedule);
    } else {
        #[cfg(test)]
        lower_scalar(func, ctx, dest, a, b, schedule);
    }
}

#[cfg(test)]
fn lower_scalar<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    dest: TileVar,
    a: TileVar,
    b: TileVar,
    schedule: &crate::tile::MatMulSchedule,
) {
    let normalized = ctx.tile_layouts[&dest] == TileLayout::Bf16MmaAccumulator128Normalized;
    let accumulators = ctx.fragment_regs[&dest].clone();
    let (wr, wc) = warp_coordinates(schedule.block_threads.0, schedule.warp_topology);
    let lane = Expr::ThreadIdx(Dim::X);
    let group = Expr::FloorDiv(Box::new(lane.clone()), 4);
    let pair = Expr::Mod(Box::new(lane), 4) * 2usize;
    let a = ctx.reg_to_shared.get(&a).copied().unwrap_or(a);
    let b = ctx.reg_to_shared.get(&b).copied().unwrap_or(b);
    for k in (0..schedule.block_tile.k).step_by(16) {
        let left: Vec<_> = (0..2)
            .map(|m| {
                (0..4)
                    .map(|i| {
                        let row =
                            wr.clone() * 2usize + group.clone() + constant(m * 16 + (i % 2) * 8);
                        let col = pair.clone() + constant(k + (i / 2) * 8);
                        let (row, col) = normalized_coordinates(row, col, normalized);
                        let addr = shared_coordinate_address(func, ctx, a, &row, &col);
                        let dst = func.add_b32_register();
                        func.add_inst(Inst::LdSharedB32 {
                            dst: dst.clone(),
                            addr,
                        });
                        dst
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        for n in 0..4 {
            for half in 0..2 {
                let right = (0..2)
                    .map(|i| {
                        let row = pair.clone() + constant(k + i * 8);
                        let col = wc.clone() * 4usize + group.clone() + constant(n * 16 + half * 8);
                        let (row, col) = normalized_coordinates(row, col, normalized);
                        let addr = shared_coordinate_address(func, ctx, b, &row, &col);
                        let dst = func.add_b32_register();
                        func.add_inst(Inst::LdSharedB32 {
                            dst: dst.clone(),
                            addr,
                        });
                        dst
                    })
                    .collect::<Vec<_>>();
                for (m, lhs) in left.iter().enumerate() {
                    let offset = 8 * (m * 4 + n) + 4 * half;
                    let accum = accumulators[offset..offset + 4].to_vec();
                    func.add_inst(Inst::MmaBf16M16N8K16 {
                        d_frags: accum.clone(),
                        a_frags: lhs.clone(),
                        b_frags: right.clone(),
                        c_frags: accum,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
fn normalized_coordinates(row: Expr, col: Expr, normalized: bool) -> (Expr, Expr) {
    if normalized {
        (
            crate::ptx::graph::normalized_mma_backend_expr(&row),
            crate::ptx::graph::normalized_mma_backend_expr(&col),
        )
    } else {
        (row, col)
    }
}
