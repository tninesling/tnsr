//! Separate test-only NN128x64 diagnostic. The strict128-square remains unchanged.
use super::*;
use crate::tile::{MatMulAccumulatorExchange, MatMulFragmentLoad, MatMulTile};
mod gpu;
mod tests;
const SHARED_RECT: usize = 29696;
fn template(r: &MatMulRegion, target: PtxTarget) -> Result<(TileIR, MatMulRegion)> {
    guard(r)?;
    let mut selected = r.clone();
    selected.schedule.block_tile = MatMulTile {
        m: 128,
        n: 64,
        k: 32,
    };
    selected.schedule.warp_topology = (4, 2);
    selected.schedule.warp_fragments = (2, 2);
    selected.schedule.accumulator_exchange = MatMulAccumulatorExchange::FullTile;
    selected.schedule.fragment_load = MatMulFragmentLoad::PreloadAll;
    selected.configure_pipeline(
        MatMulSharedLayout::Padded,
        MatMulPipeline::Auto,
        target.supports_async_copy(),
        target.matmul_resources,
    )?;
    ensure!(
        selected.schedule.pipeline_stages == 2
            && selected.schedule.async_operands == (true, true)
            && selected.schedule.copy_bytes == 16
            && selected.schedule.reuse_shared_memory,
        "complete rectangular pipeline required"
    );
    ensure!(
        selected.schedule.operand_layouts.0 == crate::tile::SharedLayout::padded(32, 8)?
            && selected.schedule.operand_layouts.1 == crate::tile::SharedLayout::padded(64, 8)?,
        "A40/B72 layout required"
    );
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.add_matmul_regions(std::slice::from_ref(&selected))?;
    let ir = tiles.matmul_region_kernels[0].ir.clone();
    ir.validate_layouts()?;
    Ok((ir, selected))
}
fn candidate(r: &MatMulRegion, target: PtxTarget) -> Result<TileIR> {
    let (mut ir, selected) = template(r, target)?;
    // Preserve the generated copy/sentinel/drain prefix. Only the old full F32
    // output exchange, its dead view, and its arena over-allocation are removed.
    let first = ir
        .body
        .stmts
        .iter()
        .position(|s| matches!(s, Stmt::ConvertLayout { .. }))
        .context("missing original full exchange")?;
    let mut end = first;
    while matches!(
        ir.body.stmts.get(end.wrapping_sub(1)),
        Some(Stmt::ViewSharedArena { .. })
    ) {
        end -= 1;
    }
    ensure!(
        matches!(ir.body.stmts.get(end.wrapping_sub(1)), Some(Stmt::Barrier)),
        "preserve final drain barrier"
    );
    ir.body.stmts.truncate(end);
    let mut accumulator = None;
    fn patch(block: &mut crate::tile::Block) -> Result<()> {
        for stmt in &mut block.stmts {
            match stmt {
                Stmt::MatMul { schedule, .. } => {
                    schedule.fragment_load = MatMulFragmentLoad::StreamB;
                    ensure!(
                        schedule.is_nn128x64_ldmatrix_diagnostic(),
                        "exact rectangular descriptor"
                    );
                }
                Stmt::ForLoop { body, .. } => patch(body)?,
                _ => {}
            }
        }
        Ok(())
    }
    for s in &mut ir.body.stmts {
        match s {
            Stmt::AllocSharedArena { bytes, .. } => {
                ensure!(*bytes == 32768, "original full exchange arena changed");
                *bytes = SHARED_RECT;
            }
            Stmt::AllocTile {
                var,
                space: MemorySpace::Fragment,
                layout,
                ..
            } => {
                ensure!(
                    accumulator.is_none() && *layout == selected.schedule.accumulator_tile_layout(),
                    "one original accumulator"
                );
                accumulator = Some(*var);
                *layout = TileLayout::Bf16MmaAccumulator128x64LdmatrixNn;
            }
            _ => {}
        }
    }
    patch(&mut ir.body)?;
    ir.body.stmts.push(Stmt::StoreBf16MmaCoalesced {
        src: accumulator.context("missing accumulator")?,
        output_param: "output_0".into(),
        bias_param: None,
        m: M,
        n: N,
    });
    ir.kernel_name = "test_nn128x64_ldmatrix_plain_bf16_coalesced".into();
    ir.validate_layouts()?;
    Ok(ir)
}
