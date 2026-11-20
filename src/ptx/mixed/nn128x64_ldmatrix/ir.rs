//! Exact qualified NN128x64 generator. Kernel name retained for complete PTX identity.
use super::*;
use crate::tile::{MatMulAccumulatorExchange, MatMulFragmentLoad, MatMulTile};
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
pub(super) fn candidate(r: &MatMulRegion, target: PtxTarget) -> Result<TileIR> {
    let (mut ir, selected) = template(r, target)?;
    // Preserve the generated copy/sentinel/drain prefix. Only the old full F32
    // output exchange, its dead view, and its arena over-allocation are removed.
    let first = ir
        .body
        .stmts
        .iter()
        .position(|s| matches!(s, Stmt::ConvertLayout { .. }))
        .context("missing original full exchange")?;
    let suffix = &ir.body.stmts[first..];
    ensure!(
        suffix.len() == 98 && matches!(suffix[1], Stmt::Barrier),
        "original32-round full exchange suffix changed"
    );
    let Stmt::ConvertLayout {
        dest: exchange,
        coordinates: None,
        ..
    } = suffix[0]
    else {
        anyhow::bail!("original full exchange changed");
    };
    for round in suffix[2..].chunks_exact(3) {
        let Stmt::AllocTile {
            var,
            space: MemorySpace::Register,
            layout: TileLayout::ThreadScalar,
            dtype: DType::F32,
            rows: 1,
            cols: 1,
        } = round[0]
        else {
            anyhow::bail!("original scalar output allocation changed")
        };
        ensure!(
            matches!(&round[1],Stmt::ConvertLayout{dest,src,coordinates:Some(_)}if *dest==var&&*src==exchange),
            "original shared extraction changed"
        );
        ensure!(
            matches!(&round[2],Stmt::StoreGlobalPredicated{dest_param,src,..}if dest_param=="output_0"&&*src==var),
            "original plain output store changed"
        );
    }
    ensure!(
        ir.params.len() == 3
            && ir.params.iter().all(|p| p.dtype == DType::BF16)
            && ir
                .params
                .iter()
                .map(|p| p.name.as_str())
                .eq(["input_0", "input_1", "output_0"]),
        "exact plain3-pointer generated ABI required"
    );
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

fn same_view(a: &VirtualTensor, b: &VirtualTensor) -> Result<bool> {
    Ok(a.source == b.source
        && a.dtype == b.dtype
        && a.shape == b.shape
        && a.predicate.is_none()
        && a.out_of_bounds.is_none()
        && a.access.normalize_in_domain(&a.shape)? == b.access.normalize_in_domain(&b.shape)?)
}
pub(super) fn guard(r: &MatMulRegion) -> Result<()> {
    ensure!(
        (r.m, r.n, r.k) == (M, N, K)
            && r.batch_shape.is_empty()
            && r.inputs.len() == 2
            && r.outputs.len() == 1
            && r.epilogue_operations.is_empty()
            && r.outputs[0].value == r.matmul_value
            && r.output_shape == [M, N]
            && r.lhs_shape == [M, K]
            && r.rhs_shape == [K, N],
        "exact plain NN target only"
    );
    ensure!(
        r.inputs[0].value == r.lhs_value && r.inputs[1].value == r.rhs_value,
        "original A/B ABI order required"
    );
    for (input, logical, allowed) in [
        (&r.inputs[0], vec![M, K], vec![vec![32, 512, K]]),
        (&r.inputs[1], vec![K, N], vec![vec![K, N]]),
    ] {
        ensure!(
            allowed.contains(&input.source_shape),
            "unknown whole-volume source"
        );
        ensure!(
            input
                .source_shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                == logical.iter().try_fold(1usize, |n, &d| n.checked_mul(d)),
            "whole volume changed"
        );
        let expected =
            VirtualTensor::identity(input.tensor.source, input.source_shape.clone(), DType::BF16)
                .reshape(logical.clone())?;
        ensure!(
            same_view(&input.tensor, &expected)?,
            "not an exact whole-volume identity/flatten"
        );
        crate::tile::ordinary_bf16_block::aligned_operand(input, &logical, false)
            .context("NN whole-map copy16 proof")?;
    }
    let s = r.schedule;
    ensure!(
        s.block_tile
            == (crate::tile::MatMulTile {
                m: 64,
                n: 64,
                k: 32
            })
            && s.block_threads == (32, 8, 1)
            && s.warp_topology == (2, 2)
            && s.warp_fragments == (2, 2)
            && s.pipeline_stages == 2
            && s.async_operands == (true, true)
            && s.copy_bytes == 16
            && s.reuse_shared_memory
            && s.effective_grid_group_m() <= 1
            && s.operand_layouts.0 == crate::tile::SharedLayout::padded(32, 8)?
            && s.operand_layouts.1 == crate::tile::SharedLayout::padded(64, 8)?,
        "complete accepted64 NN baseline required"
    );
    Ok(())
}
