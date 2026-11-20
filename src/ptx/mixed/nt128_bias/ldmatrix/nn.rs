//! Test-only real B32 FFN dX: [16384,1408]x[1408,512], no bias.
//! No production selector, defaults, cache policy or accepted NT helper changes.
use super::*;
use crate::tile::{Dim, Expr, MemorySpace, TileLayout};
mod gpu;
mod one_cta;
mod rectangular;
mod tests;
const M: usize = 16384;
const N: usize = 512;
const K: usize = 1408;
const SHARED: usize = 37888;

fn graph() -> TensorGraph<half::bf16> {
    use crate::tensor::TensorExpr;
    let a = TensorExpr::<half::bf16>::input("dy", vec![32, 512, K]).reshape(vec![M, K]);
    let b = TensorExpr::<half::bf16>::input("weight", vec![K, N]);
    a.matmul(b).into()
}
fn fixture(target: PtxTarget) -> Result<MatMulRegion> {
    let graph = graph();
    let plan = crate::ptx::PtxExecutionPlan::build_with_reduction_mode(
        &graph,
        crate::ptx::PtxReductionMode::DeterministicTree,
    )?;
    let [r] = plan.matmul_regions() else {
        anyhow::bail!("one real rebased NN region required");
    };
    let mut r = r.clone();
    r.schedule = MatMulSchedule::select_with_resources(
        M,
        N,
        K,
        DType::BF16,
        target.matmul_capabilities(),
        crate::tile::MatMulPrecision::Tf32x3,
        target.matmul_resources,
    )
    .with_shared_memory_reuse(true);
    r.configure_pipeline(
        MatMulSharedLayout::Padded,
        MatMulPipeline::Auto,
        target.supports_async_copy(),
        target.matmul_resources,
    )?;
    guard(&r)?;
    Ok(r)
}
fn same_view(a: &VirtualTensor, b: &VirtualTensor) -> Result<bool> {
    Ok(a.source == b.source
        && a.dtype == b.dtype
        && a.shape == b.shape
        && a.predicate.is_none()
        && a.out_of_bounds.is_none()
        && a.access.normalize_in_domain(&a.shape)? == b.access.normalize_in_domain(&b.shape)?)
}
fn guard(r: &MatMulRegion) -> Result<()> {
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
        (&r.inputs[0], vec![M, K], vec![vec![M, K], vec![32, 512, K]]),
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
fn window_ir(r: &MatMulRegion, target: PtxTarget) -> Result<(TileIR, MatMulRegion)> {
    guard(r)?;
    let mut selected = r.clone();
    selected.schedule = crate::tile::large_tile_policy::square128_streamed::geometry(
        r.schedule,
        &r.output_shape,
        target.matmul_resources,
    )
    .context("test-only128 square geometry guard")?;
    selected.configure_pipeline(
        MatMulSharedLayout::Padded,
        MatMulPipeline::Auto,
        target.supports_async_copy(),
        target.matmul_resources,
    )?;
    ensure!(
        selected.schedule.is_square128_streamed()
            && selected.schedule.pipeline_stages == 2
            && selected.schedule.async_operands == (true, true)
            && selected.schedule.copy_bytes == 16
            && selected.schedule.reuse_shared_memory,
        "retain128 async2/copy16/reuse"
    );
    ensure!(
        selected.schedule.operand_layouts.0 == crate::tile::SharedLayout::padded(32, 8)?
            && selected.schedule.operand_layouts.1 == crate::tile::SharedLayout::padded(128, 8)?
            && selected.schedule.shared_mem_bytes() == SHARED,
        "NN requires A40/B136 two-stage37KiB"
    );
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.add_matmul_regions(std::slice::from_ref(&selected))?;
    let ir = tiles.matmul_region_kernels[0].ir.clone();
    ir.validate_layouts()?;
    ensure!(
        ir.params.len() == 3
            && ir.params.iter().all(|p| p.dtype == DType::BF16)
            && ir
                .params
                .iter()
                .map(|p| p.name.as_str())
                .eq(["input_0", "input_1", "output_0"]),
        "exact no-bias3-pointer ABI"
    );
    Ok((ir, selected))
}
fn candidate(r: &MatMulRegion, target: PtxTarget) -> Result<TileIR> {
    let (mut ir, selected) = window_ir(r, target)?;
    let first = ir
        .body
        .stmts
        .iter()
        .position(|s| matches!(s, Stmt::StoreAccumulatorWindow { .. }))
        .context("missing old windows")?;
    ensure!(
        matches!(
            ir.body.stmts[..first]
                .iter()
                .rev()
                .find(|s| !matches!(s, Stmt::ViewSharedArena { .. })),
            Some(Stmt::Barrier)
        ),
        "keep final async drain/barrier"
    );
    let suffix = &ir.body.stmts[first..];
    ensure!(suffix.len() == 15, "four-window suffix invariant");
    for pass in 0..4 {
        let i = pass * 4;
        ensure!(
            matches!(suffix[i],Stmt::StoreAccumulatorWindow{fragment_origin,fragment_shape:(1,2),..}if fragment_origin==(pass/2,(pass%2)*2))
                && matches!(suffix[i + 1], Stmt::Barrier),
            "old window coordinates changed"
        );
        let Stmt::ForLoop {
            start, end, body, ..
        } = &suffix[i + 2]
        else {
            anyhow::bail!("old store loop changed");
        };
        ensure!(
            *start == Expr::Const(0)
                && *end == Expr::Const(16)
                && body.stmts.len() == 3
                && matches!(body.stmts.last(), Some(Stmt::StoreGlobalPredicated { .. })),
            "plain output must have no arithmetic/bias"
        );
        if pass != 3 {
            ensure!(
                matches!(suffix[i + 3], Stmt::Barrier),
                "old exchange retirement changed"
            );
        }
    }
    let mut accumulator = None;
    for s in &mut ir.body.stmts[..first] {
        if let Stmt::AllocTile {
            var,
            space: MemorySpace::Fragment,
            layout,
            ..
        } = s
        {
            ensure!(
                accumulator.is_none() && *layout == selected.schedule.accumulator_tile_layout(),
                "single original WMMA accumulator"
            );
            accumulator = Some(*var);
            *layout = TileLayout::Bf16MmaAccumulator128LdmatrixNn;
        }
    }
    let src = accumulator.context("missing accumulator")?;
    ir.body.stmts.truncate(first);
    ir.body.stmts.push(Stmt::StoreBf16MmaCoalesced {
        src,
        output_param: "output_0".into(),
        bias_param: None,
        m: M,
        n: N,
    });
    ir.kernel_name = "test_nn_ldmatrix_plain_bf16_coalesced".into();
    ir.validate_layouts()?;
    Ok(ir)
}
fn source(r: &MatMulRegion, target: PtxTarget, body: TileIR) -> Result<String> {
    let mut t = TileGraph::from_graph(&TensorGraph::<f32>::new());
    t.add_matmul_regions(std::slice::from_ref(r))?;
    t.matmul_region_kernels[0].ir = body;
    t.optimize_indices()?;
    Ok(PtxGraph::from(t).with_target(target).module_source())
}
