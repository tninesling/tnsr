//! Default-off exact B32 FFN dX replacement after the final allocation rewrite.
//! Original Programs retain all storage/module ownership; rejection changes nothing.
use super::*;
use crate::tile::{
    MatMulPipeline, MatMulRegion, MatMulSharedLayout, MemorySpace, Stmt, TileIR, TileLayout,
    VirtualTensor,
};
use cudarc::driver::CudaFunction;
#[cfg(test)]
mod gpu_tests;
mod ir;
#[cfg(test)]
mod tests;
const M: usize = 16384;
const N: usize = 512;
const K: usize = 1408;

pub(super) fn requested(op: &TensorGraphNode<f32>, lhs: &[usize]) -> bool {
    matches!(op,TensorGraphNode::MatMul{shape}if shape==&[M,N]) && lhs == [M, K]
}
pub(super) struct Pending {
    owner: NodeIndex,
    region: MatMulRegion,
    function: CudaFunction,
    medium_square: usize,
}
fn whole(b: Binding, shape: &[usize]) -> bool {
    b.dtype == DType::BF16
        && b.pointer != 0
        && b.pointer.is_multiple_of(16)
        && shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)) == Some(b.len)
}
fn span(b: Binding) -> Result<(u64, u64)> {
    Ok((
        b.pointer,
        b.pointer
            .checked_add(u64::try_from(
                b.len
                    .checked_mul(b.dtype.size_bytes())
                    .context("BF16 byte extent overflow")?,
            )?)
            .context("BF16 pointer extent overflow")?,
    ))
}
fn known_counters<D: CudaDType>(p: &Program<D>) -> bool {
    p.plain_bf16_output_counts == (1, 0)
        && p.medium_bf16_tile_counts.0 <= 1
        && p.medium_bf16_tile_counts.1 == 0
        && p.ordinary_bf16_four_warp_count == 0
        && !p.bf16_eight_compute_warps
        && p.column_reduction_counts == (0, 0)
        && p.large_bf16_tile_counts == (0, 0)
        && p.large_square128_streamed_counts == (0, 0)
        && p.batched_attention_bf16_tile_counts
            == crate::tile::BatchedAttentionTileCounts::default()
        && p.indexed_loss_cooperative_ordered_backward_programs == 0
        && p.indexed_loss_cooperative_ordered_forward_programs == 0
        && p.contiguous_bias_reshape_alias_steps == 0
        && p.contiguous_bias_reshape_fallback_steps == 0
}
pub(super) fn pending<D: CudaDType>(owner: NodeIndex, p: &Program<D>) -> Result<Pending> {
    ensure!(
        D::HostType::TILE_DTYPE == DType::BF16 && known_counters(p),
        "unknown original ordinary64 counter family"
    );
    let [old] = p.kernels.as_slice() else {
        anyhow::bail!("requires complete single-kernel original Program")
    };
    let metadata = p
        .metadata
        .as_ref()
        .context("original typed Program metadata missing")?;
    let [step_index] = metadata.kernel_steps.as_slice() else {
        anyhow::bail!("requires one original physical step")
    };
    let step = metadata
        .plan
        .steps()
        .get(*step_index)
        .context("original step absent")?;
    let crate::ptx::PtxPlanAction::MatMulRegion(region_id) = step.action else {
        anyhow::bail!("original step is not plain MatMulRegion")
    };
    let [region] = metadata.plan.matmul_regions() else {
        anyhow::bail!("requires one original MatMulRegion")
    };
    ensure!(region_id == 0, "original region index changed");
    ir::guard(region)?;
    let [output_node] = step.outputs.as_slice() else {
        anyhow::bail!("requires one original output")
    };
    let output = *p
        .values
        .get(output_node)
        .context("original output binding missing")?;
    ensure!(
        whole(output, &[M, N])
            && output.pointer == p.output.pointer
            && output.len == p.output.len
            && output.dtype == p.output.dtype,
        "original root binding changed"
    );
    let mut expected = Vec::new();
    for input in &region.inputs {
        let b = *p
            .values
            .get(&input.tensor.source)
            .context("rebased physical input binding missing")?;
        ensure!(
            whole(b, &input.source_shape),
            "original physical input is not complete BF16 volume"
        );
        expected.push(b.pointer);
    }
    expected.push(output.pointer);
    ensure!(
        old.pointers == expected && old.pointers.len() == 3,
        "original plain3-pointer ABI changed"
    );
    check_launch(old)?;
    Ok(Pending {
        owner,
        region: region.clone(),
        function: old.function.clone(),
        medium_square: usize::from(p.medium_bf16_tile_counts.0 > 0),
    })
}
fn check_launch(k: &Kernel) -> Result<()> {
    ensure!(
        k.config.grid_dim == (8, 256, 1)
            && k.config.block_dim == (32, 8, 1)
            && k.config.shared_mem_bytes == 0
            && k.zero_before.is_none(),
        "complete original ordinary64 launch changed"
    );
    Ok(())
}
fn same_instance(
    owner: Option<NodeIndex>,
    identity: &str,
    pending_owner: NodeIndex,
    pending_identity: &str,
) -> bool {
    owner == Some(pending_owner) && identity == pending_identity
}
fn current<'a>(
    p: &Pending,
    kernels: &'a [Kernel],
    owners: &[Binding],
) -> Result<(usize, &'a Kernel)> {
    let identity = nt128_bias::ldmatrix::function_identity(&p.function)?;
    let mut matches = Vec::new();
    for (index, k) in kernels.iter().enumerate() {
        if k.owner == Some(p.owner)
            && same_instance(
                k.owner,
                &nt128_bias::ldmatrix::function_identity(&k.function)?,
                p.owner,
                &identity,
            )
        {
            matches.push((index, k));
        }
    }
    let [(index, old)] = matches.as_slice() else {
        anyhow::bail!("missing or ambiguous current owner+original CUfunction")
    };
    check_launch(old)?;
    ir::guard(&p.region)?;
    ensure!(old.pointers.len() == 3, "current3-pointer ABI changed");
    let bindings = old
        .pointers
        .iter()
        .zip([M * K, K * N, M * N])
        .map(|(&pointer, len)| Binding {
            pointer,
            len,
            dtype: DType::BF16,
        })
        .collect::<Vec<_>>();
    validate_ranges(&bindings, owners)?;
    Ok((*index, *old))
}
fn validate_ranges(bindings: &[Binding], owners: &[Binding]) -> Result<()> {
    ensure!(
        bindings.len() == 3 && bindings.iter().map(|b| b.len).eq([M * K, K * N, M * N]),
        "exact3-pointer binding inventory/lengths required"
    );
    for b in bindings {
        ensure!(
            whole(*b, &[b.len]),
            "current BF16 pointer alignment/dtype invalid"
        );
        let (lo, hi) = span(*b)?;
        let count = owners
            .iter()
            .filter(|o| o.dtype == DType::BF16 && span(**o).is_ok_and(|(a, z)| a <= lo && hi <= z))
            .count();
        ensure!(
            count == 1,
            "current range requires one live typed allocation owner, found {count}"
        );
    }
    let output = bindings[2];
    ensure!(
        output.pointer.is_multiple_of(128),
        "whole-WMMA fallback: output not128B aligned"
    );
    let (lo, hi) = span(output)?;
    for b in &bindings[..2] {
        let (a, z) = span(*b)?;
        ensure!(hi <= a || z <= lo, "current output overlaps input");
    }
    Ok(())
}
fn debit(plain: usize, medium: usize, original_medium: usize) -> Result<(usize, usize)> {
    Ok((
        plain
            .checked_sub(1)
            .context("original plain64 counter unavailable")?,
        medium
            .checked_sub(original_medium)
            .context("original medium-square counter unavailable")?,
    ))
}
struct Staged {
    index: usize,
    kernel: Kernel,
    ptx: String,
    identity: String,
    diagnostic: String,
    plain_count: usize,
    medium_count: usize,
}
fn stage(p: &Pending, e: &MixedExecutor, owners: &[Binding]) -> Result<Staged> {
    let (index, old) = current(p, &e.kernels, owners)?;
    let (plain_count, medium_count) = debit(
        e.info.plain_bf16_output_programs,
        e.info.medium_bf16_square_programs,
        p.medium_square,
    )?;
    let target = PtxTarget::from_context(e.stream.context())?;
    let candidate = ir::candidate(&p.region, target)?;
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.add_matmul_regions(std::slice::from_ref(&p.region))?;
    ensure!(
        tiles.matmul_region_kernels.len() == 1,
        "requires one exact candidate region"
    );
    tiles.matmul_region_kernels[0].ir = candidate;
    tiles.optimize_indices()?;
    let graph = PtxGraph::from(tiles).with_target(target);
    let ptx = graph.module_source();
    let name = graph
        .matmul_region_kernel_name(0)
        .context("NN128x64 entry absent")?;
    let module = e.stream.context().load_module(Ptx::from_src(&ptx))?;
    let function = module.load_function(name)?;
    let registers = function.num_regs()?;
    let shared = function.shared_size_bytes()?;
    let local = function.local_size_bytes()?;
    let maximum = function.max_threads_per_block()?;
    let capacity = function.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?;
    ensure!(
        (1..=128).contains(&registers)
            && shared <= 40960
            && local == 0
            && maximum >= 256
            && capacity >= 2,
        "NN128x64 resource rejection regs={registers} shared={shared} local={local} maxthreads={maximum} APIcapacity={capacity}"
    );
    #[cfg(test)]
    ensure!(
        !tests::forced_rejection(),
        "forced NN128x64 rejection after resource qualification"
    );
    let identity = nt128_bias::ldmatrix::function_identity(&function)?;
    let mut kernel = old.clone();
    kernel.function = function;
    kernel.config.grid_dim = (8, 128, 1);
    kernel.label = format!(
        "node={} NN128x64 ldmatrix plain selected tile=128x64x32 threads256 copy16 async2",
        p.owner.index()
    );
    let diagnostic = format!(
        "node {}: NN128x64 selected M={M} N={N} K={K} registers={registers} shared={shared} local={local} APIcapacity={capacity} at256",
        p.owner.index()
    );
    Ok(Staged {
        index,
        kernel,
        ptx,
        identity,
        diagnostic,
        plain_count,
        medium_count,
    })
}
pub(super) fn apply(e: &mut MixedExecutor) {
    if e.nn128x64_pending.is_empty() {
        return;
    }
    let owners = e
        .buffers
        .iter()
        .map(|b| binding(b, &e.stream))
        .chain(
            e.programs
                .iter()
                .filter_map(|p| p.downcast_ref::<Program<BF16>>())
                .flat_map(|p| p.storage_bindings(&e.stream)),
        )
        .collect::<Vec<_>>();
    let pending = std::mem::take(&mut e.nn128x64_pending);
    for p in pending {
        #[cfg(test)]
        let before = tests::snapshot(e);
        match stage(&p, e, &owners) {
            Ok(s) => {
                e.kernels[s.index] = s.kernel;
                e.nn128x64_sources.push((p.owner, s.ptx, s.identity));
                e.info.plain_bf16_output_programs = s.plain_count;
                e.info.medium_bf16_square_programs = s.medium_count;
                e.info.nn128x64_ldmatrix_programs += 1;
                e.info.nn128x64_ldmatrix_selections.push(s.diagnostic);
            }
            Err(reason) => {
                #[cfg(test)]
                assert_eq!(
                    before,
                    tests::snapshot(e),
                    "failed NN128x64 stage changed executable/storage/counters"
                );
                e.info.nn128x64_ldmatrix_fallback_programs += 1;
                e.info
                    .nn128x64_ldmatrix_rejections
                    .push(format!("node {}: {reason:#}", p.owner.index()));
            }
        }
    }
}
