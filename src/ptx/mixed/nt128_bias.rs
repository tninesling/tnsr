//! Default-off postcompile NT128 bias selection inside Mixed only.
//! The baseline Program is fully compiled first; every rejection keeps its
//! executable, launch, arguments, storage and counters unchanged.
use super::*;
use crate::{
    tensor::BinaryOp,
    tile::{
        Block, MatMulPipeline, MatMulRegion, MatMulSchedule, MatMulSharedLayout, RegionOpKind,
        ScalarBinaryOp, ScalarExpr, ScalarMemory, ScalarPredicate, Stmt, TileIR, TileVar,
        VirtualTensor,
    },
};
use anyhow::{Context, Result, ensure};
use cudarc::driver::CudaFunction;

#[cfg(test)]
mod direct_mma;
pub(in crate::ptx::mixed) mod ir;
pub(in crate::ptx::mixed) mod ldmatrix;
#[cfg(test)]
mod tests;

pub(super) enum Outcome {
    Selected,
    Retained(String),
}

pub(super) fn shape_candidate(output: &[usize], lhs: &[usize]) -> bool {
    let ([m, n], [am, k]) = (output, lhs) else {
        return false;
    };
    m == am
        && *m >= 2048
        && *n >= 256
        && *k >= 64
        && m.is_multiple_of(128)
        && n.is_multiple_of(128)
        && k.is_multiple_of(32)
}
struct Staged {
    kernel: Kernel,
    ptx: String,
    plan: String,
}
fn whole(binding: Binding, shape: &[usize]) -> bool {
    binding.dtype == DType::BF16
        && binding.pointer != 0
        && binding.pointer.is_multiple_of(16)
        && shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)) == Some(binding.len)
}
fn span(binding: Binding) -> Option<(u64, u64)> {
    Some((
        binding.pointer,
        binding.pointer.checked_add(
            u64::try_from(binding.len.checked_mul(binding.dtype.size_bytes())?).ok()?,
        )?,
    ))
}
fn known_counters<D: CudaDType>(p: &Program<D>) -> bool {
    p.medium_bf16_tile_counts.0 <= 1
        && p.medium_bf16_tile_counts.1 == 0
        && p.ordinary_bf16_four_warp_count == 0
        && !p.bf16_eight_compute_warps
        && p.column_reduction_counts == (0, 0)
        && p.large_bf16_tile_counts == (0, 0)
        && p.large_square128_streamed_counts == (0, 0)
        && p.plain_bf16_output_counts == (0, 0)
        && p.batched_attention_bf16_tile_counts
            == crate::tile::BatchedAttentionTileCounts::default()
        && p.indexed_loss_cooperative_ordered_backward_programs == 0
        && p.indexed_loss_cooperative_ordered_forward_programs == 0
}
fn stage<D: CudaDType>(p: &Program<D>, stream: &Arc<CudaStream>) -> Result<Staged> {
    ensure!(
        D::HostType::TILE_DTYPE == DType::BF16
            && p.output.dtype == DType::BF16
            && p.root_fused_matmul,
        "requires BF16 fused Linear root"
    );
    let [old] = p.kernels.as_slice() else {
        anyhow::bail!("requires complete single-kernel original Program");
    };
    ensure!(
        old.zero_before.is_none() && known_counters(p),
        "unsupported original counter family or zero-before"
    );
    let metadata = p
        .metadata
        .as_ref()
        .context("original selected plan metadata absent")?;
    let [step_index] = metadata.kernel_steps.as_slice() else {
        anyhow::bail!("requires one typed physical step");
    };
    let step = metadata
        .plan
        .steps()
        .get(*step_index)
        .context("original step absent")?;
    let crate::ptx::PtxPlanAction::MatMulRegion(region_id) = step.action else {
        anyhow::bail!("original single step is not fused MatMulRegion");
    };
    ensure!(
        metadata.plan.matmul_regions().len() == 1,
        "requires exactly one original MatMulRegion"
    );
    let region = metadata
        .plan
        .matmul_regions()
        .get(region_id)
        .context("original rebased region absent")?;
    let schedule = region.schedule;
    ensure!(
        schedule.block_tile
            == crate::tile::MatMulTile {
                m: 64,
                n: 64,
                k: 32
            }
            && schedule.block_threads == (32, 8, 1)
            && schedule.warp_topology == (2, 2)
            && schedule.warp_fragments == (2, 2)
            && schedule.pipeline_stages == 2
            && schedule.async_operands == (true, true)
            && schedule.copy_bytes == 16
            && schedule.reuse_shared_memory
            && schedule.grid_group_m == 1,
        "requires qualified ordinary64 baseline, async2/copy16/reuse"
    );
    ensure!(
        old.config.block_dim == schedule.block_threads
            && old.config.shared_mem_bytes == 0
            && old.config.grid_dim
                == (
                    u32::try_from(region.n.div_ceil(64))?,
                    u32::try_from(region.m.div_ceil(64))?,
                    1
                ),
        "original launch differs from selected ordinary64 plan"
    );
    let slot = ir::bias_slot(region)?;
    ensure!(
        slot == 2 && step.outputs.len() == 1,
        "requires one output and original A/W/bias order"
    );
    let output = *p
        .values
        .get(&step.outputs[0])
        .context("original output binding absent")?;
    ensure!(
        output.pointer == p.output.pointer
            && output.len == p.output.len
            && output.dtype == p.output.dtype
            && whole(output, &region.output_shape),
        "output/root binding changed"
    );
    let inputs = region
        .inputs
        .iter()
        .map(|input| {
            let binding = *p
                .values
                .get(&input.tensor.source)
                .context("rebased physical source binding absent")?;
            ensure!(
                whole(binding, &input.source_shape),
                "requires whole aligned BF16 physical input"
            );
            Ok(binding)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut expected = inputs.iter().map(|b| b.pointer).collect::<Vec<_>>();
    expected.push(output.pointer);
    ensure!(
        old.pointers == expected && expected.len() == 4,
        "original four-pointer ABI changed"
    );
    let (lo, hi) = span(output).context("output interval overflow")?;
    for &input in &inputs {
        let (a, b) = span(input).context("input interval overflow")?;
        ensure!(hi <= a || b <= lo, "output aliases input");
    }
    let target = PtxTarget::from_context(stream.context())?;
    let (candidate, selected, _) = ir::candidate(region, target)?;
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.add_matmul_regions(std::slice::from_ref(region))?;
    ensure!(
        tiles.matmul_region_kernels.len() == 1,
        "candidate region mapping changed"
    );
    tiles.matmul_region_kernels[0].ir = candidate;
    tiles.optimize_indices()?;
    let graph = PtxGraph::from(tiles).with_target(target);
    let ptx = graph.module_source();
    let name = graph
        .matmul_region_kernel_name(0)
        .context("candidate entry absent")?;
    let module = stream.context().load_module(Ptx::from_src(&ptx))?;
    let function = module.load_function(name)?;
    qualify(&function)?;
    #[cfg(test)]
    ensure!(
        !tests::forced_rejection(),
        "test-only forced rejection after successful resource qualification"
    );
    let kernel = Kernel {
        owner: old.owner,
        label: format!(
            "NT128 bias selected tile=128x128x32 blockwarps=8 copy16 async2 arena=40960 {selected:?}"
        ),
        function,
        config: LaunchConfig {
            grid_dim: (
                u32::try_from(region.n / 128)?,
                u32::try_from(region.m / 128)?,
                1,
            ),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        },
        pointers: old.pointers.clone(),
        zero_before: None,
    };
    Ok(Staged {
        kernel,
        ptx,
        plan: format!(
            "Mixed postcompile NT128 bias selected; original plan below is proof-only, not the selected geometry:\n{}",
            p.plan
        ),
    })
}
fn qualify(function: &CudaFunction) -> Result<()> {
    let registers = function.num_regs()?;
    let local = function.local_size_bytes()?;
    let shared = function.shared_size_bytes()?;
    let maximum = function.max_threads_per_block()?;
    let capacity = function.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?;
    ensure!(
        (1..=128).contains(&registers)
            && local == 0
            && shared == 40960
            && maximum >= 256
            && capacity >= 2,
        "NT128 resource rejection: registers={registers},shared={shared},local={local},max_threads={maximum},APIcapacity={capacity}; requires <=128/40KiB/0local/capacity>=2@256"
    );
    Ok(())
}
pub(super) fn replace<D: CudaDType>(
    node: NodeIndex,
    program: &mut Program<D>,
    stream: &Arc<CudaStream>,
) -> Outcome {
    #[cfg(test)]
    let before = tests::snapshot(node, program);
    #[cfg(test)]
    let executable_before = tests::ExecutableSnapshot::capture(program);
    let outcome = match stage(program, stream) {
        Err(reason) => Outcome::Retained(format!("{reason:#}")),
        Ok(selected) => {
            // All proof, allocations, compilation and driver queries succeeded.
            // No fallible operation follows the first executable publication.
            program.kernels = vec![selected.kernel];
            program.ptx = selected.ptx;
            program.plan = selected.plan;
            program.medium_bf16_tile_counts = (0, 0);
            // This immutable plan proved baseline bindings, but its64 schedule
            // must not be mistaken for the newly installed128 geometry.
            program.metadata = None;
            Outcome::Selected
        }
    };
    #[cfg(test)]
    {
        if matches!(outcome, Outcome::Retained(_)) {
            assert_eq!(
                executable_before,
                tests::ExecutableSnapshot::capture(program),
                "rejected candidate changed original Program"
            );
        }
        tests::record(before, &outcome, program);
    }
    #[cfg(not(test))]
    let _ = node;
    outcome
}
