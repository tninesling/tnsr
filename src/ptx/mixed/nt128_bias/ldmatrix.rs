//! Final post-reuse replacement of the qualified NT WMMA kernel.
//! Pending records contain no allocation addresses. All current typed ownership
//! and output alignment checks precede compilation and atomic publication.
use super::*;
mod ir;
#[cfg(test)]
mod nn;
#[cfg(test)]
mod tests;

pub(in crate::ptx::mixed) struct Pending {
    owner: NodeIndex,
    region: MatMulRegion,
    old_function: CudaFunction,
}

pub(in crate::ptx::mixed) fn exact_shape(r: &MatMulRegion) -> bool {
    matches!((r.m, r.n, r.k), (16384, 1408, 512) | (16384, 1536, 512))
}

/// Borrow the immutable rebased plan before the existing WMMA stage consumes
/// it. No clone occurs for other shapes. Its full ABI is proved by that stage.
pub(in crate::ptx::mixed) fn region<D: CudaDType>(p: &Program<D>) -> Option<MatMulRegion> {
    let metadata = p.metadata.as_ref()?;
    let [r] = metadata.plan.matmul_regions() else {
        return None;
    };
    exact_shape(r).then(|| r.clone())
}

pub(in crate::ptx::mixed) fn pending<D: CudaDType>(
    owner: NodeIndex,
    region: MatMulRegion,
    p: &Program<D>,
) -> Option<Pending> {
    let [kernel] = p.kernels.as_slice() else {
        return None;
    };
    Some(Pending {
        owner,
        region,
        old_function: kernel.function.clone(),
    })
}

pub(in crate::ptx::mixed) fn function_identity(function: &CudaFunction) -> Result<String> {
    // cudarc0.17 exposes no public function handle getter. Its pinned derived
    // Debug representation exposes the handle; reject any format change rather
    // than relying on entry labels, attributes or unsafe private field access.
    parse_function_identity(&format!("{function:?}"))
}
fn parse_function_identity(text: &str) -> Result<String> {
    ensure!(
        text.starts_with("CudaFunction { cu_function:"),
        "unknown cudarc function identity representation"
    );
    ensure!(
        text.matches("cu_function:").count() == 1,
        "ambiguous cudarc function field"
    );
    let (identity, _) = text
        .split_once(", module:")
        .context("missing function module identity boundary")?;
    let handle = identity
        .strip_prefix("CudaFunction { cu_function: 0x")
        .context("unexpected function pointer representation")?;
    ensure!(
        !handle.is_empty()
            && handle.bytes().all(|b| b.is_ascii_hexdigit())
            && handle.bytes().any(|b| b != b'0'),
        "invalid function pointer identity"
    );
    Ok(identity.to_owned())
}

fn owned_range(b: Binding, owners: &[Binding]) -> Result<()> {
    let (lo, hi) = span(b).context("required typed range overflow")?;
    ensure!(whole(b, &[b.len]), "unaligned or invalid BF16 ABI");
    let containing = owners
        .iter()
        .filter(|owner| {
            owner.dtype == b.dtype && span(**owner).is_some_and(|(a, z)| a <= lo && hi <= z)
        })
        .count();
    ensure!(
        containing == 1,
        "required BF16 range must have one live allocation owner, found {containing}"
    );
    Ok(())
}

fn same_instance(
    owner: Option<NodeIndex>,
    identity: &str,
    expected_owner: NodeIndex,
    expected_identity: &str,
) -> bool {
    owner == Some(expected_owner) && identity == expected_identity
}

fn current_kernel<'a>(
    pending: &Pending,
    kernels: &'a [Kernel],
    owners: &[Binding],
) -> Result<(usize, &'a Kernel)> {
    let identity = function_identity(&pending.old_function)?;
    // Ordinary F32 Programs may carry LOCAL numeric owner IDs. Match the
    // retained module's exact function identity as well as the global Linear
    // owner; never reinterpret those unrelated local IDs as global semantics.
    let mut matches = Vec::new();
    for (index, kernel) in kernels.iter().enumerate() {
        if kernel.owner == Some(pending.owner)
            && same_instance(
                kernel.owner,
                &function_identity(&kernel.function)?,
                pending.owner,
                &identity,
            )
        {
            matches.push((index, kernel));
        }
    }
    let [(index, old)] = matches.as_slice() else {
        anyhow::bail!("missing or ambiguous current semantic kernel");
    };
    let r = &pending.region;
    ensure!(
        exact_shape(r) && super::ir::bias_slot(r)? == 2,
        "exact qualified shape/ordered bias proof lost"
    );
    ensure!(
        function_identity(&old.function)? == function_identity(&pending.old_function)?
            && old.zero_before.is_none()
            && old.config.block_dim == (32, 8, 1)
            && old.config.grid_dim == (u32::try_from(r.n / 128)?, u32::try_from(r.m / 128)?, 1)
            && old.config.shared_mem_bytes == 0
            && old.pointers.len() == 4,
        "current complete WMMA function/launch/ABI changed"
    );
    let lengths = [
        r.m.checked_mul(r.k),
        r.n.checked_mul(r.k),
        Some(r.n),
        r.m.checked_mul(r.n),
    ];
    let bindings = old
        .pointers
        .iter()
        .zip(lengths)
        .map(|(&pointer, len)| {
            Ok(Binding {
                pointer,
                len: len.context("required element count overflow")?,
                dtype: DType::BF16,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    for &b in &bindings {
        owned_range(b, owners)?;
    }
    let output = bindings[3];
    ensure!(
        output.pointer.is_multiple_of(128),
        "whole WMMA fallback: output is not 128-byte aligned"
    );
    let (lo, hi) = span(output).context("output extent overflow")?;
    for &b in &bindings[..3] {
        let (a, z) = span(b).context("input extent overflow")?;
        ensure!(hi <= a || z <= lo, "current output overlaps input");
    }
    Ok((*index, *old))
}

struct Staged {
    index: usize,
    kernel: Kernel,
    ptx: String,
    diagnostic: String,
    identity: String,
}

fn stage(pending: &Pending, executor: &MixedExecutor, owners: &[Binding]) -> Result<Staged> {
    let (index, old) = current_kernel(pending, &executor.kernels, owners)?;
    let target = PtxTarget::from_context(executor.stream.context())?;
    let candidate = ir::candidate(&pending.region, target)?;
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.add_matmul_regions(std::slice::from_ref(&pending.region))?;
    ensure!(
        tiles.matmul_region_kernels.len() == 1,
        "one candidate region required"
    );
    tiles.matmul_region_kernels[0].ir = candidate;
    tiles.optimize_indices()?;
    let graph = PtxGraph::from(tiles).with_target(target);
    let ptx = graph.module_source();
    let name = graph
        .matmul_region_kernel_name(0)
        .context("ldmatrix entry absent")?;
    let module = executor.stream.context().load_module(Ptx::from_src(&ptx))?;
    let function = module.load_function(name)?;
    super::qualify(&function)?;
    #[cfg(test)]
    ensure!(
        !tests::forced_rejection(),
        "forced ldmatrix rejection after resource qualification"
    );
    let diagnostic = format!(
        "node {}: ldmatrix coalesced selected M={} N={} K={} registers={} shared={} local={} APIcapacity={} at256",
        pending.owner.index(),
        pending.region.m,
        pending.region.n,
        pending.region.k,
        function.num_regs()?,
        function.shared_size_bytes()?,
        function.local_size_bytes()?,
        function.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?
    );
    let identity = function_identity(&function)?;
    let mut kernel = old.clone();
    kernel.function = function;
    kernel.label = format!(
        "node={} NT128 bias ldmatrix coalesced selected tile=128x128x32 blockwarps=8 copy16 async2 arena=40960",
        pending.owner.index()
    );
    Ok(Staged {
        index,
        kernel,
        ptx,
        diagnostic,
        identity,
    })
}

/// No subsequent allocator compaction is allowed. The replacement has exactly
/// the old kernel's read/write ABI, and the old module stays owned by Program.
pub(in crate::ptx::mixed) fn apply(executor: &mut MixedExecutor) {
    if executor.nt128_ldmatrix_pending.is_empty() {
        return;
    }
    let owners = executor
        .buffers
        .iter()
        .map(|b| binding(b, &executor.stream))
        .chain(
            executor
                .programs
                .iter()
                .filter_map(|p| p.downcast_ref::<Program<BF16>>())
                .flat_map(|p| p.storage_bindings(&executor.stream)),
        )
        .collect::<Vec<_>>();
    let pending = std::mem::take(&mut executor.nt128_ldmatrix_pending);
    for p in pending {
        #[cfg(test)]
        let before = tests::executable_snapshot(executor);
        match stage(&p, executor, &owners) {
            Ok(staged) => {
                // Every fallible proof/compilation/query completed before commit.
                executor.kernels[staged.index] = staged.kernel;
                executor
                    .nt128_ldmatrix_sources
                    .push((p.owner, staged.ptx, staged.identity));
                executor.info.fused_nt128_bias_ldmatrix_programs += 1;
                executor
                    .info
                    .fused_nt128_bias_ldmatrix_selections
                    .push(staged.diagnostic);
            }
            Err(reason) => {
                #[cfg(test)]
                assert_eq!(
                    before,
                    tests::executable_snapshot(executor),
                    "rejected late stage changed the WMMA executable/storage metadata"
                );
                executor.info.fused_nt128_bias_ldmatrix_fallback_programs += 1;
                executor
                    .info
                    .fused_nt128_bias_ldmatrix_rejections
                    .push(format!("node {}: {reason:#}", p.owner.index()));
            }
        }
    }
}
