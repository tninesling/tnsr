//! Bounded Mixed-only special Split-K schedule policy.
//!
//! Selected only AFTER the current split_k.rs packing decisions and wider
//! fallback have completed, immediately BEFORE partial TileIR creation. Never
//! call this from ordinary/batched planning. This helper cannot allocate/repack
//! buffers; it returns a schedule/IR for the SAME already-bound operands.
//!
//! Initial scope is the profiled old32x32x64, padded A-column/B-row, four-part
//! family only. Existing64x64 cases remain unchanged. Each part still covers its
//! original contiguous K/4 keys and increasing K16 sequence. partK%64=0 makes
//! K64→K32 staging retain exactly the same number of MMA instructions, including
//! no extra zero groups. F32 partial storage and the existing left-associated
//! (((p0+p1)+p2)+p3) reducer (NO initial+0) plus ONE BF16 RN store are mandatory.
//!
//! API: MixedOptions.split_k_bf16_tile_policy, defaultOff. No Program or
//! ordinary PtxExecutor setter is appropriate: this path bypasses Program.
//! Append codegen_tag() to a COMPLETE special-kernel identity that also includes
//! target/precision, shapes, exact input access maps/dtypes, selected schedule,
//! outputF32 dtype, and reducer ABI. Driver JIT already hashes full PTX; distinct
//! policy-tagged kernel names/diagnostics make identity explicit. Owner capture
//! caches must retain typed bindings/module/function identity and invalidate all
//! variants if a future mutable policy setter rebuilds kernels. A policy tag
//! ALONE is not a sufficient module or pointer-cache key. Root coordinates any
//! global signature separately; this immutable per-executor special path has no global setter.
#[cfg(any(feature = "cuda", test))]
use super::{
    DType, MatMulAccumulatorLayout, MatMulPlan, MatMulRegion, MatMulResources, MatMulSchedule,
    MatMulTile, RegionInput, SharedLayout, VirtualTensor,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SplitKBf16TilePolicy {
    #[default]
    Off,
    Square64,
    /// Experimental qualified B32 physical-family rectangle; never a default.
    Rectangle128x64,
}
#[cfg(any(feature = "cuda", test))]
impl SplitKBf16TilePolicy {
    pub(crate) fn codegen_tag(self) -> &'static [u8] {
        match self {
            Self::Off => b"tnsr/splitk/existing/v1",
            Self::Square64 => b"tnsr/splitk/parts4/square64-k32/f32partials/left4-bf16rn/v1",
            Self::Rectangle128x64 => {
                b"tnsr/splitk/parts4/rectangle128x64-k32/f32partials/left4-bf16rn/v1"
            }
        }
    }
}
#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Debug)]
pub(crate) struct SplitKSquareCandidate {
    baseline: MatMulRegion,
    schedule: MatMulSchedule,
}

#[cfg(any(feature = "cuda", test))]
fn extent(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |a, &b| a.checked_mul(b))
}
#[cfg(any(feature = "cuda", test))]
fn plain_partition_map(input: &RegionInput, m: usize, n: usize, part_k: usize, lhs: bool) -> bool {
    let shape = if lhs {
        vec![4, m, part_k]
    } else {
        vec![4, part_k, n]
    };
    let Some(total_k) = part_k.checked_mul(4) else {
        return false;
    };
    if input.tensor.shape != shape
        || input.tensor.dtype != DType::BF16
        || input.tensor.predicate.is_some()
        || input.tensor.out_of_bounds.is_some()
        || extent(&input.source_shape) != extent(&shape)
        || extent(&shape).is_none()
    {
        return false;
    }
    let Some(actual) = super::matmul_copy::physical_affine(input).ok().flatten() else {
        return false;
    };
    if actual.0 != 0 {
        return false;
    } // no offset slice/Narrow or padded broadcast
    let base =
        VirtualTensor::identity(input.tensor.source, input.source_shape.clone(), DType::BF16);
    // Reconstruct exactly a full-volume physical transpose on the LHS, then
    // the original contiguous four-K partitioning. Source may retain harmless
    // batch/sequence axes; exact flattened affine equality proves EVERY index.
    let expected = if lhs {
        base.reshape(vec![total_k, m])
            .and_then(|v| v.permute(&[1, 0]))
            .and_then(|v| v.reshape(vec![m, 4, part_k]))
            .and_then(|v| v.permute(&[1, 0, 2]))
    } else {
        base.reshape(vec![4, part_k, n])
    };
    let Ok(expected) = expected else {
        return false;
    };
    let expected = RegionInput {
        node: input.node,
        value: input.value,
        tensor: expected,
        source_shape: input.source_shape.clone(),
    };
    super::matmul_copy::physical_affine(&expected)
        .ok()
        .flatten()
        .is_some_and(|map| map == actual)
}

#[cfg(any(feature = "cuda", test))]
impl SplitKBf16TilePolicy {
    /// Metadata proof only; the caller must invoke configure/partial_ir and
    /// retain the COMPLETE original region.schedule on any None result.
    pub(crate) fn candidate(
        self,
        region: &MatMulRegion,
        parts: usize,
        total_k: usize,
        resources: MatMulResources,
        allow_shared_reuse: bool,
    ) -> Option<SplitKSquareCandidate> {
        let old = region.schedule;
        let (m, n, k) = (region.m, region.n, region.k);
        if self == Self::Off
            || !allow_shared_reuse
            || parts != 4
            || k.checked_mul(parts)? != total_k
            || m < 128
            || n < 128
            || total_k < 2048
            || total_k / m.max(n) < 4
            || k < 64
            || !k.is_multiple_of(64)
            || !m.is_multiple_of(64)
            || !n.is_multiple_of(64)
            || region.batch_shape != [4]
            || region.output_shape != [4, m, n]
            || region.lhs_shape != [4, m, k]
            || region.rhs_shape != [4, k, n]
            || old.logical_shape != (MatMulTile { m, n, k })
            || old.plan != MatMulPlan::TensorCoreBF16
            || old.storage_dtype != DType::BF16
            || old.operand_dtype != DType::BF16
            || old.block_tile
                != (MatMulTile {
                    m: 32,
                    n: 32,
                    k: 64,
                })
            || old.instruction_tile
                != (MatMulTile {
                    m: 16,
                    n: 16,
                    k: 16,
                })
            || old.block_threads != (32, 8, 1)
            || old.warp_topology != (2, 2)
            || old.warp_fragments != (1, 1)
            || old.accumulator_layout != MatMulAccumulatorLayout::WarpFragment
            || old.grid_group_m != 1
            || old.pipeline_stages != 2
            || old.copy_bytes != 16
            || old.async_operands != (true, true)
            || old.reuse_shared_memory
            || old.shared_mem_bytes() != 24576
            || (
                old.operand_layouts.0.column_major,
                old.operand_layouts.1.column_major,
            ) != (true, false)
            || region.inputs.len() != 2
            || region.outputs.len() != 1
            || !region.epilogue_operations.is_empty()
            || region.outputs[0].value != region.matmul_value
        {
            return None;
        }
        extent(&[4, m, n])?.checked_mul(4)?;
        extent(&[4, m, k])?;
        extent(&[4, k, n])?;
        let ctas = 4usize.checked_mul(m / 64)?.checked_mul(n / 64)?;
        if ctas < 128 || m / 64 > u16::MAX as usize || n / 64 > i32::MAX as usize {
            return None;
        }
        let lhs = region.inputs.iter().find(|x| x.value == region.lhs_value)?;
        let rhs = region.inputs.iter().find(|x| x.value == region.rhs_value)?;
        if !plain_partition_map(lhs, m, n, k, true) || !plain_partition_map(rhs, m, n, k, false) {
            return None;
        }
        if (
            super::matmul_copy::copy_orientation(lhs, 8).ok()??,
            super::matmul_copy::copy_orientation(rhs, 8).ok()??,
        ) != (true, false)
        {
            return None;
        }
        let mut schedule = old;
        schedule.block_tile = MatMulTile {
            m: 64,
            n: 64,
            k: 32,
        };
        schedule.warp_topology = (2, 2);
        schedule.warp_fragments = (2, 2);
        schedule.operand_layouts = (SharedLayout::row_major(32), SharedLayout::row_major(64));
        schedule.pipeline_stages = 1;
        schedule.copy_bytes = 4;
        schedule.async_operands = (false, false);
        schedule.reuse_shared_memory = true;
        schedule
            .conservative_two_block_fit(resources)
            .then_some(SplitKSquareCandidate {
                baseline: region.clone(),
                schedule,
            })
    }
}

#[cfg(feature = "cuda")]
impl SplitKSquareCandidate {
    pub(crate) fn configure(
        &self,
        region: &MatMulRegion,
        resources: MatMulResources,
        supports_async: bool,
    ) -> Option<MatMulSchedule> {
        if !supports_async || region != &self.baseline {
            return None;
        }
        let mut probe = region.clone();
        probe.schedule = self.schedule;
        probe
            .configure_pipeline(
                super::MatMulSharedLayout::Padded,
                super::MatMulPipeline::Auto,
                true,
                resources,
            )
            .ok()?;
        let s = probe.schedule;
        if s.block_tile != self.schedule.block_tile
            || s.logical_shape != self.baseline.schedule.logical_shape
            || s.instruction_tile != self.baseline.schedule.instruction_tile
            || s.block_threads != (32, 8, 1)
            || s.warp_topology != (2, 2)
            || s.warp_fragments != (2, 2)
            || s.pipeline_stages != 2
            || s.copy_bytes != 16
            || s.async_operands != (true, true)
            || !s.reuse_shared_memory
            || (
                s.operand_layouts.0.column_major,
                s.operand_layouts.1.column_major,
            ) != (true, false)
            || s.shared_mem_bytes() != 18432
            || !s.conservative_two_block_fit(resources)
        {
            return None;
        }
        Some(s)
    }

    /// Construct only the partial GEMM. This does not touch existing operand
    /// buffers, repack decisions, output shape or the separate final reducer.
    /// It explicitly retains F32 partial stores despite BF16 input graph dtype.
    pub(crate) fn partial_ir(
        &self,
        region: &MatMulRegion,
        resources: MatMulResources,
        supports_async: bool,
    ) -> Option<(MatMulSchedule, super::TileIR)> {
        let schedule = self.configure(region, resources, supports_async)?;
        let mut probe = region.clone();
        probe.schedule = schedule;
        let mut tiles = super::TileGraph::from_graph(&crate::graph::TensorGraph::<f32>::new());
        tiles.add_matmul_regions(&[probe]).ok()?;
        let mut ir = tiles.matmul_region_kernels.pop()?.ir;
        ir.kernel_name = "split_k_square64_partial_f32_v1".into();
        if ir.params.iter().filter(|p| p.is_input).count() != 2
            || ir.params.iter().filter(|p| !p.is_input).count() != 1
        {
            return None;
        }
        for p in &mut ir.params {
            if !p.is_input {
                p.dtype = DType::F32;
            } else if p.dtype != DType::BF16 {
                return None;
            }
        }
        ir.validate_layouts().ok()?;
        Some((schedule, ir))
    }
}

#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
mod rectangle;
