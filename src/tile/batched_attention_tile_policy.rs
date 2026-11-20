//! Opt-in exact BF16 S512/D64 batched attention rectangle selection.
//!
//! This selector changes ownership/staging only. It retains the ordinary
//! increasing-K16 MMA sequence and the final BF16 storage round. No attention
//! normalization or gradient algebra is changed. Existing policies are untouched.
#[cfg(any(feature = "cuda", test))]
use super::{
    DType, MatMulAccumulatorLayout, MatMulPlan, MatMulRegion, MatMulResources, MatMulSchedule,
    MatMulTile, RegionInput, VirtualTensor,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BatchedAttentionBf16TilePolicy {
    #[default]
    Off,
    TallScores,
    WideScores,
    /// Admit lower batch/head counts only when each role's checked CTA grid
    /// still meets the existing minimum. All physical-map and final proofs apply.
    TallScoresByGrid,
}

/// Actually installed shape-role and rectangle counts, after final copy,
/// pipeline and resource admission. A requested policy alone contributes zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BatchedAttentionTileCounts {
    pub scores: usize,
    pub query_width: usize,
    pub width_query: usize,
    pub tall: usize,
    pub wide: usize,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AttentionTileRole {
    Scores,
    QueryWidth,
    WidthQuery,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Debug)]
pub(crate) struct AttentionTileCandidate {
    pub schedule: MatMulSchedule,
    pub role: AttentionTileRole,
    output_shape: Vec<usize>,
    orientations: (bool, bool),
}

#[cfg(any(feature = "cuda", test))]
fn extent(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1_usize, |n, &d| n.checked_mul(d))
}

// Match whole access maps against explicitly bijective full-volume view chains.
// An aligned sample or a unit stride alone does NOT prove there is no broadcast.
#[cfg(any(feature = "cuda", test))]
fn plain_attention_view(input: &RegionInput, shape: &[usize]) -> bool {
    if shape.len() != 4
        || input.tensor.shape != shape
        || input.tensor.dtype != DType::BF16
        || input.tensor.predicate.is_some()
        || input.tensor.out_of_bounds.is_some()
        || input.tensor.access.results.len() != input.source_shape.len()
        || extent(&input.source_shape) != extent(shape)
        || extent(shape).is_none()
    {
        return false;
    }
    let (batch, heads) = (shape[0], shape[1]);
    let base =
        VirtualTensor::identity(input.tensor.source, input.source_shape.clone(), DType::BF16);
    let Some(actual_affine) = super::matmul_copy::physical_affine(input).ok().flatten() else {
        return false;
    };
    let same = |expected: &VirtualTensor| {
        let expected_input = RegionInput {
            node: input.node,
            value: input.value,
            tensor: expected.clone(),
            source_shape: input.source_shape.clone(),
        };
        expected.shape == input.tensor.shape
            && super::matmul_copy::physical_affine(&expected_input)
                .ok()
                .flatten()
                .is_some_and(|map| map == actual_affine)
    };
    // Contiguous BH matrices (including harmless full-volume flatten/reshape).
    if let Ok(expected) = base.reshape(shape.to_vec())
        && same(&expected)
    {
        return true;
    }
    // Last-two-axis transpose of contiguous BH matrices.
    let transposed = base
        .reshape(vec![batch, heads, shape[3], shape[2]])
        .and_then(|v| v.permute(&[0, 1, 3, 2]));
    if transposed.as_ref().is_ok_and(same) {
        return true;
    }
    // Actual Q/K/V and dO storage: token-major BSHD, viewed as BHSD or BHDS.
    // Square attention matrices deliberately do not gain an arbitrary batch
    // permutation through this branch.
    let axes = match (shape[2], shape[3]) {
        (512, 64) => [0, 2, 1, 3],
        (64, 512) => [0, 2, 3, 1],
        _ => return false,
    };
    base.reshape(vec![batch, 512, heads, 64])
        .and_then(|v| v.permute(&axes))
        .as_ref()
        .is_ok_and(same)
}

#[cfg(any(feature = "cuda", test))]
fn operands(region: &MatMulRegion) -> Option<(bool, bool)> {
    let [batch, heads, m, n]: [usize; 4] = region.output_shape.as_slice().try_into().ok()?;
    let expected = [
        vec![batch, heads, m, region.k],
        vec![batch, heads, region.k, n],
    ];
    if region.lhs_shape != expected[0] || region.rhs_shape != expected[1] {
        return None;
    }
    let input = |value| region.inputs.iter().find(|input| input.value == value);
    let (lhs, rhs) = (input(region.lhs_value)?, input(region.rhs_value)?);
    if !plain_attention_view(lhs, &expected[0]) || !plain_attention_view(rhs, &expected[1]) {
        return None;
    }
    Some((
        super::matmul_copy::copy_orientation(lhs, 8).ok()??,
        super::matmul_copy::copy_orientation(rhs, 8).ok()??,
    ))
}

#[cfg(any(feature = "cuda", test))]
impl BatchedAttentionBf16TilePolicy {
    /// Preselect only a closed, ordinary single-output BF16 MatMul. Call
    /// `configure` before installation; this raw geometry is NOT a final schedule.
    pub(crate) fn candidate(
        self,
        region: &MatMulRegion,
        resources: MatMulResources,
        allow_shared_reuse: bool,
    ) -> Option<AttentionTileCandidate> {
        let original = region.schedule;
        let [batch, heads, m, n]: [usize; 4] = region.output_shape.as_slice().try_into().ok()?;
        let batch_heads = batch.checked_mul(heads)?;
        let minimum_batch_heads = if self == Self::TallScoresByGrid {
            1
        } else {
            64
        };
        if self == Self::Off
            || !allow_shared_reuse
            || batch == 0
            || heads == 0
            || !(minimum_batch_heads..=65535).contains(&batch_heads)
            || extent(&region.output_shape)? > u32::MAX as usize
            || region.batch_shape != [batch, heads]
            || region.m != m
            || region.n != n
            || original.logical_shape != (MatMulTile { m, n, k: region.k })
            || original.plan != MatMulPlan::TensorCoreBF16
            || original.storage_dtype != DType::BF16
            || original.operand_dtype != DType::BF16
            || original.instruction_tile
                != (MatMulTile {
                    m: 16,
                    n: 16,
                    k: 16,
                })
            || original.accumulator_layout != MatMulAccumulatorLayout::WarpFragment
            || !region.epilogue_operations.is_empty()
            || region.inputs.len() != 2
            || region.outputs.len() != 1
            || region.outputs[0].value != region.matmul_value
            || original.warp_fragments != (1, 1)
            || !matches!(
                original.block_tile,
                MatMulTile {
                    m: 16,
                    n: 16,
                    k: 16
                } | MatMulTile {
                    m: 16,
                    n: 32,
                    k: 32
                } | MatMulTile {
                    m: 32,
                    n: 32,
                    k: 32
                }
            )
        {
            return None;
        }
        let (role, tile_m, tile_n) = match (m, n, region.k) {
            (512, 512, 64) => (
                AttentionTileRole::Scores,
                if self == Self::WideScores { 64 } else { 128 },
                if self == Self::WideScores { 128 } else { 64 },
            ),
            (512, 64, 512) => (AttentionTileRole::QueryWidth, 128, 64),
            (64, 512, 512) => (AttentionTileRole::WidthQuery, 64, 128),
            _ => return None,
        };
        // Explicit full-tile scope; no tail/zero-fill assumptions are borrowed
        // from another schedule. Count all batched CTAs, not one head's grid.
        if !m.is_multiple_of(tile_m) || !n.is_multiple_of(tile_n) {
            return None;
        }
        let ctas = batch_heads
            .checked_mul(m / tile_m)?
            .checked_mul(n / tile_n)?;
        if ctas < 128 {
            return None;
        }
        let orientations = operands(region)?;
        let schedule = original.bf16_rectangle_geometry(tile_m, tile_n)?;
        schedule
            .conservative_two_block_fit(resources)
            .then_some(AttentionTileCandidate {
                schedule,
                role,
                output_shape: region.output_shape.clone(),
                orientations,
            })
    }
}

#[cfg(feature = "cuda")]
impl AttentionTileCandidate {
    /// New-policy-only forced two-stage admission. No global Auto heuristic is
    /// broadened, and an explicit synchronous user request remains a fallback.
    /// Callers must restore the complete original schedule when this returns None.
    pub(crate) fn configure(
        &self,
        region: &MatMulRegion,
        shared: super::MatMulSharedLayout,
        pipeline: super::MatMulPipeline,
        supports_async: bool,
        resources: MatMulResources,
    ) -> Option<MatMulSchedule> {
        if !supports_async
            || pipeline == super::MatMulPipeline::Synchronous
            || region.output_shape != self.output_shape
            || operands(region)? != self.orientations
        {
            return None;
        }
        let mut probe = region.clone();
        probe.schedule = self.schedule;
        probe
            .configure_pipeline(
                shared,
                super::MatMulPipeline::DoubleBuffered,
                true,
                resources,
            )
            .ok()?;
        let s = probe.schedule;
        if s.block_tile != self.schedule.block_tile
            || s.instruction_tile != self.schedule.instruction_tile
            || s.logical_shape != self.schedule.logical_shape
            || s.block_threads != (32, 8, 1)
            || s.warp_topology != self.schedule.warp_topology
            || s.warp_fragments != (2, 2)
            || s.accumulator_layout != MatMulAccumulatorLayout::WarpFragment
            || s.storage_dtype != DType::BF16
            || s.operand_dtype != DType::BF16
            || !s.reuse_shared_memory
            || s.pipeline_stages != 2
            || s.async_operands != (true, true)
            || s.copy_bytes != 16
            || (
                s.operand_layouts.0.column_major,
                s.operand_layouts.1.column_major,
            ) != self.orientations
            || !s.operand_layouts.0.supports_wmma(DType::BF16)
            || !s.operand_layouts.1.supports_wmma(DType::BF16)
            || !s.conservative_two_block_fit(resources)
        {
            return None;
        }
        Some(s)
    }
}

#[cfg(test)]
mod tests;
