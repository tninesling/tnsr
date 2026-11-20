//! Explicit numerical experiment: four contiguous K parts for a small ordinary
//! output grid. Existing long-contraction eligibility/policy is deliberately
//! separate. Partial storage is F32; only the ordered left-four reducer rounds
//! to BF16. This proof cannot repack or allocate operands.
use super::{
    DType, MatMulAccumulatorExchange, MatMulAccumulatorLayout, MatMulFragmentLoad, MatMulPlan,
    MatMulRegion, MatMulResources, MatMulSchedule, MatMulTile, RegionInput, SharedLayout,
    VirtualTensor,
};

#[derive(Clone, Debug)]
pub(crate) struct SmallGridCandidate {
    baseline: MatMulRegion,
    schedule: MatMulSchedule,
}

fn extent(shape: &[usize]) -> Option<usize> {
    let volume = shape.iter().try_fold(1_usize, |a, &b| a.checked_mul(b))?;
    (volume > 0 && volume.checked_mul(2)? <= isize::MAX as usize).then_some(volume)
}

impl SmallGridCandidate {
    /// Structural ordinary shape proof. It intentionally does not admit the
    /// existing long Split-K family, which must run first without modification.
    pub(crate) fn eligible_shape(m: usize, n: usize, total_k: usize) -> bool {
        let checked = || -> Option<bool> {
            let elements = m.checked_mul(n)?;
            let ctas = (m / 64).checked_mul(n / 64)?;
            Some(
                m >= 64
                    && n >= 64
                    && total_k >= 512
                    && m.is_multiple_of(64)
                    && n.is_multiple_of(64)
                    && total_k.is_multiple_of(128)
                    && elements <= 512 * 1024
                    && (1..128).contains(&ctas)
                    && !(m >= 128 && n >= 128 && total_k >= 2048 && total_k / m.max(n) >= 4)
                    && extent(&[m, total_k]).is_some()
                    && extent(&[total_k, n]).is_some(),
            )
        };
        checked().unwrap_or(false)
    }

    pub(crate) fn candidate(
        region: &MatMulRegion,
        total_k: usize,
        resources: MatMulResources,
        allow_shared_reuse: bool,
    ) -> Option<Self> {
        let (m, n, part_k) = (region.m, region.n, region.k);
        let old = region.schedule;
        if !allow_shared_reuse
            || !Self::eligible_shape(m, n, total_k)
            || part_k.checked_mul(4)? != total_k
            || region.batch_shape != [4]
            || region.output_shape != [4, m, n]
            || region.lhs_shape != [4, m, part_k]
            || region.rhs_shape != [4, part_k, n]
            || region.inputs.len() != 2
            || region.outputs.len() != 1
            || !region.epilogue_operations.is_empty()
            || region.outputs[0].value != region.matmul_value
            || old.logical_shape != (MatMulTile { m, n, k: part_k })
            || old.plan != MatMulPlan::TensorCoreBF16
            || old.storage_dtype != DType::BF16
            || old.operand_dtype != DType::BF16
            || old.instruction_tile
                != (MatMulTile {
                    m: 16,
                    n: 16,
                    k: 16,
                })
        {
            return None;
        }
        extent(&[4, m, n])?.checked_mul(4)?;
        let lhs = region
            .inputs
            .iter()
            .find(|input| input.value == region.lhs_value)?;
        let rhs = region
            .inputs
            .iter()
            .find(|input| input.value == region.rhs_value)?;
        full_partition_map(lhs, m, n, part_k, true)?;
        full_partition_map(rhs, m, n, part_k, false)?;
        // The selected partial geometry is independent of ordinary tile policies.
        // Explicitly reset streamed-fragment choices from unrelated experiments.
        let mut schedule = old;
        schedule.block_tile = MatMulTile {
            m: 64,
            n: 64,
            k: 32,
        };
        schedule.block_threads = (32, 8, 1);
        schedule.warp_topology = (2, 2);
        schedule.warp_fragments = (2, 2);
        schedule.accumulator_layout = MatMulAccumulatorLayout::WarpFragment;
        schedule.accumulator_exchange = MatMulAccumulatorExchange::FullTile;
        schedule.fragment_load = MatMulFragmentLoad::PreloadAll;
        schedule.grid_group_m = 1;
        schedule.operand_layouts = (SharedLayout::row_major(32), SharedLayout::row_major(64));
        schedule.pipeline_stages = 1;
        schedule.copy_bytes = 4;
        schedule.async_operands = (false, false);
        schedule.reuse_shared_memory = true;
        schedule
            .conservative_two_block_fit(resources)
            .then_some(Self {
                baseline: region.clone(),
                schedule,
            })
    }
}

/// Prove EVERY physical index against one of two full-volume matrix layouts,
/// followed by the exact four contiguous logical-K partitions. No sampling,
/// offsets, broadcasts, predicates, negative strides or padded row gaps.
fn full_partition_map(
    input: &RegionInput,
    m: usize,
    n: usize,
    part_k: usize,
    lhs: bool,
) -> Option<()> {
    let total_k = part_k.checked_mul(4)?;
    let shape = if lhs {
        vec![4, m, part_k]
    } else {
        vec![4, part_k, n]
    };
    if input.tensor.shape != shape
        || input.tensor.dtype != DType::BF16
        || input.tensor.predicate.is_some()
        || input.tensor.out_of_bounds.is_some()
        || extent(&input.source_shape)? != extent(&shape)?
    {
        return None;
    }
    let actual = super::matmul_copy::physical_affine(input).ok()??;
    if actual.0 != 0 {
        return None;
    }
    let base =
        VirtualTensor::identity(input.tensor.source, input.source_shape.clone(), DType::BF16);
    for column_major in [false, true] {
        let logical = match (lhs, column_major) {
            (true, false) => base.reshape(vec![m, total_k]),
            (true, true) => base
                .reshape(vec![total_k, m])
                .and_then(|v| v.permute(&[1, 0])),
            (false, false) => base.reshape(vec![total_k, n]),
            (false, true) => base
                .reshape(vec![n, total_k])
                .and_then(|v| v.permute(&[1, 0])),
        }
        .ok()?;
        let expected = if lhs {
            logical
                .reshape(vec![m, 4, part_k])
                .and_then(|v| v.permute(&[1, 0, 2]))
        } else {
            logical.reshape(vec![4, part_k, n])
        }
        .ok()?;
        let expected = RegionInput {
            node: input.node,
            value: input.value,
            tensor: expected,
            source_shape: input.source_shape.clone(),
        };
        if super::matmul_copy::physical_affine(&expected).ok()?? == actual
            && super::matmul_copy::copy_orientation(input, 8).ok()?? == column_major
        {
            return Some(());
        }
    }
    None
}

#[cfg(feature = "cuda")]
impl SmallGridCandidate {
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
            || s.logical_shape != self.schedule.logical_shape
            || s.instruction_tile != self.schedule.instruction_tile
            || s.block_threads != (32, 8, 1)
            || s.warp_topology != (2, 2)
            || s.warp_fragments != (2, 2)
            || s.pipeline_stages != 2
            || s.copy_bytes != 16
            || s.async_operands != (true, true)
            || !s.reuse_shared_memory
            || s.grid_group_m != 1
            || s.accumulator_layout != MatMulAccumulatorLayout::WarpFragment
            || s.accumulator_exchange != MatMulAccumulatorExchange::FullTile
            || s.fragment_load != MatMulFragmentLoad::PreloadAll
            || !s.conservative_two_block_fit(resources)
        {
            return None;
        }
        Some(s)
    }

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
        ir.kernel_name = "small_grid_split_k_partial_f32_v1".into();
        if ir.params.iter().filter(|p| p.is_input).count() != 2
            || ir.params.iter().filter(|p| !p.is_input).count() != 1
        {
            return None;
        }
        for p in &mut ir.params {
            if p.is_input {
                if p.dtype != DType::BF16 {
                    return None;
                }
            } else {
                p.dtype = DType::F32;
            }
        }
        ir.validate_layouts().ok()?;
        Some((schedule, ir))
    }
}

#[cfg(test)]
mod tests;
