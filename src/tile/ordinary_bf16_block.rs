//! Opt-in ordinary rank-two BF16 block contraction. Install only after final
//! operand/pipeline selection. Only the declared block changes; TileIR must then
//! be regenerated so cooperative copies and scalar output rounds cover the tile.
use super::{
    DType, MatMulAccumulatorLayout, MatMulPlan, MatMulRegion, MatMulResources, MatMulSchedule,
    MatMulTile, RegionInput,
};

pub(crate) fn candidate(
    original: MatMulSchedule,
    output_shape: &[usize],
    resources: MatMulResources,
) -> Option<MatMulSchedule> {
    let shape = original.logical_shape;
    if output_shape != [shape.m, shape.n]
        || original.plan != MatMulPlan::TensorCoreBF16
        || original.storage_dtype != DType::BF16
        || original.operand_dtype != DType::BF16
        || original.accumulator_layout != MatMulAccumulatorLayout::WarpFragment
        || original.instruction_tile
            != (MatMulTile {
                m: 16,
                n: 16,
                k: 16,
            })
        || original.block_tile
            != (MatMulTile {
                m: 64,
                n: 64,
                k: 32,
            })
        || original.block_threads != (32, 8, 1)
        || original.warp_topology != (2, 2)
        || original.warp_fragments != (2, 2)
        || original.pipeline_stages != 2
        || original.async_operands != (true, true)
        || original.copy_bytes != 16
        || !original.reuse_shared_memory
        || shape.m < 2048
        || shape.n < 256
        || shape.k < 64
        || !shape.m.is_multiple_of(64)
        || !shape.n.is_multiple_of(64)
        || !shape.k.is_multiple_of(32)
        || !original.operand_layouts.0.supports_wmma(DType::BF16)
        || !original.operand_layouts.1.supports_wmma(DType::BF16)
    {
        return None;
    }
    original
        .operand_layouts
        .0
        .validate(64, 32, DType::BF16)
        .ok()?;
    original
        .operand_layouts
        .1
        .validate(32, 64, DType::BF16)
        .ok()?;
    let _checked_stage_bytes = original
        .operand_layouts
        .0
        .allocation_bytes(64, 32, DType::BF16)
        .checked_add(
            original
                .operand_layouts
                .1
                .allocation_bytes(32, 64, DType::BF16),
        )?
        .checked_mul(2)?;
    if !original.conservative_two_block_fit(resources) {
        return None;
    }
    for (a, b) in [(shape.m, shape.k), (shape.k, shape.n), (shape.m, shape.n)] {
        if a.checked_mul(b)? > isize::MAX as usize {
            return None;
        }
    }
    let (rows, cols) = (shape.m / 64, shape.n / 64);
    if rows > 65535 || cols > i32::MAX as usize || rows.checked_mul(cols)? < 128 {
        return None;
    }
    let mut selected = original;
    selected.block_threads = (32, 4, 1);
    // Same padded operand stages and F32 exchange, including their lifetime
    // overlay. The reserve uses the new actual thread count, never hardcoded256.
    (selected.shared_mem_bytes() == original.shared_mem_bytes()
        && selected.conservative_two_block_fit(resources))
    .then_some(selected)
}

/// Prove the whole rank-two physical map, including bounds and nonoverlap.
/// Positive aligned row gaps are supported; broadcast/negative/wrapping or
/// predicated maps do not acquire a smaller cooperative staging block.
pub(crate) fn aligned_operand(
    input: &RegionInput,
    shape: &[usize],
    column_major: bool,
) -> Option<()> {
    if shape.len() != 2 || input.tensor.shape != shape || input.tensor.dtype != DType::BF16 {
        return None;
    }
    if super::matmul_copy::copy_orientation(input, 8).ok()?? != column_major {
        return None;
    }
    let (constant, strides) = super::matmul_copy::physical_affine(input).ok()??;
    let [row, col]: [i64; 2] = strides.as_slice().try_into().ok()?;
    let minor = usize::from(!column_major);
    let major_stride = if column_major { col } else { row };
    if constant < 0 || row <= 0 || col <= 0 || i128::from(major_stride) < shape[minor] as i128 {
        return None;
    }
    let last = i128::from(constant)
        .checked_add((shape[0] - 1) as i128 * i128::from(row))?
        .checked_add((shape[1] - 1) as i128 * i128::from(col))?;
    let physical = input
        .source_shape
        .iter()
        .try_fold(1_usize, |v, &d| v.checked_mul(d))?;
    (last < physical as i128).then_some(())
}

pub(crate) fn region_candidate(
    region: &MatMulRegion,
    resources: MatMulResources,
) -> Option<MatMulSchedule> {
    if region.output_shape != [region.m, region.n]
        || region.schedule.logical_shape
            != (MatMulTile {
                m: region.m,
                n: region.n,
                k: region.k,
            })
        || !region.batch_shape.is_empty()
        || region.lhs_shape != [region.m, region.k]
        || region.rhs_shape != [region.k, region.n]
    {
        return None;
    }
    let selected = candidate(region.schedule, &region.output_shape, resources)?;
    for (value, shape, orientation) in [
        (
            region.lhs_value,
            &region.lhs_shape,
            selected.operand_layouts.0.column_major,
        ),
        (
            region.rhs_value,
            &region.rhs_shape,
            selected.operand_layouts.1.column_major,
        ),
    ] {
        let input = region.inputs.iter().find(|input| input.value == value)?;
        aligned_operand(input, shape, orientation)?;
    }
    Some(selected)
}

#[cfg(test)]
mod tests;
