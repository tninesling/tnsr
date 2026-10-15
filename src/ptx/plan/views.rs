//! Layout profitability for reduction-lane accesses.
use super::*;

/// Compare a unit lane step with the same reduction over materialized row-major
/// storage. This is a profitability sample, not a legality proof. In particular,
/// serial batch-gradient reductions need contiguous neighboring output fibers;
/// keeping a transposed matrix virtual can turn those reads into large strides.
pub(super) fn reduction_lane_stride_worsens(
    view: &VirtualTensor,
    source: &[usize],
    axis: usize,
    schedule: ReductionSchedule,
) -> Result<bool> {
    if view.shape.contains(&0) {
        return Ok(false);
    }
    let lane_axis = match schedule {
        ReductionSchedule::Serial => view
            .shape
            .iter()
            .enumerate()
            .rfind(|&(dimension, &extent)| dimension != axis && extent > 1)
            .map(|(dimension, _)| dimension),
        _ => view
            .shape
            .get(axis)
            .filter(|&&extent| extent > 1)
            .map(|_| axis),
    };
    let Some(lane_axis) = lane_axis else {
        return Ok(false);
    };
    let offset = |coordinates: &[usize]| -> Result<usize> {
        view.source_index(coordinates)?.iter().zip(source).try_fold(
            0usize,
            |offset, (&coordinate, &extent)| {
                offset
                    .checked_mul(extent)
                    .and_then(|x| x.checked_add(coordinate))
                    .context("reduction layout offset overflow")
            },
        )
    };
    let mut coordinates = vec![0; view.shape.len()];
    let base = offset(&coordinates)?;
    coordinates[lane_axis] = 1;
    let stride = offset(&coordinates)?.abs_diff(base);
    let contiguous = view.shape[lane_axis + 1..]
        .iter()
        .try_fold(1usize, |stride, &extent| stride.checked_mul(extent))
        .context("reduction layout stride overflow")?;
    Ok(stride > contiguous)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::DType;

    #[test]
    fn reduction_layout_distinguishes_serial_fibers_and_cooperative_axis_lanes() {
        let source = vec![2, 3, 5];
        let view = VirtualTensor::identity(NodeIndex::new(0), source.clone(), DType::F32)
            .permute(&[0, 2, 1])
            .unwrap();
        assert!(
            reduction_lane_stride_worsens(&view, &source, 0, ReductionSchedule::Serial).unwrap()
        );
        let empty = VirtualTensor::identity(NodeIndex::new(0), vec![1, 0], DType::F32)
            .broadcast_axis(0, 8)
            .unwrap();
        assert!(
            !reduction_lane_stride_worsens(&empty, &[1, 0], 1, ReductionSchedule::Serial).unwrap()
        );
        assert!(
            !reduction_lane_stride_worsens(&view, &source, 2, ReductionSchedule::Serial).unwrap()
        );
        assert!(
            reduction_lane_stride_worsens(
                &view,
                &source,
                2,
                ReductionSchedule::Block { threads: 128 }
            )
            .unwrap()
        );
        assert!(
            !reduction_lane_stride_worsens(
                &view,
                &source,
                1,
                ReductionSchedule::Block { threads: 128 }
            )
            .unwrap()
        );
        let broadcast = VirtualTensor::identity(NodeIndex::new(0), vec![1, 3, 5], DType::F32)
            .broadcast_axis(0, 8)
            .unwrap();
        assert!(
            !reduction_lane_stride_worsens(&broadcast, &[1, 3, 5], 0, ReductionSchedule::Serial)
                .unwrap()
        );
    }
}
