//! Opt-in column-tiled reductions retaining the existing Block128 arithmetic DAG.
//!
//! Integration must keep the 128-contributor arithmetic DAG: contributor t
//! visits t + 128*k, then shared-memory offsets 64,32,16,8,4,2,1. Only the
//! assignment of independent columns to physical threads changes. Keep every
//! producer/epilogue SSA operation and typed load/store unchanged, including
//! the BF16 result's final round-to-nearest conversion and preceding copies.
//!
//! Proposed lowering: row_lane = tid / q; column_lane = tid % q. Scratch is
//! [128][q], indexed row_lane*q + column_lane. Every thread participates in
//! every barrier, including invalid tail columns; predicate their global
//! loads/stores and seed their accumulators with the reduction identity.
//! Do not append arithmetic padding terms to valid columns. In particular,
//! do not rewrite a producer Mul followed by accumulation Add: existing JIT
//! FMA contraction must retain the same opportunity and compiler settings.
//!
//! Before activation, integration still needs actual generated PTX/SASS
//! comparison, function/device resource checks, exact CPU/CUDA finite bits,
//! nonfinite classes/signed zero, changed-feed replay, and cache identity tests.
//! No hardware occupancy or bandwidth benefit follows from this descriptor.

#[cfg(any(feature = "cuda", test))]
use super::{DType, IndexMap, ReductionInputDomain, ReductionRegion, ReductionSchedule};
#[cfg(any(feature = "cuda", test))]
use crate::tensor::ReduceOp;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ColumnReductionPolicy {
    #[default]
    Off,
    Four,
    Eight,
}

/// Device and compiled-function hard limits, not an occupancy estimate.
/// A caller must recheck the actual JIT function's maximum thread count;
/// register pressure can make it lower than the device limit.
#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ColumnReductionLimits {
    pub max_threads_per_block: u32,
    pub shared_bytes_per_block: usize,
    pub max_grid_x: u32,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ColumnReductionTile {
    pub columns_per_block: usize,
    pub batches: usize,
    pub rows: usize,
    pub columns: usize,
    pub grid_x: u32,
    pub threads: u32,
    pub shared_bytes: usize,
}

#[cfg(any(feature = "cuda", test))]
impl ColumnReductionPolicy {
    /// Prove a deliberately narrower domain than generic reduction lowering.
    /// Returning a descriptor does not mutate the region or erase copies,
    /// producer operations, reduced epilogues, casts or graph outputs.
    pub(crate) fn select(
        self,
        region: &ReductionRegion,
        limits: ColumnReductionLimits,
    ) -> Option<ColumnReductionTile> {
        let q = match self {
            Self::Off => return None,
            Self::Four => 4usize,
            Self::Eight => 8usize,
        };
        if region.schedule != (ReductionSchedule::Block { threads: 128 })
            || region.op != ReduceOp::Sum
            || region.input_shape.contains(&0)
            || region.inputs.is_empty()
            || region.outputs.is_empty()
            || !region.broadcast_members.is_empty()
            || !region.full_epilogue_operations.is_empty()
            || region.outputs.len() != region.output_shapes.len()
        {
            return None;
        }
        let dtype = region.inputs.first()?.tensor.dtype;
        let (batches, rows, columns) = match (dtype, region.axis, region.input_shape.as_slice()) {
            (DType::F32, 1, &[b, s, c]) => (b, s, c),
            (DType::BF16, 0, &[r, c]) => (1, r, c),
            _ => return None,
        };
        let mut expected_output = region.input_shape.clone();
        expected_output[region.axis] = 1;
        if region.output_shape != expected_output
            || region
                .outputs
                .iter()
                .zip(&region.output_shapes)
                .any(|(output, shape)| {
                    region.output_domain(output) != ReductionInputDomain::Reduced
                        || *shape != expected_output
                })
        {
            return None;
        }
        // Exact identity is intentionally stronger than sampling a lane stride.
        // This excludes virtual transposes, broadcasts, slicing and padding.
        // Later expansion must prove the complete physical mapping, not merely
        // that the final logical dimension has the desired extent.
        if region.inputs.iter().any(|input| {
            let shape = match input.domain {
                ReductionInputDomain::Full => &region.input_shape,
                ReductionInputDomain::Reduced => &region.output_shape,
            };
            input.tensor.dtype != dtype
                || input.tensor.shape != *shape
                || input.source_shape != *shape
                || input.tensor.access != IndexMap::identity(shape.len())
                || input.tensor.predicate.is_some()
                || input.tensor.out_of_bounds.is_some()
        }) {
            return None;
        }
        // Bound complete physical extents before any address arithmetic.
        batches.checked_mul(rows)?.checked_mul(columns)?;
        let threads = u32::try_from(128usize.checked_mul(q)?).ok()?;
        let shared_bytes = usize::try_from(threads)
            .ok()?
            .checked_mul(size_of::<f32>())?;
        let grid_x = u32::try_from(batches.checked_mul(columns.div_ceil(q))?).ok()?;
        if threads > limits.max_threads_per_block
            || shared_bytes > limits.shared_bytes_per_block
            || grid_x > limits.max_grid_x
        {
            return None;
        }
        Some(ColumnReductionTile {
            columns_per_block: q,
            batches,
            rows,
            columns,
            grid_x,
            threads,
            shared_bytes,
        })
    }
}

#[cfg(test)]
impl ColumnReductionTile {
    /// None denotes a tail column, never permission to skip CTA barriers.
    #[cfg(test)]
    pub(crate) fn output_coordinate(self, block: usize, tid: usize) -> Option<(usize, usize)> {
        if block >= usize::try_from(self.grid_x).ok()?
            || tid >= usize::try_from(self.threads).ok()?
        {
            return None;
        }
        let tiles_per_batch = self.columns.div_ceil(self.columns_per_block);
        let batch = block / tiles_per_batch;
        let column =
            (block % tiles_per_batch) * self.columns_per_block + tid % self.columns_per_block;
        (column < self.columns).then_some((batch, column))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph::NodeIndex,
        tile::{ReductionInput, RegionOutput, RegionValue, VirtualTensor},
    };

    fn region(dtype: DType, shape: Vec<usize>, axis: usize) -> ReductionRegion {
        let mut output = shape.clone();
        output[axis] = 1;
        ReductionRegion {
            members: vec![NodeIndex::new(0), NodeIndex::new(1)],
            inputs: vec![ReductionInput {
                node: NodeIndex::new(0),
                value: RegionValue(0),
                domain: ReductionInputDomain::Full,
                tensor: VirtualTensor::identity(NodeIndex::new(0), shape.clone(), dtype),
                source_shape: shape.clone(),
            }],
            outputs: vec![RegionOutput {
                node: NodeIndex::new(1),
                value: RegionValue(1),
            }],
            output_shapes: vec![output.clone()],
            producer_operations: Vec::new(),
            epilogue_operations: Vec::new(),
            broadcast_members: Vec::new(),
            full_epilogue_operations: Vec::new(),
            reduction_input: RegionValue(0),
            reduced_value: RegionValue(1),
            input_shape: shape,
            output_shape: output,
            axis,
            op: ReduceOp::Sum,
            schedule: ReductionSchedule::Block { threads: 128 },
            serial_block_threads: 1024,
            column_policy: ColumnReductionPolicy::Off,
        }
    }

    fn limits() -> ColumnReductionLimits {
        ColumnReductionLimits {
            max_threads_per_block: 1024,
            shared_bytes_per_block: 49152,
            max_grid_x: i32::MAX as u32,
        }
    }

    #[test]
    fn column_policy_scope_resources_and_strict_mapping() {
        for (dtype, shape, axis) in [
            (DType::F32, vec![32, 512, 512], 1),
            (DType::BF16, vec![16384, 1408], 0),
        ] {
            let original = region(dtype, shape, axis);
            assert!(
                ColumnReductionPolicy::Off
                    .select(&original, limits())
                    .is_none()
            );
            for (policy, q) in [
                (ColumnReductionPolicy::Four, 4),
                (ColumnReductionPolicy::Eight, 8),
            ] {
                let tile = policy
                    .select(&original, limits())
                    .expect("eligible row-major region");
                assert_eq!(tile.threads, 128 * q);
                assert_eq!(tile.shared_bytes, 128 * q as usize * 4);
                let mut low = limits();
                low.max_threads_per_block = tile.threads - 1;
                assert!(policy.select(&original, low).is_none());
                let mut low = limits();
                low.shared_bytes_per_block = tile.shared_bytes - 1;
                assert!(policy.select(&original, low).is_none());
            }
            let mut mapped = original.clone();
            mapped.inputs[0].tensor.access.results.swap(0, 1);
            assert!(
                ColumnReductionPolicy::Four
                    .select(&mapped, limits())
                    .is_none()
            );
            let mut full_output = original.clone();
            full_output.outputs[0].value = RegionValue(0);
            assert!(
                ColumnReductionPolicy::Four
                    .select(&full_output, limits())
                    .is_none()
            );
            let mut subgroup = original;
            subgroup.schedule = ReductionSchedule::Subgroup { width: 32 };
            assert!(
                ColumnReductionPolicy::Four
                    .select(&subgroup, limits())
                    .is_none()
            );
        }
        for (dtype, shape, axis) in [
            (DType::F32, vec![32, 512, 512], 2),
            (DType::F32, vec![32, 1, 512], 0),
            (DType::F32, vec![512, 512], 0),
            (DType::BF16, vec![2, 512, 512], 1),
            (DType::BF16, vec![0, 512], 0),
        ] {
            assert!(
                ColumnReductionPolicy::Four
                    .select(&region(dtype, shape, axis), limits())
                    .is_none()
            );
        }
        let overflow = region(DType::F32, vec![usize::MAX, 512, 512], 1);
        assert!(
            ColumnReductionPolicy::Four
                .select(&overflow, limits())
                .is_none()
        );
    }

    fn sample(row: usize, column: usize, batch: usize) -> f32 {
        let value = ((row * 17 + column * 11 + batch * 7) % 31) as f32 - 15.0;
        if row.is_multiple_of(4) {
            value * 4096.0
        } else {
            value / 256.0
        }
    }

    fn reduce(mut partials: Vec<f32>) -> f32 {
        for stride in [64, 32, 16, 8, 4, 2, 1] {
            for t in 0..stride {
                let peer = partials[t + stride];
                partials[t] += peer;
            }
        }
        partials[0]
    }

    #[test]
    fn column_interleaving_preserves_addresses_accumulation_and_tree_bits() {
        for q in [4usize, 8] {
            for rows in [1usize, 127, 128, 129, 255, 256, 257, 512] {
                for columns in [1usize, 3, 4, 5, 7, 8, 9, 511, 512, 513] {
                    for batch in 0..2usize {
                        for column_base in (0..columns).step_by(q) {
                            let mut scratch = vec![0.0f32; 128 * q];
                            for t in 0..128 {
                                for c in 0..q {
                                    let column = column_base + c;
                                    if column >= columns {
                                        continue;
                                    }
                                    for row in (t..rows).step_by(128) {
                                        let old_address =
                                            batch * rows * columns + row * columns + column;
                                        let tiled_address = (batch * rows + row) * columns + column;
                                        assert_eq!(old_address, tiled_address);
                                        scratch[t * q + c] += sample(row, column, batch);
                                    }
                                }
                            }
                            for stride in [64, 32, 16, 8, 4, 2, 1] {
                                for t in 0..stride {
                                    for c in 0..q {
                                        let peer = scratch[(t + stride) * q + c];
                                        scratch[t * q + c] += peer;
                                    }
                                }
                            }
                            for (c, &actual) in scratch.iter().take(q).enumerate() {
                                let column = column_base + c;
                                if column >= columns {
                                    assert_eq!(actual.to_bits(), 0.0f32.to_bits());
                                    continue;
                                }
                                let old: Vec<_> = (0..128)
                                    .map(|t| {
                                        (t..rows).step_by(128).fold(0.0f32, |acc, row| {
                                            acc + sample(row, column, batch)
                                        })
                                    })
                                    .collect();
                                assert_eq!(actual.to_bits(), reduce(old).to_bits());
                            }
                        }
                    }
                }
            }
        }
    }

    // CPU storage oracle only. CUDA qualification must verify the emitted
    // cvt.rn.bf16.f32, NaN classes and signed zeros on the actual device.
    fn bf16_round_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        if value.is_nan() {
            return ((bits >> 16) as u16) | 0x0040;
        }
        ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
    }

    #[test]
    fn producer_fma_modes_and_bf16_final_rounding_preserve_column_bits() {
        for fused in [false, true] {
            for q in [4usize, 8] {
                let mut old = vec![vec![0.0f32; 128]; q];
                let mut tiled = vec![0.0f32; 128 * q];
                for t in 0..128 {
                    for row in (t..16384usize).step_by(128) {
                        for c in 0..q {
                            // BF16 storage values widen exactly before the
                            // generic F32 producer/accumulator instructions.
                            let x =
                                f32::from_bits(u32::from(bf16_round_bits(sample(row, c, 0))) << 16);
                            let y = f32::from_bits(
                                u32::from(bf16_round_bits(sample(row + 1, c, 1))) << 16,
                            );
                            let accumulate = |acc: f32| {
                                if fused {
                                    x.mul_add(y, acc)
                                } else {
                                    acc + x * y
                                }
                            };
                            old[c][t] = accumulate(old[c][t]);
                            tiled[t * q + c] = accumulate(tiled[t * q + c]);
                        }
                    }
                }
                for stride in [64, 32, 16, 8, 4, 2, 1] {
                    for t in 0..stride {
                        for c in 0..q {
                            let peer = tiled[(t + stride) * q + c];
                            tiled[t * q + c] += peer;
                        }
                    }
                }
                for c in 0..q {
                    let expected = reduce(old[c].clone());
                    assert_eq!(tiled[c].to_bits(), expected.to_bits());
                    assert_eq!(bf16_round_bits(tiled[c]), bf16_round_bits(expected));
                }
            }
        }
        assert_eq!(
            bf16_round_bits(1.0 + 1.0 / 256.0),
            0x3f80,
            "ties to even low mantissa"
        );
        assert_eq!(
            bf16_round_bits(1.0 + 3.0 / 256.0),
            0x3f82,
            "ties to even high mantissa"
        );
        assert_eq!(bf16_round_bits(-0.0), 0x8000);
        assert_eq!(bf16_round_bits(f32::INFINITY), 0x7f80);
        assert_ne!(bf16_round_bits(f32::NAN) & 0x007f, 0);
    }

    #[test]
    fn ragged_columns_keep_whole_cta_and_bound_output_coordinates() {
        let r = region(DType::F32, vec![2, 257, 5], 1);
        for policy in [ColumnReductionPolicy::Four, ColumnReductionPolicy::Eight] {
            let tile = policy.select(&r, limits()).expect("eligible tail columns");
            let q = tile.columns_per_block;
            for block in 0..tile.grid_x as usize {
                for tid in 0..tile.threads as usize {
                    let got = tile.output_coordinate(block, tid);
                    if let Some((batch, column)) = got {
                        assert!(batch < 2 && column < 5);
                    }
                    // Row contributors of the same column share the exact
                    // output coordinate, including neutral tail lanes.
                    assert_eq!(got, tile.output_coordinate(block, tid % q));
                }
            }
            assert_eq!(tile.output_coordinate(tile.grid_x as usize, 0), None);
        }
    }
}
