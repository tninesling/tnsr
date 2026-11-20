//! Bounded opt-in ordinary rank-two BF16 rectangles. Apply BEFORE shared
//! layout/pipeline selection with explicit arena reuse; only publish a candidate
//! after actual operand copy/layout proofs retain two stages and the existing
//! conservative two-block resource reserve. No split-K or batched policy here.
#[cfg(any(feature = "cuda", test))]
pub(crate) mod square128_streamed;
#[cfg(any(feature = "cuda", test))]
use super::{DType, MatMulPlan, MatMulResources, MatMulSchedule, MatMulTile};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LargeBf16TilePolicy {
    #[default]
    Off,
    Tall,
    Wide,
    TallThenWide,
    /// Explicit streamed ordinary square; all legacy choices remain unchanged.
    Square128Streamed,
}

#[cfg(any(feature = "cuda", test))]
impl LargeBf16TilePolicy {
    /// Ordered alternatives. Configuration may reject an otherwise geometrically
    /// valid tall candidate; TallThenWide must then try the wide candidate too.
    pub(crate) fn candidates(
        self,
        original: MatMulSchedule,
        logical_output_shape: &[usize],
        shared_memory_reuse: bool,
    ) -> Vec<MatMulSchedule> {
        let shape = original.logical_shape;
        if self == Self::Off
            || !shared_memory_reuse
            || logical_output_shape != [shape.m, shape.n]
            || original.plan != MatMulPlan::TensorCoreBF16
            || original.storage_dtype != DType::BF16
            || original.operand_dtype != DType::BF16
            || original.block_tile
                != (MatMulTile {
                    m: 64,
                    n: 64,
                    k: 32,
                })
            || original.block_threads != (32, 8, 1)
            || original.warp_topology != (2, 2)
            || original.warp_fragments != (2, 2)
            || shape.m < 2048
            || shape.n < 256
            || shape.k < 64
            || !shape.k.is_multiple_of(32)
        {
            return Vec::new();
        }
        if shape.m.checked_mul(shape.k).is_none()
            || shape.k.checked_mul(shape.n).is_none()
            || shape.m.checked_mul(shape.n).is_none()
        {
            return Vec::new();
        }
        let geometries: &[(usize, usize)] = match self {
            Self::Off | Self::Square128Streamed => &[],
            Self::Tall => &[(128, 64)],
            Self::Wide => &[(64, 128)],
            Self::TallThenWide => &[(128, 64), (64, 128)],
        };
        geometries
            .iter()
            .filter_map(|&(m, n)| {
                if !shape.m.is_multiple_of(m) || !shape.n.is_multiple_of(n) {
                    return None;
                }
                let grid = (shape.m / m).checked_mul(shape.n / n)?;
                if grid < 128 || shape.m / m > i32::MAX as usize || shape.n / n > i32::MAX as usize
                {
                    return None;
                }
                original.bf16_rectangle_geometry(m, n)
            })
            .collect()
    }

    pub(crate) fn configured_candidate_fits(
        schedule: MatMulSchedule,
        resources: MatMulResources,
    ) -> bool {
        let tile = schedule.block_tile;
        matches!((tile.m, tile.n, tile.k), (128, 64, 32) | (64, 128, 32))
            && schedule.block_threads == (32, 8, 1)
            && schedule.warp_topology == (tile.m / 32, tile.n / 32)
            && schedule.warp_fragments == (2, 2)
            && schedule.plan == MatMulPlan::TensorCoreBF16
            && schedule.storage_dtype == DType::BF16
            && schedule.operand_dtype == DType::BF16
            && schedule.reuse_shared_memory
            && schedule.pipeline_stages == 2
            && schedule.async_operands != (false, false)
            && schedule.conservative_two_block_fit(resources)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::{MatMulCapabilities, MatMulPipeline, MatMulPrecision, MatMulSharedLayout};

    fn resources() -> MatMulResources {
        MatMulResources {
            shared_bytes_per_sm: 100 * 1024,
            threads_per_sm: 1536,
            ..MatMulResources::default()
        }
    }
    fn ordinary(m: usize, n: usize, k: usize) -> MatMulSchedule {
        MatMulSchedule::select_with_resources(
            m,
            n,
            k,
            DType::BF16,
            MatMulCapabilities {
                tf32: true,
                f16: true,
                bf16: true,
            },
            MatMulPrecision::NativeHalf,
            resources(),
        )
    }

    #[test]
    fn large_rectangle_scope_full_tiles_grid_and_reuse_are_explicit() {
        for (m, n, k, rank, enabled, expected) in [
            (4096, 1408, 512, 2, true, 2),
            (2048, 512, 64, 2, true, 2),
            (2048, 256, 512, 2, true, 0),
            (2049, 512, 512, 2, true, 0),
            (4096, 1408, 65, 2, true, 0),
            (4096, 1408, 512, 3, true, 0),
            (4096, 1408, 512, 2, false, 0),
            (128, 4096, 512, 2, true, 0),
        ] {
            let shape = if rank == 2 { vec![m, n] } else { vec![1, m, n] };
            assert_eq!(
                LargeBf16TilePolicy::TallThenWide
                    .candidates(ordinary(m, n, k), &shape, enabled)
                    .len(),
                expected
            );
        }
        assert!(
            LargeBf16TilePolicy::Off
                .candidates(ordinary(4096, 512, 512), &[4096, 512], true)
                .is_empty()
        );
        assert_eq!(
            LargeBf16TilePolicy::TallThenWide
                .candidates(ordinary(2112, 512, 512), &[2112, 512], true)
                .iter()
                .map(|s| (s.block_tile.m, s.block_tile.n))
                .collect::<Vec<_>>(),
            [(64, 128)]
        );
        let mut scalar = ordinary(4096, 512, 512);
        scalar.plan = MatMulPlan::ScalarF32;
        assert!(
            LargeBf16TilePolicy::Tall
                .candidates(scalar, &[4096, 512], true)
                .is_empty()
        );
        let mut wrong_dtype = ordinary(4096, 512, 512);
        wrong_dtype.storage_dtype = DType::F32;
        assert!(
            LargeBf16TilePolicy::Tall
                .candidates(wrong_dtype, &[4096, 512], true)
                .is_empty()
        );
        let mut overflow = ordinary(4096, 512, 512);
        overflow.logical_shape.k = (usize::MAX / 32) * 32;
        assert!(
            LargeBf16TilePolicy::Tall
                .candidates(overflow, &[4096, 512], true)
                .is_empty()
        );
    }

    #[test]
    fn physical_reuse_admits_rectangles_without_relaxing_limits_or_default_guards() {
        let old = ordinary(4096, 1536, 512);
        for candidate in LargeBf16TilePolicy::TallThenWide.candidates(old, &[4096, 1536], true) {
            for orientation in [(false, false), (true, false), (false, true), (true, true)] {
                let configure = |s: MatMulSchedule, budget| {
                    s.with_operand_orientations(MatMulSharedLayout::Padded, orientation, budget)
                        .unwrap()
                        .with_operand_pipeline(
                            MatMulPipeline::DoubleBuffered,
                            true,
                            true,
                            (true, true),
                            budget,
                        )
                };
                let disjoint = configure(candidate.with_shared_memory_reuse(false), resources());
                assert_eq!(
                    disjoint.pipeline_stages, 1,
                    "old disjoint2stage guard must still reject"
                );
                let reused = configure(candidate, resources());
                assert!(LargeBf16TilePolicy::configured_candidate_fits(
                    reused,
                    resources()
                ));
                assert_eq!(reused.shared_mem_bytes(), 32768);
                assert!((26624..=30720).contains(
                    &(reused.pipeline_stages
                        * (reused.operand_layouts.0.allocation_bytes(
                            reused.block_tile.m,
                            32,
                            DType::BF16
                        ) + reused.operand_layouts.1.allocation_bytes(
                            32,
                            reused.block_tile.n,
                            DType::BF16
                        )))
                ));
                for constrained in [
                    MatMulResources {
                        shared_bytes_per_block: 32767,
                        ..resources()
                    },
                    MatMulResources {
                        shared_bytes_per_sm: 65535,
                        ..resources()
                    },
                    MatMulResources {
                        registers_per_sm: 65535,
                        ..resources()
                    },
                    MatMulResources {
                        threads_per_sm: 511,
                        ..resources()
                    },
                ] {
                    assert!(!LargeBf16TilePolicy::configured_candidate_fits(
                        reused,
                        constrained
                    ));
                }
                let sync = candidate
                    .with_shared_layout(MatMulSharedLayout::Padded, resources())
                    .unwrap()
                    .with_pipeline(MatMulPipeline::Synchronous, true, true, resources());
                assert!(!LargeBf16TilePolicy::configured_candidate_fits(
                    sync,
                    resources()
                ));
            }
        }
    }
    #[test]
    fn ordinary_false_reuse_resource_guards_and_known_schedules_remain_unchanged() {
        let original = ordinary(4096, 512, 512);
        let row = original
            .with_operand_orientations(MatMulSharedLayout::Padded, (false, false), resources())
            .unwrap()
            .with_pipeline(MatMulPipeline::Auto, true, true, resources());
        let column = original
            .with_operand_orientations(MatMulSharedLayout::Padded, (false, true), resources())
            .unwrap()
            .with_pipeline(MatMulPipeline::Auto, true, true, resources());
        assert!(!row.reuse_shared_memory && !column.reuse_shared_memory);
        assert_eq!(row.pipeline_stages, 2);
        assert_eq!(column.pipeline_stages, 2);
        assert_eq!(row.shared_mem_bytes(), 35840);
        assert_eq!(column.shared_mem_bytes(), 36864);
        assert_eq!(row.with_shared_memory_reuse(true).shared_mem_bytes(), 19456);
        assert_eq!(
            column.with_shared_memory_reuse(true).shared_mem_bytes(),
            20480
        );
        for candidate in LargeBf16TilePolicy::TallThenWide.candidates(original, &[4096, 512], true)
        {
            let unavailable = candidate
                .with_shared_layout(MatMulSharedLayout::Auto, resources())
                .unwrap()
                .with_pipeline(MatMulPipeline::DoubleBuffered, false, true, resources());
            assert!(!LargeBf16TilePolicy::configured_candidate_fits(
                unavailable,
                resources()
            ));
        }
    }
}
