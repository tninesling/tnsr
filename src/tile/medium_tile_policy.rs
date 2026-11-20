//! Opt-in bounded BF16 reuse tiles selected BEFORE ordinary shared-layout
//! and pipeline policy. Default scheduling remains unchanged.
//!
//! Pass the logical output shape, not a physical input's flattened/batched
//! storage shape. Initial scope is exactly rank two, so batched attention and
//! existing rank-three split-K partials retain their schedules. The specialized
//! mixed split-K path must continue bypassing this policy entirely.
//!
//! After selection, callers MUST rerun existing layout/orientation/copy proofs
//! and pipeline/resource checks, then apply scratch reuse. Do not inherit old
//! 16x32 or 32x32 strides, async stage descriptors, or a prior reuse allocation.
//! Threshold128 reproduces the old selector's minimum 64x64 grid, not measured
//! occupancy. Count actual final square/tall schedules in Program diagnostics;
//! requested policy flags alone do not prove activation. PtxExecutor, Mixed
//! and Program propagate an Off-default policy with a distinct cache identity.
#[cfg(any(feature = "cuda", test))]
use super::{
    DType, MatMulAccumulatorLayout, MatMulPlan, MatMulResources, MatMulSchedule, MatMulTile,
    SharedLayout,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MediumBf16TilePolicy {
    #[default]
    Off,
    SquareOnly,
    SquareThenTall,
}

#[cfg(any(feature = "cuda", test))]
impl MediumBf16TilePolicy {
    pub(crate) fn select(
        self,
        original: MatMulSchedule,
        logical_output_shape: &[usize],
        resources: MatMulResources,
    ) -> MatMulSchedule {
        let shape = original.logical_shape;
        if self == Self::Off
            || logical_output_shape != [shape.m, shape.n]
            || original.plan != MatMulPlan::TensorCoreBF16
            || original.storage_dtype != DType::BF16
            || original.operand_dtype != DType::BF16
            || original.instruction_tile
                != (MatMulTile {
                    m: 16,
                    n: 16,
                    k: 16,
                })
            || shape.m < 128
            || shape.n < 256
            || shape.k < 64
            || !shape.k.is_multiple_of(32)
            || !shape.m.is_multiple_of(64)
            || original.block_tile
                == (MatMulTile {
                    m: 64,
                    n: 64,
                    k: 32,
                })
        {
            return original;
        }
        // Only the ordinary smaller families are candidates. A custom or
        // already-expanded schedule must not silently acquire another policy.
        if !matches!(
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
        ) {
            return original;
        }
        let allowed = |n: usize| {
            if !shape.n.is_multiple_of(n) {
                return false;
            }
            let Some(ctas) = shape.m.div_ceil(64).checked_mul(shape.n.div_ceil(n)) else {
                return false;
            };
            if ctas < 128 {
                return false;
            }
            // Same disjoint, unpadded conservative initial budget as ordinary
            // selection; padding and double buffers are checked afterward.
            let shared = (64 * 32 + 32 * n) * 2 + 64 * n * 4;
            let blocks = (resources.threads_per_sm / 256)
                .min(resources.registers_per_sm / (128 * 256))
                .min(resources.shared_bytes_per_sm / shared);
            shared <= resources.shared_bytes_per_block && blocks >= 2
        };
        let n = if allowed(64) {
            64
        } else if self == Self::SquareThenTall && allowed(32) {
            32
        } else {
            return original;
        };
        let mut selected = original;
        selected.block_tile = MatMulTile { m: 64, n, k: 32 };
        selected.block_threads = (32, 8, 1);
        selected.warp_topology = (2, 2);
        selected.warp_fragments = if n == 64 { (2, 2) } else { (2, 1) };
        selected.accumulator_layout = MatMulAccumulatorLayout::WarpFragment;
        selected.operand_layouts = (SharedLayout::row_major(32), SharedLayout::row_major(n));
        selected.pipeline_stages = 1;
        selected.copy_bytes = 4;
        selected.async_operands = (false, false);
        selected.reuse_shared_memory = false;
        selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::{MatMulCapabilities, MatMulPrecision};
    #[cfg(feature = "cuda")]
    use crate::{
        graph::{TensorGraph, TensorGraphNode},
        ptx::{PtxGraph, PtxTarget},
        tile::{MatMulPipeline, MatMulRegion, MatMulSharedLayout, TileGraph},
    };
    #[cfg(feature = "cuda")]
    use anyhow::Result;

    fn resources() -> MatMulResources {
        MatMulResources {
            shared_bytes_per_sm: 100 * 1024,
            ..MatMulResources::default()
        }
    }
    fn base(m: usize, n: usize, k: usize) -> MatMulSchedule {
        MatMulSchedule::select_with_resources(
            m,
            n,
            k,
            DType::BF16,
            MatMulCapabilities {
                tf32: true,
                bf16: true,
                f16: true,
            },
            MatMulPrecision::NativeHalf,
            resources(),
        )
    }
    #[test]
    fn medium_policy_scope_grid_and_fallbacks_are_bounded() {
        assert_eq!(MediumBf16TilePolicy::default(), MediumBf16TilePolicy::Off);
        for (m, n, k, square, tall) in [
            (128, 5504, 2048, true, true),
            (128, 6144, 2048, true, true),
            (128, 4096, 2048, true, true),
            (128, 2048, 2048, false, true),
            (256, 2048, 128, true, true),
            (128, 256, 2048, false, false),
            (129, 4096, 2048, false, false),
            (128, 4096, 65, false, false),
            (128, 4096, 32, false, false),
            (128, 4128, 2048, false, true),
        ] {
            let old = base(m, n, k);
            assert_eq!(
                MediumBf16TilePolicy::Off.select(old, &[m, n], resources()),
                old
            );
            for (policy, active) in [
                (MediumBf16TilePolicy::SquareOnly, square),
                (MediumBf16TilePolicy::SquareThenTall, tall),
            ] {
                let new = policy.select(old, &[m, n], resources());
                assert_eq!(new != old, active);
                if active {
                    assert_eq!(
                        new.block_tile,
                        MatMulTile {
                            m: 64,
                            n: if square { 64 } else { 32 },
                            k: 32
                        }
                    );
                    assert_eq!(new.instruction_tile, old.instruction_tile);
                    assert_eq!(new.logical_shape, old.logical_shape);
                    assert_eq!(new.block_threads, (32, 8, 1));
                }
                assert_eq!(policy.select(old, &[1, m, n], resources()), old);
                assert_eq!(policy.select(old, &[4, m, n], resources()), old);
            }
        }
        let large = base(4096, 1408, 512);
        assert_eq!(
            MediumBf16TilePolicy::SquareThenTall.select(large, &[4096, 1408], resources()),
            large
        );
        for plan in [
            MatMulPlan::ScalarF32,
            MatMulPlan::TensorCoreF16,
            MatMulPlan::TensorCoreTf32x3,
        ] {
            let mut old = base(128, 5504, 2048);
            old.plan = plan;
            assert_eq!(
                MediumBf16TilePolicy::SquareThenTall.select(old, &[128, 5504], resources()),
                old
            );
        }
        for budget in [
            MatMulResources {
                shared_bytes_per_block: 16 * 1024,
                ..resources()
            },
            MatMulResources {
                shared_bytes_per_sm: 32 * 1024,
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
            let old = base(128, 5504, 2048);
            assert_eq!(
                MediumBf16TilePolicy::SquareOnly.select(old, &[128, 5504], budget),
                old
            );
        }
        let mut huge = base(128, 5504, 2048);
        huge.logical_shape.m = usize::MAX - 63;
        huge.logical_shape.n = usize::MAX - 63;
        assert_eq!(
            MediumBf16TilePolicy::SquareThenTall.select(
                huge,
                &[huge.logical_shape.m, huge.logical_shape.n],
                resources()
            ),
            huge
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn medium_square_and_tall_use_ordinary_layout_pipeline_and_codegen() -> Result<()> {
        for n in [5504, 6144, 4096, 2048] {
            for transpose_b in [false, true] {
                let mut graph = TensorGraph::<half::bf16>::new();
                let a = graph.graph.add_node(TensorGraphNode::Input {
                    name: "a",
                    shape: vec![128, 2048],
                });
                let b = graph.graph.add_node(TensorGraphNode::Input {
                    name: "b",
                    shape: vec![2048, n],
                });
                let out = graph.graph.add_node(TensorGraphNode::MatMul {
                    shape: vec![128, n],
                });
                graph.graph.add_edge(a, out, 0);
                graph.graph.add_edge(b, out, 1);
                let mut region = MatMulRegion::from_graph(&graph, out, vec![], vec![out])?;
                if transpose_b {
                    region.inputs[1].source_shape = vec![n, 2048];
                    region.inputs[1].tensor =
                        crate::tile::VirtualTensor::identity(b, vec![n, 2048], DType::BF16)
                            .permute(&[1, 0])?;
                }
                region.schedule = MediumBf16TilePolicy::SquareThenTall.select(
                    base(128, n, 2048),
                    &[128, n],
                    resources(),
                );
                let before = region.schedule;
                assert_eq!(before.pipeline_stages, 1);
                assert_eq!(before.async_operands, (false, false));
                region.configure_pipeline(
                    MatMulSharedLayout::Auto,
                    MatMulPipeline::Auto,
                    true,
                    resources(),
                )?;
                assert_eq!(region.schedule.pipeline_stages, 2);
                assert_eq!(region.schedule.copy_bytes, 16);
                assert!(region.schedule.shared_mem_bytes() <= resources().shared_bytes_per_block);
                assert!(resources().shared_bytes_per_sm / region.schedule.shared_mem_bytes() >= 2);
                let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
                tiles.add_matmul_regions(&[region.clone()])?;
                tiles.optimize_indices()?;
                for ir in tiles.graph.node_weights() {
                    ir.validate_layouts()?;
                }
                let source = PtxGraph::from(tiles)
                    .with_target(PtxTarget::sm80())
                    .module_source();
                let square = n != 2048;
                assert_eq!(
                    source.matches("wmma.mma").count(),
                    if square { 8 } else { 4 }
                );
                assert!(source.contains("cp.async"));
                assert!(source.contains(&format!(
                    ".shared .align 32 .b8 shared[{}];",
                    region.schedule.shared_mem_bytes()
                )));
            }
        }
        Ok(())
    }
}
