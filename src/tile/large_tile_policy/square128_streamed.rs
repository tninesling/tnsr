//! Initial square mode: plain ordinary GEMM only. No epilogue arithmetic,
//! batched operands, Split-K, or replacement of selected medium geometries.
use crate::tile::{
    DType, MatMulAccumulatorExchange, MatMulFragmentLoad, MatMulPlan, MatMulResources,
    MatMulSchedule, MatMulTile,
};
#[cfg(feature = "cuda")]
use crate::tile::{MatMulPipeline, MatMulRegion, MatMulSharedLayout};

pub(crate) fn geometry(
    base: MatMulSchedule,
    output: &[usize],
    resources: MatMulResources,
) -> Option<MatMulSchedule> {
    let s = base.logical_shape;
    if output != [s.m, s.n]
        || resources.shared_bytes_per_sm < 80 * 1024
        || base.plan != MatMulPlan::TensorCoreBF16
        || base.storage_dtype != DType::BF16
        || base.operand_dtype != DType::BF16
        || base.block_tile
            != (MatMulTile {
                m: 64,
                n: 64,
                k: 32,
            })
        || !matches!(base.block_threads, (32, 4, 1) | (32, 8, 1))
        || base.warp_topology != (2, 2)
        || base.warp_fragments != (2, 2)
        || base.accumulator_exchange != MatMulAccumulatorExchange::FullTile
        || base.fragment_load != MatMulFragmentLoad::PreloadAll
        || base.pipeline_stages != 2
        || base.async_operands != (true, true)
        || base.copy_bytes != 16
        || !base.reuse_shared_memory
        || s.m < 2048
        || s.n < 256
        || s.k < 64
        || !s.m.is_multiple_of(128)
        || !s.n.is_multiple_of(128)
        || !s.k.is_multiple_of(32)
    {
        return None;
    }
    for (a, b) in [(s.m, s.k), (s.k, s.n), (s.m, s.n)] {
        if a.checked_mul(b)?.checked_mul(2)? > isize::MAX as usize {
            return None;
        }
    }
    let (rows, cols) = (s.m / 128, s.n / 128);
    if rows > 65535 || cols > i32::MAX as usize || rows.checked_mul(cols)? < 128 {
        return None;
    }
    base.bf16_square128_streamed_geometry()
}
pub(crate) fn configured_fits(s: MatMulSchedule, r: MatMulResources) -> bool {
    r.shared_bytes_per_sm >= 80 * 1024
        && s.is_square128_streamed()
        && s.storage_dtype == DType::BF16
        && s.operand_dtype == DType::BF16
        && s.pipeline_stages == 2
        && s.async_operands == (true, true)
        && s.copy_bytes == 16
        && s.reuse_shared_memory
        && s.accumulator_exchange_bytes() == 16 * 1024
        && s.conservative_two_block_fit(r)
}
#[cfg(feature = "cuda")]
pub(crate) fn region_candidate(
    region: &MatMulRegion,
    shared: MatMulSharedLayout,
    pipeline: MatMulPipeline,
    supports_async: bool,
    resources: MatMulResources,
) -> Option<MatMulSchedule> {
    if !region.batch_shape.is_empty()
        || region.schedule.logical_shape
            != (MatMulTile {
                m: region.m,
                n: region.n,
                k: region.k,
            })
        || region.output_shape != [region.m, region.n]
        || region.lhs_shape != [region.m, region.k]
        || region.rhs_shape != [region.k, region.n]
        || !region.epilogue_operations.is_empty()
        || region.outputs.len() != 1
        || region.outputs[0].value != region.matmul_value
    {
        return None;
    }
    let selected = geometry(region.schedule, &region.output_shape, resources)?;
    for (value, shape, orientation) in [
        (
            region.lhs_value,
            &region.lhs_shape,
            region.schedule.operand_layouts.0.column_major,
        ),
        (
            region.rhs_value,
            &region.rhs_shape,
            region.schedule.operand_layouts.1.column_major,
        ),
    ] {
        let input = region.inputs.iter().find(|input| input.value == value)?;
        crate::tile::ordinary_bf16_block::aligned_operand(input, shape, orientation)?;
    }
    let mut probe = region.clone();
    probe.schedule = selected;
    probe
        .configure_pipeline(shared, pipeline, supports_async, resources)
        .ok()?;
    if (
        probe.schedule.operand_layouts.0.column_major,
        probe.schedule.operand_layouts.1.column_major,
    ) != (
        region.schedule.operand_layouts.0.column_major,
        region.schedule.operand_layouts.1.column_major,
    ) {
        return None;
    }
    configured_fits(probe.schedule, resources).then_some(probe.schedule)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::{
        MatMulAccumulatorLayout, MatMulCapabilities, MatMulPipeline, MatMulPrecision,
        MatMulSharedLayout, builder::TileIRBuilder,
    };
    fn resources() -> MatMulResources {
        MatMulResources {
            shared_bytes_per_sm: 100 * 1024,
            threads_per_sm: 1536,
            ..MatMulResources::default()
        }
    }
    fn baseline() -> MatMulSchedule {
        MatMulSchedule::select_with_resources(
            4096,
            1024,
            512,
            DType::BF16,
            MatMulCapabilities {
                bf16: true,
                tf32: true,
                f16: true,
            },
            MatMulPrecision::NativeHalf,
            resources(),
        )
        .with_shared_memory_reuse(true)
        .with_shared_layout(MatMulSharedLayout::Padded, resources())
        .unwrap()
        .with_pipeline(MatMulPipeline::DoubleBuffered, true, true, resources())
    }
    #[test]
    fn square128_streamed_shared_phases_and_scope_are_bounded() {
        let original = baseline();
        let square = geometry(original, &[4096, 1024], resources()).unwrap();
        for (orientation, expected) in [
            ((false, false), 37 * 1024),
            ((false, true), 40 * 1024),
            ((true, false), 34 * 1024),
            ((true, true), 37 * 1024),
        ] {
            let s = square
                .with_operand_orientations(MatMulSharedLayout::Padded, orientation, resources())
                .unwrap()
                .with_operand_pipeline(
                    MatMulPipeline::DoubleBuffered,
                    true,
                    true,
                    (true, true),
                    resources(),
                );
            assert!(configured_fits(s, resources()));
            assert_eq!(s.shared_mem_bytes(), expected);
            assert_eq!(s.accumulator_exchange_shape(), Some((64, 64)));
            assert_eq!(s.accumulator_exchange_bytes(), 16 * 1024);
        }
        assert!(geometry(original, &[4096, 1024], MatMulResources::default()).is_none());
        assert!(geometry(original, &[1, 4096, 1024], resources()).is_none());
        assert!(
            geometry(
                original.with_experimental_bf16_eight_compute_warps(),
                &[4096, 1024],
                resources()
            )
            .is_none()
        );
        for bad in [
            MatMulSchedule {
                logical_shape: MatMulTile {
                    m: 4097,
                    ..original.logical_shape
                },
                ..original
            },
            MatMulSchedule {
                async_operands: (true, false),
                ..original
            },
            MatMulSchedule {
                copy_bytes: 4,
                ..original
            },
            MatMulSchedule {
                reuse_shared_memory: false,
                ..original
            },
        ] {
            assert!(
                geometry(
                    bad,
                    &[bad.logical_shape.m, bad.logical_shape.n],
                    resources()
                )
                .is_none()
            );
        }
        assert_eq!(
            original.accumulator_layout,
            MatMulAccumulatorLayout::WarpFragment
        );
    }
    #[test]
    fn square128_streamed_four_windows_cover_each_logical_element_once() {
        let mut covered = vec![0u8; 128 * 128];
        for origin in [(0, 0), (0, 2), (1, 0), (1, 2)] {
            let mut scratch = vec![None; 64 * 64];
            for wr in 0..4 {
                for wc in 0..2 {
                    for f in 0..2 {
                        for r in 0..16 {
                            for c in 0..16 {
                                let sr = wr * 16 + r;
                                let sc = wc * 32 + f * 16 + c;
                                assert!(
                                    scratch[sr * 64 + sc]
                                        .replace((
                                            wr * 32 + origin.0 * 16 + r,
                                            wc * 64 + (origin.1 + f) * 16 + c
                                        ))
                                        .is_none()
                                );
                            }
                        }
                    }
                }
            }
            for round in 0..16 {
                for thread in 0..256 {
                    let index = round * 256 + thread;
                    let (sr, sc) = (index / 64, index % 64);
                    let logical = (
                        sr / 16 * 32 + origin.0 * 16 + sr % 16,
                        sc / 32 * 64 + origin.1 * 16 + sc % 32,
                    );
                    assert_eq!(scratch[index], Some(logical));
                    covered[logical.0 * 128 + logical.1] += 1;
                }
            }
        }
        assert!(covered.iter().all(|&n| n == 1));
    }
    #[test]
    fn square128_streamed_window_layout_requires_exact_source_and_bounds() {
        let s = geometry(baseline(), &[4096, 1024], resources()).unwrap();
        let make = |origin, shape| {
            let mut b = TileIRBuilder::new();
            b.start_kernel("window_layout");
            let c = b.alloc_fragment(DType::F32, 128, 128, s);
            let shared = b.alloc_shared(DType::F32, 64, 64);
            b.store_accumulator_window(shared, c, origin, shape);
            b.finish()
        };
        assert!(make((0, 0), (1, 2)).validate_layouts().is_ok());
        for (origin, shape) in [
            ((2, 0), (1, 2)),
            ((0, 1), (1, 2)),
            ((0, 0), (0, 2)),
            ((0, 0), (2, 2)),
        ] {
            assert!(make(origin, shape).validate_layouts().is_err());
        }
        let mut b = TileIRBuilder::new();
        b.start_kernel("wrong_window_source");
        let c = b.alloc_register(DType::F32, 1, 1);
        let shared = b.alloc_shared(DType::F32, 64, 64);
        b.store_accumulator_window(shared, c, (0, 0), (1, 2));
        assert!(b.finish().validate_layouts().is_err());
        let mut b = TileIRBuilder::new();
        b.start_kernel("wrong_window_destination");
        let c = b.alloc_fragment(DType::F32, 128, 128, s);
        let shared = b.alloc_shared(DType::F32, 64, 128);
        b.store_accumulator_window(shared, c, (0, 0), (1, 2));
        assert!(b.finish().validate_layouts().is_err());
    }
}
