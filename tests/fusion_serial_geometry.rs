//! Serial launch choices distribute independent fibers without changing sums.
use tnsr::{
    graph::{TensorGraph, TensorGraphNode},
    tensor::TensorExpr,
    tile::{ReductionRegion, ReductionSchedule},
};

fn region() -> ReductionRegion {
    let graph: TensorGraph<f32> = TensorExpr::constant(vec![1.0; 3 * 7], vec![3, 7])
        .reduce_sum(1)
        .into();
    let anchor = graph
        .toposort()
        .into_iter()
        .find(|&node| matches!(graph[node], TensorGraphNode::ReduceAxis { .. }))
        .unwrap();
    ReductionRegion::from_graph(&graph, vec![], anchor, vec![], vec![], vec![], vec![anchor])
        .unwrap()
}

#[test]
fn serial_geometry_is_validated_without_changing_reduction_policy() {
    let reference = region();
    assert_eq!(reference.serial_block_threads, 1024);
    for threads in [32, 64, 96, 128, 256, 512, 1024] {
        let changed = reference
            .clone()
            .with_serial_block_threads(threads)
            .unwrap();
        assert_eq!(changed.schedule, ReductionSchedule::Serial);
        assert_eq!(changed.serial_block_threads, threads);
        assert_eq!(changed.producer_operations, reference.producer_operations);
        assert_eq!(changed.epilogue_operations, reference.epilogue_operations);
        changed.lower_to_tile_ir(0).unwrap();
    }
    for threads in [0, 1, 31, 33, 1025, u32::MAX] {
        assert!(
            reference
                .clone()
                .with_serial_block_threads(threads)
                .is_err()
        );
    }
    let mut invalid = reference;
    invalid.serial_block_threads = 0;
    assert!(invalid.lower_to_tile_ir(0).is_err());
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use std::collections::HashMap;
    use tnsr::{
        Executor,
        ptx::{PtxExecutor, PtxFusionPolicy, PtxReductionMode},
    };

    fn executor() -> Option<PtxExecutor> {
        match PtxExecutor::try_new_with_reduction_mode(PtxReductionMode::Strict) {
            Ok(mut executor) => {
                executor.set_fusion_policy(PtxFusionPolicy::Greedy);
                Some(executor)
            }
            Err(error)
                if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                    && error
                        .to_string()
                        .starts_with("Failed to initialize CUDA device 0") =>
            {
                eprintln!("skipping CUDA test: {error:#}");
                None
            }
            Err(error) => panic!("CUDA initialization: {error:#}"),
        }
    }

    fn assert_bits(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn serial_geometries_preserve_cancellation_across_axes_and_output_tails() {
        let Some(mut executor) = executor() else {
            return;
        };
        // Serial is exactly 1, while other reduction associations can produce 2.
        let row = [16_777_216.0f32, 1.0, -16_777_216.0, 1.0];
        for fibers in [1, 33, 191, 257] {
            for axis in [0, 1] {
                let (shape, data) = if axis == 0 {
                    (
                        vec![4, fibers],
                        row.iter().flat_map(|&value| vec![value; fibers]).collect(),
                    )
                } else {
                    (vec![fibers, 4], row.repeat(fibers))
                };
                let input = TensorExpr::constant(data, shape);
                let graph: TensorGraph<f32> = (-(-input)).reduce_sum(axis).into();
                executor
                    .set_serial_reduction_block_threads(Some(1024))
                    .unwrap();
                let expected = executor.execute(&graph, HashMap::new()).unwrap();
                assert_eq!(expected, vec![1.0; fibers]);
                for threads in [32, 64, 128, 256, 512, 1024] {
                    executor
                        .set_serial_reduction_block_threads(Some(threads))
                        .unwrap();
                    let actual = executor.execute(&graph, HashMap::new()).unwrap();
                    assert_bits(&actual, &expected);
                    let regions = executor.execution_plan().unwrap().reduction_regions();
                    assert!(!regions.is_empty(), "must exercise region launch override");
                    assert!(
                        regions
                            .iter()
                            .all(|region| region.schedule == ReductionSchedule::Serial
                                && region.serial_block_threads == threads)
                    );
                }
            }
        }
    }

    #[test]
    fn serial_geometries_preserve_full_epilogues_and_special_values() {
        let Some(mut executor) = executor() else {
            return;
        };
        for fibers in [33, 95] {
            let values = (0..fibers)
                .flat_map(|fiber| match fiber % 3 {
                    0 => [-0.0f32, 0.0, -2.0, f32::INFINITY, f32::NAN],
                    1 => [16_777_216.0, 1.0, -16_777_216.0, 1.0, 0.0],
                    _ => [0.1, -0.2, 0.3, -0.4, 0.5],
                })
                .collect();
            let input = TensorExpr::constant(values, vec![fibers, 5]);
            let sum = input.clone().reduce_sum(1);
            let expr = input - sum.broadcast(vec![fibers, 5]);
            let graph: TensorGraph<f32> = expr.into();
            executor
                .set_serial_reduction_block_threads(Some(1024))
                .unwrap();
            let expected = executor.execute(&graph, HashMap::new()).unwrap();
            assert!(
                executor
                    .execution_plan()
                    .unwrap()
                    .reduction_regions()
                    .iter()
                    .any(|region| !region.full_epilogue_operations.is_empty())
            );
            for threads in [32, 64, 128, 256, 512, 1024] {
                executor
                    .set_serial_reduction_block_threads(Some(threads))
                    .unwrap();
                let actual = executor.execute(&graph, HashMap::new()).unwrap();
                assert_bits(&actual, &expected);
            }
        }
    }

    #[test]
    fn automatic_standalone_reductions_preserve_legacy_strict_bits() {
        let Some(mut gpu) = executor() else {
            return;
        };
        for axis in [0, 1] {
            for mean in [false, true] {
                let row: Vec<f32> = (0..192)
                    .map(|i| match i {
                        0 => 16_777_216.0,
                        1 | 3 => 1.0,
                        2 => -16_777_216.0,
                        _ => 0.0,
                    })
                    .collect();
                let data = if axis == 0 {
                    row.iter().flat_map(|&x| vec![x; 192]).collect()
                } else {
                    row.repeat(192)
                };
                let input = TensorExpr::constant(data, vec![192, 192]);
                let graph: TensorGraph<f32> = if mean {
                    input.reduce_mean(axis).into()
                } else {
                    input.reduce_sum(axis).into()
                };
                gpu.set_fusion_policy(PtxFusionPolicy::Greedy);
                let expected = gpu.execute(&graph, HashMap::new()).unwrap();
                assert!(gpu.execution_plan().unwrap().reduction_regions().is_empty());
                gpu.set_fusion_policy(PtxFusionPolicy::CostAware);
                let actual = gpu.execute(&graph, HashMap::new()).unwrap();
                assert_bits(&actual, &expected);
                let regions = gpu.execution_plan().unwrap().reduction_regions();
                assert!(
                    !regions.is_empty(),
                    "large strict anchor should have geometry alternatives"
                );
                assert!(
                    regions
                        .iter()
                        .all(|r| r.schedule == ReductionSchedule::Serial)
                );
            }
        }
        let graph: TensorGraph<f32> = TensorExpr::constant(Vec::new(), vec![0, 4])
            .reduce_mean(0)
            .into();
        let actual = gpu.execute(&graph, HashMap::new()).unwrap();
        assert!(actual.iter().all(|v| v.is_nan()));
        assert!(gpu.execution_plan().unwrap().reduction_regions().is_empty());
    }

    #[test]
    fn serial_geometry_preserves_bf16_storage_rounding() {
        use half::bf16;
        use tnsr::ptx::BF16;
        let Some(_) = executor() else {
            return;
        };
        let data = (0..37 * 17)
            .map(|i| bf16::from_f32((i % 13) as f32 / 16.0 - 0.25))
            .collect();
        let input = TensorExpr::constant(data, vec![37, 17]);
        let graph: TensorGraph<bf16> = (-(-input)).reduce_sum(1).into();
        let mut gpu = PtxExecutor::<BF16>::try_new_for_dtype().unwrap();
        gpu.set_serial_reduction_block_threads(Some(1024)).unwrap();
        let expected = gpu.execute(&graph, HashMap::new()).unwrap();
        for threads in [32, 64, 128, 256, 512] {
            gpu.set_serial_reduction_block_threads(Some(threads))
                .unwrap();
            let actual = gpu.execute(&graph, HashMap::new()).unwrap();
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            assert!(!gpu.execution_plan().unwrap().reduction_regions().is_empty());
        }
    }

    #[test]
    fn serial_geometry_override_invalidates_compilation_and_can_be_cleared() {
        let Some(mut executor) = executor() else {
            return;
        };
        let input = TensorExpr::constant(vec![0.5f32; 12], vec![3, 4]);
        let graph: TensorGraph<f32> = (-(-input)).reduce_sum(1).into();
        executor
            .set_serial_reduction_block_threads(Some(32))
            .unwrap();
        executor.execute(&graph, HashMap::new()).unwrap();
        let first = executor.compilation_count();
        executor
            .set_serial_reduction_block_threads(Some(64))
            .unwrap();
        assert!(executor.execute_compiled(&graph, HashMap::new()).is_err());
        executor.execute(&graph, HashMap::new()).unwrap();
        assert_eq!(executor.compilation_count(), first + 1);
        executor
            .set_serial_reduction_block_threads(Some(64))
            .unwrap();
        executor.execute(&graph, HashMap::new()).unwrap();
        assert_eq!(executor.compilation_count(), first + 1);
        assert!(
            executor
                .set_serial_reduction_block_threads(Some(0))
                .is_err()
        );
        executor.execute_compiled(&graph, HashMap::new()).unwrap();
        executor.set_serial_reduction_block_threads(None).unwrap();
        assert!(executor.execute_compiled(&graph, HashMap::new()).is_err());
        executor.execute(&graph, HashMap::new()).unwrap();
        assert_eq!(executor.compilation_count(), first + 2);
        assert!(
            executor
                .execution_plan()
                .unwrap()
                .reduction_regions()
                .iter()
                .all(|region| ReductionRegion::validate_serial_block_threads(
                    region.serial_block_threads
                )
                .is_ok())
        );
    }
}
