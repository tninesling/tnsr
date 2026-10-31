#![cfg(feature = "cuda")]

use std::collections::HashMap;
use tnsr::{
    graph::TensorGraph,
    ptx::{PtxExecutor, PtxPlanAction},
    tensor::TensorExpr,
    tile::{MatMulPipeline, MatMulPrecision, MatMulSharedLayout},
};

fn executors() -> Option<(PtxExecutor, PtxExecutor)> {
    let baseline = match PtxExecutor::try_new() {
        Ok(executor) => executor,
        Err(error)
            if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                && error
                    .to_string()
                    .starts_with("Failed to initialize CUDA device 0") =>
        {
            return None;
        }
        Err(error) => panic!("{error:#}"),
    };
    let mut candidate = PtxExecutor::try_new().unwrap();
    let mut baseline = baseline;
    baseline.set_matmul_precision(MatMulPrecision::StrictF32);
    candidate.set_matmul_precision(MatMulPrecision::StrictF32);
    candidate.set_experimental_register_tiles(true);
    Some((baseline, candidate))
}

fn assert_selected(executor: &PtxExecutor, eligible: bool) {
    let plan = executor.execution_plan().unwrap();
    let step = plan
        .steps()
        .iter()
        .find(|step| {
            matches!(
                step.action,
                PtxPlanAction::Kernel | PtxPlanAction::MatMulRegion(_)
            )
        })
        .unwrap();
    let schedule = match step.action {
        PtxPlanAction::MatMulRegion(id) => plan.matmul_regions()[id].schedule,
        _ => plan.matmul_schedules()[&step.node],
    };
    assert_eq!(schedule.register_thread_tile() == Some((4, 4)), eligible);
}

#[test]
fn outer_product_register_tile_matches_every_output_bit_for_views_tails_and_epilogues() {
    let Some((mut baseline, mut candidate)) = executors() else {
        return;
    };
    let (batch, m, n, k) = (2, 1025, 257, 129);
    for transpose_a in [false, true] {
        for transpose_b in [false, true] {
            let a = TensorExpr::input(
                "a",
                if transpose_a {
                    vec![batch, k, m]
                } else {
                    vec![batch, m, k]
                },
            );
            let b = TensorExpr::input("b", if transpose_b { vec![n, k] } else { vec![k, n] });
            let a = if transpose_a {
                a.reshape(vec![batch, k, m]).transpose()
            } else {
                a
            };
            let b = if transpose_b { b.transpose() } else { b };
            let product = a.matmul(b);
            // Exercise both plain output stores and bias/activation extraction.
            let graph: TensorGraph<f32> = if transpose_a {
                ((product + TensorExpr::input("bias", vec![n]))
                    * TensorExpr::input("scale", vec![n]))
                .relu()
                .into()
            } else {
                product.into()
            };
            for layout in [
                MatMulSharedLayout::Contiguous,
                MatMulSharedLayout::Padded,
                MatMulSharedLayout::Swizzled,
            ] {
                for pipeline in [MatMulPipeline::Synchronous, MatMulPipeline::DoubleBuffered] {
                    for executor in [&mut baseline, &mut candidate] {
                        executor.set_matmul_shared_layout(layout);
                        executor.set_matmul_pipeline(pipeline);
                        executor.compile(&graph).unwrap();
                    }
                    assert_selected(&baseline, false);
                    assert_selected(&candidate, true);
                    for feed in 0..2 {
                        let a = (0..batch * m * k)
                            .map(|i| (((i * 17 + feed * 13) % 101) as f32 - 50.0) / 37.0)
                            .collect();
                        let b = (0..k * n)
                            .map(|i| (((i * 23 + feed * 7) % 97) as f32 - 48.0) / 53.0)
                            .collect();
                        let mut inputs =
                            HashMap::from([("a".to_string(), a), ("b".to_string(), b)]);
                        if transpose_a {
                            inputs.insert(
                                "bias".to_string(),
                                (0..n).map(|i| (i % 11) as f32 / 29.0 - 0.17).collect(),
                            );
                            inputs.insert(
                                "scale".to_string(),
                                (0..n).map(|i| 0.5 + (i % 7) as f32 / 13.0).collect(),
                            );
                        }
                        let expected = baseline.execute_compiled(&graph, inputs.clone()).unwrap();
                        let actual = candidate.execute_compiled(&graph, inputs).unwrap();
                        assert_eq!(actual.len(), expected.len());
                        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate()
                        {
                            assert_eq!(
                                actual.to_bits(),
                                expected.to_bits(),
                                "index={index},ta={transpose_a},tb={transpose_b},layout={layout:?},pipeline={pipeline:?},feed={feed}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn outer_product_register_tile_preserves_cancellation_nonfinite_and_small_grid_fallbacks() {
    let Some((mut baseline, mut candidate)) = executors() else {
        return;
    };
    for (m, n, k) in [
        (1024, 256, 128),
        (1025, 257, 129),
        (1024, 257, 143),
        (1025, 256, 144),
        (1024, 257, 145),
        (1025, 257, 257),
        (683, 256, 512),
        (32, 256, 128),
    ] {
        let graph: TensorGraph<f32> = TensorExpr::input("a", vec![m, k])
            .matmul(TensorExpr::input("b", vec![k, n]))
            .into();
        for executor in [&mut baseline, &mut candidate] {
            executor.compile(&graph).unwrap();
        }
        assert_selected(&candidate, m >= 1024);
        for nonfinite in [false, true] {
            let mut a = vec![1.0_f32; m * k];
            let b: Vec<_> = (0..k * n).map(|i| [1e8, 1.0, -1e8][(i / n) % 3]).collect();
            if nonfinite {
                a[0] = f32::NAN;
                a[k] = f32::INFINITY;
                a[2 * k] = f32::NEG_INFINITY;
                a[3 * k..4 * k].fill(-0.0);
            }
            let inputs = HashMap::from([("a".to_string(), a), ("b".to_string(), b)]);
            let expected = baseline.execute_compiled(&graph, inputs.clone()).unwrap();
            let actual = candidate.execute_compiled(&graph, inputs).unwrap();
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    actual.to_bits() == expected.to_bits()
                        || (actual.is_nan() && expected.is_nan()),
                    "{m}x{n}x{k},index={index},nonfinite={nonfinite}: {actual} != {expected}"
                );
            }
            if !nonfinite {
                let ordered = (0..k).fold(0.0_f32, |sum, j| sum + [1e8, 1.0, -1e8][j % 3]);
                assert!(
                    actual
                        .iter()
                        .all(|value| value.to_bits() == ordered.to_bits())
                );
            }
        }
    }
}
