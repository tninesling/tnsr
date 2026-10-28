#![cfg(feature = "cuda")]
use std::collections::HashMap;
use tnsr::{
    Executor,
    graph::TensorGraph,
    ptx::{PtxExecutor, PtxFusionPolicy},
    tensor::TensorExpr,
    tile::{MatMulPipeline, MatMulPrecision, MatMulSharedLayout},
};
fn executor() -> Option<PtxExecutor> {
    match PtxExecutor::try_new() {
        Ok(mut e) => {
            e.set_matmul_precision(MatMulPrecision::StrictF32);
            Some(e)
        }
        Err(error)
            if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                && error
                    .to_string()
                    .starts_with("Failed to initialize CUDA device 0") =>
        {
            None
        }
        Err(error) => panic!("{error:#}"),
    }
}
#[test]
fn ordered_f32_matmul_covers_changed_feeds_views_layouts_and_tails() {
    let Some(mut e) = executor() else { return };
    for (case, (batch, m, n, k)) in [
        (1, 3, 5, 7),
        (2, 17, 19, 15),
        (2, 31, 33, 17),
        (2, 17, 19, 65),
        (1, 32, 48, 128),
        (1, 64, 64, 256),
        (2, 128, 256, 128),
        (1, 129, 257, 129),
    ]
    .into_iter()
    .enumerate()
    {
        for ta in [false, true] {
            for tb in [false, true] {
                let a = TensorExpr::<f32>::input(
                    "a",
                    if ta {
                        vec![batch, k, m]
                    } else {
                        vec![batch, m, k]
                    },
                );
                let b = TensorExpr::<f32>::input("b", if tb { vec![n, k] } else { vec![k, n] });
                let lhs = if ta { a.transpose() } else { a };
                let rhs = if tb { b.transpose() } else { b };
                let expression = lhs.matmul(rhs);
                let expression = if case % 2 == 0 {
                    (expression + TensorExpr::input("bias", vec![n])).relu()
                } else {
                    expression
                };
                let graph: TensorGraph<f32> = expression.into();
                for layout in [
                    MatMulSharedLayout::Contiguous,
                    MatMulSharedLayout::Padded,
                    MatMulSharedLayout::Swizzled,
                ] {
                    for pipeline in [MatMulPipeline::Synchronous, MatMulPipeline::DoubleBuffered] {
                        e.set_matmul_shared_layout(layout);
                        e.set_matmul_pipeline(pipeline);
                        e.set_fusion_policy(if case % 2 == 0 {
                            PtxFusionPolicy::CostAware
                        } else {
                            PtxFusionPolicy::Greedy
                        });
                        e.compile(&graph).unwrap();
                        for feed in 0..3 {
                            let a: Vec<f32> = (0..batch * m * k)
                                .map(|i| (((i * 17 + feed * 13) % 101) as f32 - 50.0) / 37.0)
                                .collect();
                            let b: Vec<f32> = (0..k * n)
                                .map(|i| (((i * 23 + feed * 7) % 97) as f32 - 48.0) / 53.0)
                                .collect();
                            let bias: Vec<f32> = (0..n)
                                .map(|i| ((i + feed) % 11) as f32 / 29.0 - 0.17)
                                .collect();
                            let mut inputs = HashMap::from([
                                ("a".to_string(), a.clone()),
                                ("b".to_string(), b.clone()),
                            ]);
                            if case % 2 == 0 {
                                inputs.insert("bias".to_string(), bias.clone());
                            }
                            let actual = e.execute_compiled(&graph, inputs).unwrap();
                            // RAW_CAPTURE_FINITE
                            for z in 0..batch {
                                for row in 0..m {
                                    for col in 0..n {
                                        let expected = (0..k)
                                            .map(|j| {
                                                let ai = z * m * k
                                                    + if ta { j * m + row } else { row * k + j };
                                                let bi = if tb { col * k + j } else { j * n + col };
                                                f64::from(a[ai]) * f64::from(b[bi])
                                            })
                                            .sum::<f64>();
                                        let expected = if case % 2 == 0 {
                                            (expected + f64::from(bias[col])).max(0.0)
                                        } else {
                                            expected
                                        };
                                        let value = f64::from(actual[(z * m + row) * n + col]);
                                        assert!(
                                            (value - expected).abs()
                                                <= 3e-5 + 3e-4 * expected.abs(),
                                            "case={case},transpose={ta}/{tb},layout={layout:?},pipeline={pipeline:?},feed={feed}: {value}!={expected}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn ordered_f32_matmul_preserves_cancellation_order_across_k_tiles() {
    let Some(mut e) = executor() else { return };
    for k in [16, 17, 63, 64, 65, 128, 129, 257] {
        let (m, n) = (129, 257);
        let graph: TensorGraph<f32> = TensorExpr::input("a", vec![m, k])
            .matmul(TensorExpr::input("b", vec![k, n]))
            .into();
        let a = vec![1.0_f32; m * k];
        let b: Vec<f32> = (0..k * n).map(|i| [1e8, 1.0, -1e8][(i / n) % 3]).collect();
        let expected = (0..k).fold(0.0_f32, |sum, j| sum + b[j * n]);
        let actual = e
            .execute(
                &graph,
                HashMap::from([("a".to_string(), a), ("b".to_string(), b)]),
            )
            .unwrap();
        // RAW_CAPTURE_CANCELLATION
        assert!(
            actual.iter().all(|v| v.to_bits() == expected.to_bits()),
            "K={k},expected={expected}"
        );
    }
}
#[test]
fn ordered_f32_matmul_preserves_nonfinite_classes() {
    let Some(mut e) = executor() else { return };
    for k in [17, 65, 129] {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let (m, n) = (129, 257);
            let graph: TensorGraph<f32> = TensorExpr::input("a", vec![m, k])
                .matmul(TensorExpr::input("b", vec![k, n]))
                .into();
            let mut a = vec![1.0_f32; m * k];
            a[0] = value;
            let b = vec![1.0_f32; k * n];
            let actual = e
                .execute(
                    &graph,
                    HashMap::from([("a".to_string(), a), ("b".to_string(), b)]),
                )
                .unwrap();
            // RAW_CAPTURE_NONFINITE
            for (i, &v) in actual.iter().enumerate() {
                if i < n {
                    if value.is_nan() {
                        assert!(v.is_nan());
                    } else {
                        assert_eq!(v, value);
                    }
                } else {
                    assert_eq!(v, k as f32);
                }
            }
        }
    }
}
