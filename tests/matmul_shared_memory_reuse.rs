#![cfg(feature = "cuda")]

use std::collections::HashMap;

use num_traits::{ToPrimitive, cast};
use tnsr::{
    graph::TensorGraph,
    ptx::{
        PtxExecutor,
        types::{BF16, CudaDType, F16, F32},
    },
    tensor::TensorExpr,
    tile::{MatMulPipeline, MatMulPrecision},
};

fn check<D: CudaDType>(precision: MatMulPrecision) {
    let mut executor = match PtxExecutor::<D>::try_new_for_dtype() {
        Ok(executor) => executor,
        Err(error)
            if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                && error
                    .to_string()
                    .starts_with("Failed to initialize CUDA device 0") =>
        {
            eprintln!("CUDA shared-memory qualification not run: {error:#}");
            return;
        }
        Err(error) => panic!("{error:#}"),
    };
    executor.set_matmul_precision(precision);
    for (m, n, k) in [(67, 81, 65), (2048, 256, 64), (32, 256, 0)] {
        for transpose_a in [false, true] {
            for transpose_b in [false, true] {
                let a = TensorExpr::<D::HostType>::input(
                    "a",
                    if transpose_a {
                        vec![2, k, m]
                    } else {
                        vec![2, m, k]
                    },
                );
                let b = TensorExpr::<D::HostType>::input(
                    "b",
                    if transpose_b { vec![n, k] } else { vec![k, n] },
                );
                let a = if transpose_a { a.transpose() } else { a };
                let b = if transpose_b { b.transpose() } else { b };
                let graph: TensorGraph<D::HostType> = (a.matmul(b)
                    + TensorExpr::input("bias", vec![n]))
                .relu()
                .into();
                for pipeline in [MatMulPipeline::Synchronous, MatMulPipeline::DoubleBuffered] {
                    executor.set_matmul_pipeline(pipeline);
                    let feeds = |phase: usize| {
                        let values = |len: usize, modulus: usize| {
                            (0..len)
                                .map(|i| {
                                    cast::<f32, D::HostType>(
                                        (((i * 13 + phase * 7) % modulus) as f32 - 20.0) / 23.0,
                                    )
                                    .unwrap()
                                })
                                .collect::<Vec<_>>()
                        };
                        let mut a = values(2 * m * k, 41);
                        if phase == 1 && k > 0 {
                            for (value, special) in
                                a.iter_mut()
                                    .zip([f32::NAN, f32::INFINITY, f32::NEG_INFINITY])
                            {
                                *value = cast::<f32, D::HostType>(special).unwrap();
                            }
                        }
                        HashMap::from([
                            ("a".to_owned(), a),
                            ("b".to_owned(), values(k * n, 43)),
                            ("bias".to_owned(), values(n, 47)),
                        ])
                    };
                    // Use the same executor to verify that changing the option
                    // invalidates compiled identity in both directions.
                    let mut expected = Vec::new();
                    for (iteration, enabled) in [false, true, false].into_iter().enumerate() {
                        executor.set_experimental_shared_memory_reuse(enabled);
                        if iteration > 0 {
                            let error = executor
                                .execute_compiled(&graph, feeds(0))
                                .err()
                                .expect("option toggle must invalidate compilation");
                            assert!(error.to_string().contains("No compiled graph signature"));
                        }
                        executor.compile(&graph).unwrap();
                        let plan = executor.execution_plan().unwrap();
                        assert!(plan.matmul_regions().iter().all(|r| {
                            r.schedule.reuse_shared_memory
                                == (enabled && r.schedule.plan != tnsr::tile::MatMulPlan::ScalarF32)
                        }));
                        let mut outputs = Vec::new();
                        for phase in 0..2 {
                            let actual = executor.execute_compiled(&graph, feeds(phase)).unwrap();
                            outputs.push(
                                actual
                                    .iter()
                                    .map(|x| {
                                        let value = x.to_f32().unwrap();
                                        if value.is_nan() {
                                            f32::NAN.to_bits()
                                        } else {
                                            value.to_bits()
                                        }
                                    })
                                    .collect::<Vec<_>>(),
                            );
                        }
                        if expected.is_empty() {
                            expected = outputs;
                        } else {
                            assert_eq!(
                                outputs, expected,
                                "{precision:?}/{m}x{n}x{k}/ta={transpose_a}/tb={transpose_b}/{pipeline:?}/reuse={enabled}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn shared_memory_reuse_preserves_bf16_gemm_bits() {
    check::<BF16>(MatMulPrecision::NativeHalf);
}

#[test]
fn shared_memory_reuse_preserves_f16_gemm_bits() {
    check::<F16>(MatMulPrecision::NativeHalf);
}

#[test]
fn shared_memory_reuse_preserves_tf32x3_gemm_bits() {
    check::<F32>(MatMulPrecision::Tf32x3);
}

#[test]
fn shared_memory_reuse_preserves_tf32_gemm_bits() {
    check::<F32>(MatMulPrecision::AllowTf32);
}
