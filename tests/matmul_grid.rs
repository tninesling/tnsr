#![cfg(feature = "cuda")]

use std::collections::HashMap;
use tnsr::{
    graph::TensorGraph,
    ptx::PtxExecutor,
    tensor::{Parameter, TensorExpr},
    tile::MatMulPrecision,
};

fn executor() -> Option<PtxExecutor> {
    match PtxExecutor::try_new() {
        Ok(executor) => Some(executor),
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
fn grouped_matmul_preserves_bits_for_broadcast_transposes_tails_and_epilogues() {
    let Some(mut executor) = executor() else {
        return;
    };
    for precision in [MatMulPrecision::StrictF32, MatMulPrecision::Tf32x3] {
        executor.set_matmul_precision(precision);
        for (m, n, k) in [(19, 35, 17), (67, 81, 65), (129, 257, 129)] {
            for transpose in [false, true] {
                let a = TensorExpr::<f32>::input(
                    "a",
                    if transpose {
                        vec![2, k, m]
                    } else {
                        vec![2, m, k]
                    },
                );
                let b =
                    TensorExpr::<f32>::input("b", if transpose { vec![n, k] } else { vec![k, n] });
                let (a, b) = if transpose {
                    (a.transpose(), b.transpose())
                } else {
                    (a, b)
                };
                let graph: TensorGraph<f32> = (a.matmul(b) + TensorExpr::input("bias", vec![n]))
                    .relu()
                    .into();
                let mut original = Vec::new();
                for group in [1, 2, 4, 8, 16, 32] {
                    executor.set_matmul_grid_group_m(group).unwrap();
                    let feeds = HashMap::from([
                        (
                            "a".into(),
                            (0..2 * m * k)
                                .map(|i| ((i * 13 % 97) as f32 - 48.0) / 31.0)
                                .collect(),
                        ),
                        (
                            "b".into(),
                            (0..k * n)
                                .map(|i| ((i * 19 % 89) as f32 - 44.0) / 29.0)
                                .collect(),
                        ),
                        (
                            "bias".into(),
                            (0..n).map(|i| (i % 13) as f32 / 17.0).collect(),
                        ),
                    ]);
                    if group != 1 {
                        let error = executor
                            .execute_compiled(&graph, feeds.clone())
                            .unwrap_err();
                        assert!(error.to_string().contains("No compiled graph signature"));
                    }
                    executor.compile_owned(graph.clone()).unwrap();
                    let actual = executor.execute_compiled(&graph, feeds).unwrap();
                    let bits: Vec<_> = actual.iter().map(|x| x.to_bits()).collect();
                    if group == 1 {
                        original = bits;
                    } else {
                        assert_eq!(
                            bits, original,
                            "{precision:?}/{m}x{n}x{k}/transpose={transpose}/group={group}"
                        );
                    }
                }
            }
        }
    }
    assert!(executor.set_matmul_grid_group_m(0).is_err());
    assert!(executor.set_matmul_grid_group_m(3).is_err());
}

#[test]
fn grouped_matmul_bf16_preserves_storage_bits() {
    use half::bf16;
    use tnsr::ptx::BF16;
    let Some(_) = executor() else { return };
    let mut executor = PtxExecutor::<BF16>::try_new_for_dtype().unwrap();
    executor.set_matmul_precision(MatMulPrecision::NativeHalf);
    let (m, k, n) = (161, 64, 257);
    let a: TensorExpr<bf16> = Parameter::new(
        (0..m * k)
            .map(|i| bf16::from_f32((i % 31) as f32 / 31.0 - 0.5))
            .collect(),
        vec![m, k],
    )
    .into();
    let b: TensorExpr<bf16> = Parameter::new(
        (0..k * n)
            .map(|i| bf16::from_f32((i % 29) as f32 / 29.0 - 0.5))
            .collect(),
        vec![k, n],
    )
    .into();
    let graph: TensorGraph<bf16> = a.matmul(b).relu().into();
    let mut original = Vec::new();
    for group in [1, 4, 8, 32] {
        executor.set_matmul_grid_group_m(group).unwrap();
        executor.compile_owned(graph.clone()).unwrap();
        let bits: Vec<_> = executor
            .execute_compiled(&graph, HashMap::new())
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect();
        if group == 1 {
            original = bits;
        } else {
            assert_eq!(bits, original);
        }
    }
}
