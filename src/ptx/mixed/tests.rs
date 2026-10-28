use super::*;
use crate::{
    Executor, SimpleExecutor,
    tensor::{Parameter, TensorExpr},
};

fn cuda_available() -> bool {
    let available = CudaContext::new(0).is_ok();
    assert!(
        available || std::env::var_os("TNSR_REQUIRE_CUDA").is_none(),
        "CUDA required"
    );
    available
}

#[test]
fn native_half_uses_bf16_mma_without_tf32_for_f32_regions() {
    if !cuda_available() {
        return;
    }
    let x = TensorExpr::input("x", vec![32, 32]);
    let weight: TensorExpr<f32> = Parameter::new(vec![1.0078125; 1024], vec![32, 32]).into();
    let bias: TensorExpr<f32> = Parameter::new(vec![0.125; 32], vec![32]).into();
    let mixed = x.clone().linear_bf16(weight, bias).unwrap();
    let f32_weight: TensorExpr<f32> = Parameter::new(vec![1.0001; 1024], vec![32, 32]).into();
    let plain = x.matmul(f32_weight);
    let mut graph = TensorGraph::new();
    let mixed_output = graph.add_expr(&mixed, false);
    let f32_output = graph.add_expr(&plain, false);
    let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::NativeHalf, true).unwrap();
    assert!(gpu.info().bf16_tensor_core_programs > 0);
    gpu.upload(HashMap::from([("x".into(), vec![1.0078125; 1024])]))
        .unwrap();
    gpu.run().unwrap();
    // F32 accumulation and one final bias/output rounding differs from rounded
    // BF16 products (32.5) and from a separately rounded BF16 matmul (32.5).
    assert_eq!(gpu.read(mixed_output).unwrap(), vec![32.75; 1024]);
    let expected = 32. * 1.0078125_f32 * 1.0001_f32;
    for value in gpu.read(f32_output).unwrap() {
        assert!((value - expected).abs() < 1e-5);
    }
}

#[test]
fn mixed_fused_linear_rounds_only_after_bias() {
    if !cuda_available() {
        return;
    }
    let x = TensorExpr::input("x", vec![1, 1]);
    let w: TensorExpr<f32> = Parameter::new(vec![1.0078125_f32], vec![1, 1]).into();
    let b: TensorExpr<f32> = Parameter::new(vec![-1.015625_f32], vec![1]).into();
    let y = x.linear_bf16(w, b).unwrap().cast(ValueDType::F32);
    let mut graph = TensorGraph::new();
    let output = graph.add_expr(&y, true);
    let graph = graph.with_gradients(output);
    let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, true).unwrap();
    let mut cpu = SimpleExecutor::new();
    let inputs = HashMap::from([("x".into(), vec![1.0078125])]);
    cpu.execute(&graph, inputs.clone()).unwrap();
    gpu.upload(inputs).unwrap();
    gpu.run().unwrap();
    assert_eq!(gpu.read(output).unwrap(), vec![1. / 16384.]);
    assert_eq!(gpu.read(output).unwrap(), cpu.get_value(output).unwrap());
    assert_eq!(gpu.gradients(&graph).unwrap(), cpu.get_gradients(&graph));
}

#[test]
fn mixed_exact_indices_preserve_365_and_reset_scatter_on_replay() {
    use crate::tensor::IndexTensor;
    if !cuda_available() {
        return;
    }
    let table = Parameter::new(
        (0..1200).map(|i| (i % 13) as f32 / 8.).collect(),
        vec![400, 3],
    );
    let id = table.id();
    let y = TensorExpr::from(table).embedding_indices(IndexTensor::input("ids", vec![3]));
    let y = y.cast(ValueDType::BF16).cast(ValueDType::F32);
    let loss = (y * TensorExpr::input("g", vec![3, 3]))
        .reduce_sum(0)
        .reduce_sum(1);
    let mut graph = TensorGraph::new();
    let output = graph.add_expr(&loss, false);
    let graph = graph.with_gradients(output);
    let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, true).unwrap();
    let inputs = HashMap::from([("g".into(), vec![0.5; 9])]);
    let indices = HashMap::from([("ids".into(), vec![365, 365, 1])]);
    let mut cpu = SimpleExecutor::new();
    cpu.execute_with_indices(&graph, inputs.clone(), indices.clone())
        .unwrap();
    gpu.upload_with_indices(inputs.clone(), indices).unwrap();
    for _ in 0..3 {
        gpu.run().unwrap();
        assert_eq!(gpu.read(output).unwrap(), cpu.get_value(output).unwrap());
        assert_eq!(
            gpu.gradients(&graph).unwrap()[&id],
            cpu.get_gradients(&graph)[&id]
        );
    }
    assert!(
        gpu.upload_with_indices(inputs, HashMap::from([("ids".into(), vec![365, 400, 1])]))
            .is_err()
    );
    gpu.run().unwrap();
    assert_eq!(
        gpu.gradients(&graph).unwrap()[&id],
        cpu.get_gradients(&graph)[&id]
    );
}

#[test]
fn mixed_autograd_shared_cast_and_replay() {
    if !cuda_available() {
        return;
    }
    for replay in [false, true] {
        for shared in [false, true] {
            let (graph, id) = crate::graph::mixed::tests::fanout(shared);
            let mut executor = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, replay).unwrap();
            assert!(executor.info().bf16_bytes > 0);
            executor.upload(HashMap::new()).unwrap();
            executor.run().unwrap();
            assert_eq!(
                executor.gradients(&graph).unwrap()[&id],
                vec![if shared { 1. } else { 1. + 1. / 256. }]
            );
            for _ in 0..3 {
                executor.run().unwrap();
            }
            assert_eq!(
                executor.gradients(&graph).unwrap()[&id],
                vec![if shared { 1. } else { 1. + 1. / 256. }]
            );
        }
    }
}

#[test]
fn mixed_graph_matmul_views_and_changed_masters() {
    if !cuda_available() {
        return;
    }
    // Power-of-two fixtures give an exact independent scalar CPU oracle.
    let master = Parameter::new((0..35).map(|i| (i % 7) as f32 / 8.).collect(), vec![5, 7]);
    let id = master.id();
    let weight: TensorExpr<f32> = master.clone().into();
    let x = TensorExpr::input("x", vec![3, 7]).cast(ValueDType::BF16);
    let y = x
        .matmul(weight.cast(ValueDType::BF16).transpose())
        .cast(ValueDType::F32);
    let loss = y.clone().reduce_sum(0).reduce_sum(1);
    let mut graph = TensorGraph::new();
    let output = graph.add_expr(&loss, false);
    let graph = graph.with_gradients(output);
    let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, true).unwrap();
    for step in 0..2 {
        let inputs = HashMap::from([(
            "x".into(),
            (0..21).map(|i| (i % 5) as f32 / 8. + step as f32).collect(),
        )]);
        let mut cpu = SimpleExecutor::new();
        cpu.execute(&graph, inputs.clone()).unwrap();
        gpu.upload(inputs).unwrap();
        gpu.run().unwrap();
        assert_eq!(gpu.read(output).unwrap(), cpu.get_value(output).unwrap());
        assert_eq!(
            gpu.gradients(&graph).unwrap()[&id],
            cpu.get_gradients(&graph)[&id]
        );
        master.set_values(vec![0.5; 35]).unwrap();
    }
    // Rejected uploads must preserve the previously executable inputs.
    assert!(
        gpu.upload(HashMap::from([("x".into(), vec![0.; 20])]))
            .is_err()
    );
    gpu.run().unwrap();
    assert!(!gpu.read(output).unwrap().iter().any(|x| x.is_nan()));
}

#[test]
fn mixed_cast_materializes_noncontiguous_views_once() {
    if !cuda_available() {
        return;
    }
    let x = TensorExpr::input("x", vec![2, 3])
        .transpose()
        .cast(ValueDType::BF16)
        .cast(ValueDType::F32);
    let mut graph = TensorGraph::new();
    let output = graph.add_expr(&x, false);
    let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, true).unwrap();
    gpu.upload(HashMap::from([(
        "x".into(),
        vec![1. + 1. / 256., 2., 3., 4., 5., 6.],
    )]))
    .unwrap();
    gpu.run().unwrap();
    assert_eq!(gpu.read(output).unwrap(), vec![1., 4., 2., 5., 3., 6.]);
    // Promotion is a typed read of the narrowed storage; only narrowing needs a kernel.
    assert_eq!(gpu.info().kernels, 1);
}

#[test]
fn mixed_f32_regions_save_silu_primals_across_cast_boundaries() {
    if !cuda_available() {
        return;
    }
    let master = Parameter::new(vec![-2_f32, -0.5, 0., 0.25, 1., 3.], vec![2, 3]);
    let id = master.id();
    let primal: TensorExpr<f32> = master.clone().into();
    let x = primal
        .cast(ValueDType::BF16)
        .transpose()
        .cast(ValueDType::F32);
    let y = crate::nn::silu(x).cast(ValueDType::BF16);
    let loss = (y.clone().cast(ValueDType::F32) * TensorExpr::input("a", vec![3, 2])
        + y.cast(ValueDType::F32) * TensorExpr::input("b", vec![3, 2]))
    .reduce_sum(0)
    .reduce_sum(1);
    let mut graph = TensorGraph::new();
    let output = graph.add_expr(&loss, false);
    let graph = graph.with_gradients(output);
    let mut fused = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, true).unwrap();
    let mut unfused = MixedExecutor::new_unfused(&graph, MatMulPrecision::Tf32x3, false).unwrap();
    assert!(fused.info().kernels < unfused.info().kernels);
    for step in 0..2 {
        let inputs = HashMap::from([
            ("a".into(), vec![1. + 1. / 256.; 6]),
            ("b".into(), vec![1. / 256.; 6]),
        ]);
        let mut cpu = SimpleExecutor::new();
        cpu.execute(&graph, inputs.clone()).unwrap();
        let expected = cpu.get_gradients(&graph)[&id].clone();
        for gpu in [&mut fused, &mut unfused] {
            gpu.upload(inputs.clone()).unwrap();
            gpu.run().unwrap();
            let actual = gpu.gradients(&graph).unwrap();
            assert_eq!(actual[&id], expected, "step {step}");
            assert_eq!(gpu.read(output).unwrap(), cpu.get_value(output).unwrap());
        }
        master.set_values(vec![0.5; 6]).unwrap();
    }
}

#[test]
fn mixed_identity_graph_needs_no_capture_or_kernel() {
    if !cuda_available() {
        return;
    }
    let x = TensorExpr::input("x", vec![2]).cast(ValueDType::F32);
    let mut graph = TensorGraph::new();
    let output = graph.add_expr(&x, false);
    let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, true).unwrap();
    assert_eq!(gpu.info().kernels, 0);
    assert!(!gpu.info().replay);
    gpu.upload(HashMap::from([("x".into(), vec![1., 2.])]))
        .unwrap();
    gpu.run().unwrap();
    assert_eq!(gpu.read(output).unwrap(), vec![1., 2.]);
}

#[test]
fn mixed_norm_linear_gradients_match_cpu() {
    if !cuda_available() {
        return;
    }
    let weight = Parameter::new(vec![1f32; 4], vec![4]);
    let bias = Parameter::new(vec![0f32; 4], vec![4]);
    let x = TensorExpr::input("x", vec![2, 4]);
    let normalized = crate::nn::layer_norm(x, 1, weight.clone(), bias.clone(), 1e-5);
    let y = normalized
        .linear_bf16(
            TensorExpr::constant(vec![0.5; 12], vec![4, 3]),
            TensorExpr::constant(vec![0.; 3], vec![3]),
        )
        .unwrap();
    let y = (y.cast(ValueDType::F32)
        * TensorExpr::constant(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]))
    .mean_all();
    let mut graph = TensorGraph::new();
    let output = graph.add_expr(&y, false);
    let graph = graph.with_gradients(output);
    let inputs = HashMap::from([("x".into(), vec![1., 2., 3., 4., 5., 6., 7., 8.])]);
    let mut cpu = SimpleExecutor::new();
    cpu.execute(&graph, inputs.clone()).unwrap();
    let expected = cpu.get_gradients(&graph);
    let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, true).unwrap();
    gpu.upload(inputs).unwrap();
    gpu.run().unwrap();
    let actual = gpu.gradients(&graph).unwrap();
    for id in [weight.id(), bias.id()] {
        assert!(expected[&id].iter().any(|x| x.abs() > 0.1));
        for (a, b) in actual[&id].iter().zip(&expected[&id]) {
            assert!((a - b).abs() < 1e-4, "{id}: {a} != {b}");
        }
    }
}

#[test]
fn mixed_linear_bias_reduction_matches_cpu() {
    if !cuda_available() {
        return;
    }
    let bias = Parameter::new(vec![0f32; 768], vec![768]);
    let weight = Parameter::new(
        (0..(32 * 768))
            .map(|i| ((i % 17) as f32 - 5.) / 8.)
            .collect(),
        vec![32, 768],
    );
    let x = Parameter::new(
        (0..(192 * 32))
            .map(|i| ((i % 29) as f32 - 10.) / 4.)
            .collect(),
        vec![3, 64, 32],
    );
    let y = TensorExpr::from(x.clone())
        .reshape(vec![192, 32])
        .linear_bf16(weight.clone().into(), bias.clone().into())
        .unwrap()
        .reshape(vec![3, 64, 768])
        .cast(ValueDType::F32);
    let loss = (y * TensorExpr::constant(
        (0..(192 * 768))
            .map(|i| ((i % 13) as f32 - 4.) / 8.)
            .collect(),
        vec![3, 64, 768],
    ))
    .mean_all();
    let mut graph = TensorGraph::new();
    let out = graph.add_expr(&loss, false);
    let graph = graph.with_gradients(out);
    let mut cpu = SimpleExecutor::new();
    cpu.execute(&graph, HashMap::new()).unwrap();
    let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, true).unwrap();
    gpu.upload(HashMap::new()).unwrap();
    gpu.run().unwrap();
    let actual = gpu.gradients(&graph).unwrap();
    let expected = cpu.get_gradients(&graph);
    for id in [x.id(), weight.id(), bias.id()] {
        for (a, b) in actual[&id].iter().zip(&expected[&id]) {
            assert!((a - b).abs() < 1e-4, "{id}: {a} != {b}");
        }
    }
}

#[test]
fn mixed_register_tiled_bf16_matmul_handles_edges_and_replay() {
    if !cuda_available() {
        return;
    }
    let (m, k, n) = (2048, 65, 257);
    let a = TensorExpr::input("a", vec![m, k]).cast(ValueDType::BF16);
    let b = TensorExpr::input("b", vec![k, n]).cast(ValueDType::BF16);
    let mut graph = TensorGraph::new();
    let out = graph.add_expr(&a.matmul(b), false);
    let a: Vec<f32> = (0..m * k)
        .map(|i| half::bf16::from_f32((i % 23) as f32 / 37. - 0.2).to_f32())
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| half::bf16::from_f32((i % 17) as f32 / 29. - 0.3).to_f32())
        .collect();
    for replay in [false, true] {
        let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, replay).unwrap();
        gpu.upload(HashMap::from([
            ("a".into(), a.clone()),
            ("b".into(), b.clone()),
        ]))
        .unwrap();
        for _ in 0..2 {
            gpu.run().unwrap();
            let actual = gpu.read(out).unwrap();
            for row in [0, 63, 64, 2047] {
                for col in [0, 63, 64, 255, 256] {
                    let expected: f64 = (0..k)
                        .map(|inner| f64::from(a[row * k + inner]) * f64::from(b[inner * n + col]))
                        .sum();
                    let expected = half::bf16::from_f32(expected as f32).to_f32();
                    assert!(
                        (actual[row * n + col] - expected).abs() <= 0.0078125,
                        "row {row} col {col}"
                    );
                }
            }
        }
    }
}

#[test]
fn mixed_split_k_keeps_f32_partials_and_rounds_once() {
    if !cuda_available() {
        return;
    }
    let (m, k, n) = (128, 2048, 128);
    let a = TensorExpr::input("a", vec![k, m])
        .cast(ValueDType::BF16)
        .transpose();
    let b = TensorExpr::input("b", vec![k, n]).cast(ValueDType::BF16);
    let expression = a.matmul(b);
    let mut graph = TensorGraph::new();
    let out = graph.add_expr(&expression, false);
    let mut b = vec![0.; k * n];
    for part in 0..4 {
        for col in 0..n {
            b[(part * (k / 4)) * n + col] = 1.;
            b[(part * (k / 4) + 1) * n + col] = if part == 3 { 1. / 128. } else { 1. / 256. };
        }
    }
    for replay in [false, true] {
        let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, replay).unwrap();
        assert!(
            gpu.kernel_labels()
                .iter()
                .any(|s| s.contains("split-K partial"))
        );
        gpu.upload(HashMap::from([
            ("a".into(), vec![1.; k * m]),
            ("b".into(), b.clone()),
        ]))
        .unwrap();
        for _ in 0..2 {
            gpu.run().unwrap();
            for (i, value) in gpu.read(out).unwrap().iter().enumerate() {
                assert_eq!(*value, 4.03125, "element {i}");
            }
        }
    }
}

#[test]
fn mixed_destruction_waits_for_shared_stream_capture() {
    use std::{sync::mpsc, time::Duration};
    if !cuda_available() {
        return;
    }
    let graph: TensorGraph<f32> = TensorExpr::input("x", vec![4])
        .cast(ValueDType::BF16)
        .into();
    let executor = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, true).unwrap();
    let gate = executor.gate.clone();
    let stream = executor.stream.clone();
    let kernel = executor.kernels[0].clone();
    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let task = std::thread::spawn(move || {
        let guard = gate.lock().unwrap();
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .unwrap();
        kernel.run(&stream).unwrap();
        started_tx.send(()).unwrap();
        assert!(matches!(
            done_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        let graph = stream
            .end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .unwrap()
            .unwrap();
        drop(graph);
        drop(guard);
        done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    });
    started_rx.recv().unwrap();
    drop(executor);
    done_tx.send(()).unwrap();
    task.join().unwrap();
}

#[test]
fn mixed_preparation_waits_for_shared_stream_capture() {
    use std::{sync::mpsc, time::Duration};
    if !cuda_available() {
        return;
    }
    let graph: TensorGraph<f32> = TensorExpr::input("x", vec![4])
        .cast(ValueDType::BF16)
        .into();
    let executor = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, true).unwrap();
    let gate = executor.gate.clone();
    let stream = executor.stream.clone();
    let kernel = executor.kernels[0].clone();
    let bindings = ResidentBindings {
        stream: stream.clone(),
        gate: gate.clone(),
        parameters: HashMap::new(),
    };
    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let task = std::thread::spawn(move || {
        let guard = gate.lock().unwrap();
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .unwrap();
        kernel.run(&stream).unwrap();
        started_tx.send(()).unwrap();
        assert!(matches!(
            done_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        drop(
            stream
                .end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .unwrap()
                .unwrap(),
        );
        drop(guard);
        done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    });
    started_rx.recv().unwrap();
    let prepared =
        MixedExecutor::for_resident(&graph, MatMulPrecision::Tf32x3, true, bindings).unwrap();
    done_tx.send(()).unwrap();
    task.join().unwrap();
    drop(prepared);
}

#[test]
fn mixed_transposed_bf16_async_replay_handles_changed_feeds() {
    if !cuda_available() {
        return;
    }
    for (m, k, n, ta, tb) in [
        (264, 136, 280, true, true),
        (267, 136, 280, true, true),
        (264, 136, 283, true, false),
        (264, 137, 280, false, false),
    ] {
        let a =
            TensorExpr::input("a", if ta { vec![k, m] } else { vec![m, k] }).cast(ValueDType::BF16);
        let b =
            TensorExpr::input("b", if tb { vec![n, k] } else { vec![k, n] }).cast(ValueDType::BF16);
        let a = if ta { a.transpose() } else { a };
        let b = if tb { b.transpose() } else { b };
        let mut graph = TensorGraph::new();
        let out = graph.add_expr(&a.matmul(b), false);
        for replay in [false, true] {
            let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, replay).unwrap();
            for revision in 0..3 {
                let values = |count, period| {
                    (0..count)
                        .map(|i| {
                            half::bf16::from_f32(((i + revision) % period) as f32 / 32. - 0.125)
                                .to_f32()
                        })
                        .collect::<Vec<_>>()
                };
                let av = values(k * m, 7);
                let bv = values(n * k, 5);
                gpu.upload(HashMap::from([
                    ("a".into(), av.clone()),
                    ("b".into(), bv.clone()),
                ]))
                .unwrap();
                gpu.run().unwrap();
                let actual = gpu.read(out).unwrap();
                for row in [0, 31, 32, 255, m - 1] {
                    for col in [0, 31, 32, 255, n - 1] {
                        let expected: f64 = (0..k)
                            .map(|inner| {
                                f64::from(av[if ta { inner * m + row } else { row * k + inner }])
                                    * f64::from(
                                        bv[if tb { col * k + inner } else { inner * n + col }],
                                    )
                            })
                            .sum();
                        assert!(
                            (actual[row * n + col] - half::bf16::from_f64(expected).to_f32()).abs()
                                <= f32::EPSILON,
                            "replay={replay} revision={revision} row={row} col={col}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn mixed_hybrid_split_k_preserves_ragged_edges_and_replay() {
    if !cuda_available() {
        return;
    }
    let k = 4096;
    for (m, n) in [(267, 280), (264, 283)] {
        let a = TensorExpr::input("a", vec![k, m])
            .cast(ValueDType::BF16)
            .transpose();
        let b = TensorExpr::input("b", vec![k, n]).cast(ValueDType::BF16);
        let mut graph = TensorGraph::new();
        let out = graph.add_expr(&a.matmul(b), false);
        for replay in [false, true] {
            let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, replay).unwrap();
            for revision in 0..2 {
                let values = |count, period| {
                    (0..count)
                        .map(|i| {
                            half::bf16::from_f32(((i + revision) % period) as f32 / 32. - 0.125)
                                .to_f32()
                        })
                        .collect::<Vec<_>>()
                };
                let av = values(k * m, 7);
                let bv = values(k * n, 5);
                gpu.upload(HashMap::from([
                    ("a".into(), av.clone()),
                    ("b".into(), bv.clone()),
                ]))
                .unwrap();
                gpu.run().unwrap();
                let actual = gpu.read(out).unwrap();
                for row in [0, 31, 32, 63, 64, m - 1] {
                    for col in [0, 31, 32, 63, 64, n - 1] {
                        let expected: f64 = (0..k)
                            .map(|inner| {
                                f64::from(av[inner * m + row]) * f64::from(bv[inner * n + col])
                            })
                            .sum();
                        assert!(
                            (actual[row * n + col] - half::bf16::from_f64(expected).to_f32()).abs()
                                <= f32::EPSILON,
                            "m={m} n={n} replay={replay} revision={revision} row={row} col={col}"
                        );
                    }
                }
            }
        }
    }
}
