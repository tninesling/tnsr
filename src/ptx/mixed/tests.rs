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
