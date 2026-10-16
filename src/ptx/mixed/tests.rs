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
    assert_eq!(gpu.info().kernels, 2);
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
