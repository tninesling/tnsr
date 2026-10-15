// Golden values/gradients generated with PyTorch 2.11 CPU f32, default GELU.
use super::*;
use crate::graph::TensorGraph;
use crate::tensor::Parameter;
use crate::{Executor, SimpleExecutor};
use std::collections::HashMap;
fn check(executor: &mut impl Executor<f32>) {
    let data = vec![
        f32::from_bits(0xc47a0000),
        f32::from_bits(0xc1a00000),
        f32::from_bits(0xc1200000),
        f32::from_bits(0xc0c00000),
        f32::from_bits(0xc0800000),
        f32::from_bits(0xc0000000),
        f32::from_bits(0xbf800000),
        f32::from_bits(0xbc23d70a),
        f32::from_bits(0xb58637bd),
        f32::from_bits(0x00000000),
        f32::from_bits(0x358637bd),
        f32::from_bits(0x3c23d70a),
        f32::from_bits(0x3f800000),
        f32::from_bits(0x40000000),
        f32::from_bits(0x40800000),
        f32::from_bits(0x40c00000),
        f32::from_bits(0x41200000),
        f32::from_bits(0x41a00000),
        f32::from_bits(0x447a0000),
    ];
    for (name, activation, expected, gradients) in [
        (
            "sigmoid",
            sigmoid as fn(TensorExpr<f32>) -> TensorExpr<f32>,
            vec![
                f32::from_bits(0x00000000),
                f32::from_bits(0x310da433),
                f32::from_bits(0x383e6998),
                f32::from_bits(0x3b220bbb),
                f32::from_bits(0x3c9357d1),
                f32::from_bits(0x3df420a9),
                f32::from_bits(0x3e89b2b1),
                f32::from_bits(0x3efeb852),
                f32::from_bits(0x3efffff8),
                f32::from_bits(0x3f000000),
                f32::from_bits(0x3f000004),
                f32::from_bits(0x3f00a3d7),
                f32::from_bits(0x3f3b26a8),
                f32::from_bits(0x3f617bea),
                f32::from_bits(0x3f7b6541),
                f32::from_bits(0x3f7f5df5),
                f32::from_bits(0x3f7ffd06),
                f32::from_bits(0x3f800000),
                f32::from_bits(0x3f800000),
            ],
            vec![
                f32::from_bits(0x00000000),
                f32::from_bits(0x310da433),
                f32::from_bits(0x383e6761),
                f32::from_bits(0x3b21a528),
                f32::from_bits(0x3c90b161),
                f32::from_bits(0x3dd706e0),
                f32::from_bits(0x3e4954a4),
                f32::from_bits(0x3e7ffe5d),
                f32::from_bits(0x3e800000),
                f32::from_bits(0x3e800000),
                f32::from_bits(0x3e800000),
                f32::from_bits(0x3e7ffe5d),
                f32::from_bits(0x3e4954a3),
                f32::from_bits(0x3dd706e6),
                f32::from_bits(0x3c90b170),
                f32::from_bits(0x3b21a46e),
                f32::from_bits(0x383e7dc9),
                f32::from_bits(0x00000000),
                f32::from_bits(0x00000000),
            ],
        ),
        (
            "silu",
            silu as fn(TensorExpr<f32>) -> TensorExpr<f32>,
            vec![
                f32::from_bits(0x80000000),
                f32::from_bits(0xb3310d40),
                f32::from_bits(0xb9ee03fe),
                f32::from_bits(0xbc731199),
                f32::from_bits(0xbd9357d1),
                f32::from_bits(0xbe7420a9),
                f32::from_bits(0xbe89b2b1),
                f32::from_bits(0xbba30553),
                f32::from_bits(0xb50637b9),
                f32::from_bits(0x00000000),
                f32::from_bits(0x350637c1),
                f32::from_bits(0x3ba4a8c1),
                f32::from_bits(0x3f3b26a8),
                f32::from_bits(0x3fe17bea),
                f32::from_bits(0x407b6541),
                f32::from_bits(0x40bf8678),
                f32::from_bits(0x411ffe24),
                f32::from_bits(0x41a00000),
                f32::from_bits(0x447a0000),
            ],
            vec![
                f32::from_bits(0x80000000),
                f32::from_bits(0xb32832fd),
                f32::from_bits(0xb9d63407),
                f32::from_bits(0xbc49f4cd),
                f32::from_bits(0xbd57b6d9),
                f32::from_bits(0xbdb9ed18),
                f32::from_bits(0x3d94217b),
                f32::from_bits(0x3efd70a6),
                f32::from_bits(0x3efffff0),
                f32::from_bits(0x3f000000),
                f32::from_bits(0x3f000008),
                f32::from_bits(0x3f0147ad),
                f32::from_bits(0x3f6d7bd1),
                f32::from_bits(0x3f8b9ed2),
                f32::from_bits(0x3f86bdb7),
                f32::from_bits(0x3f8193e8),
                f32::from_bits(0x3f800d65),
                f32::from_bits(0x3f800000),
                f32::from_bits(0x3f800000),
            ],
        ),
        (
            "gelu",
            gelu as fn(TensorExpr<f32>) -> TensorExpr<f32>,
            vec![
                f32::from_bits(0x00000000),
                f32::from_bits(0x00000000),
                f32::from_bits(0x00000000),
                f32::from_bits(0x00000000),
                f32::from_bits(0xb904e000),
                f32::from_bits(0xbd3a5e50),
                f32::from_bits(0xbe227688),
                f32::from_bits(0xbba28861),
                f32::from_bits(0xb50637b6),
                f32::from_bits(0x00000000),
                f32::from_bits(0x350637c4),
                f32::from_bits(0x3ba525b3),
                f32::from_bits(0x3f57625e),
                f32::from_bits(0x3ffa2d0e),
                f32::from_bits(0x407ffdec),
                f32::from_bits(0x40c00000),
                f32::from_bits(0x41200000),
                f32::from_bits(0x41a00000),
                f32::from_bits(0x447a0000),
            ],
            vec![
                f32::from_bits(0x00000000),
                f32::from_bits(0x00000000),
                f32::from_bits(0x00000000),
                f32::from_bits(0xb3000000),
                f32::from_bits(0xba040600),
                f32::from_bits(0xbdae8e0c),
                f32::from_bits(0xbdaaa148),
                f32::from_bits(0x3efbea38),
                f32::from_bits(0x3effffe6),
                f32::from_bits(0x3f000000),
                f32::from_bits(0x3f00000d),
                f32::from_bits(0x3f020ae4),
                f32::from_bits(0x3f8aaa14),
                f32::from_bits(0x3f8ae8e1),
                f32::from_bits(0x3f801081),
                f32::from_bits(0x3f800000),
                f32::from_bits(0x3f800000),
                f32::from_bits(0x3f800000),
                f32::from_bits(0x3f800000),
            ],
        ),
    ] {
        let x = Parameter::new(data.clone(), vec![data.len()]);
        let id = x.id();
        let expression = activation(TensorExpr::from(x));
        let graph: TensorGraph<f32> = expression.reduce_sum(0).into();
        let loss = *graph.toposort().last().unwrap();
        let graph = graph.with_gradients(loss);
        executor.execute(&graph, HashMap::new()).unwrap();
        let actual = executor.get_gradients(&graph).remove(&id).unwrap();
        for (&actual, &expected) in actual.iter().zip(&gradients) {
            assert!(
                (actual - expected).abs() < 4e-6,
                "{name} gradient {actual} != {expected}"
            );
        }
        // Forward values use an inference graph so fusion/liveness may retain only its sink.
        let graph: TensorGraph<f32> =
            activation(TensorExpr::constant(data.clone(), vec![data.len()])).into();
        let actual = executor.execute(&graph, HashMap::new()).unwrap();
        for (&actual, &expected) in actual.iter().zip(&expected) {
            assert!(
                (actual - expected).abs() < 2e-6,
                "{name} value {actual} != {expected}"
            );
        }
    }
}
#[test]
fn cpu_model_activations() {
    check(&mut SimpleExecutor::new());
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_model_activations() {
    let Ok(mut executor) = crate::ptx::PtxExecutor::try_new() else {
        assert_ne!(std::env::var("TNSR_REQUIRE_CUDA").as_deref(), Ok("1"));
        return;
    };
    check(&mut executor);
}
#[test]
fn clipping_boundaries_and_ties() {
    let x = Parameter::new(vec![-2., -1., 0., 1., 2.], vec![5]);
    let id = x.id();
    let graph: TensorGraph<f32> = clamp(x, -1., 1.).reduce_sum(0).into();
    let loss = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(loss);
    let mut executor = SimpleExecutor::new();
    executor.execute(&graph, HashMap::new()).unwrap();
    assert_eq!(
        executor.get_gradients(&graph)[&id],
        vec![0., 1., 1., 1., 0.]
    );
    let x = Parameter::new(vec![1., 2., 3.], vec![3]);
    let y = Parameter::new(vec![2., 2., 2.], vec![3]);
    let (a, b) = (x.id(), y.id());
    let graph: TensorGraph<f32> = minimum(x, y).reduce_sum(0).into();
    let loss = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(loss);
    executor.execute(&graph, HashMap::new()).unwrap();
    let grads = executor.get_gradients(&graph);
    assert_eq!(grads[&a], vec![1., 0.5, 0.]);
    assert_eq!(grads[&b], vec![0., 0.5, 1.]);
}
