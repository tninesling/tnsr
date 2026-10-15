use super::*;
use crate::{Executor, SimpleExecutor};

fn check(executor: &mut impl Executor<f32>) {
    let x = Parameter::new((1u16..=24).map(f32::from).collect(), vec![2, 3, 4]);
    let id = x.id();
    let slice = TensorExpr::from(x).narrow(1, 1, 2);
    let forward: graph::TensorGraph<f32> = slice.clone().into();
    assert_eq!(
        executor.execute(&forward, HashMap::new()).unwrap(),
        vec![
            5., 6., 7., 8., 9., 10., 11., 12., 17., 18., 19., 20., 21., 22., 23., 24.
        ]
    );
    let loss = slice.reduce_sum(0).reduce_sum(1).reduce_sum(2);
    let graph: graph::TensorGraph<f32> = loss.into();
    let last = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(last);
    executor.execute(&graph, HashMap::new()).unwrap();
    assert_eq!(
        executor.get_gradients(&graph)[&id],
        vec![
            0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.,
            1., 1.
        ]
    );

    let x = Parameter::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let id = x.id();
    let a = TensorExpr::from(x.clone()).narrow(1, 0, 2);
    let b = TensorExpr::from(x).narrow(1, 1, 2);
    let joined = TensorExpr::concat(&[a, b], 1);
    let forward: graph::TensorGraph<f32> = joined.clone().optimize().into();
    assert_eq!(
        executor.execute(&forward, HashMap::new()).unwrap(),
        vec![1., 2., 2., 3., 4., 5., 5., 6.]
    );
    let graph: graph::TensorGraph<f32> = joined.reduce_sum(0).reduce_sum(1).into();
    let last = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(last);
    executor.execute(&graph, HashMap::new()).unwrap();
    assert_eq!(
        executor.get_gradients(&graph)[&id],
        vec![1., 2., 1., 1., 2., 1.]
    );

    let empty = Parameter::new(vec![], vec![0, 3]);
    let full = Parameter::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let (a, b) = (empty.id(), full.id());
    let graph: graph::TensorGraph<f32> = TensorExpr::concat(&[empty.into(), full.into()], 0)
        .reduce_sum(0)
        .reduce_sum(1)
        .into();
    let last = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(last);
    executor.execute(&graph, HashMap::new()).unwrap();
    let grads = executor.get_gradients(&graph);
    assert_eq!(grads[&a], Vec::<f32>::new());
    assert_eq!(grads[&b], vec![1.; 6]);
}
#[test]
fn cpu_tensor_segments() {
    check(&mut SimpleExecutor::new());
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_tensor_segments() {
    let Ok(mut executor) = crate::ptx::PtxExecutor::try_new() else {
        assert_ne!(std::env::var("TNSR_REQUIRE_CUDA").as_deref(), Ok("1"));
        return;
    };
    check(&mut executor);
}
#[test]
#[should_panic(expected = "narrow range out of bounds")]
fn narrow_rejects_invalid_range() {
    TensorExpr::constant(vec![0.; 4], vec![4]).narrow(0, 3, 2);
}
#[test]
#[should_panic(expected = "concat shape mismatch")]
fn concat_rejects_other_axes() {
    TensorExpr::concat(
        &[
            TensorExpr::constant(vec![0.; 4], vec![2, 2]),
            TensorExpr::constant(vec![0.; 3], vec![1, 3]),
        ],
        0,
    );
}
