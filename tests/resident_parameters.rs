#![cfg(feature = "cuda")]
use std::collections::HashMap;
use std::sync::Arc;
use tnsr::graph::{TensorGraph, TensorGraphNode};
use tnsr::optimizer::SGD;
use tnsr::ptx::PtxExecutor;
use tnsr::tensor::{Parameter, TensorExpr};

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
fn parameter_upload_cache_tracks_host_optimizers_and_device_publication() {
    let Some(mut executor) = executor() else {
        return;
    };
    let parameter = Parameter::new(vec![1.0, 2.0], vec![2]);
    let graph: TensorGraph<f32> = (TensorExpr::from(parameter.clone())
        + TensorExpr::constant(vec![3.0, 4.0], vec![2]))
    .into();
    assert_eq!(
        executor
            .compile_and_execute(&graph, HashMap::new())
            .unwrap(),
        [4.0, 6.0]
    );
    assert_eq!(executor.execution_metrics().device_copies, 3);
    assert_eq!(
        executor
            .compile_and_execute(&graph, HashMap::new())
            .unwrap(),
        [4.0, 6.0]
    );
    assert_eq!(executor.execution_metrics().device_copies, 1);
    parameter.set_values(vec![5.0, 6.0]).unwrap();
    assert_eq!(
        executor
            .compile_and_execute(&graph, HashMap::new())
            .unwrap(),
        [8.0, 10.0]
    );
    assert_eq!(executor.execution_metrics().device_copies, 2);
    SGD::new(0.5).step(&graph, &HashMap::from([(parameter.id(), vec![2.0, 4.0])]));
    assert_eq!(
        executor
            .compile_and_execute(&graph, HashMap::new())
            .unwrap(),
        [7.0, 8.0]
    );
    let data = graph
        .graph
        .node_weights()
        .find_map(|node| match node {
            TensorGraphNode::Parameter { data, .. } => Some(data),
            _ => None,
        })
        .unwrap();
    data.publish_device(data.revision().unwrap(), Arc::new(|| Ok(vec![7.0, 8.0])))
        .unwrap();
    assert_eq!(
        executor
            .compile_and_execute(&graph, HashMap::new())
            .unwrap(),
        [10.0, 12.0]
    );
    assert_eq!(executor.compilation_count(), 1);
    assert_eq!(parameter.values().unwrap(), [7.0, 8.0]);
}

#[test]
fn reused_parameter_id_does_not_reuse_unrelated_storage() {
    let Some(mut executor) = executor() else {
        return;
    };
    let first: TensorGraph<f32> =
        TensorExpr::from(Parameter::new_with_id(999_999, vec![1.0], vec![1])).into();
    let second: TensorGraph<f32> =
        TensorExpr::from(Parameter::new_with_id(999_999, vec![9.0], vec![1])).into();
    assert_eq!(
        executor
            .compile_and_execute(&first, HashMap::new())
            .unwrap(),
        [1.0]
    );
    assert_eq!(
        executor
            .compile_and_execute(&second, HashMap::new())
            .unwrap(),
        [9.0]
    );
}

#[test]
fn host_index_facts_refresh_and_computed_indices_keep_validation() {
    use tnsr::tensor::Input;
    let Some(mut executor) = executor() else {
        return;
    };
    let indices: TensorExpr<f32> = Input::new("indices", vec![1]).into();
    let weight = || TensorExpr::constant(vec![10.0, 20.0, 30.0], vec![3, 1]);
    let graph: TensorGraph<f32> = weight().embedding(indices.clone()).into();
    let feed = |value| HashMap::from([("indices".into(), vec![value])]);
    assert_eq!(
        executor.compile_and_execute(&graph, feed(1.0)).unwrap(),
        [20.0]
    );
    assert!(
        executor
            .compile_and_execute(&graph, feed(0.5))
            .unwrap_err()
            .to_string()
            .contains("integer")
    );
    assert!(
        executor
            .compile_and_execute(&graph, feed(3.0))
            .unwrap_err()
            .to_string()
            .contains("out of range")
    );
    assert_eq!(
        executor.compile_and_execute(&graph, feed(2.0)).unwrap(),
        [30.0]
    );
    let computed: TensorGraph<f32> = weight()
        .embedding(indices - TensorExpr::constant(vec![1.0], vec![1]))
        .into();
    assert_eq!(
        executor.compile_and_execute(&computed, feed(3.0)).unwrap(),
        [30.0]
    );
    assert!(
        executor
            .compile_and_execute(&computed, feed(0.0))
            .unwrap_err()
            .to_string()
            .contains("non-negative")
    );
    let constant: TensorGraph<f32> = weight()
        .embedding(TensorExpr::constant(vec![f32::NAN], vec![1]))
        .into();
    assert!(
        executor
            .compile_and_execute(&constant, HashMap::new())
            .unwrap_err()
            .to_string()
            .contains("finite")
    );
}
