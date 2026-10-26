#![cfg(feature = "cuda")]
use std::collections::HashMap;
use tnsr::{
    Executor,
    graph::TensorGraph,
    ptx::{PtxExecutor, PtxFusionPolicy, PtxReductionMode},
    tensor::TensorExpr,
};

fn executor(mode: PtxReductionMode) -> Option<PtxExecutor> {
    match PtxExecutor::try_new_with_reduction_mode(mode) {
        Ok(executor) => Some(executor),
        Err(error)
            if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                && error
                    .to_string()
                    .starts_with("Failed to initialize CUDA device 0") =>
        {
            eprintln!("skipping CUDA: {error:#}");
            None
        }
        Err(error) => panic!("{error:#}"),
    }
}

fn graph(negations: usize) -> TensorGraph<f32> {
    let mut row = [1.0_f32; 32];
    row[0] = 16_777_216.0;
    row[31] = -16_777_216.0;
    let mut producer = TensorExpr::constant(row.repeat(2), vec![2, 32]);
    for _ in 0..negations {
        producer = -producer;
    }
    producer.reduce_sum(1).into()
}

#[test]
fn pointwise_fusion_budget_preserves_cooperative_reduction_values() {
    let Some(mut tree) = executor(PtxReductionMode::DeterministicTree) else {
        return;
    };
    let short = tree.execute(&graph(1), HashMap::new()).unwrap();
    let long = tree.execute(&graph(65), HashMap::new()).unwrap();
    assert_eq!(
        short.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        long.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    let Some(mut serial) = executor(PtxReductionMode::Strict) else {
        return;
    };
    let reference = serial.execute(&graph(65), HashMap::new()).unwrap();
    assert_eq!(reference, vec![0.0; 2]);
    // Independent f64 sum is -30; cancellation makes tree/order observable.
    // Merely returning the legacy serial result cannot satisfy this regression.
    assert_ne!(short, reference);
}

#[test]
fn full_epilogue_preserves_extent_selected_reduction_values() {
    for mode in [
        PtxReductionMode::DeterministicTree,
        PtxReductionMode::Online,
    ] {
        let Some(mut executor) = executor(mode) else {
            return;
        };
        // This regression exercises the lowering itself; cost selection must
        // not make it pass by separating the full-domain epilogue.
        executor.set_fusion_policy(PtxFusionPolicy::Greedy);
        for width in [31, 32, 33, 64, 255, 256] {
            for (axis, shape) in [(0, vec![width, 2]), (1, vec![2, width])] {
                // Independent f64 sum is 2. Serial, ascending subgroup and
                // descending block trees expose different f32 association.
                let mut row = vec![0.0_f32; width];
                row[..4].copy_from_slice(&[16_777_216.0, 1.0, -16_777_216.0, 1.0]);
                let data = if axis == 1 {
                    row.repeat(2)
                } else {
                    row.iter().flat_map(|&value| [value; 2]).collect()
                };
                let input = TensorExpr::constant(data, shape.clone());
                let sum = input.reduce_sum(axis);
                let bare_graph: TensorGraph<f32> = sum.clone().into();
                let bare = executor.execute(&bare_graph, HashMap::new()).unwrap();

                let mut full_graph = TensorGraph::new();
                let sum_node = full_graph.add_expr(&sum, false);
                let reduced_shape = full_graph[sum_node].shape().clone();
                let reduced = TensorExpr::node_ref(sum_node, reduced_shape);
                let full_node =
                    full_graph.add_expr(&(-reduced.clone().broadcast(shape.clone())), false);
                // An escaping reduced-domain consumer checks that the same
                // region correctly exports reduced and full-domain results.
                let reduced_node = full_graph.add_expr(&(-reduced), false);
                executor.execute(&full_graph, HashMap::new()).unwrap();
                assert!(
                    executor
                        .execution_plan()
                        .unwrap()
                        .reduction_regions()
                        .iter()
                        .any(|region| { !region.full_epilogue_operations.is_empty() }),
                    "full epilogue was not lowered: mode={mode:?}, width={width}, axis={axis}"
                );
                let full = executor.get_value(full_node).unwrap();
                let reduced = executor.get_value(reduced_node).unwrap();
                let expected_bits: Vec<_> = bare.iter().map(|value| (-value).to_bits()).collect();
                assert_eq!(
                    reduced
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected_bits,
                    "reduced export: mode={mode:?}, width={width}, axis={axis}"
                );
                for (index, value) in full.iter().enumerate() {
                    let fiber = if axis == 1 { index / width } else { index % 2 };
                    assert_eq!(
                        value.to_bits(),
                        expected_bits[fiber],
                        "full export: mode={mode:?}, width={width}, axis={axis}, index={index}"
                    );
                }
            }
        }
    }
}
