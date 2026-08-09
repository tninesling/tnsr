#![cfg(feature = "cuda")]

use std::collections::HashMap;

use tnsr::graph::TensorGraph;
use tnsr::nn::TransformerBlock;
use tnsr::ptx::PtxExecutor;
use tnsr::tensor::{Parameter, TensorExpr};
use tnsr::{Executor, SimpleExecutor};

fn ptx_executor() -> Option<PtxExecutor> {
    match PtxExecutor::try_new() {
        Ok(executor) => Some(executor),
        Err(error)
            if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                && error
                    .to_string()
                    .starts_with("Failed to initialize CUDA device 0") =>
        {
            eprintln!("skipping CUDA test: {error:#}");
            None
        }
        Err(error) => panic!("PtxExecutor initialization failed: {error:#}"),
    }
}

fn execute_cpu(graph: &TensorGraph<f32>) -> Vec<f32> {
    SimpleExecutor::new()
        .execute(graph, HashMap::new())
        .expect("CPU reference execution failed")
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "element {index}: expected {expected}, got {actual} (tolerance {tolerance})"
        );
    }
}

#[test]
fn ptx_reports_unfused_pointwise_plan_metrics() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let input = TensorExpr::constant(vec![0.25, 0.5, 1.0, 2.0], vec![4]);
    let shared = input.exp();
    let graph: TensorGraph<f32> = (shared.clone().log() + shared.relu()).into();

    ptx.compile_owned(graph.clone()).unwrap();
    let compile = ptx.compile_metrics();
    assert_eq!(compile.graph_nodes, 5);
    assert_eq!(compile.generated_kernels, 5);
    assert!(compile.ptx_source_bytes > 0);
    assert!(!compile.compile_time.is_zero());
    let plan = ptx.execution_plan().unwrap();
    assert_eq!(plan.steps().len(), 5);
    assert!(plan.steps().iter().all(|step| step.materialize));
    assert_eq!(
        plan.steps()
            .iter()
            .filter(|step| step.action == tnsr::ptx::PtxPlanAction::Kernel)
            .count(),
        4
    );

    let description = ptx.describe_unfused_plan(&graph).unwrap();
    assert!(description.contains("unfused PTX plan: 5 graph nodes, 5 generated kernels"));
    assert!(description.contains("action=upload"));
    assert_eq!(description.matches("action=kernel").count(), 4);

    ptx.execute_compiled(&graph, HashMap::new()).unwrap();
    assert_eq!(
        ptx.execution_metrics(),
        &tnsr::ptx::PtxExecutionMetrics {
            kernel_launches: 4,
            materialized_values: 5,
            materialized_bytes: 80,
            intermediate_materialized_bytes: 48,
            device_copies: 2,
        }
    );
}

#[test]
fn ptx_reports_view_copy_baseline() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let graph: TensorGraph<f32> =
        TensorExpr::constant((0..6).map(|value| value as f32).collect(), vec![6])
            .reshape(vec![2, 3])
            .flatten()
            .into();

    ptx.compile_owned(graph.clone()).unwrap();
    let description = ptx.describe_unfused_plan(&graph).unwrap();
    assert_eq!(description.matches("action=device-copy").count(), 2);

    ptx.execute_compiled(&graph, HashMap::new()).unwrap();
    let execution = ptx.execution_metrics();
    assert_eq!(execution.kernel_launches, 0);
    assert_eq!(execution.materialized_values, 3);
    assert_eq!(execution.materialized_bytes, 72);
    assert_eq!(execution.intermediate_materialized_bytes, 24);
    assert_eq!(execution.device_copies, 4);
}

#[test]
fn ptx_elementwise_length_17_is_bounds_safe() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let expression =
        TensorExpr::constant((-8..9).map(|value| value as f32).collect(), vec![17]).relu();
    let graph: TensorGraph<f32> = expression.into();

    let actual = ptx.execute(&graph, HashMap::new()).unwrap();
    let expected = execute_cpu(&graph);
    assert_close(&actual, &expected, 0.0);
}

#[test]
fn ptx_repeated_execute_compiles_same_structure_once() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let graph: TensorGraph<f32> = TensorExpr::constant(vec![-1.0, 2.0], vec![2]).relu().into();

    ptx.execute(&graph, HashMap::new()).unwrap();
    assert_eq!(ptx.compilation_count(), 1);
    ptx.reset_stats();
    ptx.clear_pool();
    ptx.execute(&graph, HashMap::new()).unwrap();
    assert_eq!(ptx.compilation_count(), 1);
}

#[test]
fn ptx_changed_graph_structure_recompiles() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let left = || TensorExpr::constant(vec![1.0, 2.0], vec![2]);
    let right = || TensorExpr::constant(vec![3.0, 4.0], vec![2]);
    let add_graph: TensorGraph<f32> = (left() + right()).into();
    let multiply_graph: TensorGraph<f32> = (left() * right()).into();

    assert_close(
        &ptx.execute(&add_graph, HashMap::new()).unwrap(),
        &[4.0, 6.0],
        0.0,
    );
    assert_eq!(ptx.compilation_count(), 1);
    assert_close(
        &ptx.execute(&multiply_graph, HashMap::new()).unwrap(),
        &[3.0, 8.0],
        0.0,
    );
    assert_eq!(ptx.compilation_count(), 2);
}

#[test]
fn ptx_parameter_update_changes_output_without_recompile() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let parameter = Parameter::new(vec![1.0, 2.0], vec![2]);
    let parameter_id = parameter.id();
    let graph: TensorGraph<f32> =
        (TensorExpr::from(parameter) * TensorExpr::constant(vec![2.0, 2.0], vec![2])).into();

    assert_close(
        &ptx.execute(&graph, HashMap::new()).unwrap(),
        &[2.0, 4.0],
        0.0,
    );
    assert_eq!(ptx.compilation_count(), 1);
    for node in graph.graph.node_weights() {
        if let tnsr::graph::TensorGraphNode::Parameter { id, data, .. } = node
            && *id == parameter_id
        {
            *data.lock().unwrap() = vec![3.0, 4.0];
        }
    }
    assert_close(
        &ptx.execute(&graph, HashMap::new()).unwrap(),
        &[6.0, 8.0],
        0.0,
    );
    assert_eq!(ptx.compilation_count(), 1);
}

#[test]
fn ptx_execute_compiled_rejects_different_graph_structure() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let compiled: TensorGraph<f32> = TensorExpr::constant(vec![1.0, 2.0], vec![2]).relu().into();
    let different: TensorGraph<f32> = (-TensorExpr::constant(vec![1.0, 2.0], vec![2])).into();
    ptx.compile_owned(compiled).unwrap();

    let error = ptx
        .execute_compiled(&different, HashMap::new())
        .unwrap_err();
    assert!(error.to_string().contains("does not match"), "{error:#}");
}

#[test]
fn ptx_conv2d_compiles_and_executes_without_fallback() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let input = TensorExpr::constant(vec![1.0; 9], vec![1, 1, 3, 3]);
    let weight = TensorExpr::constant(vec![1.0; 4], vec![1, 1, 2, 2]);
    let graph: TensorGraph<f32> = input.conv2d(weight, 1, 0).into();

    ptx.compile_owned(graph.clone()).unwrap();
    assert_eq!(ptx.compilation_count(), 1);
    assert_close(
        &ptx.execute_compiled(&graph, HashMap::new()).unwrap(),
        &[4.0; 4],
        0.0,
    );
}

#[test]
fn ptx_input_length_must_match_declared_shape() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let graph: TensorGraph<f32> = TensorExpr::input("x", vec![3]).relu().into();
    for values in [vec![1.0, 2.0], vec![1.0, 2.0, 3.0, 4.0]] {
        let error = ptx
            .execute(&graph, HashMap::from([("x".to_string(), values)]))
            .unwrap_err();
        assert!(
            error.to_string().contains("declared shape [3]"),
            "{error:#}"
        );
    }
    assert_eq!(ptx.compilation_count(), 1);
}

#[test]
fn ptx_zero_size_operations_return_empty_without_launching() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let empty = || TensorExpr::constant(Vec::new(), vec![0]);
    let expressions = vec![
        empty().relu(),
        empty() + empty(),
        empty().gt(empty()),
        empty().mask(empty()),
        TensorExpr::constant(Vec::new(), vec![0, 1]).broadcast_axis(1, 3),
        TensorExpr::constant(Vec::new(), vec![0, 3]).reduce_axis_sum(1),
        TensorExpr::constant(Vec::new(), vec![3, 0]).transpose(),
        empty().reshape(vec![0, 1]),
        TensorExpr::constant(Vec::new(), vec![0, 2, 3]).flatten(),
    ];
    for expression in expressions {
        let graph: TensorGraph<f32> = expression.into();
        assert!(ptx.execute(&graph, HashMap::new()).unwrap().is_empty());
    }
}

#[cfg(feature = "fusion")]
#[test]
fn ptx_zero_size_fused_unary_returns_empty_without_launching() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let mut graph: TensorGraph<f32> = TensorExpr::constant(Vec::new(), vec![0]).exp().log().into();
    assert_eq!(graph.apply_fusion(), 1);
    assert!(ptx.execute(&graph, HashMap::new()).unwrap().is_empty());
}

#[test]
fn ptx_implicit_broadcasting_matches_tensor_semantics() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let scalar = || TensorExpr::constant(vec![2.0], vec![]);
    let vector = || TensorExpr::constant(vec![1.0, 3.0, 5.0], vec![3]);
    let cases = [
        (scalar() + vector(), vec![3.0, 5.0, 7.0]),
        (vector() + scalar(), vec![3.0, 5.0, 7.0]),
        (scalar() - vector(), vec![1.0, -1.0, -3.0]),
        (vector() - scalar(), vec![-1.0, 1.0, 3.0]),
        (vector().gt(scalar()), vec![0.0, 1.0, 1.0]),
        (scalar().gt(vector()), vec![1.0, 0.0, 0.0]),
        (vector().mask(scalar()), vec![1.0, 3.0, 5.0]),
        (
            scalar().mask(TensorExpr::constant(vec![1.0, 0.0, 1.0], vec![3])),
            vec![2.0, 0.0, 2.0],
        ),
    ];
    for (expression, expected) in cases {
        let graph: TensorGraph<f32> = expression.into();
        assert_close(
            &ptx.execute(&graph, HashMap::new()).unwrap(),
            &expected,
            0.0,
        );
    }

    let rank_two = || TensorExpr::constant(vec![1.0, 2.0, 3.0], vec![3, 1]);
    let rank_three =
        || TensorExpr::constant((0..24).map(|value| value as f32).collect(), vec![2, 3, 4]);
    for expression in [rank_two() - rank_three(), rank_three() - rank_two()] {
        let graph: TensorGraph<f32> = expression.into();
        let expected = execute_cpu(&graph);
        let actual = ptx.execute(&graph, HashMap::new()).unwrap();
        assert_close(&actual, &expected, 0.0);
    }
}

#[test]
fn ptx_rank_three_broadcast_and_reduction_are_axis_generic() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let cases = [
        TensorExpr::constant((1..=6).map(|x| x as f32).collect(), vec![2, 1, 3])
            .broadcast_axis(1, 4),
        TensorExpr::constant((1..=6).map(|x| x as f32).collect(), vec![2, 3, 1])
            .broadcast_axis(2, 5),
        TensorExpr::constant((1..=24).map(|x| x as f32).collect(), vec![2, 4, 3])
            .reduce_axis_sum(1),
        TensorExpr::constant((1..=30).map(|x| x as f32).collect(), vec![2, 3, 5])
            .reduce_axis_mean(2),
    ];

    for expression in cases {
        let graph: TensorGraph<f32> = expression.into();
        let expected = execute_cpu(&graph);
        let actual = ptx.execute(&graph, HashMap::new()).unwrap();
        assert_close(&actual, &expected, 1e-5);
    }
}

#[test]
fn ptx_reshape_then_rank_three_arbitrary_permute_matches_cpu() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let expression =
        TensorExpr::constant((0..24).map(|value| value as f32).collect(), vec![2, 3, 4])
            .reshape(vec![4, 2, 3])
            .permute(vec![2, 0, 1]);
    let graph: TensorGraph<f32> = expression.into();

    let expected = execute_cpu(&graph);
    let actual = ptx.execute(&graph, HashMap::new()).unwrap();
    assert_close(&actual, &expected, 0.0);
}

#[test]
fn ptx_rectangular_matmul_matches_cpu() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let left = TensorExpr::constant((1..=6).map(|x| x as f32).collect(), vec![2, 3]);
    let right = TensorExpr::constant((1..=15).map(|x| x as f32 / 10.0).collect(), vec![3, 5]);
    let graph: TensorGraph<f32> = left.matmul(right).into();

    let expected = execute_cpu(&graph);
    let actual = ptx.execute(&graph, HashMap::new()).unwrap();
    assert_close(&actual, &expected, 1e-5);
}

#[test]
fn ptx_matmul_predicates_15_16_17_and_rectangular_boundaries() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    for (m, k, n) in [(15, 15, 15), (16, 16, 16), (17, 17, 17), (19, 23, 29)] {
        let left = TensorExpr::constant(
            (0..m * k)
                .map(|index| (index % 13) as f32 / 13.0 - 0.5)
                .collect(),
            vec![m, k],
        );
        let right = TensorExpr::constant(
            (0..k * n)
                .map(|index| (index % 17) as f32 / 17.0 - 0.5)
                .collect(),
            vec![k, n],
        );
        let graph: TensorGraph<f32> = left.matmul(right).into();
        let expected = execute_cpu(&graph);
        let actual = ptx.execute(&graph, HashMap::new()).unwrap();
        assert_close(&actual, &expected, 2e-4);
    }
}

#[test]
fn ptx_tf32_matmul_matches_cpu() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    for (m, k, n) in [(32, 32, 32), (33, 40, 35)] {
        let left = TensorExpr::constant(
            (0..m * k)
                .map(|index| (index % 29) as f32 / 29.0 - 0.5)
                .collect(),
            vec![m, k],
        );
        let right = TensorExpr::constant(
            (0..k * n)
                .map(|index| (index % 31) as f32 / 31.0 - 0.5)
                .collect(),
            vec![k, n],
        );
        let graph: TensorGraph<f32> = left.matmul(right).into();

        let expected = execute_cpu(&graph);
        let actual = ptx.execute(&graph, HashMap::new()).unwrap();
        assert_close(&actual, &expected, 2e-3);
    }
}

#[test]
fn ptx_broadcasted_batched_matmul_matches_cpu() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let left = TensorExpr::constant((1..=12).map(|value| value as f32).collect(), vec![2, 2, 3]);
    let right = TensorExpr::constant(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![1, 3, 2]);
    let graph: TensorGraph<f32> = left.matmul(right).into();

    let expected = execute_cpu(&graph);
    let actual = ptx.execute(&graph, HashMap::new()).unwrap();
    assert_close(&actual, &expected, 1e-5);
}

#[test]
fn ptx_two_sided_batched_matmul_broadcast_matches_cpu() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let left = TensorExpr::constant(
        (1..=12).map(|value| value as f32 / 10.0).collect(),
        vec![2, 1, 2, 3],
    );
    let right = TensorExpr::constant(
        (1..=18).map(|value| value as f32 / 20.0).collect(),
        vec![1, 3, 3, 2],
    );
    let graph: TensorGraph<f32> = left.matmul(right).into();

    let expected = execute_cpu(&graph);
    let actual = ptx.execute(&graph, HashMap::new()).unwrap();
    assert_close(&actual, &expected, 1e-5);
}

#[test]
fn ptx_tiny_transformer_block_forward_matches_cpu() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let block = TransformerBlock::new(2, 1, 2, 1e-5);
    let input = Parameter::new(vec![0.25, -0.5, 0.75, 0.1], vec![1, 2, 2]);
    let graph: TensorGraph<f32> = block.forward(input, true).into();

    let expected = execute_cpu(&graph);
    let actual = ptx.execute(&graph, HashMap::new()).unwrap();
    assert_close(&actual, &expected, 2e-4);
}
