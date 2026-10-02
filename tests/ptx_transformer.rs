#![cfg(feature = "cuda")]

use std::collections::HashMap;

use tnsr::graph::TensorGraph;
use tnsr::nn::{TransformerBlock, layer_norm, softmax};
use tnsr::ptx::{PtxExecutor, PtxReductionMode};
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

fn cooperative_ptx_executor() -> Option<PtxExecutor> {
    match PtxExecutor::try_new_with_reduction_mode(PtxReductionMode::DeterministicTree) {
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

#[test]
fn ptx_cooperative_block_reduction_matches_cpu() {
    let Some(mut ptx) = cooperative_ptx_executor() else {
        return;
    };
    let values: Vec<_> = (0..2 * 1024)
        .map(|index| (index as f32 * 0.001).sin())
        .collect();
    let input = TensorExpr::constant(values, vec![2, 1024]);
    let graph: TensorGraph<f32> = input.exp().reduce_sum(1).into();
    let expected = execute_cpu(&graph);

    ptx.compile_owned(graph.clone()).unwrap();
    assert_eq!(
        ptx.execution_plan().unwrap().reduction_regions()[0].schedule,
        tnsr::tile::ReductionSchedule::Block { threads: 128 }
    );
    let source = ptx.module_source().unwrap();
    assert!(source.contains("reduction_partials[512]"));
    assert!(!source.contains("shfl.sync"));
    assert!(
        ptx.describe_plan(&graph)
            .unwrap()
            .contains("schedule=Block { threads: 128 }")
    );
    let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap();
    assert_close(&actual, &expected, 1e-3);
}

#[test]
fn ptx_cooperative_reductions_cover_schedule_boundaries_and_middle_axes() {
    let Some(mut ptx) = cooperative_ptx_executor() else {
        return;
    };
    for width in [32, 33, 127, 256] {
        let values: Vec<_> = (0..2 * width * 3)
            .map(|index| ((index % 29) as f32 - 14.0) * 0.125)
            .collect();
        let input = || TensorExpr::constant(values.clone(), vec![2, width, 3]);
        let zero = || TensorExpr::constant(vec![0.0; 2 * width * 3], vec![2, width, 3]);
        let graphs: [TensorGraph<f32>; 3] = [
            (input() + zero()).reduce_sum(1).into(),
            (input() + zero()).reduce_mean(1).into(),
            (input() + zero()).reduce_max(1).into(),
        ];

        for graph in graphs {
            let expected = execute_cpu(&graph);
            ptx.compile_owned(graph.clone()).unwrap();
            let expected_schedule = if width < 256 {
                tnsr::tile::ReductionSchedule::Subgroup { width: 32 }
            } else {
                tnsr::tile::ReductionSchedule::Block { threads: 128 }
            };
            assert_eq!(
                ptx.execution_plan().unwrap().reduction_regions()[0].schedule,
                expected_schedule
            );
            let source = ptx.module_source().unwrap();
            assert_eq!(
                source.matches("shfl.sync.down.b32").count(),
                if width < 256 { 5 } else { 0 }
            );
            let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap();
            assert_close(&actual, &expected, 1e-5);
        }
    }
}

#[test]
fn ptx_cooperative_block_fuses_large_softmax_epilogue() {
    let Some(mut ptx) = cooperative_ptx_executor() else {
        return;
    };
    let values: Vec<_> = (0..2 * 1024)
        .map(|index| ((index % 97) as f32 - 48.0) * 0.03125)
        .collect();
    let graph: TensorGraph<f32> = softmax(TensorExpr::constant(values, vec![2, 1024]), 1).into();
    let expected = execute_cpu(&graph);

    ptx.compile_owned(graph.clone()).unwrap();
    let plan = ptx.execution_plan().unwrap();
    assert_eq!(plan.reduction_regions().len(), 1);
    let region = &plan.reduction_regions()[0];
    assert_eq!(
        region.schedule,
        tnsr::tile::ReductionSchedule::Block { threads: 128 }
    );
    assert!(!region.full_epilogue_operations.is_empty());
    let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap();
    assert_close(&actual, &expected, 2e-5);
    assert_eq!(ptx.execution_metrics().kernel_launches, 3);
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
fn ptx_reports_fused_pointwise_plan_metrics() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let input = TensorExpr::constant(vec![0.25, 0.5, 1.0, 2.0], vec![4]);
    let shared = input.exp();
    let graph: TensorGraph<f32> = (shared.clone().log() + shared.relu()).into();

    ptx.compile_owned(graph.clone()).unwrap();
    let compile = ptx.compile_metrics();
    assert_eq!(compile.graph_nodes, 5);
    assert_eq!(compile.generated_kernels, 1);
    assert!(compile.ptx_source_bytes > 0);
    assert!(!compile.compile_time.is_zero());
    let plan = ptx.execution_plan().unwrap();
    assert_eq!(plan.steps().len(), 2);
    assert!(plan.steps().iter().all(|step| step.materialize));
    assert_eq!(
        plan.steps()
            .iter()
            .filter(|step| matches!(step.action, tnsr::ptx::PtxPlanAction::PointwiseRegion(_)))
            .count(),
        1
    );

    let description = ptx.describe_plan(&graph).unwrap();
    assert!(description.contains("PTX plan: 5 graph nodes, 1 physical kernels"));
    assert!(description.contains("action=upload"));
    assert_eq!(description.matches("action=kernel").count(), 1);

    ptx.execute_compiled(&graph, HashMap::new()).unwrap();
    assert_eq!(
        ptx.execution_metrics(),
        &tnsr::ptx::PtxExecutionMetrics {
            kernel_launches: 1,
            materialized_values: 2,
            materialized_bytes: 32,
            intermediate_materialized_bytes: 0,
            device_copies: 2,
        }
    );
}

#[test]
fn ptx_fuses_multi_output_reduction_region() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let input = TensorExpr::constant(vec![0.25, 0.5, 1.0, 2.0], vec![4]);
    let shared = input.exp();
    let branch = shared.clone().log();
    let graph: TensorGraph<f32> = (shared.reduce_sum(0) + branch.reduce_sum(0)).into();

    ptx.compile_owned(graph.clone()).unwrap();
    let plan = ptx.execution_plan().unwrap();
    assert!(plan.regions().is_empty());
    assert_eq!(plan.reduction_regions().len(), 2);
    assert!(
        plan.reduction_regions()
            .iter()
            .any(|region| region.outputs.len() == 2)
    );
    assert_eq!(ptx.compile_metrics().generated_kernels, 2);

    let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap();
    let expected = execute_cpu(&graph);
    assert_close(&actual, &expected, 1e-4);
    assert_eq!(ptx.execution_metrics().kernel_launches, 2);
}

#[test]
fn ptx_fuses_pointwise_producer_reduction_and_epilogue() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let input = TensorExpr::constant(vec![-2.0, -1.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 4]);
    let graph: TensorGraph<f32> = input.relu().exp().reduce_sum(1).log().into();

    ptx.compile_owned(graph.clone()).unwrap();
    let plan = ptx.execution_plan().unwrap();
    assert_eq!(plan.reduction_regions().len(), 1);
    let region = &plan.reduction_regions()[0];
    assert_eq!(region.producer_operations.len(), 2);
    assert_eq!(region.epilogue_operations.len(), 1);
    assert_eq!(region.input_shape, vec![2, 4]);
    assert_eq!(region.output_shape, vec![2, 1]);
    assert_eq!(ptx.compile_metrics().generated_kernels, 1);

    let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap();
    let expected = execute_cpu(&graph);
    assert_close(&actual, &expected, 1e-4);
    assert_eq!(ptx.execution_metrics().kernel_launches, 1);
    assert_eq!(ptx.execution_metrics().materialized_values, 2);
}

#[test]
fn ptx_reduction_regions_support_mean_max_and_middle_axis() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let values = || {
        TensorExpr::constant(
            vec![
                -2.0, 1.0, 4.0, 3.0, -1.0, 2.0, 5.0, 6.0, -3.0, 8.0, 7.0, 9.0,
            ],
            vec![2, 3, 2],
        )
    };
    let graphs: Vec<TensorGraph<f32>> = vec![
        values().relu().reduce_mean(1).exp().into(),
        (-values()).reduce_max(1).relu().into(),
    ];

    for graph in graphs {
        ptx.compile_owned(graph.clone()).unwrap();
        assert_eq!(ptx.execution_plan().unwrap().reduction_regions().len(), 1);
        let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap();
        let expected = execute_cpu(&graph);
        assert_close(&actual, &expected, 1e-3);
        assert_eq!(ptx.execution_metrics().kernel_launches, 1);
    }
}

#[test]
fn ptx_reduction_regions_support_rank_one_first_and_final_axes() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let cases: Vec<TensorGraph<f32>> = vec![
        TensorExpr::constant(vec![-0.2, 0.1, 0.3, 0.4], vec![4])
            .relu()
            .reduce_sum(0)
            .exp()
            .into(),
        TensorExpr::constant(
            (1..=24).map(|value| value as f32 * 0.1).collect(),
            vec![2, 3, 4],
        )
        .relu()
        .reduce_mean(0)
        .exp()
        .into(),
        TensorExpr::constant(
            (1..=24).map(|value| value as f32 * 0.1).collect(),
            vec![2, 3, 4],
        )
        .relu()
        .reduce_max(2)
        .log()
        .into(),
    ];
    for graph in cases {
        ptx.compile_owned(graph.clone()).unwrap();
        assert_eq!(ptx.execution_plan().unwrap().reduction_regions().len(), 1);
        assert_eq!(ptx.compile_metrics().generated_kernels, 1);
        let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap();
        assert_close(&actual, &execute_cpu(&graph), 1e-4);
        assert_eq!(ptx.execution_metrics().kernel_launches, 1);
    }
}

#[test]
fn ptx_empty_reduced_axis_preserves_reduction_identities() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let empty = || TensorExpr::constant(Vec::new(), vec![3, 0]).relu();
    type EmptyReductionCase = (TensorGraph<f32>, fn(f32) -> bool);
    let cases: Vec<EmptyReductionCase> = vec![
        (empty().reduce_sum(1).into(), |value| value == 0.0),
        (empty().reduce_mean(1).into(), |value| value == 0.0),
        (empty().reduce_max(1).into(), |value| {
            value == f32::NEG_INFINITY
        }),
    ];
    for (graph, valid) in cases {
        ptx.compile_owned(graph.clone()).unwrap();
        let plan = ptx.execution_plan().unwrap();
        assert_eq!(plan.reduction_regions().len(), 1);
        assert_eq!(plan.reduction_regions()[0].input_shape, vec![3, 0]);
        assert_eq!(plan.reduction_regions()[0].output_shape, vec![3, 1]);
        let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap();
        let expected = execute_cpu(&graph);
        assert!(actual.iter().copied().all(valid));
        assert!(expected.iter().copied().all(valid));
        assert_eq!(ptx.execution_metrics().kernel_launches, 1);
    }
}

#[test]
fn ptx_plans_independent_reduction_regions_deterministically() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let left = TensorExpr::constant(vec![-2.0, 1.0, 3.0, 4.0, -1.0, 2.0], vec![2, 3]);
    let right = TensorExpr::constant(vec![0.5, -1.0, 2.0, 3.0, 4.0, -2.0, 1.0, 5.0], vec![2, 4]);
    let graph: TensorGraph<f32> =
        (left.relu().reduce_sum(1).log() + right.exp().reduce_mean(1).relu()).into();
    ptx.compile_owned(graph.clone()).unwrap();
    let first_plan = ptx.execution_plan().unwrap().clone();
    assert_eq!(first_plan.reduction_regions().len(), 2);
    ptx.compile_owned(graph.clone()).unwrap();
    assert_eq!(ptx.execution_plan().unwrap(), &first_plan);
    let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap();
    assert_close(&actual, &execute_cpu(&graph), 1e-4);
}

#[test]
fn ptx_reduction_region_exports_producer_and_reduced_values() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let input = TensorExpr::constant(vec![-2.0, 1.0, 3.0, 4.0, -1.0, 2.0], vec![2, 3]);
    let producer = input.relu();
    let reduced = producer.clone().reduce_sum(1);
    let weight = TensorExpr::constant(vec![0.5, -1.0, 2.0], vec![3, 1]);
    let graph: TensorGraph<f32> = (producer.matmul(weight) + reduced).into();
    ptx.compile_owned(graph.clone()).unwrap();
    let plan = ptx.execution_plan().unwrap();
    assert_eq!(plan.reduction_regions().len(), 1);
    assert_eq!(plan.reduction_regions()[0].outputs.len(), 2);
    for output in &plan.reduction_regions()[0].outputs {
        assert!(
            plan.steps()
                .iter()
                .any(|step| step.release_after.contains(&output.node)),
            "escaping region output {} was not assigned a release point",
            output.node.index()
        );
    }
    let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap();
    assert_close(&actual, &execute_cpu(&graph), 1e-5);
}

#[test]
fn ptx_reduction_region_backward_graph_matches_cpu() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let parameter = Parameter::new(vec![-2.0, 1.0, 3.0, 4.0, -1.0, 2.0], vec![2, 3]);
    let parameter_id = parameter.id();
    let loss = TensorExpr::from(parameter)
        .relu()
        .exp()
        .reduce_mean(1)
        .reduce_sum(0);
    let graph: TensorGraph<f32> = loss.into();
    let loss_node = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(loss_node);
    let mut cpu = SimpleExecutor::new();
    cpu.execute(&graph, HashMap::new()).unwrap();
    ptx.execute(&graph, HashMap::new()).unwrap();
    assert!(!ptx.execution_plan().unwrap().reduction_regions().is_empty());
    assert_close(
        &ptx.get_gradients(&graph)[&parameter_id],
        &cpu.get_gradients(&graph)[&parameter_id],
        1e-5,
    );
}

#[test]
fn ptx_reduction_region_preserves_serial_sum_order() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let values = [1.0e20f32, 1.0, -1.0e20, 3.0];
    let expected_sum = values
        .into_iter()
        .fold(0.0f32, |accumulator, value| accumulator + value);
    for mean in [false, true] {
        let input = TensorExpr::constant(values.to_vec(), vec![1, 4]);
        let zero = TensorExpr::constant(vec![0.0; 4], vec![1, 4]);
        let producer = input + zero;
        let graph: TensorGraph<f32> = if mean {
            producer.reduce_mean(1).into()
        } else {
            producer.reduce_sum(1).into()
        };
        ptx.compile_owned(graph.clone()).unwrap();
        let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap()[0];
        let expected = if mean {
            expected_sum / 4.0
        } else {
            expected_sum
        };
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn ptx_softmax_uses_reduction_region_for_exponential_sum() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let logits = TensorExpr::constant(
        vec![
            1000.0, 1001.0, 999.0, 998.0, -1000.0, -999.0, -1001.0, -998.0,
        ],
        vec![2, 4],
    );
    let graph: TensorGraph<f32> = softmax(logits, 1).into();

    ptx.compile_owned(graph.clone()).unwrap();
    let plan = ptx.execution_plan().unwrap();
    assert_eq!(plan.reduction_regions().len(), 1);
    assert_eq!(plan.reduction_regions()[0].outputs.len(), 1);
    assert_eq!(
        plan.reduction_regions()[0].full_epilogue_operations.len(),
        1
    );
    let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap();
    let expected = execute_cpu(&graph);
    assert_close(&actual, &expected, 1e-5);
    assert_eq!(ptx.execution_metrics().kernel_launches, 2);
    for row in actual.chunks_exact(4) {
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
}

#[test]
fn ptx_layer_norm_fuses_variance_and_full_epilogue() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let input = TensorExpr::constant(vec![-2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0], vec![2, 4]);
    let weight = TensorExpr::constant(vec![1.0, 0.5, 1.5, 2.0], vec![4]);
    let bias = TensorExpr::constant(vec![0.0, 0.25, -0.5, 1.0], vec![4]);
    let graph: TensorGraph<f32> = layer_norm(input, 1, weight, bias, 1e-5).into();

    ptx.compile_owned(graph.clone()).unwrap();
    let plan = ptx.execution_plan().unwrap();
    assert_eq!(plan.reduction_regions().len(), 1);
    let variance = &plan.reduction_regions()[0];
    assert!(!variance.producer_operations.is_empty());
    assert!(!variance.full_epilogue_operations.is_empty());
    let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap();
    let expected = execute_cpu(&graph);
    assert_close(&actual, &expected, 2e-4);
}

#[test]
fn ptx_rms_like_normalization_fuses_complete_epilogue() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let input = TensorExpr::constant(vec![-2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0], vec![2, 4]);
    let square = input.clone() * input.clone();
    let mean_square = square.reduce_mean(1);
    let inv_rms = ((mean_square + 1e-5).log() * -0.5)
        .exp()
        .broadcast(vec![2, 4]);
    let graph: TensorGraph<f32> = (input * inv_rms).into();

    ptx.compile_owned(graph.clone()).unwrap();
    let plan = ptx.execution_plan().unwrap();
    assert_eq!(plan.reduction_regions().len(), 1);
    let region = &plan.reduction_regions()[0];
    assert_eq!(region.producer_operations.len(), 1);
    assert!(!region.epilogue_operations.is_empty());
    assert_eq!(region.full_epilogue_operations.len(), 1);
    let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap();
    let expected = execute_cpu(&graph);
    assert_close(&actual, &expected, 2e-4);
    assert_eq!(ptx.execution_metrics().kernel_launches, 1);
}

#[test]
fn ptx_fused_region_preserves_operand_and_mask_order() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let lhs = TensorExpr::constant(vec![4.0, 2.0, 9.0, 1.0], vec![4]);
    let rhs = TensorExpr::constant(vec![2.0, 3.0, 3.0, 4.0], vec![4]);
    let ratio = (lhs.clone() - rhs.clone()) / (lhs.clone() + rhs.clone());
    let graph: TensorGraph<f32> = ratio.mask(lhs.gt(rhs)).relu().into();

    ptx.compile_owned(graph.clone()).unwrap();
    assert_eq!(ptx.execution_plan().unwrap().regions().len(), 1);
    let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap();
    let expected = execute_cpu(&graph);
    assert_close(&actual, &expected, 1e-6);
    assert_eq!(ptx.execution_metrics().kernel_launches, 1);
}

#[test]
fn ptx_zero_size_pointwise_region_does_not_launch() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let graph: TensorGraph<f32> = TensorExpr::constant(Vec::new(), vec![0, 3])
        .exp()
        .relu()
        .into();

    ptx.compile_owned(graph.clone()).unwrap();
    assert_eq!(ptx.execution_plan().unwrap().regions().len(), 1);
    assert!(
        ptx.execute_compiled(&graph, HashMap::new())
            .unwrap()
            .is_empty()
    );
    assert_eq!(ptx.execution_metrics().kernel_launches, 0);
}

#[test]
fn ptx_region_composes_bias_broadcast_access() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let matrix = TensorExpr::constant(vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], vec![2, 3]);
    let bias = TensorExpr::constant(vec![0.5, 1.0, 1.5], vec![3]);
    let graph: TensorGraph<f32> = (matrix + bias).relu().into();

    ptx.compile_owned(graph.clone()).unwrap();
    let plan = ptx.execution_plan().unwrap();
    assert_eq!(plan.regions().len(), 1);
    assert_eq!(
        plan.steps()
            .iter()
            .filter(|step| step.action == tnsr::ptx::PtxPlanAction::VirtualView)
            .count(),
        2
    );
    assert_eq!(ptx.compile_metrics().generated_kernels, 1);
    let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap();
    let expected = execute_cpu(&graph);
    assert_close(&actual, &expected, 0.0);
    assert_eq!(ptx.execution_metrics().kernel_launches, 1);
    assert_eq!(ptx.execution_metrics().materialized_values, 3);
}

#[test]
fn ptx_region_composes_permutation_access() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let input = TensorExpr::constant(vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], vec![2, 3]);
    let residual = TensorExpr::constant(vec![0.5; 6], vec![3, 2]);
    let graph: TensorGraph<f32> = (input.permute(vec![1, 0]) + residual).relu().into();

    ptx.compile_owned(graph.clone()).unwrap();
    let plan = ptx.execution_plan().unwrap();
    assert!(plan.steps().iter().any(|step| {
        step.operation == "Permute" && step.action == tnsr::ptx::PtxPlanAction::VirtualView
    }));
    let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap();
    let expected = execute_cpu(&graph);
    assert_close(&actual, &expected, 0.0);
    assert_eq!(ptx.execution_metrics().kernel_launches, 1);
}

#[test]
fn ptx_materializes_permutation_for_opaque_shared_consumer() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let input = TensorExpr::constant(vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], vec![2, 3]);
    let residual = TensorExpr::constant(vec![0.5; 6], vec![3, 2]);
    let permuted = input.permute(vec![1, 0]);
    let branch = (permuted.clone() + residual).relu().reduce_sum(0);
    let graph: TensorGraph<f32> = (branch + permuted.reduce_sum(0)).into();

    ptx.compile_owned(graph.clone()).unwrap();
    let plan = ptx.execution_plan().unwrap();
    assert!(plan.steps().iter().any(|step| {
        step.operation == "Permute" && step.action == tnsr::ptx::PtxPlanAction::Kernel
    }));
    let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap();
    let expected = execute_cpu(&graph);
    assert_close(&actual, &expected, 1e-6);
    assert_eq!(ptx.execution_metrics().kernel_launches, 3);
}

#[test]
fn ptx_materializes_non_identity_view_for_high_region_fanout() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let input = TensorExpr::constant(vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], vec![2, 3]);
    let permuted = input.permute(vec![1, 0]);
    let one = || TensorExpr::constant(vec![1.0; 6], vec![3, 2]);
    let first = (permuted.clone() + one()).relu().reduce_sum(0);
    let second = (permuted.clone() * one()).exp().reduce_sum(0);
    let third = (permuted - one()).relu().reduce_sum(0);
    let graph: TensorGraph<f32> = (first + second + third).into();

    ptx.compile_owned(graph.clone()).unwrap();
    let plan = ptx.execution_plan().unwrap();
    assert!(plan.steps().iter().any(|step| {
        step.operation == "Permute" && step.action == tnsr::ptx::PtxPlanAction::Kernel
    }));
    let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap();
    let expected = execute_cpu(&graph);
    assert_close(&actual, &expected, 1e-4);
}

#[test]
fn ptx_reports_virtual_view_baseline() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let graph: TensorGraph<f32> =
        TensorExpr::constant((0..6).map(|value| value as f32).collect(), vec![6])
            .reshape(vec![2, 3])
            .flatten()
            .into();

    ptx.compile_owned(graph.clone()).unwrap();
    let description = ptx.describe_plan(&graph).unwrap();
    assert_eq!(description.matches("action=virtual-view").count(), 2);

    let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap();
    assert_eq!(actual, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let execution = ptx.execution_metrics();
    assert_eq!(execution.kernel_launches, 0);
    assert_eq!(execution.materialized_values, 1);
    assert_eq!(execution.materialized_bytes, 24);
    assert_eq!(execution.intermediate_materialized_bytes, 0);
    assert_eq!(execution.device_copies, 2);
}

#[test]
fn ptx_plan_normalizes_composed_virtual_views() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let graph: TensorGraph<f32> =
        TensorExpr::constant((0..24).map(|value| value as f32).collect(), vec![2, 3, 4])
            .reshape(vec![1, 4, 6])
            .permute(vec![0, 2, 1])
            .broadcast_axis(0, 5)
            .into();

    ptx.compile_owned(graph.clone()).unwrap();
    let plan = ptx.execution_plan().unwrap();
    let source = plan.steps().first().unwrap().node;
    let graph_output = *graph.toposort().last().unwrap();
    let output = plan.virtual_value(graph_output).unwrap();
    assert_eq!(output.source, source);
    assert_eq!(output.shape, vec![5, 6, 4]);
    assert_eq!(output.source_index(&[4, 5, 3]).unwrap(), vec![1, 2, 3]);
    let first_plan = plan.clone();
    ptx.compile_owned(graph.clone()).unwrap();
    assert_eq!(ptx.execution_plan().unwrap(), &first_plan);

    let actual = ptx.execute_compiled(&graph, HashMap::new()).unwrap();
    let expected = execute_cpu(&graph);
    assert_close(&actual, &expected, 0.0);
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
