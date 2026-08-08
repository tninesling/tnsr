#![cfg(feature = "cuda")]

use std::collections::HashMap;

use tnsr::cuda::CudaExecutor;
use tnsr::graph::TensorGraph;
use tnsr::nn::{TransformerBlock, scaled_dot_product_attention};
use tnsr::tensor::{Constant, Parameter, TensorExpr};
use tnsr::{Executor, SimpleExecutor};

fn cuda_executor() -> Option<CudaExecutor> {
    match CudaExecutor::try_new() {
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
        Err(error) => panic!("CudaExecutor initialization failed: {error:#}"),
    }
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

fn execute_cpu(graph: &TensorGraph<f32>) -> Vec<f32> {
    SimpleExecutor::new()
        .execute(graph, HashMap::new())
        .expect("CPU reference execution failed")
}

fn reference_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    sequence: usize,
    width: usize,
) -> Vec<f32> {
    let scale = (width as f32).sqrt().recip();
    let mut output = vec![0.0; sequence * width];

    for query_row in 0..sequence {
        let scores: Vec<f32> = (0..=query_row)
            .map(|key_row| {
                (0..width)
                    .map(|column| query[query_row * width + column] * key[key_row * width + column])
                    .sum::<f32>()
                    * scale
            })
            .collect();
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator: f32 = scores.iter().map(|score| (score - max).exp()).sum();

        for (key_row, score) in scores.iter().enumerate() {
            let probability = (*score - max).exp() / denominator;
            for column in 0..width {
                output[query_row * width + column] += probability * value[key_row * width + column];
            }
        }
    }

    output
}

#[test]
fn cuda_reshape_then_rank_three_permute_matches_expected() {
    let Some(mut cuda) = cuda_executor() else {
        return;
    };
    let expression =
        TensorExpr::constant((0..12).map(|value| value as f32).collect(), vec![2, 2, 3])
            .reshape(vec![2, 3, 2])
            .permute(vec![0, 2, 1]);
    let graph: TensorGraph<f32> = expression.into();

    let actual = cuda.execute(&graph, HashMap::new()).unwrap();
    assert_close(
        &actual,
        &[0.0, 2.0, 4.0, 1.0, 3.0, 5.0, 6.0, 8.0, 10.0, 7.0, 9.0, 11.0],
        0.0,
    );
}

#[test]
fn cuda_broadcasted_batched_matmul_matches_expected() {
    let Some(mut cuda) = cuda_executor() else {
        return;
    };
    let left = TensorExpr::constant(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![2, 2, 3],
    );
    let right = TensorExpr::constant(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![1, 3, 2]);
    let graph: TensorGraph<f32> = left.matmul(right).into();

    let actual = cuda.execute(&graph, HashMap::new()).unwrap();
    assert_close(
        &actual,
        &[4.0, 5.0, 10.0, 11.0, 16.0, 17.0, 22.0, 23.0],
        1e-5,
    );
}

#[test]
fn cuda_two_sided_batched_matmul_broadcast_matches_cpu() {
    let Some(mut cuda) = cuda_executor() else {
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
    let actual = cuda.execute(&graph, HashMap::new()).unwrap();
    assert_close(&actual, &expected, 1e-5);
}

#[test]
fn cuda_middle_axis_reduction_matches_expected() {
    let Some(mut cuda) = cuda_executor() else {
        return;
    };
    let expression = TensorExpr::constant(
        vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
        vec![2, 2, 2],
    )
    .reduce_axis_sum(1);
    let graph: TensorGraph<f32> = expression.into();

    let actual = cuda.execute(&graph, HashMap::new()).unwrap();
    assert_close(&actual, &[3.0, 30.0, 7.0, 70.0], 0.0);
}

#[test]
fn cuda_causal_sdpa_matches_reference() {
    let Some(mut cuda) = cuda_executor() else {
        return;
    };
    let query = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let key = vec![1.0, 0.0, 0.0, 1.0, 1.0, -1.0];
    let value = vec![10.0, 1.0, 20.0, 2.0, 30.0, 4.0];
    let tensor = |data| Constant::new(data, vec![1, 3, 2]);
    let expression = scaled_dot_product_attention(
        tensor(query.clone()),
        tensor(key.clone()),
        tensor(value.clone()),
        true,
    );
    let graph: TensorGraph<f32> = expression.into();

    let actual = cuda.execute(&graph, HashMap::new()).unwrap();
    let expected = reference_attention(&query, &key, &value, 3, 2);
    assert_close(&actual, &expected, 1e-4);
}

#[test]
fn cuda_tiny_transformer_forward_and_backward_match_cpu() {
    let Some(mut cuda) = cuda_executor() else {
        return;
    };
    let block = TransformerBlock::new(2, 1, 2, 1e-5);
    let input = Parameter::new(vec![0.25, -0.5, 0.75, 0.1], vec![1, 2, 2]);
    let input_id = input.id();
    let output = block.forward(input, true);
    let forward_graph: TensorGraph<f32> = output.clone().into();

    let expected_output = execute_cpu(&forward_graph);
    let actual_output = cuda.execute(&forward_graph, HashMap::new()).unwrap();
    assert_close(&actual_output, &expected_output, 1e-4);

    let graph: TensorGraph<f32> = output.mean_all().into();
    let loss_node = *graph.toposort().last().unwrap();
    let gradient_graph = graph.with_gradients(loss_node);
    let mut cpu = SimpleExecutor::new();
    cpu.execute(&gradient_graph, HashMap::new()).unwrap();
    let expected_gradients = cpu.get_gradients(&gradient_graph);

    cuda.execute(&gradient_graph, HashMap::new()).unwrap();
    let actual_gradients = cuda.get_gradients(&gradient_graph);
    let parameter_ids = [
        input_id,
        block.attention.query_weight.id(),
        block.attention.query_bias.id(),
        block.attention.key_weight.id(),
        block.attention.key_bias.id(),
        block.attention.value_weight.id(),
        block.attention.value_bias.id(),
        block.attention.output_weight.id(),
        block.attention.output_bias.id(),
        block.attention_norm_weight.id(),
        block.attention_norm_bias.id(),
        block.feed_forward_norm_weight.id(),
        block.feed_forward_norm_bias.id(),
        block.feed_forward_input_weight.id(),
        block.feed_forward_input_bias.id(),
        block.feed_forward_output_weight.id(),
        block.feed_forward_output_bias.id(),
    ];

    for parameter_id in parameter_ids {
        let expected = expected_gradients
            .get(&parameter_id)
            .unwrap_or_else(|| panic!("CPU gradient missing for parameter {parameter_id}"));
        let actual = actual_gradients
            .get(&parameter_id)
            .unwrap_or_else(|| panic!("CUDA gradient missing for parameter {parameter_id}"));
        assert_close(actual, expected, 2e-3);
    }
}
