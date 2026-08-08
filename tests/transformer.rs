use std::collections::HashMap;

use tnsr::graph::TensorGraph;
use tnsr::nn::{TransformerBlock, layer_norm, scaled_dot_product_attention, softmax};
use tnsr::tensor::{Constant, Parameter, TensorExpr};
use tnsr::{Executor, Runtime};

const EPSILON: f32 = 1e-5;

fn run(expr: TensorExpr<f32>) -> Vec<f32> {
    let graph: TensorGraph<f32> = expr.into();
    Runtime::new().execute(&graph, HashMap::new()).unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: {actual} != {expected} (tolerance {tolerance})"
        );
    }
}

fn reference_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    sequence: usize,
    width: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = (width as f32).sqrt().recip();
    let mut output = vec![0.0; sequence * width];
    for query_row in 0..sequence {
        let mut scores = vec![f32::NEG_INFINITY; sequence];
        for key_row in 0..sequence {
            if !causal || key_row <= query_row {
                scores[key_row] = (0..width)
                    .map(|column| query[query_row * width + column] * key[key_row * width + column])
                    .sum::<f32>()
                    * scale;
            }
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator: f32 = scores.iter().map(|score| (score - max).exp()).sum();
        for key_row in 0..sequence {
            let probability = (scores[key_row] - max).exp() / denominator;
            for column in 0..width {
                output[query_row * width + column] += probability * value[key_row * width + column];
            }
        }
    }
    output
}

#[test]
fn reshape_permute_and_swap_axes_preserve_row_major_values() {
    let values: Vec<f32> = (0..24).map(|value| value as f32).collect();
    let x = TensorExpr::from(Constant::new(values, vec![2, 3, 4]));

    let reshaped = x.clone().reshape(vec![4, 6]);
    assert_eq!(reshaped.shape(), &vec![4, 6]);
    assert_close(
        &run(reshaped),
        &(0..24).map(|v| v as f32).collect::<Vec<_>>(),
        0.0,
    );

    let permuted = x.clone().permute(vec![1, 0, 2]);
    assert_eq!(permuted.shape(), &vec![3, 2, 4]);
    let expected: Vec<f32> = (0..3)
        .flat_map(|middle| {
            (0..2).flat_map(move |outer| {
                (0..4).map(move |inner| (outer * 12 + middle * 4 + inner) as f32)
            })
        })
        .collect();
    assert_close(&run(permuted), &expected, 0.0);
    assert_close(&run(x.swap_axes(0, 1)), &expected, 0.0);
}

#[test]
fn batched_matmul_uses_independent_batches() {
    let left = Constant::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 0.0, 1.0, 2.0, -2.0, 1.0],
        vec![2, 2, 3],
    );
    let right = Constant::new(
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0, 1.0, 3.0, -2.0, 2.0],
        vec![2, 3, 2],
    );
    let output = TensorExpr::from(left).matmul(right);

    assert_eq!(output.shape(), &vec![2, 2, 2]);
    assert_close(
        &run(output),
        &[4.0, 5.0, 10.0, 11.0, -4.0, 3.0, 0.0, -6.0],
        EPSILON,
    );
}

#[test]
fn softmax_is_stable_and_normalizes_each_row() {
    let logits = Constant::new(
        vec![1000.0, 1001.0, 1002.0, -1000.0, -1000.0, -1000.0],
        vec![2, 3],
    );
    let output = run(softmax(logits, 1));
    let denominator = 1.0 + std::f32::consts::E + std::f32::consts::E.powi(2);

    assert!(output.iter().all(|value| value.is_finite()));
    assert_close(
        &output,
        &[
            1.0 / denominator,
            std::f32::consts::E / denominator,
            std::f32::consts::E.powi(2) / denominator,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ],
        1e-5,
    );
    for row in output.chunks_exact(3) {
        assert!((row.iter().sum::<f32>() - 1.0).abs() < EPSILON);
    }
}

#[test]
fn attention_matches_reference_and_causal_rows_ignore_the_future() {
    let query = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let key = vec![1.0, 0.0, 0.0, 1.0, 1.0, -1.0];
    let value = vec![10.0, 1.0, 20.0, 2.0, 30.0, 4.0];
    let tensor = |data| Constant::new(data, vec![1, 3, 2]);

    let output = run(scaled_dot_product_attention(
        tensor(query.clone()),
        tensor(key.clone()),
        tensor(value.clone()),
        false,
    ));
    assert_close(
        &output,
        &reference_attention(&query, &key, &value, 3, 2, false),
        1e-4,
    );

    let causal = run(scaled_dot_product_attention(
        tensor(query.clone()),
        tensor(key.clone()),
        tensor(value.clone()),
        true,
    ));
    assert_close(
        &causal,
        &reference_attention(&query, &key, &value, 3, 2, true),
        1e-4,
    );
    assert_close(&causal[..2], &value[..2], EPSILON);

    let mut changed_future = value;
    changed_future[4..].copy_from_slice(&[30_000.0, -40_000.0]);
    let changed = run(scaled_dot_product_attention(
        tensor(query),
        tensor(key),
        tensor(changed_future),
        true,
    ));
    assert_close(&changed[..4], &causal[..4], EPSILON);
}

#[test]
fn layer_norm_has_zero_mean_and_unit_variance_per_row() {
    let input = Constant::new(vec![1.0, 2.0, 4.0, 8.0, -3.0, -1.0, 1.0, 3.0], vec![2, 4]);
    let weight = Constant::new(vec![1.0; 4], vec![4]);
    let bias = Constant::new(vec![0.0; 4], vec![4]);
    let output = run(layer_norm(input, 1, weight, bias, 1e-7));

    for row in output.chunks_exact(4) {
        let mean = row.iter().sum::<f32>() / row.len() as f32;
        let variance =
            row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / row.len() as f32;
        assert!(mean.abs() < EPSILON, "row mean was {mean}");
        assert!((variance - 1.0).abs() < 1e-4, "row variance was {variance}");
    }
}

#[test]
fn softmax_gradient_matches_closed_form() {
    let logits = Parameter::new(vec![0.2, -0.4, 1.1], vec![1, 3]);
    let logits_id = logits.id();
    let probabilities = softmax(logits, 1);
    let weights = Constant::new(vec![0.5, -1.0, 2.0], vec![1, 3]);
    let loss = (probabilities.clone() * weights)
        .reduce_sum(1)
        .reduce_sum(0);
    let graph: TensorGraph<f32> = loss.into();
    let loss_node = *graph.toposort().last().unwrap();
    let gradient_graph = graph.with_gradients(loss_node);
    let mut runtime = Runtime::new();
    runtime.execute(&gradient_graph, HashMap::new()).unwrap();
    let gradients = runtime.get_gradients(&gradient_graph);

    let probability_values = run(probabilities);
    let weighted_sum: f32 = probability_values
        .iter()
        .zip([0.5, -1.0, 2.0])
        .map(|(probability, weight)| probability * weight)
        .sum();
    let expected: Vec<f32> = probability_values
        .iter()
        .zip([0.5, -1.0, 2.0])
        .map(|(probability, weight)| probability * (weight - weighted_sum))
        .collect();
    assert_close(&gradients[&logits_id], &expected, 1e-4);
}

#[test]
fn transformer_block_executes_and_differentiates_all_parameters() {
    let block = TransformerBlock::new(4, 2, 8, 1e-5);
    let input = Constant::new(
        vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8],
        vec![1, 2, 4],
    );
    let output = block.forward(input, true);
    assert_eq!(output.shape(), &vec![1, 2, 4]);
    assert!(run(output.clone()).iter().all(|value| value.is_finite()));

    let graph: TensorGraph<f32> = output.mean_all().into();
    let loss_node = *graph.toposort().last().unwrap();
    let gradient_graph = graph.with_gradients(loss_node);
    let mut runtime = Runtime::new();
    runtime.execute(&gradient_graph, HashMap::new()).unwrap();
    let gradients = runtime.get_gradients(&gradient_graph);
    let parameter_ids = [
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
        let gradient = gradients
            .get(&parameter_id)
            .unwrap_or_else(|| panic!("missing gradient for parameter {parameter_id}"));
        assert!(gradient.iter().all(|value| value.is_finite()));
    }
}
