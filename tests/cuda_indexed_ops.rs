#![cfg(feature = "cuda")]

use std::collections::HashMap;

use tnsr::cuda::CudaExecutor;
use tnsr::graph::TensorGraph;
use tnsr::nn::{Gpt, GptConfig};
use tnsr::tensor::{ExprKind, TensorExpr};
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

fn parameter_id(expression: &TensorExpr<f32>) -> usize {
    match expression.kind() {
        ExprKind::Parameter { id, .. } => *id,
        _ => panic!("expected parameter expression"),
    }
}

fn gpt_parameter_ids(gpt: &Gpt) -> Vec<usize> {
    let mut ids = vec![
        gpt.token_embedding.id(),
        gpt.position_embedding.id(),
        gpt.final_norm_weight.id(),
        gpt.final_norm_bias.id(),
    ];
    for block in &gpt.blocks {
        ids.extend([
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
        ]);
    }
    ids
}

#[test]
fn cuda_repeated_id_embedding_forward_and_backward_match_cpu() {
    let Some(mut cuda) = cuda_executor() else {
        return;
    };
    let indices = TensorExpr::constant(vec![1.0, 1.0, 2.0, 0.0], vec![2, 2]);
    let weight_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let forward: TensorGraph<f32> = TensorExpr::constant(weight_data.clone(), vec![3, 2])
        .embedding(indices.clone())
        .into();
    let expected = SimpleExecutor::new()
        .execute(&forward, HashMap::new())
        .expect("CPU embedding forward failed");
    let actual = cuda
        .execute(&forward, HashMap::new())
        .expect("CUDA embedding forward failed");
    assert_close(&actual, &expected, 0.0);

    let weight = TensorExpr::parameter(weight_data, vec![3, 2]);
    let weight_id = parameter_id(&weight);
    let loss = weight.embedding(indices).mean_all();
    let graph: TensorGraph<f32> = loss.into();
    let loss_node = *graph.toposort().last().expect("embedding graph is empty");
    let graph = graph.with_gradients(loss_node);
    let mut cpu = SimpleExecutor::new();
    cpu.execute(&graph, HashMap::new())
        .expect("CPU embedding backward failed");
    cuda.execute(&graph, HashMap::new())
        .expect("CUDA embedding backward failed");
    assert_close(
        &cuda.get_gradients(&graph)[&weight_id],
        &cpu.get_gradients(&graph)[&weight_id],
        1e-6,
    );
}

#[test]
fn cuda_stable_indexed_cross_entropy_forward_and_backward_match_cpu() {
    let Some(mut cuda) = cuda_executor() else {
        return;
    };
    let logits_data = vec![1000.0, 1001.0, 999.0, -1000.0, -999.0, -1001.0];
    let targets = TensorExpr::constant(vec![1.0, 0.0], vec![2]);
    let forward: TensorGraph<f32> = TensorExpr::constant(logits_data.clone(), vec![2, 3])
        .indexed_cross_entropy(targets.clone())
        .into();
    let expected = SimpleExecutor::new()
        .execute(&forward, HashMap::new())
        .expect("CPU indexed cross entropy forward failed");
    let actual = cuda
        .execute(&forward, HashMap::new())
        .expect("CUDA indexed cross entropy forward failed");
    assert_close(&actual, &expected, 2e-5);

    let logits = TensorExpr::parameter(logits_data, vec![2, 3]);
    let logits_id = parameter_id(&logits);
    let loss = logits.indexed_cross_entropy(targets).mean_all();
    let graph: TensorGraph<f32> = loss.into();
    let loss_node = *graph
        .toposort()
        .last()
        .expect("indexed cross entropy graph is empty");
    let graph = graph.with_gradients(loss_node);
    let mut cpu = SimpleExecutor::new();
    cpu.execute(&graph, HashMap::new())
        .expect("CPU indexed cross entropy backward failed");
    cuda.execute(&graph, HashMap::new())
        .expect("CUDA indexed cross entropy backward failed");
    assert_close(
        &cuda.get_gradients(&graph)[&logits_id],
        &cpu.get_gradients(&graph)[&logits_id],
        1e-6,
    );
}

#[test]
fn cuda_invalid_indices_return_errors_without_poisoning_executor() {
    let Some(mut cuda) = cuda_executor() else {
        return;
    };
    let invalid: TensorGraph<f32> = TensorExpr::constant(vec![1.0; 6], vec![3, 2])
        .embedding(TensorExpr::constant(vec![1.5], vec![1]))
        .into();
    let error = cuda.execute(&invalid, HashMap::new()).unwrap_err();
    assert!(format!("{error:#}").contains("must be an integer"));

    let valid: TensorGraph<f32> = TensorExpr::constant(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
        .embedding(TensorExpr::constant(vec![1.0], vec![1]))
        .into();
    assert_close(
        &cuda.execute(&valid, HashMap::new()).unwrap(),
        &[3.0, 4.0],
        0.0,
    );
}

#[test]
fn cuda_tiny_gpt_loss_and_gradients_match_cpu() {
    let Some(mut cuda) = cuda_executor() else {
        return;
    };
    let gpt = Gpt::new(GptConfig {
        vocab_size: 4,
        max_sequence_length: 2,
        embed_dim: 4,
        num_heads: 2,
        feed_forward_dim: 8,
        num_layers: 1,
        layer_norm_epsilon: 1e-5,
    });
    let tokens = TensorExpr::constant(vec![0.0, 1.0], vec![1, 2]);
    let targets = TensorExpr::constant(vec![1.0, 2.0], vec![1, 2]);
    let graph: TensorGraph<f32> = gpt.loss(tokens, targets).into();
    let loss_node = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(loss_node);
    let mut cpu = SimpleExecutor::new();

    cpu.execute(&graph, HashMap::new()).unwrap();
    cuda.execute(&graph, HashMap::new()).unwrap();
    assert_close(
        &cuda.get_value(loss_node).unwrap(),
        &cpu.get_value(loss_node).unwrap(),
        1e-4,
    );

    let expected_gradients = cpu.get_gradients(&graph);
    let actual_gradients = cuda.get_gradients(&graph);
    for parameter_id in gpt_parameter_ids(&gpt) {
        assert_close(
            &actual_gradients[&parameter_id],
            &expected_gradients[&parameter_id],
            3e-3,
        );
    }
}
