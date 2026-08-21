#![cfg(feature = "cuda")]

use std::collections::HashMap;

use tnsr::graph::TensorGraph;
use tnsr::nn::{Gpt, GptConfig};
use tnsr::optimizer::Adam;
use tnsr::ptx::PtxExecutor;
use tnsr::tensor::TensorExpr;
use tnsr::{Backend, Executor, Runtime, SimpleExecutor};

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

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "element {index}: expected {expected}, got {actual} (tolerance {tolerance})"
        );
    }
}

fn tiny_gpt() -> Gpt {
    Gpt::new(GptConfig {
        vocab_size: 3,
        max_sequence_length: 2,
        embed_dim: 2,
        num_heads: 1,
        feed_forward_dim: 2,
        num_layers: 1,
        layer_norm_epsilon: 1e-5,
    })
}

fn parameter_ids(gpt: &Gpt) -> Vec<usize> {
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
fn ptx_embedding_forward_gathers_rows() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let expression = TensorExpr::constant(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![4, 2])
        .embedding(TensorExpr::constant(vec![2.0, 0.0, 3.0, 1.0], vec![2, 2]));
    let graph: TensorGraph<f32> = expression.into();
    let actual = ptx.execute(&graph, HashMap::new()).unwrap();
    assert_close(&actual, &[5.0, 6.0, 1.0, 2.0, 7.0, 8.0, 3.0, 4.0], 0.0);
}

#[test]
fn ptx_embedding_backward_scatter_adds_repeated_ids() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let indices = TensorExpr::constant(vec![1.0, 1.0, 2.0], vec![3]);
    let grad_output = TensorExpr::constant(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let expression = TensorExpr::embedding_backward(indices, grad_output, vec![3, 2]);
    let graph: TensorGraph<f32> = expression.into();
    let actual = ptx.execute(&graph, HashMap::new()).unwrap();
    assert_close(&actual, &[0.0, 0.0, 4.0, 6.0, 5.0, 6.0], 0.0);
}

#[test]
fn ptx_indexed_cross_entropy_forward_and_backward_are_stable() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let logits_data = vec![1000.0, 1001.0, 999.0, -1000.0, -999.0, -1001.0];
    let targets = TensorExpr::constant(vec![1.0, 0.0], vec![2]);
    let forward = TensorExpr::constant(logits_data.clone(), vec![2, 3])
        .indexed_cross_entropy(targets.clone());
    let forward_graph: TensorGraph<f32> = forward.into();
    let expected = SimpleExecutor::new()
        .execute(&forward_graph, HashMap::new())
        .unwrap();
    let actual = ptx.execute(&forward_graph, HashMap::new()).unwrap();
    assert_close(&actual, &expected, 2e-4);

    let backward = TensorExpr::indexed_cross_entropy_backward(
        TensorExpr::constant(logits_data, vec![2, 3]),
        targets,
        TensorExpr::constant(vec![0.5, 0.25], vec![2]),
    );
    let backward_graph: TensorGraph<f32> = backward.into();
    let expected = SimpleExecutor::new()
        .execute(&backward_graph, HashMap::new())
        .unwrap();
    let actual = ptx.execute(&backward_graph, HashMap::new()).unwrap();
    assert_close(&actual, &expected, 2e-4);
}

#[test]
fn ptx_invalid_index_error_is_recoverable() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let weight = || TensorExpr::constant(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let invalid: TensorGraph<f32> = weight()
        .embedding(TensorExpr::constant(vec![2.0], vec![1]))
        .into();
    let error = ptx.execute(&invalid, HashMap::new()).unwrap_err();
    assert!(error.to_string().contains("out of range"), "{error:#}");

    let valid: TensorGraph<f32> = weight()
        .embedding(TensorExpr::constant(vec![1.0], vec![1]))
        .into();
    let actual = ptx.execute(&valid, HashMap::new()).unwrap();
    assert_close(&actual, &[3.0, 4.0], 0.0);
}

#[test]
fn ptx_tiny_gpt_logits_loss_and_all_gradients_match_cpu() {
    let Some(mut ptx) = ptx_executor() else {
        return;
    };
    let gpt = tiny_gpt();
    let tokens = TensorExpr::constant(vec![0.0, 1.0], vec![1, 2]);
    let logits_graph: TensorGraph<f32> = gpt.forward(tokens.clone()).into();
    let expected_logits = SimpleExecutor::new()
        .execute(&logits_graph, HashMap::new())
        .unwrap();
    let actual_logits = ptx.execute(&logits_graph, HashMap::new()).unwrap();
    assert_close(&actual_logits, &expected_logits, 3e-4);

    let targets = TensorExpr::constant(vec![1.0, 2.0], vec![1, 2]);
    let loss_graph: TensorGraph<f32> = gpt.loss(tokens, targets).into();
    let loss_node = *loss_graph.toposort().last().unwrap();
    let gradient_graph = loss_graph.with_gradients(loss_node);
    let mut cpu = SimpleExecutor::new();
    let expected_loss = cpu.execute(&gradient_graph, HashMap::new()).unwrap();
    let expected_gradients = cpu.get_gradients(&gradient_graph);
    let actual_loss = ptx.execute(&gradient_graph, HashMap::new()).unwrap();
    let actual_gradients = ptx.get_gradients(&gradient_graph);
    assert_close(&actual_loss, &expected_loss, 3e-4);
    for parameter_id in parameter_ids(&gpt) {
        let expected = expected_gradients
            .get(&parameter_id)
            .unwrap_or_else(|| panic!("CPU gradient missing for parameter {parameter_id}"));
        let actual = actual_gradients
            .get(&parameter_id)
            .unwrap_or_else(|| panic!("PTX gradient missing for parameter {parameter_id}"));
        assert_close(actual, expected, 3e-3);
    }
}

#[test]
fn runtime_ptx_gpt_optimizer_steps_decrease_loss() {
    if ptx_executor().is_none() {
        return;
    }
    let gpt = tiny_gpt();
    let tokens = TensorExpr::constant(vec![0.0, 1.0], vec![1, 2]);
    let targets = TensorExpr::constant(vec![1.0, 2.0], vec![1, 2]);
    let graph: TensorGraph<f32> = gpt.loss(tokens, targets).into();
    let loss_node = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(loss_node);
    let mut runtime = Runtime::<f32>::with_backend(Backend::Ptx).unwrap();
    let mut optimizer = Adam::new(0.02);

    runtime.execute(&graph, HashMap::new()).unwrap();
    let initial_loss = runtime.get_value(loss_node).unwrap()[0];
    for _ in 0..3 {
        runtime.execute(&graph, HashMap::new()).unwrap();
        optimizer.step(&graph, &runtime.get_gradients(&graph));
    }
    runtime.execute(&graph, HashMap::new()).unwrap();
    let final_loss = runtime.get_value(loss_node).unwrap()[0];
    assert!(
        final_loss < initial_loss,
        "expected PTX training loss to decrease from {initial_loss}, got {final_loss}"
    );
}
