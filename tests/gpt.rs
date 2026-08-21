use std::collections::HashMap;

use tnsr::graph::TensorGraph;
use tnsr::nn::{Gpt, GptConfig};
use tnsr::optimizer::Adam;
use tnsr::tensor::TensorExpr;
use tnsr::{Executor, SimpleExecutor};

fn tiny_gpt() -> Gpt {
    Gpt::new(GptConfig {
        vocab_size: 4,
        max_sequence_length: 3,
        embed_dim: 4,
        num_heads: 2,
        feed_forward_dim: 8,
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
fn gpt_forward_loss_and_all_parameter_gradients_are_finite() {
    let gpt = tiny_gpt();
    let tokens = TensorExpr::constant(vec![0.0, 1.0, 1.0, 2.0], vec![2, 2]);
    let logits = gpt.forward(tokens.clone());
    assert_eq!(logits.shape(), &vec![2, 2, 4]);

    let targets = TensorExpr::constant(vec![1.0, 2.0, 2.0, 3.0], vec![2, 2]);
    let loss = gpt.loss(tokens, targets);
    let graph: TensorGraph<f32> = loss.into();
    let loss_node = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(loss_node);
    let mut executor = SimpleExecutor::new();
    executor.execute(&graph, HashMap::new()).unwrap();

    let loss = executor.get_value(loss_node).unwrap();
    assert_eq!(loss.len(), 1);
    assert!(loss[0].is_finite());
    let gradients = executor.get_gradients(&graph);
    for parameter_id in parameter_ids(&gpt) {
        let gradient = gradients
            .get(&parameter_id)
            .unwrap_or_else(|| panic!("missing gradient for parameter {parameter_id}"));
        assert!(gradient.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn tiny_gpt_overfits_one_sequence_and_generates_valid_tokens() {
    let gpt = tiny_gpt();
    let tokens = TensorExpr::constant(vec![0.0, 1.0, 2.0], vec![1, 3]);
    let targets = TensorExpr::constant(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let loss = gpt.loss(tokens, targets);
    let graph: TensorGraph<f32> = loss.into();
    let loss_node = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(loss_node);
    let mut executor = SimpleExecutor::new();
    let mut optimizer = Adam::new(0.03);

    executor.execute(&graph, HashMap::new()).unwrap();
    let initial_loss = executor.get_value(loss_node).unwrap()[0];
    for _ in 0..80 {
        executor.execute(&graph, HashMap::new()).unwrap();
        optimizer.step(&graph, &executor.get_gradients(&graph));
    }
    executor.execute(&graph, HashMap::new()).unwrap();
    let final_loss = executor.get_value(loss_node).unwrap()[0];
    assert!(
        final_loss < initial_loss * 0.5,
        "expected training loss to decrease from {initial_loss}, got {final_loss}"
    );

    let generated = gpt.generate(&mut executor, &[0], 4).unwrap();
    assert_eq!(generated.len(), 5);
    assert!(generated.iter().all(|&token| token < 4));
}
