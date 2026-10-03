use std::collections::BTreeSet;

use anyhow::{Context, Result};
use clap::Parser;
use tnsr::data::DataLoader;
use tnsr::graph::TensorGraph;
use tnsr::nn::{Gpt, GptConfig};
use tnsr::optimizer::Adam;
use tnsr::tensor::Input;
use tnsr::{Executor, Runtime};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 5)]
    epochs: usize,
    #[arg(long, default_value_t = 4)]
    batch_size: usize,
    #[arg(long, default_value_t = 8)]
    sequence_length: usize,
    #[arg(long, default_value_t = 0.01)]
    learning_rate: f32,
    #[arg(long, default_value_t = 40)]
    generate_tokens: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let text = "transformers predict the next token. transformers learn from repeated text. ";
    let corpus = text.repeat(8);
    let vocabulary: Vec<char> = corpus
        .chars()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let encoded: Vec<usize> = corpus
        .chars()
        .map(|character| vocabulary.binary_search(&character).unwrap())
        .collect();
    anyhow::ensure!(
        args.sequence_length > 0 && args.sequence_length < encoded.len(),
        "sequence length must be between 1 and {}",
        encoded.len() - 1
    );

    let sample_count = encoded.len() - args.sequence_length;
    let mut token_data = Vec::with_capacity(sample_count * args.sequence_length);
    let mut target_data = Vec::with_capacity(sample_count * args.sequence_length);
    for start in 0..sample_count {
        token_data.extend(
            encoded[start..start + args.sequence_length]
                .iter()
                .map(|&token| token as f32),
        );
        target_data.extend(
            encoded[start + 1..start + args.sequence_length + 1]
                .iter()
                .map(|&token| token as f32),
        );
    }

    let gpt = Gpt::new(GptConfig {
        vocab_size: vocabulary.len(),
        max_sequence_length: args.sequence_length,
        embed_dim: 16,
        num_heads: 4,
        feed_forward_dim: 64,
        num_layers: 2,
        layer_norm_epsilon: 1e-5,
    });
    let tokens = Input::new("tokens", vec![args.batch_size, args.sequence_length]);
    let targets = Input::new("targets", vec![args.batch_size, args.sequence_length]);
    let loss = gpt.loss(tokens, targets);
    let graph: TensorGraph<f32> = loss.into();
    let loss_node = *graph.toposort().last().context("GPT graph is empty")?;
    let graph = graph.with_gradients(loss_node);
    let mut runtime = Runtime::<f32>::new();
    let mut optimizer = Adam::new(args.learning_rate);
    let mut loader = DataLoader::new(sample_count)
        .with("tokens", &token_data, args.sequence_length)
        .with("targets", &target_data, args.sequence_length)
        .batch_size(args.batch_size)
        .shuffle(42)
        .drop_last();

    println!(
        "training {:?} GPT: vocab={}, samples={sample_count}",
        runtime.backend(),
        vocabulary.len()
    );
    for epoch in 0..args.epochs {
        let mut loss_sum = 0.0;
        let mut batches = 0usize;
        for batch in loader.iter()? {
            runtime.execute(&graph, batch.into_inputs())?;
            loss_sum += runtime
                .get_value(loss_node)
                .context("loss value was not retained")?[0];
            optimizer.step(&graph, &runtime.get_gradients(&graph));
            batches += 1;
        }
        println!(
            "epoch {:>2}: loss {:.4}",
            epoch + 1,
            loss_sum / batches as f32
        );
    }

    let prompt = [encoded[0]];
    let generated = gpt.generate(&mut runtime, &prompt, args.generate_tokens)?;
    let generated: String = generated
        .into_iter()
        .map(|token| vocabulary[token])
        .collect();
    println!("\n{generated}");
    Ok(())
}
