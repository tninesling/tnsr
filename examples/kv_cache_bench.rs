//! Matched one-token decoding; cold setup and prompt prefill are separate.
use anyhow::{Result, ensure};
use clap::Parser;
use serde_json::json;
use std::{collections::HashMap, time::Instant};
use tnsr::{
    Backend, Executor, Runtime,
    graph::TensorGraph,
    nn::{Gpt, GptConfig},
    tensor::TensorExpr,
};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "cpu")]
    backend: String,
    #[arg(long, default_value_t = 32)]
    repeats: usize,
    #[arg(long, default_value_t = 256)]
    capacity: usize,
    #[arg(long, value_delimiter = ',', default_value = "16,64,128")]
    prompts: Vec<usize>,
}

fn median(x: &[f64]) -> f64 {
    let mut sorted = x.to_vec();
    sorted.sort_by(f64::total_cmp);
    (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.repeats > 0 && !args.prompts.is_empty(),
        "nonempty prompts and positive repeats required"
    );
    ensure!(
        args.prompts.iter().all(|&p| p > 0 && p < args.capacity),
        "prompts must fit one new token within capacity"
    );
    let backend = match args.backend.as_str() {
        "cpu" => Backend::Cpu,
        #[cfg(feature = "cuda")]
        "cuda" => Backend::Ptx,
        _ => anyhow::bail!("backend must be cpu or CUDA-enabled cuda"),
    };
    let model = Gpt::new(GptConfig {
        vocab_size: 256,
        max_sequence_length: args.capacity,
        embed_dim: 64,
        num_heads: 4,
        feed_forward_dim: 256,
        num_layers: 2,
        layer_norm_epsilon: 1e-5,
    });
    let mut cases = Vec::new();
    for &prompt_len in &args.prompts {
        let tokens: Vec<usize> = (0..=prompt_len).map(|i| (i * 17 + 3) % 256).collect();
        let feeds: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
        let graph: TensorGraph<f32> = model
            .forward(TensorExpr::input("tokens", vec![1, tokens.len()]))
            .into();
        let mut reference = match backend {
            Backend::Cpu => Runtime::<f32>::with_backend(Backend::Cpu)?,
            #[cfg(feature = "cuda")]
            Backend::Cuda | Backend::Ptx => {
                let mut executor = tnsr::ptx::PtxExecutor::try_new_with_reduction_mode(
                    tnsr::ptx::PtxReductionMode::DeterministicTree,
                )?;
                executor.set_matmul_precision(tnsr::tile::MatMulPrecision::StrictF32);
                Runtime::Ptx(executor)
            }
        };
        let started = Instant::now();
        let full = reference.execute(&graph, HashMap::from([("tokens".into(), feeds.clone())]))?;
        let reference_cold = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let mut decoder = model.decoder(backend, 1)?;
        let construct = started.elapsed().as_secs_f64();
        let started = Instant::now();
        decoder.decode(&tokens[..prompt_len])?;
        let prefill = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let actual = decoder.decode(&tokens[prompt_len..])?;
        let first_decode = started.elapsed().as_secs_f64();
        let max_error = actual
            .iter()
            .zip(&full[full.len() - 256..])
            .map(|(&a, &b)| (a - b).abs())
            .fold(0., f32::max);
        ensure!(
            actual.iter().all(|x| x.is_finite()) && max_error < 2e-4,
            "cached logits mismatch: {max_error}"
        );
        let mut uncached = Vec::new();
        let mut cached = Vec::new();
        for iteration in 0..args.repeats + 3 {
            decoder.reset()?;
            decoder.decode(&tokens[..prompt_len])?;
            let mut measured = [0.; 2];
            for mode in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                let started = Instant::now();
                if mode == 0 {
                    std::hint::black_box(
                        reference
                            .execute(&graph, HashMap::from([("tokens".into(), feeds.clone())]))?,
                    );
                } else {
                    std::hint::black_box(decoder.decode(&tokens[prompt_len..])?);
                }
                measured[mode] = started.elapsed().as_secs_f64();
            }
            if iteration >= 3 {
                uncached.push(measured[0]);
                cached.push(measured[1]);
            }
        }
        let ratios: Vec<_> = uncached.iter().zip(&cached).map(|(a, b)| a / b).collect();
        let case = json!({"prompt_tokens":prompt_len,"max_logit_abs_error":max_error,"reference_cold_seconds":reference_cold,"decoder_construct_seconds":construct,"prefill_first_seconds":prefill,"decode_first_seconds":first_decode,"uncached_median_seconds":median(&uncached),"cached_median_seconds":median(&cached),"paired_median_speedup":median(&ratios),"uncached_samples_seconds":uncached,"cached_samples_seconds":cached});
        eprintln!(
            "prompt={prompt_len} uncached={:.3}ms cached={:.3}ms speedup={:.2}x",
            median(&uncached) * 1000.,
            median(&cached) * 1000.,
            median(&ratios)
        );
        cases.push(case);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"backend":args.backend,"capacity":args.capacity,"gpu_arithmetic":{"reference":"strict f32 matmuls and deterministic f32 tree reductions","cached":"strict f32 model projections; f32 online-normalized attention with runtime live-prefix bound"},"batch":1,"width":64,"layers":2,"heads":4,"feed_forward":256,"vocabulary":256,"warmups":3,"repeats":args.repeats,"protocol":"alternating AB/BA synchronized one-token decode, fixed prompt and weights; reset/prefill outside warm interval; all logits downloaded; cold setup separate","cases":cases})
        )?
    );
    Ok(())
}
