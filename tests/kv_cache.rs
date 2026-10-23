use std::collections::HashMap;
use tnsr::graph::TensorGraph;
use tnsr::nn::{Gpt, GptConfig};
use tnsr::tensor::TensorExpr;
use tnsr::{Backend, Executor, Runtime, SimpleExecutor};

fn model(capacity: usize) -> Gpt {
    Gpt::new(GptConfig {
        vocab_size: 367,
        max_sequence_length: capacity,
        embed_dim: 8,
        num_heads: 2,
        feed_forward_dim: 13,
        num_layers: 2,
        layer_norm_epsilon: 1e-5,
    })
}

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-4,
            "logit {i}: {a} vs {b}"
        );
    }
}

fn chunks_match(backend: Backend) {
    let model = model(7);
    for batch in [1, 2] {
        let mut decoder = model.decoder(backend, batch).unwrap();
        let rows: Vec<Vec<usize>> = (0..batch)
            .map(|b| (0..7).map(|i| (i * 53 + b * 109 + 365) % 367).collect())
            .collect();
        let mut reference = Runtime::<f32>::with_backend(backend).unwrap();
        for sizes in [[2, 1, 4], [1, 3, 3], [7, 0, 0]] {
            decoder.reset().unwrap();
            let mut position = 0;
            for chunk in sizes.into_iter().filter(|&c| c > 0) {
                let tokens: Vec<_> = rows
                    .iter()
                    .flat_map(|row| row[position..position + chunk].iter().copied())
                    .collect();
                let actual = decoder.decode(&tokens).unwrap();
                position += chunk;
                let prefix: Vec<_> = rows
                    .iter()
                    .flat_map(|row| row[..position].iter().map(|&t| t as f32))
                    .collect();
                let graph: TensorGraph<f32> = model
                    .forward(TensorExpr::constant(prefix, vec![batch, position]))
                    .into();
                let full = reference.execute(&graph, HashMap::new()).unwrap();
                let expected: Vec<_> = (0..batch)
                    .flat_map(|b| {
                        full[(b * position + position - chunk) * 367..(b + 1) * position * 367]
                            .iter()
                            .copied()
                    })
                    .collect();
                close(&actual, &expected);
                assert_eq!(decoder.len(), position);
            }
            assert_eq!(decoder.len(), 7);
            assert!(decoder.decode(&vec![0; batch]).is_err());
        }
    }
}

#[test]
fn cpu_kv_cache_matches_full_decode_for_batches_and_chunks() {
    chunks_match(Backend::Cpu);
}

#[cfg(feature = "cuda")]
fn cuda_available() -> bool {
    match tnsr::ptx::PtxExecutor::try_new() {
        Ok(_) => true,
        Err(error)
            if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                && error
                    .to_string()
                    .starts_with("Failed to initialize CUDA device 0") =>
        {
            false
        }
        Err(error) => panic!("{error:#}"),
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_kv_cache_matches_full_decode_for_batches_and_chunks() {
    if cuda_available() {
        chunks_match(Backend::Ptx);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_kv_cache_reset_refreshes_weights_and_reuses_programs() {
    if cuda_available() {
        validate_inputs_and_weights(Backend::Ptx);
    }
}

#[test]
fn kv_cache_validates_inputs_and_requires_reset_after_weight_changes() {
    validate_inputs_and_weights(Backend::Cpu);
}

fn validate_inputs_and_weights(backend: Backend) {
    let model = model(4);
    assert!(model.decoder(backend, 0).is_err());
    assert!(model.decoder_with_capacity(backend, 1, 0).is_err());
    assert!(model.decoder_with_capacity(backend, 1, 5).is_err());
    let mut decoder = model.decoder_with_capacity(backend, 2, 3).unwrap();
    assert_eq!((decoder.batch_size(), decoder.capacity()), (2, 3));
    assert!(model.decoder(backend, usize::MAX).is_err());
    assert!(decoder.is_empty());
    decoder.decode(&[365, 1]).unwrap();
    for invalid in [vec![], vec![0], vec![0, 367], vec![0; 6]] {
        assert!(decoder.decode(&invalid).is_err());
    }
    assert_eq!(decoder.len(), 1);
    decoder.decode(&[2, 3]).unwrap();
    let mut weights = model.token_embedding.values().unwrap();
    weights[0] += 0.3;
    model.token_embedding.set_values(weights).unwrap();
    assert!(
        decoder
            .decode(&[4, 5])
            .unwrap_err()
            .to_string()
            .contains("weights changed")
    );
    assert_eq!(decoder.len(), 2);
    decoder.reset().unwrap();
    let actual = decoder.decode(&[365, 1]).unwrap();
    let graph: TensorGraph<f32> = model
        .forward(TensorExpr::constant(vec![365., 1.], vec![2, 1]))
        .into();
    close(
        &actual,
        &SimpleExecutor::new()
            .execute(&graph, HashMap::new())
            .unwrap(),
    );
}

#[test]
fn cached_generation_matches_greedy_sliding_context() {
    let model = model(3);
    for prompt in [&[365][..], &[365, 2, 3][..], &[365, 2, 3, 4, 5][..]] {
        for new_tokens in [0, 1, 7] {
            let expected = model
                .generate(&mut SimpleExecutor::new(), prompt, new_tokens)
                .unwrap();
            let actual = model
                .generate_cached(Backend::Cpu, prompt, new_tokens)
                .unwrap();
            assert_eq!(actual, expected);
        }
    }
    assert!(model.generate_cached(Backend::Cpu, &[], 1).is_err());
    assert!(model.generate_cached(Backend::Cpu, &[367], 0).is_err());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_cached_generation_matches_sliding_context() {
    if !cuda_available() {
        return;
    }
    let model = model(3);
    let prompt = [365, 2, 3, 4];
    let expected = model
        .generate(&mut SimpleExecutor::new(), &prompt, 6)
        .unwrap();
    assert_eq!(
        model.generate_cached(Backend::Ptx, &prompt, 6).unwrap(),
        expected
    );
}

#[test]
fn cached_single_position_window_matches_uncached() {
    let model = model(1);
    assert_eq!(
        model.generate_cached(Backend::Cpu, &[365, 3], 4).unwrap(),
        model
            .generate(&mut SimpleExecutor::new(), &[365, 3], 4)
            .unwrap()
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_kv_cache_crosses_scan_tiles_with_one_compiled_chunk_shape() {
    if !cuda_available() {
        return;
    }
    let model = model(140);
    let batch = 2;
    let rows: Vec<Vec<usize>> = (0..batch)
        .map(|b| (0..140).map(|i| (i * 53 + b * 109 + 365) % 367).collect())
        .collect();
    let mut decoder = model.decoder(Backend::Ptx, batch).unwrap();
    // Run one shape through partial/full tiles and the context boundary; reset
    // exercises its runtime length changing back to zero using the same program.
    for _ in 0..2 {
        decoder.reset().unwrap();
        for position in 0..140 {
            let tokens: Vec<_> = rows.iter().map(|r| r[position]).collect();
            let actual = decoder.decode(&tokens).unwrap();
            if [
                0, 1, 30, 31, 32, 33, 62, 63, 64, 65, 126, 127, 128, 129, 139,
            ]
            .contains(&position)
            {
                let prefix: Vec<f32> = rows
                    .iter()
                    .flat_map(|r| r[..=position].iter().map(|&t| t as f32))
                    .collect();
                let graph: TensorGraph<f32> = model
                    .forward(TensorExpr::constant(prefix, vec![batch, position + 1]))
                    .into();
                let full = SimpleExecutor::new()
                    .execute(&graph, HashMap::new())
                    .unwrap();
                let expected: Vec<f32> = (0..batch)
                    .flat_map(|b| {
                        full[((b + 1) * (position + 1) - 1) * 367..(b + 1) * (position + 1) * 367]
                            .iter()
                            .copied()
                    })
                    .collect();
                close(&actual, &expected);
            }
        }
    }
}
