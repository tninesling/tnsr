use super::*;

#[test]
fn scan_runtime_bounds_ignore_poisoned_slots_and_match_wide_head_reference() -> Result<()> {
    match crate::ptx::PtxExecutor::<F32>::try_new() {
        Ok(_) => (),
        Err(error)
            if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                && error
                    .to_string()
                    .starts_with("Failed to initialize CUDA device 0") =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    }
    let owner = CudaDecoder::new(&[], &[])?;
    for columns in [4, 33, 128, 129] {
        for chunk in [1, 3] {
            let scan = AttentionScan {
                batch: 2,
                heads: 2,
                width: 2 * columns,
                capacity: 70,
                chunk,
                partitions: 1,
            };
            let count = scan.batch * chunk * scan.width;
            let samples = |n: usize, salt: usize| -> Vec<f32> {
                (0..n)
                    .map(|i| ((i * 17 + salt) % 101) as f32 / 23. - 2.)
                    .collect()
            };
            let q = samples(count, 5);
            let k = samples(count, 19);
            let v = samples(count, 37);
            let cache_k = samples(scan.batch * scan.capacity * scan.width, 41);
            let cache_v = samples(cache_k.len(), 67);
            let query = owner.stream.memcpy_stod(&q)?;
            let key = owner.stream.memcpy_stod(&k)?;
            let value = owner.stream.memcpy_stod(&v)?;
            let mut cached_key = owner.stream.alloc_zeros::<f32>(cache_k.len())?;
            let mut cached_value = owner.stream.alloc_zeros::<f32>(cache_v.len())?;
            let mut length = owner.stream.alloc_zeros::<u32>(2 + chunk * PARTITIONS)?;
            let output = owner.stream.alloc_zeros::<f32>(count)?;
            let config = if columns <= 128 {
                cudarc::driver::LaunchConfig {
                    grid_dim: (u32::try_from(scan.batch * scan.heads * chunk)?, 1, 1),
                    block_dim: (u32::try_from(columns.max(32).next_power_of_two())?, 1, 1),
                    shared_mem_bytes: 0,
                }
            } else {
                crate::ptx::target::pointwise_launch_config(u32::try_from(count)?)
            };
            let pointers = vec![
                query.device_ptr(&owner.stream).0,
                key.device_ptr(&owner.stream).0,
                value.device_ptr(&owner.stream).0,
                cached_key.device_ptr(&owner.stream).0,
                cached_value.device_ptr(&owner.stream).0,
                length.device_ptr(&owner.stream).0,
                output.device_ptr(&owner.stream).0,
            ];
            let kernel = owner.generated(scan.ir()?, pointers.clone(), config)?;
            let partials = owner.stream.alloc_zeros::<f32>(count * PARTITIONS)?;
            let normalizers = owner
                .stream
                .alloc_zeros::<f32>(scan.batch * scan.heads * chunk * PARTITIONS)?;
            let mut partial_pointers = pointers;
            partial_pointers[6] = partials.device_ptr(&owner.stream).0;
            partial_pointers.push(normalizers.device_ptr(&owner.stream).0);
            let partial = owner.generated(
                AttentionScan {
                    partitions: PARTITIONS,
                    ..scan
                }
                .ir()?,
                partial_pointers,
                scan_launch(scan.batch * scan.heads * chunk * PARTITIONS, columns)?,
            )?;
            let merge = owner.generated(
                MergeScan {
                    rows: scan.batch * scan.heads * chunk,
                    columns,
                    partitions: PARTITIONS,
                }
                .ir()?,
                vec![
                    partials.device_ptr(&owner.stream).0,
                    normalizers.device_ptr(&owner.stream).0,
                    output.device_ptr(&owner.stream).0,
                ],
                scan_launch(scan.batch * scan.heads * chunk, columns)?,
            )?;
            for prefix in [0, 1, 30, 31, 32, 33, 63, 64, 65] {
                let mut poisoned_k = vec![f32::NAN; cache_k.len()];
                let mut poisoned_v = poisoned_k.clone();
                for batch in 0..scan.batch {
                    let start = batch * scan.capacity * scan.width;
                    let end = start + prefix * scan.width;
                    poisoned_k[start..end].copy_from_slice(&cache_k[start..end]);
                    poisoned_v[start..end].copy_from_slice(&cache_v[start..end]);
                }
                owner.stream.memcpy_htod(&poisoned_k, &mut cached_key)?;
                owner.stream.memcpy_htod(&poisoned_v, &mut cached_value)?;
                owner
                    .stream
                    .memcpy_htod(&scan_lengths(prefix, chunk)?, &mut length)?;
                for scheduled in [&[&kernel][..], &[&partial, &merge][..]] {
                    for kernel in scheduled {
                        kernel.run(&owner.stream)?;
                    }
                    let actual = owner.stream.memcpy_dtov(&output)?;
                    owner.stream.synchronize()?;
                    for batch in 0..scan.batch {
                        for head in 0..scan.heads {
                            for query in 0..chunk {
                                let mut scores = Vec::new();
                                let select =
                                    |old: &[f32], new: &[f32], pos: usize, column: usize| -> f64 {
                                        let feature = head * columns + column;
                                        f64::from(if pos < prefix {
                                            old[(batch * scan.capacity + pos) * scan.width
                                                + feature]
                                        } else {
                                            new[(batch * chunk + pos - prefix) * scan.width
                                                + feature]
                                        })
                                    };
                                for pos in 0..prefix + query + 1 {
                                    let score: f64 = (0..columns)
                                        .map(|c| {
                                            f64::from(
                                                q[(batch * chunk + query) * scan.width
                                                    + head * columns
                                                    + c],
                                            ) * select(&cache_k, &k, pos, c)
                                        })
                                        .sum();
                                    scores.push(score / (columns as f64).sqrt());
                                }
                                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                                let weights: Vec<_> =
                                    scores.iter().map(|&s| (s - max).exp()).collect();
                                let denominator: f64 = weights.iter().sum();
                                for c in 0..columns {
                                    let expected: f64 = weights
                                        .iter()
                                        .enumerate()
                                        .map(|(pos, &w)| w * select(&cache_v, &v, pos, c))
                                        .sum::<f64>()
                                        / denominator;
                                    let offset =
                                        ((batch * scan.heads + head) * chunk + query) * columns + c;
                                    let value = f64::from(actual[offset]);
                                    assert!(
                                        value.is_finite() && (value - expected).abs() < 3e-5,
                                        "columns={columns} chunk={chunk} prefix={prefix} output={offset}: {value} vs {expected}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    nonfinite_partials(&owner)
}

#[test]
fn compiled_decode_reuses_length_and_bounds_workspace_independently_of_capacity() -> Result<()> {
    match crate::ptx::PtxExecutor::<F32>::try_new() {
        Ok(_) => (),
        Err(error)
            if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                && error
                    .to_string()
                    .starts_with("Failed to initialize CUDA device 0") =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    }
    let mut workspaces = Vec::new();
    for capacity in [70, 700] {
        let model = crate::nn::Gpt::new(crate::nn::GptConfig {
            vocab_size: 16,
            max_sequence_length: capacity,
            embed_dim: 8,
            num_heads: 2,
            feed_forward_dim: 16,
            num_layers: 2,
            layer_norm_epsilon: 1e-5,
        });
        let mut decoder = model.decoder(crate::Backend::Ptx, 2)?;
        decoder.decode(&[1, 2])?;
        decoder.decode(&[3, 4])?;
        decoder.reset()?;
        decoder.decode(&[5, 6])?;
        let super::super::DecoderBackend::Cuda(backend) = &decoder.backend else {
            unreachable!()
        };
        assert_eq!(backend.compiled.len(), 1);
        let compiled = &backend.compiled[&1];
        assert_eq!(
            compiled
                .kernels
                .iter()
                .filter(|k| k.label == "decoder generated scan")
                .count(),
            2
        );
        assert!(!compiled.feeds.contains_key("mask"));
        let bytes = compiled._programs.iter().map(Program::bytes).sum::<usize>()
            + compiled._storage.iter().map(|b| b.len() * 4).sum::<usize>()
            + compiled
                ._constants
                .iter()
                .map(|b| b.len() * 4)
                .sum::<usize>()
            + compiled.output.len() * 4;
        workspaces.push(bytes);
    }
    assert_eq!(
        workspaces[0], workspaces[1],
        "attention workspace must not scale with context capacity"
    );
    Ok(())
}

fn nonfinite_partials(owner: &CudaDecoder) -> Result<()> {
    let scan = AttentionScan {
        batch: 1,
        heads: 1,
        width: 4,
        chunk: 1,
        capacity: 2,
        partitions: 1,
    };
    let query = owner.stream.memcpy_stod(&[1e30f32, 0., 0., 0.])?;
    let key = owner.stream.memcpy_stod(&[1e-30f32, 0., 0., 0.])?;
    let value = owner.stream.memcpy_stod(&[0.25f32, 0.5, 0.75, 1.])?;
    let cached_key = owner.stream.memcpy_stod(&[
        -1e30f32,
        0.,
        0.,
        0.,
        f32::NAN,
        f32::NAN,
        f32::NAN,
        f32::NAN,
    ])?;
    let mut cached_value = owner.stream.alloc_zeros::<f32>(8)?;
    let length = owner.stream.memcpy_stod(&scan_lengths(1, 1)?)?;
    let output = owner.stream.alloc_zeros::<f32>(4)?;
    let partials = owner.stream.alloc_zeros::<f32>(4 * PARTITIONS)?;
    let normalizers = owner.stream.alloc_zeros::<f32>(PARTITIONS)?;
    let pointers = vec![
        query.device_ptr(&owner.stream).0,
        key.device_ptr(&owner.stream).0,
        value.device_ptr(&owner.stream).0,
        cached_key.device_ptr(&owner.stream).0,
        cached_value.device_ptr(&owner.stream).0,
        length.device_ptr(&owner.stream).0,
        output.device_ptr(&owner.stream).0,
    ];
    let single = owner.generated(scan.ir()?, pointers.clone(), scan_launch(1, 4)?)?;
    let mut partial_pointers = pointers;
    partial_pointers[6] = partials.device_ptr(&owner.stream).0;
    partial_pointers.push(normalizers.device_ptr(&owner.stream).0);
    let partial = owner.generated(
        AttentionScan {
            partitions: PARTITIONS,
            ..scan
        }
        .ir()?,
        partial_pointers,
        scan_launch(PARTITIONS, 4)?,
    )?;
    let merge = owner.generated(
        MergeScan {
            rows: 1,
            columns: 4,
            partitions: PARTITIONS,
        }
        .ir()?,
        vec![
            partials.device_ptr(&owner.stream).0,
            normalizers.device_ptr(&owner.stream).0,
            output.device_ptr(&owner.stream).0,
        ],
        scan_launch(1, 4)?,
    )?;
    for old in [2., f32::INFINITY, f32::NAN] {
        owner.stream.memcpy_htod(
            &[old, 2., 2., 2., f32::NAN, f32::NAN, f32::NAN, f32::NAN],
            &mut cached_value,
        )?;
        for scheduled in [&[&single][..], &[&partial, &merge][..]] {
            for kernel in scheduled {
                kernel.run(&owner.stream)?;
            }
            let actual = owner.stream.memcpy_dtov(&output)?;
            owner.stream.synchronize()?;
            if old.is_finite() {
                assert!((actual[0] - 0.25).abs() < 1e-6);
            } else {
                assert!(
                    actual[0].is_nan(),
                    "zero times nonfinite must propagate through partial merges"
                );
            }
            for (actual, expected) in actual[1..].iter().zip([0.5, 0.75, 1.]) {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
    }
    Ok(())
}
