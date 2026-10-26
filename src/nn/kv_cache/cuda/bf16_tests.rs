use super::*;
use half::bf16;
use serde_json::json;

fn owner() -> Result<Option<CudaDecoder>> {
    match crate::ptx::PtxExecutor::<F32>::try_new() {
        Ok(_) => Ok(Some(CudaDecoder::new(
            &[],
            &[],
            GptDecoderOptions {
                attention_precision: GptAttentionPrecision::Bf16,
                ..Default::default()
            },
        )?)),
        Err(e)
            if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                && e.to_string()
                    .starts_with("Failed to initialize CUDA device 0") =>
        {
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

fn captured_samples(owner: &CudaDecoder, kernels: &[&Kernel]) -> Result<Vec<f64>> {
    let mut samples = Vec::new();
    owner.stream.begin_capture(
        cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
    )?;
    let recorded = (|| {
        for _ in 0..20 {
            for kernel in kernels {
                kernel.run(&owner.stream)?;
            }
        }
        Ok::<_, anyhow::Error>(())
    })();
    // End capture on both success and failure; the guard destroys the
    // graph before any referenced kernel/module or device allocation.
    let deferred = owner
        .stream
        .context()
        .check_err()
        .map_err(anyhow::Error::from);
    let ended = owner.stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
    recorded.and(deferred)?;
    let captured = crate::ptx::linear::Captured {
        graph: Some(ended?.context("empty attention timing capture")?),
        stream: owner.stream.clone(),
    };
    let graph = captured
        .graph
        .as_ref()
        .context("missing attention timing graph")?;
    for _ in 0..3 {
        graph.launch()?;
    }
    for _ in 0..16 {
        let start = owner
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        graph.launch()?;
        let end = owner
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        samples.push(f64::from(start.elapsed_ms(&end)?) * 1000. / 20.);
    }
    owner.stream.synchronize()?;
    Ok(samples)
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    owner: &CudaDecoder,
    columns: usize,
    chunk: usize,
    prefix: usize,
    batch: usize,
    heads: usize,
    timed: bool,
    gain: f32,
) -> Result<serde_json::Value> {
    let width = columns * heads;
    let capacity = prefix + chunk + 7;
    let scan = AttentionScan {
        batch,
        heads,
        width,
        chunk,
        capacity,
        partitions: 1,
        dtype: DType::BF16,
    };
    let samples = |n: usize, salt: usize| -> Vec<bf16> {
        (0..n)
            .map(|i| bf16::from_f32(((i * 17 + salt) % 101) as f32 / 53. - 1.))
            .collect()
    };
    let count = batch * chunk * width;
    let scaled = |x: Vec<bf16>| {
        x.into_iter()
            .map(|x| bf16::from_f32(x.to_f32() * gain))
            .collect::<Vec<_>>()
    };
    let q = scaled(samples(count, 5));
    let k = scaled(samples(count, 19));
    let v = samples(count, 37);
    let old_k = scaled(samples(batch * capacity * width, 41));
    let old_v = samples(old_k.len(), 67);
    let mut poisoned_k = vec![bf16::NAN; old_k.len()];
    let mut poisoned_v = poisoned_k.clone();
    for b in 0..batch {
        let range = b * capacity * width..(b * capacity + prefix) * width;
        poisoned_k[range.clone()].copy_from_slice(&old_k[range.clone()]);
        poisoned_v[range.clone()].copy_from_slice(&old_v[range]);
    }
    let buffers = [&q, &k, &v, &poisoned_k, &poisoned_v]
        .into_iter()
        .map(|x| owner.stream.memcpy_stod(x))
        .collect::<Result<Vec<_>, _>>()?;
    let length = owner
        .stream
        .memcpy_stod(&scan_lengths(prefix, chunk, PARTITIONS)?)?;
    let output = owner.stream.alloc_zeros::<bf16>(count)?;
    let mut pointers: Vec<_> = buffers
        .iter()
        .map(|b| b.device_ptr(&owner.stream).0)
        .collect();
    pointers.push(length.device_ptr(&owner.stream).0);
    pointers.push(output.device_ptr(&owner.stream).0);
    let single = owner.generated(
        scan.ir()?,
        pointers.clone(),
        scan_launch(batch * heads * chunk, columns)?,
    )?;
    let partials = owner.stream.alloc_zeros::<f32>(count * PARTITIONS)?;
    let normalizers = owner
        .stream
        .alloc_zeros::<f32>(batch * heads * chunk * PARTITIONS * 2)?;
    let mut pp = pointers.clone();
    pp[6] = partials.device_ptr(&owner.stream).0;
    pp.push(normalizers.device_ptr(&owner.stream).0);
    let partial = owner.generated(
        AttentionScan {
            partitions: PARTITIONS,
            ..scan
        }
        .ir()?,
        pp,
        scan_launch(batch * heads * chunk * PARTITIONS, columns)?,
    )?;
    let merge = owner.generated(
        MergeScan {
            rows: batch * heads * chunk,
            columns,
            partitions: PARTITIONS,
            dtype: DType::BF16,
            raw_partials: false,
        }
        .ir()?,
        vec![
            partials.device_ptr(&owner.stream).0,
            normalizers.device_ptr(&owner.stream).0,
            output.device_ptr(&owner.stream).0,
        ],
        scan_launch(batch * heads * chunk, columns)?,
    )?;
    let length32 = owner
        .stream
        .memcpy_stod(&scan_lengths(prefix, chunk, 32)?)?;
    let partials32 = owner.stream.alloc_zeros::<f32>(count * 32)?;
    let normalizers32 = owner
        .stream
        .alloc_zeros::<f32>(batch * heads * chunk * 32)?;
    let mut pp32 = pointers.clone();
    pp32[5] = length32.device_ptr(&owner.stream).0;
    pp32[6] = partials32.device_ptr(&owner.stream).0;
    pp32.push(normalizers32.device_ptr(&owner.stream).0);
    let partial32 = owner.generated(
        AttentionScan {
            partitions: 32,
            ..scan
        }
        .ir()?,
        pp32,
        scan_launch(batch * heads * chunk * 32, columns)?,
    )?;
    let merge32 = owner.generated(
        MergeScan {
            rows: batch * heads * chunk,
            columns,
            partitions: 32,
            dtype: DType::BF16,
            raw_partials: false,
        }
        .ir()?,
        vec![
            partials32.device_ptr(&owner.stream).0,
            normalizers32.device_ptr(&owner.stream).0,
            output.device_ptr(&owner.stream).0,
        ],
        scan_launch(batch * heads * chunk, columns)?,
    )?;
    let tiled = if columns <= 128 && chunk >= 16 {
        Some(owner.generated(
            prefill::ir(&scan)?,
            pointers.clone(),
            prefill::launch(&scan)?,
        )?)
    } else {
        None
    };
    let make_tiled_split = |partitions| -> Result<_> {
        Ok(if columns <= 128 && chunk >= 16 {
            let mut pp = pointers.clone();
            pp[6] = partials.device_ptr(&owner.stream).0;
            pp.push(normalizers.device_ptr(&owner.stream).0);
            let scan = AttentionScan { partitions, ..scan };
            Some((
                owner.generated(prefill::ir(&scan)?, pp, prefill::launch(&scan)?)?,
                owner.generated(
                    MergeScan {
                        rows: batch * heads * chunk,
                        columns,
                        partitions,
                        dtype: DType::BF16,
                        raw_partials: true,
                    }
                    .ir()?,
                    vec![
                        partials.device_ptr(&owner.stream).0,
                        normalizers.device_ptr(&owner.stream).0,
                        output.device_ptr(&owner.stream).0,
                    ],
                    scan_launch(batch * heads * chunk, columns)?,
                )?,
            ))
        } else {
            None
        })
    };
    let tiled_split = make_tiled_split(4)?;
    let tiled_split8 = make_tiled_split(8)?;
    let tiled_split16 = make_tiled_split(16)?;
    let mut schedules = vec![
        ("scalar", vec![&single]),
        ("split", vec![&partial, &merge]),
        ("split32", vec![&partial32, &merge32]),
    ];
    if let Some(kernel) = &tiled {
        schedules.push(("tiled", vec![kernel]));
    }
    if let Some((partial, merge)) = &tiled_split {
        schedules.push(("tiled_split", vec![partial, merge]));
    }
    if let Some((partial, merge)) = &tiled_split8 {
        schedules.push(("tiled_split8", vec![partial, merge]));
    }
    if let Some((partial, merge)) = &tiled_split16 {
        schedules.push(("tiled_split16", vec![partial, merge]));
    }
    let select = |old: &[bf16], new: &[bf16], b: usize, pos: usize, c: usize| -> f64 {
        f64::from(
            if pos < prefix {
                old[(b * capacity + pos) * width + c]
            } else {
                new[(b * chunk + pos - prefix) * width + c]
            }
            .to_f32(),
        )
    };
    let mut expected = Vec::with_capacity(count);
    for b in 0..batch {
        for h in 0..heads {
            for qi in 0..chunk {
                let scores: Vec<f64> = (0..prefix + qi + 1)
                    .map(|pos| {
                        (0..columns)
                            .map(|c| {
                                f64::from(q[(b * chunk + qi) * width + h * columns + c].to_f32())
                                    * select(&old_k, &k, b, pos, h * columns + c)
                            })
                            .sum::<f64>()
                            / (columns as f64).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
                let den: f64 = weights.iter().sum();
                for c in 0..columns {
                    expected.push(
                        (weights
                            .iter()
                            .enumerate()
                            .map(|(pos, w)| w * select(&old_v, &v, b, pos, h * columns + c))
                            .sum::<f64>()
                            / den) as f32,
                    );
                }
            }
        }
    }
    let mut results = Vec::new();
    for (name, kernels) in schedules {
        for kernel in &kernels {
            kernel.run(&owner.stream)?;
        }
        let actual: Vec<f32> = owner
            .stream
            .memcpy_dtov(&output)?
            .into_iter()
            .map(bf16::to_f32)
            .collect();
        owner.stream.synchronize()?;
        let scale = expected.iter().map(|x| x.abs()).fold(1f32, f32::max);
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            actual.iter().all(|x| x.is_finite()) && max_error <= scale / 128.,
            "{name} D={columns} Q={chunk} prefix={prefix}: {max_error}"
        );
        let mut micros = Vec::new();
        let mut captured_micros = Vec::new();
        if timed {
            for _ in 0..3 {
                for kernel in &kernels {
                    kernel.run(&owner.stream)?;
                }
            }
            for _ in 0..16 {
                let start = owner
                    .stream
                    .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                for _ in 0..20 {
                    for kernel in &kernels {
                        kernel.run(&owner.stream)?;
                    }
                }
                let end = owner
                    .stream
                    .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                micros.push(f64::from(start.elapsed_ms(&end)?) * 1000. / 20.);
            }
        }
        if timed {
            captured_micros = captured_samples(owner, &kernels)?;
        }
        let kernel_captured_micros = if timed && kernels.len() > 1 {
            kernels
                .iter()
                .map(|kernel| captured_samples(owner, &[*kernel]))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let resources=kernels.iter().map(|kernel| -> Result<_> { Ok(json!({
            "registers_per_thread":kernel.function.num_regs()?,
            "local_bytes_per_thread":kernel.function.local_size_bytes()?,
            "static_shared_bytes":kernel.function.shared_size_bytes()?,
            "active_blocks_per_sm":kernel.function.occupancy_max_active_blocks_per_multiprocessor(kernel.config.block_dim.0*kernel.config.block_dim.1,0,None)?
        })) }).collect::<Result<Vec<_>>>()?;
        results.push(json!({"schedule":name,"output":actual,"max_abs_vs_f64":max_error,"device_us_samples":micros,"captured_device_us_samples":captured_micros,"resources":resources,"kernel_captured_device_us_samples":kernel_captured_micros}));
    }
    let joined = |old: &[bf16], new: &[bf16]| -> Vec<f32> {
        (0..batch)
            .flat_map(|b| {
                (0..prefix + chunk)
                    .flat_map(move |p| (0..width).map(move |c| select(old, new, b, p, c) as f32))
            })
            .collect()
    };
    let automatic_schedule = if columns <= 128 && chunk >= 16 {
        if prefill::partitions(chunk) == 8 {
            "tiled_split8"
        } else {
            "tiled_split"
        }
    } else if prefix + chunk >= SPLIT_MIN_KEYS && chunk <= 4 {
        "split32"
    } else if prefix + chunk >= SPLIT_MIN_KEYS {
        "split"
    } else {
        "scalar"
    };
    Ok(
        json!({"batch":batch,"heads":heads,"columns":columns,"chunk":chunk,"prefix":prefix,"qk_gain":gain,"automatic_schedule":automatic_schedule,
        "q":q.into_iter().map(bf16::to_f32).collect::<Vec<_>>(),"k":joined(&old_k,&k),"v":joined(&old_v,&v),"results":results}),
    )
}

#[test]
fn bf16_attention_tails_prefixes_poisoned_slots_and_physical_storage() -> Result<()> {
    let Some(owner) = owner()? else {
        return Ok(());
    };
    for columns in [4, 33, 64, 128, 129] {
        for (chunk, prefix) in [(1, 0), (3, 65), (16, 0), (17, 31), (33, 129)] {
            run_case(&owner, columns, chunk, prefix, 2, 2, false, 1.)?;
        }
    }
    run_case(&owner, 64, 17, 513, 2, 2, false, 16.)?;
    run_case(&owner, 64, 17, 31, 2, 2, false, 0.01)?;
    run_case(&owner, 64, 1025, 31, 1, 1, false, 1.)?;
    let model = crate::nn::Gpt::new(crate::nn::GptConfig {
        vocab_size: 16,
        max_sequence_length: 32,
        embed_dim: 64,
        num_heads: 2,
        feed_forward_dim: 64,
        num_layers: 2,
        layer_norm_epsilon: 1e-5,
    });
    let mut decoder = model.decoder_with_options(crate::Backend::Ptx, 2, 32, owner.options)?;
    decoder.decode(&vec![1; 32])?;
    let super::super::DecoderBackend::Cuda(backend) = &decoder.backend else {
        unreachable!()
    };
    assert!(
        backend
            .cache
            .values()
            .all(|buffer| matches!(buffer, Buffer::BF16(_)))
    );
    let bytes: usize = backend
        .cache
        .values()
        .map(|b| {
            let x = crate::ptx::linear::program::binding(b, &backend.stream);
            x.len * x.dtype.size_bytes()
        })
        .sum();
    assert_eq!(bytes, 2 * 2 * 2 * 32 * 64 * 2);
    assert!(backend.compiled[&16].split.is_empty());
    assert_eq!(
        backend.compiled[&16]
            .kernels
            .iter()
            .filter(|k| k.label == "decoder tensor-core prefill")
            .count(),
        2
    );
    Ok(())
}

/// Export exact BF16 inputs/native results for the independent PyTorch SDPA oracle.
#[test]
#[ignore = "opt-in GPU timing and SDPA fixture export"]
fn bf16_sdpa_fixtures() -> Result<()> {
    let owner = owner()?.context("CUDA required for fixture export")?;
    let path = std::env::var("TNSR_SDPA_FIXTURES")
        .context("set TNSR_SDPA_FIXTURES to an output JSON path")?;
    let selected_chunk = std::env::var("TNSR_SDPA_CHUNK")
        .ok()
        .map(|x| x.parse::<usize>())
        .transpose()?;
    let mut cases = Vec::new();
    for (batch, heads, columns, chunk, prefix) in [
        (1, 4, 64, 128, 0),
        (1, 4, 64, 512, 0),
        (1, 4, 64, 2048, 0),
        (2, 2, 33, 17, 31),
        (1, 4, 64, 1, 2048),
        (2, 2, 128, 33, 65),
        (1, 1, 129, 3, 129),
    ] {
        if selected_chunk.is_some_and(|selected| selected != chunk) {
            continue;
        }
        cases.push(run_case(
            &owner, columns, chunk, prefix, batch, heads, true, 1.,
        )?);
    }
    std::fs::write(
        path,
        serde_json::to_vec(&json!({"seed_formula":"BF16(((i*17+salt)%101)/53-1)","cases":cases}))?,
    )?;
    Ok(())
}

#[test]
fn shared_accumulator_layout_round_trip_preserves_f32_values() -> Result<()> {
    use crate::tile::*;
    let Some(owner) = owner()? else {
        return Ok(());
    };
    for operand_dtype in [DType::BF16, DType::F16, DType::TF32] {
        for (rows, fragments) in [(32, (1, 1)), (64, (2, 1))] {
            let cols = 64;
            let mut b = ScalarBuilder::default();
            let source = b.var();
            let target = b.var();
            let fragment = b.var();
            for var in [source, target] {
                b.stmts.push(Stmt::AllocTile {
                    var,
                    space: MemorySpace::Shared,
                    layout: TileLayout::Shared(SharedLayout::row_major(cols)),
                    dtype: DType::F32,
                    rows,
                    cols,
                });
            }
            b.stmts.push(Stmt::AllocTile {
                var: fragment,
                space: MemorySpace::Fragment,
                layout: TileLayout::WarpAccumulator {
                    operand_dtype,
                    block_width: 32,
                    warp_topology: (2, 4),
                    warp_fragments: fragments,
                },
                dtype: DType::F32,
                rows,
                cols,
            });
            let bound = b.var();
            let alias = b.var();
            b.stmts.push(Stmt::AllocTile {
                var: bound,
                space: MemorySpace::Register,
                layout: TileLayout::ThreadScalar,
                dtype: DType::F32,
                rows: 1,
                cols: 1,
            });
            b.stmts.push(Stmt::LoadSharedToReg {
                dest: bound,
                src: source,
            });
            b.stmts.push(Stmt::AliasShared {
                dest: alias,
                src: bound,
                layout: SharedLayout::row_major(cols),
                rows,
                cols,
            });
            let tid = Expr::ThreadIdx(Dim::Y) * 32usize + Expr::ThreadIdx(Dim::X);
            for i in 0..rows * cols / 256 {
                let index = tid.clone() + Expr::Const((i * 256) as i64);
                let value = b.load(ScalarMemory::Global("input".into()), index.clone());
                b.store(ScalarMemory::Shared(source), index, value);
            }
            b.barrier();
            b.stmts.push(Stmt::ConvertLayout {
                dest: fragment,
                src: alias,
                coordinates: None,
            });
            b.stmts.push(Stmt::ConvertLayout {
                dest: target,
                src: fragment,
                coordinates: None,
            });
            b.barrier();
            for i in 0..rows * cols / 256 {
                let index = tid.clone() + Expr::Const((i * 256) as i64);
                let value = b.load(ScalarMemory::Shared(target), index.clone());
                b.store(ScalarMemory::Global("output".into()), index, value);
            }
            let ir = TileIR {
                kernel_name: "accumulator_round_trip".into(),
                params: vec![
                    KernelParam {
                        name: "input".into(),
                        dtype: DType::F32,
                        is_input: true,
                    },
                    KernelParam {
                        name: "output".into(),
                        dtype: DType::F32,
                        is_input: false,
                    },
                ],
                body: Block { stmts: b.stmts },
                shared_mem_bytes: 2 * rows * cols * 4,
            };
            let values: Vec<f32> = (0..rows * cols)
                .map(|i| (i as f32 - 1023.5) / 73.)
                .collect();
            let input = owner.stream.memcpy_stod(&values)?;
            let output = owner.stream.alloc_zeros::<f32>(values.len())?;
            let kernel = owner.generated(
                ir,
                vec![
                    input.device_ptr(&owner.stream).0,
                    output.device_ptr(&owner.stream).0,
                ],
                cudarc::driver::LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 8, 1),
                    shared_mem_bytes: 0,
                },
            )?;
            kernel.run(&owner.stream)?;
            assert_eq!(owner.stream.memcpy_dtov(&output)?, values);
            owner.stream.synchronize()?;
        }
    }
    Ok(())
}
