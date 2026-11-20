//! Resident GEMM shared-scratch A/B; uploads, compilation and readback excluded.
//! Uploads/readback excluded; both modes compare every output bit before timing.
#[cfg(feature = "cuda")]
mod cuda {
    use anyhow::{Context, Result, ensure};
    use clap::Parser;
    use cudarc::driver::{
        CudaContext, CudaFunction, CudaGraph, CudaModule, CudaSlice, DevicePtr, LaunchConfig,
        PushKernelArg, sys,
    };
    use cudarc::nvrtc::Ptx;
    use num_traits::{ToPrimitive, cast};
    use serde_json::json;
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tnsr::{
        graph::{TensorGraph, TensorGraphNode},
        ptx::{
            PtxExecutor, PtxPlanAction,
            types::{BF16, CudaDType, F16, F32},
        },
        tensor::{Parameter, TensorExpr},
        tile::{MatMulPipeline, MatMulPlan, MatMulPrecision, MatMulSchedule},
    };

    #[derive(Parser)]
    struct Args {
        #[arg(long, default_value_t = 2048)]
        m: usize,
        #[arg(long, default_value_t = 768)]
        n: usize,
        #[arg(long, default_value_t = 256)]
        k: usize,
        #[arg(long, default_value = "bf16")]
        dtype: String,
        #[arg(long, default_value = "native")]
        precision: String,
        /// Reverse to candidate,baseline for an independent process order.
        #[arg(long, default_value = "baseline,candidate")]
        order: String,
        /// Staging policy: auto, sync, or double.
        #[arg(long, default_value = "auto")]
        pipeline: String,
        /// Save baseline.ptx and candidate.ptx in this directory.
        #[arg(long)]
        ptx_dir: Option<std::path::PathBuf>,
        /// Steady prewarm after both modules/captures exist, excluded from samples.
        #[arg(long, default_value_t = 200)]
        warmup_ms: u64,
        #[arg(long, default_value_t = 20)]
        samples: usize,
        #[arg(long, default_value_t = 16)]
        repeats: usize,
        #[arg(long)]
        transpose_a: bool,
        #[arg(long)]
        transpose_b: bool,
        /// Preserve a leading physical batch axis before flattening a transposed LHS.
        #[arg(long, default_value_t = 1)]
        reshape_batch: usize,
        /// Save complete first-mode output bits for comparisons between compiler builds.
        #[arg(long)]
        output_bits: Option<std::path::PathBuf>,
    }

    struct Mode<D: CudaDType> {
        // Graph goes first so it is destroyed while its module and storage are live.
        graph: CudaGraph,
        _module: Arc<CudaModule>,
        _function: CudaFunction,
        _storage: Vec<CudaSlice<D::HostType>>,
        enabled: bool,
        resources: serde_json::Value,
        samples_us: Vec<f64>,
    }

    pub fn run_main() -> Result<()> {
        let args = Args::parse();
        ensure!(
            args.samples >= 2
                && args.repeats > 0
                && [args.m, args.n, args.k].iter().all(|&d| d > 0),
            "positive dimensions/repeats and at least two samples required"
        );
        ensure!(
            args.reshape_batch > 0
                && (args.reshape_batch == 1
                    || (args.transpose_a && args.k.is_multiple_of(args.reshape_batch))),
            "reshape batch requires a transposed LHS and must divide K"
        );
        match args.dtype.as_str() {
            "bf16" => run::<BF16>(&args),
            "f16" => run::<F16>(&args),
            "f32" => run::<F32>(&args),
            other => anyhow::bail!("unsupported dtype {other}"),
        }
    }

    fn run<D: CudaDType>(args: &Args) -> Result<()> {
        let precision = match args.precision.as_str() {
            "strict" => MatMulPrecision::StrictF32,
            "native" => MatMulPrecision::NativeHalf,
            "tf32" => MatMulPrecision::AllowTf32,
            "tf32x3" => MatMulPrecision::Tf32x3,
            other => anyhow::bail!("unsupported precision {other}; use strict/native/tf32/tf32x3"),
        };
        ensure!(
            args.dtype == "f32"
                || matches!(
                    precision,
                    MatMulPrecision::NativeHalf | MatMulPrecision::StrictF32
                ),
            "half storage requires native or strict precision"
        );
        let pipeline = match args.pipeline.as_str() {
            "auto" => MatMulPipeline::Auto,
            "sync" => MatMulPipeline::Synchronous,
            "double" => MatMulPipeline::DoubleBuffered,
            other => anyhow::bail!("unsupported pipeline {other}; use auto/sync/double"),
        };
        let groups: Vec<bool> = args
            .order
            .split(',')
            .map(|mode| match mode {
                "baseline" => Ok(false),
                "candidate" => Ok(true),
                _ => anyhow::bail!("mode must be baseline or candidate"),
            })
            .collect::<Result<_>>()?;
        ensure!(
            groups.len() == 2 && groups.contains(&false) && groups.contains(&true),
            "order must contain baseline and candidate exactly once"
        );
        let values = |count, modulus| -> Result<Vec<D::HostType>> {
            (0..count)
                .map(|i| {
                    cast(
                        (((i % modulus) * 13 % modulus) as f32 - (modulus / 2) as f32)
                            / modulus as f32,
                    )
                    .context("input conversion failed")
                })
                .collect()
        };
        let a_len = args
            .m
            .checked_mul(args.k)
            .context("A extent overflows usize")?;
        let b_len = args
            .k
            .checked_mul(args.n)
            .context("B extent overflows usize")?;
        let output_len = args
            .m
            .checked_mul(args.n)
            .context("output extent overflows usize")?;
        let a = values(a_len, 97)?;
        let b = values(b_len, 89)?;
        let lhs: TensorExpr<D::HostType> = if args.transpose_a {
            let data = (0..args.k)
                .flat_map(|j| (0..args.m).map(move |row| row * args.k + j))
                .map(|index| a[index])
                .collect();
            let shape = if args.reshape_batch == 1 {
                vec![args.k, args.m]
            } else {
                vec![args.reshape_batch, args.k / args.reshape_batch, args.m]
            };
            TensorExpr::from(Parameter::new(data, shape))
                .reshape(vec![args.k, args.m])
                .transpose()
        } else {
            Parameter::new(a.clone(), vec![args.m, args.k]).into()
        };
        let rhs: TensorExpr<D::HostType> = if args.transpose_b {
            let data = (0..args.n)
                .flat_map(|col| (0..args.k).map(move |j| j * args.n + col))
                .map(|index| b[index])
                .collect();
            TensorExpr::from(Parameter::new(data, vec![args.n, args.k])).transpose()
        } else {
            Parameter::new(b.clone(), vec![args.k, args.n]).into()
        };
        let graph: TensorGraph<D::HostType> = lhs.matmul(rhs).into();
        let device = CudaContext::new(0)?;
        let stream = device.new_stream()?;
        let mut modes = Vec::new();
        let mut original = Vec::new();
        let mut reference_schedule: Option<MatMulSchedule> = None;
        let mut sanity = Vec::new();
        if let Some(directory) = &args.ptx_dir {
            std::fs::create_dir_all(directory).context("create PTX output directory")?;
        }
        for enabled in groups {
            let mut executor = PtxExecutor::<D>::try_new_for_dtype()?;
            executor.set_matmul_precision(precision);
            executor.set_matmul_pipeline(pipeline);
            executor.set_experimental_shared_memory_reuse(enabled);
            executor.compile(&graph)?;
            let plan = executor
                .execution_plan()
                .context("compiled plan unavailable")?;
            let step = plan
                .steps()
                .iter()
                .find(|s| {
                    matches!(
                        s.action,
                        PtxPlanAction::Kernel | PtxPlanAction::MatMulRegion(_)
                    )
                })
                .context("matmul absent")?;
            ensure!(
                step.inputs.len() == 2 && step.outputs.len() == 1,
                "probe requires one plain GEMM"
            );
            let description = executor.describe_plan(&graph)?;
            let name = description
                .lines()
                .find_map(|line| {
                    line.split("action=kernel ")
                        .nth(1)
                        .and_then(|s| s.split_whitespace().next())
                })
                .context("kernel name unavailable")?;
            if let Some(directory) = &args.ptx_dir {
                std::fs::write(
                    directory.join(if enabled {
                        "candidate.ptx"
                    } else {
                        "baseline.ptx"
                    }),
                    executor.module_source().context("PTX unavailable")?,
                )?;
            }
            let module = device.load_module(Ptx::from_src(
                executor.module_source().context("PTX unavailable")?,
            ))?;
            let function = module.load_function(name)?;
            let mut storage = Vec::new();
            for &input in &step.inputs {
                let TensorGraphNode::Parameter { data, .. } = &graph[input] else {
                    anyhow::bail!("unexpected non-parameter input")
                };
                let data = data
                    .lock()
                    .map_err(|_| anyhow::anyhow!("parameter lock poisoned"))?;
                storage.push(stream.memcpy_stod(data.as_slice())?);
            }
            storage.push(stream.alloc_zeros::<D::HostType>(output_len)?);
            let schedule = match step.action {
                PtxPlanAction::MatMulRegion(id) => plan.matmul_regions()[id].schedule,
                _ => *plan
                    .matmul_schedules()
                    .get(&step.node)
                    .context("schedule unavailable")?,
            };
            let config = LaunchConfig {
                grid_dim: (
                    u32::try_from(args.n.div_ceil(schedule.block_tile.n))?,
                    u32::try_from(args.m.div_ceil(schedule.block_tile.m))?,
                    1,
                ),
                block_dim: schedule.block_threads,
                shared_mem_bytes: 0,
            };
            ensure!(
                schedule.plan != MatMulPlan::ScalarF32 || precision == MatMulPrecision::StrictF32,
                "tensor-core schedule was not selected; use --precision strict only for an explicit scalar control"
            );
            ensure!(
                schedule.reuse_shared_memory == (enabled && schedule.plan != MatMulPlan::ScalarF32),
                "shared-memory reuse was not applied to the selected schedule"
            );
            let normalized_schedule = schedule.with_shared_memory_reuse(false);
            if let Some(reference) = reference_schedule {
                ensure!(
                    normalized_schedule == reference,
                    "baseline/candidate schedules differ beyond the reuse flag"
                );
            } else {
                reference_schedule = Some(normalized_schedule);
            }
            ensure!(
                usize::try_from(function.shared_size_bytes()?)? == schedule.shared_mem_bytes(),
                "actual static shared allocation differs from schedule accounting"
            );
            let resources = json!({
                "plan": format!("{:?}", schedule.plan),
                "reuse_shared_memory": schedule.reuse_shared_memory,
                "block_tile": [schedule.block_tile.m, schedule.block_tile.n, schedule.block_tile.k],
                "instruction_tile": [schedule.instruction_tile.m, schedule.instruction_tile.n, schedule.instruction_tile.k],
                "block_threads": schedule.block_threads,
                "warp_topology": schedule.warp_topology,
                "warp_fragments": schedule.warp_fragments,
                "pipeline_stages": schedule.pipeline_stages,
                "copy_bytes": schedule.copy_bytes,
                "async_operands": schedule.async_operands,
                "operand_layouts": [
                    {"row_stride": schedule.operand_layouts.0.row_stride, "column_major": schedule.operand_layouts.0.column_major, "xor_mask": schedule.operand_layouts.0.xor_mask},
                    {"row_stride": schedule.operand_layouts.1.row_stride, "column_major": schedule.operand_layouts.1.column_major, "xor_mask": schedule.operand_layouts.1.xor_mask}
                ],
                "registers_per_thread": function.num_regs()?,
                "static_shared_bytes": function.shared_size_bytes()?,
                "local_bytes_per_thread": function.local_size_bytes()?,
                "active_blocks_per_sm": function.occupancy_max_active_blocks_per_multiprocessor(
                    u32::try_from(schedule.thread_count())?, 0, None)?,
            });
            let pointers: Vec<u64> = storage.iter().map(|s| s.device_ptr(&stream).0).collect();
            let submit = || -> Result<()> {
                let mut launch = stream.launch_builder(&function);
                for pointer in &pointers {
                    launch.arg(pointer);
                }
                // SAFETY: pointers refer to correctly typed, live input/output storage;
                // launch geometry comes from this module's selected matmul schedule.
                unsafe { launch.launch(config) }?;
                Ok(())
            };
            submit()?;
            let actual = stream.memcpy_dtov(storage.last().context("output absent")?)?;
            let actual_f32: Vec<f32> = actual
                .iter()
                .map(|v| v.to_f32().context("output conversion failed"))
                .collect::<Result<_>>()?;
            ensure!(
                actual_f32.iter().all(|value| value.is_finite()),
                "nonfinite output in mode={enabled}"
            );
            if original.is_empty() {
                for sample in 0..32 {
                    let row = sample * 7919 % args.m;
                    let col = sample * 104729 % args.n;
                    let (expected, absolute_products) = (0..args.k).try_fold(
                        (0.0, 0.0),
                        |(sum, absolute), j| -> Result<(f64, f64)> {
                            let product = a[row * args.k + j]
                                .to_f64()
                                .context("reference conversion failed")?
                                * b[j * args.n + col]
                                    .to_f64()
                                    .context("reference conversion failed")?;
                            Ok((sum + product, absolute + product.abs()))
                        },
                    )?;
                    let value = f64::from(actual_f32[row * args.n + col]);
                    // Broad independent sanity only. Full exact finite bits across
                    // the two GPU modes are the acceptance criterion below.
                    let tolerance = 0.01 * (1.0 + expected.abs())
                        + if precision == MatMulPrecision::AllowTf32 {
                            0.002 * absolute_products
                        } else {
                            0.0
                        };
                    ensure!(
                        (value - expected).abs() <= tolerance,
                        "sample mismatch at {row},{col}: {value} != {expected}"
                    );
                    sanity.push(json!({"row":row,"column":col,"f64_reference":expected,"actual":value,"absolute_error":(value-expected).abs(),"tolerance":tolerance}));
                }
                original = actual_f32.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                if let Some(path) = &args.output_bits {
                    let bytes: Vec<u8> = original
                        .iter()
                        .flat_map(|bits| bits.to_le_bytes())
                        .collect();
                    std::fs::write(path, bytes).context("write reference output bits")?;
                }
            } else {
                ensure!(
                    actual_f32
                        .iter()
                        .map(|x| x.to_bits())
                        .eq(original.iter().copied()),
                    "shared-memory reuse changed output bits, enabled={enabled}"
                );
            }
            stream.synchronize()?;
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let recorded = (|| {
                for _ in 0..args.repeats {
                    submit()?;
                }
                Ok::<_, anyhow::Error>(())
            })();
            let ended = stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            recorded?;
            let captured = ended?.context("empty probe capture")?;
            modes.push(Mode::<D> {
                graph: captured,
                _module: module,
                _function: function,
                _storage: storage,
                enabled,
                resources,
                samples_us: Vec::new(),
            });
        }
        // Prewarm both completed captures until the wall-time budget has elapsed.
        // Synchronize per pair to bound queued work and exit with the GPU idle.
        let warmup_started = Instant::now();
        let mut warmup_pairs = 0usize;
        while warmup_started.elapsed() < Duration::from_millis(args.warmup_ms.max(200)) {
            let order = if warmup_pairs.is_multiple_of(2) {
                [0, 1]
            } else {
                [1, 0]
            };
            for index in order {
                modes[index].graph.launch()?;
            }
            stream.synchronize()?;
            warmup_pairs += 1;
        }
        let warmup_elapsed_ms = warmup_started.elapsed().as_secs_f64() * 1000.0;
        let mut pairs = Vec::new();
        for sample in 0..args.samples {
            // Reverse only. Rotating AND reversing cancels for two modes.
            let order = if sample % 2 == 0 { [0, 1] } else { [1, 0] };
            let mut execution_order = Vec::new();
            let mut baseline_us = 0.0;
            let mut candidate_us = 0.0;
            for index in order {
                let mode = &mut modes[index];
                let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                mode.graph.launch()?;
                let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                let elapsed = f64::from(start.elapsed_ms(&end)?) * 1000.0 / args.repeats as f64;
                ensure!(
                    elapsed.is_finite() && elapsed > 0.0,
                    "invalid CUDA-event interval"
                );
                mode.samples_us.push(elapsed);
                execution_order.push(if mode.enabled {
                    "candidate"
                } else {
                    "baseline"
                });
                if mode.enabled {
                    candidate_us = elapsed;
                } else {
                    baseline_us = elapsed;
                }
            }
            pairs.push(json!({"sample":sample,"execution_order":execution_order,"baseline_us":baseline_us,"candidate_us":candidate_us,"throughput_ratio":baseline_us/candidate_us}));
        }
        stream.synchronize()?;
        // Validate the final result of repeated captured execution too. These
        // readbacks happen after every timed interval and are excluded from it.
        for mode in &modes {
            let actual = stream.memcpy_dtov(mode._storage.last().context("output absent")?)?;
            let bits = actual
                .iter()
                .map(|value| {
                    value
                        .to_f32()
                        .map(f32::to_bits)
                        .context("output conversion failed")
                })
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                bits == original,
                "captured execution changed output bits, enabled={}",
                mode.enabled
            );
        }
        let summary = modes.iter().map(|mode| {
            let mean = mode.samples_us.iter().sum::<f64>() / args.samples as f64;
            let mut sorted = mode.samples_us.clone();
            sorted.sort_by(f64::total_cmp);
            let median = if sorted.len() % 2 == 0 { (sorted[sorted.len()/2-1] + sorted[sorted.len()/2]) / 2.0 } else { sorted[sorted.len()/2] };
            json!({"mode":if mode.enabled { "candidate" } else { "baseline" },"resources":mode.resources,"mean_us":mean,"median_us":median,"samples_us":mode.samples_us})
        }).collect::<Vec<_>>();
        let baseline_total = modes
            .iter()
            .find(|mode| !mode.enabled)
            .context("baseline absent")?
            .samples_us
            .iter()
            .sum::<f64>();
        let candidate_total = modes
            .iter()
            .find(|mode| mode.enabled)
            .context("candidate absent")?
            .samples_us
            .iter()
            .sum::<f64>();
        println!(
            "{}",
            json!({
                "m":args.m,"n":args.n,"k":args.k,"transpose_a":args.transpose_a,"transpose_b":args.transpose_b,"reshape_batch":args.reshape_batch,
                "dtype":args.dtype,"precision":args.precision,"pipeline_policy":args.pipeline,"construction_order":args.order,
                "samples":args.samples,"repeats":args.repeats,"warmup_pairs":warmup_pairs,"warmup_elapsed_ms":warmup_elapsed_ms,
                "qualified":true,"full_output_bits_equal":true,"full_output_bits_equal_after_timing":true,"finite_outputs":true,"output_elements":output_len,"sampled_f64_sanity":sanity,
                "protocol":"alternating AB/BA pairs after both captures exist and >=200ms prewarm; resident preallocated storage; CUDA graph repetitions; CUDA events; uploads/readback/compilation excluded; every finite output bit compared across modes before timing; scalar control requires explicit strict precision",
                "scope":"resident cache-hot single GEMM; not complete training updates",
                "paired_throughput_ratio":baseline_total/candidate_total,"pairs":pairs,"results":summary
            })
        );
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    cuda::run_main()
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("enable --features cuda for the shared-memory reuse probe");
}
