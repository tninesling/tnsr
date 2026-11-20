//! Resident comparison of experimental Online weight sharing. This measures one
//! normalized contraction, not attention score generation or a complete decoder.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaModule, CudaSlice, DevicePtr, LaunchConfig,
    PushKernelArg, sys,
};
use cudarc::nvrtc::Ptx;
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tnsr::{
    graph::{TensorGraph, TensorGraphNode},
    nn::softmax,
    ptx::{F32, PtxExecutor, PtxPlanAction, PtxReductionMode},
    tensor::TensorExpr,
};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 512)]
    rows: usize,
    #[arg(long, default_value_t = 64)]
    keys: usize,
    #[arg(long, default_value_t = 32)]
    width: usize,
    #[arg(long, default_value_t = 32)]
    samples: usize,
    #[arg(long, default_value_t = 32)]
    repeats: usize,
    #[arg(long, default_value = "baseline,candidate")]
    order: String,
    /// Steady alternating work after both graphs are ready, excluded from timing.
    #[arg(long, default_value_t = 200)]
    warmup_ms: u64,
}

struct Mode {
    // Graph destruction precedes module and bound-storage destruction.
    graph: CudaGraph,
    _module: Arc<CudaModule>,
    _function: CudaFunction,
    storage: Vec<Arc<CudaSlice<f32>>>,
    label: &'static str,
    effective_sharing: bool,
    resources: serde_json::Value,
    samples_us: Vec<f64>,
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) * 0.5
    } else {
        sorted[middle]
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        [args.rows, args.keys, args.width, args.repeats]
            .iter()
            .all(|&n| n > 0)
            && args.samples >= 2
            && args.warmup_ms >= 200,
        "positive dimensions/repeats, at least two samples and >=200ms warmup required"
    );
    let order = match args.order.as_str() {
        "baseline,candidate" => [false, true],
        "candidate,baseline" => [true, false],
        _ => anyhow::bail!("order must be baseline,candidate or candidate,baseline"),
    };
    let score_count = args
        .rows
        .checked_mul(args.keys)
        .context("score size overflow")?;
    let value_count = args
        .keys
        .checked_mul(args.width)
        .context("value size overflow")?;
    let output_count = args
        .rows
        .checked_mul(args.width)
        .context("output size overflow")?;
    let scores: Vec<f32> = (0..score_count)
        .map(|i| ((i % 97) as f32 - 48.) * 0.125)
        .collect();
    let values: Vec<f32> = (0..value_count)
        .map(|i| ((i % 89) as f32 - 44.) * 0.015625)
        .collect();
    let expression = softmax(TensorExpr::input("scores", vec![args.rows, args.keys]), 1)
        .matmul(TensorExpr::input("values", vec![args.keys, args.width]));
    let graph: TensorGraph<f32> = expression.into();
    let device = CudaContext::new(0)?;
    let stream = device.new_stream()?;
    // Both modes bind exactly the same resident, read-only input allocations.
    let resident_scores = Arc::new(stream.memcpy_stod(&scores)?);
    let resident_values = Arc::new(stream.memcpy_stod(&values)?);
    let mut modes = Vec::new();
    let mut canonical_bits: Option<Vec<u32>> = None;
    let mut maximum_reference_error = 0.0_f64;

    for sharing in order {
        let label = if sharing { "candidate" } else { "baseline" };
        let mut executor =
            PtxExecutor::<F32>::try_new_with_reduction_mode(PtxReductionMode::Online)?;
        executor.set_experimental_online_weight_sharing(sharing);
        executor.compile(&graph)?;
        let plan = executor.execution_plan().context("compiled plan absent")?;
        ensure!(
            plan.online_regions().len() == 1,
            "expected one OnlineRegion"
        );
        ensure!(
            plan.steps().iter().all(|step| matches!(
                step.action,
                PtxPlanAction::Upload | PtxPlanAction::VirtualView | PtxPlanAction::OnlineRegion(0)
            )),
            "probe requires a single online kernel without additional work"
        );
        let step = plan
            .steps()
            .iter()
            .find(|step| step.action == PtxPlanAction::OnlineRegion(0))
            .context("online step absent")?;
        let region = &plan.online_regions()[0];
        ensure!(
            region.score_shape == [args.rows, args.keys]
                && region.output_shape == [args.rows, args.width]
                && region.inputs.len() == 2
                && step.inputs.len() == 2
                && step.outputs.len() == 1
                && region.share_weights == sharing,
            "unexpected normalized contraction shape/bindings"
        );
        let description = executor.describe_plan(&graph)?;
        let name = description
            .lines()
            .find_map(|line| {
                line.split("action=kernel ")
                    .nth(1)
                    .and_then(|tail| tail.split_whitespace().next())
            })
            .context("online kernel name absent")?;
        let module = device.load_module(Ptx::from_src(
            executor.module_source().context("PTX absent")?,
        ))?;
        let function = module.load_function(name)?;
        let mut storage = Vec::new();
        for (descriptor, &input) in region.inputs.iter().zip(&step.inputs) {
            ensure!(
                descriptor.tensor.source == input,
                "input ABI order mismatch"
            );
            let TensorGraphNode::Input { name, .. } = graph[input] else {
                anyhow::bail!("online region binds a non-input source");
            };
            storage.push(match name {
                "scores" => resident_scores.clone(),
                "values" => resident_values.clone(),
                _ => anyhow::bail!("unrecognized online input {name}"),
            });
        }
        storage.push(Arc::new(stream.alloc_zeros::<f32>(output_count)?));
        let cooperative = region.block_threads();
        let effective_sharing = sharing && cooperative.is_some();
        ensure!(
            cooperative.is_some() == (args.width <= 128),
            "unexpected scalar fallback"
        );
        let config = if let Some(threads) = cooperative {
            LaunchConfig {
                grid_dim: (u32::try_from(args.rows)?, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            }
        } else {
            // Mirrors the current independent scalar-output launch helper.
            LaunchConfig {
                grid_dim: (u32::try_from(output_count)?.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            }
        };
        let active_blocks =
            function.occupancy_max_active_blocks_per_multiprocessor(config.block_dim.0, 0, None)?;
        ensure!(
            active_blocks > 0,
            "compiled kernel has no resident block at selected geometry"
        );
        let resources = json!({
            "kernel": name,
            "grid_dim": [config.grid_dim.0, config.grid_dim.1, config.grid_dim.2],
            "block_dim": [config.block_dim.0, config.block_dim.1, config.block_dim.2],
            "registers_per_thread": function.num_regs()?,
            "static_shared_bytes": function.shared_size_bytes()?,
            "local_bytes_per_thread": function.local_size_bytes()?,
            "active_blocks_per_sm": active_blocks,
        });
        let pointers: Vec<u64> = storage
            .iter()
            .map(|buffer| buffer.device_ptr(&stream).0)
            .collect();
        let submit = || -> Result<()> {
            let mut launch = stream.launch_builder(&function);
            for pointer in &pointers {
                launch.arg(pointer);
            }
            // SAFETY: arguments follow this region's input ABI and one f32 output;
            // all allocations remain live and match source/output extents. Launch
            // geometry matches the cooperative or guarded serial scan schedule.
            unsafe { launch.launch(config) }?;
            Ok(())
        };
        submit()?;
        let actual = stream.memcpy_dtov(storage.last().context("output absent")?.as_ref())?;
        ensure!(
            actual.iter().all(|x| x.is_finite()),
            "nonfinite probe output in {label}"
        );
        let bits: Vec<_> = actual.iter().map(|x| x.to_bits()).collect();
        if let Some(reference) = &canonical_bits {
            ensure!(
                &bits == reference,
                "{label} changes complete finite output bits"
            );
        } else {
            canonical_bits = Some(bits);
        }
        // Independent stable f64 reference on both modes, before any timed work.
        for sample in 0..64 {
            let row = (sample * 7919) % args.rows;
            let column = (sample * 104729) % args.width;
            let row_scores = &scores[row * args.keys..(row + 1) * args.keys];
            let maximum = row_scores
                .iter()
                .copied()
                .map(f64::from)
                .fold(f64::NEG_INFINITY, f64::max);
            let mut denominator = 0.0_f64;
            let mut numerator = 0.0_f64;
            for (key, score) in row_scores.iter().enumerate() {
                let weight = (f64::from(*score) - maximum).exp();
                denominator += weight;
                numerator += weight * f64::from(values[key * args.width + column]);
            }
            let expected = numerator / denominator;
            let error = (f64::from(actual[row * args.width + column]) - expected).abs();
            maximum_reference_error = maximum_reference_error.max(error);
            ensure!(
                error <= 1e-5 + 1e-4 * expected.abs(),
                "{label} f64 reference mismatch at {row},{column}: error={error}"
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
        // End capture even if recording fails, so stream ownership stays valid.
        let ended = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        recorded?;
        modes.push(Mode {
            graph: ended?.context("empty capture")?,
            _module: module,
            _function: function,
            storage,
            label,
            effective_sharing,
            resources,
            samples_us: Vec::new(),
        });
    }
    // Both modes must be ready before warming; interleave them to avoid making
    // compilation order determine which kernel receives the hot device.
    let warm_started = Instant::now();
    let mut warm_pairs = 0;
    while warm_started.elapsed() < Duration::from_millis(args.warmup_ms) {
        let order = if warm_pairs % 2 == 0 { [0, 1] } else { [1, 0] };
        for index in order {
            modes[index].graph.launch()?;
        }
        stream.synchronize()?;
        warm_pairs += 1;
    }
    let actual_warmup_ms = warm_started.elapsed().as_secs_f64() * 1000.;
    let mut pairs = Vec::new();
    for sample in 0..args.samples {
        // With two modes rotate+reverse would cancel. Reverse ONLY on odd pairs.
        let order = if sample % 2 == 0 { [0, 1] } else { [1, 0] };
        let mut labels = Vec::new();
        let mut baseline_us = 0.;
        let mut candidate_us = 0.;
        for index in order {
            let mode = &mut modes[index];
            let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            mode.graph.launch()?;
            let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            let elapsed_us = f64::from(start.elapsed_ms(&end)?) * 1000. / args.repeats as f64;
            ensure!(
                elapsed_us.is_finite() && elapsed_us > 0.,
                "invalid event time"
            );
            mode.samples_us.push(elapsed_us);
            labels.push(mode.label);
            if mode.label == "baseline" {
                baseline_us = elapsed_us;
            } else {
                candidate_us = elapsed_us;
            }
        }
        pairs.push(
            json!({"sample":sample, "order":labels, "baseline_us":baseline_us,
            "candidate_us":candidate_us, "throughput_ratio":baseline_us / candidate_us}),
        );
    }
    stream.synchronize()?;
    for mode in &modes {
        let actual = stream.memcpy_dtov(mode.storage.last().context("output absent")?.as_ref())?;
        ensure!(
            Some(actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>()).as_ref()
                == canonical_bits.as_ref(),
            "{} changed output bits after timing",
            mode.label
        );
    }
    let results: Vec<_> = modes
        .iter()
        .map(|mode| {
            json!({
                "mode":mode.label, "effective_weight_sharing":mode.effective_sharing,
                "resources":mode.resources,
                "mean_us":mode.samples_us.iter().sum::<f64>() / args.samples as f64,
                "median_us":median(&mode.samples_us), "samples_us":mode.samples_us,
            })
        })
        .collect();
    println!(
        "{}",
        json!({
            "rows":args.rows, "keys":args.keys, "width":args.width,
            "samples":args.samples, "repeats":args.repeats, "initial_order":args.order,
            "warmup_ms":actual_warmup_ms, "warmup_pairs":warm_pairs,
            "device":device.name()?,
            "multiprocessor_count":device.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
            "protocol":"same resident read-only inputs; distinct preallocated outputs; compilation, uploads, validation and readback excluded; >=200ms alternating warmup after both captures; CUDA-event repeated-kernel capture timings; genuine alternating AB/BA pairs",
            "scope":"softmax(scores) @ values only; no QK score generation, cached decoder or complete-model claim; intentionally repeated/cache-hot buffers",
            "validation":{"all_finite_output_bits_equal_before_and_after_timing":true,
                "sampled_f64_reference_per_mode":64, "maximum_f64_reference_error":maximum_reference_error},
            "results":results, "pairs":pairs,
        })
    );
    Ok(())
}
