//! Balanced resident GEMM grid-order comparisons; uploads/readback excluded.
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
use tnsr::{
    graph::{TensorGraph, TensorGraphNode},
    ptx::{
        PtxExecutor, PtxPlanAction,
        types::{BF16, CudaDType, F16, F32},
    },
    tensor::{Parameter, TensorExpr},
    tile::{MatMulPrecision, MatMulSchedule},
};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 2048)]
    m: usize,
    #[arg(long, default_value_t = 2048)]
    n: usize,
    #[arg(long, default_value_t = 512)]
    k: usize,
    #[arg(long, default_value = "bf16")]
    dtype: String,
    #[arg(long, default_value = "native")]
    precision: String,
    #[arg(long, default_value = "1,4,8,16")]
    groups: String,
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
    /// Save complete group-1 output bits for comparisons between compiler builds.
    #[arg(long)]
    output_bits: Option<std::path::PathBuf>,
}

struct Mode<D: CudaDType> {
    // Graph goes first so it is destroyed while its module and storage are live.
    graph: CudaGraph,
    _module: Arc<CudaModule>,
    _function: CudaFunction,
    _storage: Vec<CudaSlice<D::HostType>>,
    group: usize,
    effective_group: usize,
    resources: serde_json::Value,
    samples_us: Vec<f64>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.samples >= 2 && args.repeats > 0 && [args.m, args.n, args.k].iter().all(|&d| d > 0),
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
    let groups: Vec<usize> = args
        .groups
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    ensure!(
        groups.first() == Some(&1),
        "first group must be the row-major baseline 1"
    );
    for &group in &groups {
        MatMulSchedule::validate_grid_group_m(group)?;
    }
    let values = |count, modulus| -> Result<Vec<D::HostType>> {
        (0..count)
            .map(|i| {
                cast(((i * 13 % modulus) as f32 - (modulus / 2) as f32) / modulus as f32)
                    .context("input conversion failed")
            })
            .collect()
    };
    let a = values(args.m * args.k, 97)?;
    let b = values(args.k * args.n, 89)?;
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
    for group in groups {
        let mut executor = PtxExecutor::<D>::try_new_for_dtype()?;
        executor.set_matmul_precision(match args.precision.as_str() {
            "strict" => MatMulPrecision::StrictF32,
            "native" => MatMulPrecision::NativeHalf,
            "tf32x3" => MatMulPrecision::Tf32x3,
            other => anyhow::bail!("unsupported precision {other}"),
        });
        executor.set_matmul_grid_group_m(group)?;
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
        storage.push(stream.alloc_zeros::<D::HostType>(args.m * args.n)?);
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
        let resources = json!({
            "block_tile": [schedule.block_tile.m, schedule.block_tile.n, schedule.block_tile.k],
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
        if group == 1 {
            for sample in 0..32 {
                let row = sample * 7919 % args.m;
                let col = sample * 104729 % args.n;
                let expected = (0..args.k).try_fold(0.0, |sum, j| -> Result<f64> {
                    Ok(sum
                        + a[row * args.k + j]
                            .to_f64()
                            .context("reference conversion failed")?
                            * b[j * args.n + col]
                                .to_f64()
                                .context("reference conversion failed")?)
                })?;
                let value = f64::from(actual_f32[row * args.n + col]);
                ensure!(
                    (value - expected).abs() <= 0.01 * (1.0 + expected.abs()),
                    "sample mismatch at {row},{col}: {value} != {expected}"
                );
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
                "grid traversal changed output bits, group={group}"
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
        for _ in 0..5 {
            captured.launch()?;
        }
        stream.synchronize()?;
        modes.push(Mode::<D> {
            graph: captured,
            _module: module,
            _function: function,
            _storage: storage,
            group,
            effective_group: schedule.effective_grid_group_m(),
            resources,
            samples_us: Vec::new(),
        });
    }
    for sample in 0..args.samples {
        let mut order: Vec<_> = (0..modes.len()).collect();
        order.rotate_left(sample % modes.len());
        if sample % 2 == 1 {
            order.reverse();
        }
        for index in order {
            let mode = &mut modes[index];
            let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            mode.graph.launch()?;
            let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            mode.samples_us
                .push(f64::from(start.elapsed_ms(&end)?) * 1000.0 / args.repeats as f64);
        }
    }
    stream.synchronize()?;
    let summary: Vec<_> = modes.iter().map(|mode| {
        let mean=mode.samples_us.iter().sum::<f64>()/args.samples as f64;
        let variance=mode.samples_us.iter().map(|x|(x-mean).powi(2)).sum::<f64>()/(args.samples-1) as f64;
        let ci=2.4*(variance/args.samples as f64).sqrt();
            json!({"group_m":mode.group,"effective_group_m":mode.effective_group,"resources":mode.resources,"mean_us":mean,"ci95_us":[(mean-ci).max(0.0),mean+ci],"samples_us":mode.samples_us})
    }).collect();
    println!(
        "{}",
        json!({"m":args.m,"n":args.n,"k":args.k,"transpose_a":args.transpose_a,"transpose_b":args.transpose_b,"reshape_batch":args.reshape_batch,"dtype":args.dtype,"precision":args.precision,"repeats":args.repeats,"protocol":"balanced rotated/reversed candidate order; resident preallocated storage; CUDA graph repetitions; CUDA events; uploads/readback excluded; exact output bits compared with group1","results":summary})
    );
    Ok(())
}
