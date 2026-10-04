//! Attention-only resident CUDA-event and end-to-end profiling.
//! Usage: attention_profile [strict|cooperative|online] [sequence] [head_width] [causal|full].
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg, sys,
};
use cudarc::nvrtc::Ptx;
use num_traits::{Float, ToPrimitive, cast};
use tnsr::graph::{NodeIndex, TensorGraph, TensorGraphNode};
use tnsr::nn::softmax;
use tnsr::ptx::{
    PtxExecutor, PtxPlanAction, PtxReductionMode,
    types::{CudaDType, F32},
};
use tnsr::tensor::{Parameter, TensorExpr};
use tnsr::tile::MatMulPrecision;
use tnsr::{Executor, SimpleExecutor};

struct Launch {
    function: CudaFunction,
    config: LaunchConfig,
    pointers: Vec<u64>,
}

fn submit(stream: &Arc<CudaStream>, launches: &[Launch]) -> Result<()> {
    for launch in launches {
        let mut builder = stream.launch_builder(&launch.function);
        for pointer in &launch.pointers {
            builder.arg(pointer);
        }
        // SAFETY: Every pointer comes from a live, preallocated buffer. Inputs
        // were uploaded and synchronized before launch, all launches use this
        // stream in dependency order, and buffers stay alive until synchronization.
        unsafe { builder.launch(launch.config) }.context("resident attention launch failed")?;
    }
    Ok(())
}

fn percentile(values: &mut [f64], fraction: usize) -> f64 {
    values.sort_by(f64::total_cmp);
    values[(values.len() - 1) * fraction / 100]
}

fn main() -> Result<()> {
    profile::<F32>()
}
fn profile<D: CudaDType>() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mode = match args.first().map(String::as_str).unwrap_or("online") {
        "strict" => PtxReductionMode::Strict,
        "cooperative" => PtxReductionMode::DeterministicTree,
        "online" => PtxReductionMode::Online,
        other => anyhow::bail!("unknown reduction mode {other}"),
    };
    let number = |position: usize, default: usize| -> Result<usize> {
        args.get(position)
            .map(|s| s.parse().context("invalid dimension"))
            .unwrap_or(Ok(default))
    };
    let (m, k) = (number(1, 64)?, number(2, 32)?);
    anyhow::ensure!(m > 0 && k > 0, "dimensions must be nonzero");
    let causal = match args.get(3).map(String::as_str).unwrap_or("full") {
        "causal" => true,
        "full" => false,
        other => anyhow::bail!("unknown mask {other}"),
    };
    let scalar = |value: f32| -> Result<D::HostType> { cast(value).context("invalid scalar") };
    let tensor = |phase: usize| -> Result<TensorExpr<D::HostType>> {
        Ok(Parameter::new(
            (0..m * k)
                .map(|i| scalar(((i + phase) % 29) as f32 / 29.0 - 0.5))
                .collect::<Result<Vec<_>>>()?,
            vec![m, k],
        )
        .into())
    };
    let mut scores = tensor(0)?.matmul(tensor(3)?.transpose()) * scalar((k as f32).sqrt().recip())?;
    if causal {
        let mask = (0..m * m)
            .map(|i| {
                scalar(if i % m <= i / m {
                    0.0
                } else {
                    f32::NEG_INFINITY
                })
            })
            .collect::<Result<Vec<_>>>()?;
        scores = scores + TensorExpr::constant(mask, vec![m, m]);
    }
    let graph: TensorGraph<D::HostType> = softmax(scores, 1).matmul(tensor(7)?).into();
    let mut executor = PtxExecutor::<D>::try_new_with_reduction_mode(mode)?;
    executor.set_matmul_precision(MatMulPrecision::StrictF32);
    executor.compile_owned(graph.clone())?;
    let description = executor.describe_plan(&graph)?;
    let source = executor.module_source().context("missing PTX")?;
    let actual = executor.execute_compiled(&graph, HashMap::new())?;
    let plan = executor.execution_plan().context("missing plan")?;
    let element_bytes = std::mem::size_of::<D::HostType>();
    let trace = false;
    let device = CudaContext::new(0)?;
    let stream = device.default_stream();
    let module = device.load_module(Ptx::from_src(source))?;
    let mut storage: HashMap<NodeIndex, CudaSlice<D::HostType>> = HashMap::new();
    let mut aliases = HashMap::new();
    let names: Vec<Option<&str>> = description
        .lines()
        .skip(1)
        .map(|line| {
            line.split("action=kernel ")
                .nth(1)
                .and_then(|name| name.split_whitespace().next())
        })
        .collect();
    anyhow::ensure!(
        names.len() == plan.steps().len(),
        "plan description length mismatch"
    );
    let mut launches = Vec::new();
    for (step, name) in plan.steps().iter().zip(names) {
        if step.action == PtxPlanAction::Upload {
            let data = match &graph[step.node] {
                TensorGraphNode::Parameter { data, .. } => data
                    .lock()
                    .map_err(|_| anyhow::anyhow!("parameter lock poisoned"))?
                    .clone(),
                TensorGraphNode::Constant { data, .. } => data.as_ref().clone(),
                _ => anyhow::bail!("unsupported upload"),
            };
            storage.insert(step.node, stream.memcpy_stod(&data)?);
            continue;
        }
        if step.action == PtxPlanAction::VirtualView {
            for output in &step.virtual_outputs {
                aliases.insert(step.node, output.source);
            }
            continue;
        }
        let name = name.context("unsupported physical action in profiling graph")?;
        for (&output, shape) in step.outputs.iter().zip(&step.output_shapes) {
            storage.insert(
                output,
                stream.alloc_zeros::<D::HostType>(shape.iter().product())?,
            );
        }
        let pointer = |node: &NodeIndex| -> Result<u64> {
            let node = aliases.get(node).unwrap_or(node);
            let buffer = storage.get(node).context("resident buffer unavailable")?;
            Ok(buffer.device_ptr(&stream).0)
        };
        let inputs = step
            .inputs
            .iter()
            .map(pointer)
            .collect::<Result<Vec<_>>>()?;
        let outputs = step
            .outputs
            .iter()
            .map(pointer)
            .collect::<Result<Vec<_>>>()?;
        let function = module.load_function(name)?;
        if step.action == PtxPlanAction::Kernel
            && matches!(graph[step.node], TensorGraphNode::MatMul { .. })
        {
            let shape = &step.shape;
            let rows = shape[shape.len() - 2];
            let cols = shape[shape.len() - 1];
            let count: usize = shape[..shape.len() - 2].iter().product();
            let schedule = plan
                .matmul_schedules()
                .get(&step.node)
                .context("missing schedule")?;
            // This harness intentionally uses rank-two shared B for every case.
            for batch in 0..count {
                launches.push(Launch {
                    function: function.clone(),
                    config: LaunchConfig {
                        grid_dim: (
                            u32::try_from(cols.div_ceil(schedule.block_tile.n))?,
                            u32::try_from(rows.div_ceil(schedule.block_tile.m))?,
                            1,
                        ),
                        block_dim: schedule.block_threads,
                        shared_mem_bytes: 0,
                    },
                    pointers: vec![
                        inputs[0] + u64::try_from(batch * rows * k * element_bytes)?,
                        inputs[1],
                        outputs[0] + u64::try_from(batch * rows * cols * element_bytes)?,
                    ],
                });
            }
        } else {
            let config = if let PtxPlanAction::MatMulRegion(region_id) = step.action {
                let schedule = plan.matmul_regions()[region_id].schedule;
                let shape = &step.shape;
                LaunchConfig {
                    grid_dim: (
                        u32::try_from(shape[shape.len() - 1].div_ceil(schedule.block_tile.n))?,
                        u32::try_from(shape[shape.len() - 2].div_ceil(schedule.block_tile.m))?,
                        u32::try_from(shape[..shape.len() - 2].iter().product::<usize>())?,
                    ),
                    block_dim: schedule.block_threads,
                    shared_mem_bytes: 0,
                }
            } else if let PtxPlanAction::ReductionRegion(id) = step.action {
                match plan.reduction_regions()[id].schedule {
                    tnsr::tile::ReductionSchedule::Serial => LaunchConfig::for_num_elems(
                        u32::try_from(step.shape.iter().product::<usize>())?,
                    ),
                    tnsr::tile::ReductionSchedule::Subgroup { width } => LaunchConfig {
                        grid_dim: (u32::try_from(step.shape.iter().product::<usize>())?, 1, 1),
                        block_dim: (width, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    tnsr::tile::ReductionSchedule::Block { threads } => LaunchConfig {
                        grid_dim: (u32::try_from(step.shape.iter().product::<usize>())?, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    },
                }
            } else if let PtxPlanAction::OnlineRegion(id) = step.action {
                if let Some(threads) = plan.online_regions()[id].block_threads() {
                    LaunchConfig {
                        grid_dim: (
                            u32::try_from(step.shape.iter().product::<usize>() / k)?,
                            1,
                            1,
                        ),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    }
                } else {
                    LaunchConfig::for_num_elems(u32::try_from(
                        step.shape.iter().product::<usize>(),
                    )?)
                }
            } else {
                LaunchConfig::for_num_elems(u32::try_from(step.shape.iter().product::<usize>())?)
            };
            launches.push(Launch {
                function,
                config,
                pointers: inputs.into_iter().chain(outputs).collect(),
            });
        }
    }
    stream.synchronize()?;
    let repeats = if trace { 1 } else { 32 };
    for _ in 0..16 {
        submit(&stream, &launches)?;
    }
    stream.synchronize()?;
    let mut resident = Vec::new();
    for _ in 0..20 {
        let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        for _ in 0..repeats {
            submit(&stream, &launches)?;
        }
        let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        resident.push(f64::from(start.elapsed_ms(&end)?) * 1000.0 / f64::from(repeats));
    }
    let output = plan.graph_output().context("missing output")?;
    let resident_actual =
        stream.memcpy_dtov(storage.get(&output).context("missing output storage")?)?;
    let reference = if trace {
        actual.clone()
    } else {
        SimpleExecutor::<D::HostType>::new().execute(&graph, HashMap::new())?
    };
    let error = resident_actual.iter().zip(&reference).try_fold(
        0.0f32,
        |maximum, (&a, &b)| -> Result<f32> {
            Ok(maximum.max(
                (a.to_f32().context("invalid output")?
                    - b.to_f32().context("invalid reference")?)
                .abs(),
            ))
        },
    )?;
    let reference_scale = reference
        .iter()
        .try_fold(1.0f32, |maximum, value| -> Result<f32> {
            Ok(maximum.max(value.to_f32().context("invalid reference")?.abs()))
        })?;
    let tolerance = 1e-2
        + D::HostType::epsilon()
            .to_f32()
            .context("invalid dtype epsilon")?
            * reference_scale;
    anyhow::ensure!(
        error <= tolerance && resident_actual.len() == reference.len(),
        "resident numerical mismatch {error}"
    );
    let mut end_to_end = Vec::new();
    if !trace {
        for _ in 0..3 {
            executor.execute_compiled(&graph, HashMap::new())?;
        }
        for _ in 0..20 {
            let started = Instant::now();
            for _ in 0..5 {
                executor.execute_compiled(&graph, HashMap::new())?;
            }
            end_to_end.push(started.elapsed().as_secs_f64() * 1e6 / 5.0);
        }
    } else {
        end_to_end.push(0.0);
    }
    let compile = executor.compile_metrics();
    let execution = executor.execution_metrics();
    println!(
        "METRICS,{:.3},{:.3},{},{}",
        compile.compile_time.as_secs_f64() * 1e6,
        compile.index_optimization_time.as_secs_f64() * 1e6,
        execution.kernel_launches,
        execution.intermediate_materialized_bytes
    );
    println!(
        "RESULT,{mode:?},{m},{k},{causal},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{error}",
        launches.len(),
        percentile(&mut resident, 50),
        percentile(&mut resident, 10),
        percentile(&mut resident, 90),
        percentile(&mut end_to_end, 50),
        percentile(&mut end_to_end, 10),
        percentile(&mut end_to_end, 90)
    );
    Ok(())
}
