//! Focused matmul ablations. Run with CUDA enabled; all timings are microseconds.
//! Arguments: [batch|flat] [epilogue stage 0..3] [B] [M] [K] [N] [measure|trace].
//! Stages: matmul, +bias, +ReLU, +residual. Flat merges B and M for shared weights.
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg, sys,
};
use cudarc::nvrtc::Ptx;
use tnsr::graph::{NodeIndex, TensorGraph, TensorGraphNode};
use tnsr::ptx::{PtxExecutor, PtxPlanAction};
use tnsr::tensor::{Parameter, TensorExpr};
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
        unsafe { builder.launch(launch.config) }.context("resident matmul launch failed")?;
    }
    Ok(())
}

fn percentile(values: &mut [f64], fraction: usize) -> f64 {
    values.sort_by(f64::total_cmp);
    values[(values.len() - 1) * fraction / 100]
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flat = args.first().is_some_and(|argument| argument == "flat");
    let number = |position: usize, default: usize| -> Result<usize> {
        args.get(position)
            .map(|value| value.parse().context("invalid dimension/stage"))
            .unwrap_or(Ok(default))
    };
    let stage = number(1, 3)?;
    let (batches, m, k, n) = (
        number(2, 2)?,
        number(3, 128)?,
        number(4, 256)?,
        number(5, 768)?,
    );
    anyhow::ensure!(
        stage <= 3 && [batches, m, k, n].iter().all(|&extent| extent > 0),
        "invalid stage or zero dimension"
    );
    let trace = args.get(6).is_some_and(|mode| mode == "trace");
    let input_shape = if flat {
        vec![batches * m, k]
    } else {
        vec![batches, m, k]
    };
    let output_shape = if flat {
        vec![batches * m, n]
    } else {
        vec![batches, m, n]
    };
    let input = Parameter::new(
        (0..batches * m * k)
            .map(|index| (index % 29) as f32 / 29.0 - 0.5)
            .collect(),
        input_shape,
    );
    let weight = Parameter::new(
        (0..k * n)
            .map(|index| (index % 31) as f32 / 31.0 - 0.5)
            .collect(),
        vec![k, n],
    );
    let mut expression = TensorExpr::from(input).matmul(TensorExpr::from(weight));
    if stage >= 1 {
        expression = expression + TensorExpr::from(Parameter::new(vec![0.1; n], vec![n]));
    }
    if stage >= 2 {
        expression = expression.relu();
    }
    if stage >= 3 {
        expression = expression
            + TensorExpr::from(Parameter::new(vec![0.05; batches * m * n], output_shape));
    }
    let graph: TensorGraph<f32> = expression.into();
    let mut executor = PtxExecutor::try_new()?;
    executor.compile_owned(graph.clone())?;
    let description = executor.describe_plan(&graph)?;
    let source = executor.module_source().context("missing PTX")?;
    if let Ok(path) = std::env::var("TNSR_PROFILE_PTX") {
        std::fs::write(path, &source)?;
    }
    let actual = executor.execute_compiled(&graph, HashMap::new())?;
    let plan = executor
        .execution_plan()
        .context("missing execution plan")?;
    let device = CudaContext::new(0)?;
    let stream = device.default_stream();
    // Optional PTX-only experiment: resident timings use this module; executor
    // timings continue to measure the unmodified graph compilation.
    let resident_source = match std::env::var("TNSR_PROFILE_OVERRIDE_PTX") {
        Ok(path) => std::fs::read_to_string(path)?,
        Err(_) => source,
    };
    let module = device.load_module(Ptx::from_src(resident_source))?;
    let mut storage: HashMap<NodeIndex, CudaSlice<f32>> = HashMap::new();
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
            storage.insert(output, stream.alloc_zeros::<f32>(shape.iter().product())?);
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
            // This harness intentionally uses rank-two shared B for every case.
            for batch in 0..count {
                launches.push(Launch {
                    function: function.clone(),
                    config: LaunchConfig {
                        grid_dim: (
                            u32::try_from(cols.div_ceil(16))?,
                            u32::try_from(rows.div_ceil(16))?,
                            1,
                        ),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    },
                    pointers: vec![
                        inputs[0] + u64::try_from(batch * rows * k * 4)?,
                        inputs[1],
                        outputs[0] + u64::try_from(batch * rows * cols * 4)?,
                    ],
                });
            }
        } else {
            let config = if matches!(step.action, PtxPlanAction::MatMulRegion(_)) {
                let shape = &step.shape;
                LaunchConfig {
                    grid_dim: (
                        u32::try_from(shape[shape.len() - 1].div_ceil(16))?,
                        u32::try_from(shape[shape.len() - 2].div_ceil(16))?,
                        u32::try_from(shape[..shape.len() - 2].iter().product::<usize>())?,
                    ),
                    block_dim: (16, 16, 1),
                    shared_mem_bytes: 0,
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
        SimpleExecutor::new().execute(&graph, HashMap::new())?
    };
    let error = resident_actual
        .iter()
        .zip(&reference)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    anyhow::ensure!(
        error < 1e-2 && resident_actual.len() == reference.len(),
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
        "RESULT,{},{stage},{batches},{m},{k},{n},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{error}",
        if flat { "flat" } else { "batch" },
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
