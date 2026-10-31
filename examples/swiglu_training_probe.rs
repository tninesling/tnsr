//! Explicit stable SwiGLU VJP qualification. GPU execution requires --gpu.
use anyhow::{Result, ensure};
use clap::Parser;
use tnsr::tile::swiglu::SwiGluTraining;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 4096)]
    rows: usize,
    #[arg(long, default_value_t = 683)]
    width: usize,
    #[arg(long)]
    gpu: bool,
    /// Matched captured primitive-versus-fused samples; no full-model claim.
    #[arg(long, default_value_t = 20)]
    samples: usize,
    #[arg(long, default_value_t = 32)]
    repeats: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.samples >= 2 && args.repeats > 0,
        "at least two samples and positive repeats required"
    );
    let training = SwiGluTraining::new(vec![args.rows, args.width])?;
    println!(
        "{}",
        serde_json::json!({"shape":training.shape(), "elements":training.extent(), "forward_scalar_values":training.forward_scalar_operations(), "backward_scalar_values":training.backward_scalar_operations(), "forward_parameters":training.lower_forward()?.params.len(), "backward_parameters":training.lower_backward()?.params.len(), "automatic_selection":false})
    );
    if args.gpu {
        #[cfg(feature = "cuda")]
        return gpu(&args, &training);
        #[cfg(not(feature = "cuda"))]
        anyhow::bail!("--gpu requires --features cuda");
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn primitive_training_graph(
    source: &tnsr::graph::TensorGraph<f32, tnsr::graph::WithGrad>,
    loss: tnsr::graph::NodeIndex,
    ids: [usize; 2],
) -> Result<(tnsr::graph::TensorGraph<f32>, [tnsr::graph::NodeIndex; 3])> {
    use anyhow::Context;
    use std::collections::{HashMap, HashSet};
    use tnsr::{
        graph::{NodeIndex, TensorGraph, TensorGraphNode},
        tensor::BinaryOp,
    };
    let [weighted]: [NodeIndex; 1] = source
        .inputs(loss)
        .try_into()
        .map_err(|_| anyhow::anyhow!("primitive loss must reduce one weighted product"))?;
    let [value, incoming]: [NodeIndex; 2] = source
        .inputs(weighted)
        .try_into()
        .map_err(|_| anyhow::anyhow!("primitive weighted product must have two operands"))?;
    let metadata = source.gradient_metadata();
    let boundaries: Vec<_> = metadata
        .gradient_nodes
        .iter()
        .copied()
        .filter(|&node| {
            matches!(
                source[node],
                TensorGraphNode::Binary {
                    op: BinaryOp::Mul,
                    ..
                }
            ) && source.inputs(node).contains(&incoming)
        })
        .collect();
    ensure!(
        boundaries.len() == 1,
        "expected one incoming-gradient boundary, got {}",
        boundaries.len()
    );
    let boundary = boundaries[0];
    let roots = [
        value,
        *metadata
            .param_to_grad
            .get(&ids[0])
            .context("gate gradient absent")?,
        *metadata
            .param_to_grad
            .get(&ids[1])
            .context("up gradient absent")?,
    ];
    // Stop at seed*incoming and bind incoming directly. This removes only the
    // objective's VJP wrapper, not any of the activation derivative operations.
    let mut needed = HashSet::new();
    let mut pending = roots.to_vec();
    while let Some(node) = pending.pop() {
        if needed.insert(node) && node != boundary {
            pending.extend(source.inputs(node));
        }
    }
    ensure!(
        needed.contains(&boundary),
        "incoming boundary does not reach gradients"
    );
    let mut graph = TensorGraph::new();
    let mut remap = HashMap::new();
    for node in source.toposort().into_iter().filter(|n| needed.contains(n)) {
        let value = if node == boundary {
            TensorGraphNode::Input {
                name: "swiglu_incoming",
                shape: source[node].shape().clone(),
            }
        } else {
            source[node].clone()
        };
        let copied = graph.graph.add_node(value);
        remap.insert(node, copied);
        if node != boundary {
            for (position, input) in source.inputs(node).into_iter().enumerate() {
                graph.graph.add_edge(remap[&input], copied, position);
            }
        }
    }
    // The physical planner pins terminal outputs; no concat/copy is needed to
    // keep all three arrays observable. This graph contains no loss reduction.
    Ok((graph, roots.map(|n| remap[&n])))
}

#[cfg(feature = "cuda")]
fn gpu(args: &Args, training: &SwiGluTraining) -> Result<()> {
    use anyhow::Context;
    use cudarc::{
        driver::{
            CudaContext, CudaFunction, CudaGraph, CudaModule, CudaSlice, DevicePtr, LaunchConfig,
            PushKernelArg, sys,
        },
        nvrtc::Ptx,
    };
    use std::{collections::HashMap, sync::Arc};
    use tnsr::{
        Executor,
        graph::{NodeIndex, TensorGraph, TensorGraphNode},
        ptx::{
            PtxExecutor, PtxPlanAction,
            swiglu::{BACKWARD_KERNEL, FORWARD_KERNEL, module_source},
        },
        tensor::{Parameter, TensorExpr},
    };

    fn compare(actual: &[f32], expected: &[f32], gradient: bool) -> Result<()> {
        ensure!(
            actual.len() == expected.len(),
            "SwiGLU result length changed"
        );
        for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
            let tolerance = if gradient {
                3e-6 + 3e-3 * b.abs()
            } else {
                3e-5 + 5e-4 * b.abs()
            };
            ensure!(
                a.to_bits() == b.to_bits()
                    || a.is_nan() && b.is_nan()
                    || a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance,
                "SwiGLU {} mismatch at {i}: {a} versus {b}",
                if gradient { "gradient" } else { "forward" }
            );
        }
        Ok(())
    }
    struct Mode {
        // Destroy capture before the modules/storage it references.
        capture: CudaGraph,
        _module: Arc<CudaModule>,
        storage: Vec<CudaSlice<f32>>,
        outputs: [usize; 3],
        launches: usize,
        allocated_bytes: usize,
        samples_us: Vec<f64>,
    }
    let n = training.extent();
    let mut gate: Vec<f32> = (0..n).map(|i| ((i % 257) as f32 - 128.0) * 0.125).collect();
    let mut up: Vec<f32> = (0..n).map(|i| ((i % 31) as f32 - 15.0) * 0.125).collect();
    let mut incoming: Vec<f32> = (0..n).map(|i| ((i % 19) as f32 - 9.0) * 0.03125).collect();
    // Exercise underflow, signs, infinities and masked-branch NaNs with the
    // same primitive reference. These cases do not relax finite tolerances.
    for (i, &value) in [
        f32::NEG_INFINITY,
        -f32::MAX,
        -104.0,
        -90.0,
        -0.0,
        0.0,
        f32::MAX,
        f32::INFINITY,
        f32::NAN,
    ]
    .iter()
    .enumerate()
    .take(n)
    {
        gate[i] = value;
        up[i] = 1.25;
        incoming[i] = -0.7;
    }
    let gate_parameter = Parameter::new(gate.clone(), vec![n]);
    let up_parameter = Parameter::new(up.clone(), vec![n]);
    let ids = [gate_parameter.id(), up_parameter.id()];
    let value = tnsr::nn::silu(gate_parameter) * up_parameter;
    let forward_graph: TensorGraph<f32> = value.clone().into();
    let graph: TensorGraph<f32> = (value * TensorExpr::constant(incoming.clone(), vec![n]))
        .reduce_sum(0)
        .into();
    let loss = *graph.toposort().last().context("empty primitive graph")?;
    let graph = graph.with_gradients(loss);
    let (primitive, roots) = primitive_training_graph(&graph, loss, ids)?;
    let mut executor = PtxExecutor::try_new()?;
    let forward = executor.execute(&forward_graph, HashMap::new())?;
    executor.execute(&graph, HashMap::new())?;
    let gradients = executor.get_gradients(&graph);
    let expected = [&forward, &gradients[&ids[0]], &gradients[&ids[1]]];
    executor.compile(&primitive)?;
    let description = executor.describe_plan(&primitive)?;
    let plan = executor
        .execution_plan()
        .context("primitive physical plan absent")?;
    let descriptions: Vec<_> = description.lines().skip(1).collect();
    ensure!(
        descriptions.len() == plan.steps().len(),
        "physical description does not match plan"
    );
    let device = CudaContext::new(0)?;
    let stream = device.new_stream()?;
    let mut modes = Vec::new();

    // Ordinary physical-plan baseline, retaining generic pointwise fusion and
    // shared forward values. Preallocate all buffers, and ignore release_after:
    // allocator/liveness overhead is deliberately outside both timing scopes.
    let baseline_module = device.load_module(Ptx::from_src(
        executor
            .module_source()
            .context("primitive PTX unavailable")?,
    ))?;
    let mut storage = Vec::new();
    let mut locations: HashMap<NodeIndex, usize> = HashMap::new();
    let mut launches: Vec<(CudaFunction, LaunchConfig, Vec<u64>)> = Vec::new();
    let mut allocated_bytes = 0usize;
    for (step, description) in plan.steps().iter().zip(descriptions) {
        match step.action {
            PtxPlanAction::Upload => {
                ensure!(step.outputs.len() == 1, "upload must have one output");
                let node = step.outputs[0];
                let data = match &primitive[node] {
                    TensorGraphNode::Constant { data, .. } => data.as_ref().clone(),
                    TensorGraphNode::Parameter { data, .. } => data
                        .lock()
                        .map_err(|_| anyhow::anyhow!("primitive parameter lock poisoned"))?
                        .clone(),
                    TensorGraphNode::Input {
                        name: "swiglu_incoming",
                        ..
                    } => incoming.clone(),
                    _ => anyhow::bail!("unexpected primitive upload {}", primitive[node].name()),
                };
                allocated_bytes += data.len() * 4;
                locations.insert(node, storage.len());
                storage.push(stream.memcpy_stod(&data)?);
            }
            PtxPlanAction::VirtualView => {
                ensure!(step.inputs.len() == 1, "view must have one input");
                let source = *locations
                    .get(&step.inputs[0])
                    .context("view source is unavailable")?;
                for &output in &step.outputs {
                    locations.insert(output, source);
                }
            }
            PtxPlanAction::Kernel | PtxPlanAction::PointwiseRegion(_) => {
                for (&output, shape) in step.outputs.iter().zip(&step.output_shapes) {
                    let count = shape.iter().product::<usize>();
                    allocated_bytes += count * 4;
                    locations.insert(output, storage.len());
                    storage.push(stream.alloc_zeros::<f32>(count)?);
                }
                let name = description
                    .split("action=kernel ")
                    .nth(1)
                    .and_then(|s| s.split_whitespace().next())
                    .context("primitive kernel name absent")?;
                let pointers = step
                    .inputs
                    .iter()
                    .chain(&step.outputs)
                    .map(|node| {
                        locations
                            .get(node)
                            .copied()
                            .context("primitive binding unavailable")
                            .map(|i| storage[i].device_ptr(&stream).0)
                    })
                    .collect::<Result<Vec<u64>>>()?;
                let extent = step.shape.iter().product::<usize>();
                launches.push((
                    baseline_module.load_function(name)?,
                    LaunchConfig {
                        grid_dim: (u32::try_from(extent.div_ceil(256))?, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    pointers,
                ));
            }
            other => anyhow::bail!(
                "unsupported activation-only physical action {other:?}; refusing an unequal baseline"
            ),
        }
    }
    let outputs = roots.map(|node| locations[&node]);
    let submit = || -> Result<()> {
        for (function, config, pointers) in &launches {
            let mut launch = stream.launch_builder(function);
            for pointer in pointers {
                launch.arg(pointer);
            }
            // SAFETY: physical-plan bindings refer to the live, preallocated
            // f32 storage above; all selected kernels are bounded pointwise.
            unsafe { launch.launch(*config) }?;
        }
        Ok(())
    };
    submit()?;
    stream.synchronize()?;
    for i in 0..3 {
        compare(
            &stream.memcpy_dtov(&storage[outputs[i]])?,
            expected[i],
            i > 0,
        )?;
    }
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let recorded = (|| {
        for _ in 0..args.repeats {
            submit()?;
        }
        Ok::<_, anyhow::Error>(())
    })();
    let captured = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    recorded?;
    let capture = captured?.context("empty primitive capture")?;
    modes.push(Mode {
        capture,
        _module: baseline_module,
        storage,
        outputs,
        launches: launches.len(),
        allocated_bytes,
        samples_us: Vec::new(),
    });

    let fused_module = device.load_module(Ptx::from_src(module_source(training)?))?;
    let forward_kernel = fused_module.load_function(FORWARD_KERNEL)?;
    let backward_kernel = fused_module.load_function(BACKWARD_KERNEL)?;
    let fused_resources = serde_json::json!({"forward_registers":forward_kernel.num_regs()?, "backward_registers":backward_kernel.num_regs()?, "forward_local_bytes":forward_kernel.local_size_bytes()?, "backward_local_bytes":backward_kernel.local_size_bytes()?});
    let storage = vec![
        stream.memcpy_stod(&gate)?,
        stream.memcpy_stod(&up)?,
        stream.memcpy_stod(&incoming)?,
        stream.alloc_zeros::<f32>(n)?,
        stream.alloc_zeros::<f32>(n)?,
        stream.alloc_zeros::<f32>(n)?,
    ];
    let pointers: Vec<u64> = storage.iter().map(|s| s.device_ptr(&stream).0).collect();
    let config = LaunchConfig {
        grid_dim: (u32::try_from(n.div_ceil(256))?, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let submit = || -> Result<()> {
        let mut forward = stream.launch_builder(&forward_kernel);
        for &i in &[0, 1, 3] {
            forward.arg(&pointers[i]);
        }
        let mut backward = stream.launch_builder(&backward_kernel);
        for &i in &[0, 1, 2, 4, 5] {
            backward.arg(&pointers[i]);
        }
        // SAFETY: bindings are live, disjoint contiguous f32 arrays of n
        // elements; generated bounds checks guard the tail of both kernels.
        unsafe {
            forward.launch(config)?;
            backward.launch(config)?;
        }
        Ok(())
    };
    submit()?;
    stream.synchronize()?;
    for i in 0..3 {
        compare(&stream.memcpy_dtov(&storage[3 + i])?, expected[i], i > 0)?;
    }
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let recorded = (|| {
        for _ in 0..args.repeats {
            submit()?;
        }
        Ok::<_, anyhow::Error>(())
    })();
    let captured = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    recorded?;
    let capture = captured?.context("empty fused capture")?;
    modes.push(Mode {
        capture,
        _module: fused_module,
        storage,
        outputs: [3, 4, 5],
        launches: 2,
        allocated_bytes: 6 * n * 4,
        samples_us: Vec::new(),
    });
    for _ in 0..5 {
        for mode in &modes {
            mode.capture.launch()?;
        }
    }
    stream.synchronize()?;
    for sample in 0..args.samples {
        let order = if sample % 2 == 0 { [0, 1] } else { [1, 0] };
        for i in order {
            let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            modes[i].capture.launch()?;
            let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            modes[i]
                .samples_us
                .push(f64::from(start.elapsed_ms(&end)?) * 1000.0 / args.repeats as f64);
        }
    }
    stream.synchronize()?;
    // Re-qualify the final captured results, outside every timed interval.
    for mode in &modes {
        for (i, reference) in expected.iter().enumerate() {
            compare(
                &stream.memcpy_dtov(&mode.storage[mode.outputs[i]])?,
                reference,
                i > 0,
            )?;
        }
    }
    let median = |values: &[f64]| {
        let mut v = values.to_vec();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let baseline_total: f64 = modes[0].samples_us.iter().sum();
    let candidate_total: f64 = modes[1].samples_us.iter().sum();
    let results: Vec<_> = modes.iter().enumerate().map(|(i,m)| serde_json::json!({"variant":if i==0 {"ordinary_primitive_fusion"} else {"explicit_stable_swiglu_training"}, "launches_per_pair":m.launches,"preallocated_bytes":m.allocated_bytes,"median_us":median(&m.samples_us),"samples_us":m.samples_us})).collect();
    println!(
        "{}",
        serde_json::json!({"qualified":true,"reference":"primitive GPU stable SiLU autograd, all3 outputs, NaN classes and original finite gates", "protocol":"alternating variant order; resident storage; CUDA-graph repetitions; CUDA events; uploads/readback excluded; objective wrapper removed", "repeats":args.repeats,"results":results,"captured_activation_throughput_ratio":baseline_total/candidate_total,"fused_resources":fused_resources,"full_model_speedup":null})
    );
    Ok(())
}
