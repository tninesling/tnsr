//! Same-input, address-balanced component screen. Never a complete update.
use super::*;
use cudarc::driver::CudaGraph;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
const REPEATS: usize = 32;
const BLOCKS: usize = 48;
const MEMBERS: usize = 4;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Separate4,
    Group4,
}
impl Mode {
    fn index(self) -> usize {
        match self {
            Self::Separate4 => 0,
            Self::Group4 => 1,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Separate4 => "separate4",
            Self::Group4 => "group4",
        }
    }
}
const MODES: [Mode; 2] = [Mode::Separate4, Mode::Group4];
fn orders(block: usize, reverse: bool) -> [[Mode; 2]; 2] {
    let first = if ((block / 2) % 2 == 1) ^ reverse {
        [Mode::Group4, Mode::Separate4]
    } else {
        MODES
    };
    [first, [first[1], first[0]]]
}
fn slot(mode: Mode, block: usize) -> usize {
    (mode.index() + block % 2) % 2
}
fn construction(reverse: bool) -> Vec<(Mode, usize)> {
    let mut bindings = vec![
        (Mode::Separate4, 0),
        (Mode::Group4, 0),
        (Mode::Separate4, 1),
        (Mode::Group4, 1),
    ];
    if reverse {
        bindings.reverse();
    }
    bindings
}
fn output(member: &Member, slot: usize) -> &Outputs {
    if slot == 0 { &member.old } else { &member.new }
}
fn output_mut(member: &mut Member, slot: usize) -> &mut Outputs {
    if slot == 0 {
        &mut member.old
    } else {
        &mut member.new
    }
}
fn allocated_member(stream: &Arc<CudaStream>, len: usize, reverse: bool) -> Result<Member> {
    let input = Inputs {
        weight: stream.alloc_zeros(len + GUARD)?,
        gradient: stream.alloc_zeros(len + GUARD)?,
        moment: stream.alloc_zeros(len + GUARD)?,
        variance: stream.alloc_zeros(len + GUARD)?,
        settings: stream.alloc_zeros(10 + GUARD)?,
    };
    let first = outputs(stream, len)?;
    let second = outputs(stream, len)?;
    let (old, new) = if reverse {
        (second, first)
    } else {
        (first, second)
    };
    Ok(Member { input, old, new })
}
fn submit(
    stream: &Arc<CudaStream>,
    functions: &[CudaFunction],
    members: &mut [Member],
    len: usize,
    mode: Mode,
    slot: usize,
) -> Result<()> {
    let function = &functions[mode.index()];
    let config = LaunchConfig {
        grid_dim: (
            u32::try_from((len / 1024) * if mode == Mode::Group4 { MEMBERS } else { 1 })?,
            1,
            1,
        ),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    if mode == Mode::Separate4 {
        for member in members {
            let input = &member.input;
            let target = if slot == 0 {
                &mut member.old
            } else {
                &mut member.new
            };
            let mut launch = stream.launch_builder(function);
            launch
                .arg(&input.weight)
                .arg(&input.gradient)
                .arg(&input.moment)
                .arg(&input.variance)
                .arg(&input.settings)
                .arg(&mut target.weight)
                .arg(&mut target.moment)
                .arg(&mut target.variance)
                .arg(&mut target.health)
                .arg(&mut target.shadow);
            // SAFETY: source-qualified canonical10-pointer ABI, full typed owned
            // spans and a whole-CTA extent. Inputs/outputs stay alive throughout.
            unsafe { launch.launch(config) }?;
        }
    } else {
        let mut launch = stream.launch_builder(function);
        for member in members {
            let input = &member.input;
            let target = if slot == 0 {
                &mut member.old
            } else {
                &mut member.new
            };
            launch
                .arg(&input.weight)
                .arg(&input.gradient)
                .arg(&input.moment)
                .arg(&input.variance)
                .arg(&input.settings)
                .arg(&mut target.weight)
                .arg(&mut target.moment)
                .arg(&mut target.variance)
                .arg(&mut target.health)
                .arg(&mut target.shadow);
        }
        // SAFETY: same fresh typed spans in canonical40-pointer order; the
        // CPU-proved uniform CTA regions each address one member's local extent.
        unsafe { launch.launch(config) }?;
    }
    Ok(())
}
fn pointers(member: &Member, slot: usize, stream: &Arc<CudaStream>) -> Vec<u64> {
    let input = &member.input;
    let target = output(member, slot);
    vec![
        input.weight.device_ptr(stream).0,
        input.gradient.device_ptr(stream).0,
        input.moment.device_ptr(stream).0,
        input.variance.device_ptr(stream).0,
        input.settings.device_ptr(stream).0,
        target.weight.device_ptr(stream).0,
        target.moment.device_ptr(stream).0,
        target.variance.device_ptr(stream).0,
        target.health.device_ptr(stream).0,
        target.shadow.device_ptr(stream).0,
    ]
}
fn logical_spans(
    member: &Member,
    slot: usize,
    len: usize,
    stream: &Arc<CudaStream>,
) -> Result<[Span; 10]> {
    let p = pointers(member, slot, stream);
    let bytes = [
        len * 4,
        len * 2,
        len * 4,
        len * 4,
        40,
        len * 4,
        len * 4,
        len * 4,
        len / 64,
        len * 2,
    ];
    Ok(std::array::from_fn(|i| Span {
        address: usize::try_from(p[i]).unwrap_or(0),
        bytes: bytes[i],
    }))
}
struct Binding {
    mode: Mode,
    slot: usize,
    graph: CudaGraph,
    pointers: Vec<u64>,
}
fn graph(bindings: &[Binding], mode: Mode, slot: usize) -> Result<&Binding> {
    bindings
        .iter()
        .find(|b| b.mode == mode && b.slot == slot)
        .context("balanced capture binding missing")
}
struct Snapshot {
    weight: Vec<f32>,
    moment: Vec<f32>,
    variance: Vec<f32>,
    health: Vec<f32>,
    shadow: Vec<bf16>,
}
fn snapshot(stream: &Arc<CudaStream>, source: &Outputs) -> Result<Snapshot> {
    Ok(Snapshot {
        weight: download(stream, &source.weight)?,
        moment: download(stream, &source.moment)?,
        variance: download(stream, &source.variance)?,
        health: download(stream, &source.health)?,
        shadow: download(stream, &source.shadow)?,
    })
}
fn f32_extent(values: &[f32]) -> Value {
    json!({"elements":values.len(),"dtype":"f32"})
}
fn bf16_extent(values: &[bf16]) -> Value {
    json!({"elements":values.len(),"dtype":"bf16"})
}
fn compare_snapshot(actual: &Snapshot, expected: &Snapshot, len: usize) -> Result<Value> {
    let mut proofs = Vec::new();
    for (name, a, b, n) in [
        ("weight", &actual.weight, &expected.weight, len),
        ("moment", &actual.moment, &expected.moment, len),
        ("variance", &actual.variance, &expected.variance, len),
        ("health", &actual.health, &expected.health, len / 256),
    ] {
        ensure!(
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|(a, b)| a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan()),
            "component full F32output mismatch: {name}"
        );
        ensure!(
            a[n..].iter().all(|v| v.to_bits() == f_sentinel().to_bits()),
            "component F32canary changed: {name}"
        );
        proofs.push(json!({"name":name,"full_nonNaN_bits_NaN_classes_equal":true,"guard_elements":GUARD,"actual":f32_extent(a),"reference":f32_extent(b)}));
    }
    ensure!(
        actual.shadow.len() == expected.shadow.len()
            && actual
                .shadow
                .iter()
                .zip(&expected.shadow)
                .all(|(a, b)| a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan()),
        "component full BF16shadow mismatch"
    );
    ensure!(
        actual.shadow[len..]
            .iter()
            .all(|v| v.to_bits() == b_sentinel().to_bits()),
        "component BF16canary changed"
    );
    proofs.push(
        json!({"name":"shadow","full_nonNaN_bits_NaN_classes_equal":true,"guard_elements":GUARD,
        "actual":bf16_extent(&actual.shadow),"reference":bf16_extent(&expected.shadow)}),
    );
    Ok(json!(proofs))
}
fn verify_outputs(
    stream: &Arc<CudaStream>,
    members: &mut [Member],
    bindings: &[Binding],
    oracle: &[Snapshot],
    len: usize,
) -> Result<Value> {
    let mut proofs = Vec::new();
    for binding in bindings {
        for member in members.iter_mut() {
            poison(stream, output_mut(member, binding.slot), len)?;
        }
        binding.graph.launch()?;
        stream.synchronize()?;
        let outputs = members
            .iter()
            .zip(oracle)
            .enumerate()
            .map(|(member, (actual, reference))| -> Result<Value> {
                let actual = snapshot(stream, output(actual, binding.slot))?;
                Ok(json!({"member":member,"arrays":compare_snapshot(&actual,reference,len)?}))
            })
            .collect::<Result<Vec<_>>>()?;
        proofs.push(json!({"mode":binding.mode.name(),"slot":binding.slot,"poisoned_before_launch":true,"members":outputs}));
    }
    Ok(json!(proofs))
}
fn observe_final_outputs(
    stream: &Arc<CudaStream>,
    members: &[Member],
    oracle: &[Snapshot],
    len: usize,
) -> Result<Value> {
    let mut slots = Vec::new();
    for slot in 0..2 {
        let proofs = members
            .iter()
            .zip(oracle)
            .enumerate()
            .map(|(member, (actual, reference))| -> Result<Value> {
                let actual = snapshot(stream, output(actual, slot))?;
                Ok(json!({"member":member,"arrays":compare_snapshot(&actual,reference,len)?}))
            })
            .collect::<Result<Vec<_>>>()?;
        slots.push(
            json!({"slot":slot,"observed_before_post_poison_or_relaunch":true,"members":proofs}),
        );
    }
    Ok(json!(slots))
}
fn verify_readonly(stream: &Arc<CudaStream>, members: &[Member], feeds: &[Feed]) -> Result<Value> {
    let mut proofs = Vec::new();
    for (index, (member, feed)) in members.iter().zip(feeds).enumerate() {
        let input = &member.input;
        let mut records = Vec::new();
        for (name, actual, expected) in [
            ("weight", &input.weight, &feed.weight),
            ("moment", &input.moment, &feed.moment),
            ("variance", &input.variance, &feed.variance),
            ("settings", &input.settings, &feed.settings),
        ] {
            let actual = download(stream, actual)?;
            ensure!(
                actual.len() == expected.len()
                    && actual
                        .iter()
                        .zip(expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                "component readonly F32input changed: {name}"
            );
            records.push(json!({"name":name,"all_bits_equal":true,"actual":f32_extent(&actual),"expected":f32_extent(expected)}));
        }
        let actual = download(stream, &input.gradient)?;
        ensure!(
            actual.len() == feed.gradient.len()
                && actual
                    .iter()
                    .zip(&feed.gradient)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
            "component readonly BF16input changed"
        );
        records.push(json!({"name":"gradient","all_bits_equal":true,"actual":bf16_extent(&actual),"expected":bf16_extent(&feed.gradient)}));
        proofs.push(json!({"member":index,"arrays":records}));
    }
    Ok(json!(proofs))
}
fn owned_bytes(members: &[Member]) -> usize {
    members
        .iter()
        .map(|member| {
            let input = &member.input;
            let outputs = [&member.old, &member.new]
                .into_iter()
                .map(|o| {
                    (o.weight.len() + o.moment.len() + o.variance.len() + o.health.len()) * 4
                        + o.shadow.len() * 2
                })
                .sum::<usize>();
            (input.weight.len() + input.moment.len() + input.variance.len() + input.settings.len())
                * 4
                + input.gradient.len() * 2
                + outputs
        })
        .sum()
}
fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    (values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.
}
fn case(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    root: &Path,
    qualified: &Path,
    len: usize,
    reverse: bool,
    timing: bool,
) -> Result<Value> {
    let directory = root.join(format!("len{len}"));
    ensure!(!directory.exists(), "fresh component case output required");
    std::fs::create_dir_all(&directory)?;
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    for (region_id, ir) in [paired_ir(len, AdamKind::Vector4)?, ir(len, MEMBERS)?]
        .into_iter()
        .enumerate()
    {
        tiles
            .region_kernels
            .push(crate::tile::graph::TileRegionKernel { region_id, ir });
    }
    tiles.optimize_indices()?;
    let graph_ir = PtxGraph::from(tiles).with_target(PtxTarget::from_context(context)?);
    let source = graph_ir.to_ptx();
    let qualified_ptx = qualified.join(format!("len{len}-members4.ptx"));
    ensure!(
        std::fs::read(&qualified_ptx)? == source.as_bytes(),
        "component PTX differs from qualified paired-vector math module"
    );
    std::fs::write(directory.join("loaded-module.ptx"), &source)?;
    let names = (0..2)
        .map(|index| {
            graph_ir
                .region_kernel_name(index)
                .map(str::to_owned)
                .context("component entry missing")
        })
        .collect::<Result<Vec<_>>>()?;
    let module = context.load_module(Ptx::from_src(source))?;
    let functions = names
        .iter()
        .map(|n| module.load_function(n).map_err(anyhow::Error::from))
        .collect::<Result<Vec<_>>>()?;
    let resources=functions.iter().zip(&names).map(|(f,name)|->Result<Value>{Ok(json!({"name":name,"registers":f.num_regs()?,
        "shared_bytes":f.shared_size_bytes()?,"local_bytes":f.local_size_bytes()?,"max_threads":f.max_threads_per_block()?,
        "capacity256":f.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?}))}).collect::<Result<Vec<_>>>()?;
    std::fs::write(
        directory.join("resources.json"),
        serde_json::to_vec_pretty(&resources)?,
    )?;
    ensure!(
        functions
            .iter()
            .all(|f| f.local_size_bytes().ok() == Some(0))
            && functions[1].shared_size_bytes()? == 32
            && functions[1].num_regs()? <= 128
            && functions[1].max_threads_per_block()? >= 256
            && functions[1].occupancy_max_active_blocks_per_multiprocessor(256, 0, None)? >= 2,
        "qualified component resource policy changed"
    );
    let member_order = if reverse {
        vec![3usize, 2, 1, 0]
    } else {
        vec![0, 1, 2, 3]
    };
    let mut allocated = (0..MEMBERS).map(|_| None).collect::<Vec<Option<Member>>>();
    for &member in &member_order {
        allocated[member] = Some(allocated_member(stream, len, reverse)?);
    }
    let mut members = allocated
        .into_iter()
        .map(|m| m.context("component member allocation missing"))
        .collect::<Result<Vec<_>>>()?;
    let feeds = (0..MEMBERS)
        .map(|member| feed(len, member, 0))
        .collect::<Vec<_>>();
    for (member, feed) in members.iter_mut().zip(&feeds) {
        upload(stream, &mut member.input, feed)?;
    }
    for slot in 0..2 {
        let spans = members
            .iter()
            .map(|m| logical_spans(m, slot, len, stream))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            admitted(len, &spans),
            "component current slot spans fail full typed alignment/disjointness proof"
        );
    }
    let construction = construction(reverse);
    let mut bindings = Vec::new();
    for &(mode, slot) in &construction {
        stream.synchronize()?;
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let submitted = (|| -> Result<()> {
            for _ in 0..REPEATS {
                submit(stream, &functions, &mut members, len, mode, slot)?;
            }
            Ok(())
        })();
        let captured = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        submitted?;
        let graph = captured?.context("empty repeated paired-vector component capture")?;
        let pointers = members
            .iter()
            .flat_map(|m| pointers(m, slot, stream))
            .collect::<Vec<_>>();
        bindings.push(Binding {
            mode,
            slot,
            graph,
            pointers,
        });
    }
    for member in &mut members {
        poison(stream, output_mut(member, 0), len)?;
    }
    graph(&bindings, Mode::Separate4, 0)?.graph.launch()?;
    stream.synchronize()?;
    let oracle = members
        .iter()
        .map(|m| snapshot(stream, output(m, 0)))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        oracle.iter().all(|o| o.weight[..len]
            .iter()
            .chain(&o.moment[..len])
            .chain(&o.variance[..len])
            .all(|v| v.is_finite())
            && o.health[..len / 256].iter().all(|v| v.to_bits() == 0)
            && o.shadow[..len].iter().all(|v| v.is_finite())),
        "component finite oracle invalid"
    );
    let pre_outputs = verify_outputs(stream, &mut members, &bindings, &oracle, len)?;
    let pre_inputs = verify_readonly(stream, &members, &feeds)?;
    let mut warmup_cycles = 0usize;
    let mut warmup_seconds = 0.;
    let mut blocks = Vec::new();
    if timing {
        let begin = context.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let end = context.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let started = Instant::now();
        while started.elapsed() < Duration::from_secs(1) {
            for block in 0..4 {
                for order in orders(block, reverse) {
                    for mode in order {
                        graph(&bindings, mode, slot(mode, block))?.graph.launch()?;
                    }
                }
            }
            stream.synchronize()?;
            warmup_cycles += 1;
        }
        warmup_seconds = started.elapsed().as_secs_f64();
        for block in 0..BLOCKS {
            let mut samples = Vec::new();
            for (order_index, order) in orders(block, reverse).into_iter().enumerate() {
                for (position, mode) in order.into_iter().enumerate() {
                    let slot = slot(mode, block);
                    let binding = graph(&bindings, mode, slot)?;
                    // The only stream work between these pre-created events is
                    // one replay of the already-captured immutable-input graph.
                    begin.record(stream)?;
                    binding.graph.launch()?;
                    end.record(stream)?;
                    end.synchronize()?;
                    let event_ms = f64::from(begin.elapsed_ms(&end)?);
                    let us = event_ms * 1000. / REPEATS as f64;
                    ensure!(
                        us.is_finite() && us > 0.,
                        "invalid paired-vector component event interval"
                    );
                    samples.push(json!({"mode":mode.name(),"slot":slot,"order_index":order_index,"position":position,
                        "cuda_event_ms":event_ms,"microseconds_per_group":us,"flat40pointers":binding.pointers}));
                }
            }
            blocks.push(json!({"block":block,"orders":orders(block,reverse).map(|o|o.map(Mode::name)),
                "mode_to_slot":{"separate4":slot(Mode::Separate4,block),"group4":slot(Mode::Group4,block)},"samples":samples}));
        }
    }
    stream.synchronize()?;
    // Inspect the outputs left by the actual final measured graph replays before
    // any post-qualification poison/relaunch could overwrite that evidence.
    let final_outputs = observe_final_outputs(stream, &members, &oracle, len)?;
    let post_inputs = verify_readonly(stream, &members, &feeds)?;
    let post_outputs = verify_outputs(stream, &mut members, &bindings, &oracle, len)?;
    let mut summaries = Vec::new();
    if timing {
        for mode in MODES {
            let values = blocks
                .iter()
                .flat_map(|b| b["samples"].as_array().into_iter().flatten())
                .filter(|s| s["mode"] == mode.name())
                .filter_map(|s| s["microseconds_per_group"].as_f64())
                .collect::<Vec<_>>();
            ensure!(values.len() == 96, "component mode sample count changed");
            summaries.push(json!({"mode":mode.name(),"samples":values.len(),"median_us_per_group":median(&values),"samples_us_per_group":values}));
        }
    }
    let binding_metadata=bindings.iter().map(|b|json!({"mode":b.mode.name(),"slot":b.slot,"entry":names[b.mode.index()],
        "kernel_launches_per_group":if b.mode==Mode::Separate4 {4}else{1},"captured_kernel_nodes":REPEATS*if b.mode==Mode::Separate4 {4}else{1},
        "resources":resources[b.mode.index()],"flat40pointers":b.pointers,
        "typed_ABI_per_member":["f32","bf16","f32","f32","f32","f32","f32","f32","f32","bf16"],
        "logical_elements_per_member":[len,len,len,len,10,len,len,len,len/256,len]})).collect::<Vec<_>>();
    let result = json!({"length":len,"members":MEMBERS,"qualified":true,"timing":timing,"capture_repeats":REPEATS,
        "qualified_module_source_byte_identity":true,"qualified_ptx":qualified_ptx,"loaded_ptx":directory.join("loaded-module.ptx"),
        "member_allocation_order":member_order,"output_slot_allocation_order":if reverse {vec![1,0]}else{vec![0,1]},
        "graph_construction_order":construction.iter().map(|(m,s)|json!({"mode":m.name(),"slot":s})).collect::<Vec<_>>(),
        "bindings":binding_metadata,"resources":resources,"warmup_cycles":warmup_cycles,"warmup_seconds":warmup_seconds,
        "balanced_blocks":blocks,"summaries":summaries,"pre_output_proofs":pre_outputs,"post_output_proofs":post_outputs,
        "pre_readonly_proofs":pre_inputs,"post_readonly_proofs":post_inputs,"final_outputs_before_post_relaunch":final_outputs,
        "post_readonly_before_relaunch":true,"full_output_comparisons":200,"readonly_comparisons":40,
        "owned_tensor_bytes_including_guards":owned_bytes(&members),"extra_GPU_oracle_allocations":0,
        "memory_scope":"Explicit typed allocation payload only; CUDA allocator reserve/highwater not sampled here; whole-process NVML recorded by controller.",
        "comparison_scope":"All elements and canaries are compared against fixed source-defined inputs and the separate4 oracle; no checksums substitute for full comparisons.",
        "scope":"Four independent paired-vector Adam parameter updates with unchanged readonly inputs, repeated32 times per CUDA graph. Repetition/cache behavior differs from an owner update; no complete-model/parity or bandwidth causality claim."});
    stream.synchronize()?;
    drop(bindings);
    stream.synchronize()?;
    Ok(result)
}
fn write(root: &Path, report: &Value) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(report)?;
    std::fs::write(root.join("report.json"), &bytes)?;
    std::fs::write(root.join("results.json"), &bytes)?;
    Ok(())
}
pub(super) fn run() -> Result<()> {
    let root = PathBuf::from(
        std::env::var_os("TNSR_PAIRED_VECTOR_BATCH_TIMING_DIR")
            .context("set fresh TNSR_PAIRED_VECTOR_BATCH_TIMING_DIR")?,
    );
    std::fs::create_dir_all(&root)?;
    ensure!(
        !root.join("report.json").exists() && !root.join("results.json").exists(),
        "fresh component output required"
    );
    let reverse = std::env::var("TNSR_PAIRED_VECTOR_BATCH_REVERSE").unwrap_or_else(|_| "0".into());
    ensure!(reverse == "0" || reverse == "1", "REVERSE must be0or1");
    let reverse = reverse == "1";
    let timing =
        std::env::var("TNSR_PAIRED_VECTOR_BATCH_TIMING_ENABLED").unwrap_or_else(|_| "0".into());
    ensure!(timing == "0" || timing == "1", "TIMING_ENABLED must be0or1");
    let timing = timing == "1";
    let qualified = PathBuf::from(
        std::env::var_os("TNSR_PAIRED_VECTOR_BATCH_QUALIFIED_PTX_DIR")
            .context("set qualified paired-vector leaf PTX directory")?,
    );
    let mut lengths = vec![1_048_576usize, 2_818_048, 3_145_728];
    if reverse {
        lengths.reverse();
    }
    let mut report = json!({"qualified":false,"timing":timing,"reverse":reverse,"members":MEMBERS,"capture_repeats":REPEATS,
        "case_order":lengths,"cases":[],"full_output_comparisons":0,"readonly_comparisons":0,"performance_claim":false,
        "complete_model_claim":false,"scope":"CFG(test) component screen only, no owner dispatch. CUDA-event32-replay latency with fixed inputs has different cache/repetition behavior from actual training updates."});
    write(&root, &report)?;
    let context = CudaContext::new(0).context("component screen requires CUDA device")?;
    // SAFETY: private test owns all allocations/captures and exclusively uses one
    // ordered stream; capture/readback and destruction are synchronized.
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream()?;
    let mut cases = Vec::new();
    for len in lengths {
        cases.push(case(
            &context, &stream, &root, &qualified, len, reverse, timing,
        )?);
        report["cases"] = json!(cases);
        report["full_output_comparisons"] = (cases.len() * 200).into();
        report["readonly_comparisons"] = (cases.len() * 40).into();
        write(&root, &report)?;
    }
    stream.synchronize()?;
    report["qualified"] = true.into();
    report["all_graphs_and_readbacks_terminal"] = true.into();
    write(&root, &report)?;
    Ok(())
}
#[test]
fn paired_vector_batch_timing_protocol_balances_slots_orders_and_samples() {
    for reverse in [false, true] {
        let mut samples = std::collections::BTreeMap::new();
        let mut orientations = std::collections::BTreeMap::new();
        for block in 0..BLOCKS {
            let orders = orders(block, reverse);
            for mode in MODES {
                *orientations
                    .entry((mode.index(), slot(mode, block), orders[0][0].index()))
                    .or_insert(0usize) += 1;
            }
            for order in orders {
                for mode in order {
                    *samples
                        .entry((mode.index(), slot(mode, block)))
                        .or_insert(0usize) += 1;
                }
            }
        }
        assert_eq!(samples.len(), 4);
        assert!(samples.values().all(|&n| n == 48));
        assert_eq!(orientations.len(), 8);
        assert!(orientations.values().all(|&n| n == 12));
        assert_eq!(construction(reverse).len(), 4);
        assert_eq!(
            construction(reverse)
                .into_iter()
                .map(|(m, s)| (m.index(), s))
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            4
        );
    }
}
