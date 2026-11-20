//! Manual complete-pair equivalence; no timers, production selection or model claim.
use super::*;
use crate::graph::TensorGraph;
use crate::ptx::mixed::attention_training_online::{backward, tests as baseline};
use crate::ptx::{
    PtxGraph, PtxTarget,
    linear::{Captured, program::Kernel},
};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, sys,
};
use half::bf16;
use std::sync::Arc;
const GUARD: usize = 16;
fn f_marker() -> f32 {
    f32::from_bits(0x4355aa55)
}
fn b_marker() -> bf16 {
    bf16::from_bits(0x4355)
}
struct Outputs {
    values: [CudaSlice<bf16>; 4],
    statistics: CudaSlice<f32>,
}
fn outputs(stream: &Arc<CudaStream>, shape: Shape) -> Result<Outputs> {
    Ok(Outputs {
        values: [
            stream.memcpy_stod(&vec![b_marker(); shape.elements() + GUARD])?,
            stream.memcpy_stod(&vec![b_marker(); shape.elements() + GUARD])?,
            stream.memcpy_stod(&vec![b_marker(); shape.elements() + GUARD])?,
            stream.memcpy_stod(&vec![b_marker(); shape.elements() + GUARD])?,
        ],
        statistics: stream.memcpy_stod(&vec![f_marker(); shape.bh() * 512 * 3 + GUARD])?,
    })
}
fn poison(stream: &Arc<CudaStream>, output: &mut Outputs, shape: Shape) -> Result<()> {
    for x in &mut output.values {
        stream.memcpy_htod(&vec![b_marker(); shape.elements() + GUARD], x)?;
    }
    stream.memcpy_htod(
        &vec![f_marker(); shape.bh() * 512 * 3 + GUARD],
        &mut output.statistics,
    )?;
    Ok(())
}
fn read<T: DeviceRepr>(stream: &Arc<CudaStream>, value: &CudaSlice<T>) -> Result<Vec<T>> {
    let v = stream.memcpy_dtov(value)?;
    stream.synchronize()?;
    Ok(v)
}
fn same_bf16(a: &[bf16], b: &[bf16]) -> Result<()> {
    ensure!(a.len() == b.len(), "BF16 extent changed");
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        ensure!(
            a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
            "BF16 first mismatch {i}: {a:?}/{b:?}"
        );
    }
    Ok(())
}
fn same_f32(a: &[f32], b: &[f32]) -> Result<()> {
    ensure!(a.len() == b.len(), "F32 statistics extent changed");
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        ensure!(
            a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
            "F32 first mismatch {i}: {a:?}/{b:?}"
        );
    }
    Ok(())
}
fn compare(
    stream: &Arc<CudaStream>,
    old: &Outputs,
    new: &Outputs,
    shape: Shape,
    finite: bool,
) -> Result<usize> {
    for (a, b) in old.values.iter().zip(&new.values) {
        let a = read(stream, a)?;
        let b = read(stream, b)?;
        same_bf16(&a, &b)?;
        ensure!(
            a[shape.elements()..]
                .iter()
                .chain(&b[shape.elements()..])
                .all(|x| x.to_bits() == b_marker().to_bits()),
            "BF16 output guard changed"
        );
        if finite {
            ensure!(
                a[..shape.elements()]
                    .iter()
                    .chain(&b[..shape.elements()])
                    .all(|x| x.is_finite()),
                "finite fixture produced nonfinite BF16 output"
            );
        }
    }
    let a = read(stream, &old.statistics)?;
    let b = read(stream, &new.statistics)?;
    same_f32(&a, &b)?;
    let n = shape.bh() * 512 * 3;
    ensure!(
        a[n..]
            .iter()
            .chain(&b[n..])
            .all(|x| x.to_bits() == f_marker().to_bits()),
        "F32 statistic guard changed"
    );
    if finite {
        ensure!(
            a[..n].iter().chain(&b[..n]).all(|x| x.is_finite()),
            "finite fixture produced nonfinite statistics"
        );
    }
    Ok(5)
}
fn paired_kernels(
    functions: &[CudaFunction],
    inputs: &[CudaSlice<bf16>; 4],
    output: &Outputs,
    shape: Shape,
    stream: &Arc<CudaStream>,
    registers: bool,
) -> Result<[Kernel; 3]> {
    let input: Vec<_> = inputs.iter().map(|x| x.device_ptr(stream).0).collect();
    let out: Vec<_> = output
        .values
        .iter()
        .map(|x| x.device_ptr(stream).0)
        .collect();
    let stats = output.statistics.device_ptr(stream).0;
    Ok([
        baseline::kernel(
            functions[usize::from(registers)].clone(),
            shape,
            vec![input[0], input[1], input[2], out[0], stats],
            "forward stats leaf",
        )?,
        baseline::kernel(
            functions[2].clone(),
            shape,
            vec![
                input[0], input[1], input[2], out[0], input[3], stats, out[1],
            ],
            "UNCHANGED backward dQ+delta",
        )?,
        baseline::kernel(
            functions[3].clone(),
            shape,
            vec![
                input[0], input[1], input[2], input[3], stats, out[2], out[3],
            ],
            "UNCHANGED backward dKT+dV",
        )?,
    ])
}
fn capture(stream: &Arc<CudaStream>, kernels: &[Kernel; 3]) -> Result<Captured> {
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let submitted = kernels.iter().try_for_each(|k| k.run(stream));
    let ended = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    submitted?;
    Ok(Captured {
        graph: Some(ended?.context("empty register-statistics pair capture")?),
        stream: stream.clone(),
    })
}
#[test]
#[ignore = "manual complete forward/backward finite-bit/nonfinite-class/poison/replay proof; no timing"]
fn online_register_stats_complete_pair_equivalence_replay_and_statistics_contract() -> Result<()> {
    ensure!(
        std::env::var("TNSR_REQUIRE_CUDA").as_deref() == Ok("1"),
        "TNSR_REQUIRE_CUDA=1 is mandatory"
    );
    let root = std::path::PathBuf::from(
        std::env::var_os("TNSR_ATTENTION_REGISTER_STATS_DIR")
            .context("set fresh TNSR_ATTENTION_REGISTER_STATS_DIR")?,
    );
    ensure!(!root.exists(), "fresh leaf evidence directory required");
    std::fs::create_dir_all(&root)?;
    let context = CudaContext::new(0)?;
    // SAFETY: every allocation/module/capture belongs to this one fixture and
    // ordered stream. Synchronization precedes captures/readbacks/destruction;
    // independent graphs must not inherit uncaptured allocation-event edges.
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream()?;
    let shape = Shape::new(8, 8)?;
    let n = shape.elements();
    let stats_n = shape.bh() * 512 * 3;
    let resource_only = std::env::var_os("TNSR_ATTENTION_REGISTER_RESOURCE_ONLY").is_some();
    let mut reports = Vec::new();
    for layout in [ReadLayout::HeadMajor, ReadLayout::SequenceMajor] {
        for transport in [Transport::Scalar, Transport::Async16] {
            let directory = root.join(format!("{layout:?}-{transport:?}"));
            std::fs::create_dir(&directory)?;
            let irs = [
                ir_with_statistics(shape, layout, transport, StatisticsStorage::Global)?,
                ir_with_statistics(shape, layout, transport, StatisticsStorage::Registers)?,
                backward::d_query_ir_with_transport(shape, layout, layout, transport)?,
                backward::d_key_value_ir_with_transport(shape, layout, layout, transport)?,
            ];
            let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
            for (region_id, ir) in irs.into_iter().enumerate() {
                ir.validate_layouts()?;
                tiles
                    .region_kernels
                    .push(crate::tile::graph::TileRegionKernel { region_id, ir });
            }
            tiles.optimize_indices()?;
            let graph = PtxGraph::from(tiles).with_target(PtxTarget::from_context(&context)?);
            let ptx = graph.to_ptx();
            std::fs::write(directory.join("loaded-module.ptx"), &ptx)?;
            let module = context.load_module(cudarc::nvrtc::Ptx::from_src(ptx))?;
            let functions = (0..4)
                .map(|i| {
                    Ok(module.load_function(
                        graph
                            .region_kernel_name(i)
                            .context("leaf function absent")?,
                    )?)
                })
                .collect::<Result<Vec<_>>>()?;
            let resources=functions.iter().enumerate().map(|(i,f)|->Result<_>{let registers=f.num_regs()?;let shared=f.shared_size_bytes()?;let local=f.local_size_bytes()?;let max=f.max_threads_per_block()?;let capacity=f.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?;let role=["Global forward","Register forward","UNCHANGED dQ+delta","UNCHANGED dKT+dV"][i];Ok(serde_json::json!({"role":role,"registers":registers,"shared_bytes":shared,"local_bytes":local,"max_threads":max,"API_capacity_at_256":capacity,"diagnostic_admitted":registers>0&&registers<=192&&shared==49152&&local==0&&max>=256&&capacity>=1,"strict_resource_admitted":registers>0&&registers<=128&&shared==49152&&local==0&&max>=256&&capacity>=2,"scope":"Actual loaded resources; API capacity is not achieved occupancy."}))}).collect::<Result<Vec<_>>>()?;
            std::fs::write(
                directory.join("resources-before-math.json"),
                serde_json::to_vec_pretty(&resources)?,
            )?;
            ensure!(
                resources.iter().all(|r| r["diagnostic_admitted"] == true),
                "leaf resource screen failed before any math launch; inspect resources"
            );
            if resource_only {
                reports.push(serde_json::json!({"layout":format!("{layout:?}"),"transport":format!("{transport:?}"),"resources":resources,"math_executed":false}));
                continue;
            }
            let mut inputs = [
                stream.alloc_zeros::<bf16>(n + GUARD)?,
                stream.alloc_zeros(n + GUARD)?,
                stream.alloc_zeros(n + GUARD)?,
                stream.alloc_zeros(n + GUARD)?,
            ];
            let mut old = outputs(&stream, shape)?;
            let mut new = outputs(&stream, shape)?;
            let oldkernels = paired_kernels(&functions, &inputs, &old, shape, &stream, false)?;
            let newkernels = paired_kernels(&functions, &inputs, &new, shape, &stream, true)?;
            stream.synchronize()?;
            // Keep the same captured pointer graph across all input mutations.
            let oldgraph = capture(&stream, &oldkernels)?;
            let newgraph = capture(&stream, &newkernels)?;
            let mut comparisons = 0;
            let mut readonly = 0;
            for revision in 0..6 {
                let mut host = Vec::new();
                for salt in 1..=4 {
                    let mut logical =
                        baseline::data(shape, if revision == 5 { 0 } else { revision }, salt);
                    if revision == 5 && salt == 2 {
                        logical[511 * 64] = bf16::INFINITY;
                    }
                    let mut physical = baseline::physical(&logical, shape, layout);
                    physical.extend([b_marker(); GUARD]);
                    host.push(physical);
                }
                for (input, feed) in inputs.iter_mut().zip(&host) {
                    stream.memcpy_htod(feed, input)?;
                }
                poison(&stream, &mut old, shape)?;
                poison(&stream, &mut new, shape)?;
                oldkernels[0].run(&stream)?;
                newkernels[0].run(&stream)?;
                stream.synchronize()?;
                let before_old = read(&stream, &old.statistics)?;
                let before_new = read(&stream, &new.statistics)?;
                same_f32(&before_old, &before_new)?;
                ensure!(
                    before_old[..stats_n]
                        .iter()
                        .zip(&before_new[..stats_n])
                        .enumerate()
                        .all(|(i, (a, b))| i % 3 != 2
                            || (a.to_bits() == f_marker().to_bits()
                                && b.to_bits() == f_marker().to_bits())),
                    "forward overwrote statistics slot2"
                );
                ensure!(
                    before_old[stats_n..]
                        .iter()
                        .chain(&before_new[stats_n..])
                        .all(|x| x.to_bits() == f_marker().to_bits()),
                    "forward statistics guard changed"
                );
                same_bf16(
                    &read(&stream, &old.values[0])?,
                    &read(&stream, &new.values[0])?,
                )?;
                comparisons += 2;
                for k in oldkernels.iter().skip(1).chain(newkernels.iter().skip(1)) {
                    k.run(&stream)?;
                }
                stream.synchronize()?;
                comparisons += compare(&stream, &old, &new, shape, revision <= 2)?;
                for (output, before) in [(&old, &before_old), (&new, &before_new)] {
                    let after = read(&stream, &output.statistics)?;
                    ensure!(
                        before[..stats_n]
                            .iter()
                            .zip(&after[..stats_n])
                            .enumerate()
                            .all(|(i, (a, b))| i % 3 == 2 || a.to_bits() == b.to_bits()),
                        "unchanged backward rewrote forward m/l slots"
                    );
                    if revision <= 2 {
                        ensure!(
                            after[..stats_n]
                                .iter()
                                .enumerate()
                                .all(|(i, x)| i % 3 != 2 || x.to_bits() != f_marker().to_bits()),
                            "dQ failed to define delta slot2"
                        );
                    }
                }
                for _ in 0..2 {
                    poison(&stream, &mut old, shape)?;
                    poison(&stream, &mut new, shape)?;
                    oldgraph
                        .graph
                        .as_ref()
                        .context("old pair graph absent")?
                        .launch()?;
                    newgraph
                        .graph
                        .as_ref()
                        .context("new pair graph absent")?
                        .launch()?;
                    stream.synchronize()?;
                    comparisons += compare(&stream, &old, &new, shape, revision <= 2)?;
                }
                for (input, expected) in inputs.iter().zip(&host) {
                    let actual = read(&stream, input)?;
                    ensure!(
                        actual
                            .iter()
                            .zip(expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "readonly Q/K/V/dO or input guard changed"
                    );
                    readonly += 1;
                }
            }
            drop(newgraph);
            drop(oldgraph);
            ensure!(
                comparisons == 102 && readonly == 24,
                "leaf comparison census changed"
            );
            reports.push(serde_json::json!({"layout":format!("{layout:?}"),"transport":format!("{transport:?}"),"resources":resources,"math_executed":true,"feeds":6,"complete_array_comparisons":comparisons,"readonly_input_checks":readonly,"direct_then_graph_replays":2,"persistent_graph_across_feeds":true,"poison_before_every_replay":true,"forward_slots01_and_unchanged_slot2":true,"backward_preserves_slots01_defines_delta2":true,"finite_bits_and_nonfinite_classes_equal":true,"guards":GUARD}));
        }
    }
    std::fs::write(
        root.join("qualification.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"qualified":!resource_only,"resource_only":resource_only,"cases":reports,"total_complete_array_comparisons":if resource_only {0}else{408},"total_readonly_input_checks":if resource_only {0}else{96},"scope":"CFGTEST complete-pair leaf only; no production hook, timers, optimizer/model/performance/occupancy/parity claim."}),
        )?,
    )?;
    Ok(())
}
