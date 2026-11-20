//! Untimed independent-launch versus batched paired-vector CUDA qualification.
use super::*;
use crate::{
    graph::TensorGraph,
    ptx::{PtxGraph, PtxTarget},
    tile::TileGraph,
};
use cudarc::driver::{CudaFunction, DevicePtr, DeviceRepr, sys};
use cudarc::nvrtc::Ptx;
use half::bf16;
use std::sync::Arc;
const GUARD: usize = 16;
fn f_sentinel() -> f32 {
    f32::from_bits(0x4355aa55)
}
fn b_sentinel() -> bf16 {
    bf16::from_bits(0x4355)
}
struct Inputs {
    weight: CudaSlice<f32>,
    gradient: CudaSlice<bf16>,
    moment: CudaSlice<f32>,
    variance: CudaSlice<f32>,
    settings: CudaSlice<f32>,
}
struct Outputs {
    weight: CudaSlice<f32>,
    moment: CudaSlice<f32>,
    variance: CudaSlice<f32>,
    health: CudaSlice<f32>,
    shadow: CudaSlice<bf16>,
}
struct Member {
    input: Inputs,
    old: Outputs,
    new: Outputs,
}
struct Feed {
    weight: Vec<f32>,
    gradient: Vec<bf16>,
    moment: Vec<f32>,
    variance: Vec<f32>,
    settings: Vec<f32>,
}
fn outputs(stream: &Arc<CudaStream>, len: usize) -> Result<Outputs> {
    Ok(Outputs {
        weight: stream.memcpy_stod(&vec![f_sentinel(); len + GUARD])?,
        moment: stream.memcpy_stod(&vec![f_sentinel(); len + GUARD])?,
        variance: stream.memcpy_stod(&vec![f_sentinel(); len + GUARD])?,
        health: stream.memcpy_stod(&vec![f_sentinel(); len / 256 + GUARD])?,
        shadow: stream.memcpy_stod(&vec![b_sentinel(); len + GUARD])?,
    })
}
fn member(stream: &Arc<CudaStream>, len: usize) -> Result<Member> {
    Ok(Member {
        input: Inputs {
            weight: stream.alloc_zeros(len + GUARD)?,
            gradient: stream.alloc_zeros(len + GUARD)?,
            moment: stream.alloc_zeros(len + GUARD)?,
            variance: stream.alloc_zeros(len + GUARD)?,
            settings: stream.alloc_zeros(10 + GUARD)?,
        },
        old: outputs(stream, len)?,
        new: outputs(stream, len)?,
    })
}
fn poison(stream: &Arc<CudaStream>, output: &mut Outputs, len: usize) -> Result<()> {
    for buffer in [&mut output.weight, &mut output.moment, &mut output.variance] {
        stream.memcpy_htod(&vec![f_sentinel(); len + GUARD], buffer)?;
    }
    stream.memcpy_htod(&vec![f_sentinel(); len / 256 + GUARD], &mut output.health)?;
    stream.memcpy_htod(&vec![b_sentinel(); len + GUARD], &mut output.shadow)?;
    Ok(())
}
fn feed(len: usize, member: usize, phase: usize) -> Feed {
    let mut result = Feed {
        weight: vec![f_sentinel(); len + GUARD],
        gradient: vec![b_sentinel(); len + GUARD],
        moment: vec![f_sentinel(); len + GUARD],
        variance: vec![f_sentinel(); len + GUARD],
        settings: vec![f_sentinel(); 10 + GUARD],
    };
    let patterns = [
        0u32, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x3f807fff, 0x3f808000,
        0x3f808001, 0x3f818000, 0xbf808000, 0x7f7f7fff, 0x7f7f8000, 0x7f7f8001, 0x7f7fffff,
        0xff7fffff, 0x7fc12345, 0xffa12345,
    ];
    for i in 0..len {
        result.weight[i] = match phase {
            2 => f32::MAX,
            3 => f32::from_bits(patterns[(i + member) % patterns.len()]),
            _ => ((i % 17) as f32 - 8.) / 32. + member as f32 / 128.,
        };
        result.gradient[i] = match phase {
            0 | 4 => bf16::from_f32(((i % 13) as f32 - 6.) / 64.),
            2 | 3 => bf16::ZERO,
            _ => bf16::from_bits((i as u16).wrapping_add((member * 1237 + phase * 571) as u16)),
        };
        result.moment[i] = if phase == 2 || phase == 3 {
            0.
        } else if phase == 4 && i == member * 256 + 63 {
            f32::from_bits(0x7fc12345)
        } else {
            0.001 + member as f32 / 4096.
        };
        result.variance[i] = if phase == 2 || phase == 3 {
            0.
        } else if phase == 4 && i == member * 256 + 64 {
            -1.
        } else {
            0.01 + member as f32 / 4096.
        };
    }
    result.settings[..10].copy_from_slice(&[
        if phase == 2 || phase == 3 {
            0.
        } else {
            1e-3 * (member + 1) as f32
        },
        0.9,
        0.999,
        1. - 0.9,
        1. - 0.999,
        1. - 0.9,
        1. - 0.999,
        1e-8,
        if phase == 5 { 0.01 } else { 0. },
        if phase == 5 && member % 2 == 1 {
            0.25
        } else {
            1.
        },
    ]);
    result
}
fn upload(stream: &Arc<CudaStream>, input: &mut Inputs, feed: &Feed) -> Result<()> {
    stream.memcpy_htod(&feed.weight, &mut input.weight)?;
    stream.memcpy_htod(&feed.gradient, &mut input.gradient)?;
    stream.memcpy_htod(&feed.moment, &mut input.moment)?;
    stream.memcpy_htod(&feed.variance, &mut input.variance)?;
    stream.memcpy_htod(&feed.settings, &mut input.settings)?;
    Ok(())
}
fn launch_separate(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    members: &mut [Member],
    len: usize,
) -> Result<()> {
    for member in members {
        let input = &member.input;
        let output = &mut member.old;
        let mut launch = stream.launch_builder(function);
        launch
            .arg(&input.weight)
            .arg(&input.gradient)
            .arg(&input.moment)
            .arg(&input.variance)
            .arg(&input.settings)
            .arg(&mut output.weight)
            .arg(&mut output.moment)
            .arg(&mut output.variance)
            .arg(&mut output.health)
            .arg(&mut output.shadow);
        // SAFETY: typed independent readonly inputs and staged outputs cover the
        // whole canonical paired-vector extent; guards are outside all writes.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (u32::try_from(len / 1024)?, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
    }
    Ok(())
}
fn launch_batch(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    members: &mut [Member],
    len: usize,
) -> Result<()> {
    let blocks = (len / 1024)
        .checked_mul(members.len())
        .context("batch GPU grid overflow")?;
    let mut launch = stream.launch_builder(function);
    for member in members {
        let input = &member.input;
        let output = &mut member.new;
        launch
            .arg(&input.weight)
            .arg(&input.gradient)
            .arg(&input.moment)
            .arg(&input.variance)
            .arg(&input.settings)
            .arg(&mut output.weight)
            .arg(&mut output.moment)
            .arg(&mut output.variance)
            .arg(&mut output.health)
            .arg(&mut output.shadow);
    }
    // SAFETY: CPU-proved disjoint uniform CTA regions remap into each member's
    // canonical local extent. All ten typed spans per member remain alive here.
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (u32::try_from(blocks)?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    Ok(())
}
fn download<T: DeviceRepr>(stream: &Arc<CudaStream>, buffer: &CudaSlice<T>) -> Result<Vec<T>> {
    let values = stream.memcpy_dtov(buffer)?;
    stream.synchronize()?;
    Ok(values)
}
fn compare(stream: &Arc<CudaStream>, member: &Member, len: usize) -> Result<usize> {
    for (a, b, n) in [
        (&member.old.weight, &member.new.weight, len),
        (&member.old.moment, &member.new.moment, len),
        (&member.old.variance, &member.new.variance, len),
        (&member.old.health, &member.new.health, len / 256),
    ] {
        let a = download(stream, a)?;
        let b = download(stream, b)?;
        ensure!(a.len() == b.len(), "batch output extent mismatch");
        for (index, (a, b)) in a.iter().zip(&b).enumerate() {
            ensure!(
                a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan(),
                "batch output first mismatch at{index}: {a:?}/{b:?}"
            );
        }
        ensure!(
            a[n..]
                .iter()
                .chain(&b[n..])
                .all(|v| v.to_bits() == f_sentinel().to_bits()),
            "batch output guard changed"
        );
    }
    let a = download(stream, &member.old.shadow)?;
    let b = download(stream, &member.new.shadow)?;
    ensure!(
        a.iter()
            .zip(&b)
            .all(|(a, b)| a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan()),
        "batch BF16 shadow bits/NaN classes differ"
    );
    ensure!(
        a[len..]
            .iter()
            .chain(&b[len..])
            .all(|v| v.to_bits() == b_sentinel().to_bits()),
        "batch BF16 shadow guard changed"
    );
    Ok(5)
}
fn readonly(stream: &Arc<CudaStream>, input: &Inputs, feed: &Feed) -> Result<usize> {
    for (buffer, expected) in [
        (&input.weight, &feed.weight),
        (&input.moment, &feed.moment),
        (&input.variance, &feed.variance),
        (&input.settings, &feed.settings),
    ] {
        let actual = download(stream, buffer)?;
        ensure!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "batch readonly F32 input changed"
        );
    }
    let actual = download(stream, &input.gradient)?;
    ensure!(
        actual
            .iter()
            .zip(&feed.gradient)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "batch readonly BF16 input changed"
    );
    Ok(5)
}
fn logical_span<T: DeviceRepr>(
    buffer: &CudaSlice<T>,
    elements: usize,
    stream: &Arc<CudaStream>,
) -> Result<Span> {
    Ok(Span {
        address: usize::try_from(buffer.device_ptr(stream).0)?,
        bytes: elements
            .checked_mul(std::mem::size_of::<T>())
            .context("batch span size overflow")?,
    })
}
fn spans(member: &Member, len: usize, stream: &Arc<CudaStream>) -> Result<[Span; 10]> {
    let a = &member.input;
    let b = &member.new;
    Ok([
        logical_span(&a.weight, len, stream)?,
        logical_span(&a.gradient, len, stream)?,
        logical_span(&a.moment, len, stream)?,
        logical_span(&a.variance, len, stream)?,
        logical_span(&a.settings, 10, stream)?,
        logical_span(&b.weight, len, stream)?,
        logical_span(&b.moment, len, stream)?,
        logical_span(&b.variance, len, stream)?,
        logical_span(&b.health, len / 256, stream)?,
        logical_span(&b.shadow, len, stream)?,
    ])
}
#[test]
#[ignore = "manual untimed paired-vector batch resources/readonly/poison/replay/NaN qualification"]
fn paired_vector_batch_matches_separate_updates_with_replay_and_independent_members() -> Result<()>
{
    let context = match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => {
            ensure!(
                std::env::var_os("TNSR_REQUIRE_CUDA").is_none(),
                "CUDA required: {error}"
            );
            return Ok(());
        }
    };
    // SAFETY: this private fixture owns all modules, allocations and graphs and
    // synchronizes its one ordered stream at captures/readbacks/destruction.
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream()?;
    let root = std::path::PathBuf::from(
        std::env::var_os("TNSR_PAIRED_VECTOR_BATCH_DIR")
            .context("set fresh TNSR_PAIRED_VECTOR_BATCH_DIR")?,
    );
    std::fs::create_dir_all(&root)?;
    let mut reports = Vec::new();
    for len in [1024usize, 65536, 1_048_576, 2_818_048, 3_145_728] {
        for count in [1usize, 4] {
            let stem = format!("len{len}-members{count}");
            let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
            for (region_id, ir) in [paired_ir(len, AdamKind::Vector4)?, ir(len, count)?]
                .into_iter()
                .enumerate()
            {
                tiles
                    .region_kernels
                    .push(crate::tile::graph::TileRegionKernel { region_id, ir });
            }
            tiles.optimize_indices()?;
            let graph = PtxGraph::from(tiles).with_target(PtxTarget::from_context(&context)?);
            let source = graph.to_ptx();
            let names = (0..2)
                .map(|index| {
                    graph
                        .region_kernel_name(index)
                        .map(str::to_owned)
                        .context("batch leaf entry missing")
                })
                .collect::<Result<Vec<_>>>()?;
            let ptx_path = root.join(format!("{stem}.ptx"));
            ensure!(!ptx_path.exists(), "fresh paired batch PTX output required");
            std::fs::write(&ptx_path, &source)?;
            let module = context.load_module(Ptx::from_src(source.clone()))?;
            let functions = names
                .iter()
                .map(|n| module.load_function(n).map_err(anyhow::Error::from))
                .collect::<Result<Vec<_>>>()?;
            let resources=functions.iter().zip(&names).map(|(f,n)|->Result<_>{Ok(serde_json::json!({"name":n,
                "registers":f.num_regs()?,"shared_bytes":f.shared_size_bytes()?,"local_bytes":f.local_size_bytes()?,
                "max_threads":f.max_threads_per_block()?,"capacity256":f.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?}))}).collect::<Result<Vec<_>>>()?;
            std::fs::write(
                root.join(format!("{stem}-resources.json")),
                serde_json::to_vec_pretty(&resources)?,
            )?;
            ensure!(
                functions[1].local_size_bytes()? == 0
                    && functions[1].num_regs()? <= 128
                    && functions[1].max_threads_per_block()? >= 256
                    && functions[1].shared_size_bytes()? == 32
                    && functions[1].occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?
                        >= 2,
                "batch resource screen failed before launch; inspect retained resource/PTX evidence"
            );
            let mut members = (0..count)
                .map(|_| member(&stream, len))
                .collect::<Result<Vec<_>>>()?;
            let bindings = members
                .iter()
                .map(|m| spans(m, len, &stream))
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                admitted(len, &bindings),
                "actual batch spans failed whole/range/alignment/disjointness proof"
            );
            let mut comparisons = 0usize;
            let mut readonly_checks = 0usize;
            for phase in 0..6 {
                let feeds = (0..count)
                    .map(|member| feed(len, member, phase))
                    .collect::<Vec<_>>();
                for (member, feed) in members.iter_mut().zip(&feeds) {
                    upload(&stream, &mut member.input, feed)?;
                    poison(&stream, &mut member.old, len)?;
                    poison(&stream, &mut member.new, len)?;
                }
                launch_separate(&stream, &functions[0], &mut members, len)?;
                launch_batch(&stream, &functions[1], &mut members, len)?;
                stream.synchronize()?;
                for member in &members {
                    comparisons += compare(&stream, member, len)?;
                }
                stream
                    .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                let submitted = launch_separate(&stream, &functions[0], &mut members, len);
                let old = stream.end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                );
                submitted?;
                let old = old?.context("empty independent paired capture")?;
                stream
                    .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                let submitted = launch_batch(&stream, &functions[1], &mut members, len);
                let batch = stream.end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                );
                submitted?;
                let batch = batch?.context("empty batched paired capture")?;
                for _ in 0..2 {
                    for member in &mut members {
                        poison(&stream, &mut member.old, len)?;
                        poison(&stream, &mut member.new, len)?;
                    }
                    old.launch()?;
                    batch.launch()?;
                    stream.synchronize()?;
                    for member in &members {
                        comparisons += compare(&stream, member, len)?;
                    }
                }
                if phase == 2 {
                    for member in &members {
                        let w = download(&stream, &member.new.weight)?;
                        let h = download(&stream, &member.new.health)?;
                        let s = download(&stream, &member.new.shadow)?;
                        ensure!(
                            w[..len].iter().all(|v| v.is_finite())
                                && h[..len / 256].iter().all(|v| v.to_bits() == 0)
                                && s[..len].iter().all(|v| v.is_infinite()),
                            "batch changed finite-master/BF16-infinity health boundary"
                        );
                    }
                }
                if phase == 4 {
                    // Each member has an invalid moment/variance in its own
                    // different logical health group; cross-member health ABI
                    // mixups cannot hide behind all-zero or all-NaN partials.
                    for (index, member) in members.iter().enumerate() {
                        let health = download(&stream, &member.new.health)?;
                        ensure!(
                            health[..len / 256].iter().enumerate().all(|(group, v)| {
                                if group == index {
                                    v.is_nan()
                                } else {
                                    v.to_bits() == 0
                                }
                            }),
                            "batched independent health group ownership changed"
                        );
                    }
                }
                drop(batch);
                drop(old);
                for (member, feed) in members.iter().zip(&feeds) {
                    readonly_checks += readonly(&stream, &member.input, feed)?;
                }
            }
            reports.push(serde_json::json!({"elements_per_member":len,"members":count,"pointer_parameters":10*count,"parameter_bytes":80*count,
                "shared_scratch_bytes":32,"resources":resources,"array_comparisons":comparisons,"readonly_checks":readonly_checks,
                "feeds":6,"graph_replays_per_feed":2,"direct_before_capture":true,"poison_before_every_replay":true,
                "all65536_BF16_gradient_patterns":len>=65536,"nonNaN_bits_and_NaN_classes_exact":true,
                "health_and_BF16_shadow_guarded":true,"isolated_health_group_per_member":true,"performance_claim":false}));
        }
    }
    std::fs::write(
        root.join("qualification.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "qualified":true,"scope":"CFG(test) paired vector leaf batching only: no optimizer owner dispatch, timing, occupancy or model throughput claim.","cases":reports
        }))?,
    )?;
    Ok(())
}

#[path = "gpu/timing.rs"]
mod timing;
#[test]
#[ignore = "manual source-bound balanced component screen; default untimed, timing requires explicit environment grant"]
fn paired_vector_batch_component_timing() -> Result<()> {
    timing::run()
}

#[path = "gpu/scoped.rs"]
mod scoped_parameters;
