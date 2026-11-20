//! Manual all-pattern typed-load equivalence; no timers or optimizer publication.
use super::*;
use crate::{
    graph::NodeIndex,
    tile::{DType, VirtualTensor},
};
use cudarc::driver::{DevicePtr, PushKernelArg, sys};
use half::bf16;
#[test]
#[ignore = "manual all65536 BF16 patterns/Adam health/tails/1024+256/replay qualification"]
fn borrowed_bf16_adam_all_patterns_match_accumulate_then_f32_adam() -> Result<()> {
    let context = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var_os("TNSR_REQUIRE_CUDA").is_none(),
                "CUDA required: {e}"
            );
            return Ok(());
        }
    };
    // SAFETY: this private fixture owns the context and every allocation. All
    // copies, independent captures, launches, readbacks and destruction use the
    // same ordered stream; explicit synchronization brackets capture/readback.
    // Automatic per-allocation event tracking would otherwise add dependencies
    // on an uncaptured event from the other independent comparison graph.
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream()?;
    for len in [3usize, 257, 65539] {
        let view = VirtualTensor::identity(NodeIndex::new(0), vec![len], DType::BF16);
        let kernels = kernels::Kernels::compile(
            &context,
            vec![
                kernels::accumulate_view(&view, &[len], len, true)?,
                kernels::adam(len)?,
                kernels::adam_bf16_first(len)?,
            ],
        )?;
        let functions = kernels
            .names
            .iter()
            .map(|n| kernels.module.load_function(n).map_err(anyhow::Error::from))
            .collect::<Result<Vec<_>>>()?;
        if let Some(dir) = std::env::var_os("TNSR_BORROWED_BF16_AUDIT") {
            std::fs::create_dir_all(&dir)?;
            let resources = functions.iter().zip(&kernels.names).map(|(f,name)| -> Result<_> {
                Ok(serde_json::json!({"function":name,"registers":f.num_regs()?,"shared_bytes":f.shared_size_bytes()?,"local_bytes":f.local_size_bytes()?,"max_threads":f.max_threads_per_block()?}))
            }).collect::<Result<Vec<_>>>()?;
            std::fs::write(
                std::path::PathBuf::from(dir).join(format!("admission-len{len}.json")),
                serde_json::to_vec_pretty(&resources)?,
            )?;
        }
        assert_eq!(functions[2].local_size_bytes()?, 0);
        assert!(functions[2].max_threads_per_block()? >= 1024);
        let mut raw = stream.alloc_zeros::<bf16>(len)?;
        let mut grad = stream.alloc_zeros::<f32>(len)?;
        let w = stream.memcpy_stod(
            &(0..len)
                .map(|i| ((i % 9) as f32 - 4.) / 32.)
                .collect::<Vec<_>>(),
        )?;
        let m = stream.memcpy_stod(&vec![0.001f32; len])?;
        let v = stream.memcpy_stod(&vec![0.01f32; len])?;
        let mut settings = stream.alloc_zeros::<f32>(10)?;
        let mut old = (0..3)
            .map(|_| stream.alloc_zeros::<f32>(len))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut new = (0..3)
            .map(|_| stream.alloc_zeros::<f32>(len))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut oldhealth = stream.alloc_zeros::<f32>(len.div_ceil(256))?;
        let mut newhealth = stream.alloc_zeros::<f32>(len.div_ceil(256))?;
        for compact in [false, true] {
            let config = launch::standalone_adam(len, compact)?;
            let mut capture = |typed: bool| -> Result<_> {
                stream.synchronize()?;
                stream
                    .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                let submitted = (|| -> Result<()> {
                    if !typed {
                        let mut a = stream.launch_builder(&functions[0]);
                        a.arg(&raw).arg(&mut grad);
                        unsafe { a.launch(LaunchConfig::for_num_elems(u32::try_from(len)?)) }?;
                    }
                    let output = if typed { &mut new } else { &mut old };
                    let mut a = stream.launch_builder(&functions[if typed { 2 } else { 1 }]);
                    a.arg(&w);
                    if typed {
                        a.arg(&raw);
                    } else {
                        a.arg(&grad);
                    }
                    a.arg(&m).arg(&v).arg(&settings);
                    for value in output {
                        a.arg(value);
                    }
                    a.arg(if typed {
                        &mut newhealth
                    } else {
                        &mut oldhealth
                    });
                    unsafe { a.launch(config) }?;
                    Ok(())
                })();
                let end = stream.end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                );
                submitted?;
                end?.context("empty Adam pattern capture")
            };
            let oldgraph = capture(false)?;
            let newgraph = capture(true)?;
            for phase in 0..3 {
                let host = (0..len)
                    .map(|i| bf16::from_bits((i as u16).wrapping_add((phase * 1237) as u16)))
                    .collect::<Vec<_>>();
                stream.memcpy_htod(&host, &mut raw)?;
                let scale = [1., 0.25, 0.][phase];
                let decay = [0., 0.01, 0.][phase];
                stream.memcpy_htod(
                    &[
                        1e-3,
                        0.9,
                        0.999,
                        1. - 0.9,
                        1. - 0.999,
                        1. - 0.9,
                        1. - 0.999,
                        1e-8,
                        decay,
                        scale,
                    ],
                    &mut settings,
                )?;
                for _ in 0..2 {
                    oldgraph.launch()?;
                    newgraph.launch()?;
                    stream.synchronize()?;
                    for (a, b) in old
                        .iter()
                        .chain(std::iter::once(&oldhealth))
                        .zip(new.iter().chain(std::iter::once(&newhealth)))
                    {
                        let mut x = vec![0f32; a.len()];
                        let mut y = vec![0f32; b.len()];
                        stream.memcpy_dtoh(a, &mut x)?;
                        stream.memcpy_dtoh(b, &mut y)?;
                        stream.synchronize()?;
                        for (i, (x, y)) in x.iter().zip(y).enumerate() {
                            assert!(
                                x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
                                "len{len},compact{compact},phase{phase},element{i}: {x}/{y}"
                            );
                        }
                    }
                }
                let mut after = vec![bf16::ZERO; len];
                stream.memcpy_dtoh(&raw, &mut after)?;
                stream.synchronize()?;
                assert!(
                    host.iter()
                        .zip(after)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
            drop(newgraph);
            drop(oldgraph);
        }
        if let Some(dir) = std::env::var_os("TNSR_BORROWED_BF16_AUDIT") {
            std::fs::create_dir_all(&dir)?;
            std::fs::write(
                std::path::PathBuf::from(dir).join(format!("pattern-len{len}.json")),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"len":len,"all65536patterns":len>=65536,"compact_modes":[false,true],"feeds":3,"replays":2,"exact_nonNaN_bits":true,"NaNclasses":true,"gradient_registers":functions[2].num_regs()?,"shared":functions[2].shared_size_bytes()?,"local":functions[2].local_size_bytes()?,"max_threads":functions[2].max_threads_per_block()?,"source_pointer_alignment":raw.device_ptr(&stream).0%16,"scope":"Requested source widths differ; no allocation/DRAM/time saving measured."}),
                )?,
            )?;
        }
    }
    Ok(())
}
