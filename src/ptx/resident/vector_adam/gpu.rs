//! Manual all-pattern typed-load equivalence; no timers or optimizer publication.
use super::*;
use crate::ptx::resident::{kernels, launch};
use crate::{
    graph::NodeIndex,
    tile::{DType, VirtualTensor},
};
use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, LaunchConfig};
use cudarc::driver::{DevicePtr, PushKernelArg, sys};
use half::bf16;
#[test]
#[ignore = "manual all65536 BF16 patterns/Adam health/tails/1024+256/replay qualification"]
fn vector_adam_leaf_all_patterns_states_health_replay_readonly_and_fallback() -> Result<()> {
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
    for len in [1024usize, 65536, 66560, 2818048, 3145728] {
        let view = VirtualTensor::identity(NodeIndex::new(0), vec![len], DType::BF16);
        let kernels = compile_loaded(
            &context,
            len,
            vec![
                kernels::accumulate_view(&view, &[len], len, true)?,
                kernels::adam(len)?,
                kernels::adam_bf16_first(len)?,
                super::super::ir(len)?,
            ],
        )?;
        let functions = kernels
            .names
            .iter()
            .map(|n| kernels.module.load_function(n).map_err(anyhow::Error::from))
            .collect::<Result<Vec<_>>>()?;
        if let Some(dir) = std::env::var_os("TNSR_VECTOR_ADAM_DIR") {
            std::fs::create_dir_all(&dir)?;
            let resources = functions.iter().zip(&kernels.names).map(|(f,name)| -> Result<_> {
                Ok(serde_json::json!({"function":name,"registers":f.num_regs()?,"shared_bytes":f.shared_size_bytes()?,"local_bytes":f.local_size_bytes()?,"max_threads":f.max_threads_per_block()?,"capacity256":f.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?}))
            }).collect::<Result<Vec<_>>>()?;
            std::fs::write(
                std::path::PathBuf::from(dir).join(format!("admission-len{len}.json")),
                serde_json::to_vec_pretty(&resources)?,
            )?;
        }
        assert_eq!(functions[2].local_size_bytes()?, 0);
        assert!(functions[2].max_threads_per_block()? >= 1024);
        assert_eq!(functions[3].local_size_bytes()?, 0);
        assert!(functions[3].num_regs()? <= 128);
        assert!(functions[3].max_threads_per_block()? >= 256);
        assert!(functions[3].occupancy_max_active_blocks_per_multiprocessor(256, 0, None)? >= 2);
        assert_eq!(functions[3].shared_size_bytes()?, 32);
        let mut raw = stream.alloc_zeros::<bf16>(len)?;
        let mut grad = stream.alloc_zeros::<f32>(len)?;
        let mut w = stream.memcpy_stod(
            &(0..len)
                .map(|i| ((i % 9) as f32 - 4.) / 32.)
                .collect::<Vec<_>>(),
        )?;
        let mut m = stream.memcpy_stod(&vec![0.001f32; len])?;
        let mut v = stream.memcpy_stod(&vec![0.01f32; len])?;
        let mut settings = stream.alloc_zeros::<f32>(10)?;
        let mut old = (0..3)
            .map(|_| stream.memcpy_stod(&vec![sentinel(); len + 16]))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut new = (0..3)
            .map(|_| stream.memcpy_stod(&vec![sentinel(); len + 16]))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut oldhealth = stream.memcpy_stod(&vec![sentinel(); len.div_ceil(256) + 16])?;
        let mut newhealth = stream.memcpy_stod(&vec![sentinel(); len.div_ceil(256) + 16])?;
        let actual_spans = [
            Span {
                address: usize::try_from(w.device_ptr(&stream).0)?,
                bytes: len * 4,
            },
            Span {
                address: usize::try_from(m.device_ptr(&stream).0)?,
                bytes: len * 4,
            },
            Span {
                address: usize::try_from(v.device_ptr(&stream).0)?,
                bytes: len * 4,
            },
            Span {
                address: usize::try_from(raw.device_ptr(&stream).0)?,
                bytes: len * 2,
            },
            Span {
                address: usize::try_from(new[0].device_ptr(&stream).0)?,
                bytes: len * 4,
            },
            Span {
                address: usize::try_from(new[1].device_ptr(&stream).0)?,
                bytes: len * 4,
            },
            Span {
                address: usize::try_from(new[2].device_ptr(&stream).0)?,
                bytes: len * 4,
            },
            Span {
                address: usize::try_from(newhealth.device_ptr(&stream).0)?,
                bytes: len / 64,
            },
        ];
        assert!(super::super::admitted(len, &actual_spans, true));
        assert!(!super::super::admitted(len, &actual_spans, false));
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
                    let mut a = stream.launch_builder(&functions[if typed { 3 } else { 1 }]);
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
                    let chosen = if typed {
                        LaunchConfig {
                            grid_dim: (u32::try_from(len / 1024)?, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        }
                    } else {
                        config
                    };
                    unsafe { a.launch(chosen) }?;
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
            for phase in 0..8 {
                let host = (0..len)
                    .map(|i| bf16::from_bits((i as u16).wrapping_add((phase * 1237) as u16)))
                    .collect::<Vec<_>>();
                stream.memcpy_htod(&host, &mut raw)?;
                let wh = (0..len)
                    .map(|i| match phase {
                        3 => f32::MAX,
                        4 if i % 257 == 0 => f32::INFINITY,
                        6 => {
                            if i % 2 == 0 {
                                0.
                            } else {
                                -0.
                            }
                        }
                        _ => ((i % 9) as f32 - 4.) / 32.,
                    })
                    .collect::<Vec<_>>();
                let mh = (0..len)
                    .map(|i| {
                        if phase == 4 && i % 251 == 0 {
                            f32::NAN
                        } else if phase == 3 {
                            f32::MAX
                        } else {
                            0.001
                        }
                    })
                    .collect::<Vec<_>>();
                let vh = (0..len)
                    .map(|i| match phase {
                        5 => {
                            if i % 2 == 0 {
                                -1.
                            } else {
                                f32::INFINITY
                            }
                        }
                        7 => f32::from_bits(1),
                        _ => 0.01,
                    })
                    .collect::<Vec<_>>();
                stream.memcpy_htod(&wh, &mut w)?;
                stream.memcpy_htod(&mh, &mut m)?;
                stream.memcpy_htod(&vh, &mut v)?;

                let scale = [1., 0.25, 0., 1., 1., 1., 0., 1.][phase];
                let decay = [0., 0.01, 0., 0.1, 0., 0., 1., 0.][phase];
                let expected_settings = [
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
                ];
                stream.memcpy_htod(&expected_settings, &mut settings)?;
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
                        assert!(
                            x[x.len() - 16..]
                                .iter()
                                .all(|v| v.to_bits() == sentinel().to_bits())
                        );
                        assert!(
                            y[y.len() - 16..]
                                .iter()
                                .all(|v| v.to_bits() == sentinel().to_bits())
                        );
                        for (i, (x, y)) in x.iter().zip(y).enumerate() {
                            assert!(
                                x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
                                "len{len},compact{compact},phase{phase},element{i}: {x}/{y}"
                            );
                        }
                    }
                }
                for (gpu, host) in [(&w, &wh), (&m, &mh), (&v, &vh)] {
                    let mut after = vec![0f32; len];
                    stream.memcpy_dtoh(gpu, &mut after)?;
                    stream.synchronize()?;
                    assert!(
                        host.iter()
                            .zip(after)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    );
                }
                let mut after_settings = vec![0f32; 10];
                stream.memcpy_dtoh(&settings, &mut after_settings)?;
                stream.synchronize()?;
                assert!(
                    after_settings
                        .iter()
                        .zip(expected_settings)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
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
        if let Some(dir) = std::env::var_os("TNSR_VECTOR_ADAM_DIR") {
            std::fs::create_dir_all(&dir)?;
            std::fs::write(
                std::path::PathBuf::from(dir).join(format!("pattern-len{len}.json")),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"len":len,"all65536patterns":len>=65536,"compact_modes":[false,true],"feeds":8,"replays":2,"exact_nonNaN_bits":true,"NaNclasses":true,"gradient_registers":functions[3].num_regs()?,"shared":functions[3].shared_size_bytes()?,"local":functions[3].local_size_bytes()?,"max_threads":functions[3].max_threads_per_block()?,"source_pointer_alignment":raw.device_ptr(&stream).0%16,"scope":"Requested source widths differ; no allocation/DRAM/time saving measured."}),
                )?,
            )?;
        }
    }
    Ok(())
}

fn sentinel() -> f32 {
    f32::from_bits(0x4355aa55)
}
fn compile_loaded(
    context: &std::sync::Arc<CudaContext>,
    len: usize,
    irs: Vec<crate::tile::TileIR>,
) -> Result<kernels::Kernels> {
    let count = irs.len();
    let mut tiles = crate::tile::TileGraph::from_graph(&crate::graph::TensorGraph::<f32>::new());
    tiles.region_kernels = irs
        .into_iter()
        .enumerate()
        .map(|(region_id, ir)| crate::tile::graph::TileRegionKernel { region_id, ir })
        .collect();
    tiles.optimize_indices()?;
    let graph = crate::ptx::PtxGraph::from(tiles)
        .with_target(crate::ptx::PtxTarget::from_context(context)?);
    let names = (0..count)
        .map(|i| {
            graph
                .region_kernel_name(i)
                .map(str::to_owned)
                .context("missing vector Adam entry")
        })
        .collect::<Result<Vec<_>>>()?;
    let ptx = graph.to_ptx();
    if let Some(dir) = std::env::var_os("TNSR_VECTOR_ADAM_DIR") {
        use std::io::Write;
        std::fs::create_dir_all(&dir)?;
        let path = std::path::PathBuf::from(dir).join(format!("loaded-len{len}.ptx"));
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?
            .write_all(ptx.as_bytes())?;
    }
    let module = context.load_module(cudarc::nvrtc::Ptx::from_src(ptx))?;
    Ok(kernels::Kernels { module, names })
}
