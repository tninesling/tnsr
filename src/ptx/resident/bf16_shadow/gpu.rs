//! Independent GPU cast oracle; no timers, owner publication or model claims.
use super::*;
use crate::{
    graph::{NodeIndex, TensorGraph},
    ptx::{PtxGraph, PtxTarget},
    tile::{TileGraph, VirtualTensor},
};
use anyhow::Context;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, sys,
};
use half::bf16;
use std::sync::Arc;
const GUARD: usize = 16;
fn sentinel() -> f32 {
    f32::from_bits(0x4355aa55)
}
fn bf_sentinel() -> bf16 {
    bf16::from_bits(0x4355)
}
fn same_float(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!(
            a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
            "F32state {i}: {:08x}/{:08x}",
            a.to_bits(),
            b.to_bits()
        );
    }
}
fn download<T: cudarc::driver::DeviceRepr + Clone + Default>(
    s: &Arc<CudaStream>,
    x: &CudaSlice<T>,
) -> Result<Vec<T>> {
    let mut h = vec![T::default(); x.len()];
    s.memcpy_dtoh(x, &mut h)?;
    s.synchronize()?;
    Ok(h)
}
fn ordinary_cast_ir(len: usize) -> Result<TileIR> {
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    let node = tiles.graph.add_node(TileIRBuilder::new().finish());
    let view = VirtualTensor::identity(NodeIndex::new(0), vec![len], DType::F32);
    // This is the existing MixedExecutor materializer lowering, not a clone of
    // initialize_ir or the paired store helper under qualification.
    tiles.set_materialized_view_dtype(node, &view, &[len], DType::BF16)?;
    Ok(tiles.graph[node].clone())
}
fn config(len: usize, kind: AdamKind) -> Result<LaunchConfig> {
    let vector = matches!(kind, AdamKind::Vector4);
    Ok(LaunchConfig {
        grid_dim: (u32::try_from(len.div_ceil(1024))?, 1, 1),
        block_dim: (if vector { 256 } else { 1024 }, 1, 1),
        shared_mem_bytes: 0,
    })
}
fn launch_cast(
    s: &Arc<CudaStream>,
    f: &CudaFunction,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<bf16>,
    len: usize,
) -> Result<()> {
    let mut b = s.launch_builder(f);
    b.arg(input).arg(output);
    // SAFETY: same-stream owned input/output, reference extent <= allocations,
    // no aliases; ordinary lowering bounds-predicates all out-of-range lanes.
    unsafe {
        b.launch(crate::ptx::target::pointwise_launch_config(u32::try_from(
            len,
        )?))
    }?;
    Ok(())
}
#[test]
#[ignore = "manual paired BF16-shadow leaf arithmetic/rounding/tails/replay qualification; no timing"]
fn shadow_paired_adam_and_initialization_all_patterns_tails_health_and_replay() -> Result<()> {
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
    // SAFETY: this fixture owns every allocation/capture and synchronizes its
    // single stream at capture/readback/destruction boundaries. Independent
    // graphs must not acquire automatic uncaptured per-allocation events.
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream()?;
    let output = std::env::var_os("TNSR_SHADOW_LEAF_DIR").map(std::path::PathBuf::from);
    if let Some(p) = &output {
        std::fs::create_dir_all(p)?;
    }
    let mut reports = Vec::new();
    for len in [1usize, 1023, 1024, 65536, 66560, 2818048] {
        for kind in [AdamKind::F32, AdamKind::Bf16First, AdamKind::Vector4] {
            if matches!(kind, AdamKind::Vector4) && !len.is_multiple_of(1024) {
                continue;
            }
            eprintln!("shadow leaf {kind:?} len{len}: untimed qualification");
            let irs = vec![
                kind.original(len)?,
                paired_ir(len, kind)?,
                ordinary_cast_ir(len)?,
                initialize_ir(len)?,
            ];
            for ir in &irs {
                ir.validate_layouts()?;
            }
            let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
            tiles.region_kernels = irs
                .into_iter()
                .enumerate()
                .map(|(region_id, ir)| crate::tile::graph::TileRegionKernel { region_id, ir })
                .collect();
            tiles.optimize_indices()?;
            let graph = PtxGraph::from(tiles).with_target(PtxTarget::from_context(&context)?);
            let names = (0..4)
                .map(|i| {
                    graph
                        .region_kernel_name(i)
                        .map(str::to_owned)
                        .context("shadow leaf entry absent")
                })
                .collect::<Result<Vec<_>>>()?;
            let ptx = graph.to_ptx();
            if let Some(p) = &output {
                std::fs::write(p.join(format!("loaded-{kind:?}-len{len}.ptx")), &ptx)?;
            }
            let module = context.load_module(cudarc::nvrtc::Ptx::from_src(ptx))?;
            let functions = names
                .iter()
                .map(|n| module.load_function(n).map_err(anyhow::Error::from))
                .collect::<Result<Vec<_>>>()?;
            let mut w = stream.memcpy_stod(&vec![sentinel(); len + GUARD])?;
            let mut m = stream.memcpy_stod(&vec![sentinel(); len + GUARD])?;
            let mut v = stream.memcpy_stod(&vec![sentinel(); len + GUARD])?;
            let mut gf = stream.memcpy_stod(&vec![sentinel(); len + GUARD])?;
            let mut gb = stream.memcpy_stod(&vec![bf_sentinel(); len + GUARD])?;
            let mut settings = stream.memcpy_stod(&vec![sentinel(); 10 + GUARD])?;
            let mut old = (0..3)
                .map(|_| stream.memcpy_stod(&vec![sentinel(); len + GUARD]))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let mut new = (0..3)
                .map(|_| stream.memcpy_stod(&vec![sentinel(); len + GUARD]))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let health_len = len.div_ceil(256);
            let mut oh = stream.memcpy_stod(&vec![sentinel(); health_len + GUARD])?;
            let mut nh = stream.memcpy_stod(&vec![sentinel(); health_len + GUARD])?;
            let mut shadow = stream.memcpy_stod(&vec![bf_sentinel(); len + GUARD])?;
            let mut reference = stream.memcpy_stod(&vec![bf_sentinel(); len + GUARD])?;
            let mut initial = stream.memcpy_stod(&vec![bf_sentinel(); len + GUARD])?;
            let mut initial_reference = stream.memcpy_stod(&vec![bf_sentinel(); len + GUARD])?;
            let mut comparisons = 0usize;
            let mut readonly = 0usize;
            for phase in 0..6 {
                // Full65536 BF16 patterns when len>=65536. F32 master feeds
                // independently exercise ties +/-1ULP, signed zeros, subnormal,
                // BF16 overflow thresholds, infinity and payload-bearing NaNs.
                let patterns = [
                    0u32, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x3f807fff,
                    0x3f808000, 0x3f808001, 0x3f818000, 0xbf808000, 0x7f7f7fff, 0x7f7f8000,
                    0x7f7f8001, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345,
                    0xffa12345,
                ];
                let mut wh = vec![sentinel(); len + GUARD];
                let mut mh = wh.clone();
                let mut vh = wh.clone();
                let mut gfh = wh.clone();
                let mut gbh = vec![bf_sentinel(); len + GUARD];
                for i in 0..len {
                    wh[i] = if phase == 1 {
                        f32::MAX
                    } else if phase == 2 {
                        f32::from_bits(patterns[i % patterns.len()])
                    } else {
                        ((i % 17) as f32 - 8.) / 32.
                    };
                    mh[i] = if phase == 1 {
                        0.
                    } else if phase == 3 && i % 251 == 0 {
                        f32::from_bits(0x7fc12345)
                    } else {
                        0.001
                    };
                    vh[i] = if phase == 1 {
                        0.
                    } else if phase == 4 {
                        if i % 2 == 0 { -1. } else { f32::INFINITY }
                    } else {
                        0.01
                    };
                    gbh[i] = if phase == 1 || phase == 2 {
                        bf16::ZERO
                    } else {
                        bf16::from_bits((i as u16).wrapping_add((phase * 1237) as u16))
                    };
                    gfh[i] = if phase == 5 {
                        f32::from_bits(patterns[i % patterns.len()])
                    } else {
                        f32::from(gbh[i])
                    };
                }
                let mut sh = vec![sentinel(); 10 + GUARD];
                sh[..10].copy_from_slice(&[
                    if phase == 1 || phase == 2 { 0. } else { 1e-3 },
                    0.9,
                    0.999,
                    1. - 0.9,
                    1. - 0.999,
                    1. - 0.9,
                    1. - 0.999,
                    1e-8,
                    if phase == 3 { 0.01 } else { 0. },
                    1.,
                ]);
                stream.memcpy_htod(&wh, &mut w)?;
                stream.memcpy_htod(&mh, &mut m)?;
                stream.memcpy_htod(&vh, &mut v)?;
                stream.memcpy_htod(&gfh, &mut gf)?;
                stream.memcpy_htod(&gbh, &mut gb)?;
                stream.memcpy_htod(&sh, &mut settings)?;
                // Check init independently before every arithmetic fixture.
                launch_cast(&stream, &functions[2], &w, &mut initial_reference, len)?;
                launch_cast(&stream, &functions[3], &w, &mut initial, len)?;
                let initialized = download(&stream, &initial)?;
                let cast_initialized = download(&stream, &initial_reference)?;
                assert_eq!(
                    initialized.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    cast_initialized
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>()
                );
                assert!(
                    initialized[len..]
                        .iter()
                        .chain(cast_initialized[len..].iter())
                        .all(|x| x.to_bits() == bf_sentinel().to_bits())
                );
                comparisons += 1;
                let mut launch_pair = || -> Result<()> {
                    for (which, outputs, health) in [(0, &mut old, &mut oh), (1, &mut new, &mut nh)]
                    {
                        let mut b = stream.launch_builder(&functions[which]);
                        b.arg(&w);
                        if matches!(kind, AdamKind::F32) {
                            b.arg(&gf);
                        } else {
                            b.arg(&gb);
                        }
                        b.arg(&m).arg(&v).arg(&settings);
                        for x in outputs {
                            b.arg(x);
                        }
                        b.arg(health);
                        if which == 1 {
                            b.arg(&mut shadow);
                        }
                        // SAFETY: root-declared ABI matches owned disjoint typed
                        // allocations; generated masks cover tails, same stream.
                        unsafe { b.launch(config(len, kind)?) }?;
                    }
                    Ok(())
                };
                // Direct launch plus two graph replays of both variants. No
                // graph mutates inputs; every replay still recomputes outputs.
                launch_pair()?;
                stream.synchronize()?;
                stream
                    .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                let submitted = launch_pair();
                let captured = stream.end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                );
                submitted?;
                let captured = captured?.context("empty shadow leaf capture")?;
                for replay in 0..3 {
                    if replay > 0 {
                        // Re-poison staged outputs before each replay. A missing
                        // graph launch/write cannot pass on stale correct data.
                        for x in &mut new {
                            stream.memcpy_htod(&vec![sentinel(); len + GUARD], x)?;
                        }
                        stream.memcpy_htod(&vec![sentinel(); health_len + GUARD], &mut nh)?;
                        stream.memcpy_htod(&vec![bf_sentinel(); len + GUARD], &mut shadow)?;
                        captured.launch()?;
                    }
                    stream.synchronize()?;
                    for (a, b) in old
                        .iter()
                        .chain(std::iter::once(&oh))
                        .zip(new.iter().chain(std::iter::once(&nh)))
                    {
                        let x = download(&stream, a)?;
                        let y = download(&stream, b)?;
                        same_float(&x, &y);
                        assert!(
                            x[x.len() - GUARD..]
                                .iter()
                                .all(|x| x.to_bits() == sentinel().to_bits())
                        );
                        assert!(
                            y[y.len() - GUARD..]
                                .iter()
                                .all(|x| x.to_bits() == sentinel().to_bits())
                        );
                        comparisons += 1;
                    }
                    launch_cast(&stream, &functions[2], &old[0], &mut reference, len)?;
                    let x = download(&stream, &shadow)?;
                    let y = download(&stream, &reference)?;
                    assert!(
                        x.iter().zip(&y).all(|(x, y)| x.to_bits() == y.to_bits()),
                        "BF16shadow exact bits len{len}, {kind:?},phase{phase},replay{replay}"
                    );
                    assert!(
                        x[len..]
                            .iter()
                            .all(|x| x.to_bits() == bf_sentinel().to_bits())
                    );
                    comparisons += 1;
                    if phase == 1 {
                        let weights = download(&stream, &old[0])?;
                        let health = download(&stream, &oh)?;
                        assert!(weights[..len].iter().all(|x| x.is_finite()));
                        assert!(health[..health_len].iter().all(|x| x.to_bits() == 0));
                        assert!(x[..len].iter().all(|x| x.is_infinite()));
                    }
                }
                drop(captured);
                for (gpu, expected) in [
                    (&w, &wh),
                    (&m, &mh),
                    (&v, &vh),
                    (&gf, &gfh),
                    (&settings, &sh),
                ] {
                    let actual = download(&stream, gpu)?;
                    assert!(
                        actual
                            .iter()
                            .zip(expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    );
                    readonly += 1;
                }
                assert!(
                    download(&stream, &gb)?
                        .iter()
                        .zip(&gbh)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                readonly += 1;
            }
            reports.push(serde_json::json!({"kind":format!("{kind:?}"),"len":len,"all65536BF16gradientpatterns":len>=65536&&!matches!(kind,AdamKind::F32),"direct_plus_graph_replays":true,"feeds":6,"array_comparisons":comparisons,"readonly_checks":readonly,"finiteF32_to_BF16_infinity_keeps_F32_health":true,"paired_register_source_exact":true,"shadow_BF16_bits_equal_ordinary_materializer":true,"registers":functions[1].num_regs()?,"shared_bytes":functions[1].shared_size_bytes()?,"local_bytes":functions[1].local_size_bytes()?,"scope":"Leaf only: no owner integration, model, timers, capacity or coalescing claim."}));
        }
    }
    if let Some(p) = output {
        std::fs::write(
            p.join("qualification.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"qualified":true,"cases":reports,"scope":"No performance or model claim"}),
            )?,
        )?;
    }
    Ok(())
}
