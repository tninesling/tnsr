//! Isolated untimed leaf: existing TileGraph/production reducer expression only.
use super::super::rectangle128;
use super::*;
use crate::tile::ir::MatrixLayout;
use crate::tile::{Dim, Expr, TileIRBuilder};
use anyhow::{Context, ensure};
use cudarc::driver::DevicePtr;
const GUARD: usize = 32;
fn reducer(len: usize) -> anyhow::Result<TileIR> {
    let mut ir = TileIRBuilder::new();
    ir.start_kernel("split_bf16_sum");
    ir.add_param("partial", DType::F32, true);
    ir.add_param("output", DType::BF16, false);
    ir.bounds_check(len);
    let index = Expr::Mul(
        Box::new(Expr::BlockIdx(Dim::X)),
        Box::new(Expr::BlockDim(Dim::X)),
    ) + Expr::ThreadIdx(Dim::X);
    let layout = MatrixLayout {
        rows: 1,
        cols: 4 * len,
        row_stride: 4 * len,
    };
    let mut sum = None;
    for part in 0..4 {
        let x = ir.alloc_register(DType::F32, 1, 1);
        ir.load_global_predicated(
            x,
            "partial",
            index.clone() + Expr::Const(i64::try_from(part * len)?),
            Expr::Const(0),
            index.clone(),
            layout,
        );
        sum = Some(if let Some(old) = sum {
            let next = ir.alloc_register(DType::F32, 1, 1);
            ir.add(next, old, x);
            next
        } else {
            x
        });
    }
    ir.store(
        "output",
        sum.context("missing reducer sum")?,
        index,
        Expr::Const(0),
    );
    Ok(ir.finish())
}
fn source(irs: Vec<TileIR>, target: PtxTarget) -> anyhow::Result<(String, Vec<String>)> {
    let count = irs.len();
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.region_kernels = irs
        .into_iter()
        .enumerate()
        .map(|(region_id, ir)| crate::tile::graph::TileRegionKernel { region_id, ir })
        .collect();
    tiles.optimize_indices()?;
    let ptx = PtxGraph::from(tiles).with_target(target);
    let names = (0..count)
        .map(|i| ptx.region_kernel_name(i).unwrap().to_owned())
        .collect();
    Ok((ptx.module_source(), names))
}
fn launch_partial(
    stream: &Arc<CudaStream>,
    f: &CudaFunction,
    a: &cudarc::driver::CudaSlice<bf16>,
    b: &cudarc::driver::CudaSlice<bf16>,
    out: &mut cudarc::driver::CudaSlice<f32>,
    shape: (usize, usize),
    tile_m: usize,
) -> anyhow::Result<()> {
    let (m, n) = shape;
    let mut l = stream.launch_builder(f);
    l.arg(a).arg(b).arg(out);
    // SAFETY: exact typed whole-source maps, disjoint output, validated fulltile
    // grid [N/64,M/tileM,4], ordered private stream and retained allocations.
    unsafe {
        l.launch(LaunchConfig {
            grid_dim: (u32::try_from(n / 64)?, u32::try_from(m / tile_m)?, 4),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        })
    }?;
    Ok(())
}
fn launch_reduce(
    stream: &Arc<CudaStream>,
    f: &CudaFunction,
    p: &cudarc::driver::CudaSlice<f32>,
    out: &mut cudarc::driver::CudaSlice<bf16>,
    len: usize,
) -> anyhow::Result<()> {
    let mut l = stream.launch_builder(f);
    l.arg(p).arg(out);
    // SAFETY: identical production left4 reducer, whole4*len F32 input and len
    // BF16 output with bounds guard, all allocations alive on this stream.
    unsafe {
        l.launch(LaunchConfig {
            grid_dim: (u32::try_from(len.div_ceil(256))?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    Ok(())
}
#[test]
#[ignore = "root-reserved isolated split128 resource/math trial; no timing"]
fn split128_preserves_four_f32_partials_left4_bf16_and_poisoned_replay() -> anyhow::Result<()> {
    let context = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            ensure!(
                std::env::var_os("TNSR_REQUIRE_CUDA").is_none(),
                "CUDA required:{e}"
            );
            return Ok(());
        }
    };
    // SAFETY: private ordered stream; arrays/modules/captures retained and all
    // writes synchronize before host readback, upload or destruction.
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream()?;
    let target = PtxTarget::from_context(&context)?;
    let root = std::path::PathBuf::from(
        std::env::var_os("TNSR_SPLIT128_DIR").context("set fresh split128 output")?,
    );
    std::fs::create_dir_all(&root)?;
    ensure!(
        !root.join("qualification.json").exists(),
        "fresh proof output required"
    );
    let mut reports = vec![];
    // Represents all17 selected nodes: tallM, wideN1408 and widestN4096;
    // N1536 is the same physical orientation/divisibility, not an extra sweep.
    for (m, n, k) in [
        (512usize, 1408usize, 4096usize),
        (1408, 512, 4096),
        (512, 4096, 4096),
    ] {
        let r = fixture(m, n, k);
        let c = SplitKBf16TilePolicy::Square64
            .candidate(&r, 4, 4 * k, target.matmul_resources, true)
            .context("current Square64 proof")?;
        let (_, old) = c
            .partial_ir(&r, target.matmul_resources, true)
            .context("current partial")?;
        let (_, new) = rectangle128::candidate(&r, target.matmul_resources, true)?;
        let (old_source, _) = source(vec![old.clone()], target)?;
        let (ptx, names) = source(vec![old, new, reducer(m * n)?], target)?;
        let boundary = ptx
            .find(&format!(".visible .entry {}", names[1]))
            .context("candidate entry")?;
        ensure!(
            ptx[..boundary].trim_end() == old_source.trim_end(),
            "canonical prefix changed"
        );
        let key = format!("m{m}-n{n}-k{k}");
        std::fs::write(root.join(format!("{key}-loaded.ptx")), &ptx)?;
        std::fs::write(root.join(format!("{key}-canonical.ptx")), &old_source)?;
        let module = context.load_module(Ptx::from_src(ptx))?;
        let functions = names
            .iter()
            .map(|name| module.load_function(name))
            .collect::<Result<Vec<_>, _>>()?;
        let resources=functions.iter().zip(&names).map(|(f,name)|->anyhow::Result<_>{Ok(serde_json::json!({"name":name,"registers":f.num_regs()?,"shared_bytes":f.shared_size_bytes()?,"local_bytes":f.local_size_bytes()?,"max_threads":f.max_threads_per_block()?,"capacity256":f.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?}))}).collect::<anyhow::Result<Vec<_>>>()?;
        std::fs::write(
            root.join(format!("{key}-resources.json")),
            serde_json::to_vec_pretty(&resources)?,
        )?;
        ensure!(
            functions[1].num_regs()? <= 128
                && functions[1].shared_size_bytes()? == 32768
                && functions[1].local_size_bytes()? == 0
                && functions[1].max_threads_per_block()? >= 256
                && functions[1].occupancy_max_active_blocks_per_multiprocessor(256, 0, None)? >= 2,
            "resource gate rejected; raw evidence retained"
        );
        let plane = m * n;
        let asize = 4 * k * m;
        let bsize = 4 * k * n;
        let mut a = stream.alloc_zeros::<bf16>(asize + GUARD)?;
        let mut b = stream.alloc_zeros::<bf16>(bsize + GUARD)?;
        let mut p0 = stream.alloc_zeros::<f32>(4 * plane + GUARD)?;
        let mut p1 = stream.alloc_zeros::<f32>(4 * plane + GUARD)?;
        let mut y0 = stream.alloc_zeros::<bf16>(plane + GUARD)?;
        let mut y1 = stream.alloc_zeros::<bf16>(plane + GUARD)?;
        let spans = [
            (a.device_ptr(&stream).0, a.len() * 2),
            (b.device_ptr(&stream).0, b.len() * 2),
            (p0.device_ptr(&stream).0, p0.len() * 4),
            (p1.device_ptr(&stream).0, p1.len() * 4),
            (y0.device_ptr(&stream).0, y0.len() * 2),
            (y1.device_ptr(&stream).0, y1.len() * 2),
        ];
        for (i, (start, size)) in spans.iter().enumerate() {
            ensure!(*start % 16 == 0, "alignment");
            for (other, bytes) in &spans[..i] {
                ensure!(
                    start
                        .checked_add(u64::try_from(*size)?)
                        .context("range overflow")?
                        <= *other
                        || other
                            .checked_add(u64::try_from(*bytes)?)
                            .context("range overflow")?
                            <= *start,
                    "alias"
                );
            }
        }
        std::fs::write(
            root.join(format!("{key}-bindings.json")),
            serde_json::to_vec_pretty(
                &serde_json::json!({"spans":spans,"typed_inputs":"BF16,BF16","typed_partials":"F32,F32","typed_outputs":"BF16,BF16","disjoint":true,"grid64":[n/64,m/64,4],"grid128":[n/64,m/128,4]}),
            )?,
        )?;
        let poison_p = vec![f32::from_bits(0x7fc12345); 4 * plane + GUARD];
        let poison_y = vec![bf16::from_bits(0x7fc1); plane + GUARD];
        let mut comparisons = 0usize;
        let mut readonly = 0usize;
        let mut oracle_checks = 0usize;
        macro_rules! submit {
            () => {{
                (|| -> anyhow::Result<()> {
                    launch_partial(&stream, &functions[0], &a, &b, &mut p0, (m, n), 64)?;
                    launch_partial(&stream, &functions[1], &a, &b, &mut p1, (m, n), 128)?;
                    launch_reduce(&stream, &functions[2], &p0, &mut y0, plane)?;
                    launch_reduce(&stream, &functions[2], &p1, &mut y1, plane)
                })()
            }};
        }
        stream.begin_capture(
            cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )?;
        let submitted = submit!();
        let graph=stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
        submitted?;
        let graph = graph?.context("empty graph")?;
        for phase in 0..3 {
            let mut av = (0..asize + GUARD)
                .map(|i| bf16::from_f32(((i + phase * 3) % 23) as f32 / 64. - 0.125))
                .collect::<Vec<_>>();
            let mut bv = (0..bsize + GUARD)
                .map(|i| bf16::from_f32(((i + phase) % 19) as f32 / 32. - 0.25))
                .collect::<Vec<_>>();
            if phase == 1 {
                av[..asize].fill(bf16::from_bits(0x8000));
                bv[..bsize].fill(bf16::ONE);
            }
            if phase == 2 {
                for part in 0..4 {
                    av[part * k * m + part] = bf16::INFINITY;
                    av[part * k * m + m + part] = bf16::NAN;
                    bv[part * k * n + part] = bf16::ZERO;
                    bv[part * k * n + n + part] = bf16::NEG_INFINITY;
                }
            }
            stream.memcpy_htod(&av, &mut a)?;
            stream.memcpy_htod(&bv, &mut b)?;
            for replay in 0..3 {
                stream.memcpy_htod(&poison_p, &mut p0)?;
                stream.memcpy_htod(&poison_p, &mut p1)?;
                stream.memcpy_htod(&poison_y, &mut y0)?;
                stream.memcpy_htod(&poison_y, &mut y1)?;
                if replay == 0 {
                    submit!()?;
                } else {
                    graph.launch()?;
                }
                stream.synchronize()?;
                let pv0 = stream.memcpy_dtov(&p0)?;
                let pv1 = stream.memcpy_dtov(&p1)?;
                let yv0 = stream.memcpy_dtov(&y0)?;
                let yv1 = stream.memcpy_dtov(&y1)?;
                stream.synchronize()?;
                if phase < 2 {
                    ensure!(
                        pv0[..4 * plane]
                            .iter()
                            .chain(&pv1[..4 * plane])
                            .all(|v| v.is_finite())
                            && yv0[..plane]
                                .iter()
                                .chain(&yv1[..plane])
                                .all(|v| v.is_finite()),
                        "finite feed left NaN poison/nonfinite outputs"
                    );
                }
                if phase == 0 {
                    for part in 0..4 {
                        for row in [0, m / 2, m - 1] {
                            for col in [0, n / 2, n - 1] {
                                let expected = (0..k)
                                    .map(|kk| {
                                        f64::from(av[(part * k + kk) * m + row].to_f32())
                                            * f64::from(bv[(part * k + kk) * n + col].to_f32())
                                    })
                                    .sum::<f64>();
                                let index = part * plane + row * n + col;
                                ensure!(
                                    pv0[index].to_bits() == (expected as f32).to_bits()
                                        && pv1[index].to_bits() == (expected as f32).to_bits(),
                                    "sampled independent dyadic F64 partial mismatch"
                                );
                                oracle_checks += 1;
                            }
                        }
                    }
                }
                for (&x, &y) in pv0[..4 * plane].iter().zip(&pv1[..4 * plane]) {
                    same(x, y);
                }
                for part in 0..4 {
                    ensure!(
                        pv0[part * plane..(part + 1) * plane]
                            .iter()
                            .any(|v| v.to_bits() != 0x7fc12345),
                        "poison remained entire plane"
                    );
                }
                for i in 0..plane {
                    let expected = left_four(&pv0, i, plane);
                    let x = yv0[i];
                    let y = yv1[i];
                    ensure!(
                        (x.to_bits() == y.to_bits() || x.is_nan() && y.is_nan())
                            && (x.to_bits() == expected.to_bits()
                                || x.is_nan() && expected.is_nan()),
                        "reducer bits/classes mismatch"
                    );
                }
                ensure!(
                    pv0[4 * plane..]
                        .iter()
                        .chain(&pv1[4 * plane..])
                        .all(|x| x.to_bits() == 0x7fc12345),
                    "partial guard"
                );
                ensure!(
                    yv0[plane..]
                        .iter()
                        .chain(&yv1[plane..])
                        .all(|x| x.to_bits() == 0x7fc1),
                    "BF16 guard"
                );
                comparisons += 6;
            }
            let actual_a = stream.memcpy_dtov(&a)?;
            let actual_b = stream.memcpy_dtov(&b)?;
            stream.synchronize()?;
            ensure!(
                actual_a
                    .iter()
                    .zip(&av)
                    .all(|(x, y)| x.to_bits() == y.to_bits())
                    && actual_b
                        .iter()
                        .zip(&bv)
                        .all(|(x, y)| x.to_bits() == y.to_bits()),
                "readonly input/guard changed"
            );
            readonly += 2;
        }
        reports.push(serde_json::json!({"m":m,"n":n,"part_k":k,"resources":resources,"full_array_comparisons":comparisons,"readonly_arrays":readonly,"feeds":3,"direct_launches_per_feed":1,"poisoned_capture_replays_per_feed":2,"same_graph_reused_across_all_feeds":true,"capture_count":1,"finite_phases_fully_finite":true,"independent_F64_dyadic_sample_checks":oracle_checks,"four_F32_partial_planes_exact":true,"left4_single_BF16_RN_exact":true,"canonical_prefix_exact":true}));
    }
    std::fs::write(
        root.join("qualification.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"qualified_math":true,"timing_performed":false,"scope":"Isolated CFGTEST split128x64; no production adoption","cases":reports}),
        )?,
    )?;
    Ok(())
}

#[test]
fn split128_canonical_ptx_prefix_and_actual_reducer_left_association() -> anyhow::Result<()> {
    let target = PtxTarget::sm80();
    for (m, n, k) in [(512, 1408, 4096), (1408, 512, 4096), (512, 4096, 4096)] {
        let r = fixture(m, n, k);
        let c = SplitKBf16TilePolicy::Square64
            .candidate(&r, 4, 4 * k, resources(), true)
            .unwrap();
        let (_, old) = c.partial_ir(&r, resources(), true).unwrap();
        let (_, new) = rectangle128::candidate(&r, resources(), true)?;
        let (before, _) = source(vec![old.clone()], target)?;
        let (combined, names) = source(vec![old, new, reducer(m * n)?], target)?;
        let marker = format!(".visible .entry {}", names[1]);
        assert_eq!(
            combined[..combined.find(&marker).unwrap()].trim_end(),
            before.trim_end()
        );
        let reducer = reducer(m * n)?;
        assert_eq!(
            reducer.params.iter().map(|p| p.dtype).collect::<Vec<_>>(),
            [DType::F32, DType::BF16]
        );
        let adds = reducer
            .body
            .stmts
            .iter()
            .filter_map(|s| {
                if let crate::tile::Stmt::Add { dest, a, b } = s {
                    Some((*dest, *a, *b))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(adds.len(), 3);
        assert_eq!(adds[1].1, adds[0].0);
        assert_eq!(adds[2].1, adds[1].0);
    }
    Ok(())
}
