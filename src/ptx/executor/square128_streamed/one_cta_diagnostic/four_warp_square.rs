//! Test-only four-warp square. Production Square policy remains eight warps.
mod wide_exchange;
use super::*;
use crate::tile::{Dim, Expr, MatMulRegion, MatMulResources, Stmt};

fn tile_graph(region: MatMulRegion) -> Result<TileGraph> {
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.add_matmul_regions(&[region])?;
    Ok(tiles)
}
fn four_square_region(
    mut region: MatMulRegion,
    resources: MatMulResources,
    supports_async: bool,
) -> Result<MatMulRegion> {
    ensure!(
        region.batch_shape.is_empty()
            && region.epilogue_operations.is_empty()
            && region.outputs.len() == 1
            && region.outputs[0].value == region.matmul_value
            && region.m.is_multiple_of(128)
            && region.n.is_multiple_of(128)
            && region.k >= 512
            && region.k.is_multiple_of(32),
        "unsupported square scope"
    );
    for (v, shape) in [
        (region.lhs_value, &region.lhs_shape),
        (region.rhs_value, &region.rhs_shape),
    ] {
        let input = region
            .inputs
            .iter()
            .find(|i| i.value == v)
            .context("operand absent")?;
        ensure!(
            crate::tile::ordinary_bf16_block::aligned_operand(input, shape, false).is_some(),
            "unproved aligned NN source"
        );
    }
    region.schedule = region
        .schedule
        .bf16_square128_streamed_geometry()
        .context("nonBF16")?;
    region.schedule.block_threads = (32, 4, 1);
    region.schedule.warp_topology = (2, 2);
    region.schedule.warp_fragments = (4, 4);
    region.configure_pipeline(
        MatMulSharedLayout::Padded,
        MatMulPipeline::Auto,
        supports_async,
        resources,
    )?;
    let s = region.schedule;
    ensure!(
        s.supports_streamed_fragment_windows()
            && !s.is_square128_streamed()
            && s.pipeline_stages == 2
            && s.async_operands == (true, true)
            && s.copy_bytes == 16
            && s.reuse_shared_memory
            && s.shared_mem_bytes() == 37888
            && s.accumulator_exchange_bytes() == 8192,
        "four-warp schedule proof failed"
    );
    Ok(region)
}
fn eval(expr: &Expr, thread: usize, block: (usize, usize), k: usize) -> u64 {
    let get_dim = |dim: &Dim, x: usize, y: usize, z: usize| match dim {
        Dim::X => x as u64,
        Dim::Y => y as u64,
        Dim::Z => z as u64,
    };
    match expr {
        Expr::Const(v) => *v as u64,
        Expr::Var(v) => {
            assert!(v == "k_tile" || v.starts_with("square_output_round_"));
            k as u64
        }
        Expr::ThreadIdx(d) => get_dim(d, thread % 32, thread / 32, 0),
        Expr::BlockIdx(d) => get_dim(d, block.1, block.0, 0),
        Expr::BlockDim(d) => get_dim(d, 32, 4, 1),
        Expr::Add(a, b) => eval(a, thread, block, k).wrapping_add(eval(b, thread, block, k)),
        Expr::Sub(a, b) => eval(a, thread, block, k).wrapping_sub(eval(b, thread, block, k)),
        Expr::Mul(a, b) => eval(a, thread, block, k).wrapping_mul(eval(b, thread, block, k)),
        Expr::FloorDiv(a, b) => eval(a, thread, block, k) / (*b as u64),
        Expr::Mod(a, b) => eval(a, thread, block, k) % (*b as u64),
        Expr::ShiftRight(a, b) => eval(a, thread, block, k) >> *b,
        Expr::BitAnd(a, b) => eval(a, thread, block, k) & *b,
    }
}

#[test]
fn four_square_coordinates_copy_k16_and_drain_proofs() -> Result<()> {
    let resources = MatMulResources {
        shared_bytes_per_sm: 100 * 1024,
        threads_per_sm: 1536,
        ..MatMulResources::default()
    };
    for case in CASES {
        let base = super::k64_diagnostic::fixture(case)?;
        let region = four_square_region(base, resources, true)?;
        let s = region.schedule;
        let tiles = tile_graph(region)?;
        let ir = &tiles.matmul_region_kernels[0].ir;
        ir.validate_layouts()?;
        let windows = ir
            .body
            .stmts
            .iter()
            .filter_map(|x| match x {
                Stmt::StoreAccumulatorWindow {
                    fragment_origin, ..
                } => Some(*fragment_origin),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            windows,
            (0..4)
                .flat_map(|m| [0, 2].map(|n| (m, n)))
                .collect::<Vec<_>>()
        );
        let mut covered = vec![false; 128 * 128];
        let output_loops = ir
            .body
            .stmts
            .iter()
            .filter_map(|s| match s {
                Stmt::ForLoop { loop_var, body, .. }
                    if loop_var.starts_with("square_output_round_") =>
                {
                    Some(body)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(output_loops.len(), 8);
        for (window, (m, n)) in windows.into_iter().enumerate() {
            let (row, col, global_index) = output_loops[window]
                .stmts
                .iter()
                .find_map(|s| match s {
                    Stmt::StoreGlobalPredicated {
                        row,
                        col,
                        element_index,
                        ..
                    } => Some((row, col, element_index)),
                    _ => None,
                })
                .context("window global output absent")?;
            for round in 0..16 {
                for thread in 0..128 {
                    let index = round * 128 + thread;
                    let (sr, sc) = (index / 64, index % 64);
                    let r = sr / 16 * 64 + m * 16 + sr % 16;
                    let c = sc / 32 * 64 + n * 16 + sc % 32;
                    assert!(r < 128 && c < 128);
                    assert_eq!(
                        (
                            eval(row, thread, (0, 0), round),
                            eval(col, thread, (0, 0), round)
                        ),
                        (r as u64, c as u64)
                    );
                    assert_eq!(
                        eval(global_index, thread, (0, 0), round),
                        (r * case.n + c) as u64
                    );
                    assert!(!covered[r * 128 + c]);
                    covered[r * 128 + c] = true;
                }
            }
        }
        assert!(covered.into_iter().all(|x| x));
        let (end, body) = ir
            .body
            .stmts
            .iter()
            .find_map(|x| match x {
                Stmt::ForLoop {
                    loop_var,
                    end,
                    body,
                    ..
                } if loop_var == "k_tile" => Some((end, body)),
                _ => None,
            })
            .context("Kloop absent")?;
        assert_eq!(*end, Expr::Const((case.k / 32) as i64));
        let initial = ir
            .body
            .stmts
            .iter()
            .filter(|s| matches!(s, Stmt::AsyncCopy { .. }))
            .collect::<Vec<_>>();
        let next = body
            .stmts
            .iter()
            .filter(|s| matches!(s, Stmt::AsyncCopy { .. }))
            .collect::<Vec<_>>();
        let stages = std::iter::once((&initial, 0, false))
            .chain((0..case.k / 32).map(|k| (&next, k, k + 1 == case.k / 32)));
        for (stmts, ks, sentinel) in stages {
            for lhs in [true, false] {
                let mut local = vec![false; 4096];
                let (rows, cols) = if lhs { (128, 32) } else { (32, 128) };
                for stmt in stmts {
                    let Stmt::AsyncCopy {
                        src_param,
                        tile_row,
                        tile_col,
                        row,
                        col,
                        element_index,
                        layout,
                        copy_bytes,
                        ..
                    } = stmt
                    else {
                        unreachable!()
                    };
                    if (src_param == "input_0") != lhs {
                        continue;
                    }
                    assert_eq!(*copy_bytes, 16);
                    for thread in 0..128 {
                        let (r, c) = (
                            eval(tile_row, thread, (case.m / 128 - 1, case.n / 128 - 1), ks)
                                as usize,
                            eval(tile_col, thread, (case.m / 128 - 1, case.n / 128 - 1), ks)
                                as usize,
                        );
                        assert!(r < rows && c + 7 < cols && c % 8 == 0);
                        for lane in 0..8 {
                            assert!(!local[r * cols + c + lane]);
                            local[r * cols + c + lane] = true;
                        }
                        let (gr, gc) = (
                            eval(row, thread, (case.m / 128 - 1, case.n / 128 - 1), ks),
                            eval(col, thread, (case.m / 128 - 1, case.n / 128 - 1), ks),
                        );
                        let valid = gr < layout.rows as u64 && gc < layout.cols as u64;
                        assert_eq!(valid, !sentinel);
                        if valid {
                            assert!(gc + 7 < layout.cols as u64);
                            assert_eq!(
                                eval(
                                    element_index,
                                    thread,
                                    (case.m / 128 - 1, case.n / 128 - 1),
                                    ks
                                ),
                                gr * layout.row_stride as u64 + gc
                            );
                        }
                    }
                }
                assert!(local.into_iter().all(|x| x));
            }
        }
        let mut t = tiles;
        t.optimize_indices()?;
        let mut target = PtxTarget::sm80();
        target.compute_capability = (8, 9);
        target.matmul_resources = resources;
        let source = PtxGraph::from(t).with_target(target).module_source();
        assert_eq!(source.matches("wmma.mma.sync").count(), 32);
        assert_eq!(source.matches("wmma.load.a.sync").count(), 8);
        assert_eq!(source.matches("wmma.load.b.sync").count(), 8);
        let first = source
            .find("wmma.store.d.sync")
            .context("window store absent")?;
        assert!(source[..first].rfind("cp.async.wait_group 0").is_some());
        assert!(source[..first].rfind("bar.sync").is_some());
        assert_eq!(s.warp_fragments.0 * s.warp_fragments.1 * 8, 128);
    }
    Ok(())
}
fn build_four_square(
    case: Case,
    stream: &Arc<CudaStream>,
    a: &[bf16],
    b: &[bf16],
    dir: &Path,
) -> Result<Mode> {
    let graph: TensorGraph<bf16> = TensorExpr::input("a", vec![case.m, case.k])
        .matmul(TensorExpr::input("b", vec![case.k, case.n]))
        .into();
    let mut e = PtxExecutor::<BF16>::with_stream(
        stream.clone(),
        PtxReductionMode::DeterministicTree,
        Arc::new(Mutex::new(())),
    )?;
    e.set_matmul_precision(MatMulPrecision::NativeHalf);
    e.set_matmul_shared_layout(MatMulSharedLayout::Padded);
    e.set_matmul_pipeline(MatMulPipeline::Auto);
    e.set_experimental_shared_memory_reuse(true);
    e.set_experimental_medium_bf16_tile_policy(MediumBf16TilePolicy::SquareThenTall);
    e.compile(&graph)?;
    let plan = e.execution_plan().context("baseline plan absent")?;
    let (_, step) = schedule(plan)?;
    let PtxPlanAction::MatMulRegion(id) = step.action else {
        anyhow::bail!("diagnostic requires selected plain region")
    };
    let region = four_square_region(
        plan.matmul_regions()[id].clone(),
        e.target.matmul_resources,
        e.target.supports_async_copy(),
    )?;
    let selected = region.schedule;
    let mut tiles = tile_graph(region)?;
    tiles.optimize_indices()?;
    let ptx = PtxGraph::from(tiles).with_target(e.target);
    let source = ptx.module_source();
    let module = stream.context().load_module(Ptx::from_src(&source))?;
    let name = ptx
        .matmul_region_kernel_name(0)
        .context("four-warp Square function absent")?;
    let function = module.load_function(name)?;
    let (registers, local, shared, max_threads, capacity) = (
        function.num_regs()?,
        function.local_size_bytes()?,
        function.shared_size_bytes()?,
        function.max_threads_per_block()?,
        function.occupancy_max_active_blocks_per_multiprocessor(128, 0, None)?,
    );
    let accepted = registers > 0
        && registers <= 255
        && local == 0
        && shared > 0
        && shared <= 48 * 1024
        && max_threads >= 128
        && capacity >= 2;
    let resources = json!({"mode":"test_only_four_square","function":name,"registers_per_thread":registers,"local_bytes":local,"shared_bytes":shared,"max_threads_per_block":max_threads,"static_active_block_capacity":capacity,"capacity_is_not_measured_occupancy":true,"schedule":format!("{selected:?}"),"required_resource_gate_passed":accepted,"production_selection_unchanged":true,"production_activation_claim":false});
    std::fs::write(dir.join("test_only_four_square.ptx"), source)?;
    write_json(
        &dir.join("test_only_four_square-resources.json"),
        &resources,
    )?;
    ensure!(
        accepted,
        "four-warp Square actual resource admission failed: {resources}"
    );
    ensure!(
        shared == 37888,
        "four-warp Square physical shared accounting mismatch"
    );
    let storage = vec![
        stream.memcpy_stod(a)?,
        stream.memcpy_stod(b)?,
        stream.alloc_zeros::<bf16>(case.m * case.n)?,
    ];
    let pointers = storage.iter().map(|s| s.device_ptr(stream).0).collect();
    Ok(Mode {
        capture: None,
        function,
        _module: module,
        storage,
        pointers,
        config: LaunchConfig {
            grid_dim: (
                u32::try_from(case.n / 128)?,
                u32::try_from(case.m / 128)?,
                1,
            ),
            block_dim: selected.block_threads,
            shared_mem_bytes: 0,
        },
        schedule: selected,
        name: "test_only_four_square",
        resources,
        samples_us: Vec::new(),
    })
}

#[test]
#[ignore = "manual test-only four-warp Square paired component diagnostic; no production selector change"]
fn four_square_component_resource_equivalence_and_paired_timing() -> Result<()> {
    let directory = PathBuf::from(
        std::env::var_os("TNSR_FOUR_SQUARE_DIAG_DIR")
            .context("set explicit four-warp Square artifact directory")?,
    );
    std::fs::create_dir_all(&directory)?;
    let reverse = std::env::var("TNSR_FOUR_SQUARE_DIAG_REVERSE").as_deref() == Ok("1");
    let timing = std::env::var("TNSR_FOUR_SQUARE_DIAG_TIMING").as_deref() == Ok("1");
    let repeats = 32;
    let samples = 32;
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let mut cases = CASES.to_vec();
    if reverse {
        cases.reverse();
    }
    let mut results = Vec::new();
    for case in cases {
        let dir = directory.join(case.name);
        std::fs::create_dir_all(&dir)?;
        let a = (0..case.m * case.k)
            .map(|i| bf16::from_f32(((i % 97 * 13 % 97) as f32 - 48.) / 97.))
            .collect::<Vec<_>>();
        let b = (0..case.k * case.n)
            .map(|i| bf16::from_f32(((i % 89 * 17 % 89) as f32 - 44.) / 89.))
            .collect::<Vec<_>>();
        let mut modes = Vec::new();
        for candidate in if reverse {
            [true, false]
        } else {
            [false, true]
        } {
            let mode = if candidate {
                build_four_square(case, &stream, &a, &b, &dir)
            } else {
                build(case, false, &stream, &a, &b, &dir)
            };
            match mode {
                Ok(m) => modes.push(m),
                Err(error) => {
                    write_json(
                        &dir.join("result.json"),
                        &json!({"status":"stopped_resource_or_compile","error":format!("{error:#}"),"timing_performed":false}),
                    )?;
                    return Err(error);
                }
            }
        }
        let original =
            PathBuf::from("target/paper-kernels/native-vs-torch-bf16-mm/campaign/native0")
                .join(case.name)
                .join("accepted_baseline.ptx");
        if original.exists() {
            ensure!(
                std::fs::read(original)? == std::fs::read(dir.join("accepted_baseline.ptx"))?,
                "accepted K32 baseline PTX changed"
            );
        }
        let (expected, checks) = qualify(case, &mut modes, &stream, &a, &b)?;
        let mut pairs = Vec::new();
        let mut warmups = 0;
        let mut warmup_ms = 0.;
        if timing {
            for mode in &mut modes {
                mode.capture(&stream, repeats)?;
            }
            let began = Instant::now();
            while began.elapsed() < Duration::from_millis(500) {
                for i in if warmups % 2 == 0 { [0, 1] } else { [1, 0] } {
                    modes[i].capture.as_mut().unwrap().launch()?;
                }
                stream.synchronize()?;
                warmups += 1;
            }
            warmup_ms = began.elapsed().as_secs_f64() * 1000.;
            for sample in 0..samples {
                let mut baseline = 0.;
                let mut candidate = 0.;
                let mut order = Vec::new();
                for i in if sample % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let mode = &mut modes[i];
                    let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                    mode.capture.as_mut().unwrap().launch()?;
                    let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                    let us = f64::from(start.elapsed_ms(&end)?) * 1000. / repeats as f64;
                    ensure!(us > 0. && us.is_finite(), "invalid event interval");
                    mode.samples_us.push(us);
                    order.push(mode.name);
                    if mode.schedule.block_threads.1 == 4 {
                        candidate = us;
                    } else {
                        baseline = us;
                    }
                }
                pairs.push(json!({"sample":sample,"execution_order":order,"accepted_k32_us":baseline,"test_only_four_square_us":candidate,"baseline_over_candidate":baseline/candidate}));
            }
            stream.synchronize()?;
            for mode in &modes {
                compare(
                    &expected,
                    &mode.read(&stream)?,
                    "four-warp Square post-capture finite",
                )?;
            }
        }
        let summaries=modes.iter().map(|mode|json!({"mode":mode.name,"resources":mode.resources,"samples_us":mode.samples_us,"median_us":if timing{Some(median(&mode.samples_us))}else{None}})).collect::<Vec<_>>();
        let result = json!({"status":"qualified_test_fixture_only","case":case.name,"m":case.m,"n":case.n,"k":case.k,"orientation":"NN","qualification":checks,"timing_performed":timing,"post_capture_bits_equal":timing.then_some(true),"samples":if timing{samples}else{0},"graph_kernel_repeats":repeats,"warmup_pairs":warmups,"warmup_elapsed_ms":warmup_ms,"pairs":pairs,"summaries":summaries,"production_selector_changed":false,"scope":"Accepted64²K32 versus test-only128²K32 four-warp Square. Four compute/block warps,4x4fragments,FragmentWindows1x2,StreamB,F32K16order/BF16RN unchanged; not complete updates"});
        write_json(&dir.join("result.json"), &result)?;
        results.push(result);
        write_json(
            &directory.join("report.json"),
            &json!({"cases":results,"construction_order":if reverse{"candidate,baseline"}else{"baseline,candidate"},"scope":"test-only four-warp Square diagnostic; no production selector/API/cache/default changes"}),
        )?;
    }
    Ok(())
}
