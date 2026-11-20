//! Test-only exchange(2,4). All MMA/copies/geometry match the existing4warp probe.
use super::*;
use crate::tile::{DType, MatMulAccumulatorExchange, MatMulTile, SharedLayout, VirtualTensor};

fn resources() -> MatMulResources {
    MatMulResources {
        shared_bytes_per_sm: 100 * 1024,
        threads_per_sm: 1536,
        ..MatMulResources::default()
    }
}
fn target() -> PtxTarget {
    let mut t = PtxTarget::sm80();
    t.compute_capability = (8, 9);
    t.matmul_resources = resources();
    t
}
fn wide_region(base: MatMulRegion) -> Result<MatMulRegion> {
    let mut r = four_square_region(base, resources(), true)?;
    r.schedule.accumulator_exchange = MatMulAccumulatorExchange::FragmentWindows {
        fragment_rows: 2,
        fragment_cols: 4,
    };
    ensure!(
        r.schedule.is_four_warp_wide_exchange_diagnostic() && !r.schedule.is_square128_streamed(),
        "not isolated4warp wide exchange"
    );
    ensure!(
        r.schedule.shared_mem_bytes() == 37888 && r.schedule.accumulator_exchange_bytes() == 32768,
        "exchange arena changed"
    );
    Ok(r)
}
fn module(r: MatMulRegion) -> Result<String> {
    let mut tiles = tile_graph(r)?;
    tiles.optimize_indices()?;
    Ok(PtxGraph::from(tiles).with_target(target()).module_source())
}
#[test]
fn wide_exchange_coverage_retirement_and_same_compute_proofs() -> Result<()> {
    for case in CASES {
        let base = super::super::k64_diagnostic::fixture(case)?;
        let old = four_square_region(base.clone(), resources(), true)?;
        let wide = wide_region(base)?;
        let mut same = wide.schedule;
        same.accumulator_exchange = old.schedule.accumulator_exchange;
        assert_eq!(same, old.schedule);
        let oldtiles = tile_graph(old)?;
        let tiles = tile_graph(wide)?;
        let oldbody = &oldtiles.matmul_region_kernels[0].ir.body.stmts;
        let ir = &tiles.matmul_region_kernels[0].ir;
        ir.validate_layouts()?;
        assert_eq!(ir.shared_mem_bytes, 37888);
        let body = &ir.body.stmts;
        let first = body
            .iter()
            .position(|s| matches!(s, Stmt::StoreAccumulatorWindow { .. }))
            .unwrap();
        let first_old = oldbody
            .iter()
            .position(|s| matches!(s, Stmt::StoreAccumulatorWindow { .. }))
            .unwrap();
        // The only preceding change is the exchange view extent; operands,
        // copies, zero, entire Kloop and async sentinel drain are identical.
        let mut prefix = body[..first - 1].to_vec();
        for stmt in &mut prefix {
            if let Stmt::MatMul { schedule, .. } = stmt {
                schedule.accumulator_exchange = oldbody
                    .iter()
                    .find_map(|s| match s {
                        Stmt::ForLoop { body, .. } => body.stmts.iter().find_map(|s| match s {
                            Stmt::MatMul { schedule, .. } => Some(schedule.accumulator_exchange),
                            _ => None,
                        }),
                        _ => None,
                    })
                    .unwrap();
            }
            if let Stmt::ForLoop { body, .. } = stmt {
                for s in &mut body.stmts {
                    if let Stmt::MatMul { schedule, .. } = s {
                        schedule.accumulator_exchange =
                            MatMulAccumulatorExchange::FragmentWindows {
                                fragment_rows: 1,
                                fragment_cols: 2,
                            };
                    }
                }
            }
        }
        assert_eq!(
            format!("{prefix:?}"),
            format!("{:?}", &oldbody[..first_old - 1])
        );
        let wait = body[..first]
            .iter()
            .rposition(|s| matches!(s, Stmt::AsyncWait))
            .unwrap();
        assert!(matches!(body[wait + 1], Stmt::Barrier));
        let windows = body
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                Stmt::StoreAccumulatorWindow {
                    fragment_origin,
                    fragment_shape,
                    ..
                } => {
                    assert_eq!(*fragment_shape, (2, 4));
                    Some((i, *fragment_origin))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            windows.iter().map(|(_, o)| *o).collect::<Vec<_>>(),
            [(0, 0), (2, 0)]
        );
        assert_eq!(
            body[first..]
                .iter()
                .filter(|s| matches!(s, Stmt::Barrier))
                .count(),
            3
        );
        let mut covered = vec![false; 128 * 128];
        let mut fragments = vec![false; 16];
        for (pass, (position, origin)) in windows.into_iter().enumerate() {
            assert!(matches!(body[position + 1], Stmt::Barrier));
            let Stmt::ForLoop {
                start,
                end,
                body: out,
                ..
            } = &body[position + 2]
            else {
                panic!("output loop missing")
            };
            assert_eq!(*start, Expr::Const(0));
            assert_eq!(*end, Expr::Const(64));
            assert!(
                !out.stmts
                    .iter()
                    .any(|s| matches!(s, Stmt::Barrier | Stmt::AsyncWait))
            );
            let (row, col, index) = out
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
                .unwrap();
            for round in 0..64 {
                for thread in 0..128 {
                    let i = round * 128 + thread;
                    let (sr, sc) = (i / 128, i % 128);
                    let r = sr / 32 * 64 + origin.0 * 16 + sr % 32;
                    let c = sc / 64 * 64 + sc % 64;
                    assert!(r < 128 && c < 128);
                    assert!(!covered[r * 128 + c]);
                    covered[r * 128 + c] = true;
                    assert_eq!(
                        (
                            eval(row, thread, (0, 0), round),
                            eval(col, thread, (0, 0), round)
                        ),
                        (r as u64, c as u64)
                    );
                    assert_eq!(eval(index, thread, (0, 0), round), (r * case.n + c) as u64);
                    assert!((sr * 128 + sc) * 4 + 4 <= 32768);
                }
            }
            for m in 0..2 {
                for n in 0..4 {
                    let fragment = (origin.0 + m) * 4 + origin.1 + n;
                    assert!(fragment < 16);
                    assert!(!fragments[fragment]);
                    fragments[fragment] = true;
                }
            }
            if pass == 0 {
                assert!(matches!(body[position + 3], Stmt::Barrier));
            } else {
                assert_eq!(position + 3, body.len());
            }
        }
        assert!(covered.into_iter().all(|v| v));
        assert!(fragments.into_iter().all(|v| v));
        let source = module(tiles_to_region_for_source(case)?)?;
        assert_eq!(source.matches("wmma.mma.sync").count(), 32);
        assert_eq!(source.matches("wmma.load.a.sync").count(), 8);
        assert_eq!(source.matches("wmma.load.b.sync").count(), 8);
        assert_eq!(source.matches("wmma.store.d.sync").count(), 16);
    }
    Ok(())
}
fn tiles_to_region_for_source(case: Case) -> Result<MatMulRegion> {
    wide_region(super::super::k64_diagnostic::fixture(case)?)
}

#[test]
fn wide_exchange_invalid_shapes_stride_and_fusion_rejected() -> Result<()> {
    let r = wide_region(super::super::k64_diagnostic::fixture(CASES[0])?)?;
    let mut invalid = r.schedule;
    invalid.block_threads = (32, 8, 1);
    invalid.warp_topology = (4, 2);
    invalid.warp_fragments = (2, 4);
    assert!(!invalid.supports_streamed_fragment_windows());
    let mut invalid = r.schedule;
    invalid.block_tile = MatMulTile {
        m: 64,
        n: 128,
        k: 32,
    };
    assert!(!invalid.supports_streamed_fragment_windows());
    let tiles = tile_graph(r.clone())?;
    for origin in [(1, 0), (0, 1), (4, 0)] {
        let mut ir = tiles.matmul_region_kernels[0].ir.clone();
        if let Some(Stmt::StoreAccumulatorWindow {
            fragment_origin, ..
        }) = ir
            .body
            .stmts
            .iter_mut()
            .find(|s| matches!(s, Stmt::StoreAccumulatorWindow { .. }))
        {
            *fragment_origin = origin;
        }
        assert!(ir.validate_layouts().is_err());
    }
    let mut ir = tiles.matmul_region_kernels[0].ir.clone();
    if let Some(Stmt::ViewSharedArena { layout, .. }) = ir.body.stmts.iter_mut().rev().find(|s| {
        matches!(
            s,
            Stmt::ViewSharedArena {
                dtype: DType::F32,
                ..
            }
        )
    }) {
        *layout = SharedLayout::row_major(64);
    }
    assert!(ir.validate_layouts().is_err());
    let mut fused = r;
    // A second logical output is outside the plain-output path. It must never
    // silently lose its SSA output by entering the test-only exchange branch.
    fused.outputs.push(fused.outputs[0].clone());
    if let Ok(tiles) = tile_graph(fused) {
        assert!(
            tiles.matmul_region_kernels[0]
                .ir
                .validate_layouts()
                .is_err()
        );
    }
    Ok(())
}

#[test]
fn wide_exchange_all_eight_legacy_and_four_old_four_warp_ptx_are_identical() -> Result<()> {
    let legacy = PathBuf::from("target/paper-kernels/four-warp-square-diagnostic/legacy-proof");
    for (ta, tb) in [(false, false), (false, true), (true, false), (true, true)] {
        let case = Case {
            name: "legacy",
            m: 2048,
            n: 1024,
            k: 128,
        };
        let mut original = super::super::k64_diagnostic::fixture(case)?;
        for (input, transposed) in original.inputs.iter_mut().zip([ta, tb]) {
            if transposed {
                let mut shape = input.source_shape.clone();
                shape.swap(0, 1);
                input.tensor =
                    VirtualTensor::identity(input.tensor.source, shape.clone(), DType::BF16)
                        .permute(&[1, 0])?;
                input.source_shape = shape;
            }
        }
        original.configure_pipeline(
            MatMulSharedLayout::Padded,
            MatMulPipeline::DoubleBuffered,
            true,
            resources(),
        )?;
        let baseline = module(original.clone())?;
        let file = legacy.join(format!("baseline-{ta}-{tb}.ptx"));
        ensure!(
            baseline.as_bytes() == std::fs::read(&file)?,
            "legacy FullTile changed: {}",
            file.display()
        );
        original.schedule = crate::tile::large_tile_policy::square128_streamed::region_candidate(
            &original,
            MatMulSharedLayout::Padded,
            MatMulPipeline::DoubleBuffered,
            true,
            resources(),
        )
        .context("legacy candidate absent")?;
        let source = module(original)?;
        let file = legacy.join(format!("candidate-{ta}-{tb}.ptx"));
        ensure!(
            source.as_bytes() == std::fs::read(&file)?,
            "legacy streamed8 changed: {}",
            file.display()
        );
    }
    for case in CASES {
        let old = four_square_region(
            super::super::k64_diagnostic::fixture(case)?,
            resources(),
            true,
        )?;
        let source = module(old)?;
        let file = PathBuf::from("target/paper-kernels/four-warp-square-diagnostic/qualification")
            .join(case.name)
            .join("test_only_four_square.ptx");
        ensure!(
            source.as_bytes() == std::fs::read(&file)?,
            "old4warp1x2 changed: {}",
            file.display()
        );
    }
    Ok(())
}

fn build_wide_square(
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
    let region = wide_region(plan.matmul_regions()[id].clone())?;
    let selected = region.schedule;
    let mut tiles = tile_graph(region)?;
    tiles.optimize_indices()?;
    let ptx = PtxGraph::from(tiles).with_target(e.target);
    let source = ptx.module_source();
    let module = stream.context().load_module(Ptx::from_src(&source))?;
    let name = ptx
        .matmul_region_kernel_name(0)
        .context("four-warp wide-exchange Square function absent")?;
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
    let resources = json!({"mode":"test_only_four_square_wide_exchange","function":name,"registers_per_thread":registers,"local_bytes":local,"shared_bytes":shared,"max_threads_per_block":max_threads,"static_active_block_capacity":capacity,"capacity_is_not_measured_occupancy":true,"schedule":format!("{selected:?}"),"required_resource_gate_passed":accepted,"production_selection_unchanged":true,"production_activation_claim":false});
    std::fs::write(dir.join("test_only_four_square_wide_exchange.ptx"), source)?;
    write_json(
        &dir.join("test_only_four_square_wide_exchange-resources.json"),
        &resources,
    )?;
    ensure!(
        accepted,
        "four-warp wide-exchange Square actual resource admission failed: {resources}"
    );
    ensure!(
        shared == 37888,
        "four-warp wide-exchange Square physical shared accounting mismatch"
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
        name: "test_only_four_square_wide_exchange",
        resources,
        samples_us: Vec::new(),
    })
}

#[test]
#[ignore = "manual test-only four-warp2x4 exchange resource/output/replay qualification; no production selection"]
fn wide_exchange_actual_resources_changed_feed_and_replay() -> Result<()> {
    let directory = PathBuf::from(
        std::env::var_os("TNSR_WIDE_EXCHANGE_DIAG_DIR")
            .context("set explicit wide exchange artifact directory")?,
    );
    std::fs::create_dir_all(&directory)?;
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let mut results = Vec::new();
    for case in CASES {
        let dir = directory.join(case.name);
        std::fs::create_dir_all(&dir)?;
        let a = (0..case.m * case.k)
            .map(|i| bf16::from_f32(((i % 97 * 13 % 97) as f32 - 48.) / 97.))
            .collect::<Vec<_>>();
        let b = (0..case.k * case.n)
            .map(|i| bf16::from_f32(((i % 89 * 17 % 89) as f32 - 44.) / 89.))
            .collect::<Vec<_>>();
        let mut modes = vec![
            build_four_square(case, &stream, &a, &b, &dir)?,
            build_wide_square(case, &stream, &a, &b, &dir)?,
        ];
        let (expected, checks) = qualify(case, &mut modes, &stream, &a, &b)?;
        for mode in &mut modes {
            mode.capture(&stream, 4)?;
        }
        for revision in 0..2 {
            let aa = a
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    if i % 3 == revision {
                        *x
                    } else {
                        bf16::from_f32(-(*x).to_f32())
                    }
                })
                .collect::<Vec<_>>();
            let mut values = Vec::new();
            for mode in &mut modes {
                mode.upload(&stream, &aa, &b)?;
                mode.capture.as_mut().context("capture absent")?.launch()?;
                values.push(mode.read(&stream)?);
            }
            compare(&values[0], &values[1], "changed-feed captured output")?;
        }
        for mode in &mut modes {
            mode.upload(&stream, &a, &b)?;
            mode.capture.as_mut().context("capture absent")?.launch()?;
            compare(&expected, &mode.read(&stream)?, "restored captured output")?;
        }
        results.push(json!({"case":case.name,"checks":checks,"resources":modes.iter().map(|m|&m.resources).collect::<Vec<_>>(),"changed_feed_and_replay":true,"timing_performed":false}));
        write_json(
            &directory.join("qualification.json"),
            &json!({"cases":results,"production_selection_unchanged":true,"performance_claim":false}),
        )?;
    }
    Ok(())
}

fn build_strong_plain_control(
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
    e.set_experimental_plain_bf16_output(true);
    e.compile(&graph)?;
    ensure!(
        e.plain_bf16_output_counts() == (1, 0),
        "strong production plain control must actually select"
    );
    let plan = e.execution_plan().context("control plan absent")?;
    let (selected, step) = schedule(plan)?;
    ensure!(
        selected.block_tile
            == (MatMulTile {
                m: 64,
                n: 64,
                k: 32
            })
            && selected.block_threads == (32, 8, 1),
        "strong control geometry changed"
    );
    let PtxPlanAction::MatMulRegion(id) = step.action else {
        anyhow::bail!("control not plain region")
    };
    ensure!(
        e.plain_bf16_output_selected(id),
        "control selected metadata absent"
    );
    let ptx = e.ptx_graph.as_ref().context("control PTX absent")?;
    let name = ptx
        .matmul_region_kernel_name(id)
        .context("control function absent")?;
    let module = e.module.as_ref().context("control module absent")?.clone();
    let function = module.load_function(name)?;
    let registers = function.num_regs()?;
    let local = function.local_size_bytes()?;
    let shared = function.shared_size_bytes()?;
    let max_threads = function.max_threads_per_block()?;
    let capacity = function.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?;
    ensure!(
        local == 0 && max_threads >= 256 && capacity >= 2,
        "strong control resource admission failed"
    );
    let resources = json!({"mode":"accepted64_plain_output","plain_bf16_output":true,"actual_plain_regions":1,"function":name,"registers_per_thread":registers,"local_bytes":local,"shared_bytes":shared,"max_threads_per_block":max_threads,"static_active_block_capacity":capacity,"capacity_is_not_measured_occupancy":true,"schedule":format!("{selected:?}")});
    std::fs::write(dir.join("accepted64_plain_output.ptx"), ptx.module_source())?;
    write_json(
        &dir.join("accepted64_plain_output-resources.json"),
        &resources,
    )?;
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
            grid_dim: (u32::try_from(case.n / 64)?, u32::try_from(case.m / 64)?, 1),
            block_dim: selected.block_threads,
            shared_mem_bytes: 0,
        },
        schedule: selected,
        name: "accepted64_plain_output",
        resources,
        samples_us: Vec::new(),
    })
}

#[test]
#[ignore = "manual3mode component timing; requires parent clean-stage acknowledgment; no production adoption"]
fn wide_exchange_three_mode_interleaved_component_timing() -> Result<()> {
    let directory = PathBuf::from(
        std::env::var_os("TNSR_WIDE_EXCHANGE_TIMING_DIR")
            .context("set explicit timing artifact directory")?,
    );
    std::fs::create_dir_all(&directory)?;
    let reverse = std::env::var("TNSR_WIDE_EXCHANGE_TIMING_REVERSE").as_deref() == Ok("1");
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let mut cases = CASES.to_vec();
    if reverse {
        cases.reverse();
    }
    let mut results = Vec::new();
    let repeats = 32;
    let rounds = 32;
    for case in cases {
        let dir = directory.join(case.name);
        std::fs::create_dir_all(&dir)?;
        let a = (0..case.m * case.k)
            .map(|i| bf16::from_f32(((i % 97 * 13 % 97) as f32 - 48.) / 97.))
            .collect::<Vec<_>>();
        let b = (0..case.k * case.n)
            .map(|i| bf16::from_f32(((i % 89 * 17 % 89) as f32 - 44.) / 89.))
            .collect::<Vec<_>>();
        let construction = if reverse { [2, 1, 0] } else { [0, 1, 2] };
        let mut slots: [Option<Mode>; 3] = [None, None, None];
        for index in construction {
            slots[index] = Some(match index {
                0 => build_strong_plain_control(case, &stream, &a, &b, &dir)?,
                1 => build_four_square(case, &stream, &a, &b, &dir)?,
                _ => build_wide_square(case, &stream, &a, &b, &dir)?,
            });
        }
        let mut modes = slots
            .into_iter()
            .map(|s| s.context("mode not built"))
            .collect::<Result<Vec<_>>>()?;
        let (expected, strong_vs_old) = qualify(case, &mut modes[..2], &stream, &a, &b)?;
        let (_, old_vs_wide) = qualify(case, &mut modes[1..], &stream, &a, &b)?;
        for m in &mut modes {
            m.capture(&stream, repeats)?;
        }
        // Capture launch proof and full output readbacks are outside timing.
        for m in &mut modes {
            m.capture.as_mut().context("missing capture")?.launch()?;
            compare(
                &expected,
                &m.read(&stream)?,
                "pre-timing captured fulloutput",
            )?;
        }
        let began = Instant::now();
        let mut warmups = 0;
        while began.elapsed() < Duration::from_millis(500) {
            let order = if (warmups % 2 == 0) ^ reverse {
                [0, 1, 2]
            } else {
                [2, 1, 0]
            };
            for i in order {
                modes[i]
                    .capture
                    .as_mut()
                    .context("missing capture")?
                    .launch()?;
            }
            stream.synchronize()?;
            warmups += 1;
        }
        let warmup_ms = began.elapsed().as_secs_f64() * 1000.;
        let mut raw = Vec::new();
        for round in 0..rounds {
            let order = if (round % 2 == 0) ^ reverse {
                [0, 1, 2]
            } else {
                [2, 1, 0]
            };
            let mut times = [0.; 3];
            for i in order {
                let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                modes[i]
                    .capture
                    .as_mut()
                    .context("missing capture")?
                    .launch()?;
                let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                let us = f64::from(start.elapsed_ms(&end)?) * 1000. / repeats as f64;
                ensure!(us > 0. && us.is_finite(), "invalid event interval");
                times[i] = us;
                modes[i].samples_us.push(us);
            }
            raw.push(json!({"round":round,"execution_order_indices":order,"execution_order":order.map(|i|modes[i].name),"accepted64_plain_us":times[0],"old4warp1x2_us":times[1],"new4warp2x4_us":times[2],"old_over_new":times[1]/times[2],"strong_control_over_new":times[0]/times[2]}));
        }
        stream.synchronize()?;
        for m in &modes {
            compare(&expected, &m.read(&stream)?, "post-timing fulloutput")?;
        }
        let bytes_per_mode = 2 * (case.m * case.k + case.k * case.n + case.m * case.n);
        let sums = modes
            .iter()
            .map(|m| m.samples_us.iter().sum::<f64>())
            .collect::<Vec<_>>();
        let summaries=modes.iter().map(|m|json!({"mode":m.name,"resources":m.resources,"samples_us":m.samples_us,"median_us":median(&m.samples_us)})).collect::<Vec<_>>();
        let result = json!({"case":case.name,"m":case.m,"n":case.n,"k":case.k,"construction_order_indices":construction,"reversed_case_order":reverse,"graph_repeats":repeats,"measured_rounds":rounds,"warmup_rounds":warmups,"warmup_ms":warmup_ms,"prequalification":{"strong_vs_old":strong_vs_old,"old_vs_wide":old_vs_wide},"pre_post_fulloutput_pass":true,"allocation_protocol":{"device_inputs_shared_across_modes":false,"copies_per_operand":3,"separate_output_buffers":3,"bytes_per_mode":bytes_per_mode,"total_live_operand_output_bytes":3*bytes_per_mode,"cache_scope":"Repeated component GEMMs on each mode's own preallocated unchanged A/B/output buffers; all3 modes remain live. No setup, transfers, compilation or readbacks inside event intervals."},"rounds":raw,"summaries":summaries,"old_over_new_ratio_of_summed_times":sums[1]/sums[2],"strong_control_over_new_ratio_of_summed_times":sums[0]/sums[2],"scope":"component only; old4warp1x2 versus2x4 and production64square plain-output strong control; no complete-update/adoption claim"});
        write_json(&dir.join("result.json"), &result)?;
        results.push(result);
        write_json(
            &directory.join("report.json"),
            &json!({"cases":results,"reverse":reverse,"timing_excludes_setup_and_output_checks":true,"production_policy_changed":false}),
        )?;
    }
    Ok(())
}
