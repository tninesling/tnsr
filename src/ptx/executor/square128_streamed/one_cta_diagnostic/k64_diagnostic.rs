//! Test-only K32/K64 component comparison. Production selection is untouched.
use super::*;
use crate::tile::{
    DType, Dim, Expr, MatMulAccumulatorExchange, MatMulCapabilities, MatMulFragmentLoad,
    MatMulRegion, MatMulResources, MatMulTile, SharedLayout, Stmt,
};

fn k64_region(
    mut region: MatMulRegion,
    resources: MatMulResources,
    supports_async: bool,
) -> Result<MatMulRegion> {
    let base = region.schedule;
    ensure!(
        region.batch_shape.is_empty()
            && region.output_shape == [region.m, region.n]
            && region.lhs_shape == [region.m, region.k]
            && region.rhs_shape == [region.k, region.n]
            && region.inputs.len() == 2
            && region.outputs.len() == 1
            && region.outputs[0].value == region.matmul_value
            && region.epilogue_operations.is_empty()
            && base.logical_shape
                == (MatMulTile {
                    m: region.m,
                    n: region.n,
                    k: region.k
                })
            && region.m > 0
            && region.n > 0
            && region.m.is_multiple_of(64)
            && region.n.is_multiple_of(64)
            && region.k >= 512
            && region.k.is_multiple_of(64),
        "K64 requires full NN plain rank-two M/N/K with K>=512"
    );
    ensure!(
        base.block_tile
            == (MatMulTile {
                m: 64,
                n: 64,
                k: 32
            })
            && base.plan == crate::tile::MatMulPlan::TensorCoreBF16
            && base.storage_dtype == DType::BF16
            && base.operand_dtype == DType::BF16
            && base.block_threads == (32, 8, 1)
            && base.warp_topology == (2, 2)
            && base.warp_fragments == (2, 2)
            && base.accumulator_exchange == MatMulAccumulatorExchange::FullTile
            && base.fragment_load == MatMulFragmentLoad::PreloadAll
            && base.pipeline_stages == 2
            && base.copy_bytes == 16
            && base.async_operands == (true, true)
            && base.reuse_shared_memory
            && !base.operand_layouts.0.column_major
            && !base.operand_layouts.1.column_major,
        "K64 requires accepted 64² K32 ownership/staging"
    );
    for (value, shape) in [
        (region.lhs_value, &region.lhs_shape),
        (region.rhs_value, &region.rhs_shape),
    ] {
        let input = region
            .inputs
            .iter()
            .find(|i| i.value == value)
            .context("operand absent")?;
        ensure!(
            crate::tile::ordinary_bf16_block::aligned_operand(input, shape, false).is_some(),
            "unproved full aligned NN source map"
        );
    }
    // Never retain K32's A stride40. Reconstruct both layouts and the complete
    // pipeline from the new K64 dimensions before lowering any shared views.
    region.schedule.block_tile.k = 64;
    region.schedule.operand_layouts = (SharedLayout::row_major(64), SharedLayout::row_major(64));
    region.schedule.pipeline_stages = 1;
    region.schedule.async_operands = (false, false);
    region.schedule.copy_bytes = 4;
    region.configure_pipeline(
        MatMulSharedLayout::Padded,
        MatMulPipeline::Auto,
        supports_async,
        resources,
    )?;
    let selected = region.schedule;
    ensure!(
        selected.operand_layouts == (SharedLayout::padded(64, 8)?, SharedLayout::padded(64, 8)?)
            && selected.pipeline_stages == 2
            && selected.async_operands == (true, true)
            && selected.copy_bytes == 16
            && selected.reuse_shared_memory
            && selected.shared_mem_bytes() == 36 * 1024
            && selected.conservative_two_block_fit(resources),
        "K64 two-stage padded reuse/resource proof failed"
    );
    let mut normalized = selected;
    normalized.block_tile.k = base.block_tile.k;
    normalized.operand_layouts = base.operand_layouts;
    ensure!(
        normalized == base,
        "K64 changed policy beyond Ktile and its operand layouts"
    );
    Ok(region)
}

fn tile_graph(region: MatMulRegion) -> Result<TileGraph> {
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.add_matmul_regions(&[region])?;
    Ok(tiles)
}
pub(super) fn fixture(case: Case) -> Result<MatMulRegion> {
    let resources = MatMulResources {
        shared_bytes_per_sm: 100 * 1024,
        threads_per_sm: 1536,
        ..MatMulResources::default()
    };
    let graph: TensorGraph<bf16> = TensorExpr::input("a", vec![case.m, case.k])
        .matmul(TensorExpr::input("b", vec![case.k, case.n]))
        .into();
    let node = graph
        .graph
        .node_indices()
        .find(|&n| matches!(graph[n], TensorGraphNode::MatMul { .. }))
        .context("fixture GEMM absent")?;
    let mut region = MatMulRegion::from_graph(&graph, node, vec![], vec![node])?;
    region.schedule = MatMulSchedule::select_with_resources(
        case.m,
        case.n,
        case.k,
        DType::BF16,
        MatMulCapabilities {
            bf16: true,
            tf32: true,
            f16: true,
        },
        MatMulPrecision::NativeHalf,
        resources,
    )
    .with_shared_memory_reuse(true);
    region.configure_pipeline(
        MatMulSharedLayout::Padded,
        MatMulPipeline::Auto,
        true,
        resources,
    )?;
    Ok(region)
}

/// Emulate the actual unsigned TileIR index expressions before normalization.
fn eval(expr: &Expr, thread: usize, block: (usize, usize), k: usize) -> u64 {
    let get_dim = |dim: &Dim, x: usize, y: usize, z: usize| match dim {
        Dim::X => x as u64,
        Dim::Y => y as u64,
        Dim::Z => z as u64,
    };
    match expr {
        Expr::Const(v) => *v as u64,
        Expr::Var(v) => {
            assert_eq!(v, "k_tile");
            k as u64
        }
        Expr::ThreadIdx(d) => get_dim(d, thread % 32, thread / 32, 0),
        Expr::BlockIdx(d) => get_dim(d, block.1, block.0, 0),
        Expr::BlockDim(d) => get_dim(d, 32, 8, 1),
        Expr::Add(a, b) => eval(a, thread, block, k).wrapping_add(eval(b, thread, block, k)),
        Expr::Sub(a, b) => eval(a, thread, block, k).wrapping_sub(eval(b, thread, block, k)),
        Expr::Mul(a, b) => eval(a, thread, block, k).wrapping_mul(eval(b, thread, block, k)),
        Expr::FloorDiv(a, b) => eval(a, thread, block, k) / (*b as u64),
        Expr::Mod(a, b) => eval(a, thread, block, k) % (*b as u64),
        Expr::ShiftRight(a, b) => eval(a, thread, block, k) >> *b,
        Expr::BitAnd(a, b) => eval(a, thread, block, k) & *b,
    }
}
fn copies(block: &crate::tile::Block) -> Vec<&Stmt> {
    block
        .stmts
        .iter()
        .filter(|s| matches!(s, Stmt::AsyncCopy { .. }))
        .collect()
}
fn check_copy_stage(
    stmts: &[&Stmt],
    case: Case,
    block: (usize, usize),
    iteration: usize,
    sentinel: bool,
) {
    for lhs in [true, false] {
        let mut covered = vec![false; 64 * 64];
        let mut groups = 0;
        for stmt in stmts {
            let Stmt::AsyncCopy {
                src_param,
                element_index,
                row,
                col,
                tile_row,
                tile_col,
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
            for thread in 0..256 {
                let (r, c) = (
                    eval(tile_row, thread, block, iteration),
                    eval(tile_col, thread, block, iteration),
                );
                assert!(r < 64 && c < 64 && c.is_multiple_of(8) && c + 7 < 64);
                for lane in 0..8 {
                    let index = (r * 64 + c + lane) as usize;
                    assert!(!covered[index]);
                    covered[index] = true;
                }
                let (global_r, global_c) = (
                    eval(row, thread, block, iteration),
                    eval(col, thread, block, iteration),
                );
                let valid = global_r < layout.rows as u64 && global_c < layout.cols as u64;
                assert_eq!(
                    valid, !sentinel,
                    "only final +1 prefetch is wholly zero-filled"
                );
                if valid {
                    assert!(global_c + 7 < layout.cols as u64);
                    let index = eval(element_index, thread, block, iteration);
                    assert_eq!(index, global_r * layout.row_stride as u64 + global_c);
                    let length = if lhs {
                        case.m * case.k
                    } else {
                        case.k * case.n
                    };
                    assert!(index + 7 < length as u64);
                }
                groups += 1;
            }
        }
        assert_eq!(groups, 512);
        assert!(covered.into_iter().all(|v| v));
    }
}

#[test]
fn k64_scope_layout_resource_and_actual_copy_expression_proofs() -> Result<()> {
    let resources = MatMulResources {
        shared_bytes_per_sm: 100 * 1024,
        threads_per_sm: 1536,
        ..MatMulResources::default()
    };
    for case in CASES.into_iter().chain([Case { k: 576, ..CASES[0] }]) {
        let original = fixture(case)?;
        let selected = k64_region(original.clone(), resources, true)?;
        assert_eq!(selected.schedule.shared_mem_bytes(), 36864);
        let mut disjoint = selected.schedule;
        disjoint.reuse_shared_memory = false;
        assert_eq!(disjoint.shared_mem_bytes(), 53248);
        let before = (0..case.k / 32)
            .flat_map(|t| [t * 32, t * 32 + 16])
            .collect::<Vec<_>>();
        let after = (0..case.k / 64)
            .flat_map(|t| [t * 64, t * 64 + 16, t * 64 + 32, t * 64 + 48])
            .collect::<Vec<_>>();
        assert_eq!(
            before, after,
            "all per-output K16 accumulation dependencies retain order"
        );
        let tiles = tile_graph(selected)?;
        let ir = &tiles.matmul_region_kernels[0].ir;
        ir.validate_layouts()?;
        let initial = copies(&ir.body);
        assert_eq!(initial.len(), 4);
        let (end, body) = ir
            .body
            .stmts
            .iter()
            .find_map(|s| match s {
                Stmt::ForLoop {
                    loop_var,
                    end,
                    body,
                    ..
                } if loop_var == "k_tile" => Some((end, body)),
                _ => None,
            })
            .context("K loop absent")?;
        assert_eq!(*end, Expr::Const((case.k / 64) as i64));
        let next = copies(body);
        assert_eq!(next.len(), 4);
        let views = ir
            .body
            .stmts
            .iter()
            .filter_map(|s| match s {
                Stmt::ViewSharedArena {
                    dest,
                    byte_offset,
                    dtype: DType::BF16,
                    ..
                } => Some((*dest, *byte_offset)),
                _ => None,
            })
            .collect::<std::collections::HashMap<_, _>>();
        let reads = body
            .stmts
            .iter()
            .filter_map(|s| match s {
                Stmt::MatMul { a, b, .. } => Some([*a, *b]),
                _ => None,
            })
            .flatten()
            .collect::<std::collections::HashSet<_>>();
        let writes = next
            .iter()
            .filter_map(|s| match s {
                Stmt::AsyncCopy { dest, .. } => Some(*dest),
                _ => None,
            })
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(reads.len(), 2);
        assert_eq!(writes.len(), 2);
        let stages = body
            .stmts
            .iter()
            .filter_map(|s| match s {
                Stmt::SelectSharedStage {
                    dest,
                    first,
                    second,
                    stage,
                } => Some((*dest, *first, *second, stage)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(stages.len(), 4);
        for (dest, first, second, stage) in stages {
            assert_eq!(views[&second], views[&first] + 18432);
            assert!(matches!(views[&first], 0 | 9216));
            assert!(reads.contains(&dest) ^ writes.contains(&dest));
            for k in 0..case.k / 64 {
                assert_eq!(
                    eval(stage, 0, (0, 0), k),
                    ((k + usize::from(writes.contains(&dest))) % 2) as u64
                );
            }
        }
        for br in [0, case.m / 64 - 1] {
            for bc in [0, case.n / 64 - 1] {
                check_copy_stage(&initial, case, (br, bc), 0, false);
                for k in 0..case.k / 64 {
                    check_copy_stage(&next, case, (br, bc), k, k + 1 == case.k / 64);
                }
            }
        }
        let exchange = ir
            .body
            .stmts
            .iter()
            .position(|s| matches!(s, Stmt::ConvertLayout { .. }))
            .context("F32 exchange absent")?;
        assert!(matches!(ir.body.stmts[exchange - 3], Stmt::AsyncWait));
        assert!(matches!(ir.body.stmts[exchange - 2], Stmt::Barrier));
        assert!(matches!(
            ir.body.stmts[exchange - 1],
            Stmt::ViewSharedArena {
                dtype: DType::F32,
                byte_offset: 0,
                ..
            }
        ));
        assert!(k64_region(original.clone(), MatMulResources::default(), true).is_err());
        assert!(k64_region(original.clone(), resources, false).is_err());
        let mut short = original;
        short.inputs[0].source_shape[0] -= 1;
        assert!(k64_region(short, resources, true).is_err());
    }
    for case in [
        Case {
            m: 4097,
            ..CASES[0]
        },
        Case { n: 513, ..CASES[0] },
        Case { k: 515, ..CASES[0] },
        Case { k: 256, ..CASES[0] },
    ] {
        assert!(k64_region(fixture(case)?, resources, true).is_err());
    }
    Ok(())
}

#[test]
fn k64_offline_ptx_k16_count_and_final_drain_proofs() -> Result<()> {
    let mut target = PtxTarget::sm80();
    target.compute_capability = (8, 9);
    target.matmul_resources.shared_bytes_per_sm = 100 * 1024;
    target.matmul_resources.threads_per_sm = 1536;
    for case in CASES {
        let region = k64_region(fixture(case)?, target.matmul_resources, true)?;
        let mut tiles = tile_graph(region)?;
        tiles.optimize_indices()?;
        let source = PtxGraph::from(tiles).with_target(target).module_source();
        assert!(source.contains(".shared .align 32 .b8 shared[36864];"));
        assert_eq!(source.matches("wmma.mma.sync").count(), 16);
        assert_eq!(source.matches("wmma.load.a.sync").count(), 8);
        assert_eq!(source.matches("wmma.load.b.sync").count(), 8);
        assert_eq!(source.matches("cp.async.ca.shared.global").count(), 8);
        let store = source
            .find("wmma.store.d.sync")
            .context("WMMA store absent")?;
        let drain = source[..store]
            .rfind("cp.async.wait_group 0;")
            .context("final drain absent")?;
        assert!(source[drain..store].contains("bar.sync 0;"));
        if let Some(dir) = std::env::var_os("TNSR_K64_PTX_DIR") {
            std::fs::create_dir_all(&dir)?;
            std::fs::write(
                PathBuf::from(dir).join(format!("cpu-{}.ptx", case.name)),
                source,
            )?;
        }
    }
    Ok(())
}

fn build_k64(
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
    let region = k64_region(
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
        .context("K64 function absent")?;
    let function = module.load_function(name)?;
    let (registers, local, shared, max_threads, capacity) = (
        function.num_regs()?,
        function.local_size_bytes()?,
        function.shared_size_bytes()?,
        function.max_threads_per_block()?,
        function.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?,
    );
    let accepted = resources_fit(
        registers,
        local,
        shared,
        max_threads,
        capacity,
        e.target.matmul_resources.shared_bytes_per_block,
    );
    let resources = json!({"mode":"test_only_k64","function":name,"registers_per_thread":registers,"local_bytes":local,"shared_bytes":shared,"max_threads_per_block":max_threads,"static_active_block_capacity":capacity,"capacity_is_not_measured_occupancy":true,"schedule":format!("{selected:?}"),"required_resource_gate_passed":accepted,"production_selection_unchanged":true,"production_activation_claim":false});
    std::fs::write(dir.join("test_only_k64.ptx"), source)?;
    write_json(&dir.join("test_only_k64-resources.json"), &resources)?;
    ensure!(
        accepted,
        "K64 actual resource admission failed: {resources}"
    );
    ensure!(shared == 36864, "K64 physical shared accounting mismatch");
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
        name: "test_only_k64",
        resources,
        samples_us: Vec::new(),
    })
}

#[test]
#[ignore = "manual test-only K64 paired component diagnostic; no production selector change"]
fn k64_component_resource_equivalence_and_paired_timing() -> Result<()> {
    let directory = PathBuf::from(
        std::env::var_os("TNSR_K64_DIAG_DIR").context("set explicit K64 artifact directory")?,
    );
    std::fs::create_dir_all(&directory)?;
    let reverse = std::env::var("TNSR_K64_DIAG_REVERSE").as_deref() == Ok("1");
    let timing = std::env::var("TNSR_K64_DIAG_TIMING").as_deref() == Ok("1");
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
                build_k64(case, &stream, &a, &b, &dir)
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
                    if mode.schedule.block_tile.k == 64 {
                        candidate = us;
                    } else {
                        baseline = us;
                    }
                }
                pairs.push(json!({"sample":sample,"execution_order":order,"baseline_k32_us":baseline,"test_only_k64_us":candidate,"baseline_over_candidate":baseline/candidate}));
            }
            stream.synchronize()?;
            for mode in &modes {
                compare(&expected, &mode.read(&stream)?, "K64 post-capture finite")?;
            }
        }
        let summaries=modes.iter().map(|mode|json!({"mode":mode.name,"resources":mode.resources,"samples_us":mode.samples_us,"median_us":if timing{Some(median(&mode.samples_us))}else{None}})).collect::<Vec<_>>();
        let result = json!({"status":"qualified_test_fixture_only","case":case.name,"m":case.m,"n":case.n,"k":case.k,"orientation":"NN","qualification":checks,"timing_performed":timing,"post_capture_bits_equal":timing.then_some(true),"samples":if timing{samples}else{0},"graph_kernel_repeats":repeats,"warmup_pairs":warmups,"warmup_elapsed_ms":warmup_ms,"pairs":pairs,"summaries":summaries,"production_selector_changed":false,"scope":"Ktile only: accepted64²K32 versus test-only64²K64. Four compute warps/eight block warps,2x2fragments,FullTile,PreloadAll,F32K16order/BF16RN unchanged; not complete updates"});
        write_json(&dir.join("result.json"), &result)?;
        results.push(result);
        write_json(
            &directory.join("report.json"),
            &json!({"cases":results,"construction_order":if reverse{"candidate,baseline"}else{"baseline,candidate"},"scope":"test-only K64 diagnostic; no production selector/API/cache/default changes"}),
        )?;
    }
    Ok(())
}
