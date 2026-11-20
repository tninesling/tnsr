//! Isolated full-MN scalar BF16 output probe; no production hook.
use super::*;
use crate::tile::{
    DType, Dim, Expr, MatMulAccumulatorExchange, MatMulFragmentLoad, MatMulRegion, Stmt, TileIR,
};
const BASE: &str = "plain_output_base";
fn tile_graph(region: MatMulRegion) -> Result<TileGraph> {
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.add_matmul_regions(&[region])?;
    Ok(tiles)
}
fn guard(r: &MatMulRegion) -> Result<()> {
    let s = r.schedule;
    ensure!(
        r.batch_shape.is_empty()
            && r.epilogue_operations.is_empty()
            && r.outputs.len() == 1
            && r.outputs[0].value == r.matmul_value
            && r.inputs.len() == 2
            && r.output_shape == [r.m, r.n]
            && r.lhs_shape == [r.m, r.k]
            && r.rhs_shape == [r.k, r.n]
            && r.m > 0
            && r.n > 0
            && r.m.is_multiple_of(64)
            && r.n.is_multiple_of(64)
            && r.k >= 512
            && r.k.is_multiple_of(32)
            && s.logical_shape
                == (crate::tile::MatMulTile {
                    m: r.m,
                    n: r.n,
                    k: r.k
                })
            && s.plan == crate::tile::MatMulPlan::TensorCoreBF16
            && s.block_tile
                == (crate::tile::MatMulTile {
                    m: 64,
                    n: 64,
                    k: 32
                })
            && s.block_threads == (32, 8, 1)
            && s.warp_topology == (2, 2)
            && s.warp_fragments == (2, 2)
            && s.accumulator_exchange == MatMulAccumulatorExchange::FullTile
            && s.fragment_load == MatMulFragmentLoad::PreloadAll
            && s.storage_dtype == DType::BF16
            && s.operand_dtype == DType::BF16
            && s.pipeline_stages == 2
            && s.async_operands == (true, true)
            && s.copy_bytes == 16
            && s.reuse_shared_memory
            && s.effective_grid_group_m() <= 1
            && s.operand_layouts
                == (
                    crate::tile::SharedLayout::padded(32, 8)?,
                    crate::tile::SharedLayout::padded(64, 8)?
                ),
        "probe requires accepted full-MN plain rank2 BF16 geometry"
    );
    for (v, shape) in [(r.lhs_value, &r.lhs_shape), (r.rhs_value, &r.rhs_shape)] {
        let i = r
            .inputs
            .iter()
            .find(|i| i.value == v)
            .context("missing operand")?;
        ensure!(
            crate::tile::ordinary_bf16_block::aligned_operand(i, shape, false).is_some(),
            "unproved contiguous NN source"
        );
    }
    Ok(())
}
fn base_expr(n: usize) -> Expr {
    let tid = Expr::ThreadIdx(Dim::Y) * 32usize + Expr::ThreadIdx(Dim::X);
    Expr::BlockIdx(Dim::Y) * (64 * n)
        + Expr::BlockIdx(Dim::X) * 64usize
        + Expr::FloorDiv(Box::new(tid.clone()), 64) * n
        + Expr::Mod(Box::new(tid), 64)
}
// Exact affine coefficients in (1,blockY,blockX), after fixing one
// thread. Reject a non-affine/unresolved expression rather than infer equality
// from corner samples. Nonnegative checked u128 coefficients prove no wrapping
// for every admitted grid coordinate when the checked maximum fits u64.
fn affine(
    e: &Expr,
    tid: usize,
    defs: &std::collections::HashMap<String, Expr>,
) -> Option<[u128; 3]> {
    let f = |x: &Expr| affine(x, tid, defs);
    let constant = |v: u128| Some([v, 0, 0]);
    let divide = |a: [u128; 3], d: u128, rem: bool| {
        if d == 0 || !a[1].is_multiple_of(d) || !a[2].is_multiple_of(d) {
            return None;
        }
        Some(if rem {
            [a[0] % d, 0, 0]
        } else {
            [a[0] / d, a[1] / d, a[2] / d]
        })
    };
    match e {
        Expr::Const(v) => constant(u128::try_from(*v).ok()?),
        Expr::Var(v) => f(defs.get(v)?),
        Expr::ThreadIdx(Dim::X) => constant((tid % 32) as u128),
        Expr::ThreadIdx(Dim::Y) => constant((tid / 32) as u128),
        Expr::ThreadIdx(Dim::Z) | Expr::BlockIdx(Dim::Z) => constant(0),
        Expr::BlockIdx(Dim::Y) => Some([0, 1, 0]),
        Expr::BlockIdx(Dim::X) => Some([0, 0, 1]),
        Expr::BlockDim(Dim::X) => constant(32),
        Expr::BlockDim(Dim::Y) => constant(8),
        Expr::BlockDim(Dim::Z) => constant(1),
        Expr::Add(a, b) => {
            let (a, b) = (f(a)?, f(b)?);
            Some([
                a[0].checked_add(b[0])?,
                a[1].checked_add(b[1])?,
                a[2].checked_add(b[2])?,
            ])
        }
        Expr::Sub(a, b) => {
            let (a, b) = (f(a)?, f(b)?);
            Some([
                a[0].checked_sub(b[0])?,
                a[1].checked_sub(b[1])?,
                a[2].checked_sub(b[2])?,
            ])
        }
        Expr::Mul(a, b) => {
            let (mut a, mut b) = (f(a)?, f(b)?);
            if a[1] == 0 && a[2] == 0 {
                std::mem::swap(&mut a, &mut b);
            }
            if b[1] != 0 || b[2] != 0 {
                return None;
            }
            Some([
                a[0].checked_mul(b[0])?,
                a[1].checked_mul(b[0])?,
                a[2].checked_mul(b[0])?,
            ])
        }
        Expr::FloorDiv(a, d) => divide(f(a)?, *d as u128, false),
        Expr::Mod(a, d) => divide(f(a)?, *d as u128, true),
        Expr::ShiftRight(a, n) => divide(f(a)?, 1u128.checked_shl(u32::try_from(*n).ok()?)?, false),
        Expr::BitAnd(a, m) => {
            let d = u128::from(*m) + 1;
            if !d.is_power_of_two() {
                return None;
            }
            divide(f(a)?, d, true)
        }
    }
}
fn prove_stores(ir: &TileIR, case: Case) -> Result<()> {
    ensure!(
        case.m > 0 && case.n > 0 && case.m.is_multiple_of(64) && case.n.is_multiple_of(64),
        "partial output grid"
    );
    ensure!(
        ir.params.iter().filter(|p| !p.is_input).count() == 1
            && ir
                .params
                .iter()
                .any(|p| p.name == "output_0" && !p.is_input && p.dtype == DType::BF16),
        "non-BF16 output interface"
    );
    let defs = ir
        .body
        .stmts
        .iter()
        .filter_map(|s| match s {
            Stmt::LetIndex { name, value } => Some((name.clone(), value.clone())),
            _ => None,
        })
        .collect::<std::collections::HashMap<_, _>>();
    let stores = ir
        .body
        .stmts
        .iter()
        .filter_map(|s| match s {
            Stmt::StoreGlobalPredicated {
                row,
                col,
                element_index,
                ..
            } => Some((row, col, element_index)),
            _ => None,
        })
        .collect::<Vec<_>>();
    ensure!(stores.len() == 16, "expected sixteen output stores");
    for (round, (row, col, index)) in stores.iter().enumerate() {
        for tid in 0..256 {
            let t = round * 256 + tid;
            let (lr, lc) = (t / 64, t % 64);
            let expected = [(lr * case.n + lc) as u128, (64 * case.n) as u128, 64];
            ensure!(
                affine(row, tid, &defs) == Some([lr as u128, 64, 0])
                    && affine(col, tid, &defs) == Some([lc as u128, 0, 64])
                    && affine(index, tid, &defs) == Some(expected),
                "original output SSA is not the exact full-MN affine mapping"
            );
            let max = expected[0]
                + expected[1] * (case.m / 64 - 1) as u128
                + expected[2] * (case.n / 64 - 1) as u128;
            ensure!(
                max < (case.m as u128) * (case.n as u128) && max <= u64::MAX as u128,
                "output index overflow/bounds"
            );
        }
    }
    Ok(())
}
fn rewrite(ir: &mut TileIR, case: Case) -> Result<()> {
    prove_stores(ir, case)?;
    let count = ir
        .body
        .stmts
        .iter()
        .filter(|s| matches!(s, Stmt::StoreGlobalPredicated { .. }))
        .count();
    ensure!(
        count == 16,
        "expected exactly sixteen original scalar stores"
    );
    ensure!(
        !ir.body
            .stmts
            .iter()
            .any(|s| matches!(s,Stmt::LetIndex{name,..} if name==BASE)),
        "duplicate probe binding"
    );
    for s in &ir.body.stmts {
        if let Stmt::StoreGlobalPredicated {
            dest_param, layout, ..
        } = s
        {
            ensure!(
                dest_param == "output_0"
                    && layout.rows == case.m
                    && layout.cols == case.n
                    && layout.row_stride == case.n,
                "unexpected output interface"
            );
        }
    }
    // Validate the complete rewrite before committing any statement. Retain
    // original LetIndex definitions and never rerun placement: all instructions
    // through the exchange/read phase stay identical to the original program.
    let mut result = Vec::with_capacity(ir.body.stmts.len() + 1);
    let mut round = 0;
    for s in ir.body.stmts.clone() {
        if let Stmt::StoreGlobalPredicated {
            dest_param, src, ..
        } = s
        {
            if round == 0 {
                result.push(Stmt::LetIndex {
                    name: BASE.into(),
                    value: base_expr(case.n),
                });
            }
            result.push(Stmt::Store {
                dest_param,
                src,
                row_offset: Expr::Var(BASE.into()) + Expr::Const((round * 4 * case.n) as i64),
                col_offset: Expr::Const(0),
            });
            round += 1;
        } else {
            result.push(s);
        }
    }
    let mut proposed = ir.clone();
    proposed.body.stmts = result;
    proposed.validate_layouts()?;
    *ir = proposed;
    Ok(())
}
fn mainloop_prefix(source: &str) -> Result<&str> {
    let start = source.find("ld.param").context("parameter loads absent")?;
    let end = source
        .find("ld.shared.f32")
        .context("exchange scalar load absent")?;
    ensure!(end > start, "invalid prefix");
    Ok(&source[start..end])
}
fn eval(
    e: &Expr,
    tid: usize,
    block: (usize, usize),
    defs: &std::collections::HashMap<String, Expr>,
) -> u64 {
    let v = |e: &Expr| eval(e, tid, block, defs);
    match e {
        Expr::Const(x) => *x as u64,
        Expr::Var(x) => v(&defs[x]),
        Expr::ThreadIdx(Dim::X) => (tid % 32) as u64,
        Expr::ThreadIdx(Dim::Y) => (tid / 32) as u64,
        Expr::ThreadIdx(Dim::Z) => 0,
        Expr::BlockIdx(Dim::X) => block.1 as u64,
        Expr::BlockIdx(Dim::Y) => block.0 as u64,
        Expr::BlockIdx(Dim::Z) => 0,
        Expr::BlockDim(Dim::X) => 32,
        Expr::BlockDim(Dim::Y) => 8,
        Expr::BlockDim(Dim::Z) => 1,
        Expr::Add(a, b) => v(a).wrapping_add(v(b)),
        Expr::Sub(a, b) => v(a).wrapping_sub(v(b)),
        Expr::Mul(a, b) => v(a).wrapping_mul(v(b)),
        Expr::FloorDiv(a, b) => v(a) / *b as u64,
        Expr::Mod(a, b) => v(a) % *b as u64,
        Expr::ShiftRight(a, b) => v(a) >> *b,
        Expr::BitAnd(a, b) => v(a) & *b,
    }
}
#[test]
fn plain_output_full_mn_original_store_coverage_and_prefix() -> Result<()> {
    for case in CASES {
        let region = super::k64_diagnostic::fixture(case)?;
        guard(&region)?;
        let mut tiles = tile_graph(region.clone())?;
        tiles.optimize_indices()?;
        let mut original = tile_graph(region.clone())?;
        original.optimize_indices()?;
        let ir = &mut tiles.matmul_region_kernels[0].ir;
        let defs = ir
            .body
            .stmts
            .iter()
            .filter_map(|s| match s {
                Stmt::LetIndex { name, value } => Some((name.clone(), value.clone())),
                _ => None,
            })
            .collect::<std::collections::HashMap<_, _>>();
        let stores = ir
            .body
            .stmts
            .iter()
            .filter_map(|s| match s {
                Stmt::StoreGlobalPredicated {
                    row,
                    col,
                    element_index,
                    ..
                } => Some((row, col, element_index)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(stores.len(), 16);
        for block in [
            (0, 0),
            (case.m / 64 - 1, 0),
            (0, case.n / 64 - 1),
            (case.m / 64 - 1, case.n / 64 - 1),
        ] {
            let mut covered = vec![false; 4096];
            for (round, (row, col, index)) in stores.iter().enumerate() {
                for tid in 0..256 {
                    let t = round * 256 + tid;
                    let lr = t / 64;
                    let lc = t % 64;
                    let (r, c) = (eval(row, tid, block, &defs), eval(col, tid, block, &defs));
                    assert_eq!(
                        (r, c),
                        ((block.0 * 64 + lr) as u64, (block.1 * 64 + lc) as u64)
                    );
                    let value = eval(index, tid, block, &defs);
                    assert_eq!(
                        value,
                        eval(&base_expr(case.n), tid, block, &defs) + (round * 4 * case.n) as u64
                    );
                    assert_eq!(value, r * case.n as u64 + c);
                    assert!(value < (case.m * case.n) as u64);
                    assert!(!covered[lr * 64 + lc]);
                    covered[lr * 64 + lc] = true;
                }
            }
            assert!(covered.into_iter().all(|v| v));
        }
        rewrite(ir, case)?;
        let mut target = PtxTarget::sm80();
        target.compute_capability = (8, 9);
        target.matmul_resources.shared_bytes_per_sm = 100 * 1024;
        let a = PtxGraph::from(original).with_target(target).module_source();
        let b = PtxGraph::from(tiles).with_target(target).module_source();
        assert_eq!(mainloop_prefix(&a)?, mainloop_prefix(&b)?);
        assert_eq!(b.matches("st.global.b16").count(), 16);
        assert_eq!(b.matches("cvt.rn.bf16.f32").count(), 16);
        assert!(!b.contains("matmul_store_skip_"));
        let mut bad = tile_graph(region.clone())?
            .matmul_region_kernels
            .remove(0)
            .ir;
        bad.body
            .stmts
            .retain(|s| !matches!(s, Stmt::StoreGlobalPredicated { .. }));
        let snapshot = format!("{bad:?}");
        assert!(rewrite(&mut bad, case).is_err());
        assert_eq!(snapshot, format!("{bad:?}"));
        let mut bad = region.clone();
        bad.m += 1;
        assert!(guard(&bad).is_err());
        let mut bad = region.clone();
        bad.inputs[0].source_shape[0] -= 1;
        assert!(guard(&bad).is_err());
        let mut bad = region.clone();
        bad.epilogue_operations.push(crate::tile::RegionOp {
            node: bad.outputs[0].node,
            output: crate::tile::RegionValue(usize::MAX),
            kind: crate::tile::RegionOpKind::Unary {
                op: crate::tensor::UnaryOp::Neg,
                input: bad.matmul_value,
            },
        });
        assert!(guard(&bad).is_err());
        let mut bad = region.clone();
        let i = &mut bad.inputs[0];
        i.source_shape = vec![case.k, case.m];
        i.tensor = crate::tile::VirtualTensor::identity(
            i.tensor.source,
            i.source_shape.clone(),
            DType::BF16,
        )
        .permute(&[1, 0])?;
        assert!(guard(&bad).is_err());
        let mut bad = tile_graph(region)?.matmul_region_kernels.remove(0).ir;
        if let Some(Stmt::StoreGlobalPredicated { element_index, .. }) = bad
            .body
            .stmts
            .iter_mut()
            .find(|s| matches!(s, Stmt::StoreGlobalPredicated { .. }))
        {
            *element_index = Expr::Const(0);
        }
        let snapshot = format!("{bad:?}");
        assert!(rewrite(&mut bad, case).is_err());
        assert_eq!(snapshot, format!("{bad:?}"));
    }
    Ok(())
}
fn build_plain_output(
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
    let region = plan.matmul_regions()[id].clone();
    guard(&region)?;
    let selected = region.schedule;
    let mut baseline_tiles = tile_graph(region.clone())?;
    baseline_tiles.optimize_indices()?;
    let original = PtxGraph::from(baseline_tiles)
        .with_target(e.target)
        .module_source();
    let mut tiles = tile_graph(region)?;
    tiles.optimize_indices()?;
    rewrite(&mut tiles.matmul_region_kernels[0].ir, case)?;
    let ptx = PtxGraph::from(tiles).with_target(e.target);
    let source = ptx.module_source();
    ensure!(
        mainloop_prefix(&source)? == mainloop_prefix(&original)?,
        "mainloop/exchange prefix changed"
    );
    std::fs::write(dir.join("original_direct_region.ptx"), &original)?;
    let module = stream.context().load_module(Ptx::from_src(&source))?;
    let name = ptx
        .matmul_region_kernel_name(0)
        .context("plain BF16 output function absent")?;
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
    let resources = json!({"mode":"test_only_plain_output","function":name,"registers_per_thread":registers,"local_bytes":local,"shared_bytes":shared,"max_threads_per_block":max_threads,"static_active_block_capacity":capacity,"capacity_is_not_measured_occupancy":true,"schedule":format!("{selected:?}"),"required_resource_gate_passed":accepted,"production_selection_unchanged":true,"production_activation_claim":false});
    std::fs::write(dir.join("test_only_plain_output.ptx"), source)?;
    write_json(
        &dir.join("test_only_plain_output-resources.json"),
        &resources,
    )?;
    ensure!(
        accepted,
        "plain BF16 output actual resource admission failed: {resources}"
    );
    ensure!(
        shared == 19456,
        "plain BF16 output physical shared accounting mismatch"
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
            grid_dim: (u32::try_from(case.n / 64)?, u32::try_from(case.m / 64)?, 1),
            block_dim: selected.block_threads,
            shared_mem_bytes: 0,
        },
        schedule: selected,
        name: "test_only_plain_output",
        resources,
        samples_us: Vec::new(),
    })
}

#[test]
#[ignore = "manual test-only plain BF16 output paired component diagnostic; no production selector change"]
fn plain_output_component_resource_equivalence_and_paired_timing() -> Result<()> {
    let directory = PathBuf::from(
        std::env::var_os("TNSR_PLAIN_OUTPUT_DIAG_DIR")
            .context("set explicit plain BF16 output artifact directory")?,
    );
    std::fs::create_dir_all(&directory)?;
    let reverse = std::env::var("TNSR_PLAIN_OUTPUT_DIAG_REVERSE").as_deref() == Ok("1");
    let timing = std::env::var("TNSR_PLAIN_OUTPUT_DIAG_TIMING").as_deref() == Ok("1");
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
                build_plain_output(case, &stream, &a, &b, &dir)
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
                    if mode.name == "test_only_plain_output" {
                        candidate = us;
                    } else {
                        baseline = us;
                    }
                }
                pairs.push(json!({"sample":sample,"execution_order":order,"accepted_k32_us":baseline,"test_only_plain_output_us":candidate,"baseline_over_candidate":baseline/candidate}));
            }
            stream.synchronize()?;
            for mode in &modes {
                compare(
                    &expected,
                    &mode.read(&stream)?,
                    "plain BF16 output post-capture finite",
                )?;
            }
        }
        let summaries=modes.iter().map(|mode|json!({"mode":mode.name,"resources":mode.resources,"samples_us":mode.samples_us,"median_us":if timing{Some(median(&mode.samples_us))}else{None}})).collect::<Vec<_>>();
        let result = json!({"status":"qualified_test_fixture_only","case":case.name,"m":case.m,"n":case.n,"k":case.k,"orientation":"NN","qualification":checks,"timing_performed":timing,"post_capture_bits_equal":timing.then_some(true),"samples":if timing{samples}else{0},"graph_kernel_repeats":repeats,"warmup_pairs":warmups,"warmup_elapsed_ms":warmup_ms,"pairs":pairs,"summaries":summaries,"production_selector_changed":false,"scope":"Accepted64²K32 versus same kernel with plain full-MN scalar output bounds elided and canonical addresses; staging/MMA/exchange/scalarBF16RN unchanged; not complete updates"});
        write_json(&dir.join("result.json"), &result)?;
        results.push(result);
        write_json(
            &directory.join("report.json"),
            &json!({"cases":results,"construction_order":if reverse{"candidate,baseline"}else{"baseline,candidate"},"scope":"test-only plain BF16 output diagnostic; no production selector/API/cache/default changes"}),
        )?;
    }
    Ok(())
}
