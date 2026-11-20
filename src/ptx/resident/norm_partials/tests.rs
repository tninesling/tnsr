use super::*;
use crate::tensor::{Parameter, ReduceOp};
use crate::tile::{ReductionSchedule, RegionOpKind};

#[test]
fn global_rows_repair_coverage_and_interval_validity() {
    let layout = PartialLayout::new(
        vec![0..3, 3..1024, 1024..2049, 2049..4096, 4096..4103],
        5120,
    )
    .unwrap();
    assert_eq!(layout.repair_rows, vec![0, 2, 4]);
    assert_eq!(layout.rows, 5);
    let mut writes = vec![0; layout.rows];
    for range in &layout.ranges {
        for (row, writes) in writes
            .iter_mut()
            .enumerate()
            .take(range.end.div_ceil(1024))
            .skip(range.start / 1024)
        {
            if row * 1024 >= range.start && (row + 1) * 1024 <= range.end {
                *writes += 1;
            }
        }
    }
    for &row in &layout.repair_rows {
        writes[row as usize] += 1;
    }
    assert_eq!(writes, vec![1; layout.rows]);
    for (ranges, pad) in [
        (vec![], 1024),
        (std::iter::once(1..2).collect(), 1024),
        (vec![0..1, 2..3], 1024),
        (std::iter::once(0..0).collect(), 0),
        (std::iter::once(0..1).collect(), 2048),
    ] {
        assert!(PartialLayout::new(ranges, pad).is_err());
    }
    let mut state = Validity::default();
    assert!(!state.valid());
    state.begin_step();
    let first = state.begin_accumulation(true, true, true);
    assert!(first);
    assert!(!state.valid());
    state.finish_accumulation(first, true);
    assert!(state.valid());
    let second = state.begin_accumulation(true, true, false);
    assert!(!second);
    assert!(!state.valid());
    state.finish_accumulation(second, true);
    assert!(!state.valid());
    state.begin_step();
    let first = state.begin_accumulation(true, true, true);
    state.finish_accumulation(first, false);
    assert!(!state.valid());
    state.begin_step();
    assert!(!state.valid());
    assert!(!state.begin_accumulation(true, false, true));
}
#[test]
fn old_norm_ir_policy_and_literal_second_stage() {
    let input: TensorExpr<f32> = Input::new("g", vec![257, 1024]).into();
    let graph: TensorGraph<f32> = (input.clone() * input).reduce_sum(1).reduce_sum(0).into();
    let plan = crate::ptx::plan::PtxExecutionPlan::build_with_reduction_mode(
        &graph,
        PtxReductionMode::DeterministicTree,
    )
    .unwrap();
    let first = plan
        .reduction_regions()
        .iter()
        .find(|r| r.axis == 1)
        .unwrap();
    assert_eq!(first.schedule, ReductionSchedule::Block { threads: 128 });
    assert_eq!(first.input_shape, vec![257, 1024]);
    assert_eq!(first.output_shape, vec![257, 1]);
    assert!(
        first.producer_operations.iter().any(
            |op| matches!(op.kind,RegionOpKind::Binary {op:BinaryOp::Mul,lhs,rhs} if lhs==rhs)
        )
    );
    let (second, input, output) = second_stage_graph(257).unwrap();
    assert_eq!(second[input].shape(), &[257, 1]);
    assert!(matches!(
        second[output],
        TensorGraphNode::ReduceAxis {
            op: ReduceOp::Sum,
            axis: 0,
            ..
        }
    ));
    let second_plan = crate::ptx::plan::PtxExecutionPlan::build_with_reduction_mode(
        &second,
        PtxReductionMode::DeterministicTree,
    )
    .unwrap();
    assert_eq!(second_plan.reduction_regions().len(), 1);
    let r = &second_plan.reduction_regions()[0];
    assert_eq!(r.input_shape, vec![257, 1]);
    assert_eq!(r.schedule, ReductionSchedule::Block { threads: 128 });
    let view = VirtualTensor::identity(NodeIndex::new(0), vec![2048], DType::F32);
    let ir = accumulate_ir(&view, &[2048], 128..2176).unwrap();
    let repair = repair_ir().unwrap();
    let directory = std::path::Path::new("target/norm-partials-audit");
    std::fs::create_dir_all(directory).unwrap();
    for (name, ir) in [
        ("baseline-stage1", first.lower_to_tile_ir(0).unwrap()),
        ("fused-accumulation", ir),
        ("repair", repair),
    ] {
        let mut tiles = crate::tile::TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.region_kernels = vec![crate::tile::graph::TileRegionKernel { region_id: 0, ir }];
        tiles.optimize_indices().unwrap();
        let mut target = crate::ptx::PtxTarget::sm80();
        target.compute_capability = (8, 9);
        let ptx = crate::ptx::PtxGraph::from(tiles)
            .with_target(target)
            .to_ptx();
        std::fs::write(directory.join(format!("{name}.ptx")), &ptx).unwrap();
        assert!(ptx.contains("mul.f32"));
        assert!(ptx.contains("add.f32"));
        assert!(ptx.contains("bar.sync"));
        assert!(!ptx.contains("atom."));
    }
}

// Execute the emitted scalar IR in lockstep. Delayed shared publication and
// barriers under only uniform predicates catch tail early-return mistakes.
struct Machine {
    block: usize,
    regs: Vec<HashMap<TileVar, f32>>,
    indices: HashMap<String, usize>,
    globals: HashMap<String, Vec<f32>>,
    rows: Vec<u32>,
    shared: Vec<f32>,
    pending: HashMap<usize, f32>,
    writes: Vec<usize>,
    fma: bool,
}
impl Machine {
    fn index(&self, e: &Expr, lane: usize) -> usize {
        match e {
            Expr::Const(n) => *n as usize,
            Expr::Var(n) => self.indices[n],
            Expr::ThreadIdx(Dim::X) => lane,
            Expr::BlockIdx(Dim::X) => self.block,
            Expr::Add(a, b) => self.index(a, lane).wrapping_add(self.index(b, lane)),
            Expr::Sub(a, b) => self.index(a, lane).wrapping_sub(self.index(b, lane)),
            Expr::Mul(a, b) => self.index(a, lane).wrapping_mul(self.index(b, lane)),
            Expr::FloorDiv(a, n) => self.index(a, lane) / n,
            Expr::Mod(a, n) => self.index(a, lane) % n,
            Expr::ShiftRight(a, n) => self.index(a, lane) >> n,
            Expr::BitAnd(a, n) => self.index(a, lane) & (*n as usize),
            _ => panic!("unexpected index {e:?}"),
        }
    }
    fn value(&self, e: &ScalarExpr, lane: usize) -> f32 {
        match e {
            ScalarExpr::Constant(v) => *v,
            ScalarExpr::Var(v) => self.regs[lane][v],
            ScalarExpr::Binary {
                op: ScalarBinaryOp::Arithmetic(op),
                lhs,
                rhs,
            } => {
                if self.fma
                    && *op == BinaryOp::Add
                    && let ScalarExpr::Binary {
                        op: ScalarBinaryOp::Arithmetic(BinaryOp::Mul),
                        lhs: a,
                        rhs: b,
                    } = rhs.as_ref()
                {
                    return self
                        .value(a, lane)
                        .mul_add(self.value(b, lane), self.value(lhs, lane));
                }
                let (a, b) = (self.value(lhs, lane), self.value(rhs, lane));
                match op {
                    BinaryOp::Add => a + b,
                    BinaryOp::Sub => a - b,
                    BinaryOp::Mul => a * b,
                    BinaryOp::Div => a / b,
                }
            }
            _ => panic!("unexpected expression {e:?}"),
        }
    }
    fn predicate(&self, p: &ScalarPredicate, lane: usize) -> bool {
        match p {
            ScalarPredicate::IndexLt { lhs, rhs } => self.index(lhs, lane) < self.index(rhs, lane),
            _ => panic!("unexpected predicate"),
        }
    }
    fn run(&mut self, body: &Block, active: &[bool]) {
        if !active.iter().any(|&a| a) {
            return;
        }
        for stmt in &body.stmts {
            match stmt {
                Stmt::AllocTile { .. } => continue,
                Stmt::Barrier => {
                    assert!(active.iter().all(|&a| a));
                    for (i, v) in self.pending.drain() {
                        self.shared[i] = v;
                    }
                    continue;
                }
                Stmt::LoadIndex {
                    name,
                    source,
                    index,
                } => {
                    assert_eq!(source, "rows");
                    self.indices
                        .insert(name.clone(), self.rows[self.index(index, 0)] as usize);
                    continue;
                }
                Stmt::If { condition, body } => {
                    let next: Vec<_> = (0..128)
                        .map(|lane| active[lane] && self.predicate(condition, lane))
                        .collect();
                    self.run(body, &next);
                    continue;
                }
                _ => {}
            }
            for (lane, &enabled) in active.iter().enumerate() {
                if !enabled {
                    continue;
                }
                match stmt {
                    Stmt::SetScalar { dest, value } => {
                        let v = self.value(value, lane);
                        self.regs[lane].insert(*dest, v);
                    }
                    Stmt::LoadScalar {
                        dest,
                        source,
                        index,
                    } => {
                        let i = self.index(index, lane);
                        let v = match source {
                            ScalarMemory::Global(n) => self.globals[n][i],
                            ScalarMemory::Shared(_) => self.shared[i],
                        };
                        self.regs[lane].insert(*dest, v);
                    }
                    Stmt::StoreScalar {
                        target,
                        index,
                        value,
                    } => {
                        let i = self.index(index, lane);
                        let v = self.regs[lane][value];
                        match target {
                            ScalarMemory::Shared(_) => {
                                self.pending.insert(i, v);
                            }
                            ScalarMemory::Global(n) => {
                                self.globals.get_mut(n).unwrap()[i] = v;
                                if n == "partials" {
                                    self.writes[i] += 1;
                                }
                            }
                        }
                    }
                    _ => panic!("unexpected statement {:?}", std::mem::discriminant(stmt)),
                }
            }
        }
    }
}
fn row_reference(g: &[f32], fma: bool) -> f32 {
    let mut v = vec![0_f32; 128];
    for (t, part) in v.iter_mut().enumerate() {
        for k in 0..8 {
            let g = g[t + k * 128];
            *part = if fma {
                g.mul_add(g, *part)
            } else {
                *part + g * g
            };
        }
    }
    for stride in [64, 32, 16, 8, 4, 2, 1] {
        for t in 0..stride {
            v[t] += v[t + stride];
        }
    }
    v[0]
}
fn equal_bits(a: f32, b: f32) {
    assert!(
        (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
        "{a:?} vs {b:?}"
    );
}
#[test]
fn actual_accumulate_and_repair_ir_exact_bits_views_zero_nonfinite_and_dirty_padding() {
    for fma in [false, true] {
        for case in 0..5 {
            let layout = PartialLayout::new(
                vec![0..3, 3..1024, 1024..2049, 2049..4096, 4096..4103],
                5120,
            )
            .unwrap();
            let mut resident = vec![f32::NAN; layout.padded_len];
            resident[4103..].fill(if case == 4 { 0.25 } else { 0. });
            let mut partials = vec![f32::NAN; layout.rows];
            let mut counts = vec![0; layout.rows];
            for range in &layout.ranges {
                let mut source: Vec<_> = (0..range.len())
                    .map(|i| {
                        if i % 31 == 0 {
                            -0.
                        } else {
                            (i % 41) as f32 / 64. - 0.25
                        }
                    })
                    .collect();
                if case == 1 {
                    source.fill(0.);
                }
                if case == 2 {
                    source[0] = f32::INFINITY;
                }
                if case == 3 {
                    source[0] = f32::NAN;
                }
                let view =
                    VirtualTensor::identity(NodeIndex::new(0), vec![range.len()], DType::F32);
                let ir = accumulate_ir(&view, &[range.len()], range.clone()).unwrap();
                let mut m = Machine {
                    block: 0,
                    regs: vec![HashMap::new(); 128],
                    indices: HashMap::new(),
                    globals: HashMap::from([
                        ("gradient".into(), source.clone()),
                        ("accumulated".into(), vec![f32::NAN; range.len()]),
                        ("partials".into(), partials),
                    ]),
                    rows: vec![],
                    shared: vec![f32::NAN; 128],
                    pending: HashMap::new(),
                    writes: counts,
                    fma,
                };
                for block in 0..range.end.div_ceil(1024) - range.start / 1024 {
                    m.block = block;
                    m.run(&ir.body, &[true; 128]);
                }
                for (i, &x) in source.iter().enumerate() {
                    equal_bits(m.globals["accumulated"][i], 0. + x);
                    resident[range.start + i] = m.globals["accumulated"][i];
                }
                partials = m.globals.remove("partials").unwrap();
                counts = m.writes;
            }
            let ir = repair_ir().unwrap();
            let mut m = Machine {
                block: 0,
                regs: vec![HashMap::new(); 128],
                indices: HashMap::new(),
                globals: HashMap::from([
                    ("gradient".into(), resident.clone()),
                    ("partials".into(), partials),
                ]),
                rows: layout.repair_rows.clone(),
                shared: vec![f32::NAN; 128],
                pending: HashMap::new(),
                writes: counts,
                fma,
            };
            for block in 0..layout.repair_rows.len() {
                m.block = block;
                m.run(&ir.body, &[true; 128]);
            }
            assert_eq!(m.writes, vec![1; layout.rows]);
            for row in 0..layout.rows {
                equal_bits(
                    m.globals["partials"][row],
                    row_reference(&resident[row * 1024..(row + 1) * 1024], fma),
                );
            }
        }
    }
    // A rectangular virtual transpose uses the exact same source indexing as
    // ordinary accumulate_view. The short range is entirely a repair row.
    let view = VirtualTensor::identity(NodeIndex::new(0), vec![3, 7], DType::BF16)
        .permute(&[1, 0])
        .unwrap();
    let ir = accumulate_ir(&view, &[3, 7], 0..21).unwrap();
    let values: Vec<_> = (0..21)
        .map(|i| half::bf16::from_f32(i as f32 / 7.).to_f32())
        .collect();
    let mut m = Machine {
        block: 0,
        regs: vec![HashMap::new(); 128],
        indices: HashMap::new(),
        globals: HashMap::from([
            ("gradient".into(), values.clone()),
            ("accumulated".into(), vec![f32::NAN; 21]),
            ("partials".into(), vec![f32::NAN]),
        ]),
        rows: vec![],
        shared: vec![f32::NAN; 128],
        pending: HashMap::new(),
        writes: vec![0],
        fma: true,
    };
    m.run(&ir.body, &[true; 128]);
    for row in 0..7 {
        for col in 0..3 {
            equal_bits(
                m.globals["accumulated"][row * 3 + col],
                values[col * 7 + row],
            );
        }
    }
}

fn cuda_available() -> bool {
    let ok = cudarc::driver::CudaContext::new(0).is_ok();
    assert!(
        ok || std::env::var_os("TNSR_REQUIRE_CUDA").is_none(),
        "CUDA required"
    );
    ok
}
fn training_fixture(
    lengths: &[usize],
    mixed: bool,
) -> (
    Vec<Parameter<f32>>,
    TensorGraph<f32, crate::graph::WithGrad>,
) {
    let parameters: Vec<_> = lengths
        .iter()
        .map(|&len| Parameter::new(vec![0.125; len], vec![len]))
        .collect();
    let mut loss = TensorExpr::constant(vec![0.], vec![1]);
    for (i, p) in parameters.iter().enumerate() {
        let value = TensorExpr::from(p.clone());
        let value = if mixed {
            value
                .cast(crate::tensor::ValueDType::BF16)
                .cast(crate::tensor::ValueDType::F32)
        } else {
            value
        };
        let names = ["g0", "g1", "g2", "g3", "g4"];
        loss = loss + (value * TensorExpr::input(names[i], vec![lengths[i]])).reduce_sum(0);
    }
    let mut graph = TensorGraph::new();
    let root = graph.add_expr(&loss, false);
    (parameters, graph.with_gradients(root))
}
#[test]
fn owner_norm_partials_exact_norm_clip_adam_replay_and_microbatch_fallback() {
    use crate::ptx::mixed::MixedOptions;
    use crate::ptx::resident::{AdamGroup, ResidentAdam};
    if !cuda_available() {
        return;
    }
    for mixed in [false, true] {
        for replay in [false, true] {
            for lengths in [
                vec![3, 1021, 1025, 2047, 7],
                vec![16384, 16385, 17],
                vec![131072, 131073, 19],
            ] {
                let (pa, ga) = training_fixture(&lengths, mixed);
                let (pb, gb) = training_fixture(&lengths, mixed);
                let make = |parameters: &[Parameter<f32>], partial: bool| {
                    let list = parameters.iter().cloned().map(|p| (p, 0)).collect();
                    let groups = vec![AdamGroup::new(1e-4)];
                    let mut owner = if replay {
                        ResidentAdam::new_with_graph_replay(list, groups)
                    } else {
                        ResidentAdam::new(list, groups)
                    }
                    .unwrap();
                    assert!(owner.set_experimental_gradient_norm_partials(true).is_err());
                    owner
                        .set_experimental_first_accumulation_overwrite(true)
                        .unwrap();
                    owner
                        .set_experimental_gradient_norm_partials(partial)
                        .unwrap();
                    owner.begin_step().unwrap();
                    owner
                };
                let mut a = make(&pa, false);
                let mut b = make(&pb, true);
                let mut ea = if mixed {
                    None
                } else {
                    Some(a.executor().unwrap())
                };
                let mut eb = if mixed {
                    None
                } else {
                    Some(b.executor().unwrap())
                };
                let options = MixedOptions {
                    direct_master_bindings: true,
                    ..MixedOptions::default()
                };
                let mut ma = if mixed {
                    Some(
                        a.mixed_executor_with_options(
                            &ga,
                            crate::tile::MatMulPrecision::Tf32x3,
                            options,
                        )
                        .unwrap(),
                    )
                } else {
                    None
                };
                let mut mb = if mixed {
                    Some(
                        b.mixed_executor_with_options(
                            &gb,
                            crate::tile::MatMulPrecision::Tf32x3,
                            options,
                        )
                        .unwrap(),
                    )
                } else {
                    None
                };
                for (step, microbatches) in [1, 2, 0, 1].into_iter().enumerate() {
                    a.begin_step().unwrap();
                    b.begin_step().unwrap();
                    // Dirty real elements must never leak into first overwrite. Padding
                    // remains the separately tested persistent-zero owner invariant.
                    for owner in [&mut a, &mut b] {
                        let gradient = Arc::get_mut(&mut owner.gradients).unwrap();
                        owner
                            .stream
                            .memcpy_htod(&[f32::NAN; 3], &mut gradient.slice_mut(0..3))
                            .unwrap();
                    }
                    for micro in 0..microbatches {
                        let feeds: HashMap<_, _> = lengths
                            .iter()
                            .enumerate()
                            .map(|(i, &len)| {
                                let names = ["g0", "g1", "g2", "g3", "g4"];
                                (
                                    names[i].to_string(),
                                    (0..len)
                                        .map(|j| {
                                            if j % 47 == 0 {
                                                -0.
                                            } else {
                                                let seed =
                                                    (j as u32).wrapping_mul(1664525).wrapping_add(
                                                        ((step + micro) as u32)
                                                            .wrapping_mul(1013904223),
                                                    );
                                                let x = f32::from_bits(
                                                    0x3e00_0000 | (seed & 0x007f_ffff),
                                                );
                                                if j % 211 == 0 {
                                                    f32::from_bits(1 + (seed & 0x007f_ffff))
                                                } else if step == 1 && j % 7 == 0 {
                                                    x * 0.00001
                                                } else if step == 3 && j % 13 == 0 {
                                                    x * 128.
                                                } else {
                                                    x
                                                }
                                            }
                                        })
                                        .collect(),
                                )
                            })
                            .collect();
                        if mixed {
                            a.accumulate_mixed(
                                ma.as_mut().unwrap(),
                                &ga,
                                feeds.clone(),
                                HashMap::new(),
                            )
                            .unwrap();
                            b.accumulate_mixed(mb.as_mut().unwrap(), &gb, feeds, HashMap::new())
                                .unwrap();
                        } else {
                            a.accumulate(ea.as_mut().unwrap(), &ga, feeds.clone())
                                .unwrap();
                            b.accumulate(eb.as_mut().unwrap(), &gb, feeds).unwrap();
                        }
                        assert_eq!(b.norm_partial_validity.valid(), micro == 0);
                    }
                    let sa = a.step(1.).unwrap();
                    let sb = b.step(1.).unwrap();
                    let audit = std::path::Path::new("target/norm-partials-audit");
                    std::fs::create_dir_all(audit).unwrap();
                    std::fs::write(
                        audit.join(format!(
                            "actual-baseline-mixed{mixed}-replay{replay}-rows{}.ptx",
                            a.padded_len / 1024
                        )),
                        a.norm.executor.module_source().unwrap(),
                    )
                    .unwrap();
                    assert_eq!(
                        sa.gradient_norm.to_bits(),
                        sb.gradient_norm.to_bits(),
                        "mixed={mixed} replay={replay} step={step}"
                    );
                    assert_eq!(sa.gradient_scale.to_bits(), sb.gradient_scale.to_bits());
                    assert_eq!(
                        b.gradient_norm_partials_status().last_step_used,
                        microbatches == 1
                    );
                    let ag = a.gradients().unwrap();
                    let bg = b.gradients().unwrap();
                    let am = a.moments().unwrap();
                    let bm = b.moments().unwrap();
                    for (p, q) in pa.iter().zip(&pb) {
                        assert_eq!(
                            ag[&p.id()].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            bg[&q.id()].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                        );
                        assert_eq!(p.values().unwrap(), q.values().unwrap());
                        assert_eq!(am[&p.id()], bm[&q.id()]);
                    }
                    let mut pad = vec![0.; b.padded_len];
                    b.stream
                        .memcpy_dtoh(b.gradients.as_ref(), &mut pad)
                        .unwrap();
                    assert!(
                        pad[lengths.iter().sum::<usize>()..]
                            .iter()
                            .all(|&x| x == 0.)
                    );
                }
                if mixed {
                    for executor in [ma.as_ref().unwrap(), mb.as_ref().unwrap()] {
                        assert_eq!(executor.info().replay, replay);
                        assert_eq!(executor.info().direct_master_parameters, lengths.len());
                        assert_eq!(executor.transfers().parameter_copies, 0);
                    }
                } else if replay {
                    for executor in [ea.as_ref().unwrap(), eb.as_ref().unwrap()] {
                        assert!(
                            executor.replay_metrics().graph_launches > 0,
                            "{:?}",
                            executor.replay_metrics()
                        );
                        assert_eq!(executor.replay_metrics().fallbacks, 0);
                    }
                }
                // An attempted bad feed invalidates the partial fastpath before work,
                // and existing owner failure semantics forbid retry/step publication.
                b.begin_step().unwrap();
                let error = if mixed {
                    b.accumulate_mixed(mb.as_mut().unwrap(), &gb, HashMap::new(), HashMap::new())
                } else {
                    b.accumulate(eb.as_mut().unwrap(), &gb, HashMap::new())
                };
                assert!(error.is_err());
                assert!(!b.norm_partial_validity.valid());
                assert!(b.step(1.).is_err());
                b.begin_step().unwrap();
                assert!(!b.norm_partial_validity.valid());
                let status = b.step(1.).unwrap();
                assert_eq!(status.gradient_norm.to_bits(), 0_f64.to_bits());
                assert!(!b.gradient_norm_partials_status().last_step_used);
            }
        }
    }
}
#[test]
fn owner_norm_partials_nonfinite_rejects_without_publication() {
    use crate::ptx::resident::{AdamGroup, ResidentAdam};
    if !cuda_available() {
        return;
    }
    for value in [f32::NAN, f32::INFINITY, f32::MAX] {
        let (parameters, graph) = training_fixture(&[1025], false);
        let mut owner = ResidentAdam::new_with_graph_replay(
            vec![(parameters[0].clone(), 0)],
            vec![AdamGroup::new(1e-4)],
        )
        .unwrap();
        owner
            .set_experimental_first_accumulation_overwrite(true)
            .unwrap();
        owner.set_experimental_gradient_norm_partials(true).unwrap();
        let mut executor = owner.executor().unwrap();
        owner.begin_step().unwrap();
        owner
            .accumulate(
                &mut executor,
                &graph,
                HashMap::from([("g0".into(), vec![value; 1025])]),
            )
            .unwrap();
        let before = parameters[0].values().unwrap();
        assert!(owner.step(1.).is_err());
        assert_eq!(parameters[0].values().unwrap(), before);
    }
}
