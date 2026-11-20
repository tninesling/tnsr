use super::*;
use crate::tensor::{Parameter, TensorExpr};

fn fixture(
    sequence: usize,
    width: usize,
    bf16: bool,
    causal: bool,
) -> (TensorGraph<f32>, NodeIndex, NodeIndex) {
    fixture_bh(sequence, width, bf16, causal, 1, 1)
}
fn fixture_bh(
    sequence: usize,
    width: usize,
    bf16: bool,
    causal: bool,
    batch: usize,
    heads: usize,
) -> (TensorGraph<f32>, NodeIndex, NodeIndex) {
    let q = TensorExpr::input("q", vec![batch, heads, sequence, width]);
    let k = TensorExpr::input("k", vec![batch, heads, width, sequence]);
    let (q, k) = if bf16 {
        (q.cast(ValueDType::BF16), k.cast(ValueDType::BF16))
    } else {
        (q, k)
    };
    let qk = q.matmul(k).cast(ValueDType::F32);
    let mask = TensorExpr::constant(
        (0..sequence * sequence)
            .map(|i| {
                if causal && i % sequence > i / sequence {
                    f32::NEG_INFINITY
                } else {
                    0.
                }
            })
            .collect(),
        vec![1, 1, sequence, sequence],
    );
    let p = crate::nn::softmax(qk * 0.125 + mask, 3);
    let mut graph = TensorGraph::new();
    let narrowed = graph.add_expr(&p.cast(ValueDType::BF16), false);
    let probability = graph.inputs(narrowed)[0];
    (graph, probability, narrowed)
}
#[test]
fn matcher_requires_dtype_geometry_mask_and_closed_interiors() {
    assert!(!MixedOptions::default().attention_softmax_fusion);
    let (mut g, p, _) = fixture(512, 64, true, true);
    let matched = recognize(&g);
    assert_eq!(matched.len(), 1);
    assert_eq!(matched[0].output, p);
    let interior = *matched[0]
        .members
        .iter()
        .find(|&&n| {
            matches!(
                g[n],
                TensorGraphNode::Unary {
                    op: UnaryOp::Exp,
                    ..
                }
            )
        })
        .unwrap();
    let escape = g.graph.add_node(TensorGraphNode::Reshape {
        shape: g[interior].shape().clone(),
    });
    g.graph.add_edge(interior, escape, 0);
    assert!(recognize(&g).is_empty());
    for (s, d, bf16, causal) in [
        (128, 64, true, true),
        (512, 32, true, true),
        (512, 64, false, true),
        (512, 64, true, false),
    ] {
        assert!(recognize(&fixture(s, d, bf16, causal).0).is_empty());
    }
    let (mut g, _, _) = fixture(512, 64, true, true);
    let region = recognize(&g).remove(0);
    g.graph[region.output] = TensorGraphNode::Binary {
        op: BinaryOp::Div,
        shape: vec![1, 1, 512, 512],
    };
    let extra = g.graph.add_node(TensorGraphNode::Cast {
        dtype: ValueDType::BF16,
        shape: vec![1, 1, 512, 512],
    });
    g.graph.add_edge(region.output, extra, 0);
    assert!(recognize(&g).is_empty());
}
#[test]
fn compact_derivative_keeps_probability_fans_but_primitive_derivative_falls_back() {
    let (mut g, p, _) = fixture(512, 64, true, true);
    // Require differentiation through the normalization, rather than only
    // differentiating the weighting parameter downstream of it.
    let q_node = g
        .graph
        .node_indices()
        .find(|&n| matches!(g[n], TensorGraphNode::Input { name: "q", .. }))
        .unwrap();
    let q = Parameter::new(vec![0.125; 512 * 64], vec![1, 1, 512, 64]);
    g.graph[q_node] = TensorGraphNode::Parameter {
        id: q.id(),
        data: q.storage(),
        shape: q.shape().clone(),
    };
    // A nonconstant weighting avoids the trivial sum(softmax) gradient.
    let weights = g.graph.add_node(TensorGraphNode::Parameter {
        id: usize::MAX,
        data: Parameter::new(
            (0..512 * 512).map(|i| (i % 17) as f32 / 32.).collect(),
            vec![1, 1, 512, 512],
        )
        .storage()
        .clone(),
        shape: vec![1, 1, 512, 512],
    });
    let weighted = g.graph.add_node(TensorGraphNode::Binary {
        op: BinaryOp::Mul,
        shape: vec![1, 1, 512, 512],
    });
    g.graph.add_edge(p, weighted, 0);
    g.graph.add_edge(weights, weighted, 1);
    let mut last = weighted;
    let mut shape = vec![1, 1, 512, 512];
    for axis in [3, 2, 1, 0] {
        shape[axis] = 1;
        let n = g.graph.add_node(TensorGraphNode::ReduceAxis {
            op: ReduceOp::Mean,
            axis,
            shape: shape.clone(),
        });
        g.graph.add_edge(last, n, 0);
        last = n;
    }
    assert!(recognize(&g.clone().with_gradients(last)).is_empty());
    let compact = g.with_gradients_with_options(
        last,
        crate::graph::GradientOptions {
            compact_softmax: true,
        },
    );
    assert_eq!(recognize(&compact).len(), 1);
}
#[test]
fn row_kernel_typed_outputs_and_deterministic_tree_codegen() {
    let ir = row_ir(0.125).unwrap();
    assert_eq!(
        ir.params.iter().map(|p| p.dtype).collect::<Vec<_>>(),
        vec![DType::BF16, DType::F32, DType::F32, DType::BF16]
    );
    assert_eq!(ir.shared_mem_bytes, 512);
    // Two trees, each with a publication barrier, seven descending stages and
    // a final read-completion barrier. No divergent barrier is permitted.
    assert_eq!(
        ir.body
            .stmts
            .iter()
            .filter(|s| matches!(s, Stmt::Barrier))
            .count(),
        18
    );
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.region_kernels = vec![crate::tile::graph::TileRegionKernel { region_id: 0, ir }];
    tiles.optimize_indices().unwrap();
    let ptx = PtxGraph::from(tiles)
        .with_target(PtxTarget::sm80())
        .to_ptx();
    assert!(ptx.contains("ex2.approx.f32"));
    assert!(ptx.contains("cvt.rn.bf16.f32"));
    assert!(ptx.contains("bar.sync"));
    assert!(!ptx.contains("shfl.sync"));
}
#[test]
fn fused_probability_matches_baseline_nonfinites_replay_and_changed_feeds() {
    if !super::super::tests::cuda_available() {
        return;
    }
    let (g, p, narrowed) = fixture(512, 64, true, true);
    for replay in [false, true] {
        let mut baseline = MixedExecutor::new(&g, MatMulPrecision::NativeHalf, replay).unwrap();
        let mut candidate = MixedExecutor::new_with_options(
            &g,
            MatMulPrecision::NativeHalf,
            replay,
            MixedOptions {
                attention_softmax_fusion: true,
                ..MixedOptions::default()
            },
        )
        .unwrap();
        assert_eq!(candidate.info().attention_softmax_regions, 1);
        assert!(candidate.info().kernels < baseline.info().kernels);
        for revision in 0..3 {
            let mut q: Vec<_> = (0..512 * 64)
                .map(|i| ((i + revision) % 19) as f32 / 32. - 0.25)
                .collect();
            let k: Vec<_> = (0..512 * 64)
                .map(|i| ((i + revision) % 23) as f32 / 32. - 0.25)
                .collect();
            if revision == 2 {
                q[0] = f32::NAN;
                q[64] = f32::INFINITY;
                q[128] = f32::NEG_INFINITY;
            }
            let feeds = HashMap::from([("q".into(), q), ("k".into(), k)]);
            baseline.upload(feeds.clone()).unwrap();
            candidate.upload(feeds).unwrap();
            baseline.run().unwrap();
            candidate.run().unwrap();
            for node in [p, narrowed] {
                let expected = baseline.read(node).unwrap();
                let actual = candidate.read(node).unwrap();
                for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
                        "index {i}: {a:?} vs {b:?}"
                    );
                }
            }
        }
    }
}

// A lockstep interpreter of the emitted IR; shared stores publish only at a
// barrier. This exercises the actual schedule without a GPU or approximate exp.
struct RowMachine {
    row: usize,
    registers: Vec<HashMap<TileVar, f32>>,
    global: HashMap<String, Vec<f32>>,
    shared: Vec<f32>,
    pending: HashMap<usize, f32>,
}
impl RowMachine {
    fn index(&self, e: &Expr, lane: usize) -> usize {
        match e {
            Expr::Const(n) => usize::try_from(*n).unwrap(),
            Expr::ThreadIdx(Dim::X) => lane,
            Expr::BlockIdx(Dim::X) => self.row,
            Expr::Add(a, b) => self.index(a, lane) + self.index(b, lane),
            Expr::Mul(a, b) => self.index(a, lane) * self.index(b, lane),
            Expr::Mod(a, n) => self.index(a, lane) % n,
            _ => panic!("unexpected index {e:?}"),
        }
    }
    fn value(&self, e: &ScalarExpr, lane: usize) -> f32 {
        match e {
            ScalarExpr::Constant(v) => *v,
            ScalarExpr::Var(v) => self.registers[lane][v],
            ScalarExpr::Unary {
                op: UnaryOp::Exp,
                value,
            } => self.value(value, lane).exp(),
            ScalarExpr::Binary { op, lhs, rhs } => {
                let (a, b) = (self.value(lhs, lane), self.value(rhs, lane));
                match op {
                    ScalarBinaryOp::Max => a.max(b),
                    ScalarBinaryOp::AddRn => a + b,
                    ScalarBinaryOp::SubRn => a - b,
                    ScalarBinaryOp::MulRn => a * b,
                    ScalarBinaryOp::Arithmetic(op) => match op {
                        BinaryOp::Add => a + b,
                        BinaryOp::Sub => a - b,
                        BinaryOp::Mul => a * b,
                        BinaryOp::Div => a / b,
                    },
                }
            }
            _ => panic!("unexpected expression {e:?}"),
        }
    }
    fn block(&mut self, block: &Block, active: &[bool]) {
        for stmt in &block.stmts {
            if matches!(stmt, Stmt::AllocTile { .. }) {
                continue;
            }
            if matches!(stmt, Stmt::Barrier) {
                assert!(active.iter().all(|x| *x));
                for (i, v) in self.pending.drain() {
                    self.shared[i] = v;
                }
                continue;
            }
            if let Stmt::If {
                condition: ScalarPredicate::IndexLt { lhs, rhs },
                body,
            } = stmt
            {
                let enabled: Vec<_> = (0..THREADS)
                    .map(|l| active[l] && self.index(lhs, l) < self.index(rhs, l))
                    .collect();
                self.block(body, &enabled);
                continue;
            }
            for (lane, &enabled) in active.iter().enumerate() {
                if !enabled {
                    continue;
                }
                match stmt {
                    Stmt::SetScalar { dest, value } => {
                        let v = self.value(value, lane);
                        self.registers[lane].insert(*dest, v);
                    }
                    Stmt::LoadScalar {
                        dest,
                        source,
                        index,
                    } => {
                        let i = self.index(index, lane);
                        let v = match source {
                            ScalarMemory::Global(name) => self.global[name][i],
                            ScalarMemory::Shared(_) => self.shared[i],
                        };
                        self.registers[lane].insert(*dest, v);
                    }
                    Stmt::StoreScalar {
                        target,
                        index,
                        value,
                    } => {
                        let i = self.index(index, lane);
                        let mut v = self.registers[lane][value];
                        match target {
                            ScalarMemory::Shared(_) => {
                                self.pending.insert(i, v);
                            }
                            ScalarMemory::Global(name) => {
                                if name == "narrowed" {
                                    v = half::bf16::from_f32(v).to_f32();
                                }
                                self.global.get_mut(name).unwrap()[i] = v;
                            }
                        }
                    }
                    _ => panic!("unexpected statement {:?}", std::mem::discriminant(stmt)),
                }
            }
        }
    }
}
fn reference_row(qk: &[f32], mask: &[f32]) -> Vec<f32> {
    fn reduce(values: &[f32], max: bool) -> f32 {
        let mut partials = vec![if max { f32::NEG_INFINITY } else { 0. }; THREADS];
        for (lane, partial) in partials.iter_mut().enumerate() {
            for i in (lane..COLUMNS).step_by(THREADS) {
                *partial = if max {
                    (*partial).max(values[i])
                } else {
                    *partial + values[i]
                };
            }
        }
        for stride in [64, 32, 16, 8, 4, 2, 1] {
            for lane in 0..stride {
                partials[lane] = if max {
                    partials[lane].max(partials[lane + stride])
                } else {
                    partials[lane] + partials[lane + stride]
                };
            }
        }
        partials[0]
    }
    let scores: Vec<_> = qk.iter().zip(mask).map(|(q, m)| q * 0.125 + m).collect();
    let maximum = reduce(&scores, true);
    let exp: Vec<_> = scores.iter().map(|x| (x - maximum).exp()).collect();
    let denom = reduce(&exp, false);
    exp.into_iter().map(|x| x / denom).collect()
}
#[test]
fn emitted_row_ir_preserves_full_tree_and_nonfinite_ieee_arithmetic() {
    let ir = row_ir(0.125).unwrap();
    for (row, case) in [
        (0, 0),
        (127, 0),
        (511, 0),
        (31, 1),
        (31, 2),
        (31, 3),
        (31, 4),
    ] {
        let len = (row + 1) * COLUMNS;
        let mut qk: Vec<_> = (0..len)
            .map(|i| half::bf16::from_f32((i % 73) as f32 / 8. - 4.).to_f32())
            .collect();
        let mut mask = vec![f32::NEG_INFINITY; COLUMNS * COLUMNS];
        for i in 0..=row {
            mask[row * COLUMNS + i] = 0.;
        }
        match case {
            1 => qk[row * COLUMNS] = f32::NAN,
            2 => qk[row * COLUMNS] = f32::INFINITY,
            3 => qk[row * COLUMNS..].fill(f32::NEG_INFINITY),
            4 => mask[row * COLUMNS..(row + 1) * COLUMNS].fill(f32::NEG_INFINITY),
            _ => {}
        }
        let expected = reference_row(
            &qk[row * COLUMNS..],
            &mask[row * COLUMNS..(row + 1) * COLUMNS],
        );
        let mut machine = RowMachine {
            row,
            registers: vec![HashMap::new(); THREADS],
            global: HashMap::from([
                ("qk".into(), qk),
                ("mask".into(), mask),
                ("probability".into(), vec![f32::NAN; len]),
                ("narrowed".into(), vec![f32::NAN; len]),
            ]),
            shared: vec![f32::NAN; THREADS],
            pending: HashMap::new(),
        };
        machine.block(&ir.body, &[true; THREADS]);
        for (name, narrowed) in [("probability", false), ("narrowed", true)] {
            for (i, &v) in expected.iter().enumerate() {
                let actual = machine.global[name][row * COLUMNS + i];
                let expected = if narrowed {
                    half::bf16::from_f32(v).to_f32()
                } else {
                    v
                };
                assert!(
                    (actual.is_nan() && expected.is_nan())
                        || actual.to_bits() == expected.to_bits(),
                    "row {row} case {case} {name}[{i}]: {actual} vs {expected}"
                );
            }
        }
    }
}

fn comparison_diagnostic(expected: &[f32], actual: &[f32]) -> (bool, String) {
    let first = expected
        .iter()
        .zip(actual)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    let finite = expected.iter().chain(actual).all(|v| v.is_finite());
    let max_abs = expected
        .iter()
        .zip(actual)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
        .fold(0_f64, f64::max);
    let delta2: f64 = expected
        .iter()
        .zip(actual)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    let ref2: f64 = expected.iter().map(|&a| f64::from(a).powi(2)).sum();
    let first = first.map(|i| {
        format!(
            "index={i} expected={:?}({:08x}) actual={:?}({:08x})",
            expected[i],
            expected[i].to_bits(),
            actual[i],
            actual[i].to_bits()
        )
    });
    (
        expected.len() == actual.len() && first.is_none(),
        format!(
            "len={}/{} finite={finite} max_abs={max_abs:e} relative_l2={:e} first={first:?}",
            expected.len(),
            actual.len(),
            delta2.sqrt() / ref2.sqrt().max(1e-12)
        ),
    )
}
fn fixed_gradient_gate(expected: &[f32], actual: &[f32]) -> bool {
    if expected.len() != actual.len() || expected.iter().chain(actual).any(|v| !v.is_finite()) {
        return false;
    }
    let delta2: f64 = expected
        .iter()
        .zip(actual)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    let ref2: f64 = expected.iter().map(|&a| f64::from(a).powi(2)).sum();
    // The established full-model qualification gate; neither fitted to this
    // fixture nor changed for attention fusion.
    delta2.sqrt() / ref2.sqrt().max(1e-12) <= 0.02
}

fn attention_intermediate_diagnostics(
    graph: &TensorGraph<f32, WithGrad>,
    baseline: &MixedExecutor,
    candidate: &MixedExecutor,
    context: &str,
) {
    let region = recognize(graph).remove(0);
    let directory = std::path::Path::new("target/attention-debug");
    std::fs::create_dir_all(directory).unwrap();
    for (mode, executor) in [("baseline", baseline), ("candidate", candidate)] {
        for (i, program) in executor.programs.iter().enumerate() {
            if let Some(program) = program.downcast_ref::<Program<F32>>() {
                std::fs::write(
                    directory.join(format!("{mode}-{}-{i}.ptx", context.replace(' ', "_"))),
                    &program.ptx,
                )
                .unwrap();
                std::fs::write(
                    directory.join(format!("{mode}-{}-{i}.plan", context.replace(' ', "_"))),
                    &program.plan,
                )
                .unwrap();
            }
        }
        std::fs::write(
            directory.join(format!(
                "{mode}-{}-kernel-labels.txt",
                context.replace(' ', "_")
            )),
            executor.kernel_labels().join("\n"),
        )
        .unwrap();
    }
    std::fs::write(
        directory.join("graph.txt"),
        graph
            .toposort()
            .iter()
            .map(|&n| {
                format!(
                    "{} {} {:?} inputs={:?}",
                    n.index(),
                    graph[n].name(),
                    graph[n].shape(),
                    graph.inputs(n)
                )
            })
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    let mut nodes = vec![
        ("QK", region.qk),
        ("P32", region.output),
        ("P16", region.narrowed),
    ];
    for n in graph.toposort() {
        if graph[n].shape() != &region.shape {
            continue;
        }
        if n != region.qk && matches!(graph[n], TensorGraphNode::MatMul { .. }) {
            nodes.push(("dP", n));
        }
        if n != region.narrowed
            && matches!(
                graph[n],
                TensorGraphNode::Cast {
                    dtype: ValueDType::BF16,
                    ..
                }
            )
        {
            nodes.push(("dScore32", graph.inputs(n)[0]));
            nodes.push(("dScore16", n));
        }
    }
    for (label, n) in nodes {
        match (baseline.read(n), candidate.read(n)) {
            (Ok(a), Ok(b)) => eprintln!(
                "{context} {label} node={} {}",
                n.index(),
                comparison_diagnostic(&a, &b).1
            ),
            (a, b) => eprintln!(
                "{context} {label} node={} unavailable baseline={:?} candidate={:?}",
                n.index(),
                a.err(),
                b.err()
            ),
        }
    }
}

#[test]
fn fused_compact_attention_training_preserves_parameter_gradients() {
    if !super::super::tests::cuda_available() {
        return;
    }
    let (mut graph, _, narrowed) = fixture(512, 64, true, true);
    let q_node = graph
        .graph
        .node_indices()
        .find(|&n| matches!(graph[n], TensorGraphNode::Input { name: "q", .. }))
        .unwrap();
    let q = Parameter::new(
        (0..512 * 64)
            .map(|i| (i % 17) as f32 / 32. - 0.25)
            .collect(),
        vec![1, 1, 512, 64],
    );
    graph.graph[q_node] = TensorGraphNode::Parameter {
        id: q.id(),
        data: q.storage(),
        shape: q.shape().clone(),
    };
    let value = graph.add_expr(
        &TensorExpr::input("v", vec![1, 1, 512, 64]).cast(ValueDType::BF16),
        false,
    );
    let context = graph.graph.add_node(TensorGraphNode::MatMul {
        shape: vec![1, 1, 512, 64],
    });
    graph.graph.add_edge(narrowed, context, 0);
    graph.graph.add_edge(value, context, 1);
    let promoted = graph.graph.add_node(TensorGraphNode::Cast {
        dtype: ValueDType::F32,
        shape: vec![1, 1, 512, 64],
    });
    graph.graph.add_edge(context, promoted, 0);
    let mut loss = promoted;
    let mut shape = vec![1, 1, 512, 64];
    for axis in [3, 2, 1, 0] {
        shape[axis] = 1;
        let n = graph.graph.add_node(TensorGraphNode::ReduceAxis {
            op: ReduceOp::Mean,
            axis,
            shape: shape.clone(),
        });
        graph.graph.add_edge(loss, n, 0);
        loss = n;
    }
    let graph = graph.with_gradients_with_options(
        loss,
        crate::graph::GradientOptions {
            compact_softmax: true,
        },
    );
    assert_eq!(recognize(&graph).len(), 1);
    for replay in [false, true] {
        let mut baseline = MixedExecutor::new(&graph, MatMulPrecision::NativeHalf, replay).unwrap();
        let mut candidate = MixedExecutor::new_with_options(
            &graph,
            MatMulPrecision::NativeHalf,
            replay,
            MixedOptions {
                attention_softmax_fusion: true,
                ..MixedOptions::default()
            },
        )
        .unwrap();
        assert_eq!(candidate.info().attention_softmax_regions, 1);
        // Keep the owner alive throughout compilation/replay/readback. Prefix
        // reuse must retain QK past its skipped promotion through fused P.
        let parameters = vec![(q.clone(), 0)];
        let groups = vec![crate::ptx::resident::AdamGroup::new(1e-4)];
        let mut owner = if replay {
            crate::ptx::resident::ResidentAdam::new_with_graph_replay(parameters, groups)
        } else {
            crate::ptx::resident::ResidentAdam::new(parameters, groups)
        }
        .unwrap();
        owner.begin_step().unwrap();
        let mut resident = owner
            .mixed_executor_with_options(
                &graph,
                MatMulPrecision::NativeHalf,
                MixedOptions {
                    attention_softmax_fusion: true,
                    incremental_allocation_reuse: true,
                    ..MixedOptions::default()
                },
            )
            .unwrap();
        assert_eq!(resident.info().attention_softmax_regions, 1);
        assert!(resident.info().reused_allocations > 0);
        assert_eq!(resident.info().replay, replay);
        for revision in 0..2 {
            let feeds = HashMap::from([
                (
                    "k".into(),
                    (0..512 * 64)
                        .map(|i| ((i + revision) % 23) as f32 / 32. - 0.25)
                        .collect(),
                ),
                (
                    "v".into(),
                    (0..512 * 64)
                        .map(|i| ((i + revision) % 29) as f32 / 32. - 0.25)
                        .collect(),
                ),
            ]);
            baseline.upload(feeds.clone()).unwrap();
            candidate.upload(feeds.clone()).unwrap();
            resident.upload(feeds).unwrap();
            baseline.run().unwrap();
            candidate.run().unwrap();
            resident.run().unwrap();
            assert_eq!(baseline.read(loss).unwrap(), candidate.read(loss).unwrap());
            let region = recognize(&graph).remove(0);
            let mut exact_nodes = vec![("P32", region.output), ("P16", region.narrowed)];
            exact_nodes.extend(
                graph
                    .graph
                    .node_indices()
                    .filter(|&n| {
                        n != region.qk
                            && graph[n].shape() == &region.shape
                            && matches!(graph[n], TensorGraphNode::MatMul { .. })
                    })
                    .map(|n| ("dP", n)),
            );
            for (label, node) in exact_nodes {
                let expected = baseline.read(node).unwrap();
                let actual = candidate.read(node).unwrap();
                let (exact, diagnostic) = comparison_diagnostic(&expected, &actual);
                assert!(
                    exact,
                    "{label} replay={replay} revision={revision}: {diagnostic}"
                );
            }
            let expected = baseline.gradients(&graph).unwrap();
            assert_eq!(baseline.read(loss).unwrap(), resident.read(loss).unwrap());
            let comparisons = [
                (false, candidate.gradients(&graph).unwrap()),
                (true, resident.gradients(&graph).unwrap()),
            ];
            let (same_residency, residency_diagnostic) =
                comparison_diagnostic(&comparisons[0].1[&q.id()], &comparisons[1].1[&q.id()]);
            assert!(
                same_residency,
                "fused residency mismatch replay={replay} revision={revision}: {residency_diagnostic}"
            );
            let mut failure = None;
            for (is_resident, actual) in comparisons {
                let context = format!("replay={replay} revision={revision} resident={is_resident}");
                let (_, diagnostic) = comparison_diagnostic(&expected[&q.id()], &actual[&q.id()]);
                // Removing the forward region can make the ordinary planner
                // fuse P*dP into its sum: FFMA replaces a stored FMUL product
                // followed by FADD. P32/P16 and dP are exact above; residency
                // must remain exact. This baseline comparison uses the same
                // immutable 2% relative-L2 gate as full-model qualification.
                let qualified = fixed_gradient_gate(&expected[&q.id()], &actual[&q.id()]);
                eprintln!("Q-gradient {context} qualified={qualified}: {diagnostic}");
                if !qualified {
                    eprintln!("Q-gradient {context}: {diagnostic}");
                    attention_intermediate_diagnostics(
                        &graph,
                        &baseline,
                        if is_resident { &resident } else { &candidate },
                        &context,
                    );
                    failure = Some(format!("Q-gradient {context}: {diagnostic}"));
                }
            }
            assert!(failure.is_none(), "{}", failure.unwrap_or_default());
        }
    }
}

#[test]
fn matcher_proves_batch_head_replication_and_rejects_escaping_mask_views() {
    for (batch, heads) in [(2, 2), (8, 8), (1, 16)] {
        let (mut graph, p, _) = fixture_bh(512, 64, true, true, batch, heads);
        let regions = recognize(&graph);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].output, p);
        assert_eq!(regions[0].shape, vec![batch, heads, 512, 512]);
        assert_eq!(graph[regions[0].mask].shape(), &[1, 1, 512, 512]);
        let wrapper = *regions[0]
            .members
            .iter()
            .find(|&&n| matches!(graph[n], TensorGraphNode::BroadcastAxis { axis: 0 | 1, .. }))
            .unwrap();
        let escape = graph.graph.add_node(TensorGraphNode::Reshape {
            shape: graph[wrapper].shape().clone(),
        });
        graph.graph.add_edge(wrapper, escape, 0);
        assert!(recognize(&graph).is_empty());
    }
}
#[test]
fn fused_batch_head_mask_replication_matches_baseline() {
    if !super::super::tests::cuda_available() {
        return;
    }
    let (graph, p, narrowed) = fixture_bh(512, 64, true, true, 2, 2);
    for replay in [false, true] {
        let mut baseline = MixedExecutor::new(&graph, MatMulPrecision::NativeHalf, replay).unwrap();
        let mut candidate = MixedExecutor::new_with_options(
            &graph,
            MatMulPrecision::NativeHalf,
            replay,
            MixedOptions {
                attention_softmax_fusion: true,
                ..MixedOptions::default()
            },
        )
        .unwrap();
        assert_eq!(candidate.info().attention_softmax_regions, 1);
        for revision in 0..2 {
            let feeds = HashMap::from([
                (
                    "q".into(),
                    (0..4 * 512 * 64)
                        .map(|i| ((i + revision) % 19) as f32 / 32. - 0.25)
                        .collect(),
                ),
                (
                    "k".into(),
                    (0..4 * 512 * 64)
                        .map(|i| ((i + revision) % 23) as f32 / 32. - 0.25)
                        .collect(),
                ),
            ]);
            baseline.upload(feeds.clone()).unwrap();
            candidate.upload(feeds).unwrap();
            baseline.run().unwrap();
            candidate.run().unwrap();
            for node in [p, narrowed] {
                assert_eq!(
                    baseline
                        .read(node)
                        .unwrap()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    candidate
                        .read(node)
                        .unwrap()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}
