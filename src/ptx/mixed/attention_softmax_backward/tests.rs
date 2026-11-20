use super::*;
use crate::tensor::{Parameter, TensorExpr};
use crate::tile::TileVar;

fn fixture(
    sequence: usize,
    batch: usize,
    heads: usize,
    compact: bool,
) -> (TensorGraph<f32, WithGrad>, NodeIndex, Parameter<f32>) {
    let shape = vec![batch, heads, sequence, 64];
    let q = Parameter::new(
        (0..batch * heads * sequence * 64)
            .map(|i| (i % 17) as f32 / 32. - 0.25)
            .collect(),
        shape.clone(),
    );
    let query = TensorExpr::from(q.clone()).cast(ValueDType::BF16);
    let key = TensorExpr::input("k", vec![batch, heads, 64, sequence]).cast(ValueDType::BF16);
    let value = TensorExpr::input("v", shape).cast(ValueDType::BF16);
    let mask = TensorExpr::constant(
        (0..sequence * sequence)
            .map(|i| {
                if i % sequence > i / sequence {
                    f32::NEG_INFINITY
                } else {
                    0.
                }
            })
            .collect(),
        vec![1, 1, sequence, sequence],
    );
    let probability = crate::nn::softmax(query.matmul(key).cast(ValueDType::F32) * 0.125 + mask, 3);
    let loss = probability
        .cast(ValueDType::BF16)
        .matmul(value)
        .cast(ValueDType::F32)
        .mean_all();
    let mut graph = TensorGraph::new();
    let loss = graph.add_expr(&loss, false);
    (
        graph.with_gradients_with_options(
            loss,
            crate::graph::GradientOptions {
                compact_softmax: compact,
            },
        ),
        loss,
        q,
    )
}
#[test]
fn backward_matcher_closed_bf16_boundary_and_geometry() {
    assert!(!MixedOptions::default().attention_softmax_backward_fusion);
    for (batch, heads) in [(1, 1), (2, 2), (8, 8), (1, 16)] {
        let (graph, _, _) = fixture(512, batch, heads, true);
        let regions = recognize(&graph);
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        assert_eq!(r.shape, vec![batch, heads, 512, 512]);
        assert_eq!(graph.inputs(r.narrowed), vec![r.output]);
        assert!(!r.members.contains(&r.probability));
        assert!(!r.members.contains(&r.incoming));
        let consumers: Vec<_> = graph
            .graph
            .neighbors_directed(r.output, Direction::Outgoing)
            .collect();
        assert_eq!(consumers, vec![r.narrowed]);
    }
    assert!(recognize(&fixture(128, 1, 1, true).0).is_empty());
    assert!(recognize(&fixture(513, 1, 1, true).0).is_empty());
    assert!(recognize(&fixture(512, 1, 1, false).0).is_empty());
    for escape_final in [false, true] {
        let (mut graph, _, _) = fixture(512, 1, 1, true);
        let r = recognize(&graph).remove(0);
        let node = if escape_final {
            r.output
        } else {
            *r.members
                .iter()
                .find(|&&n| {
                    matches!(
                        graph[n],
                        TensorGraphNode::ReduceAxis {
                            op: ReduceOp::Sum,
                            ..
                        }
                    )
                })
                .unwrap()
        };
        let escape = graph.graph.add_node(TensorGraphNode::Reshape {
            shape: graph[node].shape().clone(),
        });
        graph.graph.add_edge(node, escape, 0);
        assert!(recognize(&graph).is_empty());
    }
    let (mut graph, _, _) = fixture(512, 1, 1, true);
    let r = recognize(&graph).remove(0);
    graph.graph[r.incoming] = TensorGraphNode::Cast {
        dtype: ValueDType::BF16,
        shape: r.shape.clone(),
    };
    assert!(recognize(&graph).is_empty());
}
#[test]
fn backward_row_ir_tree_and_dtype_audit() {
    let ir = row_ir().unwrap();
    assert_eq!(
        ir.params.iter().map(|p| p.dtype).collect::<Vec<_>>(),
        vec![DType::F32, DType::BF16, DType::BF16]
    );
    assert_eq!(ir.shared_mem_bytes, 512);
    assert_eq!(
        ir.body
            .stmts
            .iter()
            .filter(|s| matches!(s, Stmt::Barrier))
            .count(),
        9
    );
    let stages: Vec<_> = ir
        .body
        .stmts
        .iter()
        .filter_map(|s| {
            if let Stmt::If {
                condition:
                    ScalarPredicate::IndexLt {
                        lhs: Expr::ThreadIdx(Dim::X),
                        rhs: Expr::Const(stride),
                    },
                ..
            } = s
            {
                Some(*stride)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(stages, vec![64, 32, 16, 8, 4, 2, 1]);
    let stores = ir
        .body
        .stmts
        .iter()
        .filter(|s| {
            matches!(
                s,
                Stmt::StoreScalar {
                    target: ScalarMemory::Global(_),
                    ..
                }
            )
        })
        .count();
    assert_eq!(stores, 4);
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.region_kernels = vec![crate::tile::graph::TileRegionKernel { region_id: 0, ir }];
    tiles.optimize_indices().unwrap();
    let ptx = PtxGraph::from(tiles)
        .with_target(PtxTarget::sm80())
        .to_ptx();
    assert!(ptx.contains("cvt.rn.bf16.f32"));
    assert!(!ptx.contains("ex2.approx"));
    assert!(!ptx.contains("shfl.sync"));
}

// Lockstep interpretation of the actual emitted row IR. Shared stores publish
// only at a uniform barrier; CPU arithmetic represents the noncontracted member
// of the permitted fused-producer policy, with a separate f64 reference below.
struct Machine {
    row: usize,
    registers: Vec<HashMap<TileVar, f32>>,
    globals: HashMap<String, Vec<f32>>,
    shared: Vec<f32>,
    pending: HashMap<usize, f32>,
}
impl Machine {
    fn index(&self, e: &Expr, lane: usize) -> usize {
        match e {
            Expr::Const(n) => usize::try_from(*n).unwrap(),
            Expr::ThreadIdx(Dim::X) => lane,
            Expr::BlockIdx(Dim::X) => self.row,
            Expr::Add(a, b) => self.index(a, lane) + self.index(b, lane),
            Expr::Mul(a, b) => self.index(a, lane) * self.index(b, lane),
            _ => panic!("unexpected index {e:?}"),
        }
    }
    fn value(&self, e: &ScalarExpr, lane: usize) -> f32 {
        match e {
            ScalarExpr::Constant(v) => *v,
            ScalarExpr::Var(v) => self.registers[lane][v],
            ScalarExpr::Binary {
                op: ScalarBinaryOp::Arithmetic(op),
                lhs,
                rhs,
            } => {
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
    fn block(&mut self, block: &Block, active: &[bool]) {
        for stmt in &block.stmts {
            if matches!(stmt, Stmt::AllocTile { .. }) {
                continue;
            }
            if matches!(stmt, Stmt::Barrier) {
                assert!(active.iter().all(|v| *v));
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
                            ScalarMemory::Global(name) => self.globals[name][i],
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
                        let v = self.registers[lane][value];
                        match target {
                            ScalarMemory::Shared(_) => {
                                self.pending.insert(i, v);
                            }
                            ScalarMemory::Global(name) => {
                                self.globals.get_mut(name).unwrap()[i] =
                                    half::bf16::from_f32(v).to_f32();
                            }
                        }
                    }
                    _ => panic!("unexpected statement {:?}", std::mem::discriminant(stmt)),
                }
            }
        }
    }
}
fn reference(p: &[f32], g: &[f32]) -> Vec<f32> {
    let mut partials = vec![0_f32; THREADS];
    for (lane, partial) in partials.iter_mut().enumerate() {
        for i in (lane..COLUMNS).step_by(THREADS) {
            *partial += p[i] * g[i];
        }
    }
    for stride in [64, 32, 16, 8, 4, 2, 1] {
        for lane in 0..stride {
            partials[lane] += partials[lane + stride];
        }
    }
    p.iter()
        .zip(g)
        .map(|(&p, &g)| half::bf16::from_f32((p * (g - partials[0])) * 0.125).to_f32())
        .collect()
}
#[test]
fn backward_emitted_ir_zero_nonfinite_rows_and_bf16_rounding() {
    let ir = row_ir().unwrap();
    for (row, case) in [
        (0, 0),
        (1, 0),
        (512, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
    ] {
        let len = (row + 1) * COLUMNS;
        let mut p: Vec<_> = (0..len).map(|i| (i % 23) as f32 / 8192.).collect();
        let mut g: Vec<_> = (0..len)
            .map(|i| half::bf16::from_f32((i % 19) as f32 / 16. - 0.5).to_f32())
            .collect();
        match case {
            1 => g.fill(0.),
            2 => {
                p.fill(0.);
                g[0] = f32::INFINITY;
            }
            3 => g[0] = f32::NAN,
            4 => p.fill(f32::NAN),
            5 => {
                g[0] = f32::INFINITY;
                g[1] = f32::NEG_INFINITY;
            }
            _ => {}
        }
        let expected = reference(&p[row * COLUMNS..], &g[row * COLUMNS..]);
        let mut machine = Machine {
            row,
            registers: vec![HashMap::new(); THREADS],
            globals: HashMap::from([
                ("probability".into(), p),
                ("incoming".into(), g),
                ("narrowed".into(), vec![f32::NAN; len]),
            ]),
            shared: vec![f32::NAN; THREADS],
            pending: HashMap::new(),
        };
        machine.block(&ir.body, &[true; THREADS]);
        if case <= 1 {
            let p = &machine.globals["probability"][row * COLUMNS..];
            let g = &machine.globals["incoming"][row * COLUMNS..];
            let dot64: f64 = p
                .iter()
                .zip(g)
                .map(|(&p, &g)| f64::from(p) * f64::from(g))
                .sum();
            let magnitude: f64 = p
                .iter()
                .zip(g)
                .map(|(&p, &g)| (f64::from(p) * f64::from(g)).abs())
                .sum();
            // Standard gamma_(2n) bound for at most n rounded products and n
            // additions; FFMA uses fewer roundings. Include gradual-underflow
            // absolute errors without introducing a data-fitted tolerance.
            let rounds = (2 * COLUMNS) as f64;
            let u = 2_f64.powi(-24);
            let gamma = rounds * u / (1. - rounds * u);
            let bound = gamma * magnitude + rounds * f64::from(f32::from_bits(1));
            assert!((f64::from(machine.shared[0]) - dot64).abs() <= bound);
        }
        for (i, &expected) in expected.iter().enumerate() {
            let actual = machine.globals["narrowed"][row * COLUMNS + i];
            assert!(
                (expected.is_nan() && actual.is_nan()) || expected.to_bits() == actual.to_bits(),
                "row={row} case={case} index={i}: {actual} vs {expected}"
            );
        }
    }
}
fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let numerator: f64 = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    let denominator: f64 = a.iter().map(|&a| f64::from(a).powi(2)).sum();
    numerator.sqrt() / denominator.sqrt().max(1e-12)
}
#[test]
fn backward_mixed_training_replay_and_incremental_lifetimes() {
    if !super::super::tests::cuda_available() {
        return;
    }
    for (batch, heads) in [(1, 1), (2, 2)] {
        let (graph, loss, q) = fixture(512, batch, heads, true);
        let descriptor = recognize(&graph).remove(0);
        for replay in [false, true] {
            let options = MixedOptions {
                attention_softmax_fusion: true,
                attention_softmax_backward_fusion: true,
                ..MixedOptions::default()
            };
            let mut baseline = MixedExecutor::new_with_options(
                &graph,
                MatMulPrecision::NativeHalf,
                replay,
                MixedOptions {
                    attention_softmax_fusion: true,
                    ..MixedOptions::default()
                },
            )
            .unwrap();
            let mut candidate = MixedExecutor::new_with_options(
                &graph,
                MatMulPrecision::NativeHalf,
                replay,
                options,
            )
            .unwrap();
            let groups = vec![crate::ptx::resident::AdamGroup::new(1e-4)];
            let parameters = vec![(q.clone(), 0)];
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
                        incremental_allocation_reuse: true,
                        ..options
                    },
                )
                .unwrap();
            assert_eq!(candidate.info().attention_softmax_backward_regions, 1);
            assert_eq!(resident.info().attention_softmax_backward_regions, 1);
            assert_eq!(resident.info().replay, replay);
            assert!(!candidate.values.contains_key(&descriptor.output));
            assert!(candidate.info().kernels < baseline.info().kernels);
            for revision in 0..2 {
                let feeds = HashMap::from([
                    (
                        "k".into(),
                        (0..batch * heads * 512 * 64)
                            .map(|i| ((i + revision) % 23) as f32 / 32. - 0.25)
                            .collect(),
                    ),
                    (
                        "v".into(),
                        (0..batch * heads * 512 * 64)
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
                assert_eq!(candidate.read(loss).unwrap(), resident.read(loss).unwrap());
                for n in [descriptor.probability, descriptor.incoming] {
                    assert_eq!(baseline.read(n).unwrap(), candidate.read(n).unwrap());
                }
                let expected = baseline.gradients(&graph).unwrap();
                let actual = candidate.gradients(&graph).unwrap();
                let pooled = resident.gradients(&graph).unwrap();
                assert!(
                    expected[&q.id()]
                        .iter()
                        .chain(actual[&q.id()].iter())
                        .all(|v| v.is_finite())
                );
                let error = rel_l2(&expected[&q.id()], &actual[&q.id()]);
                assert!(
                    error <= 0.02,
                    "batch={batch} heads={heads} replay={replay} revision={revision} Q-gradient relative-L2={error}"
                );
                assert_eq!(
                    actual[&q.id()]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    pooled[&q.id()]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}
