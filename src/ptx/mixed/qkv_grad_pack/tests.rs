use super::*;
use crate::tensor::TensorExpr;
use crate::tile::{DType, ScalarValue};

fn fixture(zero: f32) -> TensorGraph<f32> {
    let (b, s, h, d) = (2, 5, 2, 3);
    let w = h * d;
    let q = TensorExpr::input("q", vec![b, h, s, d])
        .cast(ValueDType::BF16)
        .permute(vec![0, 2, 1, 3])
        .reshape(vec![b, s, w]);
    let k = TensorExpr::input("k", vec![b, h, d, s])
        .cast(ValueDType::BF16)
        .transpose()
        .permute(vec![0, 2, 1, 3])
        .reshape(vec![b, s, w]);
    let v = TensorExpr::input("v", vec![b, h, s, d])
        .cast(ValueDType::BF16)
        .permute(vec![0, 2, 1, 3])
        .reshape(vec![b, s, w]);
    let z = |n| TensorExpr::constant(vec![zero], vec![]).broadcast(vec![b, s, n]);
    let q = TensorExpr::concat(&[q, z(2 * w)], 2).cast(ValueDType::BF16);
    let k = TensorExpr::concat(&[z(w), k, z(w)], 2).cast(ValueDType::BF16);
    let v = TensorExpr::concat(&[z(2 * w), v], 2).cast(ValueDType::BF16);
    ((v + k) + q).into()
}
fn raw_values(region: &QkvGradPack, g: &TensorGraph<f32>) -> HashMap<NodeIndex, Value> {
    region
        .inputs
        .iter()
        .map(|input| {
            let shape = g[input.source].shape().clone();
            (
                input.source,
                Value {
                    binding: Binding {
                        pointer: input.source.index() as u64 + 1,
                        len: shape.iter().product(),
                        dtype: DType::BF16,
                    },
                    view: VirtualTensor::identity(input.source, shape.clone(), DType::BF16),
                    source_shape: shape,
                    chain: Vec::new(),
                },
            )
        })
        .collect()
}
#[test]
fn qkv_pack_matches_actual_full_decoder_and_preserves_all_primal_narrows() {
    use super::super::test_decoder::{Config, synthetic};
    for (width, heads, sequence, batch, hidden, vocab) in
        [(16, 4, 5, 2, 24, 31), (512, 8, 512, 32, 1408, 4096)]
    {
        let model = synthetic(Config {
            width,
            layers: 4,
            heads,
            hidden,
            sequence,
            batch,
            vocab,
            epsilon: 1e-5,
            learning_rate: 1e-4,
            max_grad_norm: 1.0,
        });
        let (g, _) = model.training_graph().unwrap();
        let narrow: HashSet<_> = g
            .graph
            .node_indices()
            .filter(|n| matches!(g[*n], TensorGraphNode::Narrow { .. }))
            .collect();
        assert_eq!(narrow.len(), 12);
        let regions = recognize(&g);
        assert_eq!(regions.len(), 4);
        for r in regions {
            assert_eq!(r.shape, vec![batch, sequence, 3 * width]);
            assert!(r.members.is_disjoint(&narrow));
            assert!(
                r.members
                    .iter()
                    .all(|n| g.gradient_metadata().gradient_nodes.contains(n))
            );
            let mut windows: Vec<_> = r.inputs.iter().map(|i| i.window).collect();
            windows.sort_unstable();
            assert_eq!(windows, vec![0, 1, 2]);
            assert!(r.inputs.iter().all(|i| !r.members.contains(&i.source)));
        }
    }
}
#[test]
fn qkv_pack_rejects_escapes_nonpositive_zero_dtypes_arithmetic_and_predicated_sources() {
    let base = fixture(0.0);
    let regions = recognize(&base);
    assert_eq!(regions.len(), 1);
    for zero in [-0.0, 1.0, f32::NAN] {
        assert!(recognize(&fixture(zero)).is_empty());
    }
    let r = &regions[0];
    for member in r.members.iter().copied().filter(|n| *n != r.output) {
        let mut g = base.clone();
        let sink = g.graph.add_node(TensorGraphNode::Reshape {
            shape: g[member].shape().clone(),
        });
        g.graph.add_edge(member, sink, 0);
        assert!(recognize(&g).is_empty(), "escaped {}", g[member].name());
    }
    let mut g = base.clone();
    let cast = *r
        .members
        .iter()
        .find(|n| {
            matches!(g[**n], TensorGraphNode::Cast { .. })
                && matches!(g[g.inputs(**n)[0]], TensorGraphNode::Concat { .. })
        })
        .unwrap();
    g.graph[cast] = TensorGraphNode::Cast {
        dtype: ValueDType::F32,
        shape: r.shape.clone(),
    };
    assert!(recognize(&g).is_empty());
    let mut g = base.clone();
    let concat = *r
        .members
        .iter()
        .find(|n| matches!(g[**n], TensorGraphNode::Concat { .. }))
        .unwrap();
    let input = g
        .inputs(concat)
        .into_iter()
        .find(|n| !matches!(g[*n], TensorGraphNode::BroadcastAxis { .. }))
        .unwrap();
    let edge = g.graph.find_edge(input, concat).unwrap();
    let ordinal = g.graph[edge];
    g.graph.remove_edge(edge);
    let one = g.graph.add_node(TensorGraphNode::Constant {
        shape: g[input].shape().clone(),
        data: Arc::new(vec![1.0; g[input].shape().iter().product()]),
    });
    let arithmetic = g.graph.add_node(TensorGraphNode::Binary {
        op: BinaryOp::Mul,
        shape: g[input].shape().clone(),
    });
    g.graph.add_edge(input, arithmetic, 0);
    g.graph.add_edge(one, arithmetic, 1);
    g.graph.add_edge(arithmetic, concat, ordinal);
    assert!(recognize(&g).is_empty());
    // The runtime descriptor check also rejects OOB metadata even if a caller
    // supplies a malformed internal Value rather than an ordinary graph view.
    let mut values = raw_values(r, &base);
    let source = r.inputs[0].source;
    values.get_mut(&source).unwrap().view.out_of_bounds = Some(ScalarValue::F32(0));
    assert!(logical_inputs(r, &values).is_err());
    let mut values = raw_values(r, &base);
    values.get_mut(&source).unwrap().view.predicate =
        Some(crate::tile::IndexPredicate::And(Vec::new()));
    assert!(logical_inputs(r, &values).is_err());
    // Rank-two physical MatMul output rewrites cannot satisfy the raw BHSD
    // prototype, even if a later reshape could manufacture the same elements.
    let mut rank_two = base.clone();
    rank_two.graph[source] = TensorGraphNode::MatMul { shape: vec![10, 6] };
    assert!(recognize(&rank_two).is_empty());
}
#[test]
fn qkv_pack_ir_keeps_bf16_rounds_selected_loads_and_no_collectives() {
    let g = fixture(0.0);
    let regions = recognize(&g);
    let r = &regions[0];
    let values = logical_inputs(r, &raw_values(r, &g)).unwrap();
    let ir = kernel::ir(r, &values).unwrap();
    assert_eq!(ir.shared_mem_bytes, 1024);
    assert!(ir.params.iter().all(|p| p.dtype == DType::BF16));
    fn counts(stmts: &[crate::tile::Stmt]) -> (usize, usize, usize) {
        use crate::tile::Stmt;
        let mut result = (0, 0, 0);
        for s in stmts {
            match s {
                Stmt::LoadScalar {
                    source: crate::tile::ScalarMemory::Global(_),
                    ..
                } => result.0 += 1,
                Stmt::StoreScalar {
                    target: crate::tile::ScalarMemory::Shared(_),
                    ..
                } => result.1 += 1,
                Stmt::If { body, .. } => {
                    let inner = counts(&body.stmts);
                    result.0 += inner.0;
                    result.1 += inner.1;
                    result.2 += inner.2;
                }
                Stmt::Barrier | Stmt::WarpReduce { .. } => result.2 += 1,
                _ => {}
            }
        }
        result
    }
    assert_eq!(counts(&ir.body.stmts), (3, 4, 0));
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.region_kernels = vec![crate::tile::graph::TileRegionKernel { region_id: 0, ir }];
    let ptx = PtxGraph::from(tiles)
        .with_target(PtxTarget::sm80())
        .to_ptx();
    assert!(ptx.contains("cvt.rn.bf16.f32"));
    assert!(ptx.contains("ld.shared.b16"));
    assert!(!ptx.contains("bar.sync"));
}

fn assert_same(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
            "index {i}: {a:?}/{:08x} vs {b:?}/{:08x}",
            a.to_bits(),
            b.to_bits()
        );
    }
}

#[test]
fn qkv_pack_gpu_replay_changed_feeds_and_nonfinites() {
    if !super::super::tests::cuda_available() {
        return;
    }
    let graph = fixture(0.0);
    let root = recognize(&graph)[0].output;
    for replay in [false, true] {
        let mut baseline = MixedExecutor::new(&graph, MatMulPrecision::NativeHalf, replay).unwrap();
        let mut candidate = MixedExecutor::new_with_options(
            &graph,
            MatMulPrecision::NativeHalf,
            replay,
            MixedOptions {
                qkv_grad_pack_fusion: true,
                ..MixedOptions::default()
            },
        )
        .unwrap();
        assert_eq!(candidate.info().qkv_grad_pack_regions, 1);
        assert!(candidate.info().kernels < baseline.info().kernels);
        for revision in 0..4 {
            let pattern = [
                0.,
                -0.,
                1. + 1. / 256.,
                -1. - 3. / 256.,
                f32::from_bits(1),
                f32::MIN_POSITIVE,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::from_bits(0x7fc1_0000),
                f32::from_bits(0x7f81_0000),
            ];
            let feeds = ["q", "k", "v"]
                .into_iter()
                .enumerate()
                .map(|(j, name)| {
                    (
                        name.into(),
                        (0..60)
                            .map(|i| pattern[(i + 3 * j + revision) % pattern.len()])
                            .collect(),
                    )
                })
                .collect::<HashMap<_, _>>();
            baseline.upload(feeds.clone()).unwrap();
            candidate.upload(feeds).unwrap();
            baseline.run().unwrap();
            candidate.run().unwrap();
            assert_same(
                &baseline.read(root).unwrap(),
                &candidate.read(root).unwrap(),
            );
        }
    }
}

#[test]
fn qkv_pack_gpu_actual_decoder_updates_replay_and_incremental_lifetimes() {
    if !super::super::tests::cuda_available() {
        return;
    }
    use super::super::test_decoder::{Config, synthetic};
    use crate::ptx::resident::{AdamGroup, ResidentAdam};
    let config = Config {
        width: 16,
        layers: 4,
        heads: 4,
        hidden: 24,
        sequence: 5,
        batch: 2,
        vocab: 31,
        epsilon: 1e-5,
        learning_rate: 1e-4,
        max_grad_norm: 1.0,
    };
    let models = [synthetic(config.clone()), synthetic(config)];
    let graphs = models
        .iter()
        .map(|m| m.training_graph().unwrap())
        .collect::<Vec<_>>();
    let mut owners = models
        .iter()
        .map(|m| {
            ResidentAdam::new_with_graph_replay(
                m.parameters
                    .iter()
                    .map(|(_, p)| (p.parameter.clone(), 0))
                    .collect(),
                vec![AdamGroup::new(1e-4)],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    for o in &mut owners {
        o.begin_step().unwrap();
    }
    let mut executors = owners
        .iter_mut()
        .enumerate()
        .map(|(i, o)| {
            o.mixed_executor_with_options(
                &graphs[i].0,
                MatMulPrecision::Tf32x3,
                MixedOptions {
                    qkv_grad_pack_fusion: i == 1,
                    incremental_allocation_reuse: true,
                    direct_master_bindings: true,
                    transposed_bf16_outputs: true,
                    attention_softmax_fusion: true,
                    attention_softmax_backward_fusion: true,
                    ..MixedOptions::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(executors[0].info().qkv_grad_pack_regions, 0);
    assert_eq!(executors[1].info().qkv_grad_pack_regions, 4);
    eprintln!(
        "decoder packing kernels {} -> {}",
        executors[0].info().kernels,
        executors[1].info().kernels
    );
    assert!(executors[1].info().kernels < executors[0].info().kernels);
    for step in 0..3 {
        for i in 0..2 {
            if step > 0 {
                owners[i].begin_step().unwrap();
            }
            owners[i]
                .accumulate_mixed(
                    &mut executors[i],
                    &graphs[i].0,
                    HashMap::new(),
                    models[i].config.feeds().unwrap(),
                )
                .unwrap();
            owners[i].step(1.0).unwrap();
        }
        assert_same(
            &executors[0].read(graphs[0].1).unwrap(),
            &executors[1].read(graphs[1].1).unwrap(),
        );
        let grads = [
            owners[0].gradients().unwrap(),
            owners[1].gradients().unwrap(),
        ];
        let moments = [owners[0].moments().unwrap(), owners[1].moments().unwrap()];
        for ((name, a), (_, b)) in models[0].parameters.iter().zip(models[1].parameters.iter()) {
            let (a, b) = (&a.parameter, &b.parameter);
            let _ = name;
            assert_same(&grads[0][&a.id()], &grads[1][&b.id()]);
            assert_same(&a.values().unwrap(), &b.values().unwrap());
            assert_same(&moments[0][&a.id()].first, &moments[1][&b.id()].first);
            assert_same(&moments[0][&a.id()].second, &moments[1][&b.id()].second);
        }
    }
}

#[test]
fn qkv_pack_gpu_raw_bf16_signed_zero_and_nan_payload_diagnostic() {
    if !super::super::tests::cuda_available() {
        return;
    }
    let graph = fixture(0.0);
    let descriptor = recognize(&graph).remove(0);
    let mut outputs = Vec::new();
    for enabled in [false, true] {
        let mut executor = MixedExecutor::new_with_options(
            &graph,
            MatMulPrecision::NativeHalf,
            false,
            MixedOptions {
                qkv_grad_pack_fusion: enabled,
                ..MixedOptions::default()
            },
        )
        .unwrap();
        // Expose owned program allocations to this diagnostic without moving or
        // changing any address. They remain owned until executor destruction.
        for p in &mut executor.programs {
            if let Some(p) = p.downcast_mut::<Program<F32>>() {
                executor.buffers.extend(p.drain_storage());
            }
            if let Some(p) = p.downcast_mut::<Program<BF16>>() {
                executor.buffers.extend(p.drain_storage());
            }
        }
        let raw: HashSet<_> = descriptor
            .inputs
            .iter()
            .map(|i| executor.values[&i.source].binding.pointer)
            .collect();
        let patterns = [
            0u16, 0x8000, 0x0001, 0x8001, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc1, 0x7f81, 0xffa5,
        ];
        for (j, input) in descriptor.inputs.iter().enumerate() {
            let target = executor.values[&input.source].binding;
            let values = (0..target.len)
                .map(|i| half::bf16::from_bits(patterns[(i + j * 3) % patterns.len()]))
                .collect::<Vec<_>>();
            let buffer = executor
                .buffers
                .iter_mut()
                .find(|b| binding(b, &executor.stream).pointer == target.pointer)
                .unwrap();
            let Buffer::BF16(buffer) = buffer else {
                panic!("raw BF16 source");
            };
            executor.stream.memcpy_htod(&values, buffer).unwrap();
        }
        let mut skipped = 0;
        for k in &executor.kernels {
            if k.label.contains("F32 to BF16") && k.pointers.last().is_some_and(|p| raw.contains(p))
            {
                skipped += 1;
                continue;
            }
            k.run(&executor.stream).unwrap();
        }
        assert_eq!(skipped, 3, "raw input cast writers must be skipped");
        executor.stream.synchronize().unwrap();
        let target = executor.values[&descriptor.output].binding;
        let buffer = executor
            .buffers
            .iter()
            .find(|b| binding(b, &executor.stream).pointer == target.pointer)
            .unwrap();
        let Buffer::BF16(buffer) = buffer else {
            panic!("BF16 output");
        };
        outputs.push(executor.stream.memcpy_dtov(buffer).unwrap());
    }
    let mut payload_differences = 0;
    for (&a, &b) in outputs[0].iter().zip(&outputs[1]) {
        if a.is_nan() || b.is_nan() {
            assert!(a.is_nan() && b.is_nan());
            payload_differences += usize::from(a.to_bits() != b.to_bits());
        } else {
            assert_eq!(a.to_bits(), b.to_bits(), "finite/Inf/signed-zero mismatch");
        }
    }
    eprintln!("raw BF16 NaN payload differences vs original GPU: {payload_differences}");
}
