use super::*;
use crate::tensor::TensorExpr;
use crate::tile::RegionValue;
use crate::{Executor, SimpleExecutor};

fn fixture(
    g: f32,
    u: f32,
    incoming: f32,
) -> (
    TensorGraph<f32, WithGrad>,
    NodeIndex,
    [usize; 2],
    [NodeIndex; 3],
) {
    let gate = Parameter::new(vec![g], vec![1]);
    let up = Parameter::new(vec![u], vec![1]);
    let ids = [gate.id(), up.id()];
    let value = crate::nn::silu(
        TensorExpr::from(gate)
            .cast(ValueDType::BF16)
            .cast(ValueDType::F32),
    )
    .cast(ValueDType::BF16)
        * TensorExpr::from(up).cast(ValueDType::BF16);
    let mut graph = TensorGraph::new();
    let output = graph.add_expr(&value, false);
    let prefix = graph.len();
    let mut graph = graph.with_gradients(output);
    graph.graph[NodeIndex::new(prefix)] = TensorGraphNode::Constant {
        data: Arc::new(vec![incoming]),
        shape: vec![1],
    };
    let parameter = |id| {
        graph
            .graph
            .node_indices()
            .find(|&n| matches!(graph[n],TensorGraphNode::Parameter {id:i,..}if i==id))
            .unwrap()
    };
    let cast = |source| {
        graph
            .graph
            .neighbors_directed(source, Direction::Outgoing)
            .find(|&n| {
                matches!(
                    graph[n],
                    TensorGraphNode::Cast {
                        dtype: ValueDType::BF16,
                        ..
                    }
                )
            })
            .unwrap()
    };
    let x = cast(parameter(ids[0]));
    let u = cast(parameter(ids[1]));
    let seed = graph
        .graph
        .node_indices()
        .find(|&n| {
            n.index() >= prefix
                && one(&graph, n) == Some(NodeIndex::new(prefix))
                && matches!(
                    graph[n],
                    TensorGraphNode::Cast {
                        dtype: ValueDType::BF16,
                        ..
                    }
                )
        })
        .unwrap();
    (graph, output, ids, [x, u, seed])
}
fn same(a: f32, b: f32) {
    assert!(
        (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
        "{a:?}({:08x}) vs {b:?}({:08x})",
        a.to_bits(),
        b.to_bits()
    );
}
#[test]
fn extracted_mixed_program_preserves_every_storage_boundary_and_primitive_nonfinite_vjp() {
    for gate in [
        f32::NEG_INFINITY,
        -f32::MAX,
        -104.,
        -90.,
        -20.,
        -1.,
        -f32::MIN_POSITIVE,
        -0.,
        0.,
        f32::from_bits(1),
        0.125 + 1. / 2048.,
        1.,
        20.,
        90.,
        f32::MAX,
        f32::INFINITY,
        f32::NAN,
    ] {
        for up in [-2., -0., 0., 1. + 1. / 256., f32::INFINITY, f32::NAN] {
            for incoming in [-0.7, -0., 0., 2., f32::INFINITY, f32::NAN] {
                let (graph, output, ids, inputs) = fixture(gate, up, incoming);
                let dtypes = value_dtypes(&graph);
                let gradients = &graph.gradient_metadata().param_to_grad;
                let a = normalize(&graph, &dtypes, gradients[&ids[0]], &mut HashMap::new());
                let b = normalize(&graph, &dtypes, gradients[&ids[1]], &mut HashMap::new());
                let source = HashMap::from([(inputs[0], 0), (inputs[1], 1), (inputs[2], 2)]);
                let f = ScalarProgram::extract(&graph, &[output], &source).unwrap();
                let back = ScalarProgram::extract(&graph, &[a, b], &source).unwrap();
                let mut cpu = SimpleExecutor::new();
                cpu.execute(&graph, HashMap::new()).unwrap();
                let values = [gate, up, incoming].map(|x| half::bf16::from_f32(x).to_f32());
                same(f.evaluate(&values)[0], cpu.get_value(output).unwrap()[0]);
                let actual = back.evaluate(&values);
                let reference = cpu.get_gradients(&graph);
                same(actual[0], reference[&ids[0]][0]);
                same(actual[1], reference[&ids[1]][0]);
            }
        }
    }
}
#[test]
fn actual_decoder_mixed_pair_matcher_and_selective_saved_primal_checkpoints() {
    use super::super::test_decoder::{Config, synthetic};
    let config = Config {
        width: 32,
        layers: 4,
        heads: 2,
        hidden: 67,
        sequence: 3,
        batch: 2,
        vocab: 37,
        epsilon: 1e-5,
        learning_rate: 1e-4,
        max_grad_norm: 1.,
    };
    let model = synthetic(config);
    let inference: TensorGraph<f32> = model.loss().unwrap().into();
    assert!(recognize(&inference).unwrap().is_empty());
    let (graph, _) = model
        .training_graph_with_options(crate::graph::GradientOptions {
            compact_softmax: true,
        })
        .unwrap();
    let regions = recognize(&graph).unwrap();
    assert_eq!(
        regions.len(),
        4,
        "actual decoder must not be a no-op matcher"
    );
    for r in &regions {
        assert_eq!(r.shape, vec![2, 3, 67]);
        assert!(r.saved_f32_checkpoints.len() >= 3);
        assert!(!r.members.contains(&r.inputs[0]));
        assert!(!r.members.contains(&r.inputs[1]));
        assert!(!r.members.contains(&r.inputs[2]));
        assert!(r.members.contains(&r.hidden));
        assert!(r.members.contains(&r.gate_gradient));
        assert!(r.members.contains(&r.up_gradient));
        let dtypes = value_dtypes(&graph);
        assert!(planned_contiguous(
            r,
            &graph,
            &dtypes,
            MixedOptions::default()
        ));
        assert!(planned_contiguous(
            r,
            &graph,
            &dtypes,
            MixedOptions {
                physical_order_casts: true,
                transposed_bf16_outputs: true,
                ..Default::default()
            }
        ));
        for (&logical, &root) in r.inputs.iter().zip(&r.storage_roots) {
            assert_eq!(flat_storage_root(&graph, &dtypes, logical), Some(root));
        }
        // Descriptor validation is mandatory even if the graph producer normally
        // allocates contiguous storage: other backend options can retain views.
        let count = r.shape.iter().product();
        let mut bindings: HashMap<_, _> = r
            .storage_roots
            .iter()
            .map(|&root| {
                let shape = vec![count];
                let base = VirtualTensor::identity(root, shape.clone(), DType::BF16);
                (
                    root,
                    Value {
                        binding: Binding {
                            pointer: 0,
                            len: count,
                            dtype: DType::BF16,
                        },
                        view: base.reshape(r.shape.clone()).unwrap(),
                        source_shape: shape,
                        chain: vec![TensorGraphNode::Reshape {
                            shape: r.shape.clone(),
                        }],
                    },
                )
            })
            .collect();
        assert_eq!(contiguous_values(r, &bindings, false).unwrap().len(), 2);
        assert_eq!(contiguous_values(r, &bindings, true).unwrap().len(), 3);
        let value = bindings.get_mut(&r.storage_roots[2]).unwrap();
        value.view = value.view.permute(&[1, 0, 2]).unwrap();
        value.chain.push(TensorGraphNode::Permute {
            axes: vec![1, 0, 2],
            shape: value.view.shape.clone(),
        });
        assert!(contiguous_values(r, &bindings, false).is_some());
        assert!(contiguous_values(r, &bindings, true).is_none());
    }
    let mut reordered = graph.clone();
    let input = regions[0].inputs[0];
    let shape = reordered[input].shape().clone();
    let axes = vec![1, 0, 2];
    let permuted_shape = axes.iter().map(|&axis| shape[axis]).collect();
    let permuted = reordered.graph.add_node(TensorGraphNode::Permute {
        axes,
        shape: permuted_shape,
    });
    reordered.graph.add_edge(input, permuted, 0);
    assert!(flat_storage_root(&reordered, &value_dtypes(&reordered), permuted).is_none());
    let mut gate_escape = graph.clone();
    let gate = graph.inputs(regions[0].hidden)[0];
    assert_eq!(value_dtypes(&graph)[&gate], ValueDType::BF16);
    let sink = gate_escape.graph.add_node(TensorGraphNode::Reshape {
        shape: graph[gate].shape().clone(),
    });
    gate_escape.graph.add_edge(gate, sink, 0);
    assert_eq!(
        recognize(&gate_escape).unwrap().len(),
        3,
        "the forward rounded gate can be register-only only when the union is closed"
    );
    // Escape of any formerly saved scalar local must preserve the old path.
    let mut escaped = graph.clone();
    let r = &regions[0];
    let node = *r.saved_f32_checkpoints.iter().next().unwrap();
    let sink = escaped.graph.add_node(TensorGraphNode::Reshape {
        shape: escaped[node].shape().clone(),
    });
    escaped.graph.add_edge(node, sink, 0);
    assert_eq!(recognize(&escaped).unwrap().len(), 3);
}
#[test]
fn lower_mixed_pair_has_internal_bf16_rounds_and_no_cross_lane_barriers() {
    use super::super::test_decoder::{Config, synthetic};
    let model = synthetic(Config {
        width: 32,
        layers: 1,
        heads: 2,
        hidden: 67,
        sequence: 3,
        batch: 2,
        vocab: 37,
        epsilon: 1e-5,
        learning_rate: 1e-4,
        max_grad_norm: 1.,
    });
    let (graph, _) = model.training_graph().unwrap();
    let mut regions = recognize(&graph).unwrap();
    let r = regions.remove(0);
    let sources: Vec<_> = r
        .inputs
        .iter()
        .enumerate()
        .map(|(i, &node)| RegionInput {
            node,
            value: RegionValue(i),
            tensor: VirtualTensor::identity(node, r.shape.clone(), DType::BF16),
            source_shape: r.shape.clone(),
        })
        .collect();
    let f = r.lower_forward(&sources[..2], 256).unwrap();
    let b = r.lower_backward(&sources, 256).unwrap();
    assert_eq!(f.params.len(), 3);
    assert_eq!(b.params.len(), 5);
    assert!(f.params.iter().all(|p| p.dtype == DType::BF16));
    assert!(b.params.iter().all(|p| p.dtype == DType::BF16));
    for ir in [f, b] {
        assert!(!ir.body.stmts.iter().any(|s| matches!(s, Stmt::Barrier)));
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.region_kernels = vec![crate::tile::graph::TileRegionKernel { region_id: 0, ir }];
        tiles.optimize_indices().unwrap();
        let ptx = PtxGraph::from(tiles)
            .with_target(PtxTarget::sm80())
            .to_ptx();
        assert!(ptx.contains("cvt.rn.bf16.f32"));
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(!ptx.contains("bar.sync"));
    }
}

#[test]
fn actual_b32_decoder_keeps_four_contiguous_pairs_under_current_options() {
    use super::super::test_decoder::{Config, synthetic};
    let model = synthetic(Config {
        width: 512,
        layers: 4,
        heads: 8,
        hidden: 1408,
        sequence: 512,
        batch: 32,
        vocab: 4096,
        epsilon: 1e-5,
        learning_rate: 1e-4,
        max_grad_norm: 1.,
    });
    let (graph, _) = model
        .training_graph_with_options(crate::graph::GradientOptions {
            compact_softmax: true,
        })
        .unwrap();
    let regions = recognize(&graph).unwrap();
    assert_eq!(regions.len(), 4);
    let dtypes = value_dtypes(&graph);
    for r in &regions {
        assert_eq!(r.shape, vec![32, 512, 1408]);
        assert!(planned_contiguous(
            r,
            &graph,
            &dtypes,
            MixedOptions {
                physical_order_casts: true,
                transposed_bf16_outputs: true,
                qkv_grad_pack_fusion: true,
                ..Default::default()
            }
        ));
    }
}

fn same_slice(a: &[f32], b: &[f32]) {
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
fn swiglu_pair_gpu_actual_decoder_updates_replay_and_incremental_lifetimes() {
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
        o.set_experimental_first_accumulation_overwrite(true)
            .unwrap();
        o.set_experimental_gradient_norm_partials(true).unwrap();
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
                    qkv_grad_pack_fusion: true,
                    swiglu_pair_fusion: i == 1,
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
    assert_eq!(executors[0].info().swiglu_pair_forward_regions, 0);
    assert_eq!(executors[1].info().swiglu_pair_forward_regions, 4);
    assert_eq!(executors[1].info().swiglu_pair_backward_regions, 4);
    assert!(executors.iter().all(|e| e.info().replay));
    eprintln!(
        "decoder SwiGLU kernels {} -> {}",
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
            assert!(owners[i].gradient_norm_partials_status().last_step_used);
            assert_eq!(executors[i].transfers().parameter_copies, 0);
        }
        same_slice(
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
            same_slice(&grads[0][&a.id()], &grads[1][&b.id()]);
            same_slice(&a.values().unwrap(), &b.values().unwrap());
            same_slice(&moments[0][&a.id()].first, &moments[1][&b.id()].first);
            same_slice(&moments[0][&a.id()].second, &moments[1][&b.id()].second);
        }
    }
}

#[test]
fn swiglu_pair_actual_baseline_and_candidate_ptx_audit() {
    if !super::super::tests::cuda_available() {
        return;
    }
    use super::super::test_decoder::{Config, synthetic};
    let model = synthetic(Config {
        width: 16,
        layers: 1,
        heads: 4,
        hidden: 24,
        sequence: 5,
        batch: 2,
        vocab: 31,
        epsilon: 1e-5,
        learning_rate: 1e-4,
        max_grad_norm: 1.,
    });
    let (graph, _) = model.training_graph().unwrap();
    let region = recognize(&graph).unwrap().remove(0);
    let directory = std::path::Path::new("target/swiglu-pair-audit");
    std::fs::create_dir_all(directory).unwrap();
    let baseline = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, false).unwrap();
    let mut metadata = Vec::new();
    for (i, program) in baseline.programs.iter().enumerate() {
        if let Some(program) = program.downcast_ref::<Program<F32>>() {
            std::fs::write(
                directory.join(format!("baseline-f32-{i}.ptx")),
                &program.ptx,
            )
            .unwrap();
            std::fs::write(
                directory.join(format!("baseline-f32-{i}.plan")),
                &program.plan,
            )
            .unwrap();
            metadata.push(serde_json::json!({"index":i,"labels":program.kernels.iter().map(|k|&k.label).collect::<Vec<_>>() }));
        }
    }
    for (i, program) in baseline.programs.iter().enumerate() {
        if let Some(program) = program.downcast_ref::<Program<BF16>>()
            && program
                .kernels
                .iter()
                .any(|k| k.label.contains("output=[2, 5, 24]"))
        {
            std::fs::write(
                directory.join(format!("baseline-bf16-{i}.ptx")),
                &program.ptx,
            )
            .unwrap();
            std::fs::write(
                directory.join(format!("baseline-bf16-{i}.plan")),
                &program.plan,
            )
            .unwrap();
        }
    }
    for backward in [false, true] {
        let sources: Vec<_> = region.storage_roots[..if backward { 3 } else { 2 }]
            .iter()
            .enumerate()
            .map(|(i, &node)| {
                let shape = graph[node].shape().clone();
                RegionInput {
                    node,
                    value: RegionValue(i),
                    tensor: VirtualTensor::identity(node, shape.clone(), DType::BF16),
                    source_shape: shape,
                }
            })
            .collect();
        let ir = if backward {
            region.lower_backward(&sources, 256).unwrap()
        } else {
            region.lower_forward(&sources, 256).unwrap()
        };
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.region_kernels = vec![crate::tile::graph::TileRegionKernel { region_id: 0, ir }];
        tiles.optimize_indices().unwrap();
        let ptx = PtxGraph::from(tiles)
            .with_target(PtxTarget::from_context(baseline.stream.context()).unwrap())
            .to_ptx();
        std::fs::write(
            directory.join(if backward {
                "candidate-backward.ptx"
            } else {
                "candidate-forward.ptx"
            }),
            ptx,
        )
        .unwrap();
    }
    std::fs::write(
        directory.join("actual-baseline-programs.json"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .unwrap();
    std::fs::write(directory.join("checkpoint-nodes.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "shape":region.shape,"checkpoints":region.saved_f32_checkpoints.iter().map(|n|serde_json::json!({"node":n.index(),"operation":graph[*n].name()})).collect::<Vec<_>>()
    })).unwrap()).unwrap();
}

#[test]
fn swiglu_pair_gpu_raw_bf16_boundaries_nonfinites_and_tails() {
    if !super::super::tests::cuda_available() {
        return;
    }
    use super::super::test_decoder::{Config, synthetic};
    let model = synthetic(Config {
        width: 16,
        layers: 1,
        heads: 4,
        hidden: 27,
        sequence: 5,
        batch: 2,
        vocab: 31,
        epsilon: 1e-5,
        learning_rate: 1e-4,
        max_grad_norm: 1.,
    });
    let (graph, _) = model.training_graph().unwrap();
    let r = recognize(&graph).unwrap().remove(0);
    let mut executors = [false, true].map(|enabled| {
        let mut e = MixedExecutor::new_with_options(
            &graph,
            MatMulPrecision::Tf32x3,
            false,
            MixedOptions {
                swiglu_pair_fusion: enabled,
                ..Default::default()
            },
        )
        .unwrap();
        for p in &mut e.programs {
            if let Some(p) = p.downcast_mut::<Program<F32>>() {
                e.buffers.extend(p.drain_storage());
            }
            if let Some(p) = p.downcast_mut::<Program<BF16>>() {
                e.buffers.extend(p.drain_storage());
            }
        }
        e.upload_with_indices(HashMap::new(), model.config.feeds().unwrap())
            .unwrap();
        e
    });
    assert_eq!(executors[1].info().swiglu_pair_forward_regions, 1);
    assert_eq!(executors[1].info().swiglu_pair_backward_regions, 1);
    // Include signed zeros, both subnormal signs, finite overflow magnitudes,
    // sigmoid underflow, exact BF16 halfway sources, and quiet/signaling NaNs.
    let patterns = [
        0x0000, 0x8000, 0x0001, 0x8001, 0x0080, 0x8080, 0x3f80, 0xbf80, 0x3f81, 0xbf81, 0x42b4,
        0xc2b4, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1, 0x7f81, 0xffa5,
    ];
    let extent = r.shape.iter().product::<usize>();
    // Exhaust every BF16 gate bit pattern as well, keeping up/GH finite so
    // NaNs in unrelated operands cannot hide a finite rounding mismatch.
    let finite = [
        0x0000, 0x8000, 0x3f80, 0xbf80, 0x3f81, 0xbf81, 0x3f33, 0xbf33,
    ];
    for revision in 0..4 + 65_536_usize.div_ceil(extent) {
        for e in &mut executors {
            let mut raw = HashSet::new();
            for (j, &root) in r.storage_roots.iter().enumerate() {
                let target = e.values[&root].binding;
                raw.insert(target.pointer);
                let values: Vec<_> = (0..target.len)
                    .map(|i| {
                        half::bf16::from_bits(if revision < 4 {
                            patterns[(i * (j + 1) + j * 7 + revision) % patterns.len()]
                        } else if j == 0 {
                            ((revision - 4) * extent + i) as u16
                        } else {
                            finite[(i / (j * 7) + revision * j) % finite.len()]
                        })
                    })
                    .collect();
                let buffer = e
                    .buffers
                    .iter_mut()
                    .find(|b| binding(b, &e.stream).pointer == target.pointer)
                    .unwrap();
                let Buffer::BF16(buffer) = buffer else {
                    panic!("raw BF16 storage");
                };
                e.stream.memcpy_htod(&values, buffer).unwrap();
            }
            let mut skipped = 0;
            for k in &e.kernels {
                if k.pointers.last().is_some_and(|p| raw.contains(p)) {
                    skipped += 1;
                    continue;
                }
                k.run(&e.stream).unwrap();
            }
            assert_eq!(
                skipped, 3,
                "must suppress exactly the three raw leaf writers"
            );
            e.stream.synchronize().unwrap();
            e.has_run = true;
        }
        for (name, node) in [
            ("hidden", r.hidden),
            ("gate gradient", r.gate_gradient),
            ("up gradient", r.up_gradient),
        ] {
            let a = executors[0].read(node).unwrap();
            let b = executors[1].read(node).unwrap();
            assert_eq!(a.len(), b.len());
            for (i, (&a, &b)) in a.iter().zip(&b).enumerate() {
                assert!(
                    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
                    "{name}, revision {revision}, index {i}: {a:?}/{:08x} vs {b:?}/{:08x}",
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
    }
}
