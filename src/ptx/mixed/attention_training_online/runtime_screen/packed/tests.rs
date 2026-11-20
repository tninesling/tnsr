use super::*;
use crate::ptx::mixed::test_decoder::{Config, synthetic};

#[test]
fn packed_runtime_actual_scopes_reserve_pack_outputs_before_prefix_reuse() -> Result<()> {
    for batch in [8, 16, 32] {
        let model = synthetic(Config {
            width: 512,
            heads: 8,
            sequence: 512,
            batch,
            hidden: 1408,
            layers: 4,
            vocab: 4096,
            epsilon: 1e-5,
            learning_rate: 1e-4,
            max_grad_norm: 1.,
        });
        let (graph, _) = model.training_graph_with_options(crate::graph::GradientOptions {
            compact_softmax: true,
        })?;
        let old = Preparation::with_packed_backward(&graph, false);
        let selected = Preparation::with_packed_backward(&graph, true);
        assert_eq!(selected.regions.len(), 4);
        for r in &selected.regions {
            assert_eq!(r.proof.required_qkv_packs.len(), 1);
            let pack = *r.proof.required_qkv_packs.iter().next().unwrap();
            assert!(r.reservations.contains(&pack));
            assert!(selected.needed.contains(&pack));
            assert!(
                !old.regions
                    .iter()
                    .find(|old| old.proof.outputs == r.proof.outputs)
                    .unwrap()
                    .reservations
                    .contains(&pack)
            );
            let p = &selected.packs[&pack];
            let readers = p
                .readers
                .as_ref()
                .expect("real pack consumers must resolve");
            assert!(!readers.numeric.is_empty());
            assert!(
                readers
                    .numeric
                    .iter()
                    .all(|node| selected.needed.contains(node))
            );
            let mut last = HashMap::new();
            selected.extend_last_consumers(&mut last);
            assert!(last[&pack] >= r.proof.backward_position);
            let mut windows = p.inputs.iter().map(|i| i.window).collect::<Vec<_>>();
            windows.sort_unstable();
            assert_eq!(windows, [0, 1, 2]);
            for input in &p.inputs {
                assert_eq!(input.source, r.proof.outputs[input.window + 1]);
            }
            assert!(
                r.proof
                    .surviving_consumers
                    .iter()
                    .filter(|(node, _, _)| r.proof.outputs[1..].contains(node))
                    .all(|(_, consumer, _)| *consumer == pack)
            );
        }
    }
    Ok(())
}
#[test]
fn packed_runtime_owner_prefix_rejects_offset_dtype_and_ambiguous_owners() {
    let b = |pointer, len, dtype| Binding {
        pointer,
        len,
        dtype,
    };
    let target = b(128, 64, DType::BF16);
    let owner = b(128, 128, DType::BF16);
    assert_eq!(owner_from_bindings(target, &[owner]).unwrap().len, 128);
    assert!(owner_from_bindings(target, &[]).is_err());
    assert!(owner_from_bindings(b(144, 64, DType::BF16), &[owner]).is_err());
    assert!(owner_from_bindings(b(128, 129, DType::BF16), &[owner]).is_err());
    assert!(owner_from_bindings(target, &[b(128, 128, DType::F32)]).is_err());
    assert!(owner_from_bindings(target, &[owner, owner]).is_err());
    assert!(owner_from_bindings(b(u64::MAX - 1, 64, DType::BF16), &[owner]).is_err());
}
#[test]
fn packed_runtime_canonical_pack_views_reject_transpose_and_predicates() -> Result<()> {
    let shape = Shape::new(16, 8)?;
    let node = NodeIndex::new(17);
    for window in 0..3 {
        let good = expected_view(node, shape, window).unwrap();
        assert!(same_logical_view(&good, &good));
        let wrong = VirtualTensor::identity(node, vec![16, 512, 512], DType::BF16);
        assert!(!same_logical_view(&wrong, &good));
        let mut wrong = good.clone();
        wrong.source = NodeIndex::new(18);
        assert!(!same_logical_view(&wrong, &good));
        let mut wrong = good.clone();
        wrong.dtype = DType::F32;
        assert!(!same_logical_view(&wrong, &good));
    }
    let g: TensorGraph<f32> = crate::tensor::TensorExpr::input("source", vec![4, 8])
        .cast(crate::tensor::ValueDType::BF16)
        .reshape(vec![8, 4])
        .into();
    let root = g
        .graph
        .node_indices()
        .find(|n| {
            g.graph
                .neighbors_directed(*n, Direction::Outgoing)
                .next()
                .is_none()
        })
        .unwrap();
    assert!(
        readers(&g, root).is_none(),
        "retained alias sinks are not closed pack readers"
    );
    Ok(())
}

#[test]
fn packed_runtime_rejected_consumers_reproduce_four_identity_bf16_view_cast_chains() -> Result<()> {
    // Exact failed B8 graph, including loss_seed; no CUDA initialization.
    let f = super::super::gpu_tests::state_fixture_with_vocab(8, 4096)?;
    let g = &f.graph;
    let dtypes = crate::graph::mixed::value_dtypes(g);
    let preparation = Preparation::with_packed_backward(g, true);
    assert_eq!(preparation.regions.len(), 4);
    let mut records = Vec::new();
    for region in &preparation.regions {
        let output = *region.proof.required_qkv_packs.iter().next().unwrap();
        let readers = preparation.packs[&output].readers.as_ref().unwrap();
        assert_eq!(readers.identity_casts.len(), 1);
        let proof = &readers.identity_casts[0];
        assert_eq!(g.inputs(proof.parent), vec![output]);
        assert!(matches!(g[proof.parent], TensorGraphNode::Reshape { .. }));
        assert_eq!(g[proof.parent].shape(), &vec![4096, 1536]);
        assert_eq!(g.inputs(proof.node), vec![proof.parent]);
        assert!(matches!(
            g[proof.node],
            TensorGraphNode::Cast {
                dtype: crate::tensor::ValueDType::BF16,
                ..
            }
        ));
        assert_eq!(dtypes[&proof.parent], crate::tensor::ValueDType::BF16);
        assert_eq!(g[proof.node].shape(), g[proof.parent].shape());
        let expected = VirtualTensor::identity(output, g[output].shape().clone(), DType::BF16)
            .reshape(vec![4096, 1536])?;
        assert!(same_logical_view(&proof.expected, &expected));
        assert!(same_chain(&proof.chain, &[g[proof.parent].clone()]));
        assert!(
            preparation.needed.contains(&proof.node) && preparation.needed.contains(&proof.parent)
        );
        assert!(!readers.numeric.contains(&proof.node));
        let mut kinds = readers
            .numeric
            .iter()
            .map(|n| g[*n].name())
            .collect::<Vec<_>>();
        kinds.sort_unstable();
        assert_eq!(kinds, ["MatMul", "MatMul", "ReduceAxisSum"]);
        records.push(serde_json::json!({"online_output":region.proof.outputs[0].index(),"packed_output":output.index(),"metadata_reshape":proof.parent.index(),"previously_rejected_bf16_cast":proof.node.index(),"source_dtype":"BF16","cast_dtype":"BF16","root_shape":g[output].shape(),"alias_shape":proof.expected.shape,"chain":proof.chain.iter().map(|n|n.name()).collect::<Vec<_>>(),"numeric_readers":readers.numeric.iter().map(|n|serde_json::json!({"node":n.index(),"op":g[*n].name(),"dtype":format!("{:?}",dtypes[n])})).collect::<Vec<_>>() }));
    }
    if let Some(path) = std::env::var_os("TNSR_ONLINE_PACKED_RUNTIME_AUDIT") {
        let path = std::path::PathBuf::from(path);
        std::fs::create_dir_all(&path)?;
        std::fs::write(
            path.join("identity-bf16-view-cast-descriptors-cpu.json"),
            serde_json::to_vec_pretty(&records)?,
        )?;
    }
    Ok(())
}

#[test]
fn packed_runtime_identity_cast_requires_exact_live_bf16_root_view_and_chain() {
    let output = NodeIndex::new(5);
    let shape = vec![8, 512, 1536];
    let root = Value {
        binding: Binding {
            pointer: 128,
            len: shape.iter().product(),
            dtype: DType::BF16,
        },
        view: VirtualTensor::identity(output, shape.clone(), DType::BF16),
        source_shape: shape.clone(),
        chain: Vec::new(),
    };
    let mut alias = root.clone();
    alias.view = alias.view.reshape(vec![4096, 1536]).unwrap();
    alias.chain.push(TensorGraphNode::Reshape {
        shape: vec![4096, 1536],
    });
    let proof = IdentityCast {
        node: NodeIndex::new(7),
        parent: NodeIndex::new(6),
        expected: alias.view.clone(),
        chain: alias.chain.clone(),
    };
    assert!(physical_identity_cast(
        output,
        &shape,
        &root,
        &proof,
        (&alias, &alias)
    ));
    let mut wrong = alias.clone();
    wrong.binding.dtype = DType::F32;
    assert!(!physical_identity_cast(
        output,
        &shape,
        &root,
        &proof,
        (&alias, &wrong)
    ));
    let mut wrong = alias.clone();
    wrong.binding.pointer += 16;
    assert!(!physical_identity_cast(
        output,
        &shape,
        &root,
        &proof,
        (&wrong, &wrong)
    ));
    let mut wrong = alias.clone();
    wrong.binding.len -= 1;
    assert!(!physical_identity_cast(
        output,
        &shape,
        &root,
        &proof,
        (&wrong, &wrong)
    ));
    let mut wrong = alias.clone();
    wrong.view.source = NodeIndex::new(9);
    assert!(!physical_identity_cast(
        output,
        &shape,
        &root,
        &proof,
        (&wrong, &wrong)
    ));
    let mut wrong = alias.clone();
    wrong.source_shape = vec![4096, 1536];
    assert!(!physical_identity_cast(
        output,
        &shape,
        &root,
        &proof,
        (&wrong, &wrong)
    ));
    let mut wrong = alias.clone();
    wrong.view = VirtualTensor::identity(output, vec![4096, 1536], DType::BF16);
    assert!(!physical_identity_cast(
        output,
        &shape,
        &root,
        &proof,
        (&wrong, &wrong)
    ));
    let mut wrong = alias.clone();
    wrong.chain.clear();
    assert!(!physical_identity_cast(
        output,
        &shape,
        &root,
        &proof,
        (&wrong, &wrong)
    ));
    let mut wrong = alias.clone();
    wrong.chain = vec![TensorGraphNode::Flatten {
        shape: vec![4096, 1536],
    }];
    assert!(!physical_identity_cast(
        output,
        &shape,
        &root,
        &proof,
        (&wrong, &wrong)
    ));
    // Disproven F32-promotion hypothesis stays excluded, including its views.
    let x = crate::tensor::TensorExpr::input("x", vec![4, 8]).cast(crate::tensor::ValueDType::BF16);
    let g: TensorGraph<f32> = x
        .cast(crate::tensor::ValueDType::F32)
        .reshape(vec![8, 4])
        .into();
    let output = g
        .graph
        .node_indices()
        .find(|n| {
            matches!(
                g[*n],
                TensorGraphNode::Cast {
                    dtype: crate::tensor::ValueDType::BF16,
                    ..
                }
            )
        })
        .unwrap();
    let proof = readers(&g, output).unwrap();
    assert!(proof.identity_casts.is_empty());
    assert_eq!(proof.numeric.len(), 1);
    assert!(matches!(
        g[proof.numeric[0]],
        TensorGraphNode::Cast {
            dtype: crate::tensor::ValueDType::F32,
            ..
        }
    ));
}

#[test]
#[ignore = "manual CPU-only exact rejected-consumer ancestry dump; no CUDA initialization"]
fn packed_runtime_descriptor_diagnostic_cpu_only() -> Result<()> {
    let f = super::super::gpu_tests::state_fixture_with_vocab(8, 4096)?;
    let g = &f.graph;
    let dtypes = crate::graph::mixed::value_dtypes(g);
    let preparation = Preparation::with_packed_backward(g, true);
    let mut records = Vec::new();
    for r in &preparation.regions {
        let pack = *r.proof.required_qkv_packs.iter().next().unwrap();
        let mut nodes = Vec::new();
        let mut pending = vec![(pack, 0)];
        let mut seen = HashSet::new();
        while let Some((node, depth)) = pending.pop() {
            if !seen.insert(node) {
                continue;
            }
            let next = g
                .graph
                .neighbors_directed(node, Direction::Outgoing)
                .collect::<Vec<_>>();
            nodes.push(serde_json::json!({"node":node.index(),"name":g[node].name(),"dtype":format!("{:?}",dtypes[&node]),"shape":g[node].shape(),"inputs":g.inputs(node).iter().map(|n|n.index()).collect::<Vec<_>>(),"next":next.iter().map(|n|n.index()).collect::<Vec<_>>() }));
            if depth < 6
                && (node == pack
                    || matches!(
                        g[node],
                        TensorGraphNode::Cast { .. }
                            | TensorGraphNode::Reshape { .. }
                            | TensorGraphNode::Flatten { .. }
                            | TensorGraphNode::Permute { .. }
                            | TensorGraphNode::Transpose { .. }
                    ))
            {
                pending.extend(next.into_iter().map(|n| (n, depth + 1)));
            }
        }
        records.push(serde_json::json!({"online_output":r.proof.outputs[0].index(),"pack":pack.index(),"nodes":nodes}));
    }
    let path = std::path::PathBuf::from(
        std::env::var_os("TNSR_ONLINE_PACKED_RUNTIME_AUDIT")
            .context("CPU descriptor artifact directory required")?,
    );
    std::fs::create_dir_all(&path)?;
    std::fs::write(
        path.join("reader-ancestry-diagnostic.json"),
        serde_json::to_vec_pretty(&records)?,
    )?;
    Ok(())
}
