use super::*;
use crate::ptx::mixed::test_decoder::{Config, synthetic};
use petgraph::visit::EdgeRef;
fn config(width: usize, sequence: usize, batch: usize) -> Config {
    Config {
        width,
        layers: 4,
        heads: width / 64,
        hidden: ((8 * width / 3).div_ceil(64)) * 64,
        sequence,
        batch,
        vocab: 4096,
        epsilon: 1e-5,
        learning_rate: 1e-4,
        max_grad_norm: 1.,
    }
}
fn graph(c: Config) -> TensorGraph<f32, crate::graph::WithGrad> {
    synthetic(c)
        .training_graph_with_options(crate::graph::GradientOptions {
            compact_softmax: true,
        })
        .unwrap()
        .0
}
fn packs<G>(g: &TensorGraph<f32, G>) -> HashSet<NodeIndex> {
    crate::ptx::mixed::qkv_grad_pack::recognize(g)
        .into_iter()
        .map(|p| p.output)
        .collect()
}
fn inspect<G>(g: &TensorGraph<f32, G>, p: &CompletePair) {
    assert_eq!(p.shape[2..], [512, 512]);
    assert_eq!(p.inputs.len(), 4);
    assert_eq!(p.outputs[0], p.contractions[1]);
    assert_eq!(&p.outputs[1..], &p.contractions[3..]);
    assert_eq!(p.quadratic[0], p.contractions[0]);
    assert_eq!(p.quadratic[3], p.contractions[2]);
    assert_eq!(p.output_views.len(), 4);
    for (n, v) in p.outputs.iter().zip(&p.output_views) {
        assert_eq!(v.source, *n);
        assert_eq!(v.shape, *g[*n].shape());
        assert_eq!(v.dtype, DType::BF16);
        assert!(p.members.contains(n));
    }
    assert!(p.forward_position < p.backward_position);
    assert!(p.outputs[1..].contains(&p.backward_boundary));
    for (output, _, position) in &p.surviving_consumers {
        assert!(p.outputs.contains(output));
        assert!(
            *position
                > if *output == p.outputs[0] {
                    p.forward_position
                } else {
                    p.backward_position
                }
        );
    }
    assert!(p.required_qkv_packs.iter().all(|n| packs(g).contains(n)));
}
#[test]
fn complete_pair_actual_four_layer_b8_b16_b32_has_four_conditional_structural_proofs() {
    for batch in [8, 16, 32] {
        let g = graph(config(512, 512, batch));
        let assumed = packs(&g);
        assert_eq!(assumed.len(), 4);
        let regions = recognize(&g, &assumed);
        assert_eq!(regions.len(), 4, "B{batch}");
        for r in &regions {
            inspect(&g, r);
        }
    }
}
#[test]
fn complete_pair_other_five_established_suite_cases_reject_policy() {
    for (w, s, b) in [
        (2048, 128, 1),
        (1024, 512, 1),
        (512, 128, 2),
        (512, 512, 2),
        (1024, 128, 2),
    ] {
        let g = graph(config(w, s, b));
        assert!(recognize(&g, &packs(&g)).is_empty(), "W{w}S{s}B{b}");
    }
}
fn extra_node<G>(g: &mut TensorGraph<f32, G>, source: NodeIndex, view: bool) -> NodeIndex {
    let n = g.graph.add_node(if view {
        TensorGraphNode::Reshape {
            shape: g[source].shape().clone(),
        }
    } else {
        TensorGraphNode::Binary {
            op: crate::tensor::BinaryOp::Add,
            shape: g[source].shape().clone(),
        }
    });
    g.graph.add_edge(source, n, 0);
    if !view {
        g.graph.add_edge(source, n, 1);
    }
    n
}
#[test]
fn complete_pair_rejects_square_view_escape_extra_probability_and_wrong_do() {
    let base = graph(config(512, 512, 8));
    let regions = recognize(&base, &packs(&base));
    assert_eq!(regions.len(), 4);
    let target = &regions[0];
    for view in [false, true] {
        let mut g = base.clone();
        extra_node(&mut g, target.quadratic[1], view);
        let proofs = recognize(&g, &packs(&g));
        assert_eq!(proofs.len(), 3);
        assert!(
            proofs
                .iter()
                .all(|r| r.contractions[0] != target.contractions[0])
        );
    }
    let mut g = base.clone();
    let dp = target.contractions[2];
    let input = g.graph.add_node(TensorGraphNode::Input {
        name: "wrong_do",
        shape: g[target.inputs[3]].shape().clone(),
    });
    let wrong = g.graph.add_node(TensorGraphNode::Cast {
        dtype: ValueDType::BF16,
        shape: g[target.inputs[3]].shape().clone(),
    });
    // Attach a distinct canonical BF16 source of the same extent, then change
    // only the dP operand. dV still consumes the original dO, so pairing rejects.
    g.graph.add_edge(input, wrong, 0);
    let edge = g
        .graph
        .edges_directed(dp, Direction::Incoming)
        .find(|e| *e.weight() == 0)
        .unwrap()
        .id();
    g.graph.remove_edge(edge);
    g.graph.add_edge(wrong, dp, 0);
    let proofs = recognize(&g, &packs(&g));
    assert_eq!(proofs.len(), 3);
    assert!(
        proofs
            .iter()
            .all(|r| r.contractions[0] != target.contractions[0])
    );
}
#[test]
fn complete_pair_rejects_early_f32_view_consumer() {
    use crate::tensor::TensorExpr;
    let x = TensorExpr::<f32>::input("x", vec![2, 8]).cast(ValueDType::BF16);
    let mut g = TensorGraph::new();
    let source = g.add_expr(&x, false);
    let promoted = g.graph.add_node(TensorGraphNode::Cast {
        dtype: ValueDType::F32,
        shape: vec![2, 8],
    });
    g.graph.add_edge(source, promoted, 0);
    let reshaped = g
        .graph
        .add_node(TensorGraphNode::Reshape { shape: vec![1, 16] });
    g.graph.add_edge(promoted, reshaped, 0);
    assert!(
        g.graph
            .neighbors_directed(source, Direction::Outgoing)
            .next()
            .is_some()
    );
    let pos = g
        .toposort()
        .into_iter()
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect::<HashMap<_, _>>();
    let dtype = value_dtypes(&g);
    assert_eq!(dtype[&reshaped], ValueDType::F32);
    assert!(
        consumers(
            &g,
            source,
            g.graph.node_count() + 1,
            &HashSet::from([source]),
            &ConsumerContext {
                positions: &pos,
                dtypes: &dtype,
                packs: &[]
            },
            &mut HashSet::new()
        )
        .is_none()
    );
}

#[test]
fn complete_pair_inherits_literal_causal_mask_and_exact_scale_proof() {
    let base = graph(config(512, 512, 8));
    assert_eq!(recognize(&base, &packs(&base)).len(), 4);
    let mut custom_mask = base.clone();
    let mut masks = 0;
    for n in custom_mask.graph.node_indices().collect::<Vec<_>>() {
        if let TensorGraphNode::Constant { data, shape } = &custom_mask[n]
            && shape == &[1, 1, 512, 512]
        {
            let mut changed = (**data).clone();
            changed[1] = 0.;
            custom_mask.graph[n] = TensorGraphNode::Constant {
                data: std::sync::Arc::new(changed),
                shape: shape.clone(),
            };
            masks += 1;
        }
    }
    assert!(masks > 0);
    assert!(recognize(&custom_mask, &packs(&custom_mask)).is_empty());
    let mut custom_scale = base.clone();
    let mut scales = 0;
    for n in custom_scale.graph.node_indices().collect::<Vec<_>>() {
        if let TensorGraphNode::Constant { data, shape } = &custom_scale[n]
            && shape.is_empty()
            && data.len() == 1
            && data[0].to_bits() == 0.125f32.to_bits()
        {
            custom_scale.graph[n] = TensorGraphNode::Constant {
                data: std::sync::Arc::new(vec![0.25]),
                shape: shape.clone(),
            };
            scales += 1;
        }
    }
    assert!(scales > 0);
    assert!(recognize(&custom_scale, &packs(&custom_scale)).is_empty());
}
