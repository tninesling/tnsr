use super::*;
use crate::tensor::{BinaryOp, TensorExpr};

struct Fixture {
    graph: TensorGraph<f32>,
    output: NodeIndex,
}

fn fixture(batch: usize, heads: usize, sequence: usize, width: usize, early: bool) -> Fixture {
    let query = TensorExpr::input("q", vec![batch, heads, sequence, width]).cast(ValueDType::BF16);
    let key = TensorExpr::input("k", vec![batch, heads, width, sequence]).cast(ValueDType::BF16);
    let value = TensorExpr::input("v", vec![batch, heads, sequence, width]).cast(ValueDType::BF16);
    let mask = TensorExpr::constant(
        (0..sequence * sequence)
            .map(|i| {
                if i % sequence > i / sequence {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect(),
        vec![1, 1, sequence, sequence],
    );
    let p = crate::nn::softmax(query.matmul(key).cast(ValueDType::F32) * 0.125 + mask, 3);
    let value = if early {
        value
            + p.clone()
                .reduce_mean(3)
                .cast(ValueDType::BF16)
                .broadcast(vec![batch, heads, sequence, width])
    } else {
        value
    };
    let output = p.cast(ValueDType::BF16).matmul(value);
    let mut graph = TensorGraph::new();
    let output = graph.add_expr(&output, false);
    Fixture { graph, output }
}

#[test]
fn training_attention_union_retains_checkpoints_and_delays_publication() {
    let fixture = fixture(8, 8, 512, 64, false);
    let matched = recognize(&fixture.graph);
    assert_eq!(matched.len(), 1);
    let region = &matched[0];
    assert_eq!(region.output, fixture.output);
    assert_eq!(region.shape, vec![8, 8, 512, 512]);
    assert_eq!(region.scale.to_bits(), 0.125_f32.to_bits());
    assert_eq!(
        fixture.graph.inputs(region.output),
        vec![region.narrowed, region.value]
    );
    assert_eq!(
        fixture.graph.inputs(region.qk),
        vec![region.query, region.key]
    );
    assert!(region.members.contains(&region.qk));
    for node in [region.probability, region.narrowed, region.output] {
        assert!(region.members.contains(&node));
    }
    for node in [region.query, region.key, region.value, region.mask] {
        assert!(!region.members.contains(&node));
    }
    assert!(recognize(&self::fixture(8, 8, 512, 64, true).graph).is_empty());
}

#[test]
fn training_attention_union_rejects_wrong_geometry_and_qk_escape() {
    for (batch, heads, sequence, width) in [(2, 8, 512, 64), (8, 8, 128, 64), (8, 8, 512, 32)] {
        assert!(recognize(&fixture(batch, heads, sequence, width, false).graph).is_empty());
    }
    let mut fixture = fixture(8, 8, 512, 64, false);
    let qk = recognize(&fixture.graph).remove(0).qk;
    let escape = fixture.graph.graph.add_node(TensorGraphNode::Reshape {
        shape: fixture.graph[qk].shape().clone(),
    });
    fixture.graph.graph.add_edge(qk, escape, 0);
    assert!(recognize(&fixture.graph).is_empty());
}

#[test]
fn training_attention_union_allows_late_probability_consumers_only() {
    let mut fixture = fixture(8, 8, 512, 64, false);
    let region = recognize(&fixture.graph).remove(0);
    // Depending on PV guarantees that this consumer cannot execute before the
    // retained probability checkpoint is published by the union.
    let mean = fixture.graph.graph.add_node(TensorGraphNode::ReduceAxis {
        op: crate::tensor::ReduceOp::Mean,
        axis: 3,
        shape: vec![8, 8, 512, 1],
    });
    fixture.graph.graph.add_edge(region.output, mean, 0);
    let broadcast = fixture
        .graph
        .graph
        .add_node(TensorGraphNode::BroadcastAxis {
            axis: 3,
            shape: vec![8, 8, 512, 512],
        });
    fixture.graph.graph.add_edge(mean, broadcast, 0);
    let consumer = fixture.graph.graph.add_node(TensorGraphNode::Binary {
        op: BinaryOp::Mul,
        shape: vec![8, 8, 512, 512],
    });
    fixture
        .graph
        .graph
        .add_edge(region.probability, consumer, 0);
    fixture.graph.graph.add_edge(broadcast, consumer, 1);
    assert_eq!(recognize(&fixture.graph).len(), 1);
}

#[test]
fn training_attention_union_rejects_early_read_through_bf16_alias() {
    let mut fixture = fixture(8, 8, 512, 64, false);
    let region = recognize(&fixture.graph).remove(0);
    let alias = fixture.graph.graph.add_node(TensorGraphNode::Permute {
        axes: vec![0, 1, 3, 2],
        shape: vec![8, 8, 512, 512],
    });
    fixture.graph.graph.add_edge(region.narrowed, alias, 0);
    let mean = fixture.graph.graph.add_node(TensorGraphNode::ReduceAxis {
        op: crate::tensor::ReduceOp::Mean,
        axis: 3,
        shape: vec![8, 8, 512, 1],
    });
    fixture.graph.graph.add_edge(alias, mean, 0);
    let broadcast = fixture
        .graph
        .graph
        .add_node(TensorGraphNode::BroadcastAxis {
            axis: 3,
            shape: vec![8, 8, 512, 64],
        });
    fixture.graph.graph.add_edge(mean, broadcast, 0);
    let value = fixture.graph.graph.add_node(TensorGraphNode::Binary {
        op: BinaryOp::Add,
        shape: vec![8, 8, 512, 64],
    });
    fixture.graph.graph.add_edge(region.value, value, 0);
    fixture.graph.graph.add_edge(broadcast, value, 1);
    let old = fixture
        .graph
        .graph
        .find_edge(region.value, region.output)
        .unwrap();
    fixture.graph.graph.remove_edge(old);
    fixture.graph.graph.add_edge(value, region.output, 1);
    assert!(recognize(&fixture.graph).is_empty());
}

#[test]
fn training_attention_union_actual_decoder_structural_scope() -> anyhow::Result<()> {
    use crate::ptx::mixed::test_decoder::{Config, synthetic};
    let mut records = Vec::new();
    for (width, sequence, batch, heads, hidden, expected) in [
        (2048, 128, 1, 32, 5504, 0),
        (1024, 512, 1, 16, 2752, 0),
        (512, 512, 8, 8, 1408, 4),
        (512, 512, 16, 8, 1408, 4),
        (512, 512, 32, 8, 1408, 4),
        (512, 128, 2, 8, 1408, 0),
        (512, 512, 2, 8, 1408, 0),
        (1024, 128, 2, 16, 2752, 0),
    ] {
        let model = synthetic(Config {
            width,
            sequence,
            batch,
            heads,
            hidden,
            layers: 4,
            vocab: 4096,
            epsilon: 1e-5,
            learning_rate: 1e-4,
            max_grad_norm: 1.0,
        });
        let (graph, _) = model.training_graph_with_options(crate::graph::GradientOptions {
            compact_softmax: true,
        })?;
        let regions = recognize(&graph);
        assert_eq!(regions.len(), expected, "W{width}/S{sequence}/B{batch}");
        let mut claimed = HashSet::new();
        let mut nodes = Vec::new();
        for region in regions {
            assert!(region.members.is_disjoint(&claimed));
            assert!(!region.early_checkpoint_aliases.is_empty());
            assert!(region.early_checkpoint_aliases.is_disjoint(&region.members));
            claimed.extend(region.members.iter().copied());
            nodes.push(serde_json::json!({
                "q":region.query.index(),"k":region.key.index(),"v":region.value.index(),
                "qk":region.qk.index(),"probability":region.probability.index(),
                "narrowed":region.narrowed.index(),"output":region.output.index(),
                "shape":region.shape,"members":region.members.len(),
                "early_checkpoint_aliases":region.early_checkpoint_aliases.iter().map(|n|n.index()).collect::<Vec<_>>(),
            }));
        }
        records.push(serde_json::json!({
            "config":model.config,"regions":nodes,"expected":expected,
            "scope":"Exact established decoder config and graph closure only; not binding/copy/resource admission or GPU activation"
        }));
    }
    if let Some(path) = std::env::var_os("TNSR_ATTENTION_TRAINING_FORWARD_AUDIT") {
        std::fs::create_dir_all(&path)?;
        std::fs::write(
            std::path::Path::new(&path).join("actual-decoder-structural-scope.json"),
            serde_json::to_vec_pretty(&records)?,
        )?;
    }
    Ok(())
}
