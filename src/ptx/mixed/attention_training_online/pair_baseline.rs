//! Test-only complete old-native attention VJP with an external vector seed.
//! A temporary scalar is used for AD construction only; it never reaches PTX.
use super::leaf::Shape;
use crate::{
    graph::{GradientOptions, NodeIndex, TensorGraph, TensorGraphNode},
    tensor::{TensorExpr, ValueDType},
};
use anyhow::{Context, Result, ensure};
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};

mod gpu;
mod prepare;
mod timing;
mod timing_async16;

pub(super) struct Graph {
    pub graph: TensorGraph<f32>,
    /// O, dQ, dKT, dV, at the original BF16 MatMul storage boundaries.
    pub outputs: [NodeIndex; 4],
    /// Q/K/V/dO physical BF16 materialization roots, before canonical views.
    pub inputs: [NodeIndex; 4],
    pub roles: HashMap<NodeIndex, &'static str>,
    pub construction: serde_json::Value,
}

pub(super) fn graph(shape: Shape) -> Result<Graph> {
    build(shape, false)
}

/// Independent full dot-product AD reference; never a timing baseline.
pub(super) fn validation_graph(shape: Shape) -> Result<Graph> {
    build(shape, true)
}

fn build(shape: Shape, validation_only: bool) -> Result<Graph> {
    let physical = vec![shape.batch, 512, shape.heads * 64];
    let head = vec![shape.batch, shape.heads, 512, 64];
    let parameter = |id| {
        TensorExpr::parameter_with_id(id, vec![0.0_f32; shape.elements()], physical.clone())
            .cast(ValueDType::BF16)
    };
    let roots = [parameter(0), parameter(1), parameter(2)];
    let mut primal = TensorGraph::new();
    let input_nodes = roots
        .iter()
        .map(|x| primal.add_expr(x, false))
        .collect::<Vec<_>>();
    let view = |input: TensorExpr<f32>| {
        input
            .reshape(vec![shape.batch, 512, shape.heads, 64])
            .permute(vec![0, 2, 1, 3])
    };
    let query = view(TensorExpr::node_ref(input_nodes[0], physical.clone()));
    let key = view(TensorExpr::node_ref(input_nodes[1], physical.clone())).swap_axes(2, 3);
    let value = view(TensorExpr::node_ref(input_nodes[2], physical.clone()));
    let mask = TensorExpr::constant(
        (0..512 * 512)
            .map(|i| {
                if i % 512 > i / 512 {
                    f32::NEG_INFINITY
                } else {
                    0.
                }
            })
            .collect(),
        vec![1, 1, 512, 512],
    );
    let probabilities =
        crate::nn::softmax(query.matmul(key).cast(ValueDType::F32) * 0.125 + mask, 3);
    let output = probabilities.cast(ValueDType::BF16).matmul(value);
    let output = primal.add_expr(&output, false);
    let external_root_expr =
        TensorExpr::input("do", vec![shape.batch * 512, shape.heads * 64]).cast(ValueDType::BF16);
    let external_root = primal.add_expr(&external_root_expr, false);
    let external_expr =
        TensorExpr::node_ref(external_root, vec![shape.batch * 512, shape.heads * 64])
            .reshape(physical.clone())
            .reshape(vec![shape.batch, 512, shape.heads, 64])
            .permute(vec![0, 2, 1, 3]);
    let external = primal.add_expr(&external_expr, false);
    let mut scalar = TensorExpr::node_ref(output, head.clone());
    if validation_only {
        scalar = scalar * TensorExpr::node_ref(external, head.clone());
    }
    for axis in (0..4).rev() {
        scalar = scalar.reduce_axis_sum(axis);
    }
    let scalar = primal.add_expr(&scalar, false);
    let mut differentiated = primal.with_gradients_with_options(
        scalar,
        GradientOptions {
            compact_softmax: true,
        },
    );
    let traces = super::do_trace::trace(&differentiated)?;
    ensure!(
        traces.len() == 1,
        "old baseline requires one complete real AD attention pair"
    );
    let trace = &traces[0];
    let incoming = NodeIndex::new(trace.incoming_do);
    let dp = NodeIndex::new(trace.old_dp);
    let dv = NodeIndex::new(trace.old_dv);
    let qk = NodeIndex::new(trace.qk);
    ensure!(
        NodeIndex::new(trace.pv) == output,
        "AD attention output identity changed"
    );
    let matmuls: Vec<_> = differentiated
        .graph
        .node_indices()
        .filter(|&n| matches!(differentiated[n], TensorGraphNode::MatMul { .. }))
        .collect();
    ensure!(
        matmuls.len() == 6,
        "attention must contain exactly six contractions, no unmatched loss GEMM"
    );
    let remaining: Vec<_> = matmuls
        .iter()
        .copied()
        .filter(|n| ![qk, output, dp, dv].contains(n))
        .collect();
    let dq = *remaining
        .iter()
        .find(|&&n| differentiated[n].shape() == &head)
        .context("dQ contraction missing")?;
    let dkt = *remaining
        .iter()
        .find(|&&n| differentiated[n].shape() == &vec![shape.batch, shape.heads, 64, 512])
        .context("dKT contraction missing")?;

    // Rewire the already generated primitive VJP at its original BF16 dO
    // boundary. No scalar seed, reduction or multiply-by-dO is executed.
    let mut uses: Vec<_> = differentiated
        .graph
        .edges_directed(incoming, petgraph::Direction::Outgoing)
        .map(|edge| (edge.id(), edge.target(), *edge.weight()))
        .collect();
    ensure!(
        uses.iter().all(|(_, n, _)| [dp, dv].contains(n)) && uses.len() == 2,
        "external seed replacement must cover exactly the original dP/dV readers"
    );
    if !validation_only {
        uses.sort_by_key(|(edge, _, _)| std::cmp::Reverse(edge.index()));
        for (edge, _, _) in &uses {
            differentiated.graph.remove_edge(*edge);
        }
        for (_, target, slot) in uses {
            differentiated.graph.add_edge(external, target, slot);
        }
    }
    let output_nodes = [output, dq, dkt, dv];
    let mut retained = HashSet::new();
    let mut pending = output_nodes.to_vec();
    while let Some(node) = pending.pop() {
        if retained.insert(node) {
            pending.extend(differentiated.inputs(node));
        }
    }
    ensure!(
        validation_only || (!retained.contains(&scalar) && !retained.contains(&incoming)),
        "temporary AD scalar/seed branch must be absent from the executable baseline"
    );
    if validation_only {
        retained.extend(differentiated.graph.node_indices());
    }
    let mut graph = TensorGraph::new();
    let mut remap = HashMap::new();
    for old in differentiated.toposort() {
        if !retained.contains(&old) {
            continue;
        }
        let node = match &differentiated[old] {
            TensorGraphNode::Parameter { id, shape, .. } => TensorGraphNode::Input {
                name: ["q", "k", "v"][*id],
                shape: shape.clone(),
            },
            other => other.clone(),
        };
        let new = graph.graph.add_node(node);
        remap.insert(old, new);
        for edge in differentiated
            .graph
            .edges_directed(old, petgraph::Direction::Incoming)
        {
            graph
                .graph
                .add_edge(remap[&edge.source()], new, *edge.weight());
        }
    }
    let outputs = output_nodes.map(|n| remap[&n]);
    let inputs = [
        input_nodes[0],
        input_nodes[1],
        input_nodes[2],
        external_root,
    ]
    .map(|n| remap[&n]);
    let dtypes = crate::graph::mixed::value_dtypes(&graph);
    ensure!(
        outputs
            .iter()
            .all(|n| dtypes.get(n) == Some(&ValueDType::BF16)),
        "all old component output boundaries must be BF16"
    );
    let roles = [
        (qk, "QK"),
        (output, "PV"),
        (dp, "dP"),
        (dv, "dV"),
        (dq, "dQ"),
        (dkt, "dKT"),
    ]
    .into_iter()
    .map(|(n, role)| (remap[&n], role))
    .collect();
    Ok(Graph {
        graph,
        outputs,
        inputs,
        roles,
        construction: serde_json::json!({
            "external_gradient_seed":true,"scalar_used_for_AD_construction_only":true,
            "executable_loss_or_seed_nodes":if validation_only { serde_json::Value::String("present: untimed full-dot AD validation only".into()) } else { serde_json::json!(0) },
            "validation_only":validation_only,"contractions":6,"compact_softmax":true,
            "output_order":["O16","dQ16","dKT16","dV16"],
            "boundary_policy":"actual primitive AD and original BF16 score/probability/gradient casts; no hand-written compact VJP"
        }),
    })
}

#[test]
fn old_attention_pair_external_seed_retains_six_typed_contractions_and_no_loss() -> Result<()> {
    let baseline = graph(Shape::new(8, 8)?)?;
    assert_eq!(baseline.roles.len(), 6);
    assert_eq!(
        baseline
            .graph
            .graph
            .node_weights()
            .filter(|n| matches!(n, TensorGraphNode::Input { .. }))
            .count(),
        4
    );
    assert!(!baseline.graph.graph.node_weights().any(|n| matches!(
        n,
        TensorGraphNode::ReduceAxis { .. }
    ) && n.shape().iter().product::<usize>()
        == 1));
    assert_eq!(baseline.inputs.len(), 4);
    assert_eq!(baseline.outputs.len(), 4);
    assert_eq!(baseline.construction["executable_loss_or_seed_nodes"], 0);
    let reference = validation_graph(Shape::new(8, 8)?)?;
    assert!(reference.graph.len() > baseline.graph.len());
    assert_eq!(reference.roles.len(), 6);
    assert_eq!(reference.construction["validation_only"], true);
    Ok(())
}
