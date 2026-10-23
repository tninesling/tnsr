//! Extract ordinary tensor subgraphs between externally scheduled scan boundaries.
use crate::graph::{NodeIndex, TensorGraph, TensorGraphNode};
use crate::ptx::linear::Binding;
use anyhow::{Context, Result};
use std::collections::HashMap;

pub(super) struct Stage {
    pub graph: TensorGraph<f32>,
    pub inputs: HashMap<NodeIndex, Binding>,
    pub outputs: Vec<NodeIndex>,
}

pub(super) fn extract(
    graph: &TensorGraph<f32>,
    outputs: &[NodeIndex],
    available: &HashMap<NodeIndex, Binding>,
) -> Result<Stage> {
    fn visit(
        source: &TensorGraph<f32>,
        node: NodeIndex,
        stage: &mut Stage,
        nodes: &mut HashMap<NodeIndex, NodeIndex>,
        available: &HashMap<NodeIndex, Binding>,
    ) -> NodeIndex {
        if let Some(&id) = nodes.get(&node) {
            return id;
        }
        let id = if let Some(&binding) = available.get(&node) {
            let id = stage.graph.graph.add_node(TensorGraphNode::Input {
                name: "bound",
                shape: source[node].shape().clone(),
            });
            stage.inputs.insert(id, binding);
            id
        } else {
            let inputs: Vec<_> = source
                .inputs(node)
                .into_iter()
                .map(|i| visit(source, i, stage, nodes, available))
                .collect();
            let id = stage.graph.graph.add_node(source[node].clone());
            for (slot, input) in inputs.into_iter().enumerate() {
                stage.graph.graph.add_edge(input, id, slot);
            }
            id
        };
        nodes.insert(node, id);
        id
    }
    let mut stage = Stage {
        graph: TensorGraph::new(),
        inputs: HashMap::new(),
        outputs: Vec::new(),
    };
    let mut nodes = HashMap::new();
    for &node in outputs {
        let input = visit(graph, node, &mut stage, &mut nodes, available);
        // External consumers must keep the requested value alive across a stage.
        let sink = stage.graph.graph.add_node(TensorGraphNode::Reshape {
            shape: graph[node].shape().clone(),
        });
        stage.graph.graph.add_edge(input, sink, 0);
        stage.outputs.push(sink);
    }
    for node in stage.graph.graph.node_indices() {
        if matches!(stage.graph[node], TensorGraphNode::Input { .. }) {
            stage
                .inputs
                .get(&node)
                .context("decoder stage has an unbound input")?;
        }
    }
    Ok(stage)
}
