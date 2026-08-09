use crate::graph::{NodeIndex, TensorGraph, TensorGraphNode, liveness};
use crate::tensor::Shape;

/// Physical action used to produce one value in a PTX execution plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtxPlanAction {
    Upload,
    Kernel,
    DeviceCopy,
}

/// One value-producing step in a PTX execution plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtxPlanStep {
    pub node: NodeIndex,
    pub inputs: Vec<NodeIndex>,
    pub operation: &'static str,
    pub shape: Shape,
    pub action: PtxPlanAction,
    pub materialize: bool,
    pub release_after: Vec<NodeIndex>,
}

/// Immutable physical plan compiled for the PTX backend.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PtxExecutionPlan {
    steps: Vec<PtxPlanStep>,
}

impl PtxExecutionPlan {
    pub(crate) fn unfused<G>(graph: &TensorGraph<f32, G>) -> Self {
        let order = graph.toposort();
        let releases = liveness::analyze(graph, &order);
        let steps = order
            .into_iter()
            .enumerate()
            .map(|(position, node_index)| {
                let node = &graph[node_index];
                let action = match node {
                    TensorGraphNode::Constant { .. }
                    | TensorGraphNode::Input { .. }
                    | TensorGraphNode::Parameter { .. } => PtxPlanAction::Upload,
                    TensorGraphNode::Reshape { .. } | TensorGraphNode::Flatten { .. } => {
                        PtxPlanAction::DeviceCopy
                    }
                    _ => PtxPlanAction::Kernel,
                };
                PtxPlanStep {
                    node: node_index,
                    inputs: graph.inputs(node_index),
                    operation: node.name(),
                    shape: node.shape().clone(),
                    action,
                    materialize: true,
                    release_after: releases.free_after(position).to_vec(),
                }
            })
            .collect();
        Self { steps }
    }

    pub fn steps(&self) -> &[PtxPlanStep] {
        &self.steps
    }
}
