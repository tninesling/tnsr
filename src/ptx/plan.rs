use std::collections::HashMap;

use anyhow::{Context, Result};
use petgraph::Direction;

use crate::graph::{NodeIndex, TensorGraph, TensorGraphNode, liveness};
use crate::tensor::Shape;
use crate::tile::{DType, VirtualTensor};

/// Physical action used to produce one value in a PTX execution plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtxPlanAction {
    Upload,
    Kernel,
    DeviceCopy,
    VirtualView,
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
    pub virtual_output: VirtualTensor,
}

/// Immutable physical plan compiled for the PTX backend.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PtxExecutionPlan {
    steps: Vec<PtxPlanStep>,
}

impl PtxExecutionPlan {
    pub(crate) fn unfused<G>(graph: &TensorGraph<f32, G>) -> Result<Self> {
        let order = graph.toposort();
        let releases = liveness::analyze(graph, &order);
        let mut trailing_view_path = HashMap::new();
        for &node_index in order.iter().rev() {
            let node = &graph[node_index];
            let is_view = matches!(
                node,
                TensorGraphNode::Reshape { .. }
                    | TensorGraphNode::Flatten { .. }
                    | TensorGraphNode::Transpose { .. }
                    | TensorGraphNode::Permute { .. }
                    | TensorGraphNode::BroadcastAxis { .. }
            );
            let consumers_are_trailing_views = graph
                .graph
                .neighbors_directed(node_index, Direction::Outgoing)
                .all(|consumer| trailing_view_path.get(&consumer) == Some(&true));
            trailing_view_path.insert(node_index, is_view && consumers_are_trailing_views);
        }
        let mut virtual_values: HashMap<NodeIndex, VirtualTensor> = HashMap::new();
        let mut steps = Vec::with_capacity(order.len());
        for (position, node_index) in order.into_iter().enumerate() {
            let node = &graph[node_index];
            let inputs = graph.inputs(node_index);
            let action = match node {
                TensorGraphNode::Constant { .. }
                | TensorGraphNode::Input { .. }
                | TensorGraphNode::Parameter { .. } => PtxPlanAction::Upload,
                TensorGraphNode::Reshape { .. } | TensorGraphNode::Flatten { .. } => {
                    PtxPlanAction::VirtualView
                }
                TensorGraphNode::Transpose { .. }
                | TensorGraphNode::Permute { .. }
                | TensorGraphNode::BroadcastAxis { .. }
                    if trailing_view_path[&node_index] =>
                {
                    PtxPlanAction::VirtualView
                }
                _ => PtxPlanAction::Kernel,
            };
            let input_view = || {
                let input = inputs.first().context("view operation has no input")?;
                virtual_values
                    .get(input)
                    .context("view input has no virtual tensor")
            };
            let view_output = match node {
                TensorGraphNode::Reshape { shape } | TensorGraphNode::Flatten { shape } => {
                    input_view()?.reshape(shape.clone())?
                }
                TensorGraphNode::Transpose { shape } => {
                    let mut axes: Vec<usize> = (0..shape.len()).collect();
                    let rank = axes.len();
                    anyhow::ensure!(rank >= 2, "transpose plan requires rank >= 2");
                    axes.swap(rank - 2, rank - 1);
                    input_view()?.permute(&axes)?
                }
                TensorGraphNode::Permute { axes, .. } => input_view()?.permute(axes)?,
                TensorGraphNode::BroadcastAxis { axis, shape } => {
                    input_view()?.broadcast_axis(*axis, shape[*axis])?
                }
                _ => VirtualTensor::identity(node_index, node.shape().clone(), DType::F32),
            };
            let virtual_output = if action == PtxPlanAction::VirtualView {
                view_output
            } else {
                VirtualTensor::identity(node_index, node.shape().clone(), DType::F32)
            };
            virtual_values.insert(node_index, virtual_output.clone());
            steps.push(PtxPlanStep {
                node: node_index,
                inputs,
                operation: node.name(),
                shape: node.shape().clone(),
                action,
                materialize: action != PtxPlanAction::VirtualView,
                release_after: releases.free_after(position).to_vec(),
                virtual_output,
            });
        }
        Ok(Self { steps })
    }

    pub fn steps(&self) -> &[PtxPlanStep] {
        &self.steps
    }
}
