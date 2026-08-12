use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use petgraph::Direction;
use petgraph::algo::toposort;
use petgraph::graph::Graph;
use petgraph::unionfind::UnionFind;
use petgraph::visit::EdgeRef;

use crate::graph::{NodeIndex, TensorGraph, TensorGraphNode};
use crate::tensor::Shape;
use crate::tile::{DType, FusionRegion, VirtualTensor};

const MAX_POINTWISE_REGION_OPS: usize = 64;

/// Physical action used to produce values in a PTX execution plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtxPlanAction {
    Upload,
    Kernel,
    DeviceCopy,
    VirtualView,
    PointwiseRegion(usize),
}

/// One physical step in a PTX execution plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtxPlanStep {
    /// Representative node retained for diagnostics and single-node dispatch.
    pub node: NodeIndex,
    pub members: Vec<NodeIndex>,
    pub inputs: Vec<NodeIndex>,
    pub outputs: Vec<NodeIndex>,
    pub operation: &'static str,
    pub shape: Shape,
    pub action: PtxPlanAction,
    pub materialize: bool,
    pub release_after: Vec<NodeIndex>,
    pub virtual_outputs: Vec<VirtualTensor>,
}

/// Immutable physical plan compiled for the PTX backend.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PtxExecutionPlan {
    steps: Vec<PtxPlanStep>,
    regions: Vec<FusionRegion>,
    graph_output: Option<NodeIndex>,
}

#[derive(Debug, Clone, Copy)]
enum PlanUnit {
    Node(NodeIndex),
    Region(usize),
}

impl PtxExecutionPlan {
    pub(crate) fn build<G>(graph: &TensorGraph<f32, G>) -> Result<Self> {
        let order = graph.toposort();
        let graph_output = order.last().copied();
        let regions = form_pointwise_regions(graph, &order)?;
        match Self::build_with_regions(graph, &order, regions, graph_output) {
            Ok(plan) => Ok(plan),
            Err(error)
                if error
                    .to_string()
                    .contains("fusion plan contraction contains a cycle") =>
            {
                Self::build_with_regions(graph, &order, Vec::new(), graph_output)
            }
            Err(error) => Err(error),
        }
    }

    fn build_with_regions<G>(
        graph: &TensorGraph<f32, G>,
        order: &[NodeIndex],
        regions: Vec<FusionRegion>,
        graph_output: Option<NodeIndex>,
    ) -> Result<Self> {
        let mut node_to_region = HashMap::new();
        for (region_id, region) in regions.iter().enumerate() {
            for &member in &region.members {
                node_to_region.insert(member, region_id);
            }
        }

        let mut units = Vec::new();
        let mut region_units = HashMap::new();
        let mut node_units = HashMap::new();
        for &node in order {
            if let Some(&region_id) = node_to_region.get(&node) {
                region_units.entry(region_id).or_insert_with(|| {
                    let unit = units.len();
                    units.push(PlanUnit::Region(region_id));
                    unit
                });
            } else {
                let unit = units.len();
                units.push(PlanUnit::Node(node));
                node_units.insert(node, unit);
            }
        }

        let unit_for = |node: NodeIndex| -> usize {
            node_to_region
                .get(&node)
                .map(|region| region_units[region])
                .unwrap_or_else(|| node_units[&node])
        };
        let mut quotient = Graph::<usize, ()>::new();
        let quotient_nodes: Vec<_> = (0..units.len())
            .map(|unit| quotient.add_node(unit))
            .collect();
        let mut quotient_edges = HashSet::new();
        for edge in graph.graph.edge_references() {
            let source = unit_for(edge.source());
            let target = unit_for(edge.target());
            if source != target && quotient_edges.insert((source, target)) {
                quotient.add_edge(quotient_nodes[source], quotient_nodes[target], ());
            }
        }
        let physical_order = toposort(&quotient, None)
            .map_err(|_| anyhow::anyhow!("fusion plan contraction contains a cycle"))?;

        let trailing_view_path = trailing_view_paths(graph, order);
        let mut virtual_values: HashMap<NodeIndex, VirtualTensor> = HashMap::new();
        let mut steps = Vec::with_capacity(units.len());
        for unit_index in physical_order.into_iter().map(|node| quotient[node]) {
            match units[unit_index] {
                PlanUnit::Region(region_id) => {
                    let region = &regions[region_id];
                    let outputs: Vec<_> = region.outputs.iter().map(|output| output.node).collect();
                    let virtual_outputs: Vec<_> = outputs
                        .iter()
                        .map(|&output| {
                            VirtualTensor::identity(output, region.shape.clone(), DType::F32)
                        })
                        .collect();
                    for output in &virtual_outputs {
                        virtual_values.insert(output.source, output.clone());
                    }
                    steps.push(PtxPlanStep {
                        node: *outputs.last().context("pointwise region has no outputs")?,
                        members: region.members.clone(),
                        inputs: region.inputs.iter().map(|input| input.node).collect(),
                        outputs,
                        operation: "PointwiseRegion",
                        shape: region.shape.clone(),
                        action: PtxPlanAction::PointwiseRegion(region_id),
                        materialize: true,
                        release_after: Vec::new(),
                        virtual_outputs,
                    });
                }
                PlanUnit::Node(node_index) => {
                    let node = &graph[node_index];
                    let inputs = graph.inputs(node_index);
                    let action = node_action(node, node_index, &trailing_view_path);
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
                        members: vec![node_index],
                        inputs,
                        outputs: vec![node_index],
                        operation: node.name(),
                        shape: node.shape().clone(),
                        action,
                        materialize: action != PtxPlanAction::VirtualView,
                        release_after: Vec::new(),
                        virtual_outputs: vec![virtual_output],
                    });
                }
            }
        }
        assign_plan_liveness(graph, &mut steps);
        Ok(Self {
            steps,
            regions,
            graph_output,
        })
    }

    pub fn steps(&self) -> &[PtxPlanStep] {
        &self.steps
    }

    pub fn regions(&self) -> &[FusionRegion] {
        &self.regions
    }

    pub fn graph_output(&self) -> Option<NodeIndex> {
        self.graph_output
    }

    pub fn virtual_value(&self, node: NodeIndex) -> Option<&VirtualTensor> {
        self.steps.iter().find_map(|step| {
            step.outputs
                .iter()
                .position(|&output| output == node)
                .map(|position| &step.virtual_outputs[position])
        })
    }
}

fn is_pointwise(node: &TensorGraphNode<f32>) -> bool {
    matches!(
        node,
        TensorGraphNode::Unary { .. }
            | TensorGraphNode::Binary { .. }
            | TensorGraphNode::Gt { .. }
            | TensorGraphNode::Mask { .. }
    )
}

fn form_pointwise_regions<G>(
    graph: &TensorGraph<f32, G>,
    order: &[NodeIndex],
) -> Result<Vec<FusionRegion>> {
    let mut eligible = HashSet::new();
    for &node in order {
        if is_pointwise(&graph[node])
            && graph
                .inputs(node)
                .iter()
                .all(|&input| graph[input].shape() == graph[node].shape())
        {
            eligible.insert(node);
        }
    }
    let mut union = UnionFind::new(graph.graph.node_count());
    for edge in graph.graph.edge_references() {
        let source = edge.source();
        let target = edge.target();
        if eligible.contains(&source)
            && eligible.contains(&target)
            && graph[source].shape() == graph[target].shape()
        {
            union.union(source.index(), target.index());
        }
    }
    let mut components: HashMap<usize, Vec<NodeIndex>> = HashMap::new();
    for &node in order {
        if eligible.contains(&node) {
            components
                .entry(union.find(node.index()))
                .or_default()
                .push(node);
        }
    }
    let mut regions = Vec::new();
    for component in components
        .into_values()
        .filter(|members| members.len() >= 2)
    {
        for members in component.chunks(MAX_POINTWISE_REGION_OPS) {
            if members.len() < 2 {
                continue;
            }
            let members = members.to_vec();
            let member_set: HashSet<_> = members.iter().copied().collect();
            let outputs = members
                .iter()
                .filter(|&&member| {
                    let mut consumers = graph.graph.neighbors_directed(member, Direction::Outgoing);
                    match consumers.next() {
                        None => true,
                        Some(first) => {
                            !member_set.contains(&first)
                                || consumers.any(|consumer| !member_set.contains(&consumer))
                        }
                    }
                })
                .copied()
                .collect();
            regions.push(FusionRegion::from_graph(graph, members, outputs)?);
        }
    }
    regions.sort_by_key(|region| {
        order
            .iter()
            .position(|node| node == &region.members[0])
            .unwrap_or(usize::MAX)
    });
    Ok(regions)
}

fn trailing_view_paths<G>(
    graph: &TensorGraph<f32, G>,
    order: &[NodeIndex],
) -> HashMap<NodeIndex, bool> {
    let mut trailing = HashMap::new();
    for &node_index in order.iter().rev() {
        let is_view = matches!(
            graph[node_index],
            TensorGraphNode::Reshape { .. }
                | TensorGraphNode::Flatten { .. }
                | TensorGraphNode::Transpose { .. }
                | TensorGraphNode::Permute { .. }
                | TensorGraphNode::BroadcastAxis { .. }
        );
        let consumers_are_views = graph
            .graph
            .neighbors_directed(node_index, Direction::Outgoing)
            .all(|consumer| trailing.get(&consumer) == Some(&true));
        trailing.insert(node_index, is_view && consumers_are_views);
    }
    trailing
}

fn node_action(
    node: &TensorGraphNode<f32>,
    node_index: NodeIndex,
    trailing_view_path: &HashMap<NodeIndex, bool>,
) -> PtxPlanAction {
    match node {
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
    }
}

fn assign_plan_liveness<G>(graph: &TensorGraph<f32, G>, steps: &mut [PtxPlanStep]) {
    let pinned: HashSet<_> = graph
        .graph
        .node_indices()
        .filter(|&node| {
            graph
                .graph
                .neighbors_directed(node, Direction::Outgoing)
                .next()
                .is_none()
        })
        .collect();
    let mut last_use = HashMap::new();
    for (position, step) in steps.iter().enumerate() {
        for &output in &step.outputs {
            last_use.entry(output).or_insert(position);
        }
        for &input in &step.inputs {
            last_use.insert(input, position);
        }
    }
    for (&value, &position) in &last_use {
        if !pinned.contains(&value) {
            steps[position].release_after.push(value);
        }
    }
    for step in steps {
        step.release_after.sort_by_key(|node| node.index());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorExpr;

    #[test]
    fn pointwise_diamond_forms_one_closed_region_without_graph_mutation() {
        let input = TensorExpr::constant(vec![0.25; 4], vec![4]);
        let shared = input.exp();
        let graph: TensorGraph<f32> = (shared.clone().log() + shared.relu()).into();
        let node_count = graph.graph.node_count();
        let edge_count = graph.graph.edge_count();

        let plan = PtxExecutionPlan::build(&graph).unwrap();
        assert_eq!(plan.regions.len(), 1);
        assert_eq!(plan.regions[0].members.len(), 4);
        assert_eq!(plan.regions[0].inputs.len(), 1);
        assert_eq!(plan.regions[0].outputs.len(), 1);
        assert_eq!(graph.graph.node_count(), node_count);
        assert_eq!(graph.graph.edge_count(), edge_count);
    }

    #[test]
    fn large_pointwise_chain_splits_at_resource_limit() {
        let mut expression = TensorExpr::constant(vec![0.25; 4], vec![4]);
        for _ in 0..130 {
            expression = expression.relu();
        }
        let graph: TensorGraph<f32> = expression.into();

        let plan = PtxExecutionPlan::build(&graph).unwrap();
        let sizes: Vec<_> = plan
            .regions
            .iter()
            .map(|region| region.members.len())
            .collect();
        assert_eq!(sizes, vec![64, 64, 2]);
    }
}
