use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use petgraph::Direction;
use petgraph::algo::toposort;
use petgraph::graph::Graph;
use petgraph::unionfind::UnionFind;
use petgraph::visit::EdgeRef;

use crate::graph::{NodeIndex, TensorGraph, TensorGraphNode};
use crate::tensor::Shape;
use crate::tile::{
    FusionRegion, MatMulCapabilities, MatMulPrecision, MatMulRegion, MatMulResources,
    MatMulSchedule, MatMulSharedLayout, ReductionRegion, ReductionSchedule, VirtualTensor,
};

const MAX_POINTWISE_REGION_OPS: usize = 64;
const MAX_VIRTUAL_INDEX_OPS: usize = 64;
const MAX_NON_IDENTITY_VIRTUAL_FANOUT: usize = 2;
// A full-shape epilogue is evaluated serially by each output-fiber thread.
// Recomputing more than one warp of elements this way regresses normalization
// workloads; larger axes keep their broadcast-back epilogue parallel until a
// warp/block reduction schedule is available.
const MAX_SERIAL_FULL_EPILOGUE_AXIS: usize = 32;
const CUDA_WARP_SIZE: u32 = 32;
const CUDA_BLOCK_REDUCTION_THREADS: u32 = 128;
const MIN_BLOCK_REDUCTION_AXIS: usize = 256;
const MAX_COOPERATIVE_FULL_EPILOGUE_OPS: usize = 2;
const MAX_MATMUL_EPILOGUE_OPS: usize = 16;

/// Floating-point ordering policy used when planning PTX reductions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PtxReductionMode {
    /// Preserve the existing increasing-index accumulation order.
    #[default]
    Strict,
    /// Permit a fixed warp reduction tree for eligible regions.
    DeterministicTree,
}

/// Physical action used to produce values in a PTX execution plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtxPlanAction {
    Upload,
    Kernel,
    DeviceCopy,
    VirtualView,
    PointwiseRegion(usize),
    ReductionRegion(usize),
    MatMulRegion(usize),
}

/// One physical step in a PTX execution plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtxPlanStep {
    /// Representative node retained for diagnostics and single-node dispatch.
    pub node: NodeIndex,
    pub members: Vec<NodeIndex>,
    pub inputs: Vec<NodeIndex>,
    pub outputs: Vec<NodeIndex>,
    pub output_shapes: Vec<Shape>,
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
    reduction_regions: Vec<ReductionRegion>,
    matmul_regions: Vec<MatMulRegion>,
    matmul_schedules: HashMap<NodeIndex, MatMulSchedule>,
    graph_output: Option<NodeIndex>,
}

#[derive(Debug, Clone, Copy)]
enum PlanUnit {
    Node(NodeIndex),
    Region(usize),
    ReductionRegion(usize),
    MatMulRegion(usize),
}

impl PtxExecutionPlan {
    #[cfg(test)]
    pub(crate) fn build<D: crate::tile::TileDType, G>(graph: &TensorGraph<D, G>) -> Result<Self> {
        Self::build_with_reduction_mode(graph, PtxReductionMode::Strict)
    }

    pub(crate) fn build_with_reduction_mode<D: crate::tile::TileDType, G>(
        graph: &TensorGraph<D, G>,
        reduction_mode: PtxReductionMode,
    ) -> Result<Self> {
        let order = graph.toposort();
        let graph_output = order.last().copied();
        let (reduction_regions, mut claimed) =
            form_reduction_regions(graph, &order, reduction_mode)?;
        let (matmul_regions, matmul_claimed) = form_matmul_regions(graph, &order, &claimed)?;
        claimed.extend(matmul_claimed);
        let regions = form_pointwise_regions(graph, &order, &claimed)?;
        match Self::build_with_regions(
            graph,
            &order,
            regions,
            reduction_regions,
            matmul_regions,
            graph_output,
        ) {
            Ok(plan) => Ok(plan),
            Err(error)
                if error
                    .to_string()
                    .contains("fusion plan contraction contains a cycle") =>
            {
                Self::build_with_regions(
                    graph,
                    &order,
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    graph_output,
                )
            }
            Err(error) => Err(error),
        }
    }

    fn build_with_regions<D: crate::tile::TileDType, G>(
        graph: &TensorGraph<D, G>,
        order: &[NodeIndex],
        mut regions: Vec<FusionRegion>,
        mut reduction_regions: Vec<ReductionRegion>,
        mut matmul_regions: Vec<MatMulRegion>,
        graph_output: Option<NodeIndex>,
    ) -> Result<Self> {
        let mut node_to_region = HashMap::new();
        for (region_id, region) in regions.iter().enumerate() {
            for &member in &region.members {
                anyhow::ensure!(
                    node_to_region.insert(member, region_id).is_none(),
                    "pointwise fusion regions overlap at node {}",
                    member.index()
                );
            }
        }
        let mut node_to_reduction_region = HashMap::new();
        for (region_id, region) in reduction_regions.iter().enumerate() {
            for &member in &region.members {
                anyhow::ensure!(
                    !node_to_region.contains_key(&member),
                    "pointwise and reduction fusion regions overlap at node {}",
                    member.index()
                );
                anyhow::ensure!(
                    node_to_reduction_region.insert(member, region_id).is_none(),
                    "reduction fusion regions overlap at node {}",
                    member.index()
                );
            }
        }
        let mut node_to_matmul_region = HashMap::new();
        for (region_id, region) in matmul_regions.iter().enumerate() {
            for &member in &region.members {
                anyhow::ensure!(
                    !node_to_region.contains_key(&member)
                        && !node_to_reduction_region.contains_key(&member),
                    "matmul and existing fusion regions overlap at node {}",
                    member.index()
                );
                anyhow::ensure!(
                    node_to_matmul_region.insert(member, region_id).is_none(),
                    "matmul fusion regions overlap at node {}",
                    member.index()
                );
            }
        }

        let mut units = Vec::new();
        let mut region_units = HashMap::new();
        let mut reduction_region_units = HashMap::new();
        let mut matmul_region_units = HashMap::new();
        let mut node_units = HashMap::new();
        for &node in order {
            if let Some(&region_id) = node_to_region.get(&node) {
                region_units.entry(region_id).or_insert_with(|| {
                    let unit = units.len();
                    units.push(PlanUnit::Region(region_id));
                    unit
                });
            } else if let Some(&region_id) = node_to_reduction_region.get(&node) {
                reduction_region_units.entry(region_id).or_insert_with(|| {
                    let unit = units.len();
                    units.push(PlanUnit::ReductionRegion(region_id));
                    unit
                });
            } else if let Some(&region_id) = node_to_matmul_region.get(&node) {
                matmul_region_units.entry(region_id).or_insert_with(|| {
                    let unit = units.len();
                    units.push(PlanUnit::MatMulRegion(region_id));
                    unit
                });
            } else {
                let unit = units.len();
                units.push(PlanUnit::Node(node));
                node_units.insert(node, unit);
            }
        }

        let unit_for = |node: NodeIndex| -> usize {
            if let Some(region) = node_to_region.get(&node) {
                region_units[region]
            } else if let Some(region) = node_to_reduction_region.get(&node) {
                reduction_region_units[region]
            } else if let Some(region) = node_to_matmul_region.get(&node) {
                matmul_region_units[region]
            } else {
                node_units[&node]
            }
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

        let virtual_view_path = virtual_view_paths(
            graph,
            order,
            &node_to_region,
            &node_to_reduction_region,
            &node_to_matmul_region,
            &matmul_regions,
        );
        let mut virtual_values: HashMap<NodeIndex, VirtualTensor> = HashMap::new();
        let mut steps = Vec::with_capacity(units.len());
        for unit_index in physical_order.into_iter().map(|node| quotient[node]) {
            match units[unit_index] {
                PlanUnit::Region(region_id) => {
                    let region = &mut regions[region_id];
                    for input in &mut region.inputs {
                        let tensor =
                            virtual_values.get(&input.node).cloned().unwrap_or_else(|| {
                                VirtualTensor::identity(
                                    input.node,
                                    graph[input.node].shape().clone(),
                                    D::TILE_DTYPE,
                                )
                            });
                        input.source_shape = graph[tensor.source].shape().clone();
                        input.tensor = tensor;
                    }
                    let outputs: Vec<_> = region.outputs.iter().map(|output| output.node).collect();
                    let virtual_outputs: Vec<_> = outputs
                        .iter()
                        .map(|&output| {
                            VirtualTensor::identity(output, region.shape.clone(), D::TILE_DTYPE)
                        })
                        .collect();
                    for output in &virtual_outputs {
                        virtual_values.insert(output.source, output.clone());
                    }
                    steps.push(PtxPlanStep {
                        node: *outputs.last().context("pointwise region has no outputs")?,
                        members: region.members.clone(),
                        inputs: region
                            .inputs
                            .iter()
                            .map(|input| input.tensor.source)
                            .collect(),
                        outputs,
                        output_shapes: region
                            .outputs
                            .iter()
                            .map(|_| region.shape.clone())
                            .collect(),
                        operation: "PointwiseRegion",
                        shape: region.shape.clone(),
                        action: PtxPlanAction::PointwiseRegion(region_id),
                        materialize: true,
                        release_after: Vec::new(),
                        virtual_outputs,
                    });
                }
                PlanUnit::ReductionRegion(region_id) => {
                    let region = &mut reduction_regions[region_id];
                    anyhow::ensure!(
                        region.outputs.len() == region.output_shapes.len(),
                        "reduction region {region_id} output metadata is inconsistent"
                    );
                    for input in &mut region.inputs {
                        let tensor =
                            virtual_values.get(&input.node).cloned().unwrap_or_else(|| {
                                VirtualTensor::identity(
                                    input.node,
                                    graph[input.node].shape().clone(),
                                    D::TILE_DTYPE,
                                )
                            });
                        input.source_shape = graph[tensor.source].shape().clone();
                        input.tensor = tensor;
                    }
                    let outputs: Vec<_> = region.outputs.iter().map(|output| output.node).collect();
                    let virtual_outputs: Vec<_> = outputs
                        .iter()
                        .map(|&output| {
                            VirtualTensor::identity(
                                output,
                                graph[output].shape().clone(),
                                D::TILE_DTYPE,
                            )
                        })
                        .collect();
                    for output in &virtual_outputs {
                        virtual_values.insert(output.source, output.clone());
                    }
                    steps.push(PtxPlanStep {
                        node: *outputs.last().context("reduction region has no outputs")?,
                        members: region.members.clone(),
                        inputs: region
                            .inputs
                            .iter()
                            .map(|input| input.tensor.source)
                            .collect(),
                        outputs,
                        output_shapes: region.output_shapes.clone(),
                        operation: "ReductionRegion",
                        shape: region.output_shape.clone(),
                        action: PtxPlanAction::ReductionRegion(region_id),
                        materialize: true,
                        release_after: Vec::new(),
                        virtual_outputs,
                    });
                }
                PlanUnit::MatMulRegion(region_id) => {
                    let region = &mut matmul_regions[region_id];
                    for input in &mut region.inputs {
                        let tensor =
                            virtual_values.get(&input.node).cloned().unwrap_or_else(|| {
                                VirtualTensor::identity(
                                    input.node,
                                    graph[input.node].shape().clone(),
                                    D::TILE_DTYPE,
                                )
                            });
                        input.source_shape = graph[tensor.source].shape().clone();
                        input.tensor = tensor;
                    }
                    let outputs: Vec<_> = region.outputs.iter().map(|output| output.node).collect();
                    let virtual_outputs: Vec<_> = outputs
                        .iter()
                        .map(|&output| {
                            VirtualTensor::identity(
                                output,
                                region.output_shape.clone(),
                                D::TILE_DTYPE,
                            )
                        })
                        .collect();
                    for output in &virtual_outputs {
                        virtual_values.insert(output.source, output.clone());
                    }
                    steps.push(PtxPlanStep {
                        node: *outputs.last().context("matmul region has no outputs")?,
                        members: region.members.clone(),
                        inputs: region
                            .inputs
                            .iter()
                            .map(|input| input.tensor.source)
                            .collect(),
                        outputs,
                        output_shapes: region
                            .outputs
                            .iter()
                            .map(|_| region.output_shape.clone())
                            .collect(),
                        operation: "MatMulRegion",
                        shape: region.output_shape.clone(),
                        action: PtxPlanAction::MatMulRegion(region_id),
                        materialize: true,
                        release_after: Vec::new(),
                        virtual_outputs,
                    });
                }
                PlanUnit::Node(node_index) => {
                    let node = &graph[node_index];
                    let inputs = graph.inputs(node_index);
                    let mut action = node_action(node, node_index, &virtual_view_path);
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
                        _ => {
                            VirtualTensor::identity(node_index, node.shape().clone(), D::TILE_DTYPE)
                        }
                    };
                    let consumer_regions: HashSet<_> = graph
                        .graph
                        .neighbors_directed(node_index, Direction::Outgoing)
                        .filter_map(|consumer| node_to_region.get(&consumer).copied())
                        .collect();
                    let consumer_count = if consumer_regions.is_empty() {
                        graph
                            .graph
                            .neighbors_directed(node_index, Direction::Outgoing)
                            .count()
                    } else {
                        consumer_regions.len()
                    };
                    let expensive_reduction_access = graph
                        .graph
                        .neighbors_directed(node_index, Direction::Outgoing)
                        .filter_map(|consumer| node_to_reduction_region.get(&consumer))
                        .any(|&region_id| {
                            let region = &reduction_regions[region_id];
                            region.input_shape[region.axis] > MAX_SERIAL_FULL_EPILOGUE_AXIS
                                && !view_output.access.is_identity()
                        });
                    if action == PtxPlanAction::VirtualView
                        && consumer_count != 0
                        && (expensive_reduction_access
                            || view_output.access.operation_count() > MAX_VIRTUAL_INDEX_OPS
                            || (!view_output.access.is_identity()
                                && consumer_count > MAX_NON_IDENTITY_VIRTUAL_FANOUT))
                    {
                        action = PtxPlanAction::Kernel;
                    }
                    let virtual_output = if action == PtxPlanAction::VirtualView {
                        view_output
                    } else {
                        VirtualTensor::identity(node_index, node.shape().clone(), D::TILE_DTYPE)
                    };
                    virtual_values.insert(node_index, virtual_output.clone());
                    steps.push(PtxPlanStep {
                        node: node_index,
                        members: vec![node_index],
                        inputs,
                        outputs: vec![node_index],
                        output_shapes: vec![node.shape().clone()],
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
        let mut matmul_schedules = HashMap::new();
        for step in &steps {
            if step.action == PtxPlanAction::Kernel
                && matches!(graph[step.node], TensorGraphNode::MatMul { .. })
            {
                let inputs = graph.inputs(step.node);
                let shape = graph[step.node].shape();
                let lhs = graph[inputs[0]].shape();
                matmul_schedules.insert(
                    step.node,
                    MatMulSchedule::select_for_dtype(
                        shape[shape.len() - 2],
                        shape[shape.len() - 1],
                        lhs[lhs.len() - 1],
                        D::TILE_DTYPE,
                        MatMulCapabilities {
                            tf32: true,
                            f16: false,
                            bf16: false,
                        },
                        MatMulPrecision::AllowTf32,
                    ),
                );
            }
        }
        Ok(Self {
            steps,
            regions,
            reduction_regions,
            matmul_regions,
            matmul_schedules,
            graph_output,
        })
    }

    pub fn steps(&self) -> &[PtxPlanStep] {
        &self.steps
    }

    pub fn regions(&self) -> &[FusionRegion] {
        &self.regions
    }

    pub fn reduction_regions(&self) -> &[ReductionRegion] {
        &self.reduction_regions
    }

    pub(crate) fn schedule_matmuls(
        &mut self,
        capabilities: MatMulCapabilities,
        precision: MatMulPrecision,
        resources: MatMulResources,
        shared_layout: MatMulSharedLayout,
    ) -> Result<()> {
        for schedule in self.matmul_schedules.values_mut() {
            let shape = schedule.logical_shape;
            *schedule = MatMulSchedule::select_with_resources(
                shape.m,
                shape.n,
                shape.k,
                schedule.storage_dtype,
                capabilities,
                precision,
                resources,
            )
            .with_shared_layout(shared_layout, resources)?;
        }
        for region in &mut self.matmul_regions {
            region.schedule = MatMulSchedule::select_with_resources(
                region.m,
                region.n,
                region.k,
                region.inputs[0].tensor.dtype,
                capabilities,
                precision,
                resources,
            )
            .with_shared_layout(shared_layout, resources)?;
        }
        Ok(())
    }

    pub fn matmul_schedules(&self) -> &HashMap<NodeIndex, MatMulSchedule> {
        &self.matmul_schedules
    }

    pub fn matmul_regions(&self) -> &[MatMulRegion] {
        &self.matmul_regions
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

    pub fn value_shape(&self, node: NodeIndex) -> Option<&Shape> {
        self.steps.iter().find_map(|step| {
            step.outputs
                .iter()
                .position(|&output| output == node)
                .map(|position| &step.output_shapes[position])
        })
    }
}

fn is_pointwise<D: crate::tile::TileDType>(node: &TensorGraphNode<D>) -> bool {
    matches!(
        node,
        TensorGraphNode::Unary { .. }
            | TensorGraphNode::Binary { .. }
            | TensorGraphNode::Gt { .. }
            | TensorGraphNode::Mask { .. }
    )
}

fn form_reduction_regions<D: crate::tile::TileDType, G>(
    graph: &TensorGraph<D, G>,
    order: &[NodeIndex],
    reduction_mode: PtxReductionMode,
) -> Result<(Vec<ReductionRegion>, HashSet<NodeIndex>)> {
    let positions: HashMap<_, _> = order
        .iter()
        .enumerate()
        .map(|(position, &node)| (node, position))
        .collect();
    let mut regions = Vec::new();
    let mut claimed = HashSet::new();
    for &anchor in order.iter().rev() {
        let output_shape = match &graph[anchor] {
            TensorGraphNode::ReduceAxis { shape, .. } if !claimed.contains(&anchor) => shape,
            _ => continue,
        };
        let anchor_inputs = graph.inputs(anchor);
        if anchor_inputs.len() != 1 {
            continue;
        }
        let input_shape = graph[anchor_inputs[0]].shape();

        let mut producer_set = HashSet::new();
        let mut producer_stack = vec![anchor_inputs[0]];
        while let Some(node) = producer_stack.pop() {
            if claimed.contains(&node)
                || !is_pointwise(&graph[node])
                || graph[node].shape() != input_shape
                || !producer_set.insert(node)
            {
                continue;
            }
            producer_stack.extend(graph.inputs(node));
        }
        let mut producer_members: Vec<_> = producer_set.iter().copied().collect();
        producer_members.sort_by_key(|node| positions[node]);
        let producer_escapes = producer_set.iter().any(|&member| {
            graph
                .graph
                .neighbors_directed(member, Direction::Outgoing)
                .any(|consumer| consumer != anchor && !producer_set.contains(&consumer))
        });

        let mut epilogue_set = HashSet::new();
        let mut epilogue_stack: Vec<_> = graph
            .graph
            .neighbors_directed(anchor, Direction::Outgoing)
            .collect();
        while let Some(node) = epilogue_stack.pop() {
            if claimed.contains(&node)
                || !is_pointwise(&graph[node])
                || graph[node].shape() != output_shape
                || !epilogue_set.insert(node)
            {
                continue;
            }
            epilogue_stack.extend(graph.graph.neighbors_directed(node, Direction::Outgoing));
        }
        if producer_escapes {
            epilogue_set.clear();
        }
        let mut epilogue_members: Vec<_> = epilogue_set.iter().copied().collect();
        epilogue_members.sort_by_key(|node| positions[node]);

        let mut broadcast_members = Vec::new();
        let mut full_epilogue_set = HashSet::new();
        if reduction_mode == PtxReductionMode::DeterministicTree
            || input_shape
                .get(match &graph[anchor] {
                    TensorGraphNode::ReduceAxis { axis, .. } => *axis,
                    _ => unreachable!(),
                })
                .is_some_and(|&extent| extent <= MAX_SERIAL_FULL_EPILOGUE_AXIS)
        {
            for &reduced_source in std::iter::once(&anchor).chain(epilogue_members.iter()) {
                for consumer in graph
                    .graph
                    .neighbors_directed(reduced_source, Direction::Outgoing)
                {
                    if matches!(graph[consumer], TensorGraphNode::BroadcastAxis { axis, ref shape } if axis < shape.len() && shape == input_shape)
                    {
                        broadcast_members.push(consumer);
                    }
                }
            }
        }
        broadcast_members.sort_by_key(|node| positions[node]);
        broadcast_members.dedup();
        if !broadcast_members.is_empty() {
            let mut stack: Vec<_> = broadcast_members
                .iter()
                .flat_map(|&bridge| graph.graph.neighbors_directed(bridge, Direction::Outgoing))
                .collect();
            while let Some(node) = stack.pop() {
                if claimed.contains(&node)
                    || !is_pointwise(&graph[node])
                    || graph[node].shape() != input_shape
                    || !full_epilogue_set.insert(node)
                {
                    continue;
                }
                stack.extend(graph.graph.neighbors_directed(node, Direction::Outgoing));
            }
        }
        let mut full_epilogue_members: Vec<_> = full_epilogue_set.iter().copied().collect();
        full_epilogue_members.sort_by_key(|node| positions[node]);
        if reduction_mode == PtxReductionMode::DeterministicTree
            && full_epilogue_members.len() > MAX_COOPERATIVE_FULL_EPILOGUE_OPS
        {
            broadcast_members.clear();
            full_epilogue_members.clear();
        }
        if producer_members.is_empty()
            && epilogue_members.is_empty()
            && full_epilogue_members.is_empty()
        {
            continue;
        }
        if producer_members.len() + epilogue_members.len() + full_epilogue_members.len()
            > MAX_POINTWISE_REGION_OPS
        {
            continue;
        }

        let member_set: HashSet<_> = producer_members
            .iter()
            .chain(std::iter::once(&anchor))
            .chain(&epilogue_members)
            .chain(&broadcast_members)
            .chain(&full_epilogue_members)
            .copied()
            .collect();
        let mut outputs: Vec<_> = member_set
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
        outputs.sort_by_key(|node| positions[node]);
        let mut region = ReductionRegion::from_graph(
            graph,
            producer_members,
            anchor,
            epilogue_members,
            broadcast_members,
            full_epilogue_members,
            outputs,
        )?;
        if reduction_mode == PtxReductionMode::DeterministicTree {
            let reduction_extent = region.input_shape[region.axis];
            if reduction_extent >= MIN_BLOCK_REDUCTION_AXIS
                || !region.full_epilogue_operations.is_empty()
            {
                region.schedule = ReductionSchedule::Block {
                    threads: CUDA_BLOCK_REDUCTION_THREADS,
                };
            } else if reduction_extent >= CUDA_WARP_SIZE as usize {
                region.schedule = ReductionSchedule::Subgroup {
                    width: CUDA_WARP_SIZE,
                };
            }
        }
        claimed.extend(region.members.iter().copied());
        regions.push(region);
    }
    regions.sort_by_key(|region| positions[&region.members[0]]);
    Ok((regions, claimed))
}

fn form_matmul_regions<D: crate::tile::TileDType, G>(
    graph: &TensorGraph<D, G>,
    order: &[NodeIndex],
    already_claimed: &HashSet<NodeIndex>,
) -> Result<(Vec<MatMulRegion>, HashSet<NodeIndex>)> {
    let positions: HashMap<_, _> = order
        .iter()
        .enumerate()
        .map(|(position, &node)| (node, position))
        .collect();
    let mut regions = Vec::new();
    let mut claimed = already_claimed.clone();

    for &anchor in order {
        let output_shape = match &graph[anchor] {
            TensorGraphNode::MatMul { shape }
                if !claimed.contains(&anchor)
                    && shape.len() >= 2
                    && shape.iter().all(|&x| x > 0) =>
            {
                shape
            }
            _ => continue,
        };
        // CUDA grid.z is limited to 65535. Larger batches retain the existing
        // per-matrix path instead of constructing an unlaunchable fused kernel.
        let batch_count = output_shape[..output_shape.len() - 2]
            .iter()
            .try_fold(1usize, |count, &extent| count.checked_mul(extent));
        if batch_count.is_none_or(|count| count > 65535) {
            continue;
        }
        let anchor_inputs = graph.inputs(anchor);
        if anchor_inputs.len() != 2
            || graph[anchor_inputs[0]].shape().len() < 2
            || graph[anchor_inputs[1]].shape().len() < 2
            || graph[anchor_inputs[0]].shape().last() == Some(&0)
        {
            continue;
        }

        let mut epilogue_set = HashSet::new();
        let mut stack: Vec<_> = graph
            .graph
            .neighbors_directed(anchor, Direction::Outgoing)
            .collect();
        while let Some(node) = stack.pop() {
            if claimed.contains(&node)
                || !is_pointwise(&graph[node])
                || graph[node].shape() != output_shape
                || !epilogue_set.insert(node)
            {
                continue;
            }
            stack.extend(graph.graph.neighbors_directed(node, Direction::Outgoing));
        }
        if epilogue_set.is_empty() || epilogue_set.len() > MAX_MATMUL_EPILOGUE_OPS {
            continue;
        }
        let mut epilogue_members: Vec<_> = epilogue_set.iter().copied().collect();
        epilogue_members.sort_by_key(|node| positions[node]);
        let member_set: HashSet<_> = std::iter::once(anchor)
            .chain(epilogue_members.iter().copied())
            .collect();
        let mut outputs: Vec<_> = member_set
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
        outputs.sort_by_key(|node| positions[node]);
        let region = MatMulRegion::from_graph(graph, anchor, epilogue_members, outputs)?;
        claimed.extend(region.members.iter().copied());
        regions.push(region);
    }

    let newly_claimed = claimed
        .difference(already_claimed)
        .copied()
        .collect::<HashSet<_>>();
    Ok((regions, newly_claimed))
}

fn form_pointwise_regions<D: crate::tile::TileDType, G>(
    graph: &TensorGraph<D, G>,
    order: &[NodeIndex],
    claimed: &HashSet<NodeIndex>,
) -> Result<Vec<FusionRegion>> {
    let mut eligible = HashSet::new();
    for &node in order {
        if !claimed.contains(&node)
            && is_pointwise(&graph[node])
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

fn virtual_view_paths<D: crate::tile::TileDType, G>(
    graph: &TensorGraph<D, G>,
    order: &[NodeIndex],
    node_to_region: &HashMap<NodeIndex, usize>,
    node_to_reduction_region: &HashMap<NodeIndex, usize>,
    node_to_matmul_region: &HashMap<NodeIndex, usize>,
    matmul_regions: &[MatMulRegion],
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
        let consumers_accept_virtual = graph
            .graph
            .neighbors_directed(node_index, Direction::Outgoing)
            .all(|consumer| {
                node_to_region.contains_key(&consumer)
                    || node_to_reduction_region.contains_key(&consumer)
                    || node_to_matmul_region
                        .get(&consumer)
                        .is_some_and(|&region_id| {
                            let region = &matmul_regions[region_id];
                            region.inputs.iter().any(|input| input.node == node_index)
                        })
                    || trailing.get(&consumer) == Some(&true)
            });
        trailing.insert(node_index, is_view && consumers_accept_virtual);
    }
    trailing
}

fn node_action<D: crate::tile::TileDType>(
    node: &TensorGraphNode<D>,
    node_index: NodeIndex,
    virtual_view_path: &HashMap<NodeIndex, bool>,
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
            if virtual_view_path[&node_index] =>
        {
            PtxPlanAction::VirtualView
        }
        _ => PtxPlanAction::Kernel,
    }
}

fn assign_plan_liveness<D: crate::tile::TileDType, G>(
    graph: &TensorGraph<D, G>,
    steps: &mut [PtxPlanStep],
) {
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
    fn rank_two_matmul_claims_same_shape_pointwise_epilogue() {
        let lhs = TensorExpr::constant(vec![0.25; 4 * 8], vec![4, 8]);
        let rhs = TensorExpr::constant(vec![0.5; 8 * 6], vec![8, 6]);
        let bias = TensorExpr::constant(vec![0.1; 4 * 6], vec![4, 6]);
        let graph: TensorGraph<f32> = (lhs.matmul(rhs) + bias).relu().into();

        let plan = PtxExecutionPlan::build(&graph).unwrap();
        assert!(plan.regions.is_empty());
        assert_eq!(plan.matmul_regions.len(), 1);
        assert_eq!(plan.matmul_regions[0].epilogue_operations.len(), 2);
        assert_eq!(plan.matmul_regions[0].members.len(), 3);
        assert_eq!(
            plan.steps
                .iter()
                .filter(|step| matches!(step.action, PtxPlanAction::MatMulRegion(0)))
                .count(),
            1
        );
    }

    #[test]
    fn standalone_and_zero_k_remain_unfused_batched_epilogue_fuses() {
        let standalone: TensorGraph<f32> = TensorExpr::constant(vec![0.25; 2 * 3], vec![2, 3])
            .matmul(TensorExpr::constant(vec![0.5; 3 * 4], vec![3, 4]))
            .into();
        assert!(
            PtxExecutionPlan::build(&standalone)
                .unwrap()
                .matmul_regions
                .is_empty()
        );

        let batched: TensorGraph<f32> = TensorExpr::constant(vec![0.25; 2 * 2 * 3], vec![2, 2, 3])
            .matmul(TensorExpr::constant(vec![0.5; 2 * 3 * 4], vec![2, 3, 4]))
            .relu()
            .into();
        assert_eq!(
            PtxExecutionPlan::build(&batched)
                .unwrap()
                .matmul_regions
                .len(),
            1
        );

        let zero_k: TensorGraph<f32> = TensorExpr::constant(Vec::new(), vec![2, 0])
            .matmul(TensorExpr::constant(Vec::new(), vec![0, 4]))
            .relu()
            .into();
        assert!(
            PtxExecutionPlan::build(&zero_k)
                .unwrap()
                .matmul_regions
                .is_empty()
        );
    }

    #[test]
    fn batched_matmul_grid_limit_keeps_per_matrix_fallback() {
        let lhs = TensorExpr::constant(vec![0.25; 65536], vec![65536, 1, 1]);
        let rhs = TensorExpr::constant(vec![0.5], vec![1, 1]);
        let graph: TensorGraph<f32> = lhs.matmul(rhs).relu().into();
        let plan = PtxExecutionPlan::build(&graph).unwrap();
        assert!(plan.matmul_regions().is_empty());
        assert_eq!(plan.matmul_schedules().len(), 1);
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

    #[test]
    fn reduction_region_claims_producer_and_reduced_epilogue() {
        let input = TensorExpr::constant(vec![0.25; 8], vec![2, 4]);
        let graph: TensorGraph<f32> = input.relu().exp().reduce_sum(1).log().into();

        let plan = PtxExecutionPlan::build(&graph).unwrap();
        assert!(plan.regions.is_empty());
        assert_eq!(plan.reduction_regions.len(), 1);
        let region = &plan.reduction_regions[0];
        assert_eq!(region.producer_operations.len(), 2);
        assert_eq!(region.epilogue_operations.len(), 1);
        assert_eq!(region.members.len(), 4);
        assert!(
            plan.steps
                .iter()
                .any(|step| matches!(step.action, PtxPlanAction::ReductionRegion(0)))
        );
    }

    #[test]
    fn serial_full_epilogue_stops_at_one_warp() {
        for (width, expected_full_epilogue_operations) in [(32, 1), (33, 0)] {
            let input = TensorExpr::constant(vec![0.25; 2 * width], vec![2, width]);
            let producer = input.exp();
            let sum = producer.clone().reduce_sum(1).broadcast(vec![2, width]);
            let graph: TensorGraph<f32> = (producer / sum).into();

            let plan = PtxExecutionPlan::build(&graph).unwrap();
            assert_eq!(plan.reduction_regions.len(), 1);
            assert_eq!(
                plan.reduction_regions[0].full_epilogue_operations.len(),
                expected_full_epilogue_operations
            );
        }
    }

    #[test]
    fn large_reduction_keeps_full_epilogue_parallel() {
        let input = TensorExpr::constant(vec![0.25; 128], vec![2, 64]);
        let producer = input.exp();
        let sum = producer.clone().reduce_sum(1).broadcast(vec![2, 64]);
        let graph: TensorGraph<f32> = (producer / sum).into();

        let plan = PtxExecutionPlan::build(&graph).unwrap();
        assert_eq!(plan.reduction_regions.len(), 1);
        assert!(
            plan.reduction_regions[0]
                .full_epilogue_operations
                .is_empty()
        );
        assert!(plan.steps.iter().any(|step| {
            step.operation == "BroadcastAxis" && step.action == PtxPlanAction::Kernel
        }));
        assert!(
            plan.steps
                .iter()
                .any(|step| { step.operation == "Div" && step.action == PtxPlanAction::Kernel })
        );
    }

    #[test]
    fn cooperative_mode_selects_subgroup_schedule_without_changing_strict_default() {
        let input = TensorExpr::constant(vec![0.25; 2 * 64], vec![2, 64]);
        let graph: TensorGraph<f32> = input.exp().reduce_sum(1).into();

        let strict = PtxExecutionPlan::build(&graph).unwrap();
        assert_eq!(
            strict.reduction_regions[0].schedule,
            ReductionSchedule::Serial
        );

        let cooperative = PtxExecutionPlan::build_with_reduction_mode(
            &graph,
            PtxReductionMode::DeterministicTree,
        )
        .unwrap();
        assert_eq!(
            cooperative.reduction_regions[0].schedule,
            ReductionSchedule::Subgroup {
                width: CUDA_WARP_SIZE
            }
        );
    }

    #[test]
    fn cooperative_mode_selects_block_at_extent_boundary() {
        for (width, expected) in [
            (
                MIN_BLOCK_REDUCTION_AXIS - 1,
                ReductionSchedule::Subgroup {
                    width: CUDA_WARP_SIZE,
                },
            ),
            (
                MIN_BLOCK_REDUCTION_AXIS,
                ReductionSchedule::Block {
                    threads: CUDA_BLOCK_REDUCTION_THREADS,
                },
            ),
        ] {
            let input = TensorExpr::constant(vec![0.25; 2 * width], vec![2, width]);
            let graph: TensorGraph<f32> = input.exp().reduce_sum(1).into();
            let plan = PtxExecutionPlan::build_with_reduction_mode(
                &graph,
                PtxReductionMode::DeterministicTree,
            )
            .unwrap();
            assert_eq!(plan.reduction_regions[0].schedule, expected);
        }
    }

    #[test]
    fn cooperative_mode_fuses_large_full_epilogue_with_block_schedule() {
        let width = 64;
        let input = TensorExpr::constant(vec![0.25; 2 * width], vec![2, width]);
        let producer = input.exp();
        let sum = producer.clone().reduce_sum(1).broadcast(vec![2, width]);
        let graph: TensorGraph<f32> = (producer / sum).into();
        let plan = PtxExecutionPlan::build_with_reduction_mode(
            &graph,
            PtxReductionMode::DeterministicTree,
        )
        .unwrap();

        assert_eq!(plan.reduction_regions.len(), 1);
        assert_eq!(
            plan.reduction_regions[0].schedule,
            ReductionSchedule::Block {
                threads: CUDA_BLOCK_REDUCTION_THREADS
            }
        );
        assert_eq!(plan.reduction_regions[0].full_epilogue_operations.len(), 1);
    }

    #[test]
    fn cooperative_mode_rejects_expensive_full_epilogue() {
        let width = MIN_BLOCK_REDUCTION_AXIS;
        let input = TensorExpr::constant(vec![0.25; 2 * width], vec![2, width]);
        let producer = input.exp();
        let sum = producer.clone().reduce_sum(1).broadcast(vec![2, width]);
        let graph: TensorGraph<f32> = (producer / sum).relu().exp().into();
        let plan = PtxExecutionPlan::build_with_reduction_mode(
            &graph,
            PtxReductionMode::DeterministicTree,
        )
        .unwrap();

        assert_eq!(plan.reduction_regions.len(), 1);
        assert!(
            plan.reduction_regions[0]
                .full_epilogue_operations
                .is_empty()
        );
        assert!(
            plan.steps
                .iter()
                .any(|step| matches!(step.action, PtxPlanAction::PointwiseRegion(_)))
        );
    }
}
