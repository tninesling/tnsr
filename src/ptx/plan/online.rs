//! Recognize normalization algebra, then inline legal score producers. This pass
//! never adds an attention operation or changes the user's differentiation graph.
use super::*;
use crate::tensor::{BinaryOp, ReduceOp, UnaryOp};
use crate::tile::{OnlineExpr, OnlineRegion, RegionInput, RegionValue, TileDType};

impl PtxExecutionPlan {
    pub fn online_regions(&self) -> &[OnlineRegion] {
        &self.online_regions
    }

    pub(crate) fn fuse_online<D: TileDType, G>(&mut self, graph: &TensorGraph<D, G>) -> Result<()> {
        if D::TILE_DTYPE != crate::tile::DType::F32 {
            return Ok(());
        }
        for output in graph.toposort() {
            let boundaries = self.online_regions.iter().map(|r| r.output).collect();
            let Some(mut region) = recognize(graph, output, &boundaries) else {
                continue;
            };
            let members: HashSet<_> = region.members.iter().copied().collect();
            // Every selected physical step must be entirely inside the candidate.
            // Otherwise its additional outputs/computation must remain materialized.
            let selected: Vec<_> = self
                .steps
                .iter()
                .enumerate()
                .filter(|(_, s)| s.members.iter().any(|n| members.contains(n)))
                .map(|(i, _)| i)
                .collect();
            let Some(&last) = selected.last() else {
                continue;
            };
            if selected
                .iter()
                .any(|&i| self.steps[i].members.iter().any(|n| !members.contains(n)))
            {
                continue;
            }
            let mut candidate_steps = self.steps.clone();
            for input in &mut region.inputs {
                input.tensor = resolve_view(graph, input.node, &members, &mut candidate_steps)?;
                input.source_shape = graph[input.tensor.source].shape().clone();
            }
            // Insertion at the last replaced step is safe only when all inputs
            // are already available and no output consumer executes earlier.
            if region.inputs.iter().any(|input| {
                self.steps
                    .iter()
                    .position(|s| s.outputs.contains(&input.tensor.source))
                    .is_none_or(|i| i >= last)
            }) {
                continue;
            }
            if self.steps[..last]
                .iter()
                .enumerate()
                .any(|(i, s)| !selected.contains(&i) && s.inputs.contains(&output))
            {
                continue;
            }
            let id = self.online_regions.len();
            let step = PtxPlanStep {
                node: output,
                members: region.members.clone(),
                inputs: region.inputs.iter().map(|i| i.tensor.source).collect(),
                outputs: vec![output],
                output_shapes: vec![region.output_shape.clone()],
                operation: "OnlineNormalizedContraction",
                shape: region.output_shape.clone(),
                action: PtxPlanAction::OnlineRegion(id),
                materialize: true,
                release_after: Vec::new(),
                virtual_outputs: vec![VirtualTensor::identity(
                    output,
                    region.output_shape.clone(),
                    D::TILE_DTYPE,
                )],
            };
            self.steps = candidate_steps
                .iter()
                .enumerate()
                .filter_map(|(i, old)| {
                    if i == last {
                        Some(step.clone())
                    } else if selected.contains(&i) {
                        None
                    } else {
                        Some(old.clone())
                    }
                })
                .collect();
            self.online_regions.push(region);
        }
        for step in &mut self.steps {
            step.release_after.clear();
        }
        assign_plan_liveness(graph, &mut self.steps);
        Ok(())
    }
}

fn recognize<D: TileDType, G>(
    graph: &TensorGraph<D, G>,
    output: NodeIndex,
    boundaries: &HashSet<NodeIndex>,
) -> Option<OnlineRegion> {
    if !matches!(graph[output], TensorGraphNode::MatMul { .. }) {
        return None;
    }
    let operands = graph.inputs(output);
    let [normalized, weights] = operands.as_slice() else {
        return None;
    };
    let rank = graph[*normalized].shape().len();
    if rank < 2 {
        return None;
    }
    let axis = rank - 1;
    let mut members = vec![output];
    let [exponential, sum_broadcast] = binary(graph, *normalized, BinaryOp::Div)?;
    members.push(*normalized);
    let sum = broadcast_reduction(
        graph,
        sum_broadcast,
        ReduceOp::Sum,
        axis,
        exponential,
        &mut members,
    )?;
    let [shift] = unary(graph, exponential, UnaryOp::Exp)?;
    let [score, max_broadcast] = binary(graph, shift, BinaryOp::Sub)?;
    broadcast_reduction(
        graph,
        max_broadcast,
        ReduceOp::Max,
        axis,
        score,
        &mut members,
    )?;
    members.extend([exponential, shift]);
    let shape = graph[score].shape().clone();
    let weight_shape = graph[*weights].shape();
    if shape.len() != rank
        || shape.contains(&0)
        || weight_shape.len() != rank
        || weight_shape[..rank - 2] != shape[..rank - 2]
        || weight_shape[rank - 2] != shape[axis]
        || weight_shape[axis] == 0
    {
        return None;
    }
    if graph[sum].shape()[axis] != 1 {
        return None;
    }
    let mut inputs = Vec::new();
    let score_expr = producer(
        graph,
        score,
        &shape,
        &mut members,
        &mut inputs,
        boundaries,
        0,
    )?;
    let weight_input = add_input(graph, *weights, &mut inputs);
    let member_set: HashSet<_> = members.iter().copied().collect();
    if members.iter().any(|&n| {
        n != output
            && graph
                .graph
                .neighbors_directed(n, Direction::Outgoing)
                .any(|consumer| !member_set.contains(&consumer))
    }) {
        return None;
    }
    members.sort_by_key(|n| n.index());
    members.dedup();
    Some(OnlineRegion {
        members,
        inputs,
        score: score_expr,
        weights: weight_input,
        output,
        score_shape: shape,
        output_shape: graph[output].shape().clone(),
    })
}

fn binary<D: TileDType, G>(
    graph: &TensorGraph<D, G>,
    node: NodeIndex,
    op: BinaryOp,
) -> Option<[NodeIndex; 2]> {
    if !matches!(graph[node], TensorGraphNode::Binary { op: actual, .. } if actual == op) {
        return None;
    }
    graph.inputs(node).try_into().ok()
}
fn unary<D: TileDType, G>(
    graph: &TensorGraph<D, G>,
    node: NodeIndex,
    op: UnaryOp,
) -> Option<[NodeIndex; 1]> {
    if !matches!(graph[node], TensorGraphNode::Unary { op: actual, .. } if actual == op) {
        return None;
    }
    graph.inputs(node).try_into().ok()
}
fn broadcast_reduction<D: TileDType, G>(
    graph: &TensorGraph<D, G>,
    node: NodeIndex,
    op: ReduceOp,
    axis: usize,
    input: NodeIndex,
    members: &mut Vec<NodeIndex>,
) -> Option<NodeIndex> {
    if !matches!(graph[node], TensorGraphNode::BroadcastAxis { axis: a, .. } if a == axis) {
        return None;
    }
    let [reduce]: [NodeIndex; 1] = graph.inputs(node).try_into().ok()?;
    if !matches!(graph[reduce], TensorGraphNode::ReduceAxis { op: actual, axis: a, .. } if actual == op && a == axis)
        || graph.inputs(reduce) != [input]
    {
        return None;
    }
    members.extend([node, reduce]);
    Some(reduce)
}
fn add_input<D: TileDType, G>(
    graph: &TensorGraph<D, G>,
    node: NodeIndex,
    inputs: &mut Vec<RegionInput>,
) -> usize {
    if let Some(index) = inputs.iter().position(|i| i.node == node) {
        return index;
    }
    let index = inputs.len();
    inputs.push(RegionInput {
        node,
        value: RegionValue(index),
        tensor: VirtualTensor::identity(node, graph[node].shape().clone(), D::TILE_DTYPE),
        source_shape: graph[node].shape().clone(),
    });
    index
}
fn producer<D: TileDType, G>(
    graph: &TensorGraph<D, G>,
    node: NodeIndex,
    shape: &Shape,
    members: &mut Vec<NodeIndex>,
    inputs: &mut Vec<RegionInput>,
    boundaries: &HashSet<NodeIndex>,
    depth: usize,
) -> Option<OnlineExpr> {
    if depth > 16 || graph[node].shape() != shape {
        return None;
    }
    if boundaries.contains(&node) {
        return Some(OnlineExpr::Input(add_input(graph, node, inputs)));
    }
    let operands = graph.inputs(node);
    let expression = match graph[node] {
        TensorGraphNode::Unary { op, .. } => OnlineExpr::Unary(
            op,
            Box::new(producer(
                graph,
                operands[0],
                shape,
                members,
                inputs,
                boundaries,
                depth + 1,
            )?),
        ),
        TensorGraphNode::Binary { op, .. } => OnlineExpr::Binary(
            op,
            Box::new(producer(
                graph,
                operands[0],
                shape,
                members,
                inputs,
                boundaries,
                depth + 1,
            )?),
            Box::new(producer(
                graph,
                operands[1],
                shape,
                members,
                inputs,
                boundaries,
                depth + 1,
            )?),
        ),
        TensorGraphNode::MatMul { .. } => {
            let lhs = graph[operands[0]].shape();
            let rhs = graph[operands[1]].shape();
            let rank = shape.len();
            if lhs.len() != rank
                || rhs.len() != rank
                || lhs[..rank - 2] != shape[..rank - 2]
                || rhs[..rank - 2] != shape[..rank - 2]
            {
                return None;
            }
            OnlineExpr::Dot {
                lhs: add_input(graph, operands[0], inputs),
                rhs: add_input(graph, operands[1], inputs),
                width: lhs[rank - 1],
            }
        }
        _ => return Some(OnlineExpr::Input(add_input(graph, node, inputs))),
    };
    members.push(node);
    Some(expression)
}

// Revisit view materialization after contraction: access maps that were costly
// when traversed by separate reductions may now be consumed directly by a scan.
fn resolve_view<D: TileDType, G>(
    graph: &TensorGraph<D, G>,
    node: NodeIndex,
    members: &HashSet<NodeIndex>,
    steps: &mut [PtxPlanStep],
) -> Result<VirtualTensor> {
    let position = steps
        .iter()
        .position(|s| s.outputs.contains(&node))
        .context("Online input has no producing step")?;
    if !view_feeds_region(graph, node, members) {
        return Ok(steps[position].virtual_outputs[steps[position]
            .outputs
            .iter()
            .position(|n| *n == node)
            .context("Online view output missing")?]
        .clone());
    }
    let operands = graph.inputs(node);
    let input = resolve_view(graph, operands[0], members, steps)?;
    let tensor = match &graph[node] {
        TensorGraphNode::Reshape { shape } | TensorGraphNode::Flatten { shape } => {
            input.reshape(shape.clone())?
        }
        TensorGraphNode::Permute { axes, .. } => input.permute(axes)?,
        TensorGraphNode::Transpose { shape } => {
            let mut axes: Vec<_> = (0..shape.len()).collect();
            axes.swap(shape.len() - 2, shape.len() - 1);
            input.permute(&axes)?
        }
        TensorGraphNode::BroadcastAxis { axis, shape } => {
            input.broadcast_axis(*axis, shape[*axis])?
        }
        _ => return Ok(input),
    };
    steps[position].action = PtxPlanAction::VirtualView;
    steps[position].materialize = false;
    steps[position].virtual_outputs = vec![tensor.clone()];
    Ok(tensor)
}
fn view_feeds_region<D: TileDType, G>(
    graph: &TensorGraph<D, G>,
    node: NodeIndex,
    members: &HashSet<NodeIndex>,
) -> bool {
    matches!(
        graph[node],
        TensorGraphNode::Reshape { .. }
            | TensorGraphNode::Flatten { .. }
            | TensorGraphNode::Transpose { .. }
            | TensorGraphNode::Permute { .. }
            | TensorGraphNode::BroadcastAxis { .. }
    ) && graph
        .graph
        .neighbors_directed(node, Direction::Outgoing)
        .all(|n| members.contains(&n) || view_feeds_region(graph, n, members))
}
