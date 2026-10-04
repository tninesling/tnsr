//! Recognize normalization algebra, then inline legal score producers. This pass
//! never adds an attention operation or changes the user's differentiation graph.
use super::*;
mod contraction;
mod normalization;
use crate::tensor::{BinaryOp, ReduceOp, UnaryOp};
use crate::tile::{IndexExpr, IndexMap, IndexedSource};
use crate::tile::{OnlineConsumer, OnlineExpr, OnlineRegion, RegionInput, RegionValue, TileDType};
use contraction::{apply_view, contraction, view_root};
use normalization::{Normalization, match_normalization};

impl PtxExecutionPlan {
    pub fn online_regions(&self) -> &[OnlineRegion] {
        &self.online_regions
    }

    pub(crate) fn fuse_online<D: TileDType, G>(&mut self, graph: &TensorGraph<D, G>) -> Result<()> {
        if D::TILE_DTYPE != crate::tile::DType::F32 {
            return Ok(());
        }
        // Prefer consumers first; only then try the normalization region alone.
        let order = graph.toposort();
        for (output, normalizer_only) in order
            .iter()
            .map(|&n| (n, false))
            .chain(order.iter().map(|&n| (n, true)))
        {
            let boundaries = self.online_regions.iter().map(|r| r.output).collect();
            let Some(mut region) = recognize(graph, output, &boundaries, normalizer_only) else {
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
                operation: match region.consumer {
                    OnlineConsumer::Normalizer => "OnlineNormalizer",
                    OnlineConsumer::Sum(_) => "OnlineNormalizedContraction",
                },
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

/// Move division by the row-invariant normalizer outside a homogeneous sum.
/// This consumer rule is independent of score producers and normalization matching.
fn normalized_input<D: TileDType, G>(
    graph: &TensorGraph<D, G>,
    node: NodeIndex,
) -> Option<Normalization> {
    let [exponential, broadcast] = binary(graph, node, BinaryOp::Div)?;
    let axis = graph[exponential].shape().len().checked_sub(1)?;
    if !matches!(graph[broadcast], TensorGraphNode::BroadcastAxis { axis: a, .. } if a == axis) {
        return None;
    }
    let [sum]: [NodeIndex; 1] = graph.inputs(broadcast).try_into().ok()?;
    let mut normalization = match_normalization(graph, sum)?;
    if normalization.exponential != exponential {
        return None;
    }
    normalization.members.extend([broadcast, node]);
    Some(normalization)
}

fn recognize<D: TileDType, G>(
    graph: &TensorGraph<D, G>,
    output: NodeIndex,
    boundaries: &HashSet<NodeIndex>,
    normalizer_only: bool,
) -> Option<OnlineRegion> {
    let (mut normalization, matched) = if normalizer_only {
        (match_normalization(graph, output)?, None)
    } else {
        let mut sum = contraction(graph, output)?;
        let mut matched = None;
        for argument in 0..2 {
            let (view, views) = view_root(graph, sum.operands[argument])?;
            let Some(n) = normalized_input(graph, view.source) else {
                continue;
            };
            let shape = graph[n.score].shape();
            let output_shape = graph[output].shape();
            // The existing schedule handles one normalized row per output row.
            if output_shape.len() != shape.len()
                || output_shape[..shape.len() - 1] != shape[..shape.len() - 1]
                || sum.extent != shape[shape.len() - 1]
            {
                continue;
            }
            let mut domain = output_shape.clone();
            domain.push(sum.extent);
            let mapped = view
                .access
                .compose(&sum.accesses[argument])
                .ok()?
                .normalize_in_domain(&domain)
                .ok()?;
            let mut expected = IndexMap::identity(shape.len());
            expected.results[shape.len() - 1] = IndexExpr::IterDim(shape.len());
            if mapped != expected.normalize_in_domain(&domain).ok()? {
                continue;
            }
            sum.accesses[argument] = mapped;
            let mut n = n;
            n.members.extend(views);
            if argument == 1 {
                sum.operands.swap(0, 1);
                sum.accesses.swap(0, 1);
            }
            matched = Some((n, sum));
            break;
        }
        let (n, sum) = matched?;
        (n, Some(sum))
    };
    normalization.members.push(output);
    let shape = graph[normalization.score].shape().clone();
    let mut members = normalization.members;
    if let Some(sum) = &matched {
        members.extend(&sum.members);
    }
    let mut inputs = Vec::new();
    let score_expr = producer(
        graph,
        normalization.score,
        &shape,
        &mut members,
        &mut inputs,
        boundaries,
        0,
    )?;
    let consumer = if let Some(sum) = matched {
        let weight = add_input(graph, sum.operands[1], &mut inputs);
        OnlineConsumer::Sum(sum.bind(IndexedSource::Argument, IndexedSource::Input(weight)))
    } else {
        OnlineConsumer::Normalizer
    };
    let member_set: HashSet<_> = members.iter().copied().collect();
    if members.iter().any(|&n| {
        n != output
            && graph
                .graph
                .neighbors_directed(n, Direction::Outgoing)
                .any(|c| !member_set.contains(&c))
    }) {
        return None;
    }
    members.sort_by_key(|n| n.index());
    members.dedup();
    Some(OnlineRegion {
        members,
        inputs,
        score: score_expr,
        consumer,
        output,
        score_shape: shape,
        output_shape: graph[output].shape().clone(),
    })
}

pub(super) fn binary<D: TileDType, G>(
    graph: &TensorGraph<D, G>,
    node: NodeIndex,
    op: BinaryOp,
) -> Option<[NodeIndex; 2]> {
    if !matches!(graph[node], TensorGraphNode::Binary { op: actual, .. } if actual == op) {
        return None;
    }
    graph.inputs(node).try_into().ok()
}
pub(super) fn unary<D: TileDType, G>(
    graph: &TensorGraph<D, G>,
    node: NodeIndex,
    op: UnaryOp,
) -> Option<[NodeIndex; 1]> {
    if !matches!(graph[node], TensorGraphNode::Unary { op: actual, .. } if actual == op) {
        return None;
    }
    graph.inputs(node).try_into().ok()
}
pub(super) fn broadcast_reduction<D: TileDType, G>(
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
    let indexed = contraction(graph, node);
    let mut child = |node| producer(graph, node, shape, members, inputs, boundaries, depth + 1);
    let expression = match graph[node] {
        TensorGraphNode::Unary { op, .. } => OnlineExpr::Unary(op, Box::new(child(operands[0])?)),
        TensorGraphNode::Binary { op, .. } => OnlineExpr::Binary(
            op,
            Box::new(child(operands[0])?),
            Box::new(child(operands[1])?),
        ),
        _ if indexed.is_some() => {
            let sum = indexed?;
            let lhs = add_input(graph, sum.operands[0], inputs);
            let rhs = add_input(graph, sum.operands[1], inputs);
            members.extend(&sum.members);
            OnlineExpr::Sum(Box::new(
                sum.bind(IndexedSource::Input(lhs), IndexedSource::Input(rhs)),
            ))
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
    let tensor = apply_view(graph, node, input)?;
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
