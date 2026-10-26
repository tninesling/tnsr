//! Optional derivative algebra, independent of backend scheduling.
use super::{NodeIndex, TensorGraph, TensorGraphNode};
use crate::tensor::{BinaryOp, ReduceOp, UnaryOp};
use petgraph::Direction;
use std::collections::HashSet;

/// Explicit numerical choices for automatic differentiation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GradientOptions {
    /// Use `p * (g - sum(p * g, axis))` for a recognized stable softmax.
    ///
    /// This changes f32 operation/reduction order and may change rounding,
    /// overflow, and nonfinite propagation. The displayed IEEE arithmetic
    /// formula defines the derivative, including when inputs are nonfinite;
    /// all-masked rows produce NaNs. Intended for finite incoming gradients
    /// and finite scores, optionally masked by negative infinity. Probability
    /// tensors still materialize; this does not provide memory-linear attention.
    ///
    /// Only homogeneous f32 graphs are eligible. Graphs containing any casts,
    /// other dtypes, unsupported shapes, or escaping normalization intermediates
    /// retain primitive derivatives. The original forward graph is unchanged.
    pub compact_softmax: bool,
}

/// Match only the precise stable-softmax graph. Interior fanout must remain
/// within the normalization: bypassing its derivative is otherwise unsafe.
pub(super) fn softmax<D, G>(
    graph: &TensorGraph<D, G>,
    output: NodeIndex,
) -> Option<(NodeIndex, usize)> {
    let [exp, denominator] = binary(graph, output, BinaryOp::Div)?;
    let sum = unbroadcast(graph, denominator)?;
    let TensorGraphNode::ReduceAxis {
        op: ReduceOp::Sum,
        axis,
        ..
    } = graph[sum]
    else {
        return None;
    };
    if graph.inputs(sum) != [exp] {
        return None;
    }
    let [shift] = unary(graph, exp, UnaryOp::Exp)?;
    let [score, maximum_broadcast] = binary(graph, shift, BinaryOp::Sub)?;
    let maximum = unbroadcast(graph, maximum_broadcast)?;
    if !matches!(graph[maximum], TensorGraphNode::ReduceAxis {op: ReduceOp::Max, axis: a, ..} if a == axis)
        || graph.inputs(maximum) != [score]
    {
        return None;
    }
    let shape = graph[score].shape();
    if axis >= shape.len() || shape.contains(&0) {
        return None;
    }
    let mut reduced_shape = shape.clone();
    reduced_shape[axis] = 1;
    if [output, exp, shift, denominator, maximum_broadcast]
        .iter()
        .any(|&node| graph[node].shape() != shape)
        || [sum, maximum]
            .iter()
            .any(|&node| graph[node].shape() != &reduced_shape)
        || !valid_broadcast(graph, denominator, sum, axis)
        || !valid_broadcast(graph, maximum_broadcast, maximum, axis)
    {
        return None;
    }
    let members: HashSet<_> = [
        output,
        exp,
        shift,
        sum,
        denominator,
        maximum,
        maximum_broadcast,
    ]
    .into_iter()
    .collect();
    if members.iter().any(|&node| {
        node != output
            && graph
                .graph
                .neighbors_directed(node, Direction::Outgoing)
                .any(|consumer| !members.contains(&consumer))
    }) {
        return None;
    }
    Some((score, axis))
}

fn unbroadcast<D, G>(graph: &TensorGraph<D, G>, node: NodeIndex) -> Option<NodeIndex> {
    if matches!(graph[node], TensorGraphNode::BroadcastAxis { .. }) {
        let [input] = graph.inputs(node).try_into().ok()?;
        Some(input)
    } else {
        Some(node)
    }
}

fn valid_broadcast<D, G>(
    graph: &TensorGraph<D, G>,
    output: NodeIndex,
    reduction: NodeIndex,
    axis: usize,
) -> bool {
    output == reduction
        || matches!(graph[output], TensorGraphNode::BroadcastAxis {axis: a, ..} if a == axis)
            && graph.inputs(output) == [reduction]
}

fn binary<D, G>(
    graph: &TensorGraph<D, G>,
    node: NodeIndex,
    op: BinaryOp,
) -> Option<[NodeIndex; 2]> {
    matches!(graph[node], TensorGraphNode::Binary {op: actual, ..} if actual == op)
        .then(|| graph.inputs(node).try_into().ok())
        .flatten()
}

fn unary<D, G>(graph: &TensorGraph<D, G>, node: NodeIndex, op: UnaryOp) -> Option<[NodeIndex; 1]> {
    matches!(graph[node], TensorGraphNode::Unary {op: actual, ..} if actual == op)
        .then(|| graph.inputs(node).try_into().ok())
        .flatten()
}
