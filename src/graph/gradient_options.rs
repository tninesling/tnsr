//! Optional derivative algebra, independent of backend scheduling.
use super::{NodeIndex, TensorGraph, TensorGraphNode};
use crate::tensor::{BinaryOp, ReduceOp, UnaryOp, ValueDType};
use petgraph::Direction;
use std::collections::{HashMap, HashSet};

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
    /// Only locally f32 normalizations are eligible, including within explicitly
    /// mixed graphs. The score and every normalization intermediate must be
    /// f32; interior casts, BF16 normalizations, unsupported shapes and escaping
    /// intermediates retain primitive derivatives. Casts before the score or
    /// after the probability output keep their ordinary gradient dtype/rounding
    /// boundaries. The original forward graph is unchanged.
    pub compact_softmax: bool,
}

/// Match only the precise stable-softmax graph. Interior fanout must remain
/// within the normalization: bypassing its derivative is otherwise unsafe.
pub(super) fn softmax<D, G>(
    graph: &TensorGraph<D, G>,
    output: NodeIndex,
    primal_dtypes: &HashMap<NodeIndex, ValueDType>,
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
    // A nonempty map includes every floating primal node, not just BF16 ones.
    // Missing entries are accepted only for the homogeneous f32 case (the
    // caller checks the graph's host dtype). Do not cross a storage-rounding
    // boundary inside the matched arithmetic.
    if std::iter::once(&score)
        .chain(&members)
        .any(|node| !primal_dtypes.is_empty() && primal_dtypes.get(node) != Some(&ValueDType::F32))
    {
        return None;
    }
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
