//! The normalization rule captures only the max broadcast and exponential sum.
//! It does not inspect attention, normalized probabilities, or weighted consumers.
use super::*;

pub(super) struct Normalization {
    pub score: NodeIndex,
    pub exponential: NodeIndex,
    pub members: Vec<NodeIndex>,
}

pub(super) fn match_normalization<D: TileDType, G>(
    graph: &TensorGraph<D, G>,
    sum: NodeIndex,
) -> Option<Normalization> {
    let TensorGraphNode::ReduceAxis {
        op: ReduceOp::Sum,
        axis,
        ..
    } = graph[sum]
    else {
        return None;
    };
    let [exponential]: [NodeIndex; 1] = graph.inputs(sum).try_into().ok()?;
    let [shift] = unary(graph, exponential, UnaryOp::Exp)?;
    let [score, broadcast] = binary(graph, shift, BinaryOp::Sub)?;
    let shape = graph[score].shape();
    if shape.is_empty()
        || shape.contains(&0)
        || axis + 1 != shape.len()
        || graph[sum].shape()[axis] != 1
    {
        return None;
    }
    let mut members = vec![sum, exponential, shift];
    broadcast_reduction(graph, broadcast, ReduceOp::Max, axis, score, &mut members)?;
    Some(Normalization {
        score,
        exponential,
        members,
    })
}
