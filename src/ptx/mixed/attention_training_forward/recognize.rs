//! Checkpoint-retaining QK -> normalization -> PV closure.
//!
//! Emit at PV, after Q/K/V are available, and retain the original P32/P16/PV
//! identities. Earlier BF16 metadata-only views are permitted; an earlier arithmetic
//! reader makes this union ineligible. Retained bindings must exist for those
//! views before the fused kernel publishes the checkpoint contents at PV.
use crate::{
    graph::{NodeIndex, TensorGraph, TensorGraphNode, mixed::value_dtypes},
    tensor::ValueDType,
};
use petgraph::Direction;
use std::collections::{HashMap, HashSet};

pub(in crate::ptx::mixed) struct Region {
    pub query: NodeIndex,
    pub key: NodeIndex,
    pub value: NodeIndex,
    pub mask: NodeIndex,
    pub qk: NodeIndex,
    pub probability: NodeIndex,
    pub narrowed: NodeIndex,
    pub output: NodeIndex,
    pub scale: f32,
    pub shape: Vec<usize>,
    pub members: HashSet<NodeIndex>,
    pub early_checkpoint_aliases: HashSet<NodeIndex>,
}

fn checkpoint_reads_after<G>(
    graph: &TensorGraph<f32, G>,
    node: NodeIndex,
    emit: NodeIndex,
    positions: &HashMap<NodeIndex, usize>,
    dtypes: &HashMap<NodeIndex, ValueDType>,
    members: &HashSet<NodeIndex>,
    aliases: &mut HashSet<NodeIndex>,
) -> bool {
    graph
        .graph
        .neighbors_directed(node, Direction::Outgoing)
        .all(|consumer| {
            if members.contains(&consumer) || positions[&consumer] > positions[&emit] {
                return true;
            }
            // These BF16 node kinds take MixedExecutor's direct metadata branch.
            // F32 views can instead enter an automatic F32 region and materialize,
            // so they deliberately remain ineligible before publication.
            super::super::is_view(&graph[consumer])
                && dtypes.get(&consumer) == Some(&ValueDType::BF16)
                && graph.inputs(consumer).as_slice() == [node]
                && (aliases.contains(&consumer) || {
                    aliases.insert(consumer);
                    checkpoint_reads_after(
                        graph, consumer, emit, positions, dtypes, members, aliases,
                    )
                })
        })
}

pub(in crate::ptx::mixed) fn recognize<G>(graph: &TensorGraph<f32, G>) -> Vec<Region> {
    let positions: HashMap<_, _> = graph
        .toposort()
        .into_iter()
        .enumerate()
        .map(|(position, node)| (node, position))
        .collect();
    let dtypes = value_dtypes(graph);
    let mut claimed = HashSet::new();
    let mut regions = Vec::new();
    for normalization in super::super::attention_softmax::recognize(graph) {
        let shape = &normalization.shape;
        let Some(batch_heads) = shape[0].checked_mul(shape[1]) else {
            continue;
        };
        if batch_heads < 64 || shape[2..] != [512, 512] {
            continue;
        }
        let candidates: Vec<_> = graph
            .graph
            .neighbors_directed(normalization.narrowed, Direction::Outgoing)
            .filter(|&node| {
                matches!(graph[node], TensorGraphNode::MatMul { .. })
                    && graph[node].shape() == &[shape[0], shape[1], 512, 64]
                    && dtypes.get(&node) == Some(&ValueDType::BF16)
                    && graph.inputs(node).first() == Some(&normalization.narrowed)
            })
            .collect();
        let [output] = candidates.as_slice() else {
            continue;
        };
        let output = *output;
        let [query, key]: [NodeIndex; 2] = match graph.inputs(normalization.qk).try_into() {
            Ok(inputs) => inputs,
            Err(_) => continue,
        };
        let [_, value]: [NodeIndex; 2] = match graph.inputs(output).try_into() {
            Ok(inputs) => inputs,
            Err(_) => continue,
        };
        if graph[value].shape() != &[shape[0], shape[1], 512, 64]
            || [query, key, value]
                .iter()
                .any(|node| dtypes.get(node) != Some(&ValueDType::BF16))
        {
            continue;
        }
        let mut members = normalization.members;
        members.extend([normalization.qk, output]);
        // Every suppressed interior is closed. Metadata views of checkpoints
        // can precede PV, while every real external reader must follow it.
        let retained = [normalization.output, normalization.narrowed, output];
        let mut early_checkpoint_aliases = HashSet::new();
        let closed = members.iter().all(|&node| {
            if retained.contains(&node) {
                checkpoint_reads_after(
                    graph,
                    node,
                    output,
                    &positions,
                    &dtypes,
                    &members,
                    &mut early_checkpoint_aliases,
                )
            } else {
                graph
                    .graph
                    .neighbors_directed(node, Direction::Outgoing)
                    .all(|consumer| members.contains(&consumer))
            }
        });
        if !closed
            || members.iter().any(|node| claimed.contains(node))
            || [query, key, value, normalization.mask]
                .iter()
                .any(|node| members.contains(node) || positions[node] >= positions[&output])
        {
            continue;
        }
        claimed.extend(members.iter().copied());
        regions.push(Region {
            query,
            key,
            value,
            mask: normalization.mask,
            qk: normalization.qk,
            probability: normalization.output,
            narrowed: normalization.narrowed,
            output,
            scale: normalization.scale,
            shape: normalization.shape,
            members,
            early_checkpoint_aliases,
        });
    }
    regions
}

#[cfg(test)]
mod tests;
