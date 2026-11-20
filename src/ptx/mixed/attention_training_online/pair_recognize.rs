//! Test-only complete numerical-pair closure. No kernels, allocations or hook.
//! A selected-pack set is an explicit conditional scheduling assumption: a
//! future runtime must verify those actual semantic owners before admission.
use crate::{
    graph::{NodeIndex, TensorGraph, TensorGraphNode, mixed::value_dtypes},
    tensor::ValueDType,
    tile::{DType, VirtualTensor},
};
use petgraph::Direction;
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub(super) struct CompletePair {
    /// Q, logical transposed K, V, existing rounded BF16 dO.
    pub inputs: [NodeIndex; 4],
    /// Original O,dQ,dKT,dV identities; all logical dense BF16 views.
    pub outputs: [NodeIndex; 4],
    pub output_views: [VirtualTensor; 4],
    pub shape: Vec<usize>,
    /// QK,P32,P16,dP16,DS16 have no surviving numeric readers.
    pub quadratic: [NodeIndex; 5],
    /// Exact semantic contraction owners: QK,PV,dP,dQ,dKT,dV.
    pub contractions: [NodeIndex; 6],
    pub members: HashSet<NodeIndex>,
    pub forward_position: usize,
    pub backward_boundary: NodeIndex,
    pub backward_position: usize,
    /// (retained output, surviving numeric/effective pack consumer, position).
    pub surviving_consumers: Vec<(NodeIndex, NodeIndex, usize)>,
    pub required_qkv_packs: HashSet<NodeIndex>,
}
fn one<G>(g: &TensorGraph<f32, G>, n: NodeIndex) -> Option<NodeIndex> {
    let [v] = g.inputs(n).try_into().ok()?;
    Some(v)
}
fn volume(s: &[usize]) -> Option<usize> {
    s.iter()
        .try_fold(1usize, |n, &d| if d == 0 { None } else { n.checked_mul(d) })
}
fn full_view<G>(
    g: &TensorGraph<f32, G>,
    n: NodeIndex,
    v: &VirtualTensor,
    dtypes: &HashMap<NodeIndex, ValueDType>,
) -> Option<VirtualTensor> {
    let input = one(g, n)?;
    if g[input].shape() != &v.shape || volume(g[n].shape()) != volume(&v.shape) {
        return None;
    }
    let mut result = match &g[n] {
        TensorGraphNode::Reshape { shape } | TensorGraphNode::Flatten { shape } => {
            v.reshape(shape.clone()).ok()?
        }
        TensorGraphNode::Permute { axes, .. } => v.permute(axes).ok()?,
        TensorGraphNode::Transpose { .. } => {
            let mut axes = (0..v.shape.len()).collect::<Vec<_>>();
            if axes.len() < 2 {
                return None;
            }
            let end = axes.len();
            axes.swap(end - 2, end - 1);
            v.permute(&axes).ok()?
        }
        TensorGraphNode::Cast { dtype, .. } => {
            let old = *dtypes.get(&input)?;
            if !(*dtype == old || (*dtype == ValueDType::F32 && old == ValueDType::BF16)) {
                return None;
            }
            v.clone()
        }
        _ => return None,
    };
    result.dtype = match dtypes.get(&n)? {
        ValueDType::BF16 => DType::BF16,
        ValueDType::F32 => DType::F32,
    };
    Some(result)
}
fn ancestry<G>(
    g: &TensorGraph<f32, G>,
    n: NodeIndex,
    root: NodeIndex,
    dtypes: &HashMap<NodeIndex, ValueDType>,
    depth: usize,
) -> Option<(VirtualTensor, HashSet<NodeIndex>)> {
    if n == root {
        return Some((
            VirtualTensor::identity(root, g[root].shape().clone(), DType::BF16),
            HashSet::new(),
        ));
    }
    if depth == 0 {
        return None;
    }
    let (v, mut aliases) = ancestry(g, one(g, n)?, root, dtypes, depth - 1)?;
    let v = full_view(g, n, &v, dtypes)?;
    aliases.insert(n);
    Some((v, aliases))
}
fn operand<G>(
    g: &TensorGraph<f32, G>,
    n: NodeIndex,
    root: NodeIndex,
    transpose: bool,
    dtypes: &HashMap<NodeIndex, ValueDType>,
) -> Option<HashSet<NodeIndex>> {
    if dtypes.get(&n) != Some(&ValueDType::BF16) || dtypes.get(&root) != Some(&ValueDType::BF16) {
        return None;
    }
    let (actual, aliases) = ancestry(g, n, root, dtypes, 32)?;
    let mut expected = VirtualTensor::identity(root, g[root].shape().clone(), DType::BF16);
    if transpose {
        let mut axes = (0..expected.shape.len()).collect::<Vec<_>>();
        if axes.len() < 2 {
            return None;
        }
        let end = axes.len();
        axes.swap(end - 2, end - 1);
        expected = expected.permute(&axes).ok()?;
    }
    (actual.shape == expected.shape
        && actual.access.normalize_in_domain(&actual.shape).ok()?
            == expected.access.normalize_in_domain(&expected.shape).ok()?)
    .then_some(aliases)
}
fn contraction<G>(
    g: &TensorGraph<f32, G>,
    a: NodeIndex,
    at: bool,
    b: NodeIndex,
    bt: bool,
    shape: &[usize],
    dtypes: &HashMap<NodeIndex, ValueDType>,
) -> Option<(NodeIndex, HashSet<NodeIndex>)> {
    let mut found = Vec::new();
    for n in g.graph.node_indices() {
        if !matches!(g[n], TensorGraphNode::MatMul { .. })
            || g[n].shape() != shape
            || dtypes.get(&n) != Some(&ValueDType::BF16)
        {
            continue;
        }
        let [x, y] = g.inputs(n).try_into().ok()?;
        if let (Some(mut p), Some(q)) = (operand(g, x, a, at, dtypes), operand(g, y, b, bt, dtypes))
        {
            p.extend(q);
            found.push((n, p));
        }
    }
    let [result] = found.try_into().ok()?;
    Some(result)
}
fn dp_root<G>(
    g: &TensorGraph<f32, G>,
    incoming: NodeIndex,
    dtypes: &HashMap<NodeIndex, ValueDType>,
) -> Option<(NodeIndex, HashSet<NodeIndex>)> {
    if !matches!(
        g[incoming],
        TensorGraphNode::Cast {
            dtype: ValueDType::F32,
            ..
        }
    ) {
        return None;
    }
    let mut aliases = HashSet::from([incoming]);
    let mut n = one(g, incoming)?;
    while matches!(
        g[n],
        TensorGraphNode::Cast {
            dtype: ValueDType::BF16,
            ..
        }
    ) {
        let p = one(g, n)?;
        if dtypes.get(&p) != Some(&ValueDType::BF16) || g[p].shape() != g[n].shape() {
            return None;
        }
        aliases.insert(n);
        n = p;
    }
    (matches!(g[n], TensorGraphNode::MatMul { .. }) && dtypes.get(&n) == Some(&ValueDType::BF16))
        .then_some((n, aliases))
}
fn close_interiors<G>(
    g: &TensorGraph<f32, G>,
    members: &mut HashSet<NodeIndex>,
    outputs: &[NodeIndex; 4],
    dtypes: &HashMap<NodeIndex, ValueDType>,
) -> Option<()> {
    let mut pending = members.iter().copied().collect::<Vec<_>>();
    while let Some(n) = pending.pop() {
        if outputs.contains(&n) {
            continue;
        }
        let consumers = g
            .graph
            .neighbors_directed(n, Direction::Outgoing)
            .collect::<Vec<_>>();
        if consumers.is_empty() {
            return None;
        }
        for c in consumers {
            if members.contains(&c) {
                continue;
            }
            let dtype = match dtypes.get(&n)? {
                ValueDType::BF16 => DType::BF16,
                ValueDType::F32 => DType::F32,
            };
            let v = VirtualTensor::identity(n, g[n].shape().clone(), dtype);
            full_view(g, c, &v, dtypes)?;
            g.graph.neighbors_directed(c, Direction::Outgoing).next()?;
            members.insert(c);
            pending.push(c);
        }
    }
    Some(())
}
struct Pack {
    output: NodeIndex,
    members: HashSet<NodeIndex>,
    inputs: HashSet<NodeIndex>,
}
struct ConsumerContext<'a> {
    positions: &'a HashMap<NodeIndex, usize>,
    dtypes: &'a HashMap<NodeIndex, ValueDType>,
    packs: &'a [Pack],
}
fn consumers<G>(
    g: &TensorGraph<f32, G>,
    root: NodeIndex,
    boundary: usize,
    members: &HashSet<NodeIndex>,
    context: &ConsumerContext<'_>,
    required: &mut HashSet<NodeIndex>,
) -> Option<Vec<(NodeIndex, NodeIndex, usize)>> {
    let ConsumerContext {
        positions,
        dtypes,
        packs,
    } = context;
    let mut result = Vec::new();
    let mut pending = vec![root];
    let mut seen = HashSet::new();
    while let Some(n) = pending.pop() {
        if !seen.insert(n) {
            continue;
        }
        for c in g.graph.neighbors_directed(n, Direction::Outgoing) {
            if members.contains(&c) {
                continue;
            }
            if let Some(p) = packs
                .iter()
                .find(|p| p.inputs.contains(&root) && p.members.contains(&c))
            {
                if positions[&p.output] <= boundary {
                    return None;
                }
                required.insert(p.output);
                result.push((root, p.output, positions[&p.output]));
                continue;
            }
            let dtype = match dtypes.get(&n)? {
                ValueDType::BF16 => DType::BF16,
                ValueDType::F32 => DType::F32,
            };
            let v = VirtualTensor::identity(n, g[n].shape().clone(), dtype);
            if full_view(g, c, &v, dtypes).is_some() {
                // Early BF16 metadata views and exact BF16 promotion are safe;
                // F32 views can materialize in the baseline planner.
                if positions[&c] <= boundary
                    && dtypes.get(&c) == Some(&ValueDType::F32)
                    && !matches!(g[c], TensorGraphNode::Cast { .. })
                {
                    return None;
                }
                pending.push(c);
            } else {
                if positions[&c] <= boundary {
                    return None;
                }
                result.push((root, c, positions[&c]));
            }
        }
    }
    Some(result)
}

/// Conditional static proof only. `selected_pack_outputs` must later be checked
/// against actual compiled semantic-owner groups, not inferred from a flag.
pub(super) fn recognize<G>(
    g: &TensorGraph<f32, G>,
    selected_pack_outputs: &HashSet<NodeIndex>,
) -> Vec<CompletePair> {
    let positions = g
        .toposort()
        .into_iter()
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect::<HashMap<_, _>>();
    let dtypes = value_dtypes(g);
    let backs = crate::ptx::mixed::attention_softmax_backward::recognize(g);
    let packs = crate::ptx::mixed::qkv_grad_pack::recognize(g)
        .into_iter()
        .filter(|p| selected_pack_outputs.contains(&p.output))
        .map(|p| Pack {
            output: p.output,
            members: p.members,
            inputs: p.inputs.into_iter().map(|i| i.source).collect(),
        })
        .collect::<Vec<_>>();
    let mut result = Vec::new();
    let mut claimed = HashSet::new();
    for f in crate::ptx::mixed::attention_training_forward::online_pair_forward_seeds(g) {
        let attempt = || -> Option<CompletePair> {
            let [b] = backs
                .iter()
                .filter(|b| b.probability == f.probability && b.shape == f.shape)
                .collect::<Vec<_>>()
                .try_into()
                .ok()?;
            let (dp, dp_aliases) = dp_root(g, b.incoming, &dtypes)?;
            let [incoming, vt] = g.inputs(dp).try_into().ok()?;
            operand(g, vt, f.value, true, &dtypes)?;
            let dense = vec![f.shape[0], f.shape[1], 512, 64];
            let ktshape = vec![f.shape[0], f.shape[1], 64, 512];
            if g[incoming].shape() != &dense || dtypes.get(&incoming) != Some(&ValueDType::BF16) {
                return None;
            }
            let (dq, dqa) = contraction(g, b.narrowed, false, f.key, true, &dense, &dtypes)?;
            let (dkt, dkta) = contraction(g, f.query, true, b.narrowed, false, &ktshape, &dtypes)?;
            let (dv, dva) = contraction(g, f.narrowed, true, incoming, false, &dense, &dtypes)?;
            let contractions = [f.qk, f.output, dp, dq, dkt, dv];
            if contractions.into_iter().collect::<HashSet<_>>().len() != 6 {
                return None;
            }
            let outputs = [f.output, dq, dkt, dv];
            let mut members = f.members.clone();
            members.extend(b.members.iter().copied());
            members.extend([dp, dq, dkt, dv]);
            members.extend(dp_aliases);
            // Only aliases sourced from discarded square checkpoints belong to
            // the suppressed union. Primal input views keep unrelated fanout.
            for aliases in [dqa, dkta, dva] {
                for n in aliases {
                    if ancestry(g, n, b.narrowed, &dtypes, 32).is_some()
                        || ancestry(g, n, f.narrowed, &dtypes, 32).is_some()
                    {
                        members.insert(n);
                    }
                }
            }
            close_interiors(g, &mut members, &outputs, &dtypes)?;
            let backward_boundary = *outputs[1..].iter().max_by_key(|&&n| positions[&n])?;
            let backward_position = positions[&backward_boundary];
            let forward_position = positions[&f.output];
            if [f.query, f.key, f.value]
                .iter()
                .any(|n| positions[n] >= forward_position)
                || positions[&incoming] >= backward_position
            {
                return None;
            }
            let context = ConsumerContext {
                positions: &positions,
                dtypes: &dtypes,
                packs: &packs,
            };
            let mut required = HashSet::new();
            let mut surviving = consumers(
                g,
                f.output,
                forward_position,
                &members,
                &context,
                &mut required,
            )?;
            for &out in &outputs[1..] {
                surviving.extend(consumers(
                    g,
                    out,
                    backward_position,
                    &members,
                    &context,
                    &mut required,
                )?);
            }
            let output_views =
                outputs.map(|n| VirtualTensor::identity(n, g[n].shape().clone(), DType::BF16));
            Some(CompletePair {
                inputs: [f.query, f.key, f.value, incoming],
                outputs,
                output_views,
                shape: f.shape.clone(),
                quadratic: [f.qk, f.probability, f.narrowed, dp, b.narrowed],
                contractions,
                members,
                forward_position,
                backward_boundary,
                backward_position,
                surviving_consumers: surviving,
                required_qkv_packs: required,
            })
        };
        if let Some(pair) = attempt() {
            if pair.members.iter().any(|n| claimed.contains(n)) {
                continue;
            }
            claimed.extend(pair.members.iter().copied());
            result.push(pair);
        }
    }
    result
}
#[cfg(test)]
mod tests;
