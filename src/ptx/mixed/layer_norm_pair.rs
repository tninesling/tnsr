//! Opt-in exact F32 W512/W1024 LayerNorm row-pair rewrite.
//!
//! Original selected plans provide the per-edge numerical contract.
//! Match the actual primitive AD graph before deciding any skipped nodes. The
//! normalized F32 matrix remains an interface for the unchanged gamma reduction;
//! residual additions and gamma/beta column schedules are outside the pair.
use super::*;
use crate::tensor::{BinaryOp, Parameter, ReduceOp, TensorExpr, UnaryOp};
use petgraph::Direction;
use std::collections::{HashMap, HashSet};

mod boundaries;
#[cfg(test)]
mod gpu_tests;
mod kernel;
mod lifetime;
pub(super) mod rewrite;
mod ssa;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dimension {
    Batch,
    Sequence,
    Width,
    One,
}
fn dimensions(shape: &[usize]) -> Result<Vec<Dimension>> {
    shape
        .iter()
        .map(|&n| {
            Ok(match n {
                1 => Dimension::One,
                2 => Dimension::Batch,
                3 => Dimension::Sequence,
                512 | 1024 => Dimension::Width,
                _ => anyhow::bail!("unexpected LayerNorm template dimension {n}"),
            })
        })
        .collect()
}
fn concrete(shape: &[Dimension], geometry: [usize; 3]) -> Vec<usize> {
    shape
        .iter()
        .map(|d| match d {
            Dimension::Batch => geometry[0],
            Dimension::Sequence => geometry[1],
            Dimension::Width => geometry[2],
            Dimension::One => 1,
        })
        .collect()
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Input(usize),
    Constant(u32),
    Unary(UnaryOp),
    Binary(BinaryOp),
    Reduce(ReduceOp, usize),
    Broadcast(usize),
    Reshape,
}
#[derive(Clone, Debug)]
struct Operation {
    kind: Kind,
    shape: Vec<Dimension>,
    inputs: Vec<usize>,
    node: NodeIndex,
}
struct Pattern {
    width: usize,
    ops: Vec<Operation>,
    forward: usize,
    normalized: usize,
    mean: usize,
    variance_plus_epsilon: usize,
    inverse_std: usize,
    backward: usize,
}
fn one<G>(g: &TensorGraph<f32, G>, n: NodeIndex) -> Option<NodeIndex> {
    let [n] = g.inputs(n).try_into().ok()?;
    Some(n)
}
/// Strip only genuine F32 identities, never a BF16 promotion or primal input
/// boundary. Inputs are checked before invoking this helper.
fn normalize<G>(
    g: &TensorGraph<f32, G>,
    dtypes: &HashMap<NodeIndex, ValueDType>,
    mut n: NodeIndex,
    aliases: &mut HashSet<NodeIndex>,
) -> NodeIndex {
    while matches!(
        g[n],
        TensorGraphNode::Cast {
            dtype: ValueDType::F32,
            ..
        }
    ) {
        let Some(source) = one(g, n) else {
            break;
        };
        if dtypes.get(&source) != Some(&ValueDType::F32) || g[source].shape() != g[n].shape() {
            break;
        }
        aliases.insert(n);
        n = source;
    }
    n
}
fn kind<G>(g: &TensorGraph<f32, G>, n: NodeIndex) -> Option<Kind> {
    Some(match &g[n] {
        TensorGraphNode::Constant { data, .. } if data.len() == 1 => {
            Kind::Constant(data[0].to_bits())
        }
        TensorGraphNode::Unary { op, .. } => Kind::Unary(*op),
        TensorGraphNode::Binary { op, .. } => Kind::Binary(*op),
        TensorGraphNode::ReduceAxis { op, axis, .. } => Kind::Reduce(*op, *axis),
        TensorGraphNode::BroadcastAxis { axis, .. } => Kind::Broadcast(*axis),
        TensorGraphNode::Reshape { .. } | TensorGraphNode::Flatten { .. } => Kind::Reshape,
        _ => return None,
    })
}
impl Pattern {
    fn new(width: usize, prototype_batch: usize) -> Result<Self> {
        ensure!(
            matches!(width, 512 | 1024),
            "unsupported LayerNorm template width"
        );
        // Distinct B/S sizes preserve symbolic axis identity and both parameter
        // broadcast VJPs without allocating a benchmark-size template.
        ensure!(
            matches!(prototype_batch, 1 | 2),
            "unsupported LayerNorm template batch class"
        );
        let shape = vec![prototype_batch, 3, width];
        let x = Parameter::new(vec![0.; prototype_batch * 3 * width], shape.clone());
        let gamma = Parameter::new(vec![1.; width], vec![width]);
        let beta = Parameter::new(vec![0.; width], vec![width]);
        let expression = crate::nn::layer_norm(
            TensorExpr::from(x.clone()).cast(ValueDType::F32),
            2,
            gamma.clone(),
            beta.clone(),
            1e-5,
        );
        let mut g = TensorGraph::<f32>::new();
        let forward = g.add_expr(&expression, false);
        let loss = g.add_expr(
            &TensorExpr::node_ref(forward, shape.clone())
                .reduce_axis_sum(2)
                .reduce_axis_sum(1)
                .reduce_axis_sum(0),
            false,
        );
        let g = g.with_gradients(loss);
        let dtypes = value_dtypes(&g);
        let source = |id| {
            g.graph
                .node_indices()
                .find(|n| matches!(g[*n],TensorGraphNode::Parameter{id:i,..}if i==id))
                .context("LayerNorm template parameter missing")
        };
        let x_node = source(x.id())?;
        let z = g
            .graph
            .neighbors_directed(x_node, Direction::Outgoing)
            .find(|n| {
                matches!(
                    g[*n],
                    TensorGraphNode::Cast {
                        dtype: ValueDType::F32,
                        ..
                    }
                )
            })
            .context("LayerNorm explicit primal F32 cast missing")?;
        let mut incoming = g.gradient_metadata().param_to_grad[&beta.id()];
        // The beta VJP contains only reshape/column sums/identity casts before
        // its full incoming gradient. Keep that complete value as an input.
        loop {
            incoming = normalize(&g, &dtypes, incoming, &mut HashSet::new());
            if g[incoming].shape() == &shape {
                break;
            }
            ensure!(
                matches!(
                    g[incoming],
                    TensorGraphNode::Reshape { .. }
                        | TensorGraphNode::Flatten { .. }
                        | TensorGraphNode::ReduceAxis {
                            op: ReduceOp::Sum,
                            ..
                        }
                ),
                "LayerNorm beta template VJP changed"
            );
            incoming = one(&g, incoming).context("LayerNorm beta VJP arity changed")?;
        }
        let backward = normalize(
            &g,
            &dtypes,
            g.gradient_metadata().param_to_grad[&x.id()],
            &mut HashSet::new(),
        );
        // Source-defined forward anchors. Every edge is also matched below.
        let p = g.inputs(forward)[0];
        let normalized = g.inputs(p)[0];
        let c = g.inputs(normalized)[0];
        let inverse_std = g.inputs(normalized)[1];
        let mean = one(&g, g.inputs(c)[1]).context("LayerNorm mean broadcast changed")?;
        let t = one(&g, inverse_std).context("LayerNorm exp arity changed")?;
        let log = g.inputs(t)[0];
        let variance_plus_epsilon = one(&g, log).context("LayerNorm log arity changed")?;
        let inputs = HashMap::from([
            (z, 0),
            (source(gamma.id())?, 1),
            (source(beta.id())?, 2),
            (incoming, 3),
        ]);
        fn visit<G>(
            g: &TensorGraph<f32, G>,
            dtypes: &HashMap<NodeIndex, ValueDType>,
            mut n: NodeIndex,
            inputs: &HashMap<NodeIndex, usize>,
            values: &mut HashMap<NodeIndex, usize>,
            ops: &mut Vec<Operation>,
        ) -> Result<usize> {
            if !inputs.contains_key(&n) {
                n = normalize(g, dtypes, n, &mut HashSet::new());
            }
            if let Some(v) = values.get(&n) {
                return Ok(*v);
            }
            ensure!(
                dtypes.get(&n) == Some(&ValueDType::F32),
                "LayerNorm template must stay F32"
            );
            let (kind, args) = if let Some(i) = inputs.get(&n) {
                (Kind::Input(*i), Vec::new())
            } else {
                let k = kind(g, n).context("unsupported LayerNorm template node")?;
                let args = if matches!(k, Kind::Constant(_)) {
                    Vec::new()
                } else {
                    g.inputs(n)
                        .into_iter()
                        .map(|a| visit(g, dtypes, a, inputs, values, ops))
                        .collect::<Result<Vec<_>>>()?
                };
                (k, args)
            };
            let id = ops.len();
            ops.push(Operation {
                kind,
                shape: dimensions(g[n].shape())?,
                inputs: args,
                node: n,
            });
            values.insert(n, id);
            Ok(id)
        }
        let mut values = HashMap::new();
        let mut ops = Vec::new();
        let mut extract = |n| visit(&g, &dtypes, n, &inputs, &mut values, &mut ops);
        let forward = extract(forward)?;
        let normalized = extract(normalized)?;
        let mean = extract(mean)?;
        let variance_plus_epsilon = extract(variance_plus_epsilon)?;
        let inverse_std = extract(inverse_std)?;
        let backward = extract(backward)?;
        Ok(Self {
            width,
            ops,
            forward,
            normalized,
            mean,
            variance_plus_epsilon,
            inverse_std,
            backward,
        })
    }
}
#[derive(Clone, Default)]
struct Match {
    nodes: HashMap<usize, NodeIndex>,
    inputs: HashMap<usize, NodeIndex>,
    aliases: HashSet<NodeIndex>,
}
fn match_value<G>(
    g: &TensorGraph<f32, G>,
    dtypes: &HashMap<NodeIndex, ValueDType>,
    p: &Pattern,
    id: usize,
    mut n: NodeIndex,
    geometry: [usize; 3],
    m: &mut Match,
) -> Option<()> {
    let op = &p.ops[id];
    match op.kind {
        Kind::Input(0) => {}
        Kind::Input(_) => {
            n = normalize(g, dtypes, n, &mut HashSet::new());
        }
        _ => {
            n = normalize(g, dtypes, n, &mut m.aliases);
        }
    }
    if let Some(old) = m.nodes.get(&id) {
        return (*old == n).then_some(());
    }
    if geometry[2] != p.width
        || dtypes.get(&n) != Some(&ValueDType::F32)
        || g[n].shape() != &concrete(&op.shape, geometry)
    {
        return None;
    }
    if let Kind::Input(i) = op.kind {
        if i == 0
            && (!matches!(
                g[n],
                TensorGraphNode::Cast {
                    dtype: ValueDType::F32,
                    ..
                }
            ) || one(g, n).is_none_or(|a| {
                dtypes.get(&a) != Some(&ValueDType::F32) || g[a].shape() != g[n].shape()
            }))
        {
            return None;
        }
        if matches!(i, 1 | 2) && !matches!(g[n], TensorGraphNode::Parameter { .. }) {
            return None;
        }
        if m.inputs.get(&i).is_some_and(|old| *old != n) {
            return None;
        }
        m.inputs.insert(i, n);
        m.nodes.insert(id, n);
        return Some(());
    }
    if kind(g, n)? != op.kind {
        return None;
    }
    let args = if matches!(op.kind, Kind::Constant(_)) {
        Vec::new()
    } else {
        g.inputs(n)
    };
    if args.len() != op.inputs.len() {
        return None;
    }
    m.nodes.insert(id, n);
    for (&id, n) in op.inputs.iter().zip(args) {
        match_value(g, dtypes, p, id, n, geometry, m)?;
    }
    Some(())
}
/// Graph-only proof. Runtime descriptors must still prove the identical flat
/// mapping before skipping members; no permutation/predicate is accepted here.
fn storage_root<G>(
    g: &TensorGraph<f32, G>,
    dtypes: &HashMap<NodeIndex, ValueDType>,
    mut n: NodeIndex,
) -> Option<NodeIndex> {
    let extent = |s: &[usize]| s.iter().try_fold(1_usize, |a, &b| a.checked_mul(b));
    let count = extent(g[n].shape())?;
    loop {
        if dtypes.get(&n) != Some(&ValueDType::F32) {
            return None;
        }
        match &g[n] {
            TensorGraphNode::Reshape { .. } | TensorGraphNode::Flatten { .. } => {
                let a = one(g, n)?;
                if extent(g[a].shape()) != Some(count) {
                    return None;
                }
                n = a;
            }
            TensorGraphNode::Cast {
                dtype: ValueDType::F32,
                ..
            } => {
                let a = one(g, n)?;
                if dtypes.get(&a) == Some(&ValueDType::BF16) {
                    return Some(n);
                }
                if g[a].shape() != g[n].shape() {
                    return None;
                }
                n = a;
            }
            op if is_view(op) => return None,
            _ => return Some(n),
        }
    }
}
fn gamma_column_interface<G>(
    g: &TensorGraph<f32, G>,
    n: NodeIndex,
    incoming: NodeIndex,
    members: &HashSet<NodeIndex>,
) -> bool {
    // This interface remains a real full F32 primal. Only its original gamma
    // product may escape; preserve its producer and column reduction wholesale.
    let dtypes = value_dtypes(g);
    let external = g
        .graph
        .neighbors_directed(n, Direction::Outgoing)
        .filter(|a| !members.contains(a))
        .collect::<Vec<_>>();
    external.len() == 1
        && external.into_iter().all(|a| {
            let a = normalize(g, &dtypes, a, &mut HashSet::new());
            let args = g.inputs(a);
            matches!(
                g[a],
                TensorGraphNode::Binary {
                    op: BinaryOp::Mul,
                    ..
                }
            ) && args.len() == 2
                && normalize(g, &dtypes, args[0], &mut HashSet::new()) == incoming
                && normalize(g, &dtypes, args[1], &mut HashSet::new()) == n
        })
}
pub(super) struct LayerNormPair {
    pub shape: Vec<usize>,
    pub inputs: [NodeIndex; 4],
    pub storage_roots: [NodeIndex; 4],
    pub forward: NodeIndex,
    pub normalized: NodeIndex,
    pub mean: NodeIndex,
    pub variance_plus_epsilon: NodeIndex,
    pub inverse_std: NodeIndex,
    pub backward: NodeIndex,
    pub members: HashSet<NodeIndex>,
    pub forward_members: HashSet<NodeIndex>,
    /// SSA graph remains original primitive operations, including each Sum and
    /// both square-VJP products and the width-specific exact reciprocal.
    program: Vec<Operation>,
}
/// Exactly the two row widths and narrowly bounded sequence/batch domains.
/// Geometry is only a first guard: the selected original 18 edges and four
/// reduction trees still decide whether a particular pair can execute.
fn supported_geometry(shape: &[usize]) -> bool {
    let [batch, sequence, width] = shape else {
        return false;
    };
    let valid_domain = (*sequence == 512 && *batch >= 1) || (*sequence == 128 && *batch == 2);
    if !valid_domain || !matches!(*width, 512 | 1024) {
        return false;
    }
    batch
        .checked_mul(*sequence)
        .filter(|rows| *rows >= 256)
        .and_then(|rows| rows.checked_mul(*width))
        .is_some_and(|elements| u32::try_from(elements).is_ok())
}
pub(super) fn recognize<G>(g: &TensorGraph<f32, G>) -> Result<Vec<LayerNormPair>> {
    // Generic broadcast omits the batch-axis operation when B==1. Preserve
    // that actual forward/VJP graph instead of normalizing away extra nodes.
    let patterns = [
        Pattern::new(512, 1)?,
        Pattern::new(512, 2)?,
        Pattern::new(1024, 1)?,
        Pattern::new(1024, 2)?,
    ];
    let dtypes = value_dtypes(g);
    let topo = g.toposort();
    let positions: HashMap<_, _> = topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let mut found = Vec::new();
    let mut claimed = HashSet::new();
    for &forward in &topo {
        let shape = g[forward].shape();
        if !supported_geometry(shape)
            || !matches!(
                g[forward],
                TensorGraphNode::Binary {
                    op: BinaryOp::Add,
                    ..
                }
            )
        {
            continue;
        }
        if shape
            .iter()
            .try_fold(1_usize, |a, &b| a.checked_mul(b))
            .is_none_or(|n| u32::try_from(n).is_err())
        {
            continue;
        }
        let pattern = &patterns[2 * usize::from(shape[2] == 1024) + usize::from(shape[0] != 1)];
        let geometry = [shape[0], shape[1], shape[2]];
        let mut f = Match::default();
        if match_value(
            g,
            &dtypes,
            pattern,
            pattern.forward,
            forward,
            geometry,
            &mut f,
        )
        .is_none()
        {
            continue;
        }
        let mut forward_members: HashSet<_> = f
            .nodes
            .iter()
            .filter_map(|(&id, &n)| {
                (!matches!(pattern.ops[id].kind, Kind::Input(_) | Kind::Constant(_))).then_some(n)
            })
            .collect();
        forward_members.extend(f.aliases.iter().copied());
        for &backward in &topo {
            if !matches!(
                g[backward],
                TensorGraphNode::Binary {
                    op: BinaryOp::Add,
                    ..
                }
            ) {
                continue;
            }
            let mut full = f.clone();
            if match_value(
                g,
                &dtypes,
                pattern,
                pattern.backward,
                backward,
                geometry,
                &mut full,
            )
            .is_none()
            {
                continue;
            }
            let inputs = [
                full.inputs[&0],
                full.inputs[&1],
                full.inputs[&2],
                full.inputs[&3],
            ];
            let Some(roots) = inputs
                .iter()
                .map(|&n| storage_root(g, &dtypes, n))
                .collect::<Option<Vec<_>>>()
                .and_then(|v| <[NodeIndex; 4]>::try_from(v).ok())
            else {
                continue;
            };
            let mut members: HashSet<_> = full
                .nodes
                .iter()
                .filter_map(|(&id, &n)| {
                    (!matches!(pattern.ops[id].kind, Kind::Input(_) | Kind::Constant(_)))
                        .then_some(n)
                })
                .collect();
            members.extend(full.aliases.iter().copied());
            let normalized = full.nodes[&pattern.normalized];
            if members.iter().any(|n| claimed.contains(n))
                || members.iter().any(|n| {
                    ![forward, backward, normalized].contains(n)
                        && g.graph
                            .neighbors_directed(*n, Direction::Outgoing)
                            .any(|a| !members.contains(&a))
                })
                || !gamma_column_interface(g, normalized, inputs[3], &members)
            {
                continue;
            }
            if inputs[..3]
                .iter()
                .any(|n| positions[n] >= positions[&forward])
                || positions[&inputs[3]] >= positions[&backward]
            {
                continue;
            }
            let program = pattern
                .ops
                .iter()
                .enumerate()
                .map(|(id, op)| {
                    let mut op = op.clone();
                    op.node = full.nodes[&id];
                    op
                })
                .collect();
            claimed.extend(members.iter().copied());
            found.push(LayerNormPair {
                shape: shape.clone(),
                inputs,
                storage_roots: roots,
                forward,
                normalized,
                mean: full.nodes[&pattern.mean],
                variance_plus_epsilon: full.nodes[&pattern.variance_plus_epsilon],
                inverse_std: full.nodes[&pattern.inverse_std],
                backward,
                members,
                forward_members: forward_members.clone(),
                program,
            });
            break;
        }
    }
    Ok(found)
}

/// Pure graph preflight for the restricted materialized F32 boundary classes.
/// The ordinary typed programs store these compute results in contiguous order;
/// view roots have already been rejected by `storage_root`. Runtime metadata is
/// checked independently before any integration may declare member skips.
fn planned_contiguous<G>(pair: &LayerNormPair, g: &TensorGraph<f32, G>) -> bool {
    pair.storage_roots.iter().all(|n| {
        matches!(
            g[*n],
            TensorGraphNode::Input { .. }
                | TensorGraphNode::Parameter { .. }
                | TensorGraphNode::Constant { .. }
                | TensorGraphNode::Binary { .. }
                | TensorGraphNode::Unary { .. }
                | TensorGraphNode::MatMul { .. }
                | TensorGraphNode::Linear { .. }
                | TensorGraphNode::ReduceAxis { .. }
                | TensorGraphNode::Embedding { .. }
                | TensorGraphNode::Cast {
                    dtype: ValueDType::F32,
                    ..
                }
        )
    })
}
fn contiguous_bindings(
    pair: &LayerNormPair,
    available: &HashMap<NodeIndex, Value>,
    backward: bool,
) -> Option<Vec<Value>> {
    let extent = |s: &[usize]| s.iter().try_fold(1_usize, |a, &b| a.checked_mul(b));
    let total = extent(&pair.shape)?;
    let which = if backward { [0, 1, 3] } else { [0, 1, 2] };
    let mut result = Vec::new();
    for i in which {
        let root = pair.storage_roots[i];
        let v = available.get(&root)?;
        let expected_count = if i == 1 || i == 2 {
            pair.shape[2]
        } else {
            total
        };
        if (!matches!(v.binding.dtype, DType::F32 | DType::BF16)
            || (i != 3 && v.binding.dtype != DType::F32))
            || v.view.dtype != v.binding.dtype
            || v.binding.len != expected_count
            || extent(&v.source_shape) != Some(expected_count)
            || v.view.predicate.is_some()
            || v.view.out_of_bounds.is_some()
        {
            return None;
        }
        let mut expected =
            VirtualTensor::identity(v.view.source, v.source_shape.clone(), v.binding.dtype);
        for alias in &v.chain {
            match alias {
                TensorGraphNode::Reshape { shape } | TensorGraphNode::Flatten { shape }
                    if extent(shape) == Some(expected_count) =>
                {
                    expected = expected.reshape(shape.clone()).ok()?;
                }
                _ => return None,
            }
        }
        if expected != v.view {
            return None;
        }
        let mut physical = v.clone();
        physical.view =
            VirtualTensor::identity(v.view.source, v.source_shape.clone(), v.binding.dtype);
        physical.chain.clear();
        result.push(physical);
    }
    Some(result)
}
