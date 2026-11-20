//! Undeclared preparation for exact mixed-storage SwiGLU recomputation.
//!
//! Match the stable primitive forward/VJP DAG, including operand/gradient-Add
//! order. Recompute saved f32 locals from BF16 inputs; genuine BF16 boundaries
//! use private per-lane shared slots. Selective f32 shared checkpoints represent
//! former saved-primal stores. Their final contraction policy still requires a
//! baseline/fused PTX+SASS and real-device bit audit before enabling integration.
use super::*;
use crate::tensor::{BinaryOp, Parameter, UnaryOp};
use crate::tile::{
    Block, Dim, Expr, KernelParam, MemorySpace, RegionInput, ScalarBinaryOp, ScalarBuilder,
    ScalarExpr, ScalarMemory, SharedLayout, Stmt, TileIR, TileLayout, TileVar,
};
use petgraph::Direction;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Input(usize),
    Constant(u32),
    Unary(UnaryOp),
    Binary(BinaryOp),
    Gt,
    Mask,
    RoundBF16,
}
#[derive(Clone, Debug)]
struct Operation {
    kind: Kind,
    inputs: Vec<usize>,
    dtype: ValueDType,
    node: NodeIndex,
}
#[derive(Clone, Debug)]
struct ScalarProgram {
    ops: Vec<Operation>,
    outputs: Vec<usize>,
}
fn one<G>(g: &TensorGraph<f32, G>, n: NodeIndex) -> Option<NodeIndex> {
    let [input] = g.inputs(n).try_into().ok()?;
    Some(input)
}
fn scalar_source<G>(
    g: &TensorGraph<f32, G>,
    mut n: NodeIndex,
) -> Option<(NodeIndex, Vec<NodeIndex>)> {
    let mut aliases = Vec::new();
    loop {
        match &g[n] {
            TensorGraphNode::Constant { data, .. } if data.len() == 1 => return Some((n, aliases)),
            TensorGraphNode::Reshape { shape } | TensorGraphNode::Flatten { shape } => {
                let source = one(g, n)?;
                if shape.iter().product::<usize>() != 1
                    || g[source].shape().iter().product::<usize>() != 1
                {
                    return None;
                }
                aliases.push(n);
                n = source;
            }
            TensorGraphNode::BroadcastAxis { axis, shape } => {
                let source = one(g, n)?;
                let s = g[source].shape();
                if s.len() != shape.len()
                    || *axis >= s.len()
                    || s[*axis] != 1
                    || shape.contains(&0)
                    || s.iter()
                        .enumerate()
                        .any(|(i, &v)| i != *axis && v != shape[i])
                {
                    return None;
                }
                aliases.push(n);
                n = source;
            }
            _ => return None,
        }
    }
}
fn normalize<G>(
    g: &TensorGraph<f32, G>,
    dtypes: &HashMap<NodeIndex, ValueDType>,
    mut n: NodeIndex,
    aliases: &mut HashMap<NodeIndex, NodeIndex>,
) -> NodeIndex {
    let mut path = Vec::new();
    loop {
        if let Some((source, views)) = scalar_source(g, n) {
            path.extend(views);
            n = source;
            break;
        }
        let cast = match g[n] {
            TensorGraphNode::Cast {
                dtype: ValueDType::F32,
                ..
            } => one(g, n),
            TensorGraphNode::Cast {
                dtype: ValueDType::BF16,
                ..
            } => one(g, n).filter(|source| dtypes.get(source) == Some(&ValueDType::BF16)),
            _ => None,
        };
        let Some(source) = cast else {
            break;
        };
        path.push(n);
        n = source;
    }
    for a in path {
        aliases.insert(a, n);
    }
    n
}
fn operation<G>(g: &TensorGraph<f32, G>, n: NodeIndex) -> Option<Kind> {
    Some(match &g[n] {
        TensorGraphNode::Constant { data, .. } if data.len() == 1 => {
            Kind::Constant(data[0].to_bits())
        }
        TensorGraphNode::Unary { op, .. } => Kind::Unary(*op),
        TensorGraphNode::Binary { op, .. } => Kind::Binary(*op),
        TensorGraphNode::Gt { .. } => Kind::Gt,
        TensorGraphNode::Mask { .. } => Kind::Mask,
        TensorGraphNode::Cast {
            dtype: ValueDType::BF16,
            ..
        } => Kind::RoundBF16,
        _ => return None,
    })
}
impl ScalarProgram {
    fn extract<G>(
        g: &TensorGraph<f32, G>,
        outputs: &[NodeIndex],
        inputs: &HashMap<NodeIndex, usize>,
    ) -> Result<Self> {
        fn visit<G>(
            g: &TensorGraph<f32, G>,
            dtypes: &HashMap<NodeIndex, ValueDType>,
            n: NodeIndex,
            inputs: &HashMap<NodeIndex, usize>,
            values: &mut HashMap<NodeIndex, usize>,
            ops: &mut Vec<Operation>,
        ) -> Result<usize> {
            let n = if inputs.contains_key(&n) {
                n
            } else {
                normalize(g, dtypes, n, &mut HashMap::new())
            };
            if let Some(&v) = values.get(&n) {
                return Ok(v);
            }
            let (kind, args) = if let Some(&i) = inputs.get(&n) {
                (Kind::Input(i), Vec::new())
            } else {
                (
                    operation(g, n).context("unsupported mixed SwiGLU scalar operation")?,
                    g.inputs(n)
                        .into_iter()
                        .map(|a| visit(g, dtypes, a, inputs, values, ops))
                        .collect::<Result<_>>()?,
                )
            };
            let v = ops.len();
            ops.push(Operation {
                kind,
                inputs: args,
                dtype: dtypes.get(&n).copied().unwrap_or(ValueDType::F32),
                node: n,
            });
            values.insert(n, v);
            Ok(v)
        }
        let dtypes = value_dtypes(g);
        let mut ops = Vec::new();
        let mut values = HashMap::new();
        let outputs = outputs
            .iter()
            .map(|&n| visit(g, &dtypes, n, inputs, &mut values, &mut ops))
            .collect::<Result<_>>()?;
        Ok(Self { ops, outputs })
    }
}
struct Pattern {
    program: ScalarProgram,
    forward: usize,
    gate_gradient: usize,
    up_gradient: usize,
}
impl Pattern {
    fn new() -> Result<Self> {
        let gate = Parameter::new(vec![0_f32], vec![1]);
        let up = Parameter::new(vec![0_f32], vec![1]);
        let x = crate::tensor::TensorExpr::from(gate.clone()).cast(ValueDType::BF16);
        let u = crate::tensor::TensorExpr::from(up.clone()).cast(ValueDType::BF16);
        let hidden = crate::nn::silu(x.clone().cast(ValueDType::F32)).cast(ValueDType::BF16) * u;
        let mut graph = TensorGraph::new();
        let forward = graph.add_expr(&hidden, false);
        let prefix = graph.len();
        let graph = graph.with_gradients(forward);
        let dtypes = value_dtypes(&graph);
        let source = |id| {
            graph
                .graph
                .node_indices()
                .find(|&n| matches!(graph[n],TensorGraphNode::Parameter {id:i,..} if i==id))
                .context("mixed SwiGLU template source absent")
        };
        let cast = |source| {
            graph
                .graph
                .neighbors_directed(source, Direction::Outgoing)
                .find(|&n| {
                    matches!(
                        graph[n],
                        TensorGraphNode::Cast {
                            dtype: ValueDType::BF16,
                            ..
                        }
                    )
                })
                .context("mixed SwiGLU template BF16 source absent")
        };
        let x = cast(source(gate.id())?)?;
        let up_id = up.id();
        let up = cast(source(up_id)?)?;
        let seed = graph
            .graph
            .node_indices()
            .find(|&n| {
                n.index() >= prefix
                    && matches!(
                        graph[n],
                        TensorGraphNode::Cast {
                            dtype: ValueDType::BF16,
                            ..
                        }
                    )
                    && one(&graph, n).is_some_and(|s| s.index() == prefix)
            })
            .context("mixed SwiGLU seed contract changed")?;
        let gradients = &graph.gradient_metadata().param_to_grad;
        let a = normalize(&graph, &dtypes, gradients[&gate.id()], &mut HashMap::new());
        let b = normalize(&graph, &dtypes, gradients[&up_id], &mut HashMap::new());
        let program = ScalarProgram::extract(
            &graph,
            &[forward, a, b],
            &HashMap::from([(x, 0), (up, 1), (seed, 2)]),
        )?;
        Ok(Self {
            forward: program.outputs[0],
            gate_gradient: program.outputs[1],
            up_gradient: program.outputs[2],
            program,
        })
    }
}
#[derive(Clone, Default)]
struct Match {
    nodes: HashMap<usize, NodeIndex>,
    inputs: HashMap<usize, NodeIndex>,
    aliases: HashMap<NodeIndex, NodeIndex>,
}
fn match_value<G>(
    g: &TensorGraph<f32, G>,
    dtypes: &HashMap<NodeIndex, ValueDType>,
    pattern: &ScalarProgram,
    id: usize,
    actual: NodeIndex,
    shape: &[usize],
    matched: &mut Match,
) -> Option<()> {
    let actual = normalize(g, dtypes, actual, &mut matched.aliases);
    let op = &pattern.ops[id];
    if let Some(&old) = matched.nodes.get(&id) {
        return (old == actual).then_some(());
    }
    if let Kind::Input(input) = op.kind {
        if dtypes.get(&actual) != Some(&ValueDType::BF16) || g[actual].shape() != shape {
            return None;
        }
        if let Some(&old) = matched.inputs.get(&input)
            && old != actual
        {
            return None;
        }
        matched.inputs.insert(input, actual);
        matched.nodes.insert(id, actual);
        return Some(());
    }
    if operation(g, actual)? != op.kind
        || dtypes.get(&actual).copied().unwrap_or(ValueDType::F32) != op.dtype
    {
        return None;
    }
    if !matches!(op.kind, Kind::Constant(_)) && g[actual].shape() != shape {
        return None;
    }
    let args = if matches!(op.kind, Kind::Constant(_)) {
        Vec::new()
    } else {
        g.inputs(actual)
    };
    if args.len() != op.inputs.len() {
        return None;
    }
    matched.nodes.insert(id, actual);
    for (&child, actual) in op.inputs.iter().zip(args) {
        match_value(g, dtypes, pattern, child, actual, shape, matched)?;
    }
    Some(())
}
fn members(pattern: &ScalarProgram, matched: &Match) -> HashSet<NodeIndex> {
    let mut set: HashSet<_> = matched
        .nodes
        .iter()
        .filter_map(|(&id, &n)| {
            (!matches!(pattern.ops[id].kind, Kind::Input(_) | Kind::Constant(_))).then_some(n)
        })
        .collect();
    for (&alias, &source) in &matched.aliases {
        if set.contains(&source) {
            set.insert(alias);
        }
    }
    set
}
/// Establish a flat-storage descriptor without accepting reordering or replication.
/// Runtime bindings must additionally pass `contiguous_values`: cast/output
/// optimizations may retain a physical view even at a graph computation node.
fn flat_storage_root<G>(
    g: &TensorGraph<f32, G>,
    dtypes: &HashMap<NodeIndex, ValueDType>,
    mut node: NodeIndex,
) -> Option<NodeIndex> {
    let extent = |shape: &[usize]| shape.iter().try_fold(1_usize, |n, &d| n.checked_mul(d));
    let count = extent(g[node].shape())?;
    loop {
        if dtypes.get(&node) != Some(&ValueDType::BF16) {
            return None;
        }
        match &g[node] {
            TensorGraphNode::Reshape { .. } | TensorGraphNode::Flatten { .. } => {
                let source = one(g, node)?;
                if extent(g[source].shape()) != Some(count) {
                    return None;
                }
                node = source;
            }
            TensorGraphNode::Cast {
                dtype: ValueDType::BF16,
                ..
            } if one(g, node).is_some_and(|n| dtypes.get(&n) == Some(&ValueDType::BF16)) => {
                node = one(g, node)?;
            }
            n if is_view(n) => return None,
            _ => return Some(node),
        }
    }
}
fn contiguous_values(
    region: &SwiGluPair,
    available: &HashMap<NodeIndex, Value>,
    backward: bool,
) -> Option<Vec<Value>> {
    let count = region
        .shape
        .iter()
        .try_fold(1_usize, |n, &d| n.checked_mul(d))?;
    let number = if backward { 3 } else { 2 };
    let mut result = Vec::new();
    for &root in &region.storage_roots[..number] {
        let value = available.get(&root)?;
        let extent = |s: &[usize]| s.iter().try_fold(1_usize, |n, &d| n.checked_mul(d));
        if value.binding.dtype != DType::BF16
            || value.view.dtype != DType::BF16
            || value.binding.len != count
            || extent(&value.source_shape) != Some(count)
            || value.view.predicate.is_some()
            || value.view.out_of_bounds.is_some()
        {
            return None;
        }
        let mut expected =
            VirtualTensor::identity(value.view.source, value.source_shape.clone(), DType::BF16);
        for operation in &value.chain {
            match operation {
                TensorGraphNode::Reshape { shape } | TensorGraphNode::Flatten { shape }
                    if extent(shape) == Some(count) =>
                {
                    expected = expected.reshape(shape.clone()).ok()?
                }
                _ => return None,
            }
        }
        if expected != value.view {
            return None;
        }
        let mut physical = value.clone();
        physical.view =
            VirtualTensor::identity(value.view.source, value.source_shape.clone(), DType::BF16);
        physical.chain.clear();
        result.push(physical);
    }
    Some(result)
}
/// No runtime selection/flag is declared by this preparation module.
pub(super) struct SwiGluPair {
    pub shape: Vec<usize>,
    pub inputs: [NodeIndex; 3],
    pub storage_roots: [NodeIndex; 3],
    pub hidden: NodeIndex,
    pub gate_gradient: NodeIndex,
    pub up_gradient: NodeIndex,
    pub forward_emit: NodeIndex,
    pub backward_emit: NodeIndex,
    pub members: HashSet<NodeIndex>,
    pub saved_f32_checkpoints: HashSet<NodeIndex>,
    forward: ScalarProgram,
    backward: ScalarProgram,
}
pub(super) fn recognize<G>(g: &TensorGraph<f32, G>) -> Result<Vec<SwiGluPair>> {
    let pattern = Pattern::new()?;
    let dtypes = value_dtypes(g);
    let topo = g.toposort();
    let positions: HashMap<_, _> = topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let mut found = Vec::new();
    let mut claimed = HashSet::new();
    for &hidden in &topo {
        if !matches!(
            g[hidden],
            TensorGraphNode::Binary {
                op: BinaryOp::Mul,
                ..
            }
        ) || dtypes.get(&hidden) != Some(&ValueDType::BF16)
        {
            continue;
        }
        let shape = g[hidden].shape().clone();
        if shape.is_empty() || shape.contains(&0) {
            continue;
        }
        let mut forward = Match::default();
        if match_value(
            g,
            &dtypes,
            &pattern.program,
            pattern.forward,
            hidden,
            &shape,
            &mut forward,
        )
        .is_none()
        {
            continue;
        }
        let forward_members = members(&pattern.program, &forward);
        'candidate: for &up_gradient in &topo {
            if normalize(g, &dtypes, up_gradient, &mut HashMap::new()) != up_gradient {
                continue;
            }
            let mut up_match = forward.clone();
            if match_value(
                g,
                &dtypes,
                &pattern.program,
                pattern.up_gradient,
                up_gradient,
                &shape,
                &mut up_match,
            )
            .is_none()
            {
                continue;
            }
            for &gate_gradient in &topo {
                if normalize(g, &dtypes, gate_gradient, &mut HashMap::new()) != gate_gradient {
                    continue;
                }
                let mut full = up_match.clone();
                if match_value(
                    g,
                    &dtypes,
                    &pattern.program,
                    pattern.gate_gradient,
                    gate_gradient,
                    &shape,
                    &mut full,
                )
                .is_none()
                {
                    continue;
                }
                let inputs = [full.inputs[&0], full.inputs[&1], full.inputs[&2]];
                if inputs.iter().collect::<HashSet<_>>().len() != 3 {
                    continue;
                }
                let Some(storage_roots) = inputs
                    .iter()
                    .map(|&n| flat_storage_root(g, &dtypes, n))
                    .collect::<Option<Vec<_>>>()
                    .and_then(|v| <[NodeIndex; 3]>::try_from(v).ok())
                else {
                    continue;
                };
                let members = members(&pattern.program, &full);
                let outputs = HashSet::from([hidden, gate_gradient, up_gradient]);
                if members.iter().any(|n| claimed.contains(n))
                    || members.iter().any(|&n| {
                        !outputs.contains(&n)
                            && g.graph
                                .neighbors_directed(n, Direction::Outgoing)
                                .any(|consumer| !members.contains(&consumer))
                    })
                {
                    continue;
                }
                let backward_emit = if positions[&gate_gradient] < positions[&up_gradient] {
                    gate_gradient
                } else {
                    up_gradient
                };
                // Explicitly reject early sibling output schedules; never publish
                // an unready paired output or invent an ordering edge.
                if inputs
                    .iter()
                    .any(|n| positions[n] >= positions[&backward_emit])
                    || inputs[..2]
                        .iter()
                        .any(|n| positions[n] >= positions[&hidden])
                {
                    continue;
                }
                let saved_f32_checkpoints: HashSet<_> = forward_members
                    .iter()
                    .copied()
                    .filter(|n| {
                        dtypes.get(n) == Some(&ValueDType::F32)
                            && !matches!(g[*n], TensorGraphNode::Cast { .. })
                            && g.graph
                                .neighbors_directed(*n, Direction::Outgoing)
                                .any(|c| members.contains(&c) && !forward_members.contains(&c))
                    })
                    .collect();
                let input_map = HashMap::from([(inputs[0], 0), (inputs[1], 1), (inputs[2], 2)]);
                let f = ScalarProgram::extract(g, &[hidden], &input_map)?;
                let b = ScalarProgram::extract(g, &[gate_gradient, up_gradient], &input_map)?;
                claimed.extend(members.iter().copied());
                found.push(SwiGluPair {
                    shape: shape.clone(),
                    inputs,
                    storage_roots,
                    hidden,
                    gate_gradient,
                    up_gradient,
                    forward_emit: hidden,
                    backward_emit,
                    members,
                    saved_f32_checkpoints,
                    forward: f,
                    backward: b,
                });
                break 'candidate;
            }
        }
    }
    Ok(found)
}
fn scratch(b: &mut ScalarBuilder, dtype: DType, threads: usize) -> TileVar {
    let var = b.var();
    b.stmts.push(Stmt::AllocTile {
        var,
        space: MemorySpace::Shared,
        layout: TileLayout::Shared(SharedLayout::row_major(threads)),
        dtype,
        rows: 1,
        cols: threads,
    });
    var
}
fn checkpoint(b: &mut ScalarBuilder, slot: TileVar, value: TileVar) -> TileVar {
    let lane = Expr::ThreadIdx(Dim::X);
    b.store(ScalarMemory::Shared(slot), lane.clone(), value);
    b.load(ScalarMemory::Shared(slot), lane)
}
impl ScalarProgram {
    fn lower(
        &self,
        name: &str,
        shape: &[usize],
        sources: &[RegionInput],
        checkpoints: &HashSet<NodeIndex>,
        threads: usize,
    ) -> Result<TileIR> {
        ensure!(
            threads > 0 && threads <= 1024,
            "mixed SwiGLU block size invalid"
        );
        let extent = shape
            .iter()
            .try_fold(1_usize, |a, &n| a.checked_mul(n))
            .context("mixed SwiGLU extent overflow")?;
        let mut b = ScalarBuilder::default();
        b.stmts.push(Stmt::BoundsCheck { extent });
        let bf16 = scratch(&mut b, DType::BF16, threads);
        let f32_slot = scratch(&mut b, DType::F32, threads);
        let index = Expr::BlockIdx(Dim::X) * threads + Expr::ThreadIdx(Dim::X);
        let mut values: Vec<TileVar> = Vec::new();
        let output_set: HashSet<_> = self.outputs.iter().copied().collect();
        let consumed: HashSet<_> = self
            .ops
            .iter()
            .flat_map(|op| op.inputs.iter().copied())
            .collect();
        for (id, op) in self.ops.iter().enumerate() {
            let arg = |i| values[op.inputs[i]];
            let mut value = match op.kind {
                Kind::Input(i) => {
                    let input = sources.get(i).context("mixed SwiGLU source missing")?;
                    ensure!(
                        input.tensor.dtype == DType::BF16,
                        "mixed SwiGLU input must be BF16 storage"
                    );
                    ensure!(
                        input.tensor
                            == VirtualTensor::identity(
                                input.tensor.source,
                                input.source_shape.clone(),
                                DType::BF16
                            )
                            && input
                                .source_shape
                                .iter()
                                .try_fold(1_usize, |n, &d| n.checked_mul(d))
                                == Some(extent),
                        "mixed SwiGLU requires a proven contiguous physical input"
                    );
                    b.load(ScalarMemory::Global(format!("input_{i}")), index.clone())
                }
                Kind::Constant(bits) => b.value(ScalarExpr::Constant(f32::from_bits(bits))),
                Kind::Unary(op) => b.value(ScalarExpr::unary(op, arg(0))),
                Kind::Binary(op) => b.value(ScalarExpr::binary(
                    ScalarBinaryOp::Arithmetic(op),
                    arg(0),
                    arg(1),
                )),
                Kind::Gt => {
                    let dest = b.value(ScalarExpr::Constant(0.));
                    b.stmts.push(Stmt::Gt {
                        dest,
                        a: arg(0),
                        b: arg(1),
                    });
                    dest
                }
                Kind::Mask => {
                    let dest = b.value(ScalarExpr::Constant(0.));
                    b.stmts.push(Stmt::Mask {
                        dest,
                        values: arg(0),
                        condition: arg(1),
                    });
                    dest
                }
                Kind::RoundBF16 => arg(0),
            };
            if checkpoints.contains(&op.node) {
                value = checkpoint(&mut b, f32_slot, value);
            }
            if op.dtype == ValueDType::BF16
                && !matches!(op.kind, Kind::Input(_))
                && (!output_set.contains(&id) || consumed.contains(&id))
            {
                value = checkpoint(&mut b, bf16, value);
            }
            values.push(value);
        }
        for (i, &output) in self.outputs.iter().enumerate() {
            b.store(
                ScalarMemory::Global(format!("output_{i}")),
                index.clone(),
                values[output],
            );
        }
        let params = sources
            .iter()
            .enumerate()
            .map(|(i, s)| KernelParam {
                name: format!("input_{i}"),
                dtype: s.tensor.dtype,
                is_input: true,
            })
            .chain((0..self.outputs.len()).map(|i| KernelParam {
                name: format!("output_{i}"),
                dtype: DType::BF16,
                is_input: false,
            }))
            .collect();
        Ok(TileIR {
            kernel_name: name.into(),
            params,
            body: Block { stmts: b.stmts },
            shared_mem_bytes: threads * 6,
        })
    }
}
impl SwiGluPair {
    pub fn lower_forward(&self, sources: &[RegionInput], threads: usize) -> Result<TileIR> {
        ensure!(
            sources.len() == 2,
            "mixed SwiGLU forward source count must be two"
        );
        self.forward.lower(
            "mixed_swiglu_forward",
            &self.shape,
            sources,
            &self.saved_f32_checkpoints,
            threads,
        )
    }
    pub fn lower_backward(&self, sources: &[RegionInput], threads: usize) -> Result<TileIR> {
        ensure!(
            sources.len() == 3,
            "mixed SwiGLU source count must be three"
        );
        self.backward.lower(
            "mixed_swiglu_backward",
            &self.shape,
            sources,
            &self.saved_f32_checkpoints,
            threads,
        )
    }
}
#[cfg(test)]
impl ScalarProgram {
    fn evaluate(&self, inputs: &[f32]) -> Vec<f32> {
        let mut values = Vec::<f32>::new();
        for op in &self.ops {
            let a = |i| values[op.inputs[i]];
            let raw = match op.kind {
                Kind::Input(i) => inputs[i],
                Kind::Constant(bits) => f32::from_bits(bits),
                Kind::Unary(op) => match op {
                    UnaryOp::Neg => -a(0),
                    UnaryOp::Exp => a(0).exp(),
                    UnaryOp::Log => a(0).ln(),
                    UnaryOp::Relu => a(0).max(0.),
                },
                Kind::Binary(op) => match op {
                    BinaryOp::Add => a(0) + a(1),
                    BinaryOp::Sub => a(0) - a(1),
                    BinaryOp::Mul => a(0) * a(1),
                    BinaryOp::Div => a(0) / a(1),
                },
                Kind::Gt => {
                    if a(0) > a(1) {
                        1.
                    } else {
                        0.
                    }
                }
                Kind::Mask => {
                    if a(1) != 0. {
                        a(0)
                    } else {
                        0.
                    }
                }
                Kind::RoundBF16 => a(0),
            };
            values.push(
                if op.dtype == ValueDType::BF16 && !matches!(op.kind, Kind::Input(_)) {
                    half::bf16::from_f32(raw).to_f32()
                } else {
                    raw
                },
            );
        }
        self.outputs.iter().map(|&i| values[i]).collect()
    }
}

#[cfg(test)]
mod tests;

/// Restrict the initial planner to producers which allocate ordinary contiguous
/// outputs. A cast can preserve/reorder a physical view under other options, so
/// casts and unknown producers deliberately fall back before skipping any DAG.
pub(super) fn planned_contiguous<G>(
    region: &SwiGluPair,
    graph: &TensorGraph<f32, G>,
    dtypes: &HashMap<NodeIndex, ValueDType>,
    options: MixedOptions,
) -> bool {
    region.storage_roots.iter().all(|&root| match graph[root] {
        TensorGraphNode::Binary { .. } | TensorGraphNode::Linear { .. } => true,
        TensorGraphNode::MatMul { .. } => {
            !options.transposed_bf16_outputs
                || !super::transpose_output::eligible(graph, root, dtypes)
        }
        _ => false,
    })
}

impl MixedExecutor {
    pub(super) fn swiglu_pair(&mut self, region: &SwiGluPair, backward: bool) -> Result<()> {
        const THREADS: usize = 256;
        let inputs = contiguous_values(region, &self.values, backward)
            .context("SwiGLU physical descriptor violated its contiguous producer proof")?;
        let sources: Vec<_> = inputs
            .iter()
            .enumerate()
            .map(|(i, value)| RegionInput {
                node: region.storage_roots[i],
                value: crate::tile::RegionValue(i),
                tensor: value.view.clone(),
                source_shape: value.source_shape.clone(),
            })
            .collect();
        let ir = if backward {
            region.lower_backward(&sources, THREADS)?
        } else {
            region.lower_forward(&sources, THREADS)?
        };
        let output_nodes = if backward {
            vec![region.gate_gradient, region.up_gradient]
        } else {
            vec![region.hidden]
        };
        let mut outputs = Vec::new();
        for &node in &output_nodes {
            outputs.push(self.alloc(node, region.shape.clone(), DType::BF16)?.1);
        }
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.region_kernels = vec![crate::tile::graph::TileRegionKernel { region_id: 0, ir }];
        tiles.optimize_indices()?;
        let graph =
            PtxGraph::from(tiles).with_target(PtxTarget::from_context(self.stream.context())?);
        let name = graph
            .region_kernel_name(0)
            .context("SwiGLU pair kernel absent")?;
        let module = self
            .stream
            .context()
            .load_module(Ptx::from_src(graph.to_ptx()))?;
        let mut pointers: Vec<_> = inputs.iter().map(|v| v.binding.pointer).collect();
        pointers.extend(outputs.iter().map(|v| v.binding.pointer));
        let count = region
            .shape
            .iter()
            .try_fold(1_usize, |n, &d| n.checked_mul(d))
            .context("SwiGLU launch extent overflow")?;
        self.kernels.push(Kernel {
            owner: None,
            label: format!(
                "SwiGLU pair {} BF16 shape={:?}",
                if backward { "backward" } else { "forward" },
                region.shape
            ),
            function: module.load_function(name)?,
            config: LaunchConfig {
                grid_dim: (u32::try_from(count)?.div_ceil(THREADS as u32), 1, 1),
                block_dim: (THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            pointers,
            zero_before: None,
        });
        for (node, output) in output_nodes.into_iter().zip(outputs) {
            self.values.insert(node, output);
        }
        if backward {
            self.info.swiglu_pair_backward_regions += 1;
        } else {
            self.info.swiglu_pair_forward_regions += 1;
        }
        Ok(())
    }
}
