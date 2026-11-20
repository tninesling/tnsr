//! Closed causal attention normalization, retaining both probability storage types.
//!
//! This is a backend schedule, not a different softmax formula. Each row uses
//! the ordinary deterministic 128-thread tree, including its initial identities.
//! Removing forward nodes can change surrounding automatic f32 fusion: compact
//! backward may contract P*dP and its accumulator into FFMA instead of storing
//! rounded products first. Forward probability outputs remain exact; full-model
//! gradient/weight qualification gates still apply to this opt-in experiment.
use super::*;
use crate::tensor::{BinaryOp, ReduceOp, UnaryOp};
use crate::tile::{
    Block, Dim, Expr, KernelParam, MemorySpace, ScalarBinaryOp, ScalarBuilder, ScalarExpr,
    ScalarMemory, ScalarPredicate, SharedLayout, Stmt, TileIR, TileLayout, TileVar,
};
use petgraph::Direction;
use std::collections::HashSet;

const COLUMNS: usize = 512;
const THREADS: usize = 128;

pub(super) struct AttentionSoftmax {
    pub output: NodeIndex,
    pub narrowed: NodeIndex,
    pub qk: NodeIndex,
    pub mask: NodeIndex,
    pub scale: f32,
    pub shape: Vec<usize>,
    pub members: HashSet<NodeIndex>,
}
fn binary<G>(g: &TensorGraph<f32, G>, n: NodeIndex, op: BinaryOp) -> Option<[NodeIndex; 2]> {
    matches!(g[n], TensorGraphNode::Binary {op: x, ..} if x == op)
        .then(|| g.inputs(n).try_into().ok())
        .flatten()
}
fn single<G>(g: &TensorGraph<f32, G>, n: NodeIndex) -> Option<NodeIndex> {
    let [input] = g.inputs(n).try_into().ok()?;
    Some(input)
}
fn broadcast<G>(g: &TensorGraph<f32, G>, n: NodeIndex) -> Option<NodeIndex> {
    if !matches!(g[n], TensorGraphNode::BroadcastAxis { axis: 3, .. }) {
        return None;
    }
    single(g, n)
}

// TensorExpr lowering expands a scalar into explicit reshape/broadcast views.
// Accept only that proven constant replication, never arbitrary operand maps.
fn scalar_source<G>(
    g: &TensorGraph<f32, G>,
    mut n: NodeIndex,
    shape: &[usize],
) -> Option<(NodeIndex, HashSet<NodeIndex>)> {
    if g[n].shape() != shape {
        return None;
    }
    let mut members = HashSet::new();
    loop {
        match &g[n] {
            TensorGraphNode::BroadcastAxis {
                axis,
                shape: output,
            } => {
                let input = single(g, n)?;
                let source = g[input].shape();
                if source.len() != 4
                    || output.len() != 4
                    || *axis >= 4
                    || source[*axis] != 1
                    || source
                        .iter()
                        .enumerate()
                        .any(|(i, &extent)| i != *axis && extent != output[i])
                {
                    return None;
                }
                members.insert(n);
                n = input;
            }
            TensorGraphNode::Reshape { shape: output } if output == &[1, 1, 1, 1] => {
                let input = single(g, n)?;
                if !g[input].shape().is_empty() {
                    return None;
                }
                members.insert(n);
                return Some((input, members));
            }
            _ => return None,
        }
    }
}

fn mask_source<G>(
    g: &TensorGraph<f32, G>,
    mut n: NodeIndex,
    shape: &[usize],
) -> Option<(NodeIndex, HashSet<NodeIndex>)> {
    if g[n].shape() != shape {
        return None;
    }
    let mut members = HashSet::new();
    while let TensorGraphNode::BroadcastAxis {
        axis,
        shape: output,
    } = &g[n]
    {
        let input = single(g, n)?;
        let source = g[input].shape();
        if *axis >= 2
            || source.len() != 4
            || output.len() != 4
            || source[*axis] != 1
            || source
                .iter()
                .enumerate()
                .any(|(i, &extent)| i != *axis && extent != output[i])
        {
            return None;
        }
        members.insert(n);
        n = input;
    }
    Some((n, members))
}

/// Primitive backward escapes E/denominator and therefore deliberately fails
/// closure. Compact backward escapes only P and remains an ordinary consumer.
pub(super) fn recognize<G>(g: &TensorGraph<f32, G>) -> Vec<AttentionSoftmax> {
    let dtypes = value_dtypes(g);
    let mut found = Vec::new();
    let mut claimed = HashSet::new();
    for output in g.toposort() {
        let Some(region) = recognize_one(g, &dtypes, output) else {
            continue;
        };
        if region.members.iter().any(|n| claimed.contains(n)) {
            continue;
        }
        claimed.extend(region.members.iter().copied());
        found.push(region);
    }
    found
}
fn recognize_one<G>(
    g: &TensorGraph<f32, G>,
    dtypes: &HashMap<NodeIndex, ValueDType>,
    output: NodeIndex,
) -> Option<AttentionSoftmax> {
    let [exponential, denominator] = binary(g, output, BinaryOp::Div)?;
    let shape = g[output].shape().clone();
    if shape.len() != 4 || shape.contains(&0) || shape[2..] != [COLUMNS, COLUMNS] {
        return None;
    }
    let sum = broadcast(g, denominator)?;
    if !matches!(
        g[sum],
        TensorGraphNode::ReduceAxis {
            op: ReduceOp::Sum,
            axis: 3,
            ..
        }
    ) || single(g, sum)? != exponential
    {
        return None;
    }
    if !matches!(
        g[exponential],
        TensorGraphNode::Unary {
            op: UnaryOp::Exp,
            ..
        }
    ) {
        return None;
    }
    let shift = single(g, exponential)?;
    let [score, maximum_broadcast] = binary(g, shift, BinaryOp::Sub)?;
    let maximum = broadcast(g, maximum_broadcast)?;
    if !matches!(
        g[maximum],
        TensorGraphNode::ReduceAxis {
            op: ReduceOp::Max,
            axis: 3,
            ..
        }
    ) || single(g, maximum)? != score
    {
        return None;
    }
    let [scaled, mask_broadcast] = binary(g, score, BinaryOp::Add)?;
    let (mask, mask_members) = mask_source(g, mask_broadcast, &shape)?;
    let [promotion, scale_broadcast] = binary(g, scaled, BinaryOp::Mul)?;
    // Scale views remain ordinary graph values: compact backward legitimately
    // consumes the same broadcast coefficient. Validate their access semantics
    // but do not elide them or require their downstream fanout to be closed.
    let (scale_node, _scalar_members) = scalar_source(g, scale_broadcast, &shape)?;
    let TensorGraphNode::Constant {
        data,
        shape: scalar_shape,
    } = &g[scale_node]
    else {
        return None;
    };
    if !scalar_shape.is_empty() || data.len() != 1 || data[0].to_bits() != 0.125_f32.to_bits() {
        return None;
    }
    let scale = data[0];
    let TensorGraphNode::Constant {
        data,
        shape: mask_shape,
    } = &g[mask]
    else {
        return None;
    };
    if mask_shape != &[1, 1, COLUMNS, COLUMNS]
        || data.len() != COLUMNS * COLUMNS
        || data.iter().enumerate().any(|(i, x)| {
            x.to_bits()
                != if i % COLUMNS > i / COLUMNS {
                    f32::NEG_INFINITY.to_bits()
                } else {
                    0_f32.to_bits()
                }
        })
    {
        return None;
    }
    if !matches!(
        g[promotion],
        TensorGraphNode::Cast {
            dtype: ValueDType::F32,
            ..
        }
    ) {
        return None;
    }
    let qk = single(g, promotion)?;
    if !matches!(g[qk], TensorGraphNode::MatMul { .. })
        || dtypes.get(&qk) != Some(&ValueDType::BF16)
        || g[qk].shape() != &shape
    {
        return None;
    }
    let operands = g.inputs(qk);
    if operands.len() != 2
        || g[operands[0]].shape() != &[shape[0], shape[1], COLUMNS, 64]
        || g[operands[1]].shape() != &[shape[0], shape[1], 64, COLUMNS]
    {
        return None;
    }
    let casts: Vec<_> = g
        .graph
        .neighbors_directed(output, Direction::Outgoing)
        .filter(|&n| {
            matches!(
                g[n],
                TensorGraphNode::Cast {
                    dtype: ValueDType::BF16,
                    ..
                }
            ) && g[n].shape() == &shape
        })
        .collect();
    if casts.len() != 1 {
        return None;
    }
    let narrowed = casts[0];
    let mut members: HashSet<_> = [
        promotion,
        scaled,
        score,
        maximum,
        maximum_broadcast,
        shift,
        exponential,
        sum,
        denominator,
        output,
        narrowed,
    ]
    .into_iter()
    .collect();
    members.extend(mask_members.iter().copied());
    let mut reduced = shape.clone();
    reduced[3] = 1;
    for &n in &members {
        let expected = if n == maximum || n == sum {
            &reduced
        } else {
            &shape
        };
        if (!mask_members.contains(&n) && g[n].shape() != expected)
            || dtypes.get(&n).copied().unwrap_or(ValueDType::F32)
                != if n == narrowed {
                    ValueDType::BF16
                } else {
                    ValueDType::F32
                }
        {
            return None;
        }
        if n != output
            && n != narrowed
            && g.graph
                .neighbors_directed(n, Direction::Outgoing)
                .any(|consumer| !members.contains(&consumer))
        {
            return None;
        }
    }
    Some(AttentionSoftmax {
        output,
        narrowed,
        qk,
        mask,
        scale,
        shape,
        members,
    })
}
fn arithmetic(op: BinaryOp, a: impl Into<ScalarExpr>, b: impl Into<ScalarExpr>) -> ScalarExpr {
    ScalarExpr::binary(ScalarBinaryOp::Arithmetic(op), a, b)
}
fn tree(
    b: &mut ScalarBuilder,
    shared: TileVar,
    mut accumulator: TileVar,
    max: bool,
) -> Result<TileVar> {
    let lane = Expr::ThreadIdx(Dim::X);
    b.store(ScalarMemory::Shared(shared), lane.clone(), accumulator);
    b.barrier();
    for stride in [64, 32, 16, 8, 4, 2, 1] {
        b.when(
            ScalarPredicate::IndexLt {
                lhs: lane.clone(),
                rhs: Expr::Const(stride),
            },
            |b| {
                let peer = b.load(
                    ScalarMemory::Shared(shared),
                    lane.clone() + Expr::Const(stride),
                );
                b.set(
                    accumulator,
                    ScalarExpr::binary(
                        if max {
                            ScalarBinaryOp::Max
                        } else {
                            ScalarBinaryOp::Arithmetic(BinaryOp::Add)
                        },
                        accumulator,
                        peer,
                    ),
                );
                b.store(ScalarMemory::Shared(shared), lane.clone(), accumulator);
                Ok(())
            },
        )?;
        b.barrier();
    }
    accumulator = b.load(ScalarMemory::Shared(shared), Expr::Const(0));
    // Every lane must finish reading before the next reduction overwrites scratch.
    b.barrier();
    Ok(accumulator)
}
pub(super) fn row_ir(scale: f32) -> Result<TileIR> {
    let mut b = ScalarBuilder::default();
    let shared = b.var();
    b.stmts.push(Stmt::AllocTile {
        var: shared,
        space: MemorySpace::Shared,
        layout: TileLayout::Shared(SharedLayout::row_major(THREADS)),
        dtype: DType::F32,
        rows: 1,
        cols: THREADS,
    });
    let lane = Expr::ThreadIdx(Dim::X);
    let row = Expr::BlockIdx(Dim::X);
    let mut scores = Vec::new();
    let maximum = b.value(ScalarExpr::Constant(f32::NEG_INFINITY));
    for part in 0..4 {
        let col = lane.clone() + Expr::Const((part * THREADS) as i64);
        let qk = b.load(
            ScalarMemory::Global("qk".into()),
            row.clone() * COLUMNS + col.clone(),
        );
        let mask = b.load(
            ScalarMemory::Global("mask".into()),
            Expr::Mod(Box::new(row.clone()), COLUMNS) * COLUMNS + col,
        );
        let scaled = b.value(arithmetic(BinaryOp::Mul, qk, ScalarExpr::Constant(scale)));
        let score = b.value(arithmetic(BinaryOp::Add, scaled, mask));
        b.set(
            maximum,
            ScalarExpr::binary(ScalarBinaryOp::Max, maximum, score),
        );
        scores.push(score);
    }
    let maximum = tree(&mut b, shared, maximum, true)?;
    let denominator = b.value(ScalarExpr::Constant(0.));
    let mut exponentials = Vec::new();
    for score in scores {
        let shifted = b.value(arithmetic(BinaryOp::Sub, score, maximum));
        let exp = b.value(ScalarExpr::unary(UnaryOp::Exp, shifted));
        b.set(denominator, arithmetic(BinaryOp::Add, denominator, exp));
        exponentials.push(exp);
    }
    let denominator = tree(&mut b, shared, denominator, false)?;
    for (part, exp) in exponentials.into_iter().enumerate() {
        let index = row.clone() * COLUMNS + lane.clone() + Expr::Const((part * THREADS) as i64);
        let p = b.value(arithmetic(BinaryOp::Div, exp, denominator));
        b.store(ScalarMemory::Global("probability".into()), index.clone(), p);
        b.store(ScalarMemory::Global("narrowed".into()), index, p);
    }
    Ok(TileIR {
        kernel_name: "attention_softmax_512".into(),
        params: vec![
            KernelParam {
                name: "qk".into(),
                dtype: DType::BF16,
                is_input: true,
            },
            KernelParam {
                name: "mask".into(),
                dtype: DType::F32,
                is_input: true,
            },
            KernelParam {
                name: "probability".into(),
                dtype: DType::F32,
                is_input: false,
            },
            KernelParam {
                name: "narrowed".into(),
                dtype: DType::BF16,
                is_input: false,
            },
        ],
        body: Block { stmts: b.stmts },
        shared_mem_bytes: THREADS * 4,
    })
}
impl MixedExecutor {
    pub(super) fn attention_softmax(&mut self, region: &AttentionSoftmax) -> Result<()> {
        let qk = self
            .values
            .get(&region.qk)
            .context("attention QK binding missing")?
            .clone();
        let mask = self
            .values
            .get(&region.mask)
            .context("attention mask binding missing")?
            .clone();
        for (value, shape, dtype) in [
            (&qk, &region.shape, DType::BF16),
            (&mask, &vec![1, 1, COLUMNS, COLUMNS], DType::F32),
        ] {
            ensure!(
                value.binding.dtype == dtype
                    && value.source_shape == *shape
                    && value.view.shape == *shape
                    && value.view.access == crate::tile::IndexMap::identity(shape.len())
                    && value.view.predicate.is_none()
                    && value.view.out_of_bounds.is_none()
                    && value.chain.is_empty(),
                "attention softmax requires contiguous identity operands"
            );
        }
        let (_, probability) = self.alloc(region.output, region.shape.clone(), DType::F32)?;
        let (_, narrowed) = self.alloc(region.narrowed, region.shape.clone(), DType::BF16)?;
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.region_kernels = vec![crate::tile::graph::TileRegionKernel {
            region_id: 0,
            ir: row_ir(region.scale)?,
        }];
        tiles.optimize_indices()?;
        let graph =
            PtxGraph::from(tiles).with_target(PtxTarget::from_context(self.stream.context())?);
        let name = graph
            .region_kernel_name(0)
            .context("attention softmax kernel absent")?;
        let module = self
            .stream
            .context()
            .load_module(Ptx::from_src(graph.to_ptx()))?;
        self.kernels.push(Kernel {
            owner: Some(region.output),
            label: format!("attention softmax fused F32+BF16 shape={:?}", region.shape),
            function: module.load_function(name)?,
            config: LaunchConfig {
                grid_dim: (
                    u32::try_from(region.shape.iter().product::<usize>() / COLUMNS)?,
                    1,
                    1,
                ),
                block_dim: (THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            pointers: vec![
                qk.binding.pointer,
                mask.binding.pointer,
                probability.binding.pointer,
                narrowed.binding.pointer,
            ],
            zero_before: None,
        });
        self.values.insert(region.output, probability);
        self.values.insert(region.narrowed, narrowed);
        self.info.attention_softmax_regions += 1;
        Ok(())
    }
}
#[cfg(test)]
mod tests;
