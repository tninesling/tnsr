//! Closed S512 compact attention VJP with the existing BF16 storage boundaries.
//!
//! The dot uses the existing fused-producer policy: ascending four terms per
//! lane, then the deterministic 128-thread descending shared tree. PTX may
//! contract each product/accumulate to FFMA. A baseline planner that materializes
//! products first has different f32 rounding; the established full-model gates
//! still apply. No mask, zero-probability, or nonfinite shortcut is introduced.
use super::*;
use crate::tensor::{BinaryOp, ReduceOp};
use crate::tile::{
    Block, Dim, Expr, KernelParam, MemorySpace, ScalarBinaryOp, ScalarBuilder, ScalarExpr,
    ScalarMemory, ScalarPredicate, SharedLayout, Stmt, TileIR, TileLayout,
};
use petgraph::Direction;
use std::collections::HashSet;

const COLUMNS: usize = 512;
const THREADS: usize = 128;

pub(super) struct AttentionSoftmaxBackward {
    /// Emit at the original last F32 node, before the skipped BF16 cast.
    pub output: NodeIndex,
    pub narrowed: NodeIndex,
    pub probability: NodeIndex,
    pub incoming: NodeIndex,
    pub shape: Vec<usize>,
    pub members: HashSet<NodeIndex>,
}
fn single<G>(g: &TensorGraph<f32, G>, n: NodeIndex) -> Option<NodeIndex> {
    let [input] = g.inputs(n).try_into().ok()?;
    Some(input)
}
fn binary<G>(g: &TensorGraph<f32, G>, n: NodeIndex, op: BinaryOp) -> Option<[NodeIndex; 2]> {
    matches!(g[n],TensorGraphNode::Binary {op:x,..} if x==op)
        .then(|| g.inputs(n).try_into().ok())
        .flatten()
}
fn identity_casts<G>(
    g: &TensorGraph<f32, G>,
    mut n: NodeIndex,
    shape: &[usize],
    members: &mut HashSet<NodeIndex>,
) -> Option<NodeIndex> {
    while matches!(
        g[n],
        TensorGraphNode::Cast {
            dtype: ValueDType::F32,
            ..
        }
    ) {
        if g[n].shape() != shape {
            return None;
        }
        members.insert(n);
        n = single(g, n)?;
    }
    Some(n)
}
fn scale_replication<G>(g: &TensorGraph<f32, G>, mut n: NodeIndex, shape: &[usize]) -> Option<()> {
    if g[n].shape() != shape {
        return None;
    }
    loop {
        match &g[n] {
            TensorGraphNode::BroadcastAxis {
                axis,
                shape: output,
            } => {
                let input = single(g, n)?;
                let source = g[input].shape();
                if *axis >= 4
                    || source.len() != 4
                    || source[*axis] != 1
                    || output.len() != 4
                    || source
                        .iter()
                        .enumerate()
                        .any(|(i, &extent)| i != *axis && extent != output[i])
                {
                    return None;
                }
                n = input;
            }
            TensorGraphNode::Reshape { shape: output } if output == &[1, 1, 1, 1] => {
                let TensorGraphNode::Constant {
                    shape: source,
                    data,
                } = &g[single(g, n)?]
                else {
                    return None;
                };
                return (source.is_empty()
                    && data.len() == 1
                    && data[0].to_bits() == 0.125_f32.to_bits())
                .then_some(());
            }
            _ => return None,
        }
    }
}

pub(super) fn recognize<G>(g: &TensorGraph<f32, G>) -> Vec<AttentionSoftmaxBackward> {
    let probabilities: HashSet<_> = super::attention_softmax::recognize(g)
        .iter()
        .map(|r| r.output)
        .collect();
    let dtypes = value_dtypes(g);
    let mut claimed = HashSet::new();
    let mut regions = Vec::new();
    for n in g.toposort() {
        let Some(region) = recognize_one(g, &dtypes, &probabilities, n) else {
            continue;
        };
        if region.members.iter().any(|n| claimed.contains(n)) {
            continue;
        }
        claimed.extend(region.members.iter().copied());
        regions.push(region);
    }
    regions
}
fn recognize_one<G>(
    g: &TensorGraph<f32, G>,
    dtypes: &HashMap<NodeIndex, ValueDType>,
    probabilities: &HashSet<NodeIndex>,
    narrowed: NodeIndex,
) -> Option<AttentionSoftmaxBackward> {
    if !matches!(
        g[narrowed],
        TensorGraphNode::Cast {
            dtype: ValueDType::BF16,
            ..
        }
    ) {
        return None;
    }
    let output = single(g, narrowed)?;
    let shape = g[output].shape().clone();
    if shape.len() != 4 || shape.contains(&0) || shape[2..] != [COLUMNS, COLUMNS] {
        return None;
    }
    let mut members = HashSet::from([narrowed, output]);
    let scaled = identity_casts(g, output, &shape, &mut members)?;
    let [derivative, scale] = binary(g, scaled, BinaryOp::Mul)?;
    scale_replication(g, scale, &shape)?;
    members.insert(scaled);
    let derivative = identity_casts(g, derivative, &shape, &mut members)?;
    let [probability, centered] = binary(g, derivative, BinaryOp::Mul)?;
    if !probabilities.contains(&probability) {
        return None;
    }
    let [incoming, broadcast] = binary(g, centered, BinaryOp::Sub)?;
    if !matches!(g[broadcast], TensorGraphNode::BroadcastAxis { axis: 3, .. }) {
        return None;
    }
    let dot = single(g, broadcast)?;
    if !matches!(
        g[dot],
        TensorGraphNode::ReduceAxis {
            axis: 3,
            op: ReduceOp::Sum,
            ..
        }
    ) {
        return None;
    }
    let product = single(g, dot)?;
    if binary(g, product, BinaryOp::Mul)? != [probability, incoming] {
        return None;
    }
    // G is the already-rounded BF16 dP, promoted without arithmetic. Any
    // intervening view/cast that could change its mapping or rounding rejects.
    if !matches!(
        g[incoming],
        TensorGraphNode::Cast {
            dtype: ValueDType::F32,
            ..
        }
    ) || g[incoming].shape() != &shape
    {
        return None;
    }
    let mut dp = single(g, incoming)?;
    while matches!(
        g[dp],
        TensorGraphNode::Cast {
            dtype: ValueDType::BF16,
            ..
        }
    ) {
        if g[dp].shape() != &shape || dtypes.get(&single(g, dp)?) != Some(&ValueDType::BF16) {
            return None;
        }
        dp = single(g, dp)?;
    }
    if !matches!(g[dp], TensorGraphNode::MatMul { .. })
        || g[dp].shape() != &shape
        || dtypes.get(&dp) != Some(&ValueDType::BF16)
    {
        return None;
    }
    members.extend([derivative, centered, broadcast, dot, product]);
    let mut reduced = shape.clone();
    reduced[3] = 1;
    for &n in &members {
        if g[n].shape() != if n == dot { &reduced } else { &shape }
            || dtypes.get(&n)
                != Some(&if n == narrowed {
                    ValueDType::BF16
                } else {
                    ValueDType::F32
                })
        {
            return None;
        }
        if n != narrowed
            && g.graph
                .neighbors_directed(n, Direction::Outgoing)
                .any(|consumer| !members.contains(&consumer))
        {
            return None;
        }
    }
    Some(AttentionSoftmaxBackward {
        output,
        narrowed,
        probability,
        incoming,
        shape,
        members,
    })
}
fn arithmetic(op: BinaryOp, a: impl Into<ScalarExpr>, b: impl Into<ScalarExpr>) -> ScalarExpr {
    ScalarExpr::binary(ScalarBinaryOp::Arithmetic(op), a, b)
}
pub(super) fn row_ir() -> Result<TileIR> {
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
    let dot = b.value(ScalarExpr::Constant(0.));
    let mut pairs = Vec::new();
    for part in 0..4 {
        let index = row.clone() * COLUMNS + lane.clone() + Expr::Const((part * THREADS) as i64);
        let p = b.load(ScalarMemory::Global("probability".into()), index.clone());
        let g = b.load(ScalarMemory::Global("incoming".into()), index);
        let product = b.value(arithmetic(BinaryOp::Mul, p, g));
        b.set(dot, arithmetic(BinaryOp::Add, dot, product));
        pairs.push((p, g));
    }
    b.store(ScalarMemory::Shared(shared), lane.clone(), dot);
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
                b.set(dot, arithmetic(BinaryOp::Add, dot, peer));
                b.store(ScalarMemory::Shared(shared), lane.clone(), dot);
                Ok(())
            },
        )?;
        b.barrier();
    }
    let dot = b.load(ScalarMemory::Shared(shared), Expr::Const(0));
    b.barrier();
    for (part, (p, g)) in pairs.into_iter().enumerate() {
        let centered = b.value(arithmetic(BinaryOp::Sub, g, dot));
        let derivative = b.value(arithmetic(BinaryOp::Mul, p, centered));
        let scaled = b.value(arithmetic(
            BinaryOp::Mul,
            derivative,
            ScalarExpr::Constant(0.125),
        ));
        b.store(
            ScalarMemory::Global("narrowed".into()),
            row.clone() * COLUMNS + lane.clone() + Expr::Const((part * THREADS) as i64),
            scaled,
        );
    }
    Ok(TileIR {
        kernel_name: "attention_softmax_backward_512".into(),
        params: vec![
            KernelParam {
                name: "probability".into(),
                dtype: DType::F32,
                is_input: true,
            },
            KernelParam {
                name: "incoming".into(),
                dtype: DType::BF16,
                is_input: true,
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
    pub(super) fn attention_softmax_backward(
        &mut self,
        region: &AttentionSoftmaxBackward,
    ) -> Result<()> {
        let probability = self
            .values
            .get(&region.probability)
            .context("attention backward probability missing")?
            .clone();
        let incoming = self
            .values
            .get(&region.incoming)
            .context("attention backward incoming gradient missing")?
            .clone();
        for (value, dtype) in [(&probability, DType::F32), (&incoming, DType::BF16)] {
            ensure!(
                value.binding.dtype == dtype
                    && value.source_shape == region.shape
                    && value.view.shape == region.shape
                    && value.view.access == crate::tile::IndexMap::identity(4)
                    && value.view.predicate.is_none()
                    && value.view.out_of_bounds.is_none()
                    && value.chain.is_empty(),
                "attention backward requires identity operands"
            );
        }
        let (_, narrowed) = self.alloc(region.narrowed, region.shape.clone(), DType::BF16)?;
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.region_kernels = vec![crate::tile::graph::TileRegionKernel {
            region_id: 0,
            ir: row_ir()?,
        }];
        tiles.optimize_indices()?;
        let graph =
            PtxGraph::from(tiles).with_target(PtxTarget::from_context(self.stream.context())?);
        let name = graph
            .region_kernel_name(0)
            .context("attention backward kernel absent")?;
        let module = self
            .stream
            .context()
            .load_module(Ptx::from_src(graph.to_ptx()))?;
        self.kernels.push(Kernel {
            owner: Some(region.narrowed),
            label: format!(
                "attention softmax backward fused BF16 shape={:?}",
                region.shape
            ),
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
                probability.binding.pointer,
                incoming.binding.pointer,
                narrowed.binding.pointer,
            ],
            zero_before: None,
        });
        self.values.insert(region.narrowed, narrowed);
        self.info.attention_softmax_backward_regions += 1;
        Ok(())
    }
}
#[cfg(test)]
mod tests;
