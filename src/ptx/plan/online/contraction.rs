//! Canonical indexed multiply/sum patterns, independent of normalization.
use super::*;
use crate::tile::{IndexExpr, IndexMap, IndexedOperand, IndexedSource, IndexedSum};

pub(super) struct Contraction {
    pub operands: [NodeIndex; 2],
    pub accesses: [IndexMap; 2],
    pub extent: usize,
    pub members: Vec<NodeIndex>,
}
impl Contraction {
    pub fn bind(&self, lhs: IndexedSource, rhs: IndexedSource) -> IndexedSum {
        IndexedSum {
            extent: self.extent,
            lhs: IndexedOperand {
                source: lhs,
                access: self.accesses[0].clone(),
            },
            rhs: IndexedOperand {
                source: rhs,
                access: self.accesses[1].clone(),
            },
        }
    }
}
pub(super) fn contraction<D: TileDType, G>(
    graph: &TensorGraph<D, G>,
    output: NodeIndex,
) -> Option<Contraction> {
    let shape = graph[output].shape();
    let rank = shape.len();
    if rank == 0 || shape.contains(&0) {
        return None;
    }
    if matches!(graph[output], TensorGraphNode::MatMul { .. }) {
        let operands: [NodeIndex; 2] = graph.inputs(output).try_into().ok()?;
        let lhs = graph[operands[0]].shape();
        let rhs = graph[operands[1]].shape();
        if rank < 2
            || lhs.len() != rank
            || rhs.len() != rank
            || lhs[..rank - 2] != shape[..rank - 2]
            || rhs[..rank - 2] != shape[..rank - 2]
            || lhs[rank - 1] != rhs[rank - 2]
            || lhs[rank - 1] == 0
        {
            return None;
        }
        let mut a = IndexMap::identity(rank);
        a.results[rank - 1] = IndexExpr::IterDim(rank);
        let mut b = IndexMap::identity(rank);
        b.results[rank - 2] = IndexExpr::IterDim(rank);
        return Some(Contraction {
            operands,
            accesses: [a, b],
            extent: lhs[rank - 1],
            members: vec![output],
        });
    }
    let mut node = output;
    let mut members = vec![output];
    let mut access = IndexMap::identity(rank);
    // Reshapes around keep-dimension reductions only change output indexing.
    while matches!(
        graph[node],
        TensorGraphNode::Reshape { .. } | TensorGraphNode::Flatten { .. }
    ) {
        let [input]: [NodeIndex; 1] = graph.inputs(node).try_into().ok()?;
        access = IndexMap::reshape(graph[input].shape(), graph[node].shape())
            .ok()?
            .compose(&access)
            .ok()?;
        node = input;
        members.push(node);
    }
    let TensorGraphNode::ReduceAxis {
        op: ReduceOp::Sum,
        axis,
        ..
    } = graph[node]
    else {
        return None;
    };
    let [product]: [NodeIndex; 1] = graph.inputs(node).try_into().ok()?;
    let operands = binary(graph, product, BinaryOp::Mul)?;
    let extent = graph[product].shape()[axis];
    if extent == 0 {
        return None;
    }
    access.results[axis] = IndexExpr::IterDim(rank);
    let mut domain = shape.clone();
    domain.push(extent);
    access = access.normalize_in_domain(&domain).ok()?;
    members.push(product);
    Some(Contraction {
        operands,
        accesses: [access.clone(), access],
        extent,
        members,
    })
}

/// Trace pure input views while retaining their complete access map. This lets
/// consumer rules see the value behind broadcast/reshape, rather than syntax.
pub(super) fn view_root<D: TileDType, G>(
    graph: &TensorGraph<D, G>,
    node: NodeIndex,
) -> Option<(VirtualTensor, Vec<NodeIndex>)> {
    let operands = graph.inputs(node);
    let supported = matches!(
        graph[node],
        TensorGraphNode::Reshape { .. }
            | TensorGraphNode::Flatten { .. }
            | TensorGraphNode::Permute { .. }
            | TensorGraphNode::Transpose { .. }
            | TensorGraphNode::BroadcastAxis { .. }
    );
    if !supported {
        return Some((
            VirtualTensor::identity(node, graph[node].shape().clone(), D::TILE_DTYPE),
            Vec::new(),
        ));
    }
    let [input]: [NodeIndex; 1] = operands.try_into().ok()?;
    let (tensor, mut members) = view_root(graph, input)?;
    let tensor = match &graph[node] {
        TensorGraphNode::Reshape { shape } | TensorGraphNode::Flatten { shape } => {
            tensor.reshape(shape.clone()).ok()?
        }
        TensorGraphNode::Permute { axes, .. } => tensor.permute(axes).ok()?,
        TensorGraphNode::Transpose { shape } => {
            let mut axes: Vec<_> = (0..shape.len()).collect();
            axes.swap(shape.len() - 2, shape.len() - 1);
            tensor.permute(&axes).ok()?
        }
        TensorGraphNode::BroadcastAxis { axis, shape } => {
            tensor.broadcast_axis(*axis, shape[*axis]).ok()?
        }
        _ => return None,
    };
    members.push(node);
    Some((tensor, members))
}
