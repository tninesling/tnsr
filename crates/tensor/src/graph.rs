use std::collections::HashMap;
use std::ops::Index;
use std::sync::Arc;
use std::sync::Mutex;

use itertools::Itertools;
use petgraph::graph::Graph;
pub use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;

use crate::BinaryOp;
use crate::ReduceOp;
use crate::TensorExpr;
use crate::UnaryOp;

pub enum TensorGraphNode<D> {
    Constant { data: Arc<Vec<D>> },
    Input { name: &'static str },
    Parameter { id: usize, data: Arc<Mutex<Vec<D>>> },
    Unary { op: UnaryOp },
    Binary { op: BinaryOp },
    MatMul,
    BroadcastAxis { axis: usize },
    ReduceAxis { op: ReduceOp, axis: usize },
}

impl<D> TensorGraphNode<D> {
    pub fn name(&self) -> &'static str {
        match self {
            TensorGraphNode::Constant { .. } => "Constant",
            TensorGraphNode::Input { .. } => "Input",
            TensorGraphNode::Parameter { .. } => "Parameter",
            TensorGraphNode::Unary { op } => match op {
                UnaryOp::Neg => "Neg",
                UnaryOp::Exp => "Exp",
                UnaryOp::Log => "Log",
                UnaryOp::Relu => "Relu",
            },
            TensorGraphNode::Binary { op } => match op {
                BinaryOp::Add => "Add",
                BinaryOp::Sub => "Sub",
                BinaryOp::Mul => "Mul",
                BinaryOp::Div => "Div",
            },
            TensorGraphNode::MatMul => "MatMul",
            TensorGraphNode::BroadcastAxis { .. } => "BroadcastAxis",
            TensorGraphNode::ReduceAxis { op, .. } => match op {
                ReduceOp::Sum => "ReduceAxisSum",
                ReduceOp::Max => "ReduceAxisMax",
                ReduceOp::Mean => "ReduceAxisMean",
            },
        }
    }
}

impl<D> From<UnaryOp> for TensorGraphNode<D> {
    fn from(op: UnaryOp) -> Self {
        TensorGraphNode::Unary { op }
    }
}

impl<D> From<BinaryOp> for TensorGraphNode<D> {
    fn from(op: BinaryOp) -> Self {
        TensorGraphNode::Binary { op }
    }
}

pub struct TensorGraph<D> {
    pub graph: Graph<TensorGraphNode<D>, usize>,
    pub shapes: HashMap<NodeIndex, crate::Shape>,
}

impl<D> Default for TensorGraph<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D> TensorGraph<D> {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            shapes: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.graph.node_count()
    }

    pub fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
    }

    pub fn toposort(&self) -> Vec<NodeIndex> {
        petgraph::algo::toposort(&self.graph, None).unwrap()
    }

    pub fn inputs(&self, idx: NodeIndex) -> Vec<NodeIndex> {
        // We preserve the input order for non-commutative operations
        self.graph
            .edges_directed(idx, petgraph::Direction::Incoming)
            .sorted_by_key(|e| e.weight())
            .map(|e| e.source())
            .collect()
    }
}

impl<D> Index<NodeIndex> for TensorGraph<D> {
    type Output = TensorGraphNode<D>;

    fn index(&self, index: NodeIndex) -> &Self::Output {
        &self.graph[index]
    }
}

impl From<TensorExpr<f32>> for TensorGraph<f32> {
    fn from(expr: TensorExpr<f32>) -> Self {
        let mut graph = TensorGraph::new();
        let _ = expr.lower_to_graph(&mut graph);
        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Constant;
    use crate::ptx;
    use crate::tile;

    #[test]
    #[ignore = "PTX lowering not yet implemented"]
    fn lowers_add_to_ptx() {
        let a = Constant::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]);
        let b = Constant::new(vec![10.0f32, 20.0, 30.0, 40.0], vec![4]);
        let c = a + b;
        println!("Expr: {c:?}");

        let tensor_graph: TensorGraph<f32> = c.into();
        let tile_graph: tile::TileGraph = tensor_graph.into();
        let ptx_graph: ptx::PtxGraph = tile_graph.into();
        let ptx_module: ptx::Module = ptx_graph.into();

        insta::assert_snapshot!(ptx_module);
    }
}
