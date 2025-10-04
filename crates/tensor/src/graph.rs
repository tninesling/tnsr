use crate::{BinaryOp, ReduceOp, UnaryOp};
use itertools::Itertools;
use petgraph::{
    graph::{Graph, NodeIndex},
    visit::EdgeRef,
};
use std::{ops::Index, sync::Arc};

pub enum TensorGraphNode<D> {
    Constant { data: Arc<Vec<D>> },
    Input { name: &'static str },
    Neg,
    Exp,
    Log,
    Relu,
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Reduce { op: ReduceOp, axis: usize },
}

impl<D> From<UnaryOp> for TensorGraphNode<D> {
    fn from(op: UnaryOp) -> Self {
        match op {
            UnaryOp::Neg => TensorGraphNode::Neg,
            UnaryOp::Exp => TensorGraphNode::Exp,
            UnaryOp::Log => TensorGraphNode::Log,
            UnaryOp::Relu => TensorGraphNode::Relu,
        }
    }
}

impl<D> From<BinaryOp> for TensorGraphNode<D> {
    fn from(op: BinaryOp) -> Self {
        match op {
            BinaryOp::Add => TensorGraphNode::Add,
            BinaryOp::Sub => TensorGraphNode::Sub,
            BinaryOp::Mul => TensorGraphNode::Mul,
            BinaryOp::Div => TensorGraphNode::Div,
        }
    }
}

pub struct TensorGraph<D> {
    pub graph: Graph<TensorGraphNode<D>, usize>,
}

impl<D> TensorGraph<D> {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
        }
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
