use std::sync::{Arc, Mutex};

use petgraph::graph::Graph;

use crate::graph::{TensorGraph, TensorGraphNode};

#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub inputs: Vec<TensorView>,
    pub outputs: Vec<TensorView>,
    pub grid: GridShape,
    pub body: Vec<Op>,
    pub shared_mem_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct TensorView {
    pub shape: Vec<usize>,
    pub space: MemorySpace,
}

#[derive(Clone, Debug)]
pub enum MemorySpace {
    Global,
    Shared,
    Register,
}

#[derive(Clone, Debug)]
pub struct GridShape {
    pub block: usize,
    pub threads: usize,
}

#[derive(Clone, Debug)]
pub enum Op {
    Add {
        lhs: String,
        rhs: String,
        out: String,
        elements_per_thread: usize,
    },
}

pub struct TileGraph {
    pub graph: Graph<TileGraphNode, usize>,
}

pub enum TileGraphNode {
    Constant {
        data: Arc<Vec<f32>>,
    },
    Input {
        name: &'static str,
    },
    Parameter {
        id: usize,
        data: Arc<Mutex<Vec<f32>>>,
    },
    Function {
        func: Function,
    },
}

impl From<TensorGraph<f32>> for TileGraph {
    fn from(tensor_graph: TensorGraph<f32>) -> Self {
        let tile_graph = tensor_graph.graph.map_owned(
            |_idx, n| match n {
                TensorGraphNode::Constant { data } => {
                    TileGraphNode::Constant { data: data.clone() }
                }
                TensorGraphNode::Input { .. } => todo!(),
                TensorGraphNode::Parameter { .. } => todo!(),
                TensorGraphNode::Unary { .. } => todo!(),
                TensorGraphNode::Binary { .. } => todo!(),
                TensorGraphNode::MatMul => todo!(),
                TensorGraphNode::BroadcastAxis { .. } => todo!(),
                TensorGraphNode::ReduceAxis { .. } => todo!(),
            },
            |_idx, e| e,
        );

        Self { graph: tile_graph }
    }
}
