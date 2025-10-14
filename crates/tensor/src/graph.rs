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
use crate::ptx;
use crate::tile;

pub enum TensorGraphNode<D> {
    Constant { data: Arc<Vec<D>> },
    Input { name: &'static str },
    Parameter { id: usize, data: Arc<Mutex<Vec<D>>> },
    Neg,
    Exp,
    Log,
    Relu,
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Broadcast,
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
        func: tile::Function,
    },
}

impl From<TensorGraph<f32>> for TileGraph {
    fn from(tensor_graph: TensorGraph<f32>) -> Self {
        let tile_graph = tensor_graph.graph.map_owned(
            |idx, n| match n {
                TensorGraphNode::Constant { data } => {
                    TileGraphNode::Constant { data: data.clone() }
                }
                TensorGraphNode::Input { name } => todo!(),
                TensorGraphNode::Parameter { id, data } => todo!(),
                TensorGraphNode::Neg => todo!(),
                TensorGraphNode::Exp => todo!(),
                TensorGraphNode::Log => todo!(),
                TensorGraphNode::Relu => todo!(),
                TensorGraphNode::Add => {
                    let shape = tensor_graph.shapes[&idx].clone();
                    let total_elements: usize = shape.iter().product();

                    let threads_per_block = 256;
                    let elements_per_thread = 4;
                    let elements_per_block = threads_per_block * elements_per_thread;
                    let num_blocks = (total_elements + elements_per_block - 1) / elements_per_block;

                    let func = tile::Function {
                        name: format!("add_{}", idx.index()),
                        inputs: vec![
                            tile::TensorView {
                                shape: shape.clone(),
                                space: tile::MemorySpace::Global,
                            },
                            tile::TensorView {
                                shape: shape.clone(),
                                space: tile::MemorySpace::Global,
                            },
                        ],
                        outputs: vec![tile::TensorView {
                            shape,
                            space: tile::MemorySpace::Global,
                        }],
                        grid: tile::GridShape {
                            block: num_blocks,
                            threads: threads_per_block,
                        },
                        body: vec![tile::Op::Add {
                            lhs: "lhs".to_string(),
                            rhs: "rhs".to_string(),
                            out: "out".to_string(),
                            elements_per_thread,
                        }],
                        shared_mem_bytes: 0,
                    };
                    TileGraphNode::Function { func }
                }
                TensorGraphNode::Sub => todo!(),
                TensorGraphNode::Mul => todo!(),
                TensorGraphNode::Div => todo!(),
                TensorGraphNode::MatMul => todo!(),
                TensorGraphNode::Broadcast => todo!(),
                TensorGraphNode::Reduce { op, axis } => todo!(),
            },
            |_idx, e| e,
        );

        Self { graph: tile_graph }
    }
}

struct PtxGraph {
    pub graph: Graph<PtxGraphNode, usize>,
}

#[derive(Clone, Debug)]
pub enum PtxGraphNode {
    Function {
        func: ptx::Function,
    },
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
}

impl PtxGraphNode {
    pub fn as_function(&self) -> Option<&ptx::Function> {
        if let PtxGraphNode::Function { func } = self {
            Some(func)
        } else {
            None
        }
    }
}

impl From<TileGraph> for PtxGraph {
    fn from(tile_graph: TileGraph) -> Self {
        let ptx_graph = tile_graph.graph.map_owned(
            |idx, n| match n {
                TileGraphNode::Function {
                    func:
                        tile::Function {
                            name,
                            inputs,
                            outputs,
                            grid,
                            body,
                            shared_mem_bytes,
                        },
                } => {
                    let ptx_body = body.iter().flat_map(|op| match op {
                        tile::Op::Add {
                            lhs,
                            rhs,
                            out,
                            elements_per_thread,
                        } => {
                            let vec_width = match elements_per_thread {
                                1 => ptx::VecWidth::Scalar,
                                2 => ptx::VecWidth::V2,
                                4 => ptx::VecWidth::V4,
                                _ => panic!("Unsupported elements_per_thread"),
                            };
                            let ty = ptx::Type::F32;

                            let mut insts = Vec::new();

                            // Load lhs
                            insts.push(ptx::Inst::LdGlobal {
                                dst: (0..*elements_per_thread)
                                    .map(|i| ptx::Operand::Reg(format!("%r{}", i), ty))
                                    .collect(),
                                addr: ptx::Operand::Reg(
                                    format!(
                                        "[{} + threadIdx.x * {}]",
                                        lhs,
                                        elements_per_thread * 4
                                    ),
                                    ty,
                                ),
                                ty,
                                vec: vec_width.clone(),
                            });

                            // Load rhs
                            insts.push(ptx::Inst::LdGlobal {
                                dst: (0..*elements_per_thread)
                                    .map(|i| {
                                        ptx::Operand::Reg(
                                            format!("%r{}", i + elements_per_thread),
                                            ty,
                                        )
                                    })
                                    .collect(),
                                addr: ptx::Operand::Reg(
                                    format!(
                                        "[{} + threadIdx.x * {}]",
                                        rhs,
                                        elements_per_thread * 4
                                    ),
                                    ty,
                                ),
                                ty,
                                vec: vec_width.clone(),
                            });

                            // Add
                            for i in 0..*elements_per_thread {
                                insts.push(ptx::Inst::Add {
                                    dst: ptx::Operand::Reg(
                                        format!("%r{}", i + 2 * elements_per_thread),
                                        ty,
                                    ),
                                    a: ptx::Operand::Reg(format!("r{}", i), ty),
                                    b: ptx::Operand::Reg(
                                        format!("%r{}", i + elements_per_thread),
                                        ty,
                                    ),
                                    ty,
                                });
                            }

                            // Store out
                            insts.push(ptx::Inst::StGlobal {
                                addr: ptx::Operand::Reg(
                                    format!(
                                        "[{} + threadIdx.x * {}]",
                                        out,
                                        elements_per_thread * 4
                                    ),
                                    ty,
                                ),
                                src: (0..*elements_per_thread)
                                    .map(|i| {
                                        ptx::Operand::Reg(
                                            format!("%r{}", i + 2 * elements_per_thread),
                                            ty,
                                        )
                                    })
                                    .collect(),
                                ty,
                                vec: vec_width,
                            });

                            insts
                        }
                    });
                    let func = ptx::Function {
                        name,
                        params: inputs
                            .iter()
                            .enumerate()
                            .map(|(i, _inp)| (format!("param{}", i), ptx::Type::F32))
                            .collect(),
                        body: ptx_body.collect(),
                    };
                    PtxGraphNode::Function { func }
                }
                TileGraphNode::Constant { data } => PtxGraphNode::Constant { data: data.clone() },
                TileGraphNode::Input { name } => todo!(),
                TileGraphNode::Parameter { id, data } => todo!(),
            },
            |_idx, e| e,
        );
        Self { graph: ptx_graph }
    }
}

impl From<PtxGraph> for ptx::Module {
    fn from(ptx_graph: PtxGraph) -> Self {
        let functions = ptx_graph
            .graph
            .node_indices()
            .filter_map(|idx| ptx_graph.graph[idx].as_function().cloned())
            .collect();
        ptx::Module { functions }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Constant;

    #[test]
    fn lowers_add_to_ptx() {
        let a = Constant::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]);
        let b = Constant::new(vec![10.0f32, 20.0, 30.0, 40.0], vec![4]);
        let c = a + b;
        println!("Expr: {c:?}");

        let tensor_graph: TensorGraph<f32> = c.into();
        let tile_graph: TileGraph = tensor_graph.into();
        let ptx_graph: PtxGraph = tile_graph.into();
        let ptx_module: ptx::Module = ptx_graph.into();

        insta::assert_snapshot!(ptx_module);
    }
}
