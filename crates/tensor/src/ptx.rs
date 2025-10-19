use std::fmt;
use std::sync::{Arc, Mutex};

use itertools::Itertools;
use petgraph::graph::Graph;

use crate::tile;

#[derive(Clone, Copy, Debug)]
pub enum Type {
    F32,
    F16,
    I32,
    U32,
    Pred,
}

#[derive(Clone, Copy, Debug)]
pub enum VecWidth {
    Scalar,
    V2,
    V4,
}

#[derive(Clone, Debug)]
pub enum Operand {
    Reg(String, Type),
    Pred(String),
    ImmI32(i32),
    ImmF32(f32),
    Addr(String), // label or symbol
}

#[derive(Clone, Debug)]
pub enum Inst {
    // Data movement
    Mov {
        dst: Operand,
        src: Operand,
    },
    LdGlobal {
        dst: Vec<Operand>,
        addr: Operand,
        ty: Type,
        vec: VecWidth,
    },
    StGlobal {
        addr: Operand,
        src: Vec<Operand>,
        ty: Type,
        vec: VecWidth,
    },

    // Math
    Add {
        dst: Operand,
        a: Operand,
        b: Operand,
        ty: Type,
    },
    Mul {
        dst: Operand,
        a: Operand,
        b: Operand,
        ty: Type,
    },
    Fma {
        dst: Operand,
        a: Operand,
        b: Operand,
        c: Operand,
        ty: Type,
    },
    Max {
        dst: Operand,
        a: Operand,
        b: Operand,
        ty: Type,
    },

    // Control
    SetpLtU32 {
        dst: Operand,
        a: Operand,
        b: Operand,
    },
    Bra {
        target: String,
    },
    Label(String),
    Ret,

    // Barrier
    BarSync {
        barrier_id: u32,
    },
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub body: Vec<Inst>,
}

#[derive(Clone, Debug)]
pub struct Module {
    pub functions: Vec<Function>,
}

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, ".version 8.0")?;
        writeln!(f, ".target sm_80")?;
        writeln!(f, ".address_size 64")?;
        writeln!(f)?;

        for func in &self.functions {
            writeln!(f, "{func}\n")?;
        }

        Ok(())
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, ".visible .entry {}() {{", self.name)?;
        for inst in &self.body {
            writeln!(f, "  {inst}")?;
        }
        writeln!(f, "}}")
    }
}

impl fmt::Display for Inst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LdGlobal { dst, addr, ty, vec } => {
                let dst_regs = dst.iter().join(", ");
                write!(f, "ld.global{vec}.{ty} {{ {dst_regs} }}, [{addr}];",)
            }
            Self::Add { dst, a, b, ty } => write!(f, "add.{ty} {dst}, {a}, {b};"),
            Self::StGlobal { addr, src, ty, vec } => {
                let src_regs = src.iter().join(", ");
                write!(f, "st.global{vec}.{ty} [{addr}], {{ {src_regs} }};")
            }
            _ => todo!(),
        }
    }
}

impl fmt::Display for VecWidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar => write!(f, ""),
            Self::V2 => write!(f, ".v2"),
            Self::V4 => write!(f, ".v4"),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::I32 => write!(f, "s32"),
            Self::U32 => write!(f, "u32"),
            Self::Pred => write!(f, "pred"),
        }
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reg(name, _) => write!(f, "{name}"),
            Self::Pred(name) => write!(f, "{name}"),
            Self::ImmI32(v) => write!(f, "{v}"),
            Self::ImmF32(v) => write!(f, "{v}"),
            Self::Addr(sym) => write!(f, "[{sym}]"),
        }
    }
}

pub struct PtxGraph {
    pub graph: Graph<PtxGraphNode, usize>,
}

#[derive(Clone, Debug)]
pub enum PtxGraphNode {
    Function {
        func: Function,
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
    pub fn as_function(&self) -> Option<&Function> {
        if let PtxGraphNode::Function { func } = self {
            Some(func)
        } else {
            None
        }
    }
}

impl From<tile::TileGraph> for PtxGraph {
    fn from(tile_graph: tile::TileGraph) -> Self {
        let ptx_graph = tile_graph.graph.map_owned(
            |_idx, n| match n {
                tile::TileGraphNode::Function {
                    func:
                        tile::Function {
                            name, inputs, body, ..
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
                                1 => VecWidth::Scalar,
                                2 => VecWidth::V2,
                                4 => VecWidth::V4,
                                _ => panic!("Unsupported elements_per_thread"),
                            };
                            let ty = Type::F32;

                            let mut insts = Vec::new();

                            // Load lhs
                            insts.push(Inst::LdGlobal {
                                dst: (0..*elements_per_thread)
                                    .map(|i| Operand::Reg(format!("%r{}", i), ty))
                                    .collect(),
                                addr: Operand::Reg(
                                    format!(
                                        "[{} + threadIdx.x * {}]",
                                        lhs,
                                        elements_per_thread * 4
                                    ),
                                    ty,
                                ),
                                ty,
                                vec: vec_width,
                            });

                            // Load rhs
                            insts.push(Inst::LdGlobal {
                                dst: (0..*elements_per_thread)
                                    .map(|i| {
                                        Operand::Reg(format!("%r{}", i + elements_per_thread), ty)
                                    })
                                    .collect(),
                                addr: Operand::Reg(
                                    format!(
                                        "[{} + threadIdx.x * {}]",
                                        rhs,
                                        elements_per_thread * 4
                                    ),
                                    ty,
                                ),
                                ty,
                                vec: vec_width,
                            });

                            // Add
                            for i in 0..*elements_per_thread {
                                insts.push(Inst::Add {
                                    dst: Operand::Reg(
                                        format!("%r{}", i + 2 * elements_per_thread),
                                        ty,
                                    ),
                                    a: Operand::Reg(format!("r{}", i), ty),
                                    b: Operand::Reg(format!("%r{}", i + elements_per_thread), ty),
                                    ty,
                                });
                            }

                            // Store out
                            insts.push(Inst::StGlobal {
                                addr: Operand::Reg(
                                    format!(
                                        "[{} + threadIdx.x * {}]",
                                        out,
                                        elements_per_thread * 4
                                    ),
                                    ty,
                                ),
                                src: (0..*elements_per_thread)
                                    .map(|i| {
                                        Operand::Reg(
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
                    let func = Function {
                        name,
                        params: inputs
                            .iter()
                            .enumerate()
                            .map(|(i, _inp)| (format!("param{}", i), Type::F32))
                            .collect(),
                        body: ptx_body.collect(),
                    };
                    PtxGraphNode::Function { func }
                }
                tile::TileGraphNode::Constant { data } => {
                    PtxGraphNode::Constant { data: data.clone() }
                }
                tile::TileGraphNode::Input { .. } => todo!(),
                tile::TileGraphNode::Parameter { .. } => todo!(),
            },
            |_idx, e| e,
        );
        Self { graph: ptx_graph }
    }
}

impl From<PtxGraph> for Module {
    fn from(ptx_graph: PtxGraph) -> Self {
        let functions = ptx_graph
            .graph
            .node_indices()
            .filter_map(|idx| ptx_graph.graph[idx].as_function().cloned())
            .collect();
        Module { functions }
    }
}
