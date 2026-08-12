use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};

use crate::graph::{NodeIndex, TensorGraph, TensorGraphNode};
use crate::tensor::{BinaryOp, Shape, UnaryOp};

use super::{DType, Dim, Expr, TileIR, TileIRBuilder, TileVar};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionValue(pub usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionInput {
    pub node: NodeIndex,
    pub value: RegionValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionOutput {
    pub node: NodeIndex,
    pub value: RegionValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegionOpKind {
    Unary {
        op: UnaryOp,
        input: RegionValue,
    },
    Binary {
        op: BinaryOp,
        lhs: RegionValue,
        rhs: RegionValue,
    },
    Gt {
        lhs: RegionValue,
        rhs: RegionValue,
    },
    Mask {
        values: RegionValue,
        condition: RegionValue,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionOp {
    pub node: NodeIndex,
    pub output: RegionValue,
    pub kind: RegionOpKind,
}

/// Backend-independent scalar SSA for one same-shape pointwise region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionRegion {
    pub members: Vec<NodeIndex>,
    pub inputs: Vec<RegionInput>,
    pub outputs: Vec<RegionOutput>,
    pub operations: Vec<RegionOp>,
    pub shape: Shape,
}

impl FusionRegion {
    pub fn from_graph<G>(
        graph: &TensorGraph<f32, G>,
        members: Vec<NodeIndex>,
        output_nodes: Vec<NodeIndex>,
    ) -> Result<Self> {
        let first = members.first().context("fusion region has no members")?;
        let shape = graph[*first].shape().clone();
        let member_set: HashSet<_> = members.iter().copied().collect();
        let mut values = HashMap::new();
        let mut inputs = Vec::new();
        let mut operations = Vec::with_capacity(members.len());
        let mut next_value = 0usize;

        for &member in &members {
            anyhow::ensure!(
                graph[member].shape() == &shape,
                "pointwise fusion region contains mismatched shapes"
            );
            let node_inputs = graph.inputs(member);
            for &input in &node_inputs {
                anyhow::ensure!(
                    graph[input].shape() == &shape,
                    "pointwise fusion input shape does not match its region"
                );
                if !member_set.contains(&input) && !values.contains_key(&input) {
                    let value = RegionValue(next_value);
                    next_value += 1;
                    values.insert(input, value);
                    inputs.push(RegionInput { node: input, value });
                }
            }

            let resolve = |position: usize| -> Result<RegionValue> {
                let input = node_inputs.get(position).with_context(|| {
                    format!("{} is missing operand {position}", graph[member].name())
                })?;
                values.get(input).copied().with_context(|| {
                    format!("{} operand {position} is unavailable", graph[member].name())
                })
            };
            let kind = match &graph[member] {
                TensorGraphNode::Unary { op, .. } => {
                    anyhow::ensure!(
                        node_inputs.len() == 1,
                        "unary region operation requires one input"
                    );
                    RegionOpKind::Unary {
                        op: *op,
                        input: resolve(0)?,
                    }
                }
                TensorGraphNode::Binary { op, .. } => {
                    anyhow::ensure!(
                        node_inputs.len() == 2,
                        "binary region operation requires two inputs"
                    );
                    RegionOpKind::Binary {
                        op: *op,
                        lhs: resolve(0)?,
                        rhs: resolve(1)?,
                    }
                }
                TensorGraphNode::Gt { .. } => {
                    anyhow::ensure!(
                        node_inputs.len() == 2,
                        "Gt region operation requires two inputs"
                    );
                    RegionOpKind::Gt {
                        lhs: resolve(0)?,
                        rhs: resolve(1)?,
                    }
                }
                TensorGraphNode::Mask { .. } => {
                    anyhow::ensure!(
                        node_inputs.len() == 2,
                        "Mask region operation requires two inputs"
                    );
                    RegionOpKind::Mask {
                        values: resolve(0)?,
                        condition: resolve(1)?,
                    }
                }
                _ => anyhow::bail!(
                    "{} is not a pointwise region operation",
                    graph[member].name()
                ),
            };
            let output = RegionValue(next_value);
            next_value += 1;
            values.insert(member, output);
            operations.push(RegionOp {
                node: member,
                output,
                kind,
            });
        }

        let outputs = output_nodes
            .into_iter()
            .map(|node| {
                anyhow::ensure!(
                    member_set.contains(&node),
                    "fusion output is not a region member"
                );
                Ok(RegionOutput {
                    node,
                    value: values[&node],
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            members,
            inputs,
            outputs,
            operations,
            shape,
        })
    }

    pub fn lower_to_tile_ir(&self, region_id: usize) -> Result<TileIR> {
        let extent = self.shape.iter().try_fold(1usize, |count, &dimension| {
            count
                .checked_mul(dimension)
                .context("fusion region element count overflowed usize")
        })?;
        let mut builder = TileIRBuilder::new();
        builder.start_kernel(&format!("pointwise_region_{region_id}"));
        for index in 0..self.inputs.len() {
            builder.add_param(&format!("input_{index}"), DType::F32, true);
        }
        for index in 0..self.outputs.len() {
            builder.add_param(&format!("output_{index}"), DType::F32, false);
        }
        builder.bounds_check(extent);
        let thread_index = Expr::Add(
            Box::new(Expr::Mul(
                Box::new(Expr::BlockIdx(Dim::X)),
                Box::new(Expr::BlockDim(Dim::X)),
            )),
            Box::new(Expr::ThreadIdx(Dim::X)),
        );
        let mut registers = HashMap::new();
        for (index, input) in self.inputs.iter().enumerate() {
            let register = builder.alloc_register(DType::F32, 1, 1);
            builder.load_global_to_shared(
                register,
                &format!("input_{index}"),
                thread_index.clone(),
                Expr::Const(0),
            );
            registers.insert(input.value, register);
        }
        for operation in &self.operations {
            let output = builder.alloc_register(DType::F32, 1, 1);
            let register = |value: RegionValue| -> Result<TileVar> {
                registers
                    .get(&value)
                    .copied()
                    .context("fusion region SSA value is unavailable")
            };
            match operation.kind {
                RegionOpKind::Unary { op, input } => match op {
                    UnaryOp::Neg => builder.neg(output, register(input)?),
                    UnaryOp::Exp => builder.exp(output, register(input)?),
                    UnaryOp::Log => builder.log(output, register(input)?),
                    UnaryOp::Relu => builder.relu(output, register(input)?),
                },
                RegionOpKind::Binary { op, lhs, rhs } => match op {
                    BinaryOp::Add => builder.add(output, register(lhs)?, register(rhs)?),
                    BinaryOp::Sub => builder.sub(output, register(lhs)?, register(rhs)?),
                    BinaryOp::Mul => builder.mul(output, register(lhs)?, register(rhs)?),
                    BinaryOp::Div => builder.div(output, register(lhs)?, register(rhs)?),
                },
                RegionOpKind::Gt { lhs, rhs } => builder.gt(output, register(lhs)?, register(rhs)?),
                RegionOpKind::Mask { values, condition } => {
                    builder.mask(output, register(values)?, register(condition)?)
                }
            }
            registers.insert(operation.output, output);
        }
        for (index, output) in self.outputs.iter().enumerate() {
            builder.store(
                &format!("output_{index}"),
                registers[&output.value],
                thread_index.clone(),
                Expr::Const(0),
            );
        }
        Ok(builder.finish())
    }
}
