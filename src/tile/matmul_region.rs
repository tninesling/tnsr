use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};

use crate::graph::{NodeIndex, TensorGraph, TensorGraphNode};
use crate::tensor::Shape;

use super::{
    FusionRegion, RegionInput, RegionOp, RegionOpKind, RegionOutput, RegionValue, VirtualTensor,
};

/// One broadcast-aware batched matrix multiplication followed by same-shape pointwise operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatMulRegion {
    pub members: Vec<NodeIndex>,
    pub inputs: Vec<RegionInput>,
    pub outputs: Vec<RegionOutput>,
    pub epilogue_operations: Vec<RegionOp>,
    pub lhs_value: RegionValue,
    pub rhs_value: RegionValue,
    pub matmul_value: RegionValue,
    pub lhs_shape: Shape,
    pub rhs_shape: Shape,
    pub output_shape: Shape,
    pub batch_shape: Shape,
    pub schedule: super::MatMulSchedule,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl MatMulRegion {
    pub fn from_graph<D: crate::tile::TileDType, G>(
        graph: &TensorGraph<D, G>,
        anchor: NodeIndex,
        epilogue_members: Vec<NodeIndex>,
        output_nodes: Vec<NodeIndex>,
    ) -> Result<Self> {
        anyhow::ensure!(!output_nodes.is_empty(), "matmul region has no outputs");

        let mut member_set = HashSet::new();
        anyhow::ensure!(
            member_set.insert(anchor),
            "matmul anchor appears more than once"
        );
        for &member in &epilogue_members {
            anyhow::ensure!(
                member_set.insert(member),
                "matmul region member {} appears more than once",
                member.index()
            );
        }
        let mut listed_outputs = HashSet::new();
        for &output in &output_nodes {
            anyhow::ensure!(
                listed_outputs.insert(output),
                "matmul region output {} appears more than once",
                output.index()
            );
        }

        let output_shape = match &graph[anchor] {
            TensorGraphNode::MatMul { shape } => shape.clone(),
            _ => anyhow::bail!("matmul region anchor is not a matrix multiplication"),
        };
        let anchor_inputs = graph.inputs(anchor);
        anyhow::ensure!(
            anchor_inputs.len() == 2,
            "matmul anchor requires two inputs"
        );
        let lhs_node = anchor_inputs[0];
        let rhs_node = anchor_inputs[1];
        let lhs_shape = graph[lhs_node].shape().clone();
        let rhs_shape = graph[rhs_node].shape().clone();
        anyhow::ensure!(
            lhs_shape.len() >= 2 && rhs_shape.len() >= 2 && output_shape.len() >= 2,
            "matmul region requires rank >= 2"
        );
        let (m, k) = (
            lhs_shape[lhs_shape.len() - 2],
            lhs_shape[lhs_shape.len() - 1],
        );
        let n = rhs_shape[rhs_shape.len() - 1];
        anyhow::ensure!(k > 0, "matmul region requires a non-zero inner dimension");
        anyhow::ensure!(
            rhs_shape[rhs_shape.len() - 2] == k,
            "matmul region inner dimensions do not match"
        );
        anyhow::ensure!(
            output_shape[output_shape.len() - 2..] == [m, n],
            "matmul output shape is incompatible with its inputs"
        );
        let batch_shape = output_shape[..output_shape.len() - 2].to_vec();
        for shape in [&lhs_shape, &rhs_shape] {
            let batch = &shape[..shape.len() - 2];
            anyhow::ensure!(
                batch.len() <= batch_shape.len(),
                "matmul input batch rank exceeds output rank"
            );
            for (&input, &output) in batch.iter().rev().zip(batch_shape.iter().rev()) {
                anyhow::ensure!(
                    input == 1 || input == output,
                    "matmul batch dimension cannot broadcast"
                );
            }
        }
        let schedule = super::MatMulSchedule::select_for_dtype(
            m,
            n,
            k,
            D::TILE_DTYPE,
            super::MatMulCapabilities {
                tf32: true,
                f16: false,
                bf16: false,
            },
            super::MatMulPrecision::AllowTf32,
        );

        let mut values = HashMap::new();
        let mut inputs = Vec::new();
        let mut next_value = 0usize;
        let lhs_value = add_input(graph, lhs_node, &mut values, &mut inputs, &mut next_value);
        let rhs_value = add_input(graph, rhs_node, &mut values, &mut inputs, &mut next_value);
        let matmul_value = RegionValue(next_value);
        next_value += 1;
        values.insert(anchor, matmul_value);

        let epilogue_set: HashSet<_> = epilogue_members.iter().copied().collect();
        let mut epilogue_operations = Vec::with_capacity(epilogue_members.len());
        for &member in &epilogue_members {
            anyhow::ensure!(
                graph[member].shape() == &output_shape,
                "matmul epilogue shape mismatch"
            );
            for input in graph.inputs(member) {
                if values.contains_key(&input) {
                    continue;
                }
                anyhow::ensure!(
                    !epilogue_set.contains(&input),
                    "matmul epilogue dependency {} is unavailable; members must be topologically ordered",
                    input.index()
                );
                anyhow::ensure!(
                    graph[input].shape() == &output_shape,
                    "external matmul epilogue input {} shape mismatch",
                    input.index()
                );
                add_input(graph, input, &mut values, &mut inputs, &mut next_value);
            }
            let operation = build_pointwise_operation(graph, member, &values, &mut next_value)?;
            values.insert(member, operation.output);
            epilogue_operations.push(operation);
        }

        let outputs = output_nodes
            .into_iter()
            .map(|node| {
                anyhow::ensure!(
                    member_set.contains(&node),
                    "matmul output is not a region member"
                );
                let value = values
                    .get(&node)
                    .copied()
                    .with_context(|| format!("matmul output {} has no SSA value", node.index()))?;
                Ok(RegionOutput { node, value })
            })
            .collect::<Result<_>>()?;
        let members = std::iter::once(anchor)
            .chain(epilogue_members.iter().copied())
            .collect();

        Ok(Self {
            members,
            inputs,
            outputs,
            epilogue_operations,
            lhs_value,
            rhs_value,
            matmul_value,
            lhs_shape,
            rhs_shape,
            output_shape,
            batch_shape,
            schedule,
            m,
            n,
            k,
        })
    }
}

fn add_input<D: crate::tile::TileDType, G>(
    graph: &TensorGraph<D, G>,
    node: NodeIndex,
    values: &mut HashMap<NodeIndex, RegionValue>,
    inputs: &mut Vec<RegionInput>,
    next_value: &mut usize,
) -> RegionValue {
    if let Some(&value) = values.get(&node) {
        return value;
    }
    let value = RegionValue(*next_value);
    *next_value += 1;
    values.insert(node, value);
    inputs.push(RegionInput {
        node,
        value,
        tensor: VirtualTensor::identity(node, graph[node].shape().clone(), D::TILE_DTYPE),
        source_shape: graph[node].shape().clone(),
    });
    value
}

fn build_pointwise_operation<D: crate::tile::TileDType, G>(
    graph: &TensorGraph<D, G>,
    node: NodeIndex,
    values: &HashMap<NodeIndex, RegionValue>,
    next_value: &mut usize,
) -> Result<RegionOp> {
    let temporary = FusionRegion::from_graph(graph, vec![node], vec![node])?;
    let operation = temporary
        .operations
        .into_iter()
        .next()
        .context("pointwise matmul epilogue operation was not constructed")?;
    let remap = |value: RegionValue| -> Result<RegionValue> {
        let input = temporary
            .inputs
            .iter()
            .find(|input| input.value == value)
            .context("unexpected temporary pointwise value")?;
        values
            .get(&input.node)
            .copied()
            .context("matmul epilogue operand is unavailable")
    };
    let kind = match operation.kind {
        RegionOpKind::Unary { op, input } => RegionOpKind::Unary {
            op,
            input: remap(input)?,
        },
        RegionOpKind::Binary { op, lhs, rhs } => RegionOpKind::Binary {
            op,
            lhs: remap(lhs)?,
            rhs: remap(rhs)?,
        },
        RegionOpKind::Gt { lhs, rhs } => RegionOpKind::Gt {
            lhs: remap(lhs)?,
            rhs: remap(rhs)?,
        },
        RegionOpKind::Mask { values, condition } => RegionOpKind::Mask {
            values: remap(values)?,
            condition: remap(condition)?,
        },
    };
    let output = RegionValue(*next_value);
    *next_value += 1;
    Ok(RegionOp { node, output, kind })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TensorGraph;
    use crate::tensor::{BinaryOp, UnaryOp};

    fn add_node(
        graph: &mut TensorGraph<f32>,
        node: TensorGraphNode<f32>,
        inputs: &[NodeIndex],
    ) -> NodeIndex {
        let index = graph.graph.add_node(node);
        for (position, &input) in inputs.iter().enumerate() {
            graph.graph.add_edge(input, index, position);
        }
        index
    }

    #[test]
    fn builds_rank_two_matmul_with_pointwise_epilogue() {
        let mut graph = TensorGraph::new();
        let lhs = add_node(
            &mut graph,
            TensorGraphNode::Input {
                name: "lhs",
                shape: vec![2, 3],
            },
            &[],
        );
        let rhs = add_node(
            &mut graph,
            TensorGraphNode::Input {
                name: "rhs",
                shape: vec![3, 4],
            },
            &[],
        );
        let bias = add_node(
            &mut graph,
            TensorGraphNode::Input {
                name: "bias",
                shape: vec![2, 4],
            },
            &[],
        );
        let matmul = add_node(
            &mut graph,
            TensorGraphNode::MatMul { shape: vec![2, 4] },
            &[lhs, rhs],
        );
        let add = add_node(
            &mut graph,
            TensorGraphNode::Binary {
                op: BinaryOp::Add,
                shape: vec![2, 4],
            },
            &[matmul, bias],
        );
        let relu = add_node(
            &mut graph,
            TensorGraphNode::Unary {
                op: UnaryOp::Relu,
                shape: vec![2, 4],
            },
            &[add],
        );

        let region =
            MatMulRegion::from_graph(&graph, matmul, vec![add, relu], vec![matmul, relu]).unwrap();

        assert_eq!(region.members, vec![matmul, add, relu]);
        assert_eq!(
            region
                .inputs
                .iter()
                .map(|input| input.node)
                .collect::<Vec<_>>(),
            vec![lhs, rhs, bias]
        );
        assert_eq!((region.m, region.n, region.k), (2, 4, 3));
        assert_eq!(region.output_shape, vec![2, 4]);
        assert_eq!(region.epilogue_operations.len(), 2);
        assert_eq!(region.outputs[0].value, region.matmul_value);
        assert_eq!(
            region.outputs[1].value,
            region.epilogue_operations[1].output
        );
    }

    #[test]
    fn rejects_zero_inner_dimension() {
        let mut graph = TensorGraph::new();
        let lhs = graph.graph.add_node(TensorGraphNode::Input {
            name: "lhs",
            shape: vec![2, 0],
        });
        let rhs = graph.graph.add_node(TensorGraphNode::Input {
            name: "rhs",
            shape: vec![0, 4],
        });
        let matmul = add_node(
            &mut graph,
            TensorGraphNode::MatMul { shape: vec![2, 4] },
            &[lhs, rhs],
        );

        let error = MatMulRegion::from_graph(&graph, matmul, vec![], vec![matmul])
            .unwrap_err()
            .to_string();

        assert!(error.contains("non-zero inner dimension"));
    }

    #[test]
    fn builds_batched_matmul() {
        let mut graph = TensorGraph::new();
        let lhs = graph.graph.add_node(TensorGraphNode::Input {
            name: "lhs",
            shape: vec![2, 3, 4],
        });
        let rhs = graph.graph.add_node(TensorGraphNode::Input {
            name: "rhs",
            shape: vec![2, 4, 5],
        });
        let matmul = add_node(
            &mut graph,
            TensorGraphNode::MatMul {
                shape: vec![2, 3, 5],
            },
            &[lhs, rhs],
        );

        let region = MatMulRegion::from_graph(&graph, matmul, vec![], vec![matmul]).unwrap();
        assert_eq!(region.batch_shape, vec![2]);
        assert_eq!((region.m, region.n, region.k), (3, 5, 4));
    }
}
