use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};

use crate::graph::{NodeIndex, TensorGraph, TensorGraphNode};
use crate::tensor::{ReduceOp, Shape};

use super::{
    DType, FusionRegion, IndexExpr, RegionOp, RegionOutput, RegionValue, TileIR, TileIRBuilder,
    VirtualTensor,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionInputDomain {
    Full,
    Reduced,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReductionInput {
    pub node: NodeIndex,
    pub value: RegionValue,
    pub domain: ReductionInputDomain,
    pub tensor: VirtualTensor,
    pub source_shape: Shape,
}

/// One strict-order reduction with pointwise producers and reduced epilogues.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReductionRegion {
    pub members: Vec<NodeIndex>,
    pub inputs: Vec<ReductionInput>,
    pub outputs: Vec<RegionOutput>,
    pub output_shapes: Vec<Shape>,
    pub producer_operations: Vec<RegionOp>,
    pub epilogue_operations: Vec<RegionOp>,
    pub broadcast_members: Vec<NodeIndex>,
    pub full_epilogue_operations: Vec<RegionOp>,
    pub reduction_input: RegionValue,
    pub reduced_value: RegionValue,
    pub input_shape: Shape,
    pub output_shape: Shape,
    pub axis: usize,
    pub op: ReduceOp,
}

impl ReductionRegion {
    pub fn from_graph<G>(
        graph: &TensorGraph<f32, G>,
        producer_members: Vec<NodeIndex>,
        anchor: NodeIndex,
        epilogue_members: Vec<NodeIndex>,
        broadcast_members: Vec<NodeIndex>,
        full_epilogue_members: Vec<NodeIndex>,
        output_nodes: Vec<NodeIndex>,
    ) -> Result<Self> {
        anyhow::ensure!(!output_nodes.is_empty(), "reduction region has no outputs");
        let mut listed_members = HashSet::new();
        for &member in producer_members
            .iter()
            .chain(std::iter::once(&anchor))
            .chain(&epilogue_members)
            .chain(&broadcast_members)
            .chain(&full_epilogue_members)
        {
            anyhow::ensure!(
                listed_members.insert(member),
                "reduction region member {} appears more than once",
                member.index()
            );
        }
        let mut listed_outputs = HashSet::new();
        for &output in &output_nodes {
            anyhow::ensure!(
                listed_outputs.insert(output),
                "reduction region output {} appears more than once",
                output.index()
            );
        }

        let (op, axis, output_shape) = match &graph[anchor] {
            TensorGraphNode::ReduceAxis { op, axis, shape } => (*op, *axis, shape.clone()),
            _ => anyhow::bail!("reduction region anchor is not a reduction"),
        };
        let anchor_inputs = graph.inputs(anchor);
        anyhow::ensure!(
            anchor_inputs.len() == 1,
            "reduction anchor requires one input"
        );
        let input_shape = graph[anchor_inputs[0]].shape().clone();
        anyhow::ensure!(axis < input_shape.len(), "reduction axis is out of bounds");
        anyhow::ensure!(
            output_shape.len() == input_shape.len()
                && output_shape[axis] == 1
                && output_shape
                    .iter()
                    .zip(&input_shape)
                    .enumerate()
                    .all(|(dimension, (output, input))| dimension == axis || output == input),
            "reduction output shape is incompatible with its input"
        );

        let producer_set: HashSet<_> = producer_members.iter().copied().collect();
        let epilogue_set: HashSet<_> = epilogue_members.iter().copied().collect();
        let full_epilogue_set: HashSet<_> = full_epilogue_members.iter().copied().collect();
        let mut values = HashMap::new();
        let mut inputs = Vec::new();
        let mut next_value = 0usize;

        let mut producer_operations = Vec::new();
        for &member in &producer_members {
            anyhow::ensure!(
                graph[member].shape() == &input_shape,
                "reduction producer shape mismatch"
            );
            for input in graph.inputs(member) {
                if !producer_set.contains(&input) {
                    add_external_input(
                        graph,
                        input,
                        ReductionInputDomain::Full,
                        &input_shape,
                        &mut values,
                        &mut inputs,
                        &mut next_value,
                    )?;
                }
            }
            let operation = build_pointwise_operation(graph, member, &values, &mut next_value)?;
            values.insert(member, operation.output);
            producer_operations.push(operation);
        }
        let reduction_input_node = anchor_inputs[0];
        let reduction_input = if let Some(value) = values.get(&reduction_input_node).copied() {
            value
        } else {
            add_external_input(
                graph,
                reduction_input_node,
                ReductionInputDomain::Full,
                &input_shape,
                &mut values,
                &mut inputs,
                &mut next_value,
            )?
        };
        let reduced_value = RegionValue(next_value);
        next_value += 1;
        values.insert(anchor, reduced_value);

        let mut epilogue_operations = Vec::new();
        for &member in &epilogue_members {
            anyhow::ensure!(
                graph[member].shape() == &output_shape,
                "reduction epilogue shape mismatch"
            );
            for input in graph.inputs(member) {
                if input != anchor && !epilogue_set.contains(&input) {
                    add_external_input(
                        graph,
                        input,
                        ReductionInputDomain::Reduced,
                        &output_shape,
                        &mut values,
                        &mut inputs,
                        &mut next_value,
                    )?;
                }
            }
            let operation = build_pointwise_operation(graph, member, &values, &mut next_value)?;
            values.insert(member, operation.output);
            epilogue_operations.push(operation);
        }

        for &bridge in &broadcast_members {
            anyhow::ensure!(
                matches!(graph[bridge], TensorGraphNode::BroadcastAxis { axis: bridge_axis, .. } if bridge_axis == axis),
                "reduction region broadcast bridge is incompatible with its axis"
            );
            anyhow::ensure!(
                graph[bridge].shape() == &input_shape,
                "reduction broadcast output shape is incompatible with the full domain"
            );
            let bridge_inputs = graph.inputs(bridge);
            anyhow::ensure!(
                bridge_inputs.len() == 1,
                "reduction broadcast bridge requires one input"
            );
            let bridge_value = values
                .get(&bridge_inputs[0])
                .copied()
                .context("reduction broadcast source value is unavailable")?;
            values.insert(bridge, bridge_value);
        }
        let mut full_epilogue_operations = Vec::new();
        for &member in &full_epilogue_members {
            anyhow::ensure!(
                graph[member].shape() == &input_shape,
                "full reduction epilogue shape mismatch"
            );
            for input in graph.inputs(member) {
                if !producer_set.contains(&input)
                    && !full_epilogue_set.contains(&input)
                    && !broadcast_members.contains(&input)
                {
                    add_external_input(
                        graph,
                        input,
                        ReductionInputDomain::Full,
                        &input_shape,
                        &mut values,
                        &mut inputs,
                        &mut next_value,
                    )?;
                }
            }
            let operation = build_pointwise_operation(graph, member, &values, &mut next_value)?;
            values.insert(member, operation.output);
            full_epilogue_operations.push(operation);
        }

        let member_set: HashSet<_> = producer_members
            .iter()
            .chain(std::iter::once(&anchor))
            .chain(&epilogue_members)
            .chain(&broadcast_members)
            .chain(&full_epilogue_members)
            .copied()
            .collect();
        let output_shapes = output_nodes
            .iter()
            .map(|&node| graph[node].shape().clone())
            .collect();
        let outputs = output_nodes
            .into_iter()
            .map(|node| {
                anyhow::ensure!(
                    member_set.contains(&node),
                    "reduction output is not a member"
                );
                let value = values.get(&node).copied().with_context(|| {
                    format!("reduction output {} has no SSA value", node.index())
                })?;
                Ok(RegionOutput { node, value })
            })
            .collect::<Result<_>>()?;
        let members = producer_members
            .iter()
            .copied()
            .chain(std::iter::once(anchor))
            .chain(epilogue_members.iter().copied())
            .chain(broadcast_members.iter().copied())
            .chain(full_epilogue_members.iter().copied())
            .collect();
        Ok(Self {
            members,
            inputs,
            outputs,
            output_shapes,
            producer_operations,
            epilogue_operations,
            broadcast_members,
            full_epilogue_operations,
            reduction_input,
            reduced_value,
            input_shape,
            output_shape,
            axis,
            op,
        })
    }

    pub fn lower_to_tile_ir(&self, region_id: usize) -> Result<TileIR> {
        self.validate_lowering_inputs()?;
        let extent = self
            .output_shape
            .iter()
            .try_fold(1usize, |count, &dimension| {
                count
                    .checked_mul(dimension)
                    .context("reduction region output size overflowed usize")
            })?;
        let mut builder = TileIRBuilder::new();
        builder.start_kernel(&format!("reduction_region_{region_id}"));
        for index in 0..self.inputs.len() {
            builder.add_param(&format!("input_{index}"), DType::F32, true);
        }
        for index in 0..self.outputs.len() {
            builder.add_param(&format!("output_{index}"), DType::F32, false);
        }
        builder.bounds_check(extent);
        builder.reduction_region(self.clone());
        Ok(builder.finish())
    }

    fn validate_lowering_inputs(&self) -> Result<()> {
        anyhow::ensure!(
            self.outputs.len() == self.output_shapes.len(),
            "reduction output metadata length mismatch"
        );
        for input in &self.inputs {
            let logical_shape = match input.domain {
                ReductionInputDomain::Full => &self.input_shape,
                ReductionInputDomain::Reduced => &self.output_shape,
            };
            anyhow::ensure!(
                input.tensor.shape == *logical_shape,
                "reduction input {} virtual shape does not match its domain",
                input.node.index()
            );
            anyhow::ensure!(
                input.tensor.dtype == DType::F32,
                "reduction input {} must have f32 dtype",
                input.node.index()
            );
            anyhow::ensure!(
                input.tensor.predicate.is_none() && input.tensor.out_of_bounds.is_none(),
                "predicated reduction input {} is unsupported",
                input.node.index()
            );
            anyhow::ensure!(
                input.tensor.access.results.len() == input.source_shape.len(),
                "reduction input {} access rank does not match source storage",
                input.node.index()
            );
            for expression in &input.tensor.access.results {
                validate_index_expression(expression, logical_shape.len()).with_context(|| {
                    format!(
                        "invalid access map for reduction input {}",
                        input.node.index()
                    )
                })?;
            }
        }
        Ok(())
    }
}

fn add_external_input<G>(
    graph: &TensorGraph<f32, G>,
    node: NodeIndex,
    domain: ReductionInputDomain,
    expected_shape: &[usize],
    values: &mut HashMap<NodeIndex, RegionValue>,
    inputs: &mut Vec<ReductionInput>,
    next_value: &mut usize,
) -> Result<RegionValue> {
    anyhow::ensure!(
        graph[node].shape() == expected_shape,
        "external reduction input {} shape does not match its domain",
        node.index()
    );
    if let Some(&value) = values.get(&node) {
        let existing_domain = inputs
            .iter()
            .find(|input| input.value == value)
            .map(|input| input.domain);
        anyhow::ensure!(
            existing_domain.is_none_or(|existing| existing == domain),
            "external reduction input {} is used in incompatible domains",
            node.index()
        );
        return Ok(value);
    }
    let value = RegionValue(*next_value);
    *next_value += 1;
    values.insert(node, value);
    inputs.push(ReductionInput {
        node,
        value,
        domain,
        tensor: VirtualTensor::identity(node, graph[node].shape().clone(), DType::F32),
        source_shape: graph[node].shape().clone(),
    });
    Ok(value)
}

fn validate_index_expression(expression: &IndexExpr, rank: usize) -> Result<()> {
    match expression {
        IndexExpr::IterDim(dimension) => anyhow::ensure!(
            *dimension < rank,
            "iteration dimension {dimension} is out of bounds for rank {rank}"
        ),
        IndexExpr::Symbol(symbol) => {
            anyhow::bail!("index symbol {symbol} is unsupported by reduction lowering")
        }
        IndexExpr::Const(value) => anyhow::ensure!(*value >= 0, "negative index constant {value}"),
        IndexExpr::Add(lhs, rhs) | IndexExpr::Sub(lhs, rhs) | IndexExpr::Mul(lhs, rhs) => {
            validate_index_expression(lhs, rank)?;
            validate_index_expression(rhs, rank)?;
        }
        IndexExpr::FloorDiv(value, divisor) => {
            anyhow::ensure!(*divisor > 0, "index floor-divisor must be positive");
            validate_index_expression(value, rank)?;
        }
        IndexExpr::Mod(value, modulus) => {
            anyhow::ensure!(*modulus > 0, "index modulus must be positive");
            validate_index_expression(value, rank)?;
        }
    }
    Ok(())
}

fn build_pointwise_operation<G>(
    graph: &TensorGraph<f32, G>,
    node: NodeIndex,
    values: &HashMap<NodeIndex, RegionValue>,
    next_value: &mut usize,
) -> Result<RegionOp> {
    let members = vec![node];
    let outputs = vec![node];
    let temporary = FusionRegion::from_graph(graph, members, outputs)?;
    let operation = temporary
        .operations
        .into_iter()
        .next()
        .context("pointwise operation was not constructed")?;
    let node_inputs = graph.inputs(node);
    let remap = |value: RegionValue| -> Result<RegionValue> {
        if let Some(input) = temporary.inputs.iter().find(|input| input.value == value) {
            return values
                .get(&input.node)
                .copied()
                .context("reduction region operand is unavailable");
        }
        anyhow::bail!("unexpected temporary pointwise value")
    };
    let kind = match operation.kind {
        super::RegionOpKind::Unary { op, input } => super::RegionOpKind::Unary {
            op,
            input: remap(input)?,
        },
        super::RegionOpKind::Binary { op, lhs, rhs } => super::RegionOpKind::Binary {
            op,
            lhs: remap(lhs)?,
            rhs: remap(rhs)?,
        },
        super::RegionOpKind::Gt { lhs, rhs } => super::RegionOpKind::Gt {
            lhs: remap(lhs)?,
            rhs: remap(rhs)?,
        },
        super::RegionOpKind::Mask { values, condition } => super::RegionOpKind::Mask {
            values: remap(values)?,
            condition: remap(condition)?,
        },
    };
    anyhow::ensure!(
        node_inputs.iter().all(|input| values.contains_key(input)),
        "reduction region pointwise dependency is unavailable"
    );
    let output = RegionValue(*next_value);
    *next_value += 1;
    Ok(RegionOp { node, output, kind })
}
