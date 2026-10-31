//! Explicit stable SwiGLU training kernels; no automatic graph selection.
//!
//! The backward scalar program is extracted from primitive autodiff, preserving
//! its masks, quotients and gradient accumulation order. It deliberately avoids
//! the algebraically simplified sigmoid derivative. Forward saves only gate/up
//! operands; backward recomputes inexpensive scalar locals in registers.
use std::collections::HashMap;

use anyhow::{Context, Result, bail, ensure};

use crate::graph::{NodeIndex, TensorGraph, TensorGraphNode};
use crate::tensor::{BinaryOp, Parameter, Shape, UnaryOp};

use super::{
    Block, DType, Dim, Expr, KernelParam, ScalarBinaryOp, ScalarBuilder, ScalarExpr, ScalarMemory,
    Stmt, TileIR,
};

#[derive(Clone, Debug)]
enum Op {
    Input(usize),
    Constant(f32),
    Unary(UnaryOp, usize),
    Binary(BinaryOp, usize, usize),
    Gt(usize, usize),
    Mask(usize, usize),
}

#[derive(Clone, Debug)]
struct Program {
    ops: Vec<Op>,
    outputs: Vec<usize>,
}

/// A bounded homogeneous-F32 training schedule, independent of reduction policy.
///
/// Forward parameters: `input_0=gate`, `input_1=up`, `output_0=value`.
/// Backward: the same two inputs, `input_2=dvalue`, then `output_0=dgate`
/// and `output_1=dup`. All buffers have `shape` elements in contiguous order.
/// The existing compiler's scalar arithmetic/contraction policy is unchanged.
#[derive(Clone, Debug)]
pub struct SwiGluTraining {
    shape: Shape,
    extent: usize,
    forward: Program,
    backward: Program,
}

impl SwiGluTraining {
    pub fn new(shape: Shape) -> Result<Self> {
        ensure!(
            !shape.is_empty() && !shape.contains(&0),
            "SwiGLU shape must be nonempty"
        );
        let extent = shape.iter().try_fold(1usize, |n, &d| {
            n.checked_mul(d)
                .context("SwiGLU element count overflowed usize")
        })?;
        ensure!(i64::try_from(extent).is_ok(), "SwiGLU extent exceeds i64");
        let gate = Parameter::new(vec![0.0], vec![1]);
        let up = Parameter::new(vec![0.0], vec![1]);
        let gate_id = gate.id();
        let up_id = up.id();
        let graph: TensorGraph<f32> = (crate::nn::silu(gate) * up).into();
        let output = *graph.toposort().last().context("empty SwiGLU template")?;
        // with_gradients appends its scalar loss seed first. Replace that seed
        // with dvalue, so no extra multiply-by-one enters the actual VJP.
        let seed = NodeIndex::new(graph.graph.node_count());
        let graph = graph.with_gradients(output);
        ensure!(
            matches!(&graph[seed], TensorGraphNode::Constant {data, ..} if data.as_slice() == [1.0]),
            "SwiGLU autodiff seed contract changed"
        );
        let mut inputs = HashMap::from([(seed, 2)]);
        for node in graph.graph.node_indices() {
            if let TensorGraphNode::Parameter { id, .. } = graph[node] {
                if id == gate_id {
                    inputs.insert(node, 0);
                }
                if id == up_id {
                    inputs.insert(node, 1);
                }
            }
        }
        let gradients = &graph.gradient_metadata().param_to_grad;
        let gate_gradient = *gradients
            .get(&gate_id)
            .context("missing SwiGLU gate gradient")?;
        let up_gradient = *gradients
            .get(&up_id)
            .context("missing SwiGLU up gradient")?;
        let forward = Program::extract(&graph, &[output], &inputs)?;
        let backward = Program::extract(&graph, &[gate_gradient, up_gradient], &inputs)?;
        Ok(Self {
            shape,
            extent,
            forward,
            backward,
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn extent(&self) -> usize {
        self.extent
    }
    pub fn forward_scalar_operations(&self) -> usize {
        self.forward.ops.len()
    }
    pub fn backward_scalar_operations(&self) -> usize {
        self.backward.ops.len()
    }

    pub fn lower_forward(&self) -> Result<TileIR> {
        self.forward.lower("swiglu_forward", self.extent, 2)
    }
    pub fn lower_backward(&self) -> Result<TileIR> {
        self.backward.lower("swiglu_backward", self.extent, 3)
    }
}

impl Program {
    fn extract<G>(
        graph: &TensorGraph<f32, G>,
        outputs: &[NodeIndex],
        inputs: &HashMap<NodeIndex, usize>,
    ) -> Result<Self> {
        fn visit<G>(
            graph: &TensorGraph<f32, G>,
            node: NodeIndex,
            inputs: &HashMap<NodeIndex, usize>,
            values: &mut HashMap<NodeIndex, usize>,
            ops: &mut Vec<Op>,
        ) -> Result<usize> {
            if let Some(&value) = values.get(&node) {
                return Ok(value);
            }
            let op = if let Some(&input) = inputs.get(&node) {
                Op::Input(input)
            } else {
                let args = graph
                    .inputs(node)
                    .into_iter()
                    .map(|n| visit(graph, n, inputs, values, ops))
                    .collect::<Result<Vec<_>>>()?;
                match &graph[node] {
                    TensorGraphNode::Constant { data, .. } => {
                        ensure!(data.len() == 1, "SwiGLU template constant is not scalar");
                        Op::Constant(data[0])
                    }
                    TensorGraphNode::Unary { op, .. } => Op::Unary(*op, args[0]),
                    TensorGraphNode::Binary { op, .. } => Op::Binary(*op, args[0], args[1]),
                    TensorGraphNode::Gt { .. } => Op::Gt(args[0], args[1]),
                    TensorGraphNode::Mask { .. } => Op::Mask(args[0], args[1]),
                    TensorGraphNode::Reshape { .. } | TensorGraphNode::BroadcastAxis { .. } => {
                        ensure!(
                            graph[node].shape().iter().product::<usize>() == 1,
                            "SwiGLU template view is not scalar"
                        );
                        values.insert(node, args[0]);
                        return Ok(args[0]);
                    }
                    other => bail!("unsupported SwiGLU template operation {}", other.name()),
                }
            };
            let value = ops.len();
            ops.push(op);
            values.insert(node, value);
            Ok(value)
        }
        let mut ops = Vec::new();
        let mut values = HashMap::new();
        let outputs = outputs
            .iter()
            .map(|&n| visit(graph, n, inputs, &mut values, &mut ops))
            .collect::<Result<_>>()?;
        Ok(Self { ops, outputs })
    }

    fn lower(&self, name: &str, extent: usize, input_count: usize) -> Result<TileIR> {
        let index = Expr::Mul(
            Box::new(Expr::BlockIdx(Dim::X)),
            Box::new(Expr::BlockDim(Dim::X)),
        ) + Expr::ThreadIdx(Dim::X);
        let mut builder = ScalarBuilder::default();
        builder.stmts.push(Stmt::BoundsCheck { extent });
        let mut registers = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            let value = match *op {
                Op::Input(input) => {
                    ensure!(input < input_count, "SwiGLU input outside kernel bindings");
                    builder.load(
                        ScalarMemory::Global(format!("input_{input}")),
                        index.clone(),
                    )
                }
                Op::Constant(value) => builder.value(ScalarExpr::Constant(value)),
                Op::Unary(op, input) => builder.value(ScalarExpr::unary(op, registers[input])),
                Op::Binary(op, lhs, rhs) => builder.value(ScalarExpr::binary(
                    ScalarBinaryOp::Arithmetic(op),
                    registers[lhs],
                    registers[rhs],
                )),
                Op::Gt(lhs, rhs) => {
                    let dest = builder.var();
                    builder.stmts.push(Stmt::Gt {
                        dest,
                        a: registers[lhs],
                        b: registers[rhs],
                    });
                    dest
                }
                Op::Mask(values, condition) => {
                    let dest = builder.var();
                    builder.stmts.push(Stmt::Mask {
                        dest,
                        values: registers[values],
                        condition: registers[condition],
                    });
                    dest
                }
            };
            registers.push(value);
        }
        for (output, &value) in self.outputs.iter().enumerate() {
            builder.store(
                ScalarMemory::Global(format!("output_{output}")),
                index.clone(),
                registers[value],
            );
        }
        let params = (0..input_count)
            .map(|i| KernelParam {
                name: format!("input_{i}"),
                dtype: DType::F32,
                is_input: true,
            })
            .chain((0..self.outputs.len()).map(|i| KernelParam {
                name: format!("output_{i}"),
                dtype: DType::F32,
                is_input: false,
            }))
            .collect();
        Ok(TileIR {
            kernel_name: name.into(),
            params,
            body: Block {
                stmts: builder.stmts,
            },
            shared_mem_bytes: 0,
        })
    }

    #[cfg(test)]
    fn evaluate(&self, inputs: &[f32]) -> Vec<f32> {
        let mut values: Vec<f32> = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            let value = match *op {
                Op::Input(i) => inputs[i],
                Op::Constant(x) => x,
                Op::Unary(op, i) => match op {
                    UnaryOp::Neg => -values[i],
                    UnaryOp::Exp => values[i].exp(),
                    UnaryOp::Log => values[i].ln(),
                    UnaryOp::Relu => values[i].max(0.0),
                },
                Op::Binary(op, a, b) => match op {
                    BinaryOp::Add => values[a] + values[b],
                    BinaryOp::Sub => values[a] - values[b],
                    BinaryOp::Mul => values[a] * values[b],
                    BinaryOp::Div => values[a] / values[b],
                },
                Op::Gt(a, b) => {
                    if values[a] > values[b] {
                        1.0
                    } else {
                        0.0
                    }
                }
                Op::Mask(a, b) => {
                    if values[b] != 0.0 {
                        values[a]
                    } else {
                        0.0
                    }
                }
            };
            values.push(value);
        }
        self.outputs.iter().map(|&i| values[i]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorExpr;
    use crate::{Executor, SimpleExecutor};

    fn same(a: f32, b: f32) {
        assert!(
            a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan(),
            "{a:?} differs from {b:?}"
        );
    }

    #[test]
    fn swiglu_scalar_vjp_preserves_primitive_order_and_nonfinite_classes() {
        let kernel = SwiGluTraining::new(vec![2, 683]).unwrap();
        let cases = [
            f32::NEG_INFINITY,
            -f32::MAX,
            -104.0,
            -90.0,
            -20.0,
            -1.0,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            1.0,
            20.0,
            90.0,
            f32::MAX,
            f32::INFINITY,
            f32::NAN,
        ];
        for gate_value in cases {
            for up_value in [-2.0, 0.0, 1.25, f32::INFINITY, f32::NAN] {
                for incoming in [-0.7, 0.0, 2.0, f32::INFINITY, f32::NAN] {
                    let gate = Parameter::new(vec![gate_value], vec![1]);
                    let up = Parameter::new(vec![up_value], vec![1]);
                    let ids = [gate.id(), up.id()];
                    let graph: TensorGraph<f32> = (crate::nn::silu(gate)
                        * up
                        * TensorExpr::constant(vec![incoming], vec![1]))
                    .into();
                    let loss = *graph.toposort().last().unwrap();
                    let graph = graph.with_gradients(loss);
                    let mut cpu = SimpleExecutor::new();
                    cpu.execute(&graph, HashMap::new()).unwrap();
                    let gradients = cpu.get_gradients(&graph);
                    let actual = kernel.backward.evaluate(&[gate_value, up_value, incoming]);
                    same(actual[0], gradients[&ids[0]][0]);
                    same(actual[1], gradients[&ids[1]][0]);
                    let expected: TensorGraph<f32> =
                        (crate::nn::silu(TensorExpr::constant(vec![gate_value], vec![1]))
                            * TensorExpr::constant(vec![up_value], vec![1]))
                        .into();
                    same(
                        kernel.forward.evaluate(&[gate_value, up_value])[0],
                        SimpleExecutor::new()
                            .execute(&expected, HashMap::new())
                            .unwrap()[0],
                    );
                }
            }
        }
    }

    #[test]
    fn swiglu_kernel_has_only_operands_and_two_gradient_stores() {
        let kernel = SwiGluTraining::new(vec![4096, 683]).unwrap();
        let forward = kernel.lower_forward().unwrap();
        let backward = kernel.lower_backward().unwrap();
        assert_eq!(forward.params.len(), 3);
        assert_eq!(backward.params.len(), 5);
        assert_eq!(
            backward
                .body
                .stmts
                .iter()
                .filter(|s| matches!(s, Stmt::StoreScalar { .. }))
                .count(),
            2
        );
        assert_eq!(
            backward
                .body
                .stmts
                .iter()
                .filter(|s| matches!(s, Stmt::LoadScalar { .. }))
                .count(),
            3
        );
        assert!(SwiGluTraining::new(vec![]).is_err());
        assert!(SwiGluTraining::new(vec![0, 683]).is_err());
        assert!(SwiGluTraining::new(vec![usize::MAX, 2]).is_err());
    }
}
