//! Opt-in physical transpose of a closed BF16 weight-gradient GEMM output.
//! The returned Value retains the original logical shape and explicit view chain.
use super::*;
use petgraph::Direction;

fn sole_consumer<G>(graph: &TensorGraph<f32, G>, node: NodeIndex) -> Option<NodeIndex> {
    let mut consumers = graph.graph.neighbors_directed(node, Direction::Outgoing);
    let result = consumers.next()?;
    consumers.next().is_none().then_some(result)
}

/// Walk only backend identity casts, preserving every original graph node.
/// Each traversed value must have exactly one consumer, matching shape and
/// inferred storage dtype. Return the final wrapper and its first nonidentity
/// consumer so callers can verify the required promotion/transpose edges.
fn after_identity_casts<G>(
    graph: &TensorGraph<f32, G>,
    mut node: NodeIndex,
    dtype: ValueDType,
    shape: &[usize],
    dtypes: &HashMap<NodeIndex, ValueDType>,
) -> Option<(NodeIndex, NodeIndex)> {
    for _ in 0..graph.graph.node_count() {
        let next = sole_consumer(graph, node)?;
        if matches!(&graph[next], TensorGraphNode::Cast {dtype:d,shape:s} if *d==dtype && s==shape)
            && dtypes.get(&next) == Some(&dtype)
            && graph.inputs(next) == [node]
        {
            node = next;
        } else {
            return Some((node, next));
        }
    }
    None
}

pub(super) fn eligible<G>(
    graph: &TensorGraph<f32, G>,
    node: NodeIndex,
    dtypes: &HashMap<NodeIndex, ValueDType>,
) -> bool {
    let TensorGraphNode::MatMul { shape } = &graph[node] else {
        return false;
    };
    if shape.len() != 2 || dtypes.get(&node) != Some(&ValueDType::BF16) {
        return false;
    }
    let inputs = graph.inputs(node);
    if inputs.len() != 2
        || inputs
            .iter()
            .any(|n| dtypes.get(n) != Some(&ValueDType::BF16))
    {
        return false;
    }
    let (a, b) = (graph[inputs[0]].shape(), graph[inputs[1]].shape());
    if a.len() != 2
        || b.len() != 2
        || a[0] != shape[0]
        || b[1] != shape[1]
        || a[1] != b[0]
        || shape[0] < 16
        || shape[1] < 16
        || a[1] < 64
        || !a[1].is_multiple_of(32)
        || shape[0].saturating_mul(shape[1]).saturating_mul(a[1]) < 32 * 32 * 32
    {
        return false;
    }
    // Keep the existing long-contraction split-K path completely unchanged.
    if shape[0] >= 128 && shape[1] >= 128 && a[1] >= 2048 && a[1] / shape[0].max(shape[1]) >= 4 {
        return false;
    }
    let Some((bf16_input, cast)) =
        after_identity_casts(graph, node, ValueDType::BF16, shape, dtypes)
    else {
        return false;
    };
    if !matches!(&graph[cast],TensorGraphNode::Cast {dtype: ValueDType::F32,shape: s} if s==shape)
        || graph.inputs(cast) != [bf16_input]
        || dtypes.get(&cast) != Some(&ValueDType::F32)
    {
        return false;
    }
    let Some((f32_input, transpose)) =
        after_identity_casts(graph, cast, ValueDType::F32, shape, dtypes)
    else {
        return false;
    };
    matches!(&graph[transpose],TensorGraphNode::Transpose {shape:s} if s==&vec![shape[1],shape[0]])
        && graph.inputs(transpose) == [f32_input]
        && dtypes.get(&transpose) == Some(&ValueDType::F32)
}

fn transpose_value(source: &Value) -> Result<Value> {
    ensure!(
        source.view.shape.len() == 2,
        "physical-output transpose requires rank two"
    );
    let mut value = source.clone();
    value.view = value.view.permute(&[1, 0])?;
    value.chain.push(TensorGraphNode::Transpose {
        shape: value.view.shape.clone(),
    });
    Ok(value)
}

impl MixedExecutor {
    /// The matcher establishes the closed graph chain. Both operands and the
    /// result remain BF16; F32 promotion occurs at the original graph node.
    pub(super) fn transposed_bf16_output(
        &mut self,
        node: NodeIndex,
        op: &TensorGraphNode<f32>,
        operands: &[Value],
        precision: MatMulPrecision,
        kernel_options: ProgramKernelOptions,
    ) -> Result<Value> {
        let shape = op.shape();
        ensure!(
            matches!(op, TensorGraphNode::MatMul { .. }) && shape.len() == 2 && operands.len() == 2,
            "invalid physical-output transpose operation"
        );
        let a = transpose_value(&operands[1])?;
        let b = transpose_value(&operands[0])?;
        let physical_op = TensorGraphNode::MatMul {
            shape: vec![shape[1], shape[0]],
        };
        let start = self.kernels.len();
        let physical =
            self.operation::<BF16>(node, &physical_op, &[a, b], precision, kernel_options)?;
        let logical = transpose_value(&physical)?;
        let registered = self
            .online_attention_pair_preparation
            .as_ref()
            .map(|_| self.kernels[start..].to_vec());
        let suffix = format!(" transposed-physical-output logical={shape:?}");
        for kernel in &mut self.kernels[start..] {
            kernel.label.push_str(&suffix);
        }
        if let (Some(preparation), Some(before)) =
            (&mut self.online_attention_pair_preparation, registered)
        {
            preparation.refresh_decorated_operation(
                node,
                (&before, &self.kernels[start..]),
                &suffix,
                &logical,
            )?;
        }
        self.info.transposed_bf16_output_programs += 1;
        Ok(logical)
    }
}

#[cfg(test)]
mod tests;
