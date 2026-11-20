//! Constructor-local alias of one closed contiguous BF16 bias reshape.
//! No reduction IR, numerical policy, producer layout or global graph is changed.
use super::*;
use crate::{
    tensor::ReduceOp,
    tile::{DType, ReductionSchedule, VirtualTensor},
};
use petgraph::Direction;

#[derive(Clone, Copy, Debug)]
pub(super) struct Candidate {
    pub step: usize,
    pub source: NodeIndex,
    pub output: NodeIndex,
    pub len: usize,
}
fn extent(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |n, &d| if d > 0 { n.checked_mul(d) } else { None })
}
/// Count only the existing physical identity-copy branch, not all view nodes.
pub(super) fn requests<D: crate::tile::TileDType>(
    graph: &TensorGraph<D>,
    plan: &crate::ptx::PtxExecutionPlan,
) -> usize {
    plan.steps()
        .iter()
        .filter(|s| {
            s.action == PtxPlanAction::Kernel
                && matches!(
                    graph[s.node],
                    TensorGraphNode::Reshape { .. } | TensorGraphNode::Flatten { .. }
                )
                && s.materialized_view.is_none()
        })
        .count()
}
pub(super) fn prove<D: crate::tile::TileDType>(
    graph: &TensorGraph<D>,
    plan: &crate::ptx::PtxExecutionPlan,
    dtype: DType,
    external_root: bool,
) -> Result<Candidate> {
    ensure!(
        dtype == DType::BF16 && !external_root,
        "requires BF16 and an internally allocated reduction root"
    );
    let active = plan
        .steps()
        .iter()
        .enumerate()
        .filter(|(_, s)| !matches!(s.action, PtxPlanAction::Upload | PtxPlanAction::VirtualView))
        .collect::<Vec<_>>();
    let [(copy_index, copy), (_, reduction)] = active.as_slice() else {
        anyhow::bail!("requires exactly copy then Sum physical steps");
    };
    ensure!(
        copy.action == PtxPlanAction::Kernel
            && matches!(
                graph[copy.node],
                TensorGraphNode::Reshape { .. } | TensorGraphNode::Flatten { .. }
            )
            && copy.materialized_view.is_none()
            && copy.inputs.len() == 1
            && copy.outputs == [copy.node]
            && copy.output_shapes == [copy.shape.clone()],
        "requires a whole contiguous reshape copy"
    );
    let root = plan.graph_output().context("bias reduction root missing")?;
    let PtxPlanAction::ReductionRegion(id) = reduction.action else {
        anyhow::bail!("requires selected typed reduction region");
    };
    let r = plan
        .reduction_regions()
        .get(id)
        .context("selected reduction absent")?;
    let [rows, cols] = r.input_shape.as_slice() else {
        anyhow::bail!("requires rank-two bias reduction");
    };
    ensure!(
        *rows > 0
            && *cols > 0
            && r.axis == 0
            && r.op == ReduceOp::Sum
            && r.schedule == (ReductionSchedule::Block { threads: 128 })
            && r.output_shape == [1, *cols]
            && r.producer_operations.is_empty()
            && r.epilogue_operations.is_empty()
            && r.broadcast_members.is_empty()
            && r.full_epilogue_operations.is_empty()
            && r.inputs.len() == 1
            && r.outputs.len() == 1
            && r.members == [reduction.node]
            && r.outputs[0].node == root
            && r.outputs[0].value == r.reduced_value
            && r.reduction_input == r.inputs[0].value
            && r.output_shapes == [vec![1, *cols]]
            && reduction.node == root
            && reduction.inputs == [copy.node]
            && reduction.outputs == [root]
            && reduction.output_shapes == [vec![1, *cols]]
            && matches!(
                graph[root],
                TensorGraphNode::ReduceAxis {
                    op: ReduceOp::Sum,
                    axis: 0,
                    ..
                }
            ),
        "requires one closed typed Block128 Sum with reduced-only output"
    );
    ensure!(
        copy.shape == r.input_shape
            && r.inputs[0].source_shape == r.input_shape
            && r.inputs[0].tensor.shape == r.input_shape
            && r.inputs[0].domain == crate::tile::ReductionInputDomain::Full
            && r.inputs[0].tensor.source == copy.node
            && r.inputs[0].tensor.dtype == DType::BF16
            && r.inputs[0].tensor.predicate.is_none()
            && r.inputs[0].tensor.out_of_bounds.is_none(),
        "reduction must read the complete physical reshape result"
    );
    let expected = VirtualTensor::identity(copy.node, copy.shape.clone(), DType::BF16);
    ensure!(
        r.inputs[0]
            .tensor
            .access
            .clone()
            .normalize_in_domain(&copy.shape)?
            == expected.access.normalize_in_domain(&copy.shape)?,
        "reduction input map is not physical identity"
    );
    let source = copy.inputs[0];
    let len = extent(&copy.shape).context("invalid reshape extent")?;
    ensure!(
        extent(graph[source].shape()) == Some(len),
        "reshape source volume differs"
    );
    let consumers = graph
        .graph
        .neighbors_directed(copy.node, Direction::Outgoing)
        .collect::<Vec<_>>();
    ensure!(consumers == [root], "reshape output escapes the single Sum");
    // The selected planner proves contiguity. Independently reject arbitrary
    // source maps: only whole-volume reshape chains above an uploaded source.
    let mut ancestor = source;
    loop {
        match &graph[ancestor] {
            TensorGraphNode::Input { .. }
            | TensorGraphNode::Constant { .. }
            | TensorGraphNode::Parameter { .. } => break,
            TensorGraphNode::Reshape { .. } | TensorGraphNode::Flatten { .. } => {
                let inputs = graph.inputs(ancestor);
                ensure!(
                    inputs.len() == 1 && extent(graph[inputs[0]].shape()) == Some(len),
                    "invalid source reshape chain"
                );
                ancestor = inputs[0];
            }
            _ => anyhow::bail!("source is not a whole contiguous upload/reshape chain"),
        }
    }
    ensure!(
        plan.steps().iter().all(|s| matches!(
            s.action,
            PtxPlanAction::Upload | PtxPlanAction::VirtualView
        ) || s.node == copy.node
            || s.node == root),
        "unexpected writable physical step"
    );
    for n in graph.graph.node_indices() {
        if graph
            .graph
            .neighbors_directed(n, Direction::Outgoing)
            .next()
            .is_none()
        {
            ensure!(n == root, "additional observable graph output");
        }
    }
    Ok(Candidate {
        step: *copy_index,
        source,
        output: copy.node,
        len,
    })
}
impl Candidate {
    pub(super) fn binding(self, values: &HashMap<NodeIndex, Binding>) -> Result<Binding> {
        let b = *values
            .get(&self.source)
            .context("alias source binding absent")?;
        ensure!(
            b.dtype == DType::BF16
                && b.len == self.len
                && b.pointer != 0
                && b.pointer.is_multiple_of(16),
            "requires whole aligned BF16 source binding"
        );
        b.pointer
            .checked_add(u64::try_from(
                b.len.checked_mul(2).context("alias byte extent overflow")?,
            )?)
            .context("alias pointer overflow")?;
        Ok(b)
    }
}

#[cfg(test)]
pub(super) fn record<D: crate::tile::TileDType>(
    graph: &TensorGraph<D>,
    plan: &crate::ptx::PtxExecutionPlan,
    values: &HashMap<NodeIndex, Binding>,
    requested: usize,
    selected: usize,
    reason: &str,
) -> Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let Some(dir) = std::env::var_os("TNSR_CONTIGUOUS_BIAS_RESHAPE_AUDIT") else {
        return Ok(());
    };
    if requested == 0 {
        return Ok(());
    }
    let path = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&path)?;
    let steps = plan.steps().iter().enumerate().map(|(i,s)|serde_json::json!({"index":i,"action":format!("{:?}",s.action),"operation":s.operation,"shape":s.shape,"inputs":s.inputs.iter().map(|n|serde_json::json!({"node":n.index(),"shape":graph[*n].shape(),"binding":values.get(n).map(|b|serde_json::json!({"dtype":format!("{:?}",b.dtype),"len":b.len,"mod16":b.pointer%16}))})).collect::<Vec<_>>(),"outputs":s.outputs.iter().map(|n|n.index()).collect::<Vec<_>>(),"materialized_view":format!("{:?}",s.materialized_view)})).collect::<Vec<_>>();
    std::fs::write(
        path.join(format!(
            "program-{}-{}.json",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )),
        serde_json::to_vec_pretty(
            &serde_json::json!({"requested_steps":requested,"selected_alias_steps":selected,"reason":reason,"steps":steps,"reductions":format!("{:#?}",plan.reduction_regions()),"scope":"Original selected plan and actual bindings; no qualification or performance claim."}),
        )?,
    )?;
    Ok(())
}
#[cfg(test)]
mod tests;
