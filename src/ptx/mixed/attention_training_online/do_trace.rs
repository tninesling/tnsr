//! Static source trace only. This is not an actual-device binding admission.
use crate::{
    graph::{NodeIndex, TensorGraph, TensorGraphNode},
    tensor::ValueDType,
    tile::{DType, VirtualTensor},
};
use anyhow::{Context, Result, ensure};

#[derive(Debug, serde::Serialize)]
pub(super) struct Step {
    pub node: usize,
    pub operation: &'static str,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub inputs: Vec<usize>,
}
#[derive(Debug, serde::Serialize)]
pub(super) struct Trace {
    pub qk: usize,
    pub pv: usize,
    pub old_dp: usize,
    pub old_dv: usize,
    pub incoming_do: usize,
    pub incoming_shape: Vec<usize>,
    pub ancestry: Vec<Step>,
    pub default_operation_producer: usize,
    pub default_operation_source_shape: Vec<usize>,
    pub scope: &'static str,
}

// Logical ancestry, not physical identity: casts are recorded and may
// materialize. Actual Value/source_shape/binding length must be inspected later.
fn view_from<G>(
    g: &TensorGraph<f32, G>,
    node: NodeIndex,
    source: NodeIndex,
    remaining: usize,
) -> Option<VirtualTensor> {
    if node == source {
        return Some(VirtualTensor::identity(
            source,
            g[source].shape().clone(),
            DType::BF16,
        ));
    }
    if remaining == 0 {
        return None;
    }
    let [input]: [NodeIndex; 1] = g.inputs(node).try_into().ok()?;
    let view = view_from(g, input, source, remaining - 1)?;
    match &g[node] {
        TensorGraphNode::Cast { shape, .. } if *shape == view.shape => Some(view),
        TensorGraphNode::Reshape { shape } | TensorGraphNode::Flatten { shape } => {
            view.reshape(shape.clone()).ok()
        }
        TensorGraphNode::Permute { axes, .. } => view.permute(axes).ok(),
        TensorGraphNode::Transpose { .. } => {
            let mut axes: Vec<usize> = (0..view.shape.len()).collect();
            if axes.len() < 2 {
                return None;
            }
            let n = axes.len();
            axes.swap(n - 2, n - 1);
            view.permute(&axes).ok()
        }
        _ => None,
    }
}
fn is_transpose_of<G>(g: &TensorGraph<f32, G>, node: NodeIndex, source: NodeIndex) -> bool {
    let Some(actual) = view_from(g, node, source, 24) else {
        return false;
    };
    let root = VirtualTensor::identity(source, g[source].shape().clone(), DType::BF16);
    let mut axes: Vec<_> = (0..root.shape.len()).collect();
    if axes.len() < 2 {
        return false;
    }
    let n = axes.len();
    axes.swap(n - 2, n - 1);
    let Ok(expected) = root.permute(&axes) else {
        return false;
    };
    if actual.shape != expected.shape {
        return false;
    }
    let (Ok(actual), Ok(expected)) = (
        actual.access.normalize_in_domain(&actual.shape),
        expected.access.normalize_in_domain(&expected.shape),
    ) else {
        return false;
    };
    actual == expected
}

pub(super) fn trace<G>(g: &TensorGraph<f32, G>) -> Result<Vec<Trace>> {
    let dtypes = crate::graph::mixed::value_dtypes(g);
    let mut result = Vec::new();
    for norm in crate::ptx::mixed::attention_softmax::recognize(g) {
        let [b, h, s, t]: [usize; 4] = norm
            .shape
            .as_slice()
            .try_into()
            .context("normalizer rank")?;
        if s != 512 || t != 512 || b.checked_mul(h).is_none_or(|bh| bh < 64) {
            continue;
        }
        let output_shape = vec![b, h, s, 64];
        let pv_candidates: Vec<_> = g
            .graph
            .node_indices()
            .filter(|&n| {
                matches!(g[n], TensorGraphNode::MatMul { .. })
                    && g[n].shape() == &output_shape
                    && g.inputs(n).first() == Some(&norm.narrowed)
            })
            .collect();
        let [pv] = pv_candidates.as_slice() else {
            continue;
        };
        let pv = *pv;
        let value = g.inputs(pv)[1];
        let dv_candidates: Vec<_> = g
            .graph
            .node_indices()
            .filter(|&n| {
                if !matches!(g[n], TensorGraphNode::MatMul { .. }) || g[n].shape() != &output_shape
                {
                    return false;
                }
                let inputs = g.inputs(n);
                inputs.len() == 2 && is_transpose_of(g, inputs[0], norm.narrowed)
            })
            .collect();
        ensure!(
            dv_candidates.len() == 1,
            "expected one dV contraction for PV{}",
            pv.index()
        );
        let dv = dv_candidates[0];
        let incoming = g.inputs(dv)[1];
        ensure!(
            g[incoming].shape() == &output_shape,
            "incoming dO shape mismatch"
        );
        ensure!(
            dtypes.get(&incoming) == Some(&ValueDType::BF16),
            "initial online pair requires existing BF16 dO, not a new cast"
        );
        let dp_candidates: Vec<_> = g
            .graph
            .node_indices()
            .filter(|&n| {
                if !matches!(g[n], TensorGraphNode::MatMul { .. }) || g[n].shape() != &norm.shape {
                    return false;
                }
                let inputs = g.inputs(n);
                inputs.len() == 2 && inputs[0] == incoming && is_transpose_of(g, inputs[1], value)
            })
            .collect();
        ensure!(
            dp_candidates.len() == 1,
            "expected one dP contraction for PV{}",
            pv.index()
        );
        let mut current = incoming;
        let mut ancestry = Vec::new();
        for _ in 0..32 {
            let inputs = g.inputs(current);
            ancestry.push(Step {
                node: current.index(),
                operation: g[current].name(),
                shape: g[current].shape().clone(),
                dtype: format!("{:?}", dtypes.get(&current)),
                inputs: inputs.iter().map(|x| x.index()).collect(),
            });
            let wrapper = matches!(
                g[current],
                TensorGraphNode::Cast { .. }
                    | TensorGraphNode::Reshape { .. }
                    | TensorGraphNode::Flatten { .. }
                    | TensorGraphNode::Permute { .. }
                    | TensorGraphNode::Transpose { .. }
            );
            if !wrapper || inputs.len() != 1 {
                break;
            }
            current = inputs[0];
        }
        result.push(Trace {qk:norm.qk.index(),pv:pv.index(),old_dp:dp_candidates[0].index(),old_dv:dv.index(),incoming_do:incoming.index(),incoming_shape:g[incoming].shape().clone(),ancestry,default_operation_producer:current.index(),default_operation_source_shape:g[current].shape().clone(),scope:"static graph/gradient-rule trace only; Mixed operation normally returns identity storage, but physical-output policies and prefix reuse require later actual Value observation"});
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn online_training_actual_do_static_source_trace() -> Result<()> {
        use crate::ptx::mixed::test_decoder::{Config, synthetic};
        let model = synthetic(Config {
            width: 512,
            layers: 4,
            heads: 8,
            hidden: 1408,
            sequence: 512,
            batch: 8,
            vocab: 4096,
            epsilon: 1e-5,
            learning_rate: 1e-4,
            max_grad_norm: 1.,
        });
        let (graph, _) = model.training_graph_with_options(crate::graph::GradientOptions {
            compact_softmax: true,
        })?;
        let traces = trace(&graph)?;
        if let Some(path) = std::env::var_os("TNSR_ONLINE_ATTENTION_AUDIT") {
            std::fs::create_dir_all(&path)?;
            std::fs::write(
                std::path::Path::new(&path).join("actual-do-static-trace.json"),
                serde_json::to_vec_pretty(&traces)?,
            )?;
        }
        assert_eq!(traces.len(), 4);
        for trace in traces {
            assert_eq!(trace.incoming_shape, vec![8, 8, 512, 64]);
            assert_eq!(trace.default_operation_source_shape, vec![4096, 512]);
            assert!(
                trace
                    .ancestry
                    .iter()
                    .any(|step| step.operation == "Permute")
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod gpu_tests;
