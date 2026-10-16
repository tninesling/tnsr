//! Validate and cache the structure of nondifferentiable integer source slots.
use super::*;
use crate::tensor::indices::{IndexSource, checked_index, element_count};
use anyhow::{Context, Result, ensure};

#[derive(Default)]
pub(crate) struct IndexPlan {
    sources: Vec<IndexSourcePlan>,
}

#[derive(Clone, Copy)]
struct IndexSourcePlan {
    node: NodeIndex,
    len: usize,
    bound: Option<(usize, &'static str)>,
}

impl IndexPlan {
    pub fn new<D, G>(graph: &TensorGraph<D, G>) -> Result<Self> {
        let nodes: Vec<_> = graph
            .graph
            .node_indices()
            .filter(|&node| matches!(graph[node], TensorGraphNode::IndexSource { .. }))
            .collect();
        if nodes.is_empty() {
            return Ok(Self::default());
        }
        let float_names: HashSet<_> = graph
            .graph
            .node_weights()
            .filter_map(|node| {
                if let TensorGraphNode::Input { name, .. } = node {
                    Some(*name)
                } else {
                    None
                }
            })
            .collect();
        let mut sources = Vec::with_capacity(nodes.len());
        let mut input_names = HashSet::new();
        for node in nodes {
            let TensorGraphNode::IndexSource { source } = &graph[node] else {
                unreachable!()
            };
            if let IndexSource::Input(name) = source.source() {
                ensure!(
                    input_names.insert(*name),
                    "integer input '{name}' has inconsistent or duplicate source declarations"
                );
                ensure!(
                    !float_names.contains(name),
                    "input '{name}' has both integer and floating types"
                );
            }
            let len = element_count(source.shape())?;
            let mut bound: Option<(usize, &'static str)> = None;
            for edge in graph
                .graph
                .edges_directed(node, petgraph::Direction::Outgoing)
            {
                let consumer = edge.target();
                let slot = *edge.weight();
                let upper = match (&graph[consumer], slot) {
                    (TensorGraphNode::Embedding { .. }, 1) => *graph[graph.inputs(consumer)[0]]
                        .shape()
                        .first()
                        .context("embedding table shape is empty")?,
                    (TensorGraphNode::EmbeddingBackward { shape }, 0) => {
                        *shape.first().context("embedding gradient shape is empty")?
                    }
                    (TensorGraphNode::IndexedCrossEntropy { .. }, 1) => *graph
                        [graph.inputs(consumer)[0]]
                        .shape()
                        .last()
                        .context("logits shape is empty")?,
                    (TensorGraphNode::IndexedCrossEntropyBackward { shape }, 1) => {
                        *shape.last().context("logits gradient shape is empty")?
                    }
                    _ => anyhow::bail!(
                        "integer indices cannot feed {} argument {slot}",
                        graph[consumer].name()
                    ),
                };
                if bound.is_none_or(|(previous, _)| upper < previous) {
                    bound = Some((upper, graph[consumer].name()));
                }
            }
            sources.push(IndexSourcePlan { node, len, bound });
        }
        Ok(Self { sources })
    }

    /// Every feed is validated before any buffer is overwritten or graph launched.
    /// Compilation signatures guard node IDs, source kinds/names, shapes and edges;
    /// constant payloads are read from the current graph on every call.
    pub fn prepare<D, G>(
        &self,
        graph: &TensorGraph<D, G>,
        mut feeds: crate::tensor::IndexInputs,
    ) -> Result<HashMap<NodeIndex, Arc<Vec<u32>>>> {
        let mut values = HashMap::with_capacity(self.sources.len());
        for &IndexSourcePlan { node, len, bound } in &self.sources {
            let Some(TensorGraphNode::IndexSource { source }) = graph.graph.node_weight(node)
            else {
                anyhow::bail!("integer source structure differs from compilation");
            };
            let data = match source.source() {
                IndexSource::Input(name) => Arc::new(
                    feeds
                        .remove(*name)
                        .with_context(|| format!("integer input '{name}' not found"))?,
                ),
                IndexSource::Constant(data) => data.clone(),
            };
            ensure!(
                data.len() == len,
                "integer index source length does not match shape {:?}",
                source.shape()
            );
            if let (Some(value), Some((upper, operation))) = (data.iter().copied().max(), bound) {
                checked_index(value, upper, operation)?;
            }
            values.insert(node, data);
        }
        Ok(values)
    }
}

impl<D, G> TensorGraph<D, G> {
    pub(crate) fn index_sources(
        &self,
        feeds: crate::tensor::IndexInputs,
    ) -> Result<HashMap<NodeIndex, Arc<Vec<u32>>>> {
        IndexPlan::new(self)?.prepare(self, feeds)
    }
}
