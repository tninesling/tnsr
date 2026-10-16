//! Exact lookup indices, independent of floating-point tensor storage.
use super::{DType, ExprKind, ExprNode, Shape, TensorExpr};
use anyhow::{Context, Result, ensure};
use std::collections::HashMap;
use std::sync::Arc;

/// Named integer feeds for embedding and indexed-loss operations.
pub type IndexInputs = HashMap<String, Vec<u32>>;

#[derive(Clone, Debug)]
pub(crate) enum IndexSource {
    Input(&'static str),
    Constant(Arc<Vec<u32>>),
}

#[derive(Debug)]
struct IndexNode {
    shape: Shape,
    source: IndexSource,
}

/// An integer input or constant. Indices never pass through floating storage,
/// autocast or autograd; use `embedding_indices` / `indexed_cross_entropy_indices`.
#[derive(Clone, Debug)]
pub struct IndexTensor(Arc<IndexNode>);

impl IndexTensor {
    pub fn input(name: &'static str, shape: Shape) -> Self {
        Self(Arc::new(IndexNode {
            shape,
            source: IndexSource::Input(name),
        }))
    }

    pub fn constant(values: Vec<u32>, shape: Shape) -> Result<Self> {
        ensure!(
            values.len() == element_count(&shape)?,
            "integer constant length does not match shape {shape:?}"
        );
        Ok(Self(Arc::new(IndexNode {
            shape,
            source: IndexSource::Constant(Arc::new(values)),
        })))
    }

    /// Checked conversion of signed host indices; rejects negatives/overflow.
    pub fn from_i64(values: Vec<i64>, shape: Shape) -> Result<Self> {
        Self::constant(checked_indices(&values)?, shape)
    }

    pub fn shape(&self) -> &Shape {
        &self.0.shape
    }

    pub(crate) fn source(&self) -> &IndexSource {
        &self.0.source
    }

    pub(crate) fn same_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || matches!((self.source(), other.source()),
                (IndexSource::Input(a), IndexSource::Input(b)) if a == b && self.shape() == other.shape())
    }

    pub(crate) fn expression<D: DType>(self) -> TensorExpr<D> {
        TensorExpr(Arc::new(ExprNode {
            shape: self.shape().clone(),
            kind: ExprKind::Index { source: self },
        }))
    }
}

/// Convert signed host feeds without narrowing through any floating type.
pub fn checked_indices(values: &[i64]) -> Result<Vec<u32>> {
    values
        .iter()
        .map(|&value| {
            u32::try_from(value)
                .with_context(|| format!("index {value} cannot be represented as u32"))
        })
        .collect()
}

pub(crate) fn element_count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, &dim| {
        count
            .checked_mul(dim)
            .context("integer index shape overflows usize")
    })
}

pub(crate) fn checked_index(value: u32, upper: usize, operation: &str) -> Result<usize> {
    let index = usize::try_from(value).context("index exceeds usize")?;
    ensure!(
        index < upper,
        "{operation} index {index} is out of range for size {upper}"
    );
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{TensorGraph, TensorGraphNode};
    use crate::tensor::Parameter;
    use crate::{Executor, SimpleExecutor};

    #[test]
    fn exact_index_construction_rejects_narrowing_and_bad_shapes() {
        assert_eq!(
            checked_indices(&[257, 16_777_217, i64::from(u32::MAX)]).unwrap(),
            vec![257, 16_777_217, u32::MAX]
        );
        assert!(checked_indices(&[-1]).is_err());
        assert!(checked_indices(&[i64::from(u32::MAX) + 1]).is_err());
        assert!(IndexTensor::constant(vec![1], vec![2]).is_err());
        assert!(IndexTensor::constant(vec![], vec![usize::MAX, 2]).is_err());
        assert!(IndexTensor::constant(vec![], vec![0]).is_ok());
    }

    #[test]
    fn exact_index_inputs_deduplicate_and_accumulate_shared_lookup_gradients() {
        let weight = Parameter::new(vec![0.0f32; 683], vec![683, 1]);
        let a =
            TensorExpr::from(weight.clone()).embedding_indices(IndexTensor::input("ids", vec![3]));
        let b =
            TensorExpr::from(weight.clone()).embedding_indices(IndexTensor::input("ids", vec![3]));
        let graph: TensorGraph<f32> = (a + b).reduce_sum(1).reduce_sum(0).into();
        let output = *graph.toposort().last().unwrap();
        let forward_count = graph.len();
        let graph = graph.with_gradients(output);
        assert_eq!(
            graph
                .graph
                .node_weights()
                .filter(|n| matches!(n, TensorGraphNode::IndexSource { .. }))
                .count(),
            1
        );
        assert_eq!(graph.gradient_metadata().param_to_grad.len(), 1);
        let mut cpu = SimpleExecutor::new();
        cpu.execute_with_indices(
            &graph,
            HashMap::new(),
            HashMap::from([("ids".into(), vec![257, 365, 365])]),
        )
        .unwrap();
        let gradient = &cpu.get_gradients(&graph)[&weight.id()];
        assert_eq!(gradient[257], 2.0);
        assert_eq!(gradient[365], 4.0);
        assert_eq!(gradient.iter().sum::<f32>(), 6.0);
        assert_eq!(graph.without_gradients().len(), forward_count);
    }

    #[test]
    fn exact_index_input_errors_recover_without_rounding_ids() {
        let graph: TensorGraph<f32> =
            TensorExpr::constant((0..683).map(|i| i as f32).collect(), vec![683, 1])
                .embedding_indices(IndexTensor::input("ids", vec![1]))
                .into();
        let mut cpu = SimpleExecutor::new();
        assert!(
            cpu.execute(&graph, HashMap::new())
                .unwrap_err()
                .to_string()
                .contains("integer input")
        );
        for id in [683, 16_777_217, u32::MAX] {
            let error = cpu
                .execute_with_indices(
                    &graph,
                    HashMap::new(),
                    HashMap::from([("ids".into(), vec![id])]),
                )
                .unwrap_err();
            assert!(error.to_string().contains(&id.to_string()), "{error:#}");
        }
        assert!(
            cpu.execute_with_indices(
                &graph,
                HashMap::new(),
                HashMap::from([("ids".into(), vec![1, 2])])
            )
            .is_err()
        );
        let actual = cpu
            .execute_with_indices(
                &graph,
                HashMap::new(),
                HashMap::from([("ids".into(), vec![365])]),
            )
            .unwrap();
        assert_eq!(actual, vec![365.0]);
    }

    #[test]
    fn exact_targets_match_floating_cross_entropy_values_and_gradients() {
        let weight = Parameter::new(vec![0.0f32; 2 * 683], vec![2, 683]);
        let build = |exact| {
            let logits = TensorExpr::from(weight.clone());
            let loss = if exact {
                logits.indexed_cross_entropy_indices(
                    IndexTensor::constant(vec![257, 365], vec![2]).unwrap(),
                )
            } else {
                logits.indexed_cross_entropy(TensorExpr::constant(vec![257.0, 365.0], vec![2]))
            }
            .mean_all();
            let graph: TensorGraph<f32> = loss.into();
            let output = *graph.toposort().last().unwrap();
            (graph.with_gradients(output), output)
        };
        let (exact, output) = build(true);
        let (floating, reference_output) = build(false);
        let mut cpu = SimpleExecutor::new();
        cpu.execute(&exact, HashMap::new()).unwrap();
        let actual = cpu.get_value(output).unwrap();
        let gradients = cpu.get_gradients(&exact);
        cpu.execute(&floating, HashMap::new()).unwrap();
        assert_eq!(actual, cpu.get_value(reference_output).unwrap());
        assert_eq!(gradients, cpu.get_gradients(&floating));
    }

    #[test]
    fn exact_index_scalar_empty_and_invalid_structure() {
        let scalar: TensorGraph<f32> = TensorExpr::constant(vec![2.0, 3.0], vec![2, 1])
            .embedding_indices(IndexTensor::constant(vec![1], vec![]).unwrap())
            .into();
        let mut cpu = SimpleExecutor::new();
        assert_eq!(cpu.execute(&scalar, HashMap::new()).unwrap(), vec![3.0]);
        let empty: TensorGraph<f32> = TensorExpr::constant(vec![], vec![0, 1])
            .embedding_indices(IndexTensor::input("ids", vec![0]))
            .into();
        assert!(
            cpu.execute_with_indices(
                &empty,
                HashMap::new(),
                HashMap::from([("ids".into(), vec![])])
            )
            .unwrap()
            .is_empty()
        );
        let overflow: TensorGraph<f32> = TensorExpr::constant(vec![1.0], vec![1, 1])
            .embedding_indices(IndexTensor::input("ids", vec![usize::MAX, 2]))
            .into();
        assert!(
            cpu.execute_with_indices(&overflow, HashMap::new(), HashMap::new())
                .unwrap_err()
                .to_string()
                .contains("overflows")
        );

        let lookup = TensorExpr::constant(vec![2.0f32, 3.0], vec![2, 1])
            .embedding_indices(IndexTensor::input("ids", vec![1]));
        let collision: TensorGraph<f32> =
            (lookup.clone() + TensorExpr::input("ids", vec![1, 1])).into();
        assert!(
            cpu.execute_with_indices(
                &collision,
                HashMap::new(),
                HashMap::from([("ids".into(), vec![1])])
            )
            .unwrap_err()
            .to_string()
            .contains("both integer and floating")
        );
        let mut invalid: TensorGraph<f32> = lookup.into();
        let source = invalid
            .graph
            .node_indices()
            .find(|&node| matches!(invalid[node], TensorGraphNode::IndexSource { .. }))
            .unwrap();
        let unary = invalid.graph.add_node(TensorGraphNode::Unary {
            op: super::super::UnaryOp::Exp,
            shape: vec![1],
        });
        invalid.graph.add_edge(source, unary, 0);
        assert!(
            cpu.execute_with_indices(
                &invalid,
                HashMap::new(),
                HashMap::from([("ids".into(), vec![1])])
            )
            .unwrap_err()
            .to_string()
            .contains("integer indices cannot feed")
        );
        let lookup = |n| {
            TensorExpr::constant(vec![2.0f32, 3.0], vec![2, 1])
                .embedding_indices(IndexTensor::input("ids", vec![n]))
                .mean_all()
        };
        let inconsistent: TensorGraph<f32> = (lookup(2) + lookup(3)).into();
        assert!(
            cpu.execute_with_indices(
                &inconsistent,
                HashMap::new(),
                HashMap::from([("ids".into(), vec![1, 0])])
            )
            .unwrap_err()
            .to_string()
            .contains("inconsistent or duplicate")
        );
    }
}
