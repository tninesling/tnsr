//! Fixed, explicit mixed-BF16 decoder topology shared by the benchmark/tests.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::Path,
};
use tnsr::{
    graph::{NodeIndex, TensorGraph, WithGrad},
    nn,
    parameters::NamedParameters,
    tensor::{IndexInputs, IndexTensor, Parameter, Shape, TensorExpr, ValueDType},
};

type Tensor = TensorExpr<f32>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
    pub hidden: usize,
    pub sequence: usize,
    pub batch: usize,
    pub vocab: usize,
    pub epsilon: f32,
    pub learning_rate: f32,
    pub max_grad_norm: f32,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.width > 0 && self.width <= 8192 && self.layers > 0 && self.layers <= 128,
            "invalid decoder width/layers"
        );
        ensure!(
            self.heads > 0 && self.width.is_multiple_of(self.heads),
            "heads must divide width"
        );
        ensure!(
            self.hidden > 0
                && self.hidden <= 65536
                && self.sequence > 0
                && self.sequence <= 4096
                && self.batch > 0,
            "invalid hidden width, sequence, or batch"
        );
        ensure!(
            self.vocab > 1 && u32::try_from(self.vocab).is_ok(),
            "invalid vocabulary"
        );
        self.rows()?
            .checked_mul(17)
            .and_then(|value| value.checked_add(3))
            .context("input index arithmetic overflows usize")?;
        ensure!(
            self.epsilon.is_finite()
                && self.epsilon > 0.0
                && self.learning_rate.is_finite()
                && self.learning_rate > 0.0
                && self.max_grad_norm.is_finite()
                && self.max_grad_norm > 0.0,
            "invalid normalization/optimizer settings"
        );
        Ok(())
    }
    pub fn rows(&self) -> Result<usize> {
        self.batch
            .checked_mul(self.sequence)
            .context("token row count overflows usize")
    }
    pub fn feeds(&self) -> Result<IndexInputs> {
        let rows = self.rows()?;
        let tokens = (0..rows)
            .map(|j| u32::try_from((17 * j + 3) % self.vocab))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let targets = (0..rows)
            .map(|j| u32::try_from((17 * (j + 1) + 3) % self.vocab))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(HashMap::from([
            ("tokens".into(), tokens),
            ("targets".into(), targets),
        ]))
    }
    pub fn specs(&self) -> BTreeMap<String, Shape> {
        let w = self.width;
        let mut specs = BTreeMap::from([
            ("token.weight".into(), vec![self.vocab, w]),
            ("position.weight".into(), vec![self.sequence, w]),
            ("final_norm.weight".into(), vec![w]),
            ("final_norm.bias".into(), vec![w]),
        ]);
        for layer in 0..self.layers {
            let p = format!("layer.{layer}");
            for norm in ["attn_norm", "ffn_norm"] {
                for kind in ["weight", "bias"] {
                    specs.insert(format!("{p}.{norm}.{kind}"), vec![w]);
                }
            }
            for (name, input, output) in [
                ("qkv", w, 3 * w),
                ("out", w, w),
                ("gate", w, self.hidden),
                ("up", w, self.hidden),
                ("down", self.hidden, w),
            ] {
                specs.insert(format!("{p}.{name}.weight"), vec![output, input]);
                specs.insert(format!("{p}.{name}.bias"), vec![output]);
            }
        }
        specs
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorRecord {
    pub name: String,
    pub shape: Shape,
    pub offset: usize,
    pub count: usize,
    pub group: String,
}

pub struct Model {
    pub config: Config,
    pub parameters: NamedParameters,
    /// Original archive order, independent of NamedParameters' map iteration.
    pub records: Vec<TensorRecord>,
}

impl Model {
    pub fn load(directory: &Path) -> Result<Self> {
        let (parameters, config) = NamedParameters::load(directory)?;
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("manifest.json"))?)?;
        let records = serde_json::from_value(
            manifest
                .get("tensors")
                .context("archive tensors missing")?
                .clone(),
        )?;
        let model = Self {
            config: serde_json::from_value(config)?,
            parameters,
            records,
        };
        model.validate()?;
        Ok(model)
    }
    pub fn validate(&self) -> Result<()> {
        self.config.validate()?;
        let specs = self.config.specs();
        self.parameters
            .validate_names(&specs.keys().cloned().collect::<HashSet<_>>())?;
        for (name, shape) in &specs {
            self.parameters.get(name, shape)?;
        }
        let mut offset = 0;
        let mut seen = HashSet::new();
        for record in &self.records {
            let parameter = self.parameters.get(&record.name, &record.shape)?;
            let count = record
                .shape
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))
                .context("parameter extent overflow")?;
            ensure!(
                seen.insert(record.name.clone())
                    && record.offset == offset
                    && record.count == count
                    && record.group == "train",
                "invalid archive parameter order/group for {}",
                record.name
            );
            let entry = self
                .parameters
                .iter()
                .find(|(name, _)| *name == record.name)
                .context("archive parameter missing")?
                .1;
            ensure!(
                entry.group == "train" && parameter.id() == entry.parameter.id(),
                "wrong training group"
            );
            offset = offset
                .checked_add(count)
                .context("parameter total overflow")?;
        }
        ensure!(seen.len() == specs.len(), "archive record count mismatch");
        Ok(())
    }
    pub fn parameter_count(&self) -> usize {
        self.records.iter().map(|r| r.count).sum()
    }
    fn p(&self, name: &str, shape: &[usize]) -> Result<Parameter<f32>> {
        self.parameters.get(name, shape)
    }
    fn norm(&self, x: Tensor, prefix: &str) -> Result<Tensor> {
        Ok(nn::layer_norm(
            x.cast(ValueDType::F32),
            2,
            self.p(&format!("{prefix}.weight"), &[self.config.width])?,
            self.p(&format!("{prefix}.bias"), &[self.config.width])?,
            self.config.epsilon,
        ))
    }
    fn linear(&self, x: Tensor, prefix: &str, input: usize, output: usize) -> Result<Tensor> {
        let c = &self.config;
        Ok(x.reshape(vec![c.rows()?, input])
            .linear_bf16(
                Tensor::from(self.p(&format!("{prefix}.weight"), &[output, input])?).transpose(),
                self.p(&format!("{prefix}.bias"), &[output])?.into(),
            )?
            .reshape(vec![c.batch, c.sequence, output]))
    }
    pub fn loss(&self) -> Result<Tensor> {
        let c = &self.config;
        let (w, b, s, h, d) = (c.width, c.batch, c.sequence, c.heads, c.width / c.heads);
        let token = self.p("token.weight", &[c.vocab, w])?;
        let positions = IndexTensor::constant(
            (0..s)
                .map(u32::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?,
            vec![1, s],
        )?;
        let mut x = Tensor::from(token.clone())
            .embedding_indices(IndexTensor::input("tokens", vec![b, s]))
            + Tensor::from(self.p("position.weight", &[s, w])?).embedding_indices(positions);
        let mask = Tensor::constant(
            (0..s)
                .flat_map(|row| {
                    (0..s).map(move |col| if col > row { f32::NEG_INFINITY } else { 0.0 })
                })
                .collect(),
            vec![1, 1, s, s],
        );
        for layer in 0..c.layers {
            let p = format!("layer.{layer}");
            let normalized = self.norm(x.clone(), &format!("{p}.attn_norm"))?;
            let qkv = self.linear(normalized, &format!("{p}.qkv"), w, 3 * w)?;
            let split = |offset| {
                qkv.clone()
                    .narrow(2, offset, w)
                    .reshape(vec![b, s, h, d])
                    .permute(vec![0, 2, 1, 3])
            };
            let (q, k, v) = (split(0), split(w), split(2 * w));
            let scores = q.matmul(k.transpose()).cast(ValueDType::F32) * (d as f32).sqrt().recip()
                + mask.clone();
            let probabilities = nn::softmax(scores, 3).cast(ValueDType::BF16);
            let context = probabilities
                .matmul(v)
                .permute(vec![0, 2, 1, 3])
                .reshape(vec![b, s, w]);
            x = x + self
                .linear(context, &format!("{p}.out"), w, w)?
                .cast(ValueDType::F32);
            let normalized = self.norm(x.clone(), &format!("{p}.ffn_norm"))?;
            let gate = nn::silu(
                self.linear(normalized.clone(), &format!("{p}.gate"), w, c.hidden)?
                    .cast(ValueDType::F32),
            )
            .cast(ValueDType::BF16);
            let up = self.linear(normalized, &format!("{p}.up"), w, c.hidden)?;
            let hidden = gate * up;
            x = x + self
                .linear(hidden, &format!("{p}.down"), c.hidden, w)?
                .cast(ValueDType::F32);
        }
        let final_hidden = self.norm(x, "final_norm")?.reshape(vec![c.rows()?, w]);
        let logits = final_hidden
            .linear_bf16(
                Tensor::from(token).transpose(),
                Tensor::constant(vec![0.0; c.vocab], vec![c.vocab]),
            )?
            .cast(ValueDType::F32);
        Ok(logits
            .indexed_cross_entropy_indices(IndexTensor::input("targets", vec![c.rows()?]))
            .mean_all())
    }
    pub fn training_graph(&self) -> Result<(TensorGraph<f32, WithGrad>, NodeIndex)> {
        let graph: TensorGraph<f32> = self.loss()?.into();
        let output = graph
            .toposort()
            .last()
            .copied()
            .context("empty decoder graph")?;
        Ok((graph.with_gradients(output), output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tnsr::{Executor, SimpleExecutor, graph::TensorGraphNode};
    fn tiny() -> Model {
        let config = Config {
            width: 4,
            layers: 1,
            heads: 2,
            hidden: 7,
            sequence: 3,
            batch: 2,
            vocab: 11,
            epsilon: 1e-5,
            learning_rate: 1e-4,
            max_grad_norm: 1.0,
        };
        let mut parameters = NamedParameters::default();
        let mut records = Vec::new();
        let mut offset = 0;
        for (name, shape) in config.specs() {
            let count = shape.iter().product();
            let values = (0..count)
                .map(|i| {
                    if name.ends_with("norm.weight") {
                        1.0
                    } else if name.ends_with(".bias") {
                        0.0
                    } else {
                        ((i * 7 % 17) as f32 - 8.0) * 0.015
                    }
                })
                .collect();
            parameters
                .insert(name.clone(), values, shape.clone(), "train".into())
                .unwrap();
            records.push(TensorRecord {
                name,
                shape,
                offset,
                count,
                group: "train".into(),
            });
            offset += count;
        }
        Model {
            config,
            parameters,
            records,
        }
    }
    #[test]
    fn tiny_decoder_has_tied_head_all_parameter_gradients_and_exact_inputs() {
        let model = tiny();
        model.validate().unwrap();
        let (graph, loss) = model.training_graph().unwrap();
        let ids = graph
            .graph
            .node_weights()
            .filter_map(|node| {
                if let TensorGraphNode::Parameter { id, .. } = node {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect::<HashSet<_>>();
        assert_eq!(ids.len(), model.records.len());
        assert_eq!(
            graph.gradient_metadata().param_to_grad.len(),
            model.records.len()
        );
        assert_eq!(graph[loss].shape(), &vec![1]);
        let plain: TensorGraph<f32> = model.loss().unwrap().into();
        assert_eq!(
            plain
                .graph
                .node_weights()
                .filter(|node| matches!(node, TensorGraphNode::Linear { .. }))
                .count(),
            6
        );
        assert!(
            !plain
                .graph
                .node_weights()
                .any(|node| matches!(node, TensorGraphNode::Input { .. }))
        );
        let mut executor = SimpleExecutor::<f32>::new();
        executor
            .execute_with_indices(&graph, HashMap::new(), model.config.feeds().unwrap())
            .unwrap();
        let loss = executor.get_value(loss).unwrap()[0];
        assert!(loss.is_finite() && loss > 0.0);
        let gradients = executor.get_gradients(&graph);
        assert_eq!(gradients.len(), model.records.len());
        assert!(gradients.values().all(|g| g.iter().all(|x| x.is_finite())));
        // CPU casts intentionally preserve f32 values: this verifies topology,
        // indexing, and differentiation, not mixed-storage numerical parity.
    }
    #[test]
    fn deterministic_feeds_and_shapes_are_checked() {
        let model = tiny();
        let feeds = model.config.feeds().unwrap();
        assert_eq!(feeds["tokens"], vec![3, 9, 4, 10, 5, 0]);
        assert_eq!(feeds["targets"], vec![9, 4, 10, 5, 0, 6]);
        let mut config = model.config.clone();
        config.heads = 3;
        assert!(config.validate().is_err());
        config = model.config.clone();
        config.epsilon = f32::NAN;
        assert!(config.validate().is_err());
    }
}
