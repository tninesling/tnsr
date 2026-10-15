//! Explicit forty-five actor/critic model assembled from tnsr primitives.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use tnsr::nn;
use tnsr::parameters::NamedParameters;
use tnsr::tensor::{Parameter, Shape, TensorExpr};

type Tensor = TensorExpr<f32>;
type Specs = BTreeMap<String, (Shape, &'static str)>;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub token_dim: usize,
    pub token_heads: usize,
    pub token_layers: usize,
    pub token_max_tokens: usize,
    pub token_vocab_size: usize,
    pub token_type_vocab_size: usize,
    pub phase_embed_dim: usize,
    pub num_phases: usize,
    pub use_phase_heads: bool,
    pub use_card_conditioned_actions: bool,
    pub use_separate_critic: bool,
    pub use_privileged_critic: bool,
    pub use_rope: bool,
    pub use_swiglu: bool,
}
pub struct Model {
    pub config: Config,
    pub parameters: NamedParameters,
}
fn linear_specs(specs: &mut Specs, prefix: &str, input: usize, output: usize, group: &'static str) {
    specs.insert(format!("{prefix}.weight"), (vec![output, input], group));
    specs.insert(format!("{prefix}.bias"), (vec![output], group));
}
fn norm_specs(specs: &mut Specs, prefix: &str, width: usize, group: &'static str) {
    specs.insert(format!("{prefix}.weight"), (vec![width], group));
    specs.insert(format!("{prefix}.bias"), (vec![width], group));
}
impl Model {
    pub fn load(directory: &Path) -> Result<Self> {
        let (parameters, config) = NamedParameters::load(directory)?;
        let model = Self {
            parameters,
            config: serde_json::from_value(config)?,
        };
        model.validate()?;
        Ok(model)
    }
    pub fn validate(&self) -> Result<()> {
        let c = &self.config;
        ensure!(
            c.use_swiglu && !c.use_rope && !c.use_privileged_critic,
            "this port supports learned-position SwiGLU encoders and legal-information critics"
        );
        ensure!(
            c.token_dim > 0
                && c.token_dim <= 4096
                && c.token_heads > 0
                && c.token_dim.is_multiple_of(c.token_heads),
            "invalid model width/heads"
        );
        ensure!(
            c.token_layers > 0
                && c.token_layers <= 128
                && c.token_max_tokens > 0
                && c.token_max_tokens <= 8192,
            "invalid model layers/sequence"
        );
        ensure!(
            !c.use_card_conditioned_actions || (c.use_phase_heads && c.token_max_tokens >= 15),
            "card heads require phase heads and 15 token positions"
        );
        ensure!(
            c.phase_embed_dim > 0 && c.phase_embed_dim <= 4096 && c.num_phases > 0,
            "invalid phase embedding"
        );
        ensure!(
            c.token_vocab_size > 0
                && c.token_vocab_size <= 1 << 24
                && c.token_type_vocab_size > 0
                && c.token_type_vocab_size <= 1 << 24
                && c.num_phases <= 1 << 24,
            "invalid embedding vocabulary"
        );
        let specs = self.specs();
        self.parameters
            .validate_names(&specs.keys().cloned().collect::<HashSet<_>>())?;
        for (name, (shape, group)) in &specs {
            self.parameters.get(name, shape)?;
            let actual = self
                .parameters
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, entry)| entry.group.as_str());
            ensure!(actual == Some(*group), "{name}: wrong optimizer group");
        }
        Ok(())
    }
    fn specs(&self) -> Specs {
        let c = &self.config;
        let d = c.token_dim;
        let phase = c.phase_embed_dim;
        let hidden = (8 * d).div_ceil(3);
        let mut specs = Specs::new();
        specs.insert(
            "phase_embedding.weight".into(),
            (vec![c.num_phases, phase], "policy"),
        );
        for (prefix, group) in [("token_encoder", "policy"), ("critic_encoder", "value")] {
            if prefix == "critic_encoder" && !c.use_separate_critic {
                continue;
            }
            for (name, rows) in [
                ("token_embedding", c.token_vocab_size),
                ("token_type_embedding", c.token_type_vocab_size),
                ("position_embedding", c.token_max_tokens),
            ] {
                specs.insert(format!("{prefix}.{name}.weight"), (vec![rows, d], group));
            }
            for layer in 0..c.token_layers {
                let p = format!("{prefix}.encoder.layers.{layer}");
                norm_specs(&mut specs, &format!("{p}.attn_norm"), d, group);
                norm_specs(&mut specs, &format!("{p}.ffn_norm"), d, group);
                linear_specs(
                    &mut specs,
                    &format!("{p}.attention.qkv_proj"),
                    d,
                    3 * d,
                    group,
                );
                linear_specs(&mut specs, &format!("{p}.attention.out_proj"), d, d, group);
                for projection in ["gate_proj", "up_proj"] {
                    linear_specs(
                        &mut specs,
                        &format!("{p}.feed_forward.{projection}"),
                        d,
                        hidden,
                        group,
                    );
                }
                linear_specs(
                    &mut specs,
                    &format!("{p}.feed_forward.down_proj"),
                    hidden,
                    d,
                    group,
                );
            }
            norm_specs(&mut specs, &format!("{prefix}.output_norm"), d, group);
        }
        for (prefix, query, group) in [
            ("policy_pool", d + phase, "policy"),
            ("value_pool.pool", d, "value"),
        ] {
            norm_specs(&mut specs, &format!("{prefix}.query_norm"), query, group);
            linear_specs(&mut specs, &format!("{prefix}.query_proj"), query, d, group);
            for name in ["key_proj", "value_proj"] {
                linear_specs(&mut specs, &format!("{prefix}.{name}"), d, d, group);
            }
        }
        specs.insert("value_pool.query".into(), (vec![d], "value"));
        norm_specs(&mut specs, "policy_trunk.0", 2 * d + phase, "policy");
        linear_specs(&mut specs, "policy_trunk.1", 2 * d + phase, d, "policy");
        norm_specs(&mut specs, "policy_trunk.3", d, "policy");
        norm_specs(&mut specs, "value_trunk.0", 2 * d, "value");
        for (name, input, output) in [
            ("value_trunk.1", 2 * d, d),
            ("value_trunk.3", d, d / 2),
            ("value_trunk.5", d / 2, 1),
        ] {
            linear_specs(&mut specs, name, input, output, "value");
        }
        if c.use_phase_heads {
            linear_specs(&mut specs, "bid_head", d, 5, "policy");
            linear_specs(&mut specs, "suit_head", d, 4, "policy");
            if c.use_card_conditioned_actions {
                for name in ["play_query", "play_keys", "discard_cards"] {
                    linear_specs(
                        &mut specs,
                        &format!("card_action_heads.{name}"),
                        d,
                        d,
                        "policy",
                    );
                }
                linear_specs(&mut specs, "card_action_heads.play_bias", d, 1, "policy");
                specs.insert(
                    "card_action_heads.discard_count_embedding.weight".into(),
                    (vec![6, d], "policy"),
                );
                norm_specs(
                    &mut specs,
                    "card_action_heads.discard_trunk.0",
                    3 * d,
                    "policy",
                );
                linear_specs(
                    &mut specs,
                    "card_action_heads.discard_trunk.1",
                    3 * d,
                    d,
                    "policy",
                );
                linear_specs(
                    &mut specs,
                    "card_action_heads.discard_trunk.3",
                    d,
                    1,
                    "policy",
                );
            } else {
                linear_specs(&mut specs, "discard_head", d, 32, "policy");
                linear_specs(&mut specs, "play_head", d, 5, "policy");
            }
        } else {
            linear_specs(&mut specs, "policy_head", d, 46, "policy");
        }
        specs
    }
    fn p(&self, name: &str, shape: &[usize]) -> Result<Parameter<f32>> {
        self.parameters.get(name, shape)
    }
    fn linear(&self, prefix: &str, x: Tensor, input: usize, output: usize) -> Result<Tensor> {
        Ok(nn::linear(
            x,
            Tensor::from(self.p(&format!("{prefix}.weight"), &[output, input])?).transpose(),
            Some(self.p(&format!("{prefix}.bias"), &[output])?),
        ))
    }
    fn norm(&self, prefix: &str, x: Tensor, width: usize) -> Result<Tensor> {
        let axis = x.shape().len() - 1;
        Ok(nn::layer_norm(
            x,
            axis,
            self.p(&format!("{prefix}.weight"), &[width])?,
            self.p(&format!("{prefix}.bias"), &[width])?,
            1e-5,
        ))
    }
    fn encode(&self, prefix: &str, batch: usize) -> Result<Tensor> {
        let c = &self.config;
        let d = c.token_dim;
        let s = c.token_max_tokens;
        let heads = c.token_heads;
        let head = d / heads;
        let hidden = (8 * d).div_ceil(3);
        let tokens = Tensor::input("tokens", vec![batch, s]);
        let types = Tensor::input("token_types", vec![batch, s]);
        let mut positions: Vec<f32> = (0..s).map(|i| i as f32).collect();
        if c.use_card_conditioned_actions {
            positions[10..15].fill(10.0);
        }
        let positions = Tensor::constant(positions, vec![1, s]);
        let mut x = Tensor::from(self.p(
            &format!("{prefix}.token_embedding.weight"),
            &[c.token_vocab_size, d],
        )?)
        .embedding(tokens)
            + Tensor::from(self.p(
                &format!("{prefix}.token_type_embedding.weight"),
                &[c.token_type_vocab_size, d],
            )?)
            .embedding(types)
            + Tensor::from(self.p(&format!("{prefix}.position_embedding.weight"), &[s, d])?)
                .embedding(positions);
        x = x.mask(Tensor::input("initial_mask", vec![batch, s, 1]));
        let key_mask = Tensor::input("key_mask", vec![batch, 1, 1, s]);
        for i in 0..c.token_layers {
            let p = format!("{prefix}.encoder.layers.{i}");
            let normalized = self.norm(&format!("{p}.attn_norm"), x.clone(), d)?;
            let qkv = self.linear(&format!("{p}.attention.qkv_proj"), normalized, d, 3 * d)?;
            let split = |start| {
                qkv.clone()
                    .narrow(2, start, d)
                    .reshape(vec![batch, s, heads, head])
                    .permute(vec![0, 2, 1, 3])
            };
            let (query, key, value) = (split(0), split(d), split(2 * d));
            let scores =
                query.matmul(key.transpose()) * (head as f32).sqrt().recip() + key_mask.clone();
            let context = nn::softmax(scores, 3)
                .matmul(value)
                .permute(vec![0, 2, 1, 3])
                .reshape(vec![batch, s, d]);
            x = x + self.linear(&format!("{p}.attention.out_proj"), context, d, d)?;
            let normalized = self.norm(&format!("{p}.ffn_norm"), x.clone(), d)?;
            let gate = nn::silu(self.linear(
                &format!("{p}.feed_forward.gate_proj"),
                normalized.clone(),
                d,
                hidden,
            )?);
            let up = self.linear(&format!("{p}.feed_forward.up_proj"), normalized, d, hidden)?;
            x = x + self.linear(&format!("{p}.feed_forward.down_proj"), gate * up, hidden, d)?;
        }
        self.norm(&format!("{prefix}.output_norm"), x, d)
    }
    fn pool(
        &self,
        prefix: &str,
        encoded: Tensor,
        query: Tensor,
        query_width: usize,
        batch: usize,
    ) -> Result<Tensor> {
        let d = self.config.token_dim;
        let s = self.config.token_max_tokens;
        let query = self.norm(&format!("{prefix}.query_norm"), query, query_width)?;
        let query = self
            .linear(&format!("{prefix}.query_proj"), query, query_width, d)?
            .reshape(vec![batch, 1, d]);
        let keys = self.linear(&format!("{prefix}.key_proj"), encoded.clone(), d, d)?;
        let values = self.linear(&format!("{prefix}.value_proj"), encoded, d, d)?;
        let scores = (query * keys).reduce_sum(2).reshape(vec![batch, s])
            * (d as f32).sqrt().recip()
            + Tensor::input("pool_mask", vec![batch, s]);
        let weights = nn::softmax(scores, 1).reshape(vec![batch, s, 1]);
        Ok((weights * values).reduce_sum(1).reshape(vec![batch, d]))
    }
    pub fn forward(&self, batch: usize) -> Result<(Tensor, Tensor)> {
        ensure!(batch > 0, "empty model batch");
        let c = &self.config;
        let d = c.token_dim;
        let phase = c.phase_embed_dim;
        let encoded = self.encode("token_encoder", batch)?;
        let cls = encoded.clone().narrow(1, 0, 1).reshape(vec![batch, d]);
        let phase_embedding =
            Tensor::from(self.p("phase_embedding.weight", &[c.num_phases, phase])?)
                .embedding(Tensor::input("phases", vec![batch]));
        let query = Tensor::concat(&[cls.clone(), phase_embedding.clone()], 1);
        let pool = self.pool("policy_pool", encoded.clone(), query, d + phase, batch)?;
        let features = Tensor::concat(&[cls, pool, phase_embedding], 1);
        let features = self.norm("policy_trunk.0", features, 2 * d + phase)?;
        let features = nn::gelu(self.linear("policy_trunk.1", features, 2 * d + phase, d)?);
        let policy = self.norm("policy_trunk.3", features, d)?;
        let logits = if c.use_phase_heads {
            let (discard, play) = if c.use_card_conditioned_actions {
                self.card_heads(policy.clone(), encoded.clone(), batch)?
            } else {
                (
                    self.linear("discard_head", policy.clone(), d, 32)?,
                    self.linear("play_head", policy.clone(), d, 5)?,
                )
            };
            Tensor::concat(
                &[
                    self.linear("bid_head", policy.clone(), d, 5)?,
                    self.linear("suit_head", policy, d, 4)?,
                    discard,
                    play,
                ],
                1,
            )
        } else {
            self.linear("policy_head", policy, d, 46)?
        };
        let value_encoded = if c.use_separate_critic {
            self.encode("critic_encoder", batch)?
        } else {
            encoded
        };
        let value_cls = value_encoded
            .clone()
            .narrow(1, 0, 1)
            .reshape(vec![batch, d]);
        let query = Tensor::from(self.p("value_pool.query", &[d])?)
            .reshape(vec![1, d])
            .broadcast(vec![batch, d]);
        let pool = self.pool("value_pool.pool", value_encoded, query, d, batch)?;
        let features = self.norm(
            "value_trunk.0",
            Tensor::concat(&[value_cls, pool], 1),
            2 * d,
        )?;
        let features = nn::gelu(self.linear("value_trunk.1", features, 2 * d, d)?);
        let features = nn::gelu(self.linear("value_trunk.3", features, d, d / 2)?);
        Ok((logits, self.linear("value_trunk.5", features, d / 2, 1)?))
    }
    fn card_heads(
        &self,
        policy: Tensor,
        encoded: Tensor,
        batch: usize,
    ) -> Result<(Tensor, Tensor)> {
        let d = self.config.token_dim;
        let hand = encoded.narrow(1, 10, 5);
        let query = self
            .linear("card_action_heads.play_query", policy.clone(), d, d)?
            .reshape(vec![batch, 1, d]);
        let keys = self.linear("card_action_heads.play_keys", hand.clone(), d, d)?;
        let bias = self
            .linear("card_action_heads.play_bias", hand.clone(), d, 1)?
            .reshape(vec![batch, 5]);
        let play =
            (query * keys).reduce_sum(2).reshape(vec![batch, 5]) * (d as f32).sqrt().recip() + bias;
        let cards = self.linear("card_action_heads.discard_cards", hand, d, d)?;
        let membership: Vec<f32> = (0u8..32)
            .flat_map(|subset| (0..5).map(move |slot| f32::from((subset >> slot) & 1)))
            .collect();
        let counts: Vec<f32> = (0u8..32).map(|subset| subset.count_ones() as f32).collect();
        let scales: Vec<f32> = counts.iter().map(|&n| n.max(1.).sqrt()).collect();
        let subset = Tensor::constant(membership, vec![1, 32, 5]).matmul(cards)
            / Tensor::constant(scales, vec![1, 32, 1]);
        let count_embeddings =
            Tensor::from(self.p("card_action_heads.discard_count_embedding.weight", &[6, d])?)
                .embedding(Tensor::constant(counts, vec![32]))
                .reshape(vec![1, 32, d])
                .broadcast(vec![batch, 32, d]);
        let query = policy
            .reshape(vec![batch, 1, d])
            .broadcast(vec![batch, 32, d]);
        let features = Tensor::concat(&[subset, count_embeddings, query], 2);
        let features = self.norm("card_action_heads.discard_trunk.0", features, 3 * d)?;
        let features =
            nn::gelu(self.linear("card_action_heads.discard_trunk.1", features, 3 * d, d)?);
        let discard = self
            .linear("card_action_heads.discard_trunk.3", features, d, 1)?
            .reshape(vec![batch, 32]);
        Ok((discard, play))
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Observations {
    pub token_ids: Vec<Vec<u32>>,
    pub token_type_ids: Vec<Vec<u32>>,
    pub phase_ids: Vec<u32>,
}
impl Observations {
    pub fn inputs(&self, c: &Config) -> Result<std::collections::HashMap<String, Vec<f32>>> {
        let batch = self.token_ids.len();
        let s = c.token_max_tokens;
        ensure!(
            batch > 0 && self.token_type_ids.len() == batch && self.phase_ids.len() == batch,
            "observation batch sizes disagree or are empty"
        );
        let mut tokens = Vec::with_capacity(batch * s);
        let mut types = Vec::with_capacity(batch * s);
        let mut initial = Vec::with_capacity(batch * s);
        let mut key = Vec::with_capacity(batch * s);
        let mut pool = Vec::with_capacity(batch * s);
        for (ids, type_ids) in self.token_ids.iter().zip(&self.token_type_ids) {
            ensure!(
                ids.len() == s && type_ids.len() == s,
                "observation token width must be {s}"
            );
            let any = ids.iter().any(|&id| id != 0);
            for (position, (&id, &kind)) in ids.iter().zip(type_ids).enumerate() {
                ensure!(
                    (id as usize) < c.token_vocab_size && (kind as usize) < c.token_type_vocab_size,
                    "token or type ID out of range"
                );
                tokens.push(id as f32);
                types.push(kind as f32);
                initial.push(if id == 0 { 0. } else { 1. });
                let safe = id != 0 || (!any && position == 0);
                key.push(if safe { 0. } else { f32::MIN });
                pool.push(if safe { 0. } else { f32::NEG_INFINITY });
            }
        }
        ensure!(
            self.phase_ids
                .iter()
                .all(|&id| (id as usize) < c.num_phases),
            "phase ID out of range"
        );
        Ok(std::collections::HashMap::from([
            ("tokens".into(), tokens),
            ("token_types".into(), types),
            (
                "phases".into(),
                self.phase_ids.iter().map(|&id| id as f32).collect(),
            ),
            ("initial_mask".into(), initial),
            ("key_mask".into(), key),
            ("pool_mask".into(), pool),
        ]))
    }
}
