//! F32 PPO objective and CPU/resident CUDA optimizer orchestration matching forty-five.
use super::{Model, Observations, Tensor};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tnsr::graph::{NodeIndex, TensorGraph, WithGrad};
use tnsr::nn;
use tnsr::optimizer::Adam;
use tnsr::{Executor, Runtime};

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Anchor {
    pub probabilities: Vec<Vec<f32>>,
    pub log_probabilities: Vec<Vec<f32>>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Batch {
    pub observations: Observations,
    pub action_masks: Vec<Vec<bool>>,
    pub actions: Vec<usize>,
    pub old_log_probs: Vec<f32>,
    pub old_values: Vec<f32>,
    pub returns: Vec<f32>,
    /// Normalize once over the entire rollout, before hand/minibatch selection.
    pub advantages: Vec<f32>,
    pub anchor: Option<Anchor>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub learning_rate: f32,
    pub value_learning_rate: f32,
    pub entropy_coef: f32,
    pub value_coef: f32,
    pub clip_eps: f32,
    pub max_grad_norm: f32,
    pub anchor_kl_coef: f32,
    pub microbatch: usize,
    pub scheduler_updates: usize,
    pub min_learning_rate: f32,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            learning_rate: 5e-5,
            value_learning_rate: 2e-4,
            entropy_coef: 0.005,
            value_coef: 0.5,
            clip_eps: 0.2,
            max_grad_norm: 0.5,
            anchor_kl_coef: 0.,
            microbatch: 1024,
            scheduler_updates: 1000,
            min_learning_rate: 1e-6,
        }
    }
}
impl Settings {
    fn validate(&self) -> Result<()> {
        ensure!(
            [
                self.learning_rate,
                self.value_learning_rate,
                self.entropy_coef,
                self.value_coef,
                self.max_grad_norm,
                self.anchor_kl_coef,
                self.min_learning_rate
            ]
            .iter()
            .all(|x| x.is_finite() && *x >= 0.),
            "PPO coefficients/rates must be finite and nonnegative"
        );
        ensure!(
            self.clip_eps.is_finite() && self.clip_eps > 0. && self.clip_eps < 1.,
            "PPO clipping epsilon must lie in (0,1)"
        );
        ensure!(
            self.microbatch > 0 && self.scheduler_updates > 0,
            "PPO microbatch/schedule must be nonempty"
        );
        ensure!(
            self.min_learning_rate <= self.learning_rate.min(self.value_learning_rate),
            "minimum learning rate exceeds initial rate"
        );
        Ok(())
    }
}
impl Batch {
    fn validate(&self, model: &Model, settings: &Settings) -> Result<()> {
        self.observations.inputs(&model.config)?;
        let n = self.actions.len();
        ensure!(
            n > 0 && self.observations.token_ids.len() == n && self.action_masks.len() == n,
            "PPO batch sizes disagree"
        );
        for data in [
            &self.old_log_probs,
            &self.old_values,
            &self.returns,
            &self.advantages,
        ] {
            ensure!(
                data.len() == n && data.iter().all(|x| x.is_finite()),
                "invalid PPO scalar input"
            );
        }
        for (mask, &action) in self.action_masks.iter().zip(&self.actions) {
            ensure!(
                mask.len() == 46 && action < 46 && mask[action],
                "PPO selected action must be legal"
            );
        }
        ensure!(
            settings.anchor_kl_coef == 0. || self.anchor.is_some(),
            "bidding KL requires frozen anchor outputs"
        );
        if let Some(anchor) = &self.anchor {
            ensure!(
                anchor.probabilities.len() == n && anchor.log_probabilities.len() == n,
                "anchor batch size mismatch"
            );
            for (i, (probs, logs)) in anchor
                .probabilities
                .iter()
                .zip(&anchor.log_probabilities)
                .enumerate()
            {
                ensure!(
                    probs.len() == 46
                        && logs.len() == 46
                        && probs.iter().all(|p| p.is_finite() && *p >= 0. && *p <= 1.)
                        && logs.iter().all(|p| p.is_finite()),
                    "invalid anchor distribution"
                );
                ensure!(
                    (probs.iter().sum::<f32>() - 1.).abs() < 1e-4,
                    "anchor probabilities must sum to one"
                );
                ensure!(
                    probs
                        .iter()
                        .zip(&self.action_masks[i])
                        .all(|(p, valid)| *valid || *p == 0.),
                    "anchor assigns probability to illegal action"
                );
            }
        }
        Ok(())
    }
    fn slice(&self, start: usize, end: usize) -> Self {
        Self {
            observations: Observations {
                token_ids: self.observations.token_ids[start..end].to_vec(),
                token_type_ids: self.observations.token_type_ids[start..end].to_vec(),
                phase_ids: self.observations.phase_ids[start..end].to_vec(),
            },
            action_masks: self.action_masks[start..end].to_vec(),
            actions: self.actions[start..end].to_vec(),
            old_log_probs: self.old_log_probs[start..end].to_vec(),
            old_values: self.old_values[start..end].to_vec(),
            returns: self.returns[start..end].to_vec(),
            advantages: self.advantages[start..end].to_vec(),
            anchor: self.anchor.as_ref().map(|a| Anchor {
                probabilities: a.probabilities[start..end].to_vec(),
                log_probabilities: a.log_probabilities[start..end].to_vec(),
            }),
        }
    }
}
struct Compiled {
    graph: TensorGraph<f32, WithGrad>,
    output: NodeIndex,
    executor: Runtime<f32>,
}
fn build(model: &Model, n: usize, clip: f32) -> Result<(TensorGraph<f32, WithGrad>, NodeIndex)> {
    let (logits, values) = model.forward(n)?;
    let valid = Tensor::input("action_valid", vec![n, 46]);
    let logits = logits + Tensor::input("action_offset", vec![n, 46]);
    let maximum = logits.clone().reduce_max(1);
    let shifted = logits.clone() - maximum.clone().broadcast(vec![n, 46]);
    let normalizer = maximum + shifted.exp().reduce_sum(1).log();
    let logs = (logits.clone() - normalizer.broadcast(vec![n, 46])).mask(valid.clone());
    let probabilities = nn::softmax(logits, 1);
    let selected = (logs.clone() * Tensor::input("chosen", vec![n, 46])).reduce_sum(1);
    let entropy = (-(probabilities * logs.clone()).reduce_sum(1)).mean_all();
    let log_ratio = selected - Tensor::input("old_log_probs", vec![n, 1]);
    let ratios = log_ratio.clone().exp();
    let advantages = Tensor::input("advantages", vec![n, 1]);
    let policy = -nn::minimum(
        ratios.clone() * advantages.clone(),
        nn::clamp(ratios.clone(), 1. - clip, 1. + clip) * advantages,
    )
    .mean_all();
    let old_values = Tensor::input("old_values", vec![n, 1]);
    let returns = Tensor::input("returns", vec![n, 1]);
    let difference = values.clone() - returns.clone();
    let clipped = old_values.clone() + nn::clamp(values - old_values, -clip, clip) - returns;
    let value = nn::maximum(difference.clone() * difference, clipped.clone() * clipped).mean_all();
    let anchor = (Tensor::input("anchor_probs", vec![n, 46])
        * (Tensor::input("anchor_logs", vec![n, 46]) - logs))
        .reduce_sum(1)
        .mask(Tensor::input("bidding", vec![n, 1]))
        .reduce_sum(0)
        * Tensor::input("bid_recip", vec![]);
    let approx_kl = ((ratios.clone() - 1.) - log_ratio).mean_all();
    let clip_fraction =
        ((ratios.clone() - 1.).gt(clip) + (Tensor::from(1.) - ratios).gt(clip)).mean_all();
    let loss = policy.clone() + value.clone() * Tensor::input("value_coef", vec![])
        - entropy.clone() * Tensor::input("entropy_coef", vec![])
        + anchor.clone() * Tensor::input("anchor_coef", vec![]);
    let weighted = loss.clone() * Tensor::input("gradient_weight", vec![]);
    let parts: Vec<_> = [
        loss,
        policy,
        value,
        entropy,
        anchor,
        approx_kl,
        clip_fraction,
        weighted,
    ]
    .into_iter()
    .map(|x| x.reshape(vec![1]))
    .collect();
    let graph: TensorGraph<f32> = Tensor::concat(&parts, 0).into();
    let output = graph
        .toposort()
        .last()
        .copied()
        .context("empty PPO graph")?;
    let loss_node = *graph
        .inputs(output)
        .get(7)
        .context("missing weighted PPO objective")?;
    Ok((graph.with_gradients(loss_node), output))
}
fn inputs(
    batch: &Batch,
    model: &Model,
    settings: &Settings,
    weight: f32,
) -> Result<HashMap<String, Vec<f32>>> {
    let mut inputs = batch.observations.inputs(&model.config)?;
    let n = batch.actions.len();
    let mut valid = Vec::with_capacity(n * 46);
    let mut offsets = Vec::with_capacity(n * 46);
    let mut chosen = vec![0.; n * 46];
    for (row, mask) in batch.action_masks.iter().enumerate() {
        chosen[row * 46 + batch.actions[row]] = 1.;
        for &legal in mask {
            valid.push(if legal { 1. } else { 0. });
            offsets.push(if legal { 0. } else { f32::NEG_INFINITY });
        }
    }
    let bidding: Vec<f32> = batch
        .observations
        .phase_ids
        .iter()
        .map(|&phase| if phase == 1 { 1. } else { 0. })
        .collect();
    let bid_count = bidding.iter().sum::<f32>();
    let (probs, logs) = batch
        .anchor
        .as_ref()
        .map(|a| {
            (
                a.probabilities.iter().flatten().copied().collect(),
                a.log_probabilities.iter().flatten().copied().collect(),
            )
        })
        .unwrap_or_else(|| (vec![0.; n * 46], vec![0.; n * 46]));
    for (name, data) in [
        ("action_valid", valid),
        ("action_offset", offsets),
        ("chosen", chosen),
        ("old_log_probs", batch.old_log_probs.clone()),
        ("old_values", batch.old_values.clone()),
        ("returns", batch.returns.clone()),
        ("advantages", batch.advantages.clone()),
        ("anchor_probs", probs),
        ("anchor_logs", logs),
        ("bidding", bidding),
        (
            "bid_recip",
            vec![if bid_count > 0. {
                bid_count.recip()
            } else {
                0.
            }],
        ),
        ("value_coef", vec![settings.value_coef]),
        ("entropy_coef", vec![settings.entropy_coef]),
        ("anchor_coef", vec![settings.anchor_kl_coef]),
        ("gradient_weight", vec![weight]),
    ] {
        inputs.insert(name.into(), data);
    }
    Ok(inputs)
}

pub struct Trainer {
    policy: Adam<f32>,
    value: Adam<f32>,
    initial: Settings,
    backend: String,
    steps: usize,
    compiled: HashMap<(usize, u32), Compiled>,
    profile: bool,
    #[cfg(feature = "cuda")]
    resident: Option<tnsr::ptx::resident::ResidentAdam>,
}
impl Trainer {
    pub fn new(settings: &Settings, backend: String) -> Result<Self> {
        settings.validate()?;
        Ok(Self {
            #[cfg(feature = "cuda")]
            resident: None,
            policy: Adam::new(settings.learning_rate),
            value: Adam::new(settings.value_learning_rate),
            initial: settings.clone(),
            backend,
            steps: 0,
            compiled: HashMap::new(),
            profile: std::env::var("TNSR_PROFILE_PPO").as_deref() == Ok("1"),
        })
    }
    pub fn step(
        &mut self,
        model: &Model,
        batch: &Batch,
        settings: &Settings,
        return_gradients: bool,
    ) -> Result<serde_json::Value> {
        let total_started = Instant::now();
        let mut timing = [0f64; 7];
        let kernel_launches = std::cell::Cell::new(0usize);
        let mut graph_nodes = 0usize;
        settings.validate()?;
        batch.validate(model, settings)?;
        ensure!(
            settings.learning_rate == self.initial.learning_rate
                && settings.value_learning_rate == self.initial.value_learning_rate
                && settings.scheduler_updates == self.initial.scheduler_updates
                && settings.min_learning_rate == self.initial.min_learning_rate,
            "optimizer initial rates/schedule cannot change without starting a fresh worker"
        );
        #[cfg(feature = "cuda")]
        {
            let started = Instant::now();
            if self.backend == "cuda" && self.resident.is_none() {
                use tnsr::ptx::resident::{AdamGroup, ResidentAdam};
                let parameters = model
                    .parameters
                    .iter()
                    .map(|(_, entry)| {
                        (
                            entry.parameter.clone(),
                            usize::from(entry.group != "policy"),
                        )
                    })
                    .collect();
                self.resident = Some(ResidentAdam::new(
                    parameters,
                    vec![
                        AdamGroup::new(settings.learning_rate),
                        AdamGroup::new(settings.value_learning_rate),
                    ],
                )?);
            }
            if let Some(resident) = &mut self.resident {
                resident.begin_step()?;
            }
            timing[0] += started.elapsed().as_secs_f64();
        }
        #[cfg(feature = "cuda")]
        let using_resident = self.resident.is_some();
        #[cfg(not(feature = "cuda"))]
        let using_resident = false;
        let n = batch.actions.len();
        let mut accumulated: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut metrics = [0f64; 7];
        let mut update_graph = None;
        for start in (0..n).step_by(settings.microbatch) {
            let end = (start + settings.microbatch).min(n);
            let micro = batch.slice(start, end);
            let size = end - start;
            let weight = size as f32 / n as f32;
            let key = (size, settings.clip_eps.to_bits());
            let started = Instant::now();
            if !self.compiled.contains_key(&key) {
                let (graph, output) = build(model, size, settings.clip_eps)?;
                #[cfg(feature = "cuda")]
                let executor = if let Some(resident) = &self.resident {
                    let mut executor = resident.executor()?;
                    executor.set_matmul_precision(tnsr::tile::MatMulPrecision::StrictF32);
                    Runtime::Ptx(executor)
                } else {
                    crate::runtime(&self.backend)?
                };
                #[cfg(not(feature = "cuda"))]
                let executor = crate::runtime(&self.backend)?;
                if self.compiled.len() >= 8
                    && let Some(key) = self.compiled.keys().min().copied()
                {
                    self.compiled.remove(&key);
                }
                self.compiled.insert(
                    key,
                    Compiled {
                        graph,
                        output,
                        executor,
                    },
                );
            }
            timing[0] += started.elapsed().as_secs_f64();
            let compiled = self
                .compiled
                .get_mut(&key)
                .context("PPO compilation cache missing")?;
            let started = Instant::now();
            let feeds = inputs(&micro, model, settings, weight)?;
            timing[1] += started.elapsed().as_secs_f64();
            let started = Instant::now();
            #[cfg(feature = "cuda")]
            if let (Some(resident), Runtime::Ptx(executor)) =
                (&mut self.resident, &mut compiled.executor)
            {
                resident.accumulate(executor, &compiled.graph, feeds)?;
            } else {
                compiled.executor.execute(&compiled.graph, feeds)?;
            }
            #[cfg(not(feature = "cuda"))]
            compiled.executor.execute(&compiled.graph, feeds)?;
            timing[2] += started.elapsed().as_secs_f64();
            #[cfg(feature = "cuda")]
            if let Runtime::Ptx(executor) = &compiled.executor {
                kernel_launches
                    .set(kernel_launches.get() + executor.execution_metrics().kernel_launches);
            }
            graph_nodes = compiled.graph.graph.node_count();
            let started = Instant::now();
            let values = compiled
                .executor
                .get_value(compiled.output)
                .context("PPO metrics output was not retained")?;
            ensure!(
                values.len() == 8 && values.iter().all(|x| x.is_finite()),
                "PPO produced nonfinite losses"
            );
            for (target, &value) in metrics.iter_mut().zip(&values) {
                *target += f64::from(value) * f64::from(weight);
            }
            timing[3] += started.elapsed().as_secs_f64();
            let started = Instant::now();
            if using_resident {
                update_graph = Some(key);
                continue;
            }
            let gradients = compiled.executor.get_gradients(&compiled.graph);
            timing[4] += started.elapsed().as_secs_f64();
            let started = Instant::now();
            for (id, gradient) in gradients {
                ensure!(
                    gradient.iter().all(|x| x.is_finite()),
                    "PPO produced nonfinite gradients"
                );
                let target = accumulated
                    .entry(id)
                    .or_insert_with(|| vec![0.; gradient.len()]);
                ensure!(
                    target.len() == gradient.len(),
                    "microbatch gradient shapes disagree"
                );
                for (target, value) in target.iter_mut().zip(gradient) {
                    *target += value;
                }
            }
            timing[5] += started.elapsed().as_secs_f64();
            update_graph = Some(key);
        }
        let started = Instant::now();
        #[cfg(feature = "cuda")]
        let resident_result = if let Some(resident) = &mut self.resident {
            let status = resident.step(settings.max_grad_norm)?;
            kernel_launches.set(status.kernel_launches);
            let mut named = serde_json::Map::new();
            if return_gradients {
                let gradients = resident.gradients()?;
                for (name, entry) in model.parameters.iter() {
                    named.insert(
                        name.into(),
                        serde_json::json!(
                            gradients
                                .get(&entry.parameter.id())
                                .context("missing resident model gradient")?
                        ),
                    );
                }
            }
            Some((status.gradient_norm, status.gradient_scale, named))
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let resident_result: Option<(
            f64,
            f32,
            serde_json::Map<String, serde_json::Value>,
        )> = None;
        let (norm, scale, named) = if let Some(result) = resident_result {
            result
        } else {
            self.cpu_update(model, accumulated, update_graph, settings, return_gradients)?
        };
        self.steps += 1;
        timing[6] += started.elapsed().as_secs_f64();
        let keys = [
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "bid_anchor_kl",
            "approx_kl",
            "clip_fraction",
        ];
        let metrics: serde_json::Map<_, _> = keys
            .into_iter()
            .zip(metrics)
            .map(|(key, value)| (key.into(), serde_json::json!(value)))
            .collect();
        let mut result = serde_json::json!({"metrics":metrics,"gradient_norm_before_clip":norm,"gradient_scale":scale,"optimizer_steps":self.steps,"compiled_shapes":self.compiled.len()});
        if self.profile {
            result["profile"] = serde_json::json!({
                "compile_cache_seconds":timing[0], "input_seconds":timing[1],
                "execute_seconds":timing[2], "metrics_seconds":timing[3],
                "gradient_extraction_seconds":timing[4], "accumulate_seconds":timing[5],
                "clip_adam_finite_check_seconds":timing[6],
                "total_seconds":total_started.elapsed().as_secs_f64(),
                "kernel_launches":kernel_launches.get(),"graph_nodes":graph_nodes
            });
        }
        if return_gradients {
            result["gradients"] = serde_json::Value::Object(named);
        }
        Ok(result)
    }
    fn cpu_update(
        &mut self,
        model: &Model,
        mut accumulated: HashMap<usize, Vec<f32>>,
        update_graph: Option<(usize, u32)>,
        settings: &Settings,
        return_gradients: bool,
    ) -> Result<(f64, f32, serde_json::Map<String, serde_json::Value>)> {
        let norm = accumulated
            .values()
            .flatten()
            .map(|&x| f64::from(x).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = if settings.max_grad_norm > 0. {
            (f64::from(settings.max_grad_norm) / (norm + 1e-6)).min(1.) as f32
        } else {
            1.
        };
        for gradient in accumulated.values_mut() {
            for value in gradient {
                *value *= scale;
            }
        }
        let mut named = serde_json::Map::new();
        for (name, entry) in model.parameters.iter() {
            let gradient = accumulated
                .get(&entry.parameter.id())
                .with_context(|| format!("missing model gradient: {name}"))?;
            ensure!(
                gradient.len() == entry.parameter.shape().iter().product::<usize>(),
                "{name}: gradient shape mismatch"
            );
            if return_gradients {
                named.insert(name.into(), serde_json::json!(gradient));
            }
        }
        let (policy, value): (HashMap<_, _>, HashMap<_, _>) =
            accumulated.into_iter().partition(|(id, _)| {
                model
                    .parameters
                    .iter()
                    .any(|(_, entry)| entry.parameter.id() == *id && entry.group == "policy")
            });
        let graph = &self
            .compiled
            .get(&update_graph.context("no PPO microbatches")?)
            .context("update graph missing")?
            .graph;
        self.policy.step(graph, &policy);
        self.value.step(graph, &value);
        ensure!(
            model.parameters.iter().all(|(_, entry)| entry
                .parameter
                .values()
                .is_ok_and(|values| values.iter().all(|x| x.is_finite()))),
            "Adam produced invalid parameters"
        );
        Ok((norm, scale, named))
    }

    /// The existing trainer advances its schedule per rollout, not per minibatch.
    pub fn advance_schedule(&mut self, update: usize) -> Result<serde_json::Value> {
        ensure!(
            update <= self.initial.scheduler_updates,
            "schedule update exceeds configured horizon"
        );
        let factor = 0.5
            * (1.
                + (std::f64::consts::PI * update as f64 / self.initial.scheduler_updates as f64)
                    .cos());
        let rate = |initial: f32| {
            self.initial.min_learning_rate
                + (initial - self.initial.min_learning_rate) * factor as f32
        };
        let policy = rate(self.initial.learning_rate);
        let value = rate(self.initial.value_learning_rate);
        self.policy.set_learning_rate(policy)?;
        self.value.set_learning_rate(value)?;
        #[cfg(feature = "cuda")]
        if let Some(resident) = &mut self.resident {
            resident.set_learning_rate(0, policy)?;
            resident.set_learning_rate(1, value)?;
        }
        Ok(
            serde_json::json!({"policy_learning_rate":policy,"value_learning_rate":value,"schedule_update":update}),
        )
    }
}
