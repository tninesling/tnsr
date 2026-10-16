//! Persistent JSON-lines inference and PPO worker for the real forty-five model.
mod forty_five;
use anyhow::{Context, Result, bail, ensure};
use forty_five::{Model, Observations};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;
use tnsr::graph::TensorGraph;
use tnsr::tensor::TensorExpr;
use tnsr::{Executor, Runtime, SimpleExecutor};

fn runtime(backend: &str) -> Result<Runtime<f32>> {
    match backend {
        "cpu" => Ok(Runtime::Cpu(SimpleExecutor::new())),
        #[cfg(feature = "cuda")]
        "cuda" => {
            let mut executor = tnsr::ptx::PtxExecutor::try_new()?;
            executor.set_matmul_precision(tnsr::tile::MatMulPrecision::StrictF32);
            Ok(Runtime::Ptx(executor))
        }
        _ => bail!("unsupported backend {backend}; compile with --features cuda for GPU execution"),
    }
}
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    OptimizerState,
    Infer {
        batch: Observations,
    },
    Save {
        directory: String,
    },
    Train {
        batch: Box<forty_five::ppo::Batch>,
        settings: forty_five::ppo::Settings,
        #[serde(default)]
        return_gradients: bool,
    },
    Schedule {
        update: usize,
    },
}
struct Worker {
    model: Model,
    backend: String,
    trainer: Option<forty_five::ppo::Trainer>,
    predictions: HashMap<usize, (TensorGraph<f32>, Runtime<f32>)>,
}
impl Worker {
    fn infer(&mut self, batch: &Observations) -> Result<serde_json::Value> {
        let inputs = batch.inputs(&self.model.config)?;
        let n = batch.token_ids.len();
        if !self.predictions.contains_key(&n) {
            let (logits, value) = self.model.forward(n)?;
            let graph = TensorExpr::concat(&[logits, value], 1).into();
            if self.predictions.len() >= 8
                && let Some(key) = self.predictions.keys().min().copied()
            {
                self.predictions.remove(&key);
            }
            self.predictions.insert(n, (graph, runtime(&self.backend)?));
        }
        let (graph, runtime) = self
            .predictions
            .get_mut(&n)
            .context("prediction cache entry missing")?;
        let output = runtime.execute_with_indices(graph, inputs.floats, inputs.indices)?;
        ensure!(
            output.len() == n * 47 && output.iter().all(|x| x.is_finite()),
            "invalid model output"
        );
        let logits: Vec<_> = output
            .chunks_exact(47)
            .map(|row| row[..46].to_vec())
            .collect();
        let values: Vec<_> = output.chunks_exact(47).map(|row| row[46]).collect();
        Ok(serde_json::json!({"logits":logits,"values":values}))
    }
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 3, "usage: forty_five_worker ARCHIVE cpu|cuda");
    let model = Model::load(Path::new(&args[1]))?;
    let mut worker = Worker {
        model,
        backend: args[2].clone(),
        trainer: None,
        predictions: HashMap::new(),
    };
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
    for line in std::io::stdin().lock().lines() {
        let response = (|| -> Result<serde_json::Value> {
            match serde_json::from_str::<Request>(&line?)? {
                Request::OptimizerState => worker
                    .trainer
                    .as_ref()
                    .context("PPO trainer has not initialized")?
                    .optimizer_state(&worker.model),
                Request::Infer { batch } => worker.infer(&batch),
                Request::Train {
                    batch,
                    settings,
                    return_gradients,
                } => {
                    if worker.trainer.is_none() {
                        worker.trainer = Some(forty_five::ppo::Trainer::new(
                            &settings,
                            worker.backend.clone(),
                        )?);
                    }
                    worker
                        .trainer
                        .as_mut()
                        .context("PPO trainer missing")?
                        .step(&worker.model, &batch, &settings, return_gradients)
                }
                Request::Schedule { update } => worker
                    .trainer
                    .as_mut()
                    .context("PPO trainer has not initialized")?
                    .advance_schedule(update),
                Request::Save { directory } => {
                    ensure!(
                        !Path::new(&directory).exists(),
                        "checkpoint directory already exists"
                    );
                    worker.model.parameters.save(
                        Path::new(&directory),
                        serde_json::to_value(&worker.model.config)?,
                    )?;
                    Ok(serde_json::json!({"saved":directory}))
                }
            }
        })();
        let response =
            response.unwrap_or_else(|error| serde_json::json!({"error":format!("{error:#}")}));
        writeln!(output, "{}", serde_json::to_string(&response)?)?;
        output.flush()?;
    }
    Ok(())
}
fn main() -> Result<()> {
    std::thread::Builder::new()
        .name("forty-five".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(run)?
        .join()
        .map_err(|_| anyhow::anyhow!("model worker thread panicked"))?
}

#[cfg(test)]
#[path = "forty_five/tests.rs"]
mod tests;
