//! JSON-lines validation and balanced complete-island CUDA-event measurements.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::io::{self, BufRead, Write};
use tnsr::ptx::linear::{LinearInputs, LinearPrecision, LinearShape, LinearTraining};
#[derive(Deserialize)]
struct Spec {
    shape: LinearShape,
    branches: usize,
    replay: bool,
    #[serde(default = "one")]
    replicas: usize,
}
fn one() -> usize {
    1
}
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Request {
    Run {
        inputs: LinearInputs,
        #[serde(default)]
        clear: bool,
    },
    Bench {
        samples: usize,
        repeats: usize,
    },
    Info,
}
fn main() -> Result<()> {
    let spec: Spec = serde_json::from_slice(&std::fs::read(
        std::env::args()
            .nth(1)
            .context("expected specification JSON")?,
    )?)?;
    ensure!(
        spec.replicas > 0 && spec.replicas <= 256,
        "need 1..=256 replicas"
    );
    let mut paths = [Vec::new(), Vec::new()];
    for _ in 0..spec.replicas {
        for (index, precision) in [LinearPrecision::Tf32x3, LinearPrecision::Bf16]
            .into_iter()
            .enumerate()
        {
            paths[index].push(LinearTraining::new(
                spec.shape,
                precision,
                spec.branches,
                spec.replay,
            )?);
        }
    }
    let mut output = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let result = (|| -> Result<serde_json::Value> {
            match serde_json::from_str::<Request>(&line?)? {
                Request::Info => Ok(
                    serde_json::json!({"info":paths.iter().map(|p|p[0].info()).collect::<Vec<_>>(),"ptx":paths.iter().map(|p|p[0].module_sources()).collect::<Vec<_>>()}),
                ),
                Request::Run { inputs, clear } => {
                    let mut results = Vec::new();
                    for replicas in &mut paths {
                        let mut first = None;
                        for path in replicas {
                            if clear {
                                path.clear_gradients()?;
                            }
                            path.upload(&inputs)?;
                            path.run()?;
                            let actual = path.readback()?;
                            if let Some(expected) = &first {
                                ensure!(
                                    serde_json::to_value(expected)?
                                        == serde_json::to_value(&actual)?,
                                    "linear replica outputs differ"
                                );
                            } else {
                                first = Some(actual);
                            }
                        }
                        results.push(first.context("no linear replicas")?);
                    }
                    Ok(serde_json::json!({"results":results}))
                }
                Request::Bench { samples, repeats } => {
                    ensure!(
                        samples >= 20 && repeats > 0,
                        "need at least 20 samples and positive repeats"
                    );
                    ensure!(
                        spec.replicas == 1 || repeats == 1,
                        "rotating allocations require one call per sample"
                    );
                    for replicas in &mut paths {
                        for path in replicas {
                            path.clear_gradients()?;
                            for _ in 0..8 {
                                path.run()?;
                            }
                            path.synchronize()?;
                        }
                    }
                    let mut times = [Vec::new(), Vec::new()];
                    for sample in 0..samples {
                        for index in if (sample
                            + if spec.replicas.is_multiple_of(2) {
                                sample / spec.replicas
                            } else {
                                0
                            })
                            % 2
                            == 0
                        {
                            [0, 1]
                        } else {
                            [1, 0]
                        } {
                            times[index]
                                .push(paths[index][sample % spec.replicas].time_runs(repeats)?);
                        }
                    }
                    Ok(
                        serde_json::json!({"labels":["tf32x3","bf16"],"samples_us":times,"repeats":repeats,"replicas":spec.replicas,"physical_buffer_bytes":paths.iter().map(|p|p.iter().map(|p|p.info().f32_buffer_bytes+p.info().bf16_buffer_bytes).sum::<usize>()).collect::<Vec<_>>(),"protocol":"same-process alternating AB/BA; whole forward/backward, fresh operand/seed casts, output promotions and f32 accumulation; CUDA events; upload/readback excluded","info":paths.iter().map(|p|p[0].info()).collect::<Vec<_>>()}),
                    )
                }
            }
        })();
        let value =
            result.unwrap_or_else(|error| serde_json::json!({"error":format!("{error:#}")}));
        serde_json::to_writer(&mut output, &value)?;
        writeln!(&mut output)?;
        output.flush()?;
    }
    Ok(())
}
