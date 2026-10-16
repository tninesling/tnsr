//! Complete graph-autograd two-projection probe, with explicit mixed casts.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{
    collections::HashMap,
    io::{self, BufRead, Write},
};
use tnsr::{
    graph::TensorGraph,
    ptx::mixed::MixedExecutor,
    tensor::{Parameter, TensorExpr, ValueDType},
    tile::MatMulPrecision,
};

#[derive(Deserialize)]
struct Spec {
    rows: usize,
    inputs: usize,
    hidden: usize,
    outputs: usize,
    #[serde(default)]
    relu: bool,
}
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Request {
    Run {
        x: Vec<f32>,
        w1: Vec<f32>,
        w2: Vec<f32>,
        upstream: Vec<f32>,
    },
    Bench {
        samples: usize,
        repeats: usize,
    },
    Info,
}
fn main() -> Result<()> {
    let spec: Spec = serde_json::from_slice(&std::fs::read(
        std::env::args().nth(1).context("expected spec")?,
    )?)?;
    ensure!(
        [spec.rows, spec.inputs, spec.hidden, spec.outputs]
            .iter()
            .all(|d| *d > 0),
        "positive dimensions required"
    );
    let extents = [
        (spec.rows, spec.inputs),
        (spec.hidden, spec.inputs),
        (spec.outputs, spec.hidden),
    ];
    let shapes = extents.map(|(m, n)| vec![m, n]);
    let lengths = extents
        .into_iter()
        .map(|(m, n)| m.checked_mul(n).context("probe extent overflow"))
        .collect::<Result<Vec<_>>>()?;
    let parameters: Vec<_> = lengths
        .into_iter()
        .zip(&shapes)
        .map(|(len, shape)| Parameter::new(vec![0_f32; len], shape.clone()))
        .collect();
    let mut paths = Vec::new();
    for bf16 in [false, true] {
        let operands: Vec<TensorExpr<f32>> = parameters
            .iter()
            .map(|p| {
                let x: TensorExpr<f32> = p.clone().into();
                if bf16 { x.cast(ValueDType::BF16) } else { x }
            })
            .collect();
        let hidden = operands[0].clone().matmul(operands[1].clone().transpose());
        let hidden = if spec.relu {
            hidden.relu()
        } else {
            let hidden = if bf16 {
                hidden.cast(ValueDType::F32)
            } else {
                hidden
            };
            let hidden = tnsr::nn::silu(hidden);
            if bf16 {
                hidden.cast(ValueDType::BF16)
            } else {
                hidden
            }
        };
        let y = hidden.matmul(operands[2].clone().transpose());
        let y = if bf16 { y.cast(ValueDType::F32) } else { y };
        let mut graph = TensorGraph::new();
        let output = graph.add_expr(&y, false);
        // Reuse the same expression identity inside one lowering pass.
        // A NodeRef avoids a second forward graph when adding the objective.
        let loss = (TensorExpr::node_ref(output, y.shape().clone())
            * TensorExpr::input("g", vec![spec.rows, spec.outputs]))
        .reduce_sum(0)
        .reduce_sum(1);
        let loss_node = graph.add_expr(&loss, false);
        let mut graph = graph.with_gradients(loss_node);
        if !bf16 {
            graph.split_long_matmul_gradients()?;
        }
        let path = MixedExecutor::new_unfused(&graph, MatMulPrecision::Tf32x3, true)?;
        paths.push((graph, output, loss_node, path));
    }
    let (mixed_graph, mixed_output, mixed_loss, _) = &paths[1];
    let mixed_graph = mixed_graph.clone();
    let fused_mixed = MixedExecutor::new(&mixed_graph, MatMulPrecision::Tf32x3, true)?;
    paths.push((mixed_graph, *mixed_output, *mixed_loss, fused_mixed));
    let mut fused = MixedExecutor::new_fused_f32(&paths[0].0, MatMulPrecision::Tf32x3, true)?;
    let mut output = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let response = (|| -> Result<serde_json::Value> {
            match serde_json::from_str::<Request>(&line?)? {
                Request::Info => Ok(serde_json::json!({"info":paths.iter().map(|p| p.3.info()).collect::<Vec<_>>()})),
                Request::Run { x, w1, w2, upstream } => {
                    for (p, data) in parameters.iter().zip([x, w1, w2]) { p.set_values(data)?; }
                    let mut results = Vec::new();
                    for (graph, y, loss, path) in &mut paths {
                        path.upload(HashMap::from([("g".into(), upstream.clone())]))?;
                        path.run()?;
                        let grads = path.gradients(graph)?;
                        results.push(serde_json::json!({"output":path.read(*y)?, "loss":path.read(*loss)?,
                            "gradients":parameters.iter().map(|p| &grads[&p.id()]).collect::<Vec<_>>()}));
                    }
                    fused.upload(HashMap::from([("g".into(), upstream)]))?;
                    fused.run()?;
                    let fused_grads = fused.gradients(&paths[0].0)?;
                    Ok(serde_json::json!({"results":results, "fused_f32_gradients":parameters.iter().map(|p| &fused_grads[&p.id()]).collect::<Vec<_>>()}))
                }
                Request::Bench { samples, repeats } => {
                    ensure!(samples >= 20 && repeats > 0, "need >=20 pairs and positive repeats");
                    for (_, _, _, path) in &mut paths { for _ in 0..8 { path.run()?; } path.synchronize()?; }
                    for _ in 0..8 { fused.run()?; } fused.synchronize()?;
                    let mut times = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                    for pair in 0..samples {
                        for i in [[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1], [0, 3, 1, 2], [0, 3, 2, 1], [1, 0, 2, 3], [1, 0, 3, 2], [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 0, 2], [1, 3, 2, 0], [2, 0, 1, 3], [2, 0, 3, 1], [2, 1, 0, 3], [2, 1, 3, 0], [2, 3, 0, 1], [2, 3, 1, 0], [3, 0, 1, 2], [3, 0, 2, 1], [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 0, 1], [3, 2, 1, 0]][pair % 24] {
                            times[i].push(if i == 3 { fused.time_runs(repeats)? } else { paths[i].3.time_runs(repeats)? });
                        }
                    }
                    Ok(serde_json::json!({"labels":["f32_tf32x3_per_op", "mixed_bf16_per_op", "mixed_bf16_f32_regions", "fused_f32_tf32x3_graph"], "samples_us":times,
                        "info":paths.iter().map(|p| p.3.info()).collect::<Vec<_>>(), "repeats":repeats, "fused_info":fused.info(),
                        "protocol":"same-process balanced 24 permutations of four paths; fenced private streams; full two-projection graph forward/loss/autograd; CUDA events; f32 split-K eligible gradients; fresh casts every run; uploads/readback excluded; no optimizer"}))
                }
            }
        })().unwrap_or_else(|e| serde_json::json!({"error":format!("{e:#}")}));
        serde_json::to_writer(&mut output, &response)?;
        writeln!(&mut output)?;
        output.flush()?;
    }
    Ok(())
}
