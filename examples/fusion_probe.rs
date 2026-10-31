//! Fusion partition/geometry diagnostics; no tensors are serialized to output.
use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fs::File;
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tnsr::graph::TensorGraph;
use tnsr::ptx::{F32, PtxExecutor, PtxFusionPolicy, PtxReductionMode};
use tnsr::tensor::{Parameter, TensorExpr};
use tnsr::tile::{FusionFeatures, MatMulPrecision};
use tnsr::{Executor, SimpleExecutor};

type Gpu = PtxExecutor<F32>;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "geometry")]
    suite: String,
    #[arg(
        long,
        default_value = "128x192,192x128,192x192,192x256,256x192,256x256"
    )]
    shapes: String,
    #[arg(long, default_value = "auto,greedy,32,64,128,256,512,1024")]
    variants: String,
    #[arg(long, default_value_t = 12)]
    samples: usize,
    #[arg(long, default_value_t = 3)]
    block_ms: u64,
    #[arg(long)]
    reverse_order: bool,
}

fn tensor(shape: &[usize], seed: usize) -> TensorExpr<f32> {
    Parameter::new(
        (0..shape.iter().product())
            .map(|i| ((i + seed) % 23) as f32 / 23.0 - 0.5)
            .collect(),
        shape.to_vec(),
    )
    .into()
}

fn timing(
    mut run: impl FnMut() -> Result<Vec<f32>>,
    samples: usize,
    block_ms: u64,
) -> Result<Value> {
    for _ in 0..5 {
        black_box(run()?);
    }
    let start = Instant::now();
    let mut count = 0usize;
    while start.elapsed() < Duration::from_millis(block_ms) {
        black_box(run()?);
        count += 1;
    }
    let count = count.clamp(1, 10000);
    let mut times = Vec::with_capacity(samples);
    for _ in 0..samples {
        let t = Instant::now();
        for _ in 0..count {
            black_box(run()?);
        }
        times.push(t.elapsed().as_secs_f64() * 1e9 / count as f64);
    }
    let mean = times.iter().sum::<f64>() / samples as f64;
    let variance = times.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (samples - 1) as f64;
    let ci = 2.4 * (variance / samples as f64).sqrt();
    Ok(
        json!({"mean_ns":mean,"ci95_ns":[(mean-ci).max(0.0),mean+ci],"samples":samples,"iterations_per_sample":count}),
    )
}

fn verify(actual: &[f32], expected: &[f32]) -> Result<()> {
    ensure!(actual.len() == expected.len(), "output length changed");
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        ensure!(
            (a - b).abs() <= 5e-4 * (1.0 + b.abs()),
            "value mismatch at {index}: {a} vs {b}"
        );
    }
    Ok(())
}

fn feature(f: &FusionFeatures) -> Value {
    let mut value = json!({"shape":f.logical_shape,"launches":f.launches,"traffic_bytes":f.traffic_bytes,
        "intermediate_bytes":f.intermediate_bytes,"weighted_operations":f.weighted_operations,
        "repeated_operations":f.repeated_operations,"address_operations":f.address_operations,
        "estimated_registers_per_thread":f.estimated_registers_per_thread,
        "shared_bytes_per_block":f.shared_bytes_per_block,"threads_per_block":f.threads_per_block,
        "active_threads":f.active_threads,"serial_working_set_bytes":f.serial_working_set_bytes});
    value["reduction_extent"] = json!(f.reduction_extent);
    value["reduction_stride"] = json!(f.reduction_stride);
    value["reduction_lanes"] = json!(f.reduction_lanes);
    value["full_domain_passes"] = json!(f.full_domain_passes);
    value["non_identity_inputs"] = json!(f.non_identity_inputs);
    value["input_index_operations"] = json!(f.input_index_operations);
    value
}

fn run<G>(
    name: &str,
    m: usize,
    n: usize,
    graph: &TensorGraph<f32, G>,
    args: &Args,
    writer: &mut BufWriter<File>,
    validate: impl Fn(&Gpu) -> Result<()>,
) -> Result<()>
where
    TensorGraph<f32, G>: Clone,
{
    let expected = SimpleExecutor::new().execute(graph, HashMap::new())?;
    let mut variants: Vec<&str> = args.variants.split(',').collect();
    let shift = (m + n + name.len()) % variants.len();
    variants.rotate_left(shift);
    if args.reverse_order {
        variants.reverse();
    }
    for variant in variants {
        let mut exec = Gpu::try_new()?;
        exec.set_matmul_precision(MatMulPrecision::StrictF32);
        let tree = variant.starts_with("tree");
        if tree {
            exec = Gpu::try_new_with_reduction_mode(PtxReductionMode::DeterministicTree)?;
            exec.set_matmul_precision(MatMulPrecision::StrictF32);
        }
        exec.set_fusion_policy(
            if variant == "greedy" || variant == "tree-greedy" || variant.parse::<u32>().is_ok() {
                PtxFusionPolicy::Greedy
            } else {
                PtxFusionPolicy::CostAware
            },
        );
        if let Ok(threads) = variant
            .strip_prefix("cost")
            .unwrap_or(variant)
            .parse::<u32>()
        {
            exec.set_serial_reduction_block_threads(Some(threads))?;
        }
        exec.compile(graph)?;
        let actual = exec.execute_compiled(graph, HashMap::new())?;
        verify(&actual, &expected)?;
        validate(&exec)?;
        let plan = exec.execution_plan().context("missing plan")?;
        let features = plan.kernel_features(graph);
        let kernels: Vec<_> = features.iter().map(feature).collect();
        let decisions:Vec<_>=plan.fusion_decisions().iter().map(|decision| json!({"node":decision.node.index(),"alternatives":decision.alternatives.iter().map(|a|json!({"kind":a.kind.to_string(),"selected":a.selected,"score_ns":a.cost.total_ns(),"launch_ns":a.cost.launch_ns,"traffic_ns":a.cost.traffic_ns,"compute_ns":a.cost.compute_ns,"pressure_ns":a.cost.pressure_ns})).collect::<Vec<_>>()})).collect();
        let reductions:Vec<_>=plan.reduction_regions().iter().map(|r|json!({"input_shape":r.input_shape,"axis":r.axis,"schedule":format!("{:?}",r.schedule),"producers":r.producer_operations.len(),"reduced_epilogues":r.epilogue_operations.len(),"full_epilogues":r.full_epilogue_operations.len()})).collect();
        let root = plan.graph_output().context("no output")?;
        let output_view = plan.virtual_value(root).context("missing output view")?;
        let output_index_operations = output_view.access.operation_count();
        let output_non_identity = !output_view.access.is_identity();
        let output_elements = output_view.shape.iter().product::<usize>();
        let execution = timing(
            || exec.execute_compiled(graph, HashMap::new()),
            args.samples,
            args.block_ms,
        )?;
        let readback = timing(
            || exec.get_value(root).context("readback failed"),
            args.samples,
            args.block_ms,
        )?;
        let row = json!({"case":name,"m":m,"n":n,"variant":variant,"graph_nodes":exec.compile_metrics().graph_nodes,"output_index_operations":output_index_operations,"output_non_identity":output_non_identity,"output_elements":output_elements,"end_to_end":execution,"readback_only":readback,"compile_ns":exec.compile_metrics().compile_time.as_nanos(),"fusion_selection_ns":exec.compile_metrics().fusion_selection_time.as_nanos(),"ptx_source_bytes":exec.compile_metrics().ptx_source_bytes,"launches":exec.execution_metrics().kernel_launches,"intermediate_bytes":exec.execution_metrics().intermediate_materialized_bytes,"kernels":kernels,"reductions":reductions,"decisions":decisions});
        serde_json::to_writer(&mut *writer, &row)?;
        writeln!(writer)?;
        writer.flush()?;
        eprintln!(
            "{name}/{m}x{n}/{variant}: {:.2} us; readback {:.2} us",
            execution["mean_ns"].as_f64().unwrap_or_default() / 1000.0,
            readback["mean_ns"].as_f64().unwrap_or_default() / 1000.0
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.samples >= 4 && args.block_ms > 0,
        "invalid timing configuration"
    );
    ensure!(!args.variants.is_empty(), "need variants");
    let mut output = BufWriter::new(File::create(&args.output)?);
    for shape in args.shapes.split(',') {
        let (m, n) = shape.split_once('x').context("shape must be MxN")?;
        let m: usize = m.parse()?;
        let n: usize = n.parse()?;
        ensure!(m > 0 && n > 0, "positive dimensions required");
        let x = tensor(&[m, n], 0);
        let y = tensor(&[m, n], 1);
        let row = tensor(&[1, n], 2);
        let col = tensor(&[m, 1], 3);
        if args.suite == "geometry" || args.suite == "all" {
            for (name, expr) in [
                ("producer_axis0", (x.clone() * row.clone()).reduce_sum(0)),
                ("factored_axis0", x.clone().reduce_sum(0) * row),
                ("producer_axis1", (x.clone() * col).reduce_sum(1)),
                (
                    "reduction_chain",
                    x.clone().relu().exp().reduce_sum(1).log(),
                ),
                (
                    "full_epilogue",
                    x.clone() - x.clone().reduce_mean(1).broadcast(vec![m, n]),
                ),
            ] {
                run(
                    name,
                    m,
                    n,
                    &TensorGraph::from(expr),
                    &args,
                    &mut output,
                    |_| Ok(()),
                )?;
            }
        }
        if args.suite == "layout" || args.suite == "all" {
            ensure!(
                m.is_multiple_of(2),
                "layout probe requires an even row count"
            );
            for (name, expr) in [
                (
                    "reshape_unary",
                    x.clone()
                        .reshape(vec![m / 2, n * 2])
                        .relu()
                        .reshape(vec![m, n]),
                ),
                (
                    "transpose_after_binary",
                    (x.clone() + y.clone()).transpose(),
                ),
                (
                    "transpose_before_binary",
                    x.clone().transpose() + y.transpose(),
                ),
                (
                    "reshape_roundtrip",
                    x.clone().reshape(vec![m / 2, n * 2]).reshape(vec![m, n]),
                ),
            ] {
                run(
                    name,
                    m,
                    n,
                    &TensorGraph::from(expr),
                    &args,
                    &mut output,
                    |_| Ok(()),
                )?;
            }
        }
        if args.suite == "training" || args.suite == "all" {
            let w1 = tensor(&[n, 64], 4);
            let w2 = tensor(&[64, 16], 5);
            let logits = (x.matmul(w1) + tensor(&[1, 64], 6)).relu().matmul(w2);
            let loss = (logits.clone() * logits).reduce_mean(1).reduce_mean(0);
            let forward = TensorGraph::from(loss);
            run(
                "mlp_forward",
                m,
                n,
                &forward,
                &args,
                &mut output,
                |_| Ok(()),
            )?;
            let root = *forward.toposort().last().context("empty training graph")?;
            let backward = forward.with_gradients(root);
            let mut cpu = SimpleExecutor::new();
            cpu.execute(&backward, HashMap::new())?;
            let expected = cpu.get_gradients(&backward);
            run(
                "mlp_backward",
                m,
                n,
                &backward,
                &args,
                &mut output,
                |exec| {
                    let actual = exec.get_gradients(&backward);
                    for (id, wanted) in &expected {
                        verify(actual.get(id).context("missing gradient")?, wanted)?;
                    }
                    Ok(())
                },
            )?;
        }
        if !["geometry", "layout", "training", "all"].contains(&args.suite.as_str()) {
            bail!("unknown suite");
        }
    }
    Ok(())
}
