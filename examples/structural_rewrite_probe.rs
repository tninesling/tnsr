//! Paired structural-rewrite measurements; outputs contain timings, never tensors.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    fs::File,
    hint::black_box,
    io::{BufWriter, Write},
    path::PathBuf,
    time::{Duration, Instant},
};
use tnsr::{
    Executor, SimpleExecutor,
    graph::{
        TensorGraph, TensorGraphNode,
        rewrite::{RewriteConfig, RewriteNumerics, optimize_expr_with_numerics},
    },
    ptx::{F32, PtxExecutor, mixed::MixedExecutor},
    tensor::{Parameter, TensorExpr, ValueDType},
    tile::MatMulPrecision,
};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "cpu")]
    backend: String,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "128,256,384")]
    sizes: String,
    #[arg(long, default_value_t = 20)]
    samples: usize,
    #[arg(long, default_value_t = 10)]
    block_ms: u64,
    #[arg(long)]
    reverse_order: bool,
}

fn source(n: usize, seed: usize) -> TensorExpr<f32> {
    Parameter::new(
        (0..n * n)
            .map(|i| ((i + seed) % 32) as f32 / 32.0 - 0.5)
            .collect(),
        vec![n, n],
    )
    .into()
}

fn cases(n: usize) -> Vec<(&'static str, TensorExpr<f32>)> {
    let x = source(n, 0);
    let y = source(n, 5);
    vec![
        (
            "concat_narrow_roundtrip_rows",
            TensorExpr::concat(
                &[
                    x.clone().narrow(0, 0, n / 2),
                    x.clone().narrow(0, n / 2, n / 2),
                ],
                0,
            ),
        ),
        (
            "concat_narrow_roundtrip_cols",
            TensorExpr::concat(
                &[
                    x.clone().narrow(1, 0, n / 2),
                    x.clone().narrow(1, n / 2, n / 2),
                ],
                1,
            ),
        ),
        (
            "select_concat_segment",
            TensorExpr::concat(&[x.clone(), y], 1).narrow(1, n, n),
        ),
        (
            "nested_narrow_cols",
            x.clone().narrow(1, n / 4, n / 2).narrow(1, n / 8, n / 4),
        ),
        (
            "unary_transpose_inside_cast",
            x.clone()
                .transpose()
                .relu()
                .transpose()
                .cast(ValueDType::BF16)
                .cast(ValueDType::F32),
        ),
        (
            "unary_reshape_inside_cast",
            x.clone()
                .reshape(vec![n / 2, n * 2])
                .relu()
                .reshape(vec![n, n])
                .cast(ValueDType::BF16)
                .cast(ValueDType::F32),
        ),
        (
            "opaque_max_layout",
            x.clone().transpose().relu().transpose().reduce_max(1),
        ),
        ("unchanged_segment_control", x.narrow(1, n / 4, n / 2)),
    ]
}

fn check_bits(actual: &[f32], expected: &[f32]) -> Result<()> {
    ensure!(actual.len() == expected.len(), "output length changed");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        ensure!(
            actual.to_bits() == expected.to_bits(),
            "bit mismatch at {index}: {actual} vs {expected}"
        );
    }
    Ok(())
}

fn gradients(expr: TensorExpr<f32>) -> Result<HashMap<usize, Vec<f32>>> {
    let shape = expr.shape().clone();
    let weights = TensorExpr::constant(
        (0..shape.iter().product())
            .map(|i| [1.0f32, 2.0, 4.0][i % 3])
            .collect(),
        shape,
    );
    let mut loss = expr * weights;
    for axis in (0..loss.shape().len()).rev() {
        loss = loss.reduce_sum(axis);
    }
    let graph: TensorGraph<f32> = loss.into();
    let root = *graph.toposort().last().context("missing loss")?;
    let graph = graph.with_gradients(root);
    let mut cpu = SimpleExecutor::new();
    cpu.execute(&graph, HashMap::new())?;
    Ok(cpu.get_gradients(&graph))
}

fn check_cpu(original: &TensorExpr<f32>, optimized: &TensorExpr<f32>) -> Result<Vec<f32>> {
    let expected =
        SimpleExecutor::new().execute(&TensorGraph::from(original.clone()), HashMap::new())?;
    let actual =
        SimpleExecutor::new().execute(&TensorGraph::from(optimized.clone()), HashMap::new())?;
    check_bits(&actual, &expected)?;
    let old_gradients = gradients(original.clone())?;
    let new_gradients = gradients(optimized.clone())?;
    for (parameter, values) in &old_gradients {
        if let Some(actual) = new_gradients.get(parameter) {
            check_bits(actual, values)?;
        } else {
            // Removing an unused concat branch can remove its all-zero gradient.
            ensure!(
                values.iter().all(|value| *value == 0.0),
                "removed nonzero parameter gradient"
            );
        }
    }
    ensure!(
        new_gradients
            .keys()
            .all(|id| old_gradients.contains_key(id)),
        "introduced a parameter"
    );
    Ok(expected)
}

fn measure(mut run: impl FnMut() -> Result<Vec<f32>>, args: &Args) -> Result<Value> {
    for _ in 0..5 {
        black_box(run()?);
    }
    let t = Instant::now();
    let mut count = 0usize;
    while t.elapsed() < Duration::from_millis(args.block_ms) {
        black_box(run()?);
        count += 1;
    }
    let count = count.clamp(1, 10_000);
    let mut samples = Vec::with_capacity(args.samples);
    for _ in 0..args.samples {
        let t = Instant::now();
        for _ in 0..count {
            black_box(run()?);
        }
        samples.push(t.elapsed().as_secs_f64() * 1e9 / count as f64);
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (samples.len() - 1) as f64;
    let interval = 2.4 * (variance / samples.len() as f64).sqrt();
    Ok(
        json!({"mean_ns":mean,"ci95_ns":[(mean-interval).max(0.0),mean+interval],"sample_means_ns":samples,"iterations_per_sample":count}),
    )
}

fn run_variant(expr: TensorExpr<f32>, expected: &[f32], args: &Args) -> Result<Value> {
    let graph: TensorGraph<f32> = expr.into();
    let mixed = graph
        .graph
        .node_weights()
        .any(|node| matches!(node, TensorGraphNode::Cast { .. }));
    let root = *graph.toposort().last().context("empty graph")?;
    let mut row = json!({"graph_nodes":graph.graph.node_count(),"mixed_executor":mixed});
    if args.backend == "cpu" {
        let mut cpu = SimpleExecutor::new();
        check_bits(&cpu.execute(&graph, HashMap::new())?, expected)?;
        row["timing"] = measure(|| cpu.execute(&graph, HashMap::new()), args)?;
    } else if mixed {
        let start = Instant::now();
        let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::StrictF32, false)?;
        row["compile_ns"] = json!(start.elapsed().as_nanos());
        gpu.upload(HashMap::new())?;
        gpu.run()?;
        check_bits(&gpu.read(root)?, expected)?;
        row["kernels"] = json!(gpu.info().kernels);
        row["retained_bytes"] = json!(gpu.info().f32_bytes + gpu.info().bf16_bytes);
        row["timing"] = measure(
            || {
                gpu.run()?;
                gpu.read(root)
            },
            args,
        )?;
    } else {
        let mut gpu = PtxExecutor::<F32>::try_new()?;
        gpu.set_matmul_precision(MatMulPrecision::StrictF32);
        gpu.compile(&graph)?;
        check_bits(&gpu.execute_compiled(&graph, HashMap::new())?, expected)?;
        row["compile_ns"] = json!(gpu.compile_metrics().compile_time.as_nanos());
        row["kernels"] = json!(gpu.execution_metrics().kernel_launches);
        row["intermediate_bytes"] = json!(gpu.execution_metrics().intermediate_materialized_bytes);
        row["timing"] = measure(|| gpu.execute_compiled(&graph, HashMap::new()), args)?;
    }
    Ok(row)
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        matches!(args.backend.as_str(), "cpu" | "cuda"),
        "backend must be cpu or cuda"
    );
    ensure!(
        args.samples >= 4 && args.block_ms > 0,
        "invalid timing configuration"
    );
    let mut writer = BufWriter::new(File::create(&args.output)?);
    for n in args.sizes.split(',').map(str::parse::<usize>) {
        let n = n?;
        ensure!(
            n >= 8 && n.is_multiple_of(8),
            "sizes must be positive multiples of eight"
        );
        for (name, original) in cases(n) {
            let numerics = if name.contains("concat") || name == "nested_narrow_cols" {
                RewriteNumerics::RealAlgebra
            } else {
                RewriteNumerics::PreserveOperations
            };
            let start = Instant::now();
            let optimized =
                optimize_expr_with_numerics(&original, RewriteConfig::default(), numerics);
            let optimization_ns = start.elapsed().as_nanos();
            let expected = check_cpu(&original, &optimized)
                .with_context(|| format!("validating {name}/{n}"))?;
            let same_root = std::ptr::eq(original.kind(), optimized.kind());
            let mut variants = vec![("original", original), ("optimized", optimized)];
            if (n / 8 + name.len()).is_multiple_of(2) {
                variants.reverse();
            }
            if args.reverse_order {
                variants.reverse();
            }
            for (variant, expr) in variants {
                let mut row = run_variant(expr, &expected, &args)
                    .with_context(|| format!("running {name}/{n}/{variant}"))?;
                row["case"] = json!(name);
                row["n"] = json!(n);
                row["backend"] = json!(args.backend);
                row["variant"] = json!(variant);
                row["optimization_ns"] = json!(optimization_ns);
                row["same_root_identity"] = json!(same_root);
                row["numerics"] = json!(format!("{numerics:?}"));
                eprintln!(
                    "{name}/{n}/{variant}: {:.2} us",
                    row["timing"]["mean_ns"].as_f64().unwrap_or_default() / 1000.0
                );
                serde_json::to_writer(&mut writer, &row)?;
                writeln!(writer)?;
                writer.flush()?;
            }
        }
    }
    Ok(())
}
