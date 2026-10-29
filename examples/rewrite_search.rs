//! Reproducible paired rewrite measurements and whole-graph cost-model data.
use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tnsr::graph::TensorGraph;
use tnsr::graph::rewrite::{RewriteConfig, RewriteNumerics, optimize_expr_with_numerics};
use tnsr::tensor::{ExprKind, Parameter, TensorExpr, ValueDType};
use tnsr::{Executor, SimpleExecutor};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "cpu")]
    backend: String,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "8,16,32,64,128,256")]
    sizes: String,
    #[arg(long, default_value_t = 16)]
    samples: usize,
    #[arg(long, default_value_t = 5)]
    block_ms: u64,
    #[arg(long)]
    family: Option<String>,
    #[arg(long)]
    reverse_order: bool,
}

struct Case {
    family: &'static str,
    n: usize,
    original: TensorExpr<f32>,
    manual: TensorExpr<f32>,
    numerics: RewriteNumerics,
}

fn tensor(shape: &[usize], seed: usize) -> TensorExpr<f32> {
    Parameter::new(
        (0..shape.iter().product())
            .map(|i| ((i + seed) % 29) as f32 / 29.0 - 0.5)
            .collect(),
        shape.to_vec(),
    )
    .into()
}

fn cases(n: usize) -> Vec<Case> {
    use RewriteNumerics::{PreserveOperations as Exact, RealAlgebra as Real};
    let x = tensor(&[n, n], 0);
    let y = tensor(&[n, n], 7);
    let row = tensor(&[1, n], 1);
    let a = tensor(&[n, n / 4], 2);
    let b = tensor(&[n / 4, n], 3);
    let c = tensor(&[n / 4, n], 4);
    let cast = x.clone().cast(ValueDType::BF16).cast(ValueDType::F32);
    let singleton = tensor(&[n, 1, n, 1], 5);
    let singleton_same = tensor(&[n, 1, 1, n], 6);
    let mut out = Vec::new();
    let mut add = |family, original, manual, numerics| {
        out.push(Case {
            family,
            n,
            original,
            manual,
            numerics,
        })
    };
    add(
        "mixed_layout",
        x.clone().transpose().permute(vec![1, 0]),
        x.clone(),
        Exact,
    );
    add(
        "singleton_reshape",
        singleton.clone().permute(vec![1, 0, 3, 2]),
        singleton.reshape(vec![1, n, 1, n]),
        Exact,
    );
    add(
        "singleton_identity",
        singleton_same.clone().permute(vec![0, 2, 1, 3]),
        singleton_same,
        Exact,
    );
    add(
        "unary_layout",
        x.clone().transpose().relu().transpose(),
        x.clone().relu(),
        Exact,
    );
    add(
        "unary_reshape",
        x.clone()
            .reshape(vec![n / 2, n * 2])
            .relu()
            .reshape(vec![n, n]),
        x.clone().relu(),
        Exact,
    );
    add(
        "opaque_cast_layout",
        cast.clone().transpose().transpose(),
        cast,
        Exact,
    );
    add(
        "unary_broadcast",
        row.clone().broadcast_axis(0, n).exp(),
        row.clone().exp().broadcast_axis(0, n),
        Exact,
    );
    add(
        "mean_broadcast",
        row.clone().broadcast_axis(0, n).reduce_mean(0),
        row.clone(),
        Real,
    );
    add(
        "sum_matmul_rows",
        a.clone().matmul(b.clone()).reduce_sum(0),
        a.clone().reduce_sum(0).matmul(b.clone()),
        Real,
    );
    add(
        "sum_matmul_cols",
        a.clone().matmul(b.clone()).reduce_sum(1),
        a.clone().matmul(b.clone().reduce_sum(1)),
        Real,
    );
    add(
        "mean_matmul_rows",
        a.clone().matmul(b.clone()).reduce_mean(0),
        a.clone().reduce_mean(0).matmul(b.clone()),
        Real,
    );
    add(
        "invariant_reduction",
        (x.clone() * row.clone()).reduce_sum(0),
        x.clone().reduce_sum(0) * row,
        Real,
    );
    // Positive and negative controls, including the prior rejected CUDA rule.
    add(
        "transpose_sink",
        x.clone().transpose() + y.clone().transpose(),
        (x.clone() + y).transpose(),
        Exact,
    );
    add(
        "factor_matmul",
        a.clone().matmul(b.clone()) + a.clone().matmul(c.clone()),
        a.matmul(b + c),
        Real,
    );
    let a = tensor(&[1, n], 8);
    let b = tensor(&[n, 1], 9);
    let c = tensor(&[n, 1], 10);
    add(
        "factor_skinny",
        a.clone().matmul(b.clone()) + a.clone().matmul(c.clone()),
        a.matmul(b + c),
        Real,
    );
    out
}

fn elements(expr: &TensorExpr<f32>) -> f64 {
    expr.shape().iter().fold(1.0, |n, &d| n * d as f64)
}

fn heuristic(expr: &TensorExpr<f32>) -> f64 {
    let children = expr.children();
    let own = match expr.kind() {
        ExprKind::Constant { .. }
        | ExprKind::Parameter { .. }
        | ExprKind::Input { .. }
        | ExprKind::NodeRef { .. } => 1.0,
        ExprKind::MatMul { a, .. } => {
            2.0 * elements(expr) * *a.shape().last().unwrap_or(&1) as f64 + 64.0
        }
        ExprKind::Reshape { .. } => 1.0 + expr.shape().len() as f64,
        ExprKind::ReduceAxis { x, .. } => elements(x) + 65.0,
        ExprKind::BroadcastAxis { x, axis } => elements(x) * expr.shape()[*axis] as f64 + 66.0,
        ExprKind::Cast { .. } => return 1.0, // opaque to expression extraction
        _ => 3.0 * children.iter().map(|x| elements(x)).fold(1.0, f64::max) + 64.0,
    };
    own + children.iter().map(|x| heuristic(x)).sum::<f64>()
}

fn features(expr: &TensorExpr<f32>) -> BTreeMap<String, f64> {
    fn visit(expr: &TensorExpr<f32>, seen: &mut HashSet<usize>, f: &mut BTreeMap<String, f64>) {
        let identity = std::ptr::from_ref(expr.kind()) as usize;
        if !seen.insert(identity) {
            return;
        }
        let put = |f: &mut BTreeMap<String, f64>, key: &str, value| {
            *f.entry(key.to_string()).or_default() += value
        };
        put(f, "node_count", 1.0);
        f.entry("max_rank".into())
            .and_modify(|v| *v = v.max(expr.shape().len() as f64))
            .or_insert(expr.shape().len() as f64);
        let children = expr.children();
        let input = children.iter().map(|x| elements(x)).sum::<f64>();
        match expr.kind() {
            ExprKind::MatMul { a, .. } => {
                put(f, "matmul_count", 1.0);
                put(
                    f,
                    "flops",
                    2.0 * elements(expr) * *a.shape().last().unwrap_or(&1) as f64,
                );
            }
            ExprKind::ReduceAxis { x, axis, .. } => {
                put(f, "reduction_count", 1.0);
                put(f, "flops", elements(x));
                put(f, "reduction_extent", x.shape()[*axis] as f64);
                let stride = x.shape()[axis + 1..].iter().fold(1.0, |n, &d| n * d as f64);
                put(f, "reduction_stride", stride);
            }
            ExprKind::Transpose { .. } | ExprKind::Permute { .. } => {
                put(f, "layout_count", 1.0);
                put(f, "layout_elements", elements(expr));
            }
            ExprKind::Reshape { .. } => put(f, "reshape_count", 1.0),
            ExprKind::BroadcastAxis { .. } => {
                put(f, "broadcast_count", 1.0);
                put(f, "broadcast_elements", elements(expr));
            }
            ExprKind::Unary { .. } => {
                put(f, "unary_count", 1.0);
                put(f, "flops", elements(expr));
            }
            ExprKind::Binary { .. } => {
                put(f, "binary_count", 1.0);
                put(f, "flops", elements(expr));
            }
            ExprKind::Cast { .. } => put(f, "cast_count", 1.0),
            _ => put(f, "source_bytes", 4.0 * elements(expr)),
        }
        if !children.is_empty() {
            put(f, "read_bytes", 4.0 * input);
            put(f, "write_bytes", 4.0 * elements(expr));
        }
        for child in children {
            visit(child, seen, f);
        }
    }
    let mut out = BTreeMap::new();
    visit(expr, &mut HashSet::new(), &mut out);
    out.insert("output_elements".into(), elements(expr));
    out
}

fn graph_key(expr: &TensorExpr<f32>) -> String {
    fn hash(expr: &TensorExpr<f32>, h: &mut std::hash::DefaultHasher) {
        expr.shape().hash(h);
        std::mem::discriminant(expr.kind()).hash(h);
        match expr.kind() {
            ExprKind::Parameter { id, .. } => id.hash(h),
            ExprKind::Constant { data } => std::sync::Arc::as_ptr(data).hash(h),
            ExprKind::Input { name } => name.hash(h),
            ExprKind::NodeRef { idx } => idx.hash(h),
            ExprKind::Cast { dtype, .. } => format!("{dtype:?}").hash(h),
            ExprKind::Unary { op, .. } => format!("{op:?}").hash(h),
            ExprKind::Binary { op, .. } => format!("{op:?}").hash(h),
            ExprKind::ReduceAxis { op, axis, .. } => {
                format!("{op:?}").hash(h);
                axis.hash(h);
            }
            ExprKind::BroadcastAxis { axis, .. } => axis.hash(h),
            ExprKind::Permute { axes, .. } => axes.hash(h),
            _ => {}
        }
        for child in expr.children() {
            hash(child, h);
        }
    }
    let mut h = std::hash::DefaultHasher::new();
    hash(expr, &mut h);
    format!("{:016x}", h.finish())
}

fn measure(
    mut run: impl FnMut() -> Result<Vec<f32>>,
    samples: usize,
    block: Duration,
) -> Result<Value> {
    for _ in 0..8 {
        black_box(run()?);
    }
    let start = Instant::now();
    let mut iterations = 0usize;
    while start.elapsed() < block {
        black_box(run()?);
        iterations += 1;
    }
    let iterations = iterations.clamp(1, 10_000);
    let mut timings = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(run()?);
        }
        timings.push(start.elapsed().as_secs_f64() * 1e9 / iterations as f64);
    }
    let mean = timings.iter().sum::<f64>() / samples as f64;
    let variance = timings.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (samples - 1) as f64;
    let interval = 2.2 * (variance / samples as f64).sqrt();
    Ok(
        json!({"mean_ns":mean,"ci95_ns":[(mean-interval).max(0.0),mean+interval],"sample_means_ns":timings,"iterations_per_sample":iterations}),
    )
}

fn verify(actual: &[f32], expected: &[f32], family: &str, variant: &str) -> Result<()> {
    ensure!(
        actual.len() == expected.len(),
        "{family}/{variant}: output length differs"
    );
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        ensure!(
            (a - b).abs() <= 5e-4 * (1.0 + b.abs()),
            "{family}/{variant}[{index}]: {a} vs {b}"
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.samples >= 4, "need at least four timing samples");
    ensure!(args.block_ms > 0, "block_ms must be positive");
    let sizes = args
        .sizes
        .split(',')
        .map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(
        sizes.iter().all(|n| *n >= 4 && n % 4 == 0),
        "sizes must be multiples of four >=4"
    );
    let mut writer =
        BufWriter::new(File::create(&args.output).context("creating benchmark JSONL")?);
    for n in sizes {
        for case in cases(n) {
            if args
                .family
                .as_ref()
                .is_some_and(|filter| !case.family.contains(filter))
            {
                continue;
            }
            let expected = SimpleExecutor::new().execute(
                &TensorGraph::from(case.original.clone()),
                Default::default(),
            )?;
            let optimized = optimize_expr_with_numerics(
                &case.original,
                RewriteConfig::default(),
                case.numerics,
            );
            let mut variants = vec![
                ("original", case.original),
                ("manual", case.manual),
                ("optimized", optimized),
            ];
            let rotation = (n / 4 + case.family.bytes().map(usize::from).sum::<usize>()) % 3;
            variants.rotate_left(rotation);
            if args.reverse_order {
                variants.reverse();
            }
            for (variant, expr) in variants {
                let graph = TensorGraph::from(expr.clone());
                let f = features(&expr);
                let (timing, f) = if args.backend == "cpu" {
                    let mut exec = SimpleExecutor::new();
                    let actual = exec.execute(&graph, Default::default())?;
                    verify(&actual, &expected, case.family, variant)?;
                    (
                        measure(
                            || exec.execute(&graph, Default::default()),
                            args.samples,
                            Duration::from_millis(args.block_ms),
                        )?,
                        f,
                    )
                } else if args.backend == "cuda" {
                    #[cfg(feature = "cuda")]
                    {
                        if case.family == "opaque_cast_layout" {
                            let root = *graph.toposort().last().context("empty mixed graph")?;
                            let mut exec = tnsr::ptx::mixed::MixedExecutor::new(
                                &graph,
                                tnsr::tile::MatMulPrecision::StrictF32,
                                false,
                            )?;
                            exec.upload(Default::default())?;
                            exec.run()?;
                            verify(&exec.read(root)?, &expected, case.family, variant)?;
                            let timing = measure(
                                || {
                                    exec.run()?;
                                    exec.read(root)
                                },
                                args.samples,
                                Duration::from_millis(args.block_ms),
                            )?;
                            let mut f = f;
                            f.insert("mixed_executor".into(), 1.0);
                            f.insert("launch_count".into(), exec.info().kernels as f64);
                            f.insert("generated_kernel_count".into(), exec.info().kernels as f64);
                            f.insert(
                                "materialized_bytes".into(),
                                (exec.info().f32_bytes + exec.info().bf16_bytes) as f64,
                            );
                            (timing, f)
                        } else {
                            let mut exec = tnsr::ptx::PtxExecutor::<tnsr::ptx::F32>::try_new()?;
                            exec.set_matmul_precision(tnsr::tile::MatMulPrecision::StrictF32);
                            exec.compile(&graph)?;
                            let actual = exec.execute_compiled(&graph, Default::default())?;
                            verify(&actual, &expected, case.family, variant)?;
                            let timing = measure(
                                || exec.execute_compiled(&graph, Default::default()),
                                args.samples,
                                Duration::from_millis(args.block_ms),
                            )?;
                            let m = exec.execution_metrics();
                            let mut f = f;
                            f.insert("launch_count".into(), m.kernel_launches as f64);
                            f.insert("materialized_bytes".into(), m.materialized_bytes as f64);
                            f.insert(
                                "intermediate_bytes".into(),
                                m.intermediate_materialized_bytes as f64,
                            );
                            f.insert("device_copies".into(), m.device_copies as f64);
                            f.insert(
                                "generated_kernel_count".into(),
                                exec.compile_metrics().generated_kernels as f64,
                            );
                            (timing, f)
                        }
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        bail!("CUDA benchmarks require --features cuda");
                    }
                } else {
                    bail!("unknown backend {}", args.backend);
                };
                let row = json!({"schema_version":1,"backend":args.backend,"device":if args.backend=="cpu" {"AMD Ryzen 7 3700X"} else {"NVIDIA RTX 4080"},"precision":"strict_f32","family":case.family,"shape_key":format!("n{}",case.n),"case_id":format!("{}/{}",case.family,case.n),"variant":variant,"graph_key":graph_key(&expr),"mean_ns":timing["mean_ns"],"ci95_ns":timing["ci95_ns"],"sample_means_ns":timing["sample_means_ns"],"iterations_per_sample":timing["iterations_per_sample"],"heuristic_cost":heuristic(&expr),"features":f,"shape":expr.shape(),"numerics":format!("{:?}",case.numerics),"order_rotation":rotation,"order_reversed":args.reverse_order});
                serde_json::to_writer(&mut writer, &row)?;
                writeln!(writer)?;
                writer.flush()?;
                eprintln!(
                    "{} {}/{} {variant}: {:.3} us",
                    args.backend,
                    case.family,
                    n,
                    timing["mean_ns"].as_f64().unwrap_or(0.0) / 1000.0
                );
            }
        }
    }
    Ok(())
}
