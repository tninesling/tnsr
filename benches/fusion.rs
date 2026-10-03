use std::collections::HashMap;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tnsr::cuda::CudaExecutor;
use tnsr::graph::TensorGraph;
use tnsr::nn::{layer_norm, softmax};
use tnsr::ptx::{PtxExecutor, PtxReductionMode};
use tnsr::tensor::{Parameter, TensorExpr};
use tnsr::{Executor, SimpleExecutor};

fn pointwise_diamond() -> TensorGraph<f32> {
    let input = Parameter::new(vec![0.25; 256 * 1024], vec![256, 1024]);
    let shared = (-TensorExpr::from(input)).exp();
    (shared.clone().log() + shared.relu()).into()
}

fn reduction_chain() -> TensorGraph<f32> {
    let input = Parameter::new(vec![0.25; 256 * 1024], vec![256, 1024]);
    TensorExpr::from(input)
        .relu()
        .exp()
        .reduce_sum(1)
        .log()
        .into()
}

fn subgroup_reduction_chain() -> TensorGraph<f32> {
    let input = Parameter::new(vec![0.25; 2048 * 128], vec![2048, 128]);
    TensorExpr::from(input)
        .relu()
        .exp()
        .reduce_sum(1)
        .log()
        .into()
}

fn matmul_epilogue() -> TensorGraph<f32> {
    let input = Parameter::new(vec![0.25; 128 * 128], vec![128, 128]);
    let weight = Parameter::new(vec![0.5; 128 * 128], vec![128, 128]);
    let bias = Parameter::new(vec![0.1; 128], vec![128]);
    (TensorExpr::from(input).matmul(TensorExpr::from(weight)) + TensorExpr::from(bias))
        .relu()
        .into()
}

fn bias_broadcast_region() -> TensorGraph<f32> {
    let input = Parameter::new(vec![0.25; 256 * 1024], vec![256, 1024]);
    let bias = Parameter::new(vec![0.5; 1024], vec![1024]);
    (TensorExpr::from(input) + TensorExpr::from(bias))
        .relu()
        .into()
}

fn permutation_region() -> TensorGraph<f32> {
    let input = Parameter::new(vec![0.25; 512 * 512], vec![512, 512]);
    let residual = Parameter::new(vec![0.5; 512 * 512], vec![512, 512]);
    (TensorExpr::from(input).permute(vec![1, 0]) + TensorExpr::from(residual))
        .relu()
        .into()
}

fn softmax_reduction_region() -> TensorGraph<f32> {
    let input = Parameter::new(vec![0.25; 256 * 1024], vec![256, 1024]);
    softmax(TensorExpr::from(input), 1).into()
}

fn layer_norm_reduction_region() -> TensorGraph<f32> {
    let input = Parameter::new(vec![0.25; 256 * 1024], vec![256, 1024]);
    let weight = Parameter::new(vec![1.0; 1024], vec![1024]);
    let bias = Parameter::new(vec![0.0; 1024], vec![1024]);
    layer_norm(input, 1, weight, bias, 1e-5).into()
}

fn fusion_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_baseline");
    group.sample_size(10);
    for (name, graph) in [
        ("pointwise_diamond", pointwise_diamond()),
        ("reduction_chain", reduction_chain()),
        ("subgroup_reduction_chain", subgroup_reduction_chain()),
        ("matmul_epilogue", matmul_epilogue()),
        ("bias_broadcast_region", bias_broadcast_region()),
        ("permutation_region", permutation_region()),
        ("softmax_reduction_region", softmax_reduction_region()),
        ("layer_norm_reduction_region", layer_norm_reduction_region()),
    ] {
        let mut cpu = SimpleExecutor::new();
        group.bench_function(BenchmarkId::new("cpu", name), |benchmark| {
            benchmark.iter(|| cpu.execute(&graph, HashMap::new()).unwrap())
        });

        let mut cuda = CudaExecutor::new();
        group.bench_function(BenchmarkId::new("static_cuda", name), |benchmark| {
            benchmark.iter(|| cuda.execute(&graph, HashMap::new()).unwrap())
        });

        let mut ptx = PtxExecutor::new();
        ptx.compile_owned(graph.clone()).unwrap();
        group.bench_function(BenchmarkId::new("tile_ptx", name), |benchmark| {
            benchmark.iter(|| ptx.execute_compiled(&graph, HashMap::new()).unwrap())
        });

        if matches!(
            name,
            "reduction_chain"
                | "subgroup_reduction_chain"
                | "softmax_reduction_region"
                | "layer_norm_reduction_region"
        ) {
            let mut cooperative =
                PtxExecutor::new_with_reduction_mode(PtxReductionMode::DeterministicTree);
            cooperative.compile_owned(graph.clone()).unwrap();
            group.bench_function(
                BenchmarkId::new("tile_ptx_cooperative", name),
                |benchmark| {
                    benchmark.iter(|| {
                        cooperative
                            .execute_compiled(&graph, HashMap::new())
                            .unwrap()
                    })
                },
            );
        }
    }
    group.finish();
}

fn batched_matmul_benchmarks(c: &mut Criterion) {
    use tnsr::tile::MatMulPrecision;
    let mut group = c.benchmark_group("batched_matmul_epilogue");
    group.sample_size(10);
    eprintln!(
        "shape,precision,compile_us,launches,materialized_bytes,intermediate_bytes,estimated_global_bytes,max_absolute_error"
    );
    for (name, batches, m, k, n) in [
        ("transformer_projection", 2, 64, 128, 384),
        ("attention_scores", 8, 64, 32, 64),
        ("gpt_projection", 2, 128, 256, 768),
    ] {
        let input = Parameter::new(
            (0..batches * m * k)
                .map(|index| (index % 29) as f32 / 29.0 - 0.5)
                .collect(),
            vec![batches, m, k],
        );
        let weight = Parameter::new(
            (0..k * n)
                .map(|index| (index % 31) as f32 / 31.0 - 0.5)
                .collect(),
            vec![k, n],
        );
        let bias = Parameter::new(vec![0.1; n], vec![n]);
        let residual = Parameter::new(vec![0.05; batches * m * n], vec![batches, m, n]);
        let graph: TensorGraph<f32> = ((TensorExpr::from(input).matmul(TensorExpr::from(weight))
            + TensorExpr::from(bias))
        .relu()
            + TensorExpr::from(residual))
        .into();
        let expected = SimpleExecutor::new()
            .execute(&graph, HashMap::new())
            .unwrap();
        // Criterion reports elements/s here; one element represents one FLOP.
        group.throughput(Throughput::Elements((2 * batches * m * n * k) as u64));
        for (policy, precision) in [
            ("tf32", MatMulPrecision::AllowTf32),
            ("strict_f32", MatMulPrecision::StrictF32),
        ] {
            let mut executor = PtxExecutor::new();
            executor.set_matmul_precision(precision);
            executor.compile_owned(graph.clone()).unwrap();
            let actual = executor.execute_compiled(&graph, HashMap::new()).unwrap();
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(&actual, &expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            let tolerance = if precision == MatMulPrecision::StrictF32 {
                1e-4
            } else {
                1e-2
            };
            assert!(error <= tolerance, "{name}/{policy}: maximum error {error}");
            let schedule = executor.execution_plan().unwrap().matmul_regions()[0].schedule;
            // Count valid operand loads repeated per output block, bias/residual
            // reads, and output stores. This is a source-level traffic estimate,
            // not a measurement of DRAM transactions or cache behavior.
            let traffic = 4
                * batches
                * (m * k * n.div_ceil(schedule.block_tile.n)
                    + k * n * m.div_ceil(schedule.block_tile.m)
                    + 3 * m * n);
            let metrics = executor.execution_metrics();
            assert_eq!(metrics.kernel_launches, 1);
            assert_eq!(metrics.intermediate_materialized_bytes, 0);
            eprintln!(
                "{name},{policy},{},{},{},{},{traffic},{error}",
                executor.compile_metrics().compile_time.as_micros(),
                metrics.kernel_launches,
                metrics.materialized_bytes,
                metrics.intermediate_materialized_bytes
            );
            group.bench_function(BenchmarkId::new(policy, name), |benchmark| {
                benchmark.iter(|| executor.execute_compiled(&graph, HashMap::new()).unwrap())
            });
        }
    }
    group.finish();
}

criterion_group!(benches, fusion_benchmarks, batched_matmul_benchmarks);
criterion_main!(benches);
