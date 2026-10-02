use std::collections::HashMap;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
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

criterion_group!(benches, fusion_benchmarks);
criterion_main!(benches);
