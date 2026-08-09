use std::collections::HashMap;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tnsr::cuda::CudaExecutor;
use tnsr::graph::TensorGraph;
use tnsr::ptx::PtxExecutor;
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

fn fusion_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_baseline");
    group.sample_size(10);
    for (name, graph) in [
        ("pointwise_diamond", pointwise_diamond()),
        ("reduction_chain", reduction_chain()),
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
    }
    group.finish();
}

criterion_group!(benches, fusion_benchmarks);
criterion_main!(benches);
