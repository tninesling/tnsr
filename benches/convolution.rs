use std::collections::HashMap;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tnsr::cuda::CudaExecutor;
use tnsr::graph::TensorGraph;
use tnsr::ptx::PtxExecutor;
use tnsr::tensor::{Parameter, TensorExpr};
use tnsr::{Executor, SimpleExecutor};

fn convolution_benchmarks(c: &mut Criterion) {
    let conv_input = Parameter::new(vec![0.1; 3 * 28 * 28], vec![1, 3, 28, 28]);
    let conv_weight = Parameter::new(vec![0.01; 8 * 3 * 3 * 3], vec![8, 3, 3, 3]);
    let conv: TensorGraph<f32> = TensorExpr::from(conv_input)
        .conv2d(conv_weight, 1, 1)
        .into();

    let pool_input = Parameter::new(vec![0.1; 8 * 28 * 28], vec![1, 8, 28, 28]);
    let pool: TensorGraph<f32> = TensorExpr::from(pool_input).max_pool2d(2, 2).into();

    let mut group = c.benchmark_group("convolution");
    group.sample_size(10);
    for (name, graph) in [("conv2d", conv), ("max_pool2d", pool)] {
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

criterion_group!(benches, convolution_benchmarks);
criterion_main!(benches);
