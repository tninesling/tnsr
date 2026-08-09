use std::collections::HashMap;
use std::sync::Arc;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use tnsr::cuda::CudaExecutor;
use tnsr::graph::TensorGraph;
use tnsr::nn::TransformerBlock;
use tnsr::ptx::PtxExecutor;
use tnsr::tensor::Constant;
use tnsr::{Executor, SimpleExecutor};

struct Case {
    name: &'static str,
    batch: usize,
    sequence: usize,
    width: usize,
    heads: usize,
    feed_forward: usize,
}

fn transformer_benchmarks(c: &mut Criterion) {
    let cases = [
        Case {
            name: "tiny",
            batch: 1,
            sequence: 16,
            width: 32,
            heads: 4,
            feed_forward: 128,
        },
        Case {
            name: "small",
            batch: 1,
            sequence: 64,
            width: 64,
            heads: 4,
            feed_forward: 256,
        },
        Case {
            name: "medium",
            batch: 1,
            sequence: 128,
            width: 128,
            heads: 8,
            feed_forward: 512,
        },
    ];
    let mut group = c.benchmark_group("transformer_forward");
    group.sample_size(10);

    for case in cases {
        let block = TransformerBlock::new(case.width, case.heads, case.feed_forward, 1e-5);
        let input = Constant::new(
            vec![0.1; case.batch * case.sequence * case.width],
            vec![case.batch, case.sequence, case.width],
        );
        let graph: Arc<TensorGraph<f32>> = Arc::new(block.forward(input, true).into());

        let cpu_graph = Arc::clone(&graph);
        let mut cpu = SimpleExecutor::new();
        group.bench_function(BenchmarkId::new("cpu", case.name), |benchmark| {
            benchmark
                .iter(|| std::hint::black_box(cpu.execute(&cpu_graph, HashMap::new()).unwrap()));
        });

        let cuda_graph = Arc::clone(&graph);
        let mut cuda = CudaExecutor::new();
        group.bench_function(BenchmarkId::new("static_cuda", case.name), |benchmark| {
            benchmark
                .iter(|| std::hint::black_box(cuda.execute(&cuda_graph, HashMap::new()).unwrap()));
        });

        let compile_graph = Arc::clone(&graph);
        let mut compiler = PtxExecutor::new();
        group.bench_function(
            BenchmarkId::new("tile_ptx_compile_jit", case.name),
            |benchmark| {
                benchmark.iter_batched(
                    || (*compile_graph).clone(),
                    |graph| compiler.compile_owned(graph).unwrap(),
                    BatchSize::PerIteration,
                )
            },
        );

        let ptx_graph = Arc::clone(&graph);
        let mut ptx = PtxExecutor::new();
        ptx.compile_owned((*ptx_graph).clone()).unwrap();
        group.bench_function(BenchmarkId::new("tile_ptx", case.name), |benchmark| {
            benchmark.iter(|| {
                std::hint::black_box(ptx.execute_compiled(&ptx_graph, HashMap::new()).unwrap())
            });
        });
    }
    group.finish();
}

criterion_group!(benches, transformer_benchmarks);
criterion_main!(benches);
