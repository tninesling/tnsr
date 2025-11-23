use std::sync::Arc;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use runtime::Executor;
use runtime::cuda::CudaExecutor;
use runtime::ptx::PtxExecutor;
use tensor::Constant;
use tensor::TensorExpr;

pub fn benches(c: &mut Criterion) {
    let sizes: [usize; 4] = [32, 256, 512, 1024];

    // NOTE: These benchmarks compare CudaExecutor (static PTX) vs PtxExecutor (tiled codegen).
    // CudaExecutor uses pre-compiled PTX kernels, while PtxExecutor compiles graphs to PTX.
    // For application code, prefer using Runtime::new() or Runtime::with_backend(Backend::Cuda).
    // Example:
    //   let mut runtime = Runtime::with_backend(Backend::Cuda).unwrap();
    //   runtime.execute(&graph, inputs).unwrap();

    // Unary: neg
    {
        let mut group = c.benchmark_group("neg");
        for &n in &sizes {
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 2) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());
            let mut graph = tensor::graph::TensorGraph::<f32>::new();
            let expr = -TensorExpr::from(a.clone());
            expr.lower_to_graph(&mut graph);
            let graph = Arc::new(graph);

            {
                let graph_cuda = Arc::clone(&graph);
                group.bench_function(BenchmarkId::new("cuda", format!("{n}x{n}")), |b| {
                    let mut exec = CudaExecutor::new();
                    b.iter(|| {
                        let _ =
                            std::hint::black_box(exec.execute(&*graph_cuda, Default::default()));
                    });
                });
            }

            {
                let graph_ptx = Arc::clone(&graph);
                group.bench_function(BenchmarkId::new("ptx", format!("{n}x{n}")), |b| {
                    let mut exec = PtxExecutor::new();
                    b.iter(|| {
                        let _ = std::hint::black_box(exec.execute(&*graph_ptx, Default::default()));
                    });
                });
            }
        }
        group.finish();
    }

    // Unary: exp
    {
        let mut group = c.benchmark_group("exp");
        for &n in &sizes {
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 2) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());
            let mut graph = tensor::graph::TensorGraph::<f32>::new();
            let expr = TensorExpr::from(a.clone()).exp();
            expr.lower_to_graph(&mut graph);
            let graph = Arc::new(graph);

            {
                let graph_cuda = Arc::clone(&graph);
                group.bench_function(BenchmarkId::new("cuda", format!("{n}x{n}")), |b| {
                    let mut exec = CudaExecutor::new();
                    b.iter(|| {
                        let _ =
                            std::hint::black_box(exec.execute(&*graph_cuda, Default::default()));
                    });
                });
            }

            {
                let graph_ptx = Arc::clone(&graph);
                group.bench_function(BenchmarkId::new("ptx", format!("{n}x{n}")), |b| {
                    let mut exec = PtxExecutor::new();
                    b.iter(|| {
                        let _ = std::hint::black_box(exec.execute(&*graph_ptx, Default::default()));
                    });
                });
            }
        }
        group.finish();
    }

    // Unary: log
    {
        let mut group = c.benchmark_group("log");
        for &n in &sizes {
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 2) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());
            let mut graph = tensor::graph::TensorGraph::<f32>::new();
            let expr = TensorExpr::from(a.clone()).log();
            expr.lower_to_graph(&mut graph);
            let graph = Arc::new(graph);

            {
                let graph_cuda = Arc::clone(&graph);
                group.bench_function(BenchmarkId::new("cuda", format!("{n}x{n}")), |b| {
                    let mut exec = CudaExecutor::new();
                    b.iter(|| {
                        let _ =
                            std::hint::black_box(exec.execute(&*graph_cuda, Default::default()));
                    });
                });
            }

            {
                let graph_ptx = Arc::clone(&graph);
                group.bench_function(BenchmarkId::new("ptx", format!("{n}x{n}")), |b| {
                    let mut exec = PtxExecutor::new();
                    b.iter(|| {
                        let _ = std::hint::black_box(exec.execute(&*graph_ptx, Default::default()));
                    });
                });
            }
        }
        group.finish();
    }

    // Unary: relu
    {
        let mut group = c.benchmark_group("relu");
        for &n in &sizes {
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 2) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());
            let mut graph = tensor::graph::TensorGraph::<f32>::new();
            let expr = TensorExpr::from(a.clone()).relu();
            expr.lower_to_graph(&mut graph);
            let graph = Arc::new(graph);

            {
                let graph_cuda = Arc::clone(&graph);
                group.bench_function(BenchmarkId::new("cuda", format!("{n}x{n}")), |b| {
                    let mut exec = CudaExecutor::new();
                    b.iter(|| {
                        let _ =
                            std::hint::black_box(exec.execute(&*graph_cuda, Default::default()));
                    });
                });
            }

            {
                let graph_ptx = Arc::clone(&graph);
                group.bench_function(BenchmarkId::new("ptx", format!("{n}x{n}")), |b| {
                    let mut exec = PtxExecutor::new();
                    b.iter(|| {
                        let _ = std::hint::black_box(exec.execute(&*graph_ptx, Default::default()));
                    });
                });
            }
        }
        group.finish();
    }

    // Binary: add
    {
        let mut group = c.benchmark_group("add");
        for &n in &sizes {
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 3) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());
            let b = Constant::new(vec![0.5f32; n * n], shape.clone());
            let mut graph = tensor::graph::TensorGraph::<f32>::new();
            let expr = TensorExpr::from(a.clone()) + TensorExpr::from(b.clone());
            expr.lower_to_graph(&mut graph);
            let graph = Arc::new(graph);

            {
                let graph_cuda = Arc::clone(&graph);
                group.bench_function(BenchmarkId::new("cuda", format!("{n}x{n}")), |b| {
                    let mut exec = CudaExecutor::new();
                    b.iter(|| {
                        let _ =
                            std::hint::black_box(exec.execute(&*graph_cuda, Default::default()));
                    });
                });
            }

            {
                let graph_ptx = Arc::clone(&graph);
                group.bench_function(BenchmarkId::new("ptx", format!("{n}x{n}")), |b| {
                    let mut exec = PtxExecutor::new();
                    b.iter(|| {
                        let _ = std::hint::black_box(exec.execute(&*graph_ptx, Default::default()));
                    });
                });
            }
        }
        group.finish();
    }

    // Binary: sub
    {
        let mut group = c.benchmark_group("sub");
        for &n in &sizes {
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 3) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());
            let b = Constant::new(vec![0.5f32; n * n], shape.clone());
            let mut graph = tensor::graph::TensorGraph::<f32>::new();
            let expr = TensorExpr::from(a.clone()) - TensorExpr::from(b.clone());
            expr.lower_to_graph(&mut graph);
            let graph = Arc::new(graph);

            {
                let graph_cuda = Arc::clone(&graph);
                group.bench_function(BenchmarkId::new("cuda", format!("{n}x{n}")), |b| {
                    let mut exec = CudaExecutor::new();
                    b.iter(|| {
                        let _ =
                            std::hint::black_box(exec.execute(&*graph_cuda, Default::default()));
                    });
                });
            }

            {
                let graph_ptx = Arc::clone(&graph);
                group.bench_function(BenchmarkId::new("ptx", format!("{n}x{n}")), |b| {
                    let mut exec = PtxExecutor::new();
                    b.iter(|| {
                        let _ = std::hint::black_box(exec.execute(&*graph_ptx, Default::default()));
                    });
                });
            }
        }
        group.finish();
    }

    // Binary: mul
    {
        let mut group = c.benchmark_group("mul");
        for &n in &sizes {
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 3) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());
            let b = Constant::new(vec![0.5f32; n * n], shape.clone());
            let mut graph = tensor::graph::TensorGraph::<f32>::new();
            let expr = TensorExpr::from(a.clone()) * TensorExpr::from(b.clone());
            expr.lower_to_graph(&mut graph);
            let graph = Arc::new(graph);

            {
                let graph_cuda = Arc::clone(&graph);
                group.bench_function(BenchmarkId::new("cuda", format!("{n}x{n}")), |b| {
                    let mut exec = CudaExecutor::new();
                    b.iter(|| {
                        let _ =
                            std::hint::black_box(exec.execute(&*graph_cuda, Default::default()));
                    });
                });
            }

            {
                let graph_ptx = Arc::clone(&graph);
                group.bench_function(BenchmarkId::new("ptx", format!("{n}x{n}")), |b| {
                    let mut exec = PtxExecutor::new();
                    b.iter(|| {
                        let _ = std::hint::black_box(exec.execute(&*graph_ptx, Default::default()));
                    });
                });
            }
        }
        group.finish();
    }

    // Binary: div
    {
        let mut group = c.benchmark_group("div");
        for &n in &sizes {
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 3) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());
            let b = Constant::new(vec![0.5f32; n * n], shape.clone());
            let mut graph = tensor::graph::TensorGraph::<f32>::new();
            let expr = TensorExpr::from(a.clone()) / TensorExpr::from(b.clone());
            expr.lower_to_graph(&mut graph);
            let graph = Arc::new(graph);

            {
                let graph_cuda = Arc::clone(&graph);
                group.bench_function(BenchmarkId::new("cuda", format!("{n}x{n}")), |b| {
                    let mut exec = CudaExecutor::new();
                    b.iter(|| {
                        let _ =
                            std::hint::black_box(exec.execute(&*graph_cuda, Default::default()));
                    });
                });
            }

            {
                let graph_ptx = Arc::clone(&graph);
                group.bench_function(BenchmarkId::new("ptx", format!("{n}x{n}")), |b| {
                    let mut exec = PtxExecutor::new();
                    b.iter(|| {
                        let _ = std::hint::black_box(exec.execute(&*graph_ptx, Default::default()));
                    });
                });
            }
        }
        group.finish();
    }

    // MatMul - the key comparison benchmark
    {
        let mut group = c.benchmark_group("matmul");
        for &n in &sizes {
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 3) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());
            let b = Constant::new(vec![0.5f32; n * n], shape.clone());
            let mut graph = tensor::graph::TensorGraph::<f32>::new();
            let expr = TensorExpr::from(a.clone()).matmul(TensorExpr::from(b.clone()));
            expr.lower_to_graph(&mut graph);
            let graph = Arc::new(graph);

            {
                let graph_cuda = Arc::clone(&graph);
                group.bench_function(BenchmarkId::new("cuda", format!("{n}x{n}")), |b| {
                    let mut exec = CudaExecutor::new();
                    b.iter(|| {
                        let _ =
                            std::hint::black_box(exec.execute(&*graph_cuda, Default::default()));
                    });
                });
            }

            {
                let graph_ptx = Arc::clone(&graph);
                group.bench_function(BenchmarkId::new("ptx", format!("{n}x{n}")), |b| {
                    let mut exec = PtxExecutor::new();
                    b.iter(|| {
                        let _ = std::hint::black_box(exec.execute(&*graph_ptx, Default::default()));
                    });
                });
            }
        }
        group.finish();
    }
}

criterion_group!(benches_group, benches);
criterion_main!(benches_group);
