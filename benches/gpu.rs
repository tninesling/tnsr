use std::sync::Arc;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use tnsr::Executor;
use tnsr::cuda::CudaExecutor;
use tnsr::graph::TensorGraph;
use tnsr::ptx::{F32, PtxExecutor as GenericPtxExecutor};
use tnsr::tensor::{Constant, TensorExpr};

type PtxExecutor = GenericPtxExecutor<F32>;

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
            let mut graph = TensorGraph::<f32>::new();
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
            let mut graph = TensorGraph::<f32>::new();
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
            let mut graph = TensorGraph::<f32>::new();
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
            let mut graph = TensorGraph::<f32>::new();
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
            let mut graph = TensorGraph::<f32>::new();
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
            let mut graph = TensorGraph::<f32>::new();
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
            let mut graph = TensorGraph::<f32>::new();
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
            let mut graph = TensorGraph::<f32>::new();
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
            let mut graph = TensorGraph::<f32>::new();
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

pub fn fusion_benches(c: &mut Criterion) {
    let sizes: [usize; 4] = [32, 256, 512, 1024];

    // Benchmark: Chain of unary operations (exp -> log -> relu -> neg)
    // This is highly memory-bound and should benefit significantly from fusion.
    // Without fusion: 4 kernel launches, 3 intermediate memory reads/writes
    // With fusion: 1 kernel launch, no intermediate memory traffic
    {
        let mut group = c.benchmark_group("fusion_unary_chain");
        for &n in &sizes {
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 2) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());

            // Build expression: exp -> log -> relu -> neg
            let expr = TensorExpr::from(a.clone()).exp().log().relu();
            let expr = -expr;

            // Build graph with fusion enabled (when feature is on)
            #[cfg(feature = "fusion")]
            {
                let mut graph = TensorGraph::<f32>::new();
                expr.lower_to_graph(&mut graph);
                graph.apply_fusion();
                let graph = Arc::new(graph);

                {
                    let graph_cuda = Arc::clone(&graph);
                    group.bench_function(BenchmarkId::new("cuda_fused", format!("{n}x{n}")), |b| {
                        let mut exec = CudaExecutor::new();
                        b.iter(|| {
                            let _ = std::hint::black_box(
                                exec.execute(&*graph_cuda, Default::default()),
                            );
                        });
                    });
                }

                {
                    let graph_ptx = Arc::clone(&graph);
                    group.bench_function(BenchmarkId::new("ptx_fused", format!("{n}x{n}")), |b| {
                        let mut exec = PtxExecutor::new();
                        b.iter(|| {
                            let _ =
                                std::hint::black_box(exec.execute(&*graph_ptx, Default::default()));
                        });
                    });
                }
            }

            // Build graph without fusion (baseline)
            {
                let mut graph = TensorGraph::<f32>::new();
                expr.lower_to_graph(&mut graph);
                let graph = Arc::new(graph);

                {
                    let graph_cuda = Arc::clone(&graph);
                    group.bench_function(
                        BenchmarkId::new("cuda_unfused", format!("{n}x{n}")),
                        |b| {
                            let mut exec = CudaExecutor::new();
                            b.iter(|| {
                                let _ = std::hint::black_box(
                                    exec.execute(&*graph_cuda, Default::default()),
                                );
                            });
                        },
                    );
                }

                {
                    let graph_ptx = Arc::clone(&graph);
                    group.bench_function(
                        BenchmarkId::new("ptx_unfused", format!("{n}x{n}")),
                        |b| {
                            let mut exec = PtxExecutor::new();
                            b.iter(|| {
                                let _ = std::hint::black_box(
                                    exec.execute(&*graph_ptx, Default::default()),
                                );
                            });
                        },
                    );
                }
            }
        }
        group.finish();
    }

    // Benchmark: Longer chain (6 operations)
    // Even more memory-bound, should show larger fusion benefits
    {
        let mut group = c.benchmark_group("fusion_long_chain");
        for &n in &sizes {
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 2) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![0.5f32; n * n], shape.clone());

            // Build expression: neg -> exp -> log -> relu -> exp -> log
            let expr = TensorExpr::from(a.clone());
            let expr = -expr;
            let expr = expr.exp().log().relu().exp().log();

            // Build graph with fusion enabled (when feature is on)
            #[cfg(feature = "fusion")]
            {
                let mut graph = TensorGraph::<f32>::new();
                expr.lower_to_graph(&mut graph);
                graph.apply_fusion();
                let graph = Arc::new(graph);

                {
                    let graph_cuda = Arc::clone(&graph);
                    group.bench_function(BenchmarkId::new("cuda_fused", format!("{n}x{n}")), |b| {
                        let mut exec = CudaExecutor::new();
                        b.iter(|| {
                            let _ = std::hint::black_box(
                                exec.execute(&*graph_cuda, Default::default()),
                            );
                        });
                    });
                }

                {
                    let graph_ptx = Arc::clone(&graph);
                    group.bench_function(BenchmarkId::new("ptx_fused", format!("{n}x{n}")), |b| {
                        let mut exec = PtxExecutor::new();
                        b.iter(|| {
                            let _ =
                                std::hint::black_box(exec.execute(&*graph_ptx, Default::default()));
                        });
                    });
                }
            }

            // Build graph without fusion (baseline)
            {
                let mut graph = TensorGraph::<f32>::new();
                expr.lower_to_graph(&mut graph);
                let graph = Arc::new(graph);

                {
                    let graph_cuda = Arc::clone(&graph);
                    group.bench_function(
                        BenchmarkId::new("cuda_unfused", format!("{n}x{n}")),
                        |b| {
                            let mut exec = CudaExecutor::new();
                            b.iter(|| {
                                let _ = std::hint::black_box(
                                    exec.execute(&*graph_cuda, Default::default()),
                                );
                            });
                        },
                    );
                }

                {
                    let graph_ptx = Arc::clone(&graph);
                    group.bench_function(
                        BenchmarkId::new("ptx_unfused", format!("{n}x{n}")),
                        |b| {
                            let mut exec = PtxExecutor::new();
                            b.iter(|| {
                                let _ = std::hint::black_box(
                                    exec.execute(&*graph_ptx, Default::default()),
                                );
                            });
                        },
                    );
                }
            }
        }
        group.finish();
    }
}

// End-to-end execution of the same attention graph; compilation is outside timing.
fn attention_benches(c: &mut Criterion) {
    use tnsr::nn::scaled_dot_product_attention;
    use tnsr::ptx::PtxReductionMode;
    use tnsr::tile::MatMulPrecision;
    let mut group = c.benchmark_group("attention");
    for (sequence, width) in [(64, 32), (128, 64), (256, 64), (512, 64)] {
        let input = |phase| {
            TensorExpr::constant(
                (0..sequence * width)
                    .map(|i| ((i + phase) % 29) as f32 / 29.0 - 0.5)
                    .collect(),
                vec![1, sequence, width],
            )
        };
        for causal in [false, true] {
            let graph: TensorGraph<f32> =
                scaled_dot_product_attention(input(0), input(3), input(7), causal).into();
            for mode in [PtxReductionMode::Strict, PtxReductionMode::Online] {
                let mut executor = PtxExecutor::try_new_with_reduction_mode(mode).unwrap();
                executor.set_matmul_precision(MatMulPrecision::StrictF32);
                executor.compile(&graph).unwrap();
                group.bench_function(
                    BenchmarkId::new(
                        format!("{mode:?}"),
                        format!("{sequence}x{width}/causal={causal}"),
                    ),
                    |b| {
                        b.iter(|| {
                            std::hint::black_box(
                                executor
                                    .execute_compiled(&graph, Default::default())
                                    .unwrap(),
                            )
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches_group, benches, fusion_benches, attention_benches);
criterion_main!(benches_group);
