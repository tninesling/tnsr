use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tnsr::tensor::{Constant, TensorExpr};
use tnsr::{Executor, SimpleExecutor};

pub fn benches(c: &mut Criterion) {
    let sizes: [usize; 2] = [32, 256];

    // Unary: neg
    {
        let mut group = c.benchmark_group("neg");
        for &n in &sizes {
            let mut exec = SimpleExecutor::new();
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 2) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());
            let mut graph = tnsr::graph::TensorGraph::<f32>::new();
            let expr = -TensorExpr::from(a.clone());
            expr.lower_to_graph(&mut graph);
            let graph = Arc::new(graph);

            let graph_simple = Arc::clone(&graph);
            group.bench_with_input(
                BenchmarkId::new("simple", format!("{n}x{n}")),
                &n,
                move |b, &_| {
                    b.iter(|| {
                        let _ =
                            std::hint::black_box(exec.execute(&*graph_simple, Default::default()));
                    });
                },
            );
        }
        group.finish();
    }

    // Unary: exp
    {
        let mut group = c.benchmark_group("exp");
        for &n in &sizes {
            let mut exec = SimpleExecutor::new();
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 2) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());
            let mut graph = tnsr::graph::TensorGraph::<f32>::new();
            let expr = TensorExpr::from(a.clone()).exp();
            expr.lower_to_graph(&mut graph);
            let graph = Arc::new(graph);

            let graph_simple = Arc::clone(&graph);
            group.bench_with_input(
                BenchmarkId::new("simple", format!("{n}x{n}")),
                &n,
                move |b, &_| {
                    b.iter(|| {
                        let _ =
                            std::hint::black_box(exec.execute(&*graph_simple, Default::default()));
                    });
                },
            );
        }
        group.finish();
    }

    // Unary: log
    {
        let mut group = c.benchmark_group("log");
        for &n in &sizes {
            let mut exec = SimpleExecutor::new();
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 2) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());
            let mut graph = tnsr::graph::TensorGraph::<f32>::new();
            let expr = TensorExpr::from(a.clone()).log();
            expr.lower_to_graph(&mut graph);
            let graph = Arc::new(graph);

            let graph_simple = Arc::clone(&graph);
            group.bench_with_input(
                BenchmarkId::new("simple", format!("{n}x{n}")),
                &n,
                move |b, &_| {
                    b.iter(|| {
                        let _ =
                            std::hint::black_box(exec.execute(&*graph_simple, Default::default()));
                    });
                },
            );
        }
        group.finish();
    }

    // Unary: relu
    {
        let mut group = c.benchmark_group("relu");
        for &n in &sizes {
            let mut exec = SimpleExecutor::new();
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 2) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());
            let mut graph = tnsr::graph::TensorGraph::<f32>::new();
            let expr = TensorExpr::from(a.clone()).relu();
            expr.lower_to_graph(&mut graph);
            let graph = Arc::new(graph);

            let graph_simple = Arc::clone(&graph);
            group.bench_with_input(
                BenchmarkId::new("simple", format!("{n}x{n}")),
                &n,
                move |b, &_| {
                    b.iter(|| {
                        let _ =
                            std::hint::black_box(exec.execute(&*graph_simple, Default::default()));
                    });
                },
            );
        }
        group.finish();
    }

    // Binary: add
    {
        let mut group = c.benchmark_group("add");
        for &n in &sizes {
            let mut exec = SimpleExecutor::new();
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 3) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());
            let b = Constant::new(vec![0.5f32; n * n], shape.clone());
            let mut graph = tnsr::graph::TensorGraph::<f32>::new();
            let expr = TensorExpr::from(a.clone()) + TensorExpr::from(b.clone());
            expr.lower_to_graph(&mut graph);
            let graph = Arc::new(graph);

            let graph_simple = Arc::clone(&graph);
            group.bench_with_input(
                BenchmarkId::new("simple", format!("{n}x{n}")),
                &n,
                move |bch, &_| {
                    bch.iter(|| {
                        let _ =
                            std::hint::black_box(exec.execute(&*graph_simple, Default::default()));
                    });
                },
            );
        }
        group.finish();
    }

    // Binary: sub
    {
        let mut group = c.benchmark_group("sub");
        for &n in &sizes {
            let mut exec = SimpleExecutor::new();
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 3) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());
            let b = Constant::new(vec![0.5f32; n * n], shape.clone());
            let mut graph = tnsr::graph::TensorGraph::<f32>::new();
            let expr = TensorExpr::from(a.clone()) - TensorExpr::from(b.clone());
            expr.lower_to_graph(&mut graph);
            let graph = Arc::new(graph);

            let graph_simple = Arc::clone(&graph);
            group.bench_with_input(
                BenchmarkId::new("simple", format!("{n}x{n}")),
                &n,
                move |bch, &_| {
                    bch.iter(|| {
                        let _ =
                            std::hint::black_box(exec.execute(&*graph_simple, Default::default()));
                    });
                },
            );
        }
        group.finish();
    }

    // Binary: mul
    {
        let mut group = c.benchmark_group("mul");
        for &n in &sizes {
            let mut exec = SimpleExecutor::new();
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 3) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());
            let b = Constant::new(vec![0.5f32; n * n], shape.clone());
            let mut graph = tnsr::graph::TensorGraph::<f32>::new();
            let expr = TensorExpr::from(a.clone()) * TensorExpr::from(b.clone());
            expr.lower_to_graph(&mut graph);
            let graph = Arc::new(graph);

            let graph_simple = Arc::clone(&graph);
            group.bench_with_input(
                BenchmarkId::new("simple", format!("{n}x{n}")),
                &n,
                move |bch, &_| {
                    bch.iter(|| {
                        let _ =
                            std::hint::black_box(exec.execute(&*graph_simple, Default::default()));
                    });
                },
            );
        }
        group.finish();
    }

    // Binary: div
    {
        let mut group = c.benchmark_group("div");
        for &n in &sizes {
            let mut exec = SimpleExecutor::new();
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 3) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());
            let b = Constant::new(vec![0.5f32; n * n], shape.clone());
            let mut graph = tnsr::graph::TensorGraph::<f32>::new();
            let expr = TensorExpr::from(a.clone()) / TensorExpr::from(b.clone());
            expr.lower_to_graph(&mut graph);
            let graph = Arc::new(graph);

            let graph_simple = Arc::clone(&graph);
            group.bench_with_input(
                BenchmarkId::new("simple", format!("{n}x{n}")),
                &n,
                move |bch, &_| {
                    bch.iter(|| {
                        let _ =
                            std::hint::black_box(exec.execute(&*graph_simple, Default::default()));
                    });
                },
            );
        }
        group.finish();
    }
}

pub fn optimization_benches(c: &mut Criterion) {
    let sizes: [usize; 2] = [32, 256];

    // Benchmark: exp(log(x)) with and without optimization
    {
        let mut group = c.benchmark_group("optimize_exp_log");
        for &n in &sizes {
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>()) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.5f32; n * n], shape.clone());

            // Without optimization - exp(log(x))
            {
                let mut exec = SimpleExecutor::new();
                let mut graph = tnsr::graph::TensorGraph::<f32>::new();
                let expr = TensorExpr::from(a.clone()).log().exp();
                expr.lower_to_graph(&mut graph);
                let graph = Arc::new(graph);
                let graph_clone = Arc::clone(&graph);

                group.bench_with_input(
                    BenchmarkId::new("unoptimized", format!("{n}x{n}")),
                    &n,
                    move |b, &_| {
                        b.iter(|| {
                            let _ = std::hint::black_box(
                                exec.execute(&*graph_clone, Default::default()),
                            );
                        });
                    },
                );
            }

            // With optimization - should simplify to x
            {
                let mut exec = SimpleExecutor::new();
                let mut graph = tnsr::graph::TensorGraph::<f32>::new();
                let expr = TensorExpr::from(a.clone()).log().exp();
                graph.add_expr(&expr, true);
                let graph = Arc::new(graph);
                let graph_clone = Arc::clone(&graph);

                group.bench_with_input(
                    BenchmarkId::new("optimized", format!("{n}x{n}")),
                    &n,
                    move |b, &_| {
                        b.iter(|| {
                            let _ = std::hint::black_box(
                                exec.execute(&*graph_clone, Default::default()),
                            );
                        });
                    },
                );
            }
        }
        group.finish();
    }

    // Benchmark: transpose(transpose(x)) with and without optimization
    {
        let mut group = c.benchmark_group("optimize_transpose_transpose");
        for &n in &sizes {
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>()) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());

            // Without optimization - transpose(transpose(x))
            {
                let mut exec = SimpleExecutor::new();
                let mut graph = tnsr::graph::TensorGraph::<f32>::new();
                let expr = TensorExpr::from(a.clone()).transpose().transpose();
                expr.lower_to_graph(&mut graph);
                let graph = Arc::new(graph);
                let graph_clone = Arc::clone(&graph);

                group.bench_with_input(
                    BenchmarkId::new("unoptimized", format!("{n}x{n}")),
                    &n,
                    move |b, &_| {
                        b.iter(|| {
                            let _ = std::hint::black_box(
                                exec.execute(&*graph_clone, Default::default()),
                            );
                        });
                    },
                );
            }

            // With optimization - should simplify to x
            {
                let mut exec = SimpleExecutor::new();
                let mut graph = tnsr::graph::TensorGraph::<f32>::new();
                let expr = TensorExpr::from(a.clone()).transpose().transpose();
                graph.add_expr(&expr, true);
                let graph = Arc::new(graph);
                let graph_clone = Arc::clone(&graph);

                group.bench_with_input(
                    BenchmarkId::new("optimized", format!("{n}x{n}")),
                    &n,
                    move |b, &_| {
                        b.iter(|| {
                            let _ = std::hint::black_box(
                                exec.execute(&*graph_clone, Default::default()),
                            );
                        });
                    },
                );
            }
        }
        group.finish();
    }

    // Benchmark: -(-x) with and without optimization
    {
        let mut group = c.benchmark_group("optimize_double_neg");
        for &n in &sizes {
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>()) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());

            // Without optimization - -(-x)
            {
                let mut exec = SimpleExecutor::new();
                let mut graph = tnsr::graph::TensorGraph::<f32>::new();
                let expr = TensorExpr::from(a.clone());
                let expr = -(-expr);
                expr.lower_to_graph(&mut graph);
                let graph = Arc::new(graph);
                let graph_clone = Arc::clone(&graph);

                group.bench_with_input(
                    BenchmarkId::new("unoptimized", format!("{n}x{n}")),
                    &n,
                    move |b, &_| {
                        b.iter(|| {
                            let _ = std::hint::black_box(
                                exec.execute(&*graph_clone, Default::default()),
                            );
                        });
                    },
                );
            }

            // With optimization - should simplify to x
            {
                let mut exec = SimpleExecutor::new();
                let mut graph = tnsr::graph::TensorGraph::<f32>::new();
                let expr = TensorExpr::from(a.clone());
                let expr = -(-expr);
                graph.add_expr(&expr, true);
                let graph = Arc::new(graph);
                let graph_clone = Arc::clone(&graph);

                group.bench_with_input(
                    BenchmarkId::new("optimized", format!("{n}x{n}")),
                    &n,
                    move |b, &_| {
                        b.iter(|| {
                            let _ = std::hint::black_box(
                                exec.execute(&*graph_clone, Default::default()),
                            );
                        });
                    },
                );
            }
        }
        group.finish();
    }

    // Benchmark: Overhead of running optimization itself
    {
        let mut group = c.benchmark_group("optimization_overhead");
        for &n in &sizes {
            let shape = vec![n, n];
            let a = Constant::new(vec![1.5f32; n * n], shape.clone());

            // Measure time to optimize exp(log(x))
            {
                let expr = TensorExpr::from(a.clone()).log().exp();
                group.bench_with_input(
                    BenchmarkId::new("exp_log", format!("{n}x{n}")),
                    &n,
                    |b, &_| {
                        b.iter(|| {
                            let expr_clone = expr.clone();
                            let _ = std::hint::black_box(expr_clone.optimize());
                        });
                    },
                );
            }

            // Measure time to optimize transpose(transpose(x))
            {
                let expr = TensorExpr::from(a.clone()).transpose().transpose();
                group.bench_with_input(
                    BenchmarkId::new("transpose_transpose", format!("{n}x{n}")),
                    &n,
                    |b, &_| {
                        b.iter(|| {
                            let expr_clone = expr.clone();
                            let _ = std::hint::black_box(expr_clone.optimize());
                        });
                    },
                );
            }
        }
        group.finish();
    }
}

pub fn fusion_benches(c: &mut Criterion) {
    let sizes: [usize; 2] = [32, 256];

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
                let mut exec = SimpleExecutor::new();
                let mut graph = tnsr::graph::TensorGraph::<f32>::new();
                expr.lower_to_graph(&mut graph);
                graph.apply_fusion();
                let graph = Arc::new(graph);
                let graph_clone = Arc::clone(&graph);

                group.bench_with_input(
                    BenchmarkId::new("fused", format!("{n}x{n}")),
                    &n,
                    move |b, &_| {
                        b.iter(|| {
                            let _ = std::hint::black_box(
                                exec.execute(&*graph_clone, Default::default()),
                            );
                        });
                    },
                );
            }

            // Build graph without fusion (baseline)
            {
                let mut exec = SimpleExecutor::new();
                let mut graph = tnsr::graph::TensorGraph::<f32>::new();
                expr.lower_to_graph(&mut graph);
                let graph = Arc::new(graph);
                let graph_clone = Arc::clone(&graph);

                group.bench_with_input(
                    BenchmarkId::new("unfused", format!("{n}x{n}")),
                    &n,
                    move |b, &_| {
                        b.iter(|| {
                            let _ = std::hint::black_box(
                                exec.execute(&*graph_clone, Default::default()),
                            );
                        });
                    },
                );
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
                let mut exec = SimpleExecutor::new();
                let mut graph = tnsr::graph::TensorGraph::<f32>::new();
                expr.lower_to_graph(&mut graph);
                graph.apply_fusion();
                let graph = Arc::new(graph);
                let graph_clone = Arc::clone(&graph);

                group.bench_with_input(
                    BenchmarkId::new("fused", format!("{n}x{n}")),
                    &n,
                    move |b, &_| {
                        b.iter(|| {
                            let _ = std::hint::black_box(
                                exec.execute(&*graph_clone, Default::default()),
                            );
                        });
                    },
                );
            }

            // Build graph without fusion (baseline)
            {
                let mut exec = SimpleExecutor::new();
                let mut graph = tnsr::graph::TensorGraph::<f32>::new();
                expr.lower_to_graph(&mut graph);
                let graph = Arc::new(graph);
                let graph_clone = Arc::clone(&graph);

                group.bench_with_input(
                    BenchmarkId::new("unfused", format!("{n}x{n}")),
                    &n,
                    move |b, &_| {
                        b.iter(|| {
                            let _ = std::hint::black_box(
                                exec.execute(&*graph_clone, Default::default()),
                            );
                        });
                    },
                );
            }
        }
        group.finish();
    }
}

criterion_group!(benches_group, benches, optimization_benches, fusion_benches);
criterion_main!(benches_group);
