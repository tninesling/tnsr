use std::sync::Arc;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use runtime::Executor;
use runtime::SimpleExecutor;
#[cfg(feature = "simd")]
use runtime::simd::SimdExecutor;
use tensor::Constant;
use tensor::TensorExpr;

pub fn benches(c: &mut Criterion) {
    let simple = Arc::new(SimpleExecutor {});
    #[cfg(feature = "simd")]
    let simd = Arc::new(SimdExecutor);

    let sizes: [usize; 2] = [32, 256];

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

            let exec = simple.clone();
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

            #[cfg(feature = "simd")]
            {
                let exec = simd.clone();
                let graph_simd = Arc::clone(&graph);
                group.bench_with_input(
                    BenchmarkId::new("simd", format!("{n}x{n}")),
                    &n,
                    move |b, &_| {
                        b.iter(|| {
                            let _ = std::hint::black_box(
                                exec.execute(&*graph_simd, Default::default()),
                            );
                        });
                    },
                );
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

            let exec = simple.clone();
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

            #[cfg(feature = "simd")]
            {
                let exec = simd.clone();
                let graph_simd = Arc::clone(&graph);
                group.bench_with_input(
                    BenchmarkId::new("simd", format!("{n}x{n}")),
                    &n,
                    move |b, &_| {
                        b.iter(|| {
                            let _ = std::hint::black_box(
                                exec.execute(&*graph_simd, Default::default()),
                            );
                        });
                    },
                );
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

            let exec = simple.clone();
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

            #[cfg(feature = "simd")]
            {
                let exec = simd.clone();
                let graph_simd = Arc::clone(&graph);
                group.bench_with_input(
                    BenchmarkId::new("simd", format!("{n}x{n}")),
                    &n,
                    move |b, &_| {
                        b.iter(|| {
                            let _ = std::hint::black_box(
                                exec.execute(&*graph_simd, Default::default()),
                            );
                        });
                    },
                );
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

            let exec = simple.clone();
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

            #[cfg(feature = "simd")]
            {
                let exec = simd.clone();
                let graph_simd = Arc::clone(&graph);
                group.bench_with_input(
                    BenchmarkId::new("simd", format!("{n}x{n}")),
                    &n,
                    move |b, &_| {
                        b.iter(|| {
                            let _ = std::hint::black_box(
                                exec.execute(&*graph_simd, Default::default()),
                            );
                        });
                    },
                );
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

            let exec = simple.clone();
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

            #[cfg(feature = "simd")]
            {
                let exec = simd.clone();
                let graph_simd = Arc::clone(&graph);
                group.bench_with_input(
                    BenchmarkId::new("simd", format!("{n}x{n}")),
                    &n,
                    move |bch, &_| {
                        bch.iter(|| {
                            let _ = std::hint::black_box(
                                exec.execute(&*graph_simd, Default::default()),
                            );
                        });
                    },
                );
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

            let exec = simple.clone();
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

            #[cfg(feature = "simd")]
            {
                let exec = simd.clone();
                let graph_simd = Arc::clone(&graph);
                group.bench_with_input(
                    BenchmarkId::new("simd", format!("{n}x{n}")),
                    &n,
                    move |bch, &_| {
                        bch.iter(|| {
                            let _ = std::hint::black_box(
                                exec.execute(&*graph_simd, Default::default()),
                            );
                        });
                    },
                );
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

            let exec = simple.clone();
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

            #[cfg(feature = "simd")]
            {
                let exec = simd.clone();
                let graph_simd = Arc::clone(&graph);
                group.bench_with_input(
                    BenchmarkId::new("simd", format!("{n}x{n}")),
                    &n,
                    move |bch, &_| {
                        bch.iter(|| {
                            let _ = std::hint::black_box(
                                exec.execute(&*graph_simd, Default::default()),
                            );
                        });
                    },
                );
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

            let exec = simple.clone();
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

            #[cfg(feature = "simd")]
            {
                let exec = simd.clone();
                let graph_simd = Arc::clone(&graph);
                group.bench_with_input(
                    BenchmarkId::new("simd", format!("{n}x{n}")),
                    &n,
                    move |bch, &_| {
                        bch.iter(|| {
                            let _ = std::hint::black_box(
                                exec.execute(&*graph_simd, Default::default()),
                            );
                        });
                    },
                );
            }
        }
        group.finish();
    }
}

criterion_group!(benches_group, benches);
criterion_main!(benches_group);
