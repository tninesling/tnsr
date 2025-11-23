use std::sync::Arc;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use runtime::{Executor, SimpleExecutor};
use tensor::{Constant, Parameter};
use tensor::TensorExpr;

pub fn benches(c: &mut Criterion) {
    let sizes: [usize; 4] = [32, 256, 1024, 2048];

    // NOTE: These benchmarks use SimpleExecutor directly to measure CPU-only performance.
    // For application code, prefer using Runtime::new() or Runtime::with_backend(Backend::Cpu).
    // Example:
    //   let mut runtime = Runtime::with_backend(Backend::Cpu).unwrap();
    //   runtime.execute(&graph, inputs).unwrap();

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
            let mut graph = tensor::graph::TensorGraph::<f32>::new();
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
            let mut graph = tensor::graph::TensorGraph::<f32>::new();
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
            let mut graph = tensor::graph::TensorGraph::<f32>::new();
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
            let mut graph = tensor::graph::TensorGraph::<f32>::new();
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
            let mut graph = tensor::graph::TensorGraph::<f32>::new();
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
            let mut graph = tensor::graph::TensorGraph::<f32>::new();
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
            let mut graph = tensor::graph::TensorGraph::<f32>::new();
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
            let mut graph = tensor::graph::TensorGraph::<f32>::new();
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

    // Fusion: unary chain (relu -> log -> exp)
    #[cfg(feature = "fusion")]
    {
        let mut group = c.benchmark_group("fusion_unary_chain");
        for &n in &sizes {
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 2) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());

            // Unfused graph: relu -> log -> exp
            let mut exec_unfused = SimpleExecutor::new();
            let mut graph_unfused = tensor::graph::TensorGraph::<f32>::new();
            let expr = TensorExpr::from(a.clone()).relu().log().exp();
            expr.lower_to_graph(&mut graph_unfused);
            let graph_unfused = Arc::new(graph_unfused);

            // Fused graph: FusedUnary([relu, log, exp])
            let mut exec_fused = SimpleExecutor::new();
            let mut graph_fused = tensor::graph::TensorGraph::<f32>::new();
            let expr = TensorExpr::from(a.clone()).relu().log().exp();
            expr.lower_to_graph(&mut graph_fused);
            graph_fused.apply_fusion();
            let graph_fused = Arc::new(graph_fused);

            let graph_unfused_clone = Arc::clone(&graph_unfused);
            group.bench_with_input(
                BenchmarkId::new("unfused", format!("{n}x{n}")),
                &n,
                move |b, &_| {
                    b.iter(|| {
                        let _ = std::hint::black_box(
                            exec_unfused.execute(&*graph_unfused_clone, Default::default()),
                        );
                    });
                },
            );

            let graph_fused_clone = Arc::clone(&graph_fused);
            group.bench_with_input(
                BenchmarkId::new("fused", format!("{n}x{n}")),
                &n,
                move |b, &_| {
                    b.iter(|| {
                        let _ = std::hint::black_box(
                            exec_fused.execute(&*graph_fused_clone, Default::default()),
                        );
                    });
                },
            );
        }
        group.finish();
    }

    // Fusion: longer unary chain (neg -> relu -> log -> exp)
    #[cfg(feature = "fusion")]
    {
        let mut group = c.benchmark_group("fusion_long_chain");
        for &n in &sizes {
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 2) as u64,
            ));
            let shape = vec![n, n];
            let a = Constant::new(vec![1.0f32; n * n], shape.clone());

            // Unfused graph: neg -> relu -> log -> exp
            let mut exec_unfused = SimpleExecutor::new();
            let mut graph_unfused = tensor::graph::TensorGraph::<f32>::new();
            let expr = (-TensorExpr::from(a.clone())).relu().log().exp();
            expr.lower_to_graph(&mut graph_unfused);
            let graph_unfused = Arc::new(graph_unfused);

            // Fused graph: FusedUnary([neg, relu, log, exp])
            let mut exec_fused = SimpleExecutor::new();
            let mut graph_fused = tensor::graph::TensorGraph::<f32>::new();
            let expr = (-TensorExpr::from(a.clone())).relu().log().exp();
            expr.lower_to_graph(&mut graph_fused);
            graph_fused.apply_fusion();
            let graph_fused = Arc::new(graph_fused);

            let graph_unfused_clone = Arc::clone(&graph_unfused);
            group.bench_with_input(
                BenchmarkId::new("unfused", format!("{n}x{n}")),
                &n,
                move |b, &_| {
                    b.iter(|| {
                        let _ = std::hint::black_box(
                            exec_unfused.execute(&*graph_unfused_clone, Default::default()),
                        );
                    });
                },
            );

            let graph_fused_clone = Arc::clone(&graph_fused);
            group.bench_with_input(
                BenchmarkId::new("fused", format!("{n}x{n}")),
                &n,
                move |b, &_| {
                    b.iter(|| {
                        let _ = std::hint::black_box(
                            exec_fused.execute(&*graph_fused_clone, Default::default()),
                        );
                    });
                },
            );
        }
        group.finish();
    }

    // Fusion: gradient of unary chain (exp -> log)
    #[cfg(feature = "fusion")]
    {
        let mut group = c.benchmark_group("fusion_gradient_exp_log");
        for &n in &sizes {
            // Count both forward and backward passes
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 4) as u64,
            ));
            let shape = vec![n, n];

            // Unfused gradient graph
            let mut exec_unfused = SimpleExecutor::new();
            let x_unfused = Parameter::new(vec![1.0f32; n * n], shape.clone());
            let expr_unfused = TensorExpr::from(x_unfused).exp().log().reduce_sum(1).reduce_sum(0);
            let mut graph_unfused = tensor::graph::TensorGraph::<f32>::new();
            expr_unfused.lower_to_graph(&mut graph_unfused);
            let topo_unfused = graph_unfused.toposort();
            let loss_unfused = *topo_unfused.last().unwrap();
            let grad_graph_unfused = graph_unfused.with_gradients(loss_unfused);
            let grad_graph_unfused = Arc::new(grad_graph_unfused);

            // Fused gradient graph
            let mut exec_fused = SimpleExecutor::new();
            let x_fused = Parameter::new(vec![1.0f32; n * n], shape.clone());
            let expr_fused = TensorExpr::from(x_fused).exp().log().reduce_sum(1).reduce_sum(0);
            let mut graph_fused = tensor::graph::TensorGraph::<f32>::new();
            expr_fused.lower_to_graph(&mut graph_fused);
            graph_fused.apply_fusion();
            let topo_fused = graph_fused.toposort();
            let loss_fused = *topo_fused.last().unwrap();
            let grad_graph_fused = graph_fused.with_gradients(loss_fused);
            let grad_graph_fused = Arc::new(grad_graph_fused);

            let grad_graph_unfused_clone = Arc::clone(&grad_graph_unfused);
            group.bench_with_input(
                BenchmarkId::new("unfused", format!("{n}x{n}")),
                &n,
                move |b, &_| {
                    b.iter(|| {
                        let _ = std::hint::black_box(
                            exec_unfused.execute(&*grad_graph_unfused_clone, Default::default()),
                        );
                    });
                },
            );

            let grad_graph_fused_clone = Arc::clone(&grad_graph_fused);
            group.bench_with_input(
                BenchmarkId::new("fused", format!("{n}x{n}")),
                &n,
                move |b, &_| {
                    b.iter(|| {
                        let _ = std::hint::black_box(
                            exec_fused.execute(&*grad_graph_fused_clone, Default::default()),
                        );
                    });
                },
            );
        }
        group.finish();
    }

    // Fusion: gradient of longer chain (neg -> relu -> exp)
    #[cfg(feature = "fusion")]
    {
        let mut group = c.benchmark_group("fusion_gradient_long_chain");
        for &n in &sizes {
            group.throughput(Throughput::Bytes(
                (n * n * std::mem::size_of::<f32>() * 4) as u64,
            ));
            let shape = vec![n, n];

            // Unfused gradient graph
            let mut exec_unfused = SimpleExecutor::new();
            let x_unfused = Parameter::new(vec![1.0f32; n * n], shape.clone());
            let expr_unfused = (-TensorExpr::from(x_unfused)).relu().exp().reduce_sum(1).reduce_sum(0);
            let mut graph_unfused = tensor::graph::TensorGraph::<f32>::new();
            expr_unfused.lower_to_graph(&mut graph_unfused);
            let topo_unfused = graph_unfused.toposort();
            let loss_unfused = *topo_unfused.last().unwrap();
            let grad_graph_unfused = graph_unfused.with_gradients(loss_unfused);
            let grad_graph_unfused = Arc::new(grad_graph_unfused);

            // Fused gradient graph
            let mut exec_fused = SimpleExecutor::new();
            let x_fused = Parameter::new(vec![1.0f32; n * n], shape.clone());
            let expr_fused = (-TensorExpr::from(x_fused)).relu().exp().reduce_sum(1).reduce_sum(0);
            let mut graph_fused = tensor::graph::TensorGraph::<f32>::new();
            expr_fused.lower_to_graph(&mut graph_fused);
            graph_fused.apply_fusion();
            let topo_fused = graph_fused.toposort();
            let loss_fused = *topo_fused.last().unwrap();
            let grad_graph_fused = graph_fused.with_gradients(loss_fused);
            let grad_graph_fused = Arc::new(grad_graph_fused);

            let grad_graph_unfused_clone = Arc::clone(&grad_graph_unfused);
            group.bench_with_input(
                BenchmarkId::new("unfused", format!("{n}x{n}")),
                &n,
                move |b, &_| {
                    b.iter(|| {
                        let _ = std::hint::black_box(
                            exec_unfused.execute(&*grad_graph_unfused_clone, Default::default()),
                        );
                    });
                },
            );

            let grad_graph_fused_clone = Arc::clone(&grad_graph_fused);
            group.bench_with_input(
                BenchmarkId::new("fused", format!("{n}x{n}")),
                &n,
                move |b, &_| {
                    b.iter(|| {
                        let _ = std::hint::black_box(
                            exec_fused.execute(&*grad_graph_fused_clone, Default::default()),
                        );
                    });
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches_group, benches);
criterion_main!(benches_group);
