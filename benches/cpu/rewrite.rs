use criterion::{BenchmarkId, Criterion};
use std::hint::black_box;
use tnsr::graph::TensorGraph;
use tnsr::graph::rewrite::{RewriteConfig, RewriteNumerics, optimize_expr_with_numerics};
use tnsr::tensor::TensorExpr;
use tnsr::{Executor, SimpleExecutor};

fn tensor(shape: &[usize], seed: usize) -> TensorExpr<f32> {
    let count = shape.iter().product();
    let data = (0..count)
        .map(|i| ((i + seed) % 31) as f32 / 31.0 - 0.5)
        .collect();
    TensorExpr::constant(data, shape.to_vec())
}

pub fn cases() -> Vec<(&'static str, TensorExpr<f32>, RewriteNumerics)> {
    use RewriteNumerics::{PreserveOperations, RealAlgebra};
    let a = tensor(&[64, 32], 1);
    let b = tensor(&[32, 64], 2);
    let c = tensor(&[32, 64], 3);
    vec![
        (
            "reshape_chain",
            tensor(&[256, 256], 0)
                .reshape(vec![128, 512])
                .reshape(vec![256, 256]),
            PreserveOperations,
        ),
        (
            "permute_chain",
            tensor(&[16, 32, 64], 0)
                .permute(vec![1, 2, 0])
                .permute(vec![2, 0, 1]),
            PreserveOperations,
        ),
        (
            "transpose_add",
            tensor(&[256, 256], 0).transpose() + tensor(&[256, 256], 1).transpose(),
            PreserveOperations,
        ),
        (
            "factor_matmul",
            a.clone().matmul(b) + a.matmul(c),
            RealAlgebra,
        ),
        (
            "matrix_chain",
            tensor(&[128, 8], 0)
                .matmul(tensor(&[8, 128], 1))
                .matmul(tensor(&[128, 8], 2)),
            RealAlgebra,
        ),
        (
            "factor_large",
            {
                let a = tensor(&[256, 128], 0);
                a.clone().matmul(tensor(&[128, 256], 1)) + a.matmul(tensor(&[128, 256], 2))
            },
            RealAlgebra,
        ),
        (
            "chain_large",
            tensor(&[256, 32], 0)
                .matmul(tensor(&[32, 256], 1))
                .matmul(tensor(&[256, 32], 2)),
            RealAlgebra,
        ),
        // Skinny contraction control: factoring moves the add to a larger input.
        (
            "factor_rejected",
            {
                let a = tensor(&[2, 256], 1);
                a.clone().matmul(tensor(&[256, 2], 2)) + a.matmul(tensor(&[256, 2], 3))
            },
            RealAlgebra,
        ),
    ]
}

pub fn benches<E: Executor<f32>>(
    c: &mut Criterion,
    make_executor: impl Fn() -> E,
    execute: impl FnMut(&mut E, &TensorGraph<f32>, bool) -> Vec<f32>,
) {
    execution_benches(c, make_executor, execute);
    compile_benches(c);
}

fn execution_benches<E: Executor<f32>>(
    c: &mut Criterion,
    make_executor: impl Fn() -> E,
    mut execute: impl FnMut(&mut E, &TensorGraph<f32>, bool) -> Vec<f32>,
) {
    let mut group = c.benchmark_group("graph_rewrites");
    for (name, expr, policy) in cases() {
        let optimized = optimize_expr_with_numerics(&expr, RewriteConfig::default(), policy);
        for (variant, candidate) in [("raw", expr.clone()), ("optimized", optimized)] {
            let mut graph = TensorGraph::new();
            graph.add_expr(&candidate, false);
            let mut executor = make_executor();
            let expected = SimpleExecutor::new()
                .execute(&TensorGraph::from(expr.clone()), Default::default())
                .unwrap();
            let actual = execute(&mut executor, &graph, true);
            for (a, b) in actual.iter().zip(&expected) {
                assert!(
                    (a - b).abs() <= 2e-4 * (1.0 + b.abs()),
                    "{name}: {a} vs {b}"
                );
            }
            assert_eq!(actual.len(), expected.len());
            group.bench_function(BenchmarkId::new(name, variant), |b| {
                b.iter(|| black_box(execute(&mut executor, black_box(&graph), false)));
            });
        }
    }
    group.finish();
}

fn compile_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_rewrite_compile");
    for (name, expr, policy) in cases() {
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(optimize_expr_with_numerics(
                    black_box(&expr),
                    RewriteConfig::default(),
                    policy,
                ))
            })
        });
    }
    group.finish();
}
