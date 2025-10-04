use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use runtime::Executor;
use runtime::SimpleExecutor;
use tensor::Constant;
use tensor::Tensor;

fn bench_op<F>(c: &mut Criterion, name: &str, make: F)
where
    F: Fn() -> Box<dyn Tensor<f32>>,
{
    let mut graph = tensor::graph::TensorGraph::<f32>::new();
    let node = make();
    node.lower_to_graph(&mut graph);
    let exec = SimpleExecutor {};

    c.bench_function(name, |b| {
        b.iter(|| {
            let _ = std::hint::black_box(exec.execute(&graph, Default::default()));
        })
    });
}

pub fn benches(c: &mut Criterion) {
    let a = Constant::new(vec![1.0f32; 1024], vec![32, 32]);
    let b = Constant::new(vec![0.5f32; 1024], vec![32, 32]);

    bench_op(c, "neg_32x32", || Box::new(-a.clone()));
    bench_op(c, "exp_32x32", || Box::new(a.clone().exp()));
    bench_op(c, "log_32x32", || Box::new(a.clone().log()));
    bench_op(c, "relu_32x32", || Box::new(a.clone().relu()));
    bench_op(c, "add_32x32", || Box::new(a.clone() + b.clone()));
    bench_op(c, "sub_32x32", || Box::new(a.clone() - b.clone()));
    bench_op(c, "mul_32x32", || Box::new(a.clone() * b.clone()));
    bench_op(c, "div_32x32", || Box::new(a.clone() / b.clone()));
}

criterion_group!(benches_group, benches);
criterion_main!(benches_group);
