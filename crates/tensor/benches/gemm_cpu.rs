use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use std::sync::Arc;
use tensor::Tensor;

fn bench_naive_gemm(c: &mut Criterion) {
    runtime::register_gemm(Arc::new(kernels_simd::NaiveGemm));
    let t1 = Tensor::new(512, 512, 1.0);
    let t2 = Tensor::new(512, 512, 0.5);

    c.bench_function("naive_gemm_512x512", |b| {
        b.iter(|| {
            let _ = std::hint::black_box(t1.matmul(&t2));
        });
    });
}

fn bench_avx_gemm(c: &mut Criterion) {
    if is_x86_feature_detected!("avx2") {
        runtime::register_gemm(Arc::new(kernels_simd::Avx2Gemm));
        let t1 = Tensor::new(512, 512, 1.0);
        let t2 = Tensor::new(512, 512, 0.5);

        c.bench_function("avx2_gemm_512x512", |b| {
            b.iter(|| {
                let _ = std::hint::black_box(t1.matmul(&t2));
            });
        });
    }
}

criterion_group!(benches, bench_naive_gemm, bench_avx_gemm);
criterion_main!(benches);
