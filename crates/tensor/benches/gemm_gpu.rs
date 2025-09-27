#[cfg(not(feature = "cuda"))]
compile_error!("The 'cuda' feature must be enabled to run GPU benchmarks.");

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use cust::module::Module;
use cust::stream::Stream;
use cust::stream::StreamFlags;
use std::sync::Arc;
use tensor::Tensor;

fn bench_naive_gemm(c: &mut Criterion) {
    let _ctx = cust::quick_init().expect("Failed to initialize CUDA");
    let stream =
        Stream::new(StreamFlags::NON_BLOCKING, None).expect("Failed to create CUDA stream");
    let ptx_src = kernels_cuda::PTX;
    let module = Module::from_ptx(ptx_src, &[]).expect("Failed to load PTX module");
    let gemm = kernels_cuda::NaiveGemm {
        module: &module,
        stream: &stream,
    };

    let t1 = Tensor::new(512, 512, 1.0);
    let t2 = Tensor::new(512, 512, 0.5);

    c.bench_function("naive_gemm_512x512", |b| {
        b.iter(|| {
            let _ = std::hint::black_box(t1.matmul_with_gemm(&t2, &gemm));
        });
    });
}

/*
TiledGemm kernel fails to launch right now
fn bench_tiled_gemm(c: &mut Criterion) {
    let _ctx = cust::quick_init().expect("Failed to initialize CUDA");
    let stream =
        Stream::new(StreamFlags::NON_BLOCKING, None).expect("Failed to create CUDA stream");
    let ptx_src = kernels_cuda::PTX;
    let module = Module::from_ptx(ptx_src, &[]).expect("Failed to load PTX module");
    let gemm = kernels_cuda::TiledGemm {
        module: &module,
        stream: &stream,
    };

    let t1 = Tensor::new(512, 512, 1.0);
    let t2 = Tensor::new(512, 512, 0.5);

    c.bench_function("tiled_gemm_512x512", |b| {
        b.iter(|| {
            let _ = std::hint::black_box(t1.matmul_with_gemm(&t2, &gemm));
        });
    });
}
*/

criterion_group!(benches, bench_naive_gemm);
criterion_main!(benches);
