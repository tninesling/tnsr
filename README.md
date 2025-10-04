# Tnsr

A tensor library, written in Rust.

## Crates

- [`kernels-cuda`]- CUDA backend for kernels.
- [`runtime`] - Runtime for managing devices and dispatching to backend kernels.
- [`tensor`] - Frontend tensor type and operations.

## SIMD Support

The SIMD backend requires AVX2 support. AVX2 operations are encouraged by
compiling with `-C target-cpu=native`, which is set in `.cargo/config.toml`.

## CUDA Support

The CUDA backend requires the CUDA toolkit as well as LLVM 7, which is a
limitation of NVVM. The kernels are currently working with CUDA 12.8 and
LLVM 7.1 inside the provided Docker image. These also currently require the
latest commits on the `main` branch of [`rust-cuda`]. Build the dev image with
`./build-cuda.sh` and run an interactive shell with `./run-cuda.sh`. The CUDA
kernels will build as transitive dependencies when the `cuda` feature is enabled.

### Kernel Compilation

The `kernels-cuda` crate contains a child crate, [`cuda-device-kernels`]. The
`cuda-device-kernels` crate uses `rust-cuda` to define CUDA kernels in Rust.
When the parent `kernels-cuda` crate is built, the `build.rs` script compiles
the `cuda-device-kernels` crate to PTX using `cuda_builder`. The resulting PTX
file is bundled statically and loaded into the CUDA-enabled device at runtime
for JIT compilation.

## Benchmarking

Benchmarks are defined in [`runtime/benches`] using `criterion`. To run the
benchmarks, use:

- `cargo bench --bench cpu` - Benchmark SIMD kernels against naive
  implementation on CPU.
- `cargo bench --bench gpu --features cuda` - Benchmark CUDA kernels
  against naive implementation on GPU.

[`cuda-device-kernels`]: crates/kernels-cuda/device
[`kernels-cuda`]: crates/kernels-cuda
[`runtime`]: crates/runtime
[`tensor`]: crates/tensor
[`runtime/benches`]: crates/tensor/benches
[`rust-cuda`]: https://github.com/Rust-GPU/rust-cuda/tree/main
