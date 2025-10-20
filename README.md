# Tnsr

A tensor library, written in Rust.

## Crates

- [`nn`]- Common neural network layers and operations.
- [`runtime`] - Runtime for executing graphs of tensor operations.
- [`tensor`] - Frontend tensor type and operations.

## CUDA Support

The CUDA backend can be enabled via the `cuda` feature. This is currently
backed by static PTX kernels compiled from a `rust-cuda` project.

## Benchmarking

Benchmarks are defined in [`runtime/benches`] using `criterion`. To run the
benchmarks, use:

- `cargo bench --bench cpu` - Benchmark SIMD kernels against naive
  implementation on CPU.
- `cargo bench --bench gpu --features cuda` - Benchmark CUDA kernels
  against naive implementation on GPU.

## Tracing

Trace spans are collected using the `tracing` crate. To get a JSON dump of
Chrome-compatible trace events, us `runtime::init_chrome_tracing` at the
start of your program. The resulting trace file can be loaded in Perfetto.

[`nn`]: crates/nn
[`runtime`]: crates/runtime
[`tensor`]: crates/tensor
[`runtime/benches`]: crates/tensor/benches
