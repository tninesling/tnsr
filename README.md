# Tnsr

A tensor library, written in Rust.

## Modules

- `nn` - Common neural network layers and operations.
- `runtime` - Runtime for executing graphs of tensor operations.
- `tensor` - Frontend tensor type and operations.
- `tile` - Tile-based intermediate representation and builder.
- `ptx` - PTX IR for CUDA kernels.
- `cuda` - CUDA backend with static PTX kernels.

## CUDA Support

The CUDA backend can be enabled via the `cuda` feature. This uses static PTX
kernels located in `src/cuda/kernels.ptx`.

## Benchmarking

Benchmarks are defined in `benches/` using `criterion`. To run the benchmarks:

- `cargo bench --bench cpu` - Benchmark SIMD kernels against naive
  implementation on CPU.
- `cargo bench --bench gpu --features cuda` - Benchmark CUDA kernels
  against naive implementation on GPU.

## Tracing

Trace spans are collected using the `tracing` crate. To get a JSON dump of
Chrome-compatible trace events, use `runtime::init_chrome_tracing` at the
start of your program. The resulting trace file can be loaded in Perfetto.
