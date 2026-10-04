# Tnsr

A tensor library, written in Rust.

# Overview

Tensors are core to modern ML. To train a model, you describe a computation which operates on tensors. In `tnsr`, you build a tensor from constants, inputs, and trainable parameters. These can be added, subtracted, etc to build up a tensor expression. Tensor expressions can be evaluated directly, but they are often compiled into kernels that take advantage of specialized hardware.

The `tnsr` runtime can execute a tensor expression in a few different ways:
1. It can execute the expression directly using naive implementations of each operation on CPU.
2. It can execute the expression with statically-provided CUDA kernels on GPU.
3. It can JIT-compile the expression to PTX kernels, which are then executed on GPU.


## Computation as a graph

The first step to compiling the expression is lowering it into a computation graph. Each node in the graph represents an operation (e.g. addition, multiplication) or a tensor (e.g. input, constant). The edges represent the flow of data between nodes. A graph can be optimized by combining or reordering nodes to improve performance.

## Autograd

A good autograd engine is the heart of any good tensor framework.

One way to do automatic differentiation is to implement explicit forward and backward calculations for each operation. In the forward pass, we calculate the output of each node given its inputs. In the backward pass, we calculate the gradients of each input given the gradient of the output. This applies the chain rule of calculus to propagate gradients back through the graph.

In `tnsr`, the gradient computations are appended to the computation graph as additional nodes. This reduces the API surface of the executor implementations, which means the gradient definitions do not need to repeated for each hardware platform.

## Graph rewrites

The first step to optimizing a computation graph is to apply graph rewrites. These are transformations that replace subgraphs with more efficient equivalents. For example, you could rewrite the pattern `A + 0` to just `A`, since adding zero does not change the value.

## Operator fusion

After rewriting the graph, we can apply operator fusion. This combines multiple operations into a single kernel. For example, the sequence of operations `A + B + C` can be fused into a single kernel that computes the sum of three tensors in one pass. This helps reduce memory bandwidth because the fused computation keeps the intermediate results in registers instead of writing them back to memory.

## Memory reuse

CPU, CUDA, and PTX executors perform liveness analysis on every execution and
return dead intermediate buffers to exact-size pools. Outputs and gradients
remain available until the next execution; gradients can be returned earlier
with `release_gradients`. Consequently, `get_value` is only guaranteed for
pinned output and gradient nodes after execution.

## Decoder-only GPT

`tnsr::nn::Gpt` composes learned token and position embeddings, causal
transformer blocks, final layer normalization, and a tied language-model head.
Indexed embedding and cross-entropy operations avoid one-hot vocabulary tensors
and support autograd on both CPU and static CUDA executors.

Run the self-contained character-level training and generation example on CPU:

```bash
cargo run --release --example gpt
```

Enable CUDA to use an available NVIDIA GPU automatically:

```bash
cargo run --release --features cuda --example gpt
```

## Tile representation

For GPU hardware targets, we take advantage of tiling to improve memory access patterns. Tiling breaks down large tensor operations into smaller blocks (tiles) that fit into the faster shared memory of the GPU. This allows threads within a block to cooperate and share data, reducing the number of global memory accesses. It also allows coalesced memory accesses, which improves bandwidth utilization. The smallest primitive is a register tile, which corresponds to the size of a warp. The 16x16 register tile comes directly from the [ThunderKittens] project, which provides high-performance C++ templates for CUDA kernels.

The PTX backend selects a `MatMulSchedule` from target capabilities and uses
it for both lowering and launch configuration. Nonzero-K batched matmuls can
fuse bias, activation, and residual epilogues into one launch, including batch
broadcasts. TF32 is available on SM80+; select full F32 operands explicitly with:

```rust
use tnsr::ptx::PtxExecutor;
use tnsr::tile::MatMulPrecision;

let mut executor = PtxExecutor::try_new()?;
executor.set_matmul_precision(MatMulPrecision::StrictF32);
```

See [fusion measurements](docs/fusion-baseline.md#stage-6c-target-aware-schedules-and-batched-epilogues)
for schedule limits, numerical parity, and transformer/GPT benchmarks.

# Comparison to candle

HuggingFace's [candle] is a library focused on inference. It uses C++ implementations of CUDA kernels for GPU execution. In `candle`, each tensor operation is fallible and thus returns a `Result`. This means that instead of writing `a + b + c`, you need to propagate errors with `((a + b)? + c)?`.

The `Tensor` operations in `tnsr` are infallible because they simply build an expression tree.

# Comparison to burn

Burn aims to encode more information about tensors in the type system. A `burn::Tensor` is parameterized by its backend and its rank. In `tnsr`, the tensor operations are hardware-agnostic, since they just describe a computation. The backend execution is handled separately by the runtime, which discovers available hardware features and can schedule across heterogeneous devices.

# Development

## CUDA Support

The CUDA backend can be enabled via the `cuda` feature. This uses static PTX
kernels in `src/cuda/`. Regenerate the indexed-operation module with CUDA 12.9
or newer using:

```bash
nvcc --ptx --gpu-architecture=compute_61 --use_fast_math \
  --output-file src/cuda/indexed_kernels.ptx src/cuda/indexed_kernels.cu
```

## Benchmarking

Benchmarks are defined in `benches/` using `criterion`. To run the benchmarks:

- `cargo bench --bench cpu` - Benchmark SIMD kernels against naive
  implementation on CPU.
- `cargo bench --bench gpu --features cuda` - Benchmark CUDA kernels
  against naive implementation on GPU.

## Tracing

Trace spans are collected using the `tracing` crate. To get a JSON dump of
Chrome-compatible trace events, use `tnsr::init_chrome_tracing` at the
start of your program. The resulting trace file can be loaded in Perfetto.

[candle]: https://github.com/huggingface/candle
[ThunderKittens]: https://github.com/HazyResearch/ThunderKittens

### Half precision

`Runtime::<half::f16>` and `Runtime::<half::bf16>` support CPU, CUDA, and PTX
execution. CPU matmul and reductions accumulate in f32. The static CUDA backend
converts graphs and inputs to f32, runs its existing kernels, and narrows results
at the host boundary.

The PTX backend keeps graph buffers in native 16-bit storage, including fused
pointwise, reduction, and batched matmul regions, with f32 scalar computation and
accumulation. Eligible matmuls use native FP16 tensor cores on SM70+ or BF16 tensor
cores on SM80+, including fused epilogues. Small shapes and
`MatMulPrecision::StrictF32` use increasing-K scalar f32 accumulation.
`AllowTf32` permits these native half tensor-core schedules as well as TF32 for
f32 graphs; it never converts an f32 graph to FP16/BF16.

Native BF16 PTX requires SM80+. On older GPUs, automatic runtime selection falls
back to the static CUDA backend's f32 conversion path. Half embedding gradients
sum repeated indices in f32 per output element and narrow once, avoiding atomics
on half buffers; this correctness fallback can be slower for large vocabularies.

### Automatically fused online attention

The PTX compiler has an experimental `PtxReductionMode::Online` policy that
recognizes primitive softmax/weighted-matmul graphs and generates a streaming
normalized contraction. Eligible f32 inference avoids materializing score and
probability matrices, including attention inside the existing transformer API.
See [online attention](docs/online-attention.md) for activation, legality checks,
resident benchmarks, and current schedule limits.
