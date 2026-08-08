# Transformer and CUDA Handoff

## Branch and commits

Branch: `feature/transformer-cuda-handoff`

- `2f00d43` - Implement transformer blocks and attention
- `3240343` - Support transformers on CUDA

These commits are also present on `main`. This branch adds this handoff document
so another agent can validate and continue the work in isolation.

## Implemented functionality

The CPU path now supports:

- Public reshape, arbitrary permutation, and axis swapping.
- Rank-generic batched matmul with NumPy-style batch broadcasting.
- Autograd for reshape, permutation, batched matmul, and broadcast operands.
- Numerically stable softmax and layer normalization.
- Causal scaled dot-product attention.
- Trainable multi-head attention.
- A pre-norm transformer block with a ReLU feed-forward network.

Tensor shapes are batch-first. Transformer inputs use `[B, T, C]`; split heads
use `[B, H, T, Dh]`.

The static CUDA executor now supports the tensor operations required by that
transformer implementation:

- Reshape uses a pooled device-to-device copy.
- Permute uses a generic device gather kernel in `src/cuda/kernels.ptx`.
- Higher-rank transpose routes through the generic permutation kernel.
- Batched matmul reuses the existing `matmul` kernel with cudarc device views,
  launching once per output batch and mapping broadcasted batch indices on the
  host.
- Zero-length CUDA buffers use cudarc null slices.

## Important validation gap

The development machine has an AMD GPU (`1002:1114`, kernel driver `amdgpu`),
not an NVIDIA GPU. It has no `libcuda`, `nvidia-smi`, CUDA toolkit, or `ptxas`.

Consequently:

- Rust CUDA code was compiled but not executed on a GPU.
- The new hand-written `permute` PTX was reviewed textually but was not assembled
  by `ptxas` or JIT-loaded by an NVIDIA driver.
- CUDA integration tests skip only when CUDA device initialization is
  unavailable. PTX module-load failures are treated as test failures.

The first task on an NVIDIA machine should be:

```bash
TNSR_REQUIRE_CUDA=1 cargo test --features cuda --test cuda_transformer -- --nocapture
```

`TNSR_REQUIRE_CUDA=1` is essential: it prevents the tests from passing by
skipping unavailable CUDA hardware.

## Verification completed

The following passed locally:

```bash
cargo nextest run --no-fail-fast
cargo check --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all --check
cargo test --features cuda --test cuda_transformer
```

The default suite ran 122 tests. The four CUDA transformer tests compiled and
skipped because `libcuda` was unavailable.

## CUDA test coverage

`tests/cuda_transformer.rs` compares CUDA with explicit references or the CPU
executor for:

- Reshape followed by rank-three permutation.
- Broadcasted batched matmul.
- Causal scaled dot-product attention.
- Tiny transformer forward execution and backward gradients for every
  parameter.

## Files to inspect

- `src/tensor.rs`: expression shapes and view/matmul APIs.
- `src/graph/mod.rs`: graph nodes and autograd.
- `src/lib.rs`: CPU permutation and batched matmul reference behavior.
- `src/nn.rs`: softmax, layer norm, SDPA, multi-head attention, and transformer.
- `src/cuda/mod.rs`: CUDA dispatch, batching, permutation metadata, and buffers.
- `src/cuda/kernels.ptx`: hand-written `permute` entry.
- `tests/transformer.rs`: CPU transformer correctness and gradient coverage.
- `tests/transformer_foundations.rs`: view, broadcasting, and batched matmul tests.
- `tests/cuda_transformer.rs`: NVIDIA execution tests.

## Recommended next steps

1. Run the required CUDA test command above on an NVIDIA GPU.
2. If module loading fails, validate `src/cuda/kernels.ptx` with `ptxas` and fix
   the `permute` entry before changing Rust launch code.
3. Run the complete CUDA suite:

```bash
TNSR_REQUIRE_CUDA=1 cargo nextest run --features cuda
```

4. Compare transformer CPU and CUDA results on larger shapes and add cases for
   two-sided multi-axis batch broadcasting.
5. Benchmark the one-launch-per-batch matmul approach. Replace it with a single
   batched kernel or cuBLAS only if launch overhead is material.
6. Consider checking in CUDA source and a reproducible PTX generation command;
   the repository currently treats static PTX as source of truth.

## Known limitations

- The static matmul kernel is scalar and untiled; GPU correctness is the current
  goal, not transformer performance.
- Batched matmul launches once per output batch.
- Permutation uploads small stride arrays for each execution.
- CUDA only supports `f32`.
- The separate JIT PTX executor still rejects reshape and permutation and cannot
  execute transformer graphs.
- FlashAttention, dropout, GELU, embeddings, indexed cross-entropy, and KV cache
  are not implemented.
- AMD execution is a separate backend effort tracked by issue #39; CUDA cannot
  use the GPU in the current development machine.
