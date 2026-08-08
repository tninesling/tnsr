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

## CUDA validation outcome

The development machine has an AMD GPU (`1002:1114`, kernel driver `amdgpu`),
not an NVIDIA GPU. It has no `libcuda`, `nvidia-smi`, CUDA toolkit, or `ptxas`.

Consequently, the original implementation had only been compile-checked before
this handoff. It has now been validated on an NVIDIA GeForce RTX 4080 with
driver 580.65.06 (CUDA 13.0):

- The static PTX module loads and all CUDA kernels required by the transformer
  execute successfully.
- Reshape, rank-three permutation, two-sided batched-matmul broadcasting,
  causal attention, and transformer forward/backward execution match CPU
  references.
- GPU validation exposed a middle-axis reduction metadata bug. CUDA reduction
  kernels consume compact rank-minus-one output strides, but the executor had
  uploaded full-rank strides. The executor now uploads the expected compact
  strides, fixing transformer bias gradients and all non-final-axis reductions.
- CUDA integration tests still skip only when CUDA device initialization is
  unavailable. Set `TNSR_REQUIRE_CUDA=1` to make unavailable hardware an error.

The focused hardware validation command is:

```bash
TNSR_REQUIRE_CUDA=1 cargo test --features cuda --test cuda_transformer -- --nocapture
```

`TNSR_REQUIRE_CUDA=1` prevents the tests from passing by skipping unavailable
CUDA hardware.

## Verification completed

The following pass after NVIDIA validation:

```bash
TNSR_REQUIRE_CUDA=1 cargo nextest run --features cuda
cargo check --features cuda
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all --check
TNSR_REQUIRE_CUDA=1 cargo test --features cuda --test cuda_transformer -- --nocapture
```

The CUDA-enabled suite ran 140 tests with 140 passing and none skipped. The six
focused CUDA transformer tests all executed on the GPU.

## CUDA test coverage

`tests/cuda_transformer.rs` compares CUDA with explicit references or the CPU
executor for:

- Reshape followed by rank-three permutation.
- Broadcasted batched matmul.
- Two-sided multi-axis batched matmul broadcasting.
- Middle-axis reduction used by transformer bias gradients.
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

1. Compare transformer CPU and CUDA results on larger shapes.
2. Benchmark the one-launch-per-batch matmul approach. Replace it with a single
   batched kernel or cuBLAS only if launch overhead is material.
3. Consider checking in CUDA source and a reproducible PTX generation command;
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
