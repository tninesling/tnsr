# ThunderKittens-inspired matmul block traversal

ThunderKittens highlights cache reuse across neighboring matrix tiles and
persistent grids as distinct scheduling opportunities. Its grouped-M traversal
is worth testing independently of arithmetic or fusion changes. Its published
measurements use H100 hardware and are not predictions for the RTX 4080.
[Primary paper, §3.3 and Appendix B.1](https://arxiv.org/html/2410.20399v1).

This experiment adds `PtxExecutor::set_matmul_grid_group_m(group)`, accepting
powers of two from 1 through 32. The **default remains 1**, retaining ordinary
row-major traversal. Existing physical launch dimensions and batch coordinates
stay unchanged. The lowering reinterprets each block's linear task index to
visit neighboring M tiles before advancing N. It changes no K accumulation
order, precision policy, virtual operand binding or epilogue arithmetic.

The effective group is reduced until it divides the number of M tiles. Odd tile
counts therefore retain row-major traversal; ordinary element tails still use
the existing bounds checks. An initial general partial-group index expression
was correct but caused a severe odd-tail slowdown, so it was removed. This
conservative fallback keeps the experiment small and avoids extra integer
selection and division in the final short group.

## RTX 4080 findings

The resident driver uses preallocated device storage, repeated kernels captured
in CUDA Graphs, a private stream, and CUDA-event timings. Uploads and readback
are excluded. Candidate order rotates and reverses between samples. Every
candidate must match the complete group-1 output bitwise; sampled baseline
elements also check a CPU f64 dot-product reference.

Exploration results, group 1 versus group 4:

| Storage / arithmetic | M×N×K | Group 1, µs | Group 4, µs | Speedup |
|---|---|---:|---:|---:|
| BF16 / native tensor cores | 2048×2048×512 | 81.68 | 88.83 | 0.92× |
| BF16 / native tensor cores | 8192×8192×4096 | 8267.38 | 8939.54 | 0.92× |
| BF16 / native tensor cores | 8192×8192×128 | 495.78 | 497.82 | 1.00× |
| BF16 / native tensor cores | 4096×16384×2048 | 4378.09 | 4533.39 | 0.97× |
| F32 / strict ordered accumulation | 2048×2048×1024 | 1404.33 | 1344.20 | 1.04× |

Groups 8 and 16 did not reverse the negative BF16 result. A separate 24-sample
repeat confirmed a small strict-F32 gain, 1416.36→1357.69 µs (1.04×), while the
BF16 control again regressed, 74.24→80.71 µs (0.92×). The final odd-grid fallback
was flat, 173.84→173.82 µs at 2051×2057×513, with effective group 1 for every
requested group. Actual JIT register counts changed from 54 to 52 for the strict
case and 84 to 85 for BF16; predicted resident CTAs were unchanged and neither
case spilled to local memory. These diagnostics do not establish why memory
traversal helped one arithmetic schedule and hurt the other.

The win is narrow and the BF16 regression repeatable. These results support
keeping grouping as an **experimental forced option**,
with no new automatic heuristic. Compact measurements and intervals are saved
in [matmul-grid-order.json](benchmarks/matmul-grid-order.json).

```sh
cargo run --release --features cuda --example matmul_grid_probe -- \
  --m 2048 --n 2048 --k 1024 --dtype f32 --precision strict \
  --groups 1,4,8,16 --samples 20 --repeats 8
```

The probe also reports actual JIT registers, local/shared storage, occupancy
limits and the effective group, allowing future experiments to distinguish
memory-reuse gains from index-cost or resource regressions.

## Compatibility and remaining ideas

The backend already has swizzled shared layouts, aligned `cp.async`, two-stage
buffering, register tiles and tensor-core lowering. New directions from TK are
persistent output-tile loops, deeper occupancy-aware pipelines, and tensor-core
online attention. They require separate implementations and measurements.
The present IR caps operand pipelines at two stages and drains all outstanding
copies; a deeper ring requires extending those semantics, not just changing a
stage-count constant.

Direct transplantation of current TK kernels is unsuitable: its README example
uses Hopper TMA/WGMMA and dynamic register redistribution, while this GPU is
Ada SM89. The current project also no longer actively supports Ampere.
[Official ThunderKittens repository](https://github.com/HazyResearch/ThunderKittens).
Ordinary `cp.async` is available here; TMA bulk operations require SM90+ and
WGMMA requires SM90a. [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/).

## Verification

- Two CPU tests passed: group validation and an exhaustive grid-permutation
  check across row counts 1–39, column counts 1–13 and every supported group.
- Two focused CUDA tests passed with CUDA required: complete output bits match
  group 1 for StrictF32/TF32x3 batched/transposed/tail cases with bias/ReLU and
  native BF16. The compiled-execution test also checks that changing a group
  invalidates the previous compiled signature.
- Strict all-target/all-feature Clippy passed after the final lowering change;
  focused example/test Clippy also passed after probe diagnostics were added.
