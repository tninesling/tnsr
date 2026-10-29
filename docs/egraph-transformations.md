# Shape-aware e-graph transformations

Research and measurements: October 29, 2026. Baseline commit:
`96459566905dbee6f02302a430298e3bc4311c3e`.

## Research and selection

- [TASO (SOSP 2019)](https://github.com/jiazhihao/TASO) searches graph
  substitutions with a cost model, including joint algebra/layout optimization.
  The useful lesson here is to offer alternatives and select using tensor sizes,
  rather than count operators. Shared-contraction factoring is a direct fit for
  our expression IR; concat/split projection packing would need more IR support.
- [TENSAT (MLSys 2021)](https://proceedings.mlsys.org/paper_files/paper/2021/file/cc427d934a7f6c0663e5923f49eba531-Paper.pdf)
  represents alternatives with equality saturation. It uses measured operator
  runtimes and discusses greedy tree extraction versus DAG-aware ILP extraction.
  We retain bounded egg saturation and greedy extraction, adding shape facts and
  a deterministic work heuristic. We do not implement TENSAT's calibrated cost
  model or DAG optimizer.
- [OpenXLA's current algebraic simplifier](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc)
  eliminates identity reshapes/transposes, composes permutations, and merges
  adjacent reshapes. These are appropriate conservative transformations because
  they remove layout work without changing floating-point arithmetic order.
- [PACT 2024 tensor EqSat/MCTS research](https://arxiv.org/abs/2410.05534)
  addresses search-space growth with rule selection and better extraction.
  This argues against indiscriminately adding bidirectional distributivity and
  expansive layout rules. The new contraction search is rank-2 and bounded by
  existing runner limits.
- [EqiForge (September 2026 preprint)](https://arxiv.org/abs/2609.12330)
  combines tensor algebra and tiled execution in one equality-saturation IR,
  using early compaction and subgraph composition. Its synthesis of fused
  implementations motivates a future bridge between our expression e-graph and
  Tile IR. That is a larger project than adding graph identities; this change
  does not claim to reproduce its results.

## Implemented

`src/graph/rewrite/tensor_analysis.rs` attaches shapes to e-classes and provides
layout normalization, guarded contraction rules, cached parsed rules, and a
saturating integer work model for extraction.

Default (`PreserveOperations`):

- Eliminate same-shape reshapes; compose adjacent reshapes.
- Compose adjacent permutations; eliminate identity permutations.

Explicit `RealAlgebra` opt-in:

- `A B + A C -> A (B + C)` and `B A + C A -> (B + C) A`.
  Require rank-2 matrices and equal shapes for the operands being added.
- `(A B) C <-> A (B C)` for rank-2 matrices. The cost model chooses the
  parenthesization rather than operator count (which is identical).

The model charges matmul approximately `2 * batch * M * N * K`, pointwise/layout
work approximately three times the largest input's element count, plus a small
per-operation constant. Reshape is metadata work. It is not a GPU latency model:
it omits occupancy, tensor-core tile efficiency, fusion, sharing, and measured
launch costs. Costs saturate instead of overflowing. The extractor counts shared
subgraphs as trees. Tests explicitly cover a skinny case where moving an add
before a contraction is more expensive and factoring must be rejected.

In this initial round, casts, linear/indexed ops, convolutions, pooling, flatten,
and wide metadata kept entire expressions out of the e-graph. The
[follow-up search](egraph-rewrite-search.md) replaces that whole-tree bypass with
opaque boundaries so supported parents can optimize while unsupported subtrees
retain their identities. Existing local batched row regrouping is retained.
These benchmarks exercise supported subgraphs and do not establish training or
Transformer model speedups. Arithmetic rewrites change rounding and may alter
NaNs, infinities, signed zero, and gradients; the default policy does not enable
them.

## Rejected candidate

We implemented and measured sinking paired transposes through ordered pointwise
add/subtract/multiply/divide. On the `[256,256]` add case it improved CPU time
from about 1.20 ms to 0.72 ms, but worsened precompiled CUDA execution from
about 60 us to 636 us. The scalar work heuristic missed the backend layout/fusion
penalty. Those four rules were removed; `transpose_add` remains a control.
The exploratory CPU and GPU measurements are retained in the JSON results.
This is evidence for hardware-aware extraction before introducing broader layout
movement, rather than an argument that fewer transposes always run faster.

## Measurement protocol

CPU: AMD Ryzen 7 3700X. GPU: NVIDIA RTX 4080 (16 GiB), driver 580.196.07.
Criterion: 20 samples, 0.5 s warmup, 1 s measurement per case. Means and confidence
intervals are saved in `docs/benchmarks/graph-rewrites.json`.

CPU baseline uses the original optimizer with the new benchmark harness; the
GPU baseline uses an isolated archive of the baseline commit and the identical
harness. CUDA runs require device access outside the desktop sandbox. GPU runs
use `StrictF32` for both versions, compile once outside timing, and call
`execute_compiled` in the timed loop. GPU times include execution setup, transfers,
and synchronous host output readback, not just kernel time. No benchmark treats
execution errors as successful iterations. All cases check against CPU results
before timing.

The cases are small diagnostic graphs, not a representative workload average:

| Case | Shape / purpose |
|---|---|
| reshape_chain | `[256,256] -> [128,512] -> [256,256]` |
| permute_chain | `[16,32,64]`, permutation `[1,2,0]` then its inverse |
| transpose_add | two transposed `[256,256]` inputs added |
| factor_matmul | shared `[64,32]` left matrix, two `[32,64]` right matrices |
| matrix_chain | `[128,8] @ [8,128] @ [128,8]` |
| factor_large | shared `[256,128]` left matrix, two `[128,256]` right matrices |
| chain_large | `[256,32] @ [32,256] @ [256,32]` |
| factor_rejected | legacy benchmark ID for a skinny `[2,256]` / `[256,2]` control; factoring may win |

The rejection unit test uses `[1,256]` / `[256,1]`, where the heuristic retains
two contractions. Compilation overhead is measured separately and excludes the
first initialization of the cached rule tables.

Reproduce the current version:

```sh
cargo bench --bench cpu -- graph_rewrite --warm-up-time 0.5 --measurement-time 1 --sample-size 20
cargo bench --bench gpu --features cuda -- graph_rewrites --warm-up-time 0.5 --measurement-time 1 --sample-size 20
cargo nextest run
cargo nextest run --features cuda
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
```

## Final results

Means from the original optimizer versus the final implementation; times are
microseconds. The transpose-sinking candidate is excluded from the final version.

| Case | CPU before → after (us) | CPU speedup | GPU before → after (us) | GPU speedup |
|---|---:|---:|---:|---:|
| reshape_chain | 17.00 → 5.33 | 3.19x | 10,591.65 → 53.65 | 197.43x |
| permute_chain | 720.36 → 2.46 | 292.82x | 31.44 → 30.35 | 1.04x |
| transpose_add | 1,199.42 → 1,195.06 | 1.00x | 60.44 → 59.07 | 1.02x |
| factor_matmul | 212.96 → 96.14 | 2.22x | 16.68 → 15.82 | 1.05x |
| matrix_chain | 74.37 → 36.77 | 2.02x | 14.18 → 13.70 | 1.03x |
| factor_large | 3,157.18 → 1,252.74 | 2.52x | 68.31 → 59.80 | 1.14x |
| chain_large | 646.12 → 340.42 | 1.90x | 25.66 → 21.09 | 1.22x |
| factor_rejected | 4.51 → 2.60 | 1.73x | 23.34 → 21.81 | 1.07x |

The larger contractions deliver the clearest arithmetic wins: approximately 2.52x
CPU / 1.14x GPU for factoring, and 1.90x CPU / 1.22x GPU for reassociation.
The roughly 197x GPU reshape result is a pathological layout-chain microbenchmark,
not a model-level expectation. Permutation composition saves CPU materialization;
the GPU backend already simplifies the corresponding access maps. Small GPU
contractions and unchanged controls vary by a few percent, so their small deltas
should not be interpreted as established improvements.

Steady-state optimization overhead also improved after caching rule parsing:

| Case | Before (us) | After (us) |
|---|---:|---:|
| reshape_chain | 7.64 | 5.66 |
| permute_chain | 8.25 | 6.26 |
| transpose_add | 7.57 | 5.15 |
| factor_matmul | 46.21 | 20.20 |
| matrix_chain | 40.63 | 12.12 |
| factor_large | 52.50 | 20.64 |
| chain_large | 45.65 | 12.13 |

## Validation

- `cargo check` and `cargo check --features cuda`: passed.
- `cargo nextest run`: 256 passed, none skipped.
- `TNSR_REQUIRE_CUDA=1 cargo nextest run --features cuda -j 2`: 518 passed;
  three existing manual performance experiments remained ignored.
- Final focused CUDA rewrite rerun: 12 passed, none skipped.
- `cargo fmt --all --check`, strict all-target/all-feature Clippy, and
  `git diff --check`: passed.

Regression coverage includes layout composition and inverse elimination,
conservative floating-point behavior (NaNs, infinities and signed zero),
broadcast guards, both factoring directions, both reassociation directions,
rejection of expensive factoring, large metadata without cost overflow,
parameter gradients, and CUDA versus CPU output comparisons.

## Follow-on candidates

Prioritize calibrated GPU costs and DAG-aware extraction before broadening
arithmetic search. Next candidates are moving equal layout maps through
pointwise regions, guarded broadcast/reduction simplification, and projection
packing with concat/split support. Reduction rewrites need dtype-specific
rounding and empty-axis contracts. Unified expression/Tile saturation and
FlashAttention-style synthesis need explicit scheduling and memory-cost models.
