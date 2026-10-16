# Split-K weight gradients and reduced PTX startup

The shared dense adapter exposes long weight-gradient contractions, including
`[683,4096] @ [4096,256]`. Its original 32×32 output tile creates 176 blocks, each
running the entire K loop. The opt-in `split_long_matmul_gradients` graph pass
partitions K into four slices, computes `[4,M,K/4] @ [4,K/4,N]`, then sums the
partial outputs. The example shape exposes 704 blocks instead of 176.

The pass applies only to rank-two gradient matmuls with M,N at least 128,
K at least 2048, K at least four times the larger output extent, and K divisible
by four. It preserves original output and parameter-gradient node identities,
marks the new backward nodes for liveness/release, and is idempotent. Ordinary
matmul, view and reduction lowering handles allocation and replay; no new static
CUDA source, atomic accumulation or dtype conversion is introduced.

The forty-five adapter enables this only for CUDA training with both TF32x3 and
deterministic-tree reductions selected. Splitting changes floating-point sum
association. Each partial retains the existing three-product TF32 arithmetic,
and the final reduction has a fixed order. Default strict-f32 execution retains
its original graph. Inference and ordinary matmul APIs are unchanged.

## Numerical gates

Both actual shared and separate critics pass inference, deterministic execution
and archive round trips. Two original CPU-f32 PPO updates compare every named
clipped gradient and updated weight with the same tolerances as before. Checks
include nonzero bidding-anchor KL, schedule steps and saved archives. The usual
weighted 3/3/2 microbatches test two replay buckets; the new repeated fixture
exercises 64 transitions in one microbatch and actually invokes split-K.

| Actual critic / fixture | Named parameters | Max gradient error | Max weight error |
| --- | ---: | ---: | ---: |
| Shared / weighted 3/3/2 | 140 | 1.18e-6 | 1.23e-5 |
| Shared / 64 | 140 | 1.23e-6 | 1.10e-5 |
| Separate / weighted 3/3/2 | 229 | 6.85e-7 | 1.44e-5 |
| Separate / 64 | 229 | 1.36e-6 | 1.33e-5 |
| Shared / strict-f32, no replay | 140 | 3.67e-7 | 2.99e-6 |

The large shared graph transforms 34 contractions and the separate graph 64.
All requested replay buckets remain active with zero fallbacks. Tests cover
retained roots/forward nodes, idempotence, odd output dimensions, transposed and
reshaped operands, changed feeds without recompilation, and both CostAware and
Greedy fusion. The full suite passes 194 CPU tests and 395 CUDA-required
all-feature tests, along with CPU/CUDA checks, formatting and all-target,
all-feature Clippy. The updated two-fusion-policy CUDA regression also passes
separately. Short-update agreement does not establish long-run BF16
learning-quality parity.

## Measured training throughput

Both native workers use the same initial archive, fixed 64-transition batch,
f32 storage/optimizer, TF32x3, tree reductions and CUDA Graph replay. Every timed
request includes JSON, staging, model execution, clipping, Adam and finite
checks. Two excluded warmups are retained. Order alternates AB/BA; existing GPU
workloads stay running and GPU experiments are serialized. No builds overlap
the timed native comparisons.

| Comparison / microbatch | Pairs | Baseline median | Candidate median | Median paired throughput gain |
| --- | ---: | ---: | ---: | ---: |
| Isolated split-K / 64 | 500 | 76.47 ms | 74.26 ms | +3.4% |
| Combined vs merged #122 / 64 | 200 | 67.26 ms | 67.31 ms | +2.9% |
| Combined vs merged #122 / 8 | 100 | 109.98 ms | 109.07 ms | Within noise |

The isolated pass wins 346/500 pairs. A circular block bootstrap over paired
ratios (4,000 resamples, blocks of ten adjacent pairs) gives a 2.7–4.4% interval.
Against merged #122, the combined worker wins 137/200 pairs with a paired interval
of 2.4–4.0%. Its two unpaired time medians are effectively equal in that run:
the paired ratio and ratio of marginal medians differ under the changing load.
The short full-policy comparison below also improves about 3%. Microbatch 8
wins 53/100 pairs and its interval spans -0.7–+1.5%; no split-K contraction is
eligible there. These intervals describe these shared-GPU runs, not a claim
across hardware or workloads.

The isolated GEMM comparison uses a startup-only worker with identical entry
reuse and the original GEMM schedule. The comparison against merged #122 includes
both fixes. Absolute times from separate shared-GPU runs should not be combined.
All full-request observations and excluded warmups are retained in the evidence;
profile statistics summarize phases rather than treating GPU-waiting host timers
as CPU operator costs.

In the separate ten-sample interleaved full microbatch-64 comparison, merged
#122 takes 46.08 ms and the combined candidate 44.73 ms. Original eager
strict-f32 PyTorch takes 37.52 ms, eager BF16 40.07 ms and compiled BF16 28.39 ms.
The candidate remains about 1.6× the compiled BF16 time. At microbatch 8,
merged #122 measures 92.31 ms and the candidate 89.87 ms; the longer native
comparison above does not confirm a small-microbatch gain.

The large shared graph grows from 3,851 to 4,055 logical nodes. Physical kernels
per full update grow 2,205→2,239 because each split contraction adds a final
reduction. Cumulative logical intermediate outputs grow 7.240→7.319 GB. These
are planned outputs, not measured DRAM traffic or peak memory. Whole-worker GPU
memory in the full microbatch-64 comparison is 2,116→2,110 MiB; entry reuse
also removes duplicate module code.

## Startup and rejected schedule experiments

Identical PTX entry reuse addresses the separate cold module-JIT regression;
see the [controlled startup report](training-kernel-reuse-startup.md).
The final combined worker emits 387 unique entries for 2,016 kernel-producing
steps, with a 1,384,079-byte microbatch-64 PTX module.

The final combined worker's three fresh private-cache trials take 6.53, 6.28
and 6.26 seconds for the first update (median 6.28 s). A matching new cold
merged-#122 worker takes 15.82 s: about 60% less cold setup. Every run uses active
replay with zero fallbacks. Warm process restarts take 3.04–3.15 s for the
combined graph versus 2.73 s for that baseline trial. The additional split-K
planning adds some warm setup work; the entry-reuse-only controlled three-trial
comparison remains 2.93→2.90 s. Startup remains separate from timed steady-state
updates and its raw samples are retained.

Wider output tiles and double-buffered staging generally regressed focused
GEMMs. Reduced thread counts, larger K tiles and expanded attention tiles failed
the full-update test and were reverted. In particular, K64 staging improved one
focused FFN gradient from 370→217 µs but left the 100-pair training comparison
within noise (47 wins). That result was rejected rather than reported as a
training speedup. The accepted split-K pass adds parallel work without retaining
any of those scheduler experiments.

Remaining work includes more efficient TF32x3 kernels, adaptive split counts,
reductions and pointwise traffic, plus the separately tracked physical BF16
training path. The current measured gain remains modest relative to the overall
PyTorch gap.

## Reproduction and evidence

Use `validate_ppo.py --cuda-graphs --tree-reductions --tf32x3` for the usual
small fixture, and add `--fixture-repeats 8 --microbatch 64` for the large gate.
For direct paired measurements, use the committed `benchmark_native_pair.py`
command in the [integration README](../integrations/forty_five/README.md).
`matmul_profile` also accepts `TNSR_PROFILE_TF32X3`,
`TNSR_PROFILE_TRANSPOSE_LHS` and `TNSR_PROFILE_TRANSPOSE_RHS`, and prints the
selected schedule for focused investigations.

[split-k-startup.json](benchmarks/training-parity/split-k-startup.json) contains
executable/source hashes, every paired request time, warmups, replay summaries,
profile aggregates, numerical reports, full original-policy comparisons,
controlled startup measurements and rejected ablations. Complete generated
responses/logs remain under ignored `target/forty_five_gemm_startup/`.
