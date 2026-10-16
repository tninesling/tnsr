# Shared dense contractions in the forty-five adapter

The adapter now flattens leading activation dimensions before each shared dense
projection, then restores the output shape. For `[B,S,I] @ [I,O]`, the GEMM sees
`[B*S,I] @ [I,O]`. Autograd therefore computes the weight gradient with one
contraction over `B*S`, instead of producing `[B,I,O]` per-example matrices and
reducing them over the batch. Forward tiles can also span sequence boundaries.
The public `nn::linear` and general batched matmul contracts are unchanged.

This changes the grouping of floating-point sums. It introduces no new precision
policy, and must pass the existing inference and training tolerances. Both
matched workers use TF32x3, deterministic-tree reductions and CUDA Graph replay;
strict-f32 execution is checked separately.

## CUDA view materialization prerequisite

The first prototype failed inference and training parity and was rejected in
issue #120. Layer-by-layer comparison isolated the failure to the attention
output projection. QKV, scores, softmax, attended values and the flattened
context each matched when evaluated as separate output graphs.

A small reproducer was enough: permute `[3,2,16,4]` to `[3,16,2,4]`, reshape to
`[3,16,8]`, reshape again to `[48,8]`, and consume it in a dense projection.
The accumulated index expression exceeded the planner's complexity limit. The
planner then materialized the final reshape by copying storage, dropping the
preceding permutation. The rejected implementation had an absolute output error
of 1.2734375 on this deterministic case.

Physical view steps with noncontiguous inputs now retain their complete composed
source map. Their input
binding and liveness refer to the actual source allocation. A generated Tile IR
kernel applies the map into contiguous storage for reshape, permutation,
transpose and broadcast fallbacks. The compilation signature advances to 18.
The regressions cover ordinary and broadcast sources, both fusion policies, and
changed feeds without recompilation. A separate contiguous-storage proof keeps
reshape chains on the existing copy path and primitive view kernels on contiguous
inputs. This avoids replacing ordinary copies with mapped kernels. No
materialization limit is removed.

## Validation and measurements

Validation covers 193 CPU tests and 391 CUDA-required all-feature tests, including the
shared/separate tiny fixtures, mapped-view regression, half-precision layout
paths, residency and replay. CPU/CUDA builds, formatting and all-target,
all-feature Clippy pass. The sole documentation test is ignored by design.

Both actual critics pass strict-f32 inference, determinism and archive round
trips. Maximum logit error is 1.24e-5 for each. With TF32x3, tree reductions and
replay, two PPO updates pass the unchanged CPU-f32 oracle tolerances:

| Actual critic | Parameters | Max gradient error | Max weight error |
| --- | ---: | ---: | ---: |
| Shared | 140 | 1.18e-6 | 1.23e-5 |
| Separate | 229 | 6.85e-7 | 1.44e-5 |
| Shared, strict-f32/no replay | 140 | 3.69e-7 | 2.99e-6 |

These checks compare every named clipped gradient and updated weight, weighted
3/3/2 microbatches, nonzero anchor KL, schedule steps and saved archives. Both
replay shape buckets stay active with zero fallbacks. The earlier rejected pilot
is excluded from accepted performance evidence. Short-update agreement does not
establish learning-quality parity with the original BF16 training policy.

## Throughput and remaining work

The baseline is merged `d2f3567`, including the previous view-consumer improvement.
Both workers use identical TF32x3/tree/replay settings, the same initial archive,
and the same fixed 64-transition batch. First-use setup is excluded from timed
steady-state updates and retained in the raw evidence. External workloads stay
running; compare matched workers within each protocol rather than combining
absolute timings from separate runs.

| Protocol / microbatch | Baseline | Candidate | Throughput change |
| --- | ---: | ---: | ---: |
| Native only, alternating AB/BA, 100 pairs / 64 | 72.75 ms | 63.97 ms | +13.7% |
| Native only, alternating AB/BA, 20 pairs / 8 | 72.85 ms | 72.19 ms | Within noise |
| Full interleaved comparison, 10 samples / 64 | 61.73 ms | 49.40 ms | +25.0% |
| Full interleaved comparison, 10 samples / 8 | 121.78 ms | 120.20 ms | Within noise |

In the long run, the candidate is faster in 94 of 100 paired observations; the
median paired throughput ratio improves 14.7%. A separate short native-only
microbatch-64 run improves just 5.2% (87.13 -> 82.83 ms). The size of the gain is
load-sensitive. Use the longer 100-pair result as the primary estimate;
the small-microbatch result is effectively unchanged.

At microbatch 64, physical kernels per full update fall 2407 -> 2205 and
cumulative logical intermediate outputs fall 8.13 -> 7.24 GB. The differentiation
graph retains all 3851 nodes. Worker GPU memory falls 2214 -> 2116 MiB. At
microbatch 8, kernels fall 17352 -> 16632, logical outputs fall 8.31 -> 7.42 GB,
and worker memory remains 708 MiB. Logical outputs are neither measured DRAM
traffic nor peak allocations; GPU memory is sampled whole-worker process usage.
Every timed native update uses replay with zero fallbacks.

In the full microbatch-64 comparison, original eager strict-f32 PyTorch measures
34.86 ms and compiled-policy BF16 measures 30.23 ms. The candidate's 49.40 ms is
about 1.6x the compiled BF16 result in that run. This remains a policy-specific,
shared-GPU comparison, not an established parity result across workloads.

First-use setup regresses: the first microbatch-64 update takes 2.92 s for the
baseline and 16.51 s for the candidate in this run. The measured difference is
in executor warmup/capture (2.29 -> 15.86 s), while compile-cache time is similar
(0.60 -> 0.64 s). The cause needs separate profiling; this report makes no claim
of a startup improvement. Both excluded warmups are preserved.

A new diagnostic trace contains five captured model executions and 9605 kernels,
202 fewer per execution than the previous trace. Matmul instances fall
308 -> 182 and reduction-region instances 311 -> 243 per execution. Only six
composed views per execution need the new mapped kernel. Matmuls account for
50.2% of the candidate's summed kernel duration, reductions for 14.3% and
pointwise regions for 13.2%. Those shares identify the next GPU targets. The
traces were collected at different external load, so their total durations are
not a matched before/after speedup measurement. Host `metrics_seconds` can include
waiting for preceding GPU work and must not be read as pure CPU metric overhead.

Raw measurements, samples, worker/source hashes, numerical gates, warmups and
trace aggregates are in
[dense-contractions.json](benchmarks/training-parity/dense-contractions.json).
Generated reports and traces remain under ignored
`target/forty_five_dense_contractions_final/`. The rejected numerical prototype
and the slower blanket-materialization experiment are excluded from these final
worker measurements.
