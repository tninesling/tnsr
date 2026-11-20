# Larger-model optimization targets

The latest complete-update policy uses sparse BF16 master shadows plus vector
BF16-first Adam v2 only for W1024/S128/B2, alongside the accepted packed backward
policies for B8/B16/B32. B32 now additionally selects Rectangle128x64 Split-K
for its 17 eligible programs, with a complete Square64 fallback, plus four
register-statistics forward scopes with whole Global-pair fallback. Its latest
complete-update observation is 50.8735 ms against the retained strongest
29.4994 ms control; this is not a profile-derived prediction. The closest native observation is 6.8722 ms against the
retained strongest qualified aggregate compiled control of 6.7813 ms. Parity
remains 2/8. See the [scorecard](transformer-training-parity.md#current-scorecard-b32-register-statistics-20261119)
and [shadow measurements](transformer-training-parity.md#sparse-bf16-shadows-v2-closest-case-opt-in-with-memory-regression-november-19).

The shadow policy has a measured memory regression: +198,148,096 bytes
(188.96875 MiB) used pool memory after setup, +160 MiB reserved after setup,
+192 MiB reserved after qualification/steady state, and sampled process peaks
of 2,200 versus 2,008 MiB in both native pairs. Two owned BF16 versions add
219,152,384 bytes while physical private BF16 storage falls by only 21,004,288
bytes. The 104.5 MiB earlier logical estimate was not a pool/process prediction.
No capacity or memory-saving claim follows.

The retained Square64 B32 diagnostic below predates the Rectangle128x64 and
register-statistics complete-update experiments. No new profile measures or
attributes their gains. Historical traces remain unavailable at their
deleted `target/` paths; their retained summaries describe earlier policies.
The vector-v1 and NN candidates remain unadopted. Whole-process sampled lower
bounds, allocator peaks and logical payload calculations remain separate.

<a id="current-accepted-b32-diagnostic-november-19"></a>

## Retained Square64 B32 diagnostic (November 19)

The five-update profile used the then-accepted packed B32 Square64 policy and
retained release binary, with five warmups before CUDA-profiler-API capture.
The two fixed first-update comparisons pass all six gates. Native gradient and
weight snapshots match both accepted native trials exactly, and all 66 F64
loss/norm/scale values match the corresponding first eleven optimizer steps.
Replay remains active. This diagnostic changes no policy, throughput observation
or parity target; the suite remains at **2/8**.

| Diagnostic group | Mean summed GPU activity per update |
| --- | ---: |
| GEMMs, including split-K partials | 23.507646 ms |
| Online attention forward/backward | 11.088338 ms |
| Other reductions | 3.878813 ms |
| LayerNorm and preserved outer adds | 3.572545 ms |
| Materializations/conversions | 3.436989 ms |
| SwiGLU | 2.416239 ms |
| Indexed loss | 1.549115 ms |
| Other pointwise work | 1.198661 ms |
| Adam / health | 0.692301 ms |
| Split-K sums | 0.321128 ms |
| Accumulation / norm | 0.260328 ms |
| Embedding | 0.181128 ms |
| Loss-scalar readback materializer | 0.000960 ms |

The mean summed kernel activity is 52.104192 ms/update. GEMMs contribute about
45.1% and online attention 21.3% of that sum. The eight MatMul nodes with recorded
geometry M16384/N512/K1408 total 2.895386 ms; their FFN/input-gradient role is a
model-context hypothesis rather than an independently joined gradient-graph
identity. The 21 physical-order weight casts total only 0.160838 ms. These
observations prioritize GEMM and attention investigation without predicting
end-to-end gains or extrapolating closest-case optimizer/shadow results.

Actual attention resource records are stable across all five updates:

| Direction | Nodes/update | Mean summed ms/update | Registers/thread | Static shared B | Local/thread B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forward | 4 | 3.293470 | 56 | 49,152 | 0 |
| Packed Q + delta | 4 | 3.414680 | 72 | 49,152 | 0 |
| Packed K + V | 4 | 4.380188 | 97 | 49,152 | 0 |

All use 256 threads `(32,8,1)`, grid `(8,256,1)`, and zero dynamic shared memory.
Nsight reports zero local memory per thread. Its `localMemoryTotal` field is the
same 67,239,936 bytes for every kernel; that raw constant is not interpreted as
spill traffic or a per-node memory delta. No achieved occupancy, stall, bandwidth
or tensor-utilization counters were collected.

The trace contains five exact graph-launch correlations with 298 model kernels
each, alternating two graph variants. Ordered labels are joined through original
graph-node lineage offsets and identical name/grid/block/resource signatures;
generic kernel names alone are not semantic identities. Each update also has
59 direct launches, giving 1,785 kernels overall (357/update). The software
counter reports 356 because the additional loss-scalar materializer is outside
that counter. Ten graph memsets and 35 memcpy records are retained separately.
Parameter associations for standalone Adam/accumulation use the source-bound
registration order and checked extents, not generic function names alone.

This is a single legacy software-instrumented capture. It contains 994 small
apparent adjacent overlaps on one stream (maximum 1,472 ns). Summed durations
are neither a critical path nor unprofiled synchronized latency. The sampled
6,316 MiB process peak is a lower bound including profiling effects, separate
from allocator counters and not a capacity result. Exact native-child ancestry,
frozen desktop identity and raw sampling evidence pass independent review;
sub-sample activity remains unobservable.

The [durable evidence package](benchmarks/transformer-training-evidence/packed-b32-profile-20261119/README.md)
retains the raw trace, SQLite, helpers, source/profiler provenance, both validators,
full semantic/resource records and independent audits. A separate verified raw
archive contains five unique F32 objects—native gradients/weights, qualified
PyTorch gradients/weights and fixture weights—with nine original path aliases.
Current raw evidence is therefore retained outside `target/`. Two diagnostic
comparisons are added: diagnostic totals become 15 (12 native semantic, three
reference replay), while 68 benchmark campaigns, 1,234 attempts, 52,544 timed
updates and 820 runner comparisons remain unchanged.

## Historical retained diagnostics

The fresh W1024/S128/**B2** diagnostic uses the accepted V30 binary and case
options, five warmups and five profiled updates. Both unchanged first-update
validators pass. Gradient and postweight snapshots match both accepted native
trials byte for byte, and all eleven optimizer-step loss/norm/scale records
match both trials exactly. These profiled updates are excluded from throughput
totals and do not change the [parity scorecard](transformer-training-parity.md#current-scorecard-b32-register-statistics-20261119).

| V30 W1024/S128/B2 diagnostic group | Summed GPU activity per update |
| --- | ---: |
| Adam / health | 2.466160 ms |
| GEMM | 2.339587 ms |
| Accumulation / norm | 0.601821 ms |
| Materialization | 0.585353 ms |
| Reduction | 0.373225 ms |
| Split partials | 0.283111 ms |

Total activity is 7.184739 ms/update, including 4.079177 ms in 346 model
kernels. There are 405 actual kernels per update including optimizer work and
loss readback. Twenty borrowed-BF16 Adam launches account for 2.247374 ms;
two other standalone Adam launches account for 0.205378 ms. These standalone
kernels use 1,024-thread blocks, while five small batched Adam launches already
use 256 threads. Their resource records motivate retesting the existing
experimental Adam256 option under the current policy; they do not establish
an occupancy or latency improvement. The subsequent corrected complete-update
retest passes all gates but is slightly slower in both native pairs
(aggregate 6.972354→6.975061 ms); Adam256 is not adopted. Its fresh qualified
PyTorch CUDA-graph control is 6.7961947643 ms, tightening the target without
changing the accepted native observation or policy. The whole-process peak sampled during
capture is 2,020 MiB, a lower bound separate from allocator counters.

The trace, exact commands, semantic mapping, resource records and validation
are retained in
[`parity-profile-v30-w1024-s128`](../target/transformer-training/parity-profile-v30-w1024-s128/README.md).
Activity sums describe this diagnostic capture, not wall-time shares or
predicted full-update savings.

The newer W1024/S512/B1 diagnostic uses the retained V22 binary and then-accepted case
options, five warmups and five profiled updates. Both fixed first-update
comparisons pass all six gates; snapshots and first-step scalars exactly match
both native trials. It contains 295 model kernels and 387 total kernels per
update. Summed activity is 10.154 ms, including 6.719 ms in the model, 2.716 ms
in 27 Adam/health kernels and 0.682 ms in 60 fused accumulation/norm kernels.
Norm-partial fusion is already active here. These are diagnostic activity sums,
not wall-time shares or predicted savings. The sampled 2,210 MiB process peak
is a lower bound. Reproduction, semantic mapping and the owner source review
are in
[`parity-profile-v22-w1024-s512`](../target/transformer-training/parity-profile-v22-w1024-s512/README.md).
V27 removes 20 bias reshape copies, V28 changes eligible Adam gradient reads,
and V30 batches accumulation/norm launches in this case. This capture precedes
all three changes.

The historical V25 B32 semantic breakdown uses its retained version,
using its retained binary, exact accepted options, five warmups and five
profiled complete updates. The trace contains 322 captured model kernels per
update and 414 kernels including optimizer and readback work. Both fixed
first-update comparisons pass all six gates; snapshots and first-step scalar
bits exactly match both accepted native trials. The later profiled trajectory
is not compared with the campaign's 256-warmup trajectory.

| Diagnostic group | Summed GPU activity per update |
| --- | ---: |
| Forward linears (NT) | 9.491 ms |
| Weight-gradient split partials (TN) | 8.287 ms |
| Input-gradient GEMMs (NN) | 7.702 ms |
| Online attention forward / backward | 10.345 ms |
| F32 to BF16 materializations | 2.063 ms |
| BF16 boundary conversions | 2.782 ms |

All CUDA activity sums to 55.921 ms/update, including 54.926 ms in the captured
model. The matrix labels describe mathematical orientation relative to native
parameter storage; physical copy orientation requires the retained layout
proofs. These are diagnostic activity sums, not isolated speedups, wall-time
predictions or hardware utilization measurements. The accepted online attention
path still retains its old quadratic buffers. The 6,300 MiB process peak sampled
every 50 ms is a lower bound,
separate from exact allocator scopes.

Reproduction commands, trace, semantic mapping, layout census and numerical
qualification are retained in
[`target/transformer-training/parity-profile-v25-b32`](../target/transformer-training/parity-profile-v25-b32/README.md).
The earlier
[V20 diagnostic](transformer-training-parity.md#v20-b32-diagnostic-ordered-loss-backward-is-smaller)
remains historical evidence. The V25 activity figures precede accepted V26 NT linears, V27 copy aliases, V29
collective NT loads, V30 accumulation/norm batches and the later packed policy.
The retained B32 diagnostic above describes the preceding Square64 policy. Use the
[current parity scorecard](transformer-training-parity.md#current-scorecard-b32-register-statistics-20261119)
for unprofiled complete-update observations. The original estimates and V19
scope below are historical and do not describe the current kernel breakdown.

## Historical scope after V19

The accepted V19 change adds the isolated S128 LayerNorm extension with
ordinary four-warp blocks off. Both completed V19 timing campaigns are recorded
in the [parity report](transformer-training-parity.md); neither adds a profile.
The V18 diagnostic remains the latest captured semantic breakdown. Later unbuilt
index-fence and CE proposals have no measured evidence here.

This document preserves the original profiling pass and its then-current
estimates. The later [parity report](transformer-training-parity.md) contains the
retained V16 B32 semantic trace and the completed V17 eight-case sweep. V17
activates nine LayerNorm forward/backward pairs at W512/S512/B8–32, removing 157
model launches, and eight score GEMMs at the two eligible low-BH cases. No V17
performance profile was captured; earlier family percentages are not a measured
breakdown of its new bottlenecks.

V17 B32 takes 68.188 ms versus 29.622 ms for the fresh stronger qualified control,
while increasing owned F32/steady pool memory by 34,078,720 bytes and lifecycle
USED_HIGH by 369,688,576 bytes. Source/rounding contracts, gate results, memory
scopes and raw evidence are retained in that report. Full-suite parity remains
unachieved, and the old estimates below must not be multiplied into future gains.

## Original profiling pass

The first profiling pass suggests a **1.2–1.4x complete-update speedup** is a
useful initial goal. **1.5–2x** is a broader target requiring several changes,
not a measured improvement or a guaranteed outcome. Compiled PyTorch's existing
1.77–1.81x throughput advantage on the two larger cases establishes a concrete
comparison point for this workload, not a hardware performance ceiling.

These estimates follow the [training baseline](transformer-training-performance.md)
and the [raw profile summaries](benchmarks/transformer-training-profile.json).
No production optimization is enabled by this profiling pass.

## Measured breakdown

Nsight Systems captured five measured complete updates after one qualification
update and five warmups, using node-level CUDA graph tracing. Capture starts and
stops through the benchmark's optional `--profile` switch. Setup, compilation,
qualification snapshots and warmups are outside the capture range. These are
diagnostic traces, not replacements for unprofiled benchmark timings.

| Kernel family | 55M, width 1024 / S512 / B1 | 211M, width 2048 / S128 / B1 |
| --- | ---: | ---: |
| Matmul | 38.1% | 23.3% |
| Adam | 15.7% | 31.6% |
| Gradient accumulation | 8.6% | 16.7% |
| View materialization | 6.6% | 14.7% |
| Reductions | 12.3% | 9.1% |
| Other | 18.7% | 4.6% |

Percentages divide summed kernel activity duration, grouped by generated name
prefix. They exclude copies, memset and CPU gaps; overlapping activity must not
be added to infer wall time. Tracing overhead and five-step sample size also
limit precision. This is not a hardware utilization or bandwidth analysis.

For 211M, Adam and accumulation total about 17.44 ms of summed kernel activity
per update, versus 8.41 ms for matmul and 5.32 ms for view materialization.
Device copies total another 2.86 ms and memset 1.23 ms. Halving only Adam and
accumulation would save roughly 8.7 ms if those savings translate directly to
the critical path: about 1.27x against the unprofiled 40.89 ms baseline. This
illustrates the size of the opportunity, not a forecast from overlapping sums.

Both profiled runs retain active replay. Their full first-step gradient and
post-Adam weight snapshots are bitwise identical to the corresponding original
native trial-0 snapshots. Profiling timings are not included in the performance
scorecard.

## Memory and candidate changes

At 211M parameters, one full-size f32 array is 844,591,104 bytes (805.465 MiB).
The resident optimizer retains eight such arrays: current/staged weights,
gradients, current/staged first and second moments, and per-element health.
Together these account for approximately 6.29 GiB of the 8.13 GiB steady pool
usage. Mixed-executor f32/BF16 buffers account for nearly all the remainder.

Investigate these changes in order:

1. Reduce Adam and gradient-accumulation traffic. Preserve clipping order,
   microbatch addition semantics and transactional publication on failures.
2. Replace full-size health storage with one zero/NaN partial per 256-thread
   block. The current finite check adds `(next-next)` for each new weight and
   moment, so zero/NaN reduction can retain rejection before publication.
   For 211M, 825,344 padded partials would occupy 3.148 MiB, an estimated
   **802.316 MiB** saving against the current array. This is an allocation
   calculation, not a measured process-memory or latency improvement. Masked
   tail lanes must contribute zero and reach every block barrier; batched
   parameters need disjoint partial ranges and stable replay pointers.
3. Reduce repeated master copies and view materialization. Stable captured
   pointers and immutable published weight versions constrain direct aliasing.
4. Tune BF16 GEMMs and attention for the 55M/context-heavy cases, where matmul
   occupies a larger fraction of the trace.

A 25–40% reduction in native VRAM is a research target, not an allocation saving
already demonstrated. Compiled PyTorch uses less than half the sampled process
VRAM on the 211M case, but its different state management does not establish
which native arrays can safely be removed.

Every candidate still needs unprofiled complete-update measurements and the
unchanged numerical gates. Component speedups cannot be multiplied into a
full-model prediction.

## Reproduction

Build the example before taking the GPU lock, then capture either existing
fixture using a fresh output path:

```sh
cargo build --release --features cuda --example transformer_training_bench
flock /tmp/tnsr-bench.cuda0.lock nsys profile \
  --trace=cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=target/transformer-training/profile-new \
  target/release/examples/transformer_training_bench \
  --archive target/transformer-training/scale-screen/w2048_s128/archive \
  --output target/transformer-training/profile-new-artifacts \
  --warmups 5 --repeats 5 --profile
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
  --format csv --output target/transformer-training/profile-new-summary \
  target/transformer-training/profile-new.nsys-rep
```

Original traces and SQLite exports remain under
`target/transformer-training/profiles-v1/`; checked-in JSON retains the complete
CSV summaries, source/binary hashes and benchmark result metadata.
