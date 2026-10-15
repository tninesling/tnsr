# Batched forward/backward dispatch — issue 90

Matmul regions now support an empty pointwise epilogue. Eligible standalone and
backward contractions use the existing generated TileIR matmul kernel with batch
coordinates in `grid.z`, rather than launching once per leading matrix/head.
Broadcasts and virtual input layouts retain their general index-map lowering;
zero-K, empty outputs and batches exceeding CUDA's grid.z limit keep fallbacks.
The PTX graph cache version is bumped because physical dispatch changes.

The full forty-five training graph also exposed a fusion contraction cycle. Its
fallback previously discarded every region, which would erase the dispatch
improvement. That fallback now rebuilds independent, anchor-only matmul regions.
Single-node regions cannot introduce a cycle by contracting graph edges. Other
fusion remains conservative in this path.

## Paired measurements

Before is the readback-only implementation from #95 (`0ce289e`), not the original
unoptimized compiler. Same v8.4 archive, real seeded 64-transition batch, native
strict f32 arithmetic, one PPO update, anchor KL disabled, two excluded warmups
and three timed samples. The original PyTorch training function runs with the
same workload. Cases are interleaved on the RTX 4080, with existing GPU work left
running. CUDA tests and parity checks run separately from throughput measurements.

| Median seconds per update | Microbatch 8 | Microbatch 64 |
| --- | ---: | ---: |
| Before: readback fix, per-matrix dispatch | 0.898 | 0.363 |
| After: batched dispatch | 0.758 | 0.251 |
| Original PyTorch eager f32 | 0.250 | 0.0292 |
| Original PyTorch compiled-policy bf16 | 0.216 | 0.0328 |

Warm total update time falls **16% / 31%**. Runtime execute-stage medians fall
**0.386 -> 0.301 s / 0.267 -> 0.153 s**. These stage times include dispatch and
execution; they are not isolated GPU kernel durations.

| Kernel launches per update | Microbatch 8 | Microbatch 64 |
| --- | ---: | ---: |
| Before | 52,104 | 29,361 |
| After | 25,592 | 3,199 |

Separate Nsight Systems traces of three complete updates confirm less GPU work:

| Sum of GPU kernel durations per update | Microbatch 8 | Microbatch 64 |
| --- | ---: | ---: |
| Before | 0.227 s | 0.178 s |
| After | 0.152 s | 0.116 s |
| Matmul kernels only, before -> after | 0.104 -> 0.040 s | 0.111 -> 0.047 s |

These totals include profiler overhead and contention and overlap CPU activity;
do not add them to wall or runtime-stage times. Hardware-counter occupancy was
not measured: the driver restricts profiling counters (`RmProfilingAdminOnly=1`).

The first warmup of the new binary took 27.5 / 48.8 seconds, primarily in the
execute stage that includes first-use CUDA module compilation. This is excluded
from warm timings and retained in the raw evidence. Warm latency does not describe
cold startup. Timing variation/contention makes these diagnostic samples rather
than an exclusive-GPU performance gate.

The remaining eager-f32 gap in these samples is about **3.0x / 8.6x**. Issue 91
covers device-resident parameters, accumulated gradients and Adam state. Issue 92
covers the remaining normalization/activation backward fragmentation; issue 93
covers stable-buffer replay, and issue 94 covers mixed precision. Shared-RHS row
flattening and further matmul layout/scheduling improvements remain possible;
this change implements general batch dispatch without changing arithmetic policy.

Raw samples, stages, source/batch hashes, binary hashes and GPU trace summaries
are retained in [batched.json](benchmarks/training-parity/batched.json). Full
generated outputs are under ignored `target/forty_five_batched/`.

## Validation and reproduction

All **179 CPU / 344 all-feature CUDA-required tests** pass, along with CPU/CUDA
builds, formatting and clippy. Added tests cover standalone rank-four permuted
and broadcast operands with one launch, forward/backward shared-parameter
contractions, and the fusion-cycle fallback. Existing tests cover rank-two/three,
two-sided broadcasting, transpose, zero-K and oversized batch fallbacks.

Full-sized shared and separate-critic forty-five PPO checks compare losses,
every clipped named gradient and every Adam-updated weight over two steps,
including a nonzero frozen-anchor KL term.

For paired throughput, preserve a readback-only release worker, then follow the
[readback comparison commands](training-readback-performance.md#validation-and-reproduction)
with a fresh output directory and `--microbatch 8` or `--microbatch 64`.
For the three-update GPU kernel trace, feed that directory's `requests.jsonl`
to each binary using the [Nsight commands](training-parity-research.md#reproducing-this-research).
Do not run CUDA validation concurrently with performance measurements.
