# Resident training performance — issue 91

The parameter-storage layer and CUDA `ResidentAdam` remove unchanged source
uploads and per-microbatch gradient downloads. General TileIR kernels accumulate
virtual-layout gradients and run grouped Adam; the runtime downloads small
metrics/norm/finite-check values. Model exports and explicit diagnostics remain
synchronization boundaries. The [ownership contract](resident-training.md)
describes host mutation, lazy reads, staged finite checks and numerical policy.

## Paired real-model measurements

Baseline is merged revision `2145dd8`, before both residency stack layers. Same
v8.4 archive and real seeded 64-transition PPO batch, native strict-f32 matmuls,
anchor KL disabled, two excluded warmups and three timed samples. Cases are
interleaved on the RTX 4080 with existing GPU work left running. Correctness
validation, throughput measurements and traces run separately.

| Median seconds per update | Microbatch 8 | Microbatch 64 |
| --- | ---: | ---: |
| Before residency | 0.884 | 0.279 |
| Resident training | 0.256 | 0.170 |
| Original PyTorch eager f32 | 0.213 | 0.0360 |
| Original PyTorch compiled-policy bf16 | 0.179 | 0.0293 |

Residency improves whole-update time **3.45x / 1.64x**. The eager-f32 gap is about
**1.2x / 4.7x** in these diagnostic samples; parity across the workload envelope
has not been established. Microbatch-64 samples vary substantially under GPU
contention, so these three samples are not an exclusive-GPU performance gate.

Native host gradient extraction and accumulation disappear from the normal
update. CPU Adam/clipping/finite checking falls from roughly 39–41 ms to a
2–6 ms resident step. The execute stage is asynchronous now; waiting in the
small metrics readback includes pending GPU work. Stage attribution therefore
changes, and a larger metrics-stage time does not mean computing metrics became
more expensive. Compare total update time rather than adding overlapping stages.

## Transfer trace and remaining work

Separate Nsight traces each contain three complete updates, including first-use
uploads. Counts below divide totals by three; upload counts include initialization
and are not steady-state counts.

| Copies per update, amortized | Before, micro 8 | After, micro 8 | Before, micro 64 | After, micro 64 |
| --- | ---: | ---: | ---: | ---: |
| Host to device | 3,200 | 296.3 | 400 | 149.3 |
| Device to host | 1,216 | 10 | 152 | 3 |

The resident microbatch-8 trace downloads only about **1 KiB over all three
updates**, instead of full gradients. Weights and constants upload once; normal
subsequent uploads are observations/settings. Generated accumulation/Adam/check
kernels add work: launches increase **25,592 -> 26,856 / 3,199 -> 3,483**. GPU
kernel-duration totals in these contended traces are **0.152 -> 0.211 s /
0.117 -> 0.106 s** per update. Residency is a host/runtime improvement, not a
claim that device computation got uniformly faster. Trace outliers are large;
profiler durations overlap CPU activity and must not be added to wall time.

Large reductions, broadcasts and matmuls dominate the updated kernel trace.
The full backward graph still triggers a fusion-cycle fallback that discards
pointwise/reduction regions. Issue 92 should preserve safe backward regions and
measure the numerical policy explicitly. Issue 93 follows with stable buffers
and replay; issue 94 adds a calibrated mixed-precision policy.

Raw paired samples, stage timings, executable/source/batch hashes, compact
shared/separate correctness reports and GPU trace summaries are in
[resident.json](benchmarks/training-parity/resident.json). Generated reports are
under ignored `target/forty_five_resident/`.

## Validation and reproduction

All 182 CPU / 354 all-feature CUDA-required tests pass, plus formatting,
CPU/CUDA builds and clippy. The final module split passes all seven resident
ownership/index/optimizer tests. Actual shared (140 parameters) and separate
critic (229 parameters) PPO losses, every clipped named gradient and every
Adam-updated weight match the existing oracle over two updates with nonzero
anchor KL, including partial microbatches. Dedicated tests check three-step
weighted accumulation, moments, custom betas/decay/epsilon, separate rates,
schedule changes, shared parameters, host writes, failed finite checks, lazy
export/inference and parameter lifetime after owner destruction.

Preserve the baseline release worker, build the changed worker and use the
[paired benchmark commands](training-readback-performance.md#validation-and-reproduction)
with fresh directories and microbatch 8/64. Run
`integrations/forty_five/validate_ppo.py` on each actual model separately before
profiling. For GPU traces, feed the benchmark's three-request `requests.jsonl`
to each binary using the [Nsight commands](training-parity-research.md#reproducing-this-research).
Keep CUDA tests and validation separate from throughput runs.
