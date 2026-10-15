# Resident model replay performance

Issue [93](https://github.com/tninesling/tnsr/issues/93) adds stable private-stream
bindings and CUDA Graph replay for forward/backward. Accumulation, dynamic
clipping/status, grouped Adam, publication and export remain outside capture.
The [ownership and fallback contract](resident-replay.md) explains stream safety,
setup evaluations, changed inputs/weights and bounded shape buckets.

## Paired real-model measurements

Baseline is merged `8ff9503` (the backward/dispatch stack). Measured replay code
is `ddf00de`. Same v8.4 archive and seeded 64-transition PPO batch; two excluded
warmups and ten interleaved timed samples on RTX 4080. Each native comparison
uses the same strict/tree ordering. Existing GPU jobs remain running, and tests,
checkpoint validation, throughput and traces run separately.

| Median seconds/update | Strict micro 8 | Strict micro 64 | Tree micro 8 | Tree micro 64 |
| --- | ---: | ---: | ---: | ---: |
| Before replay | 0.310 | 0.172 | 0.227 | 0.161 |
| Replay | 0.290 | 0.162 | 0.209 | 0.152 |
| Original PyTorch eager f32 | 0.232 | 0.0354 | 0.241 | 0.0323 |
| Original PyTorch compiled-policy bf16 | 0.210 | 0.0317 | 0.211 | 0.0293 |

Replay improves whole warm updates about **5–8%** in these paired samples. The
tree microbatch-8 median is approximately the compiled-bf16 median in that run;
microbatch 64 remains about **5x** behind. This is one contended diagnostic batch,
not parity across workloads or evidence of learning-quality parity. Both replay
and tree ordering remain explicit options; the strict reference remains available.

The execute-stage wall time falls sharply (roughly 173 -> 9 ms at strict micro 8),
but that stage includes CPU work and synchronization from interleaved pageable
uploads. Most waiting shifts into the small metrics readback. It is not a pure
CPU-dispatch measurement and does not imply an 18x whole-update improvement.
The remaining GPU work and fragmentation dominate the larger-batch gap.

Both strict/tree paired runs report worker GPU memory **732 -> 786 MiB** at
microbatch 8 and **2,652 -> 2,706 MiB** at 64: about **54 MiB extra** for one
captured bucket. These are native process measurements sampled outside timing,
including workspace, streams/modules and driver graph metadata. Stable input/
parameter bindings account for 24.10 / 24.22 MB of that allocation footprint.
They do not include all workspace/driver metadata. Weak source identities avoid
retaining old generations of weights for inactive cached shapes. Eight shape/clip
entries bound capture ownership; a full multi-bucket capacity envelope remains a
separate parity gate.

## Dispatch trace and setup costs

Warm GPU kernels and logical materialization are unchanged by replay. Tree mode
still executes **25,920 / 3,360 GPU kernels** per update and materializes about
10.106 / 9.869 GB of intermediate logical values. Those byte sums are not DRAM
traffic or peak VRAM. Host model dispatch becomes **8 / 1 graph launches**.

Subtracting setup/recording from separate three-update node-level Nsight traces,
warm individual `cuLaunchKernel` calls fall **25,920 -> 1,264 / 3,360 -> 284** per
update; accumulation/Adam/status kernels remain individual launches. The graph
launches replace the remaining model calls, about a 95% reduction in individual
host kernel submissions. Stable weights refresh by D2D copy once per changed
source/bucket; observations still upload outside capture.

Raw profiles include two pure model warmups and the host calls used to record
capture. Thus microbatch-8 GPU instances are **77,760 -> 83,924** across three
updates, with 6,164 extra setup kernels; microbatch-64 instances are
**10,080 -> 16,232**, with 6,152 extra. Recording adds host launch calls but does
not execute GPU kernels. The native cold counters include the actual warmup
evaluations, and `warmup_executions=2` remains unchanged during warm samples.
Only the final model evaluation contributes gradients once to the optimizer.

Node-level Nsight tracing adds substantial graph-launch instrumentation overhead;
its durations are not an unprofiled before/after timing comparison. Use it to
check submissions, setup, transfers and retained state. GPU kernel totals include
setup and contention and must not be added to whole-update wall time. Raw traces
and API totals are retained in the evidence below.

## Validation and reproduction

All 183 CPU tests and 369 tests in the all-feature CUDA-required suite pass.
The two subsequently added alias/nonfinite recovery cases pass with all ten
resident tests, giving 371 distinct passing CUDA-feature tests. Formatting,
CPU/CUDA checks and all-feature clippy pass. The committed shared/separate PPO
fixtures require active captures for both full/partial shapes while matching
their original two-step gradients and weights.

Actual shared (140 parameters) and separate critics (229 parameters) pass two
PPO updates in both strict/tree replay modes with zero fallbacks and both shapes
captured. Every clipped gradient, updated weight, saved archive and schedule is
checked against the original CPU oracle. Maximum gradient error is below `4.4e-7`;
maximum weight error is below `1.1e-5`, within existing tolerances. Separate tests
cover changed inputs, group rates/moments, host edits, constant replacement,
concurrent published readers, exact IDs above 256, invalid-ID recovery, computed
indices, source aliases, cache clearing and rejected nonfinite updates.

Build/preserve workers from the baseline and replay revisions, then use a fresh
output directory:

```sh
TNSR_PROFILE_PPO=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
  /path/to/forty-five/.venv/bin/python integrations/forty_five/benchmark_training.py \
  ARCHIVE --forty-five-root /path/to/forty-five \
  --baseline-worker /tmp/before --worker /tmp/after --cuda-graphs \
  --microbatch 8 --repeats 10 --compile --output FRESH_OUTPUT
```

Repeat at 64; for equal-order tree comparisons add `--tree-reductions
--baseline-tree-reductions`. Run `validate_ppo.py` separately on both real models
with `--backends cuda --cuda-graphs`, with/without `--tree-reductions`. The
benchmark rejects unexpected fallback and records active shapes and per-worker
memory. For detailed Nsight kernel counts use `--cuda-graph-trace=node`, baseline
requests from `baseline-requests.jsonl` and replay requests from `requests.jsonl`.
Include `cuda_api_sum` alongside the GPU kernel/memory reports.

[replay.json](benchmarks/training-parity/replay.json) contains full paired samples,
source/batch/executable hashes, compact correctness reports, graph states and
trace/API summaries. Full generated artifacts are under ignored
`target/forty_five_replay/`. A subsequent diagnostic-only fix adds staging copies
to captured internal copy counts; it does not alter kernels or the PPO counters
used in these measurements. Mixed precision (#94) and remaining GPU work still
require implementation and calibration.
