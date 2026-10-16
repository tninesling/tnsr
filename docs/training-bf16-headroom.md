# BF16 training headroom on the current forty-five workload

Selective BF16 remains a promising next implementation avenue. A deliberately
incomplete arithmetic substitution improves paired complete-update throughput
by 15.9% at microbatch 64 and 9.7% at microbatch 8 on RTX 4080. This is a research
probe, not an enabled BF16 training policy. Production arithmetic is unchanged.
Physical mixed storage, correct forward/backward rounding and original-trainer
calibration remain required by [#94](https://github.com/tninesling/tnsr/issues/94),
[#114](https://github.com/tninesling/tnsr/issues/114) and
[#115](https://github.com/tninesling/tnsr/issues/115).

## Measure the current bottleneck first

The baseline is merged #127 (`3762720`), using the actual shared v8.4 archive,
fixed 64-transition batch, TF32x3, tree reductions and CUDA Graph replay. The
preserved worker hash is
`4b8fb24d7174bc7d7e1be873040b444a5c1563fd61b7a1c5c0ff59e31111d461`.
A fresh five-update Nsight trace counts 9,775 captured model kernels. GEMMs
account for 102.48 of 207.38 ms summed kernel time (49.4%), reductions 10.2%
and fused pointwise kernels 13.9%. These are separate diagnostic traces; summed
kernel durations and host phases are not wall-clock throughput or DRAM traffic.
The selection uses non-null `graphNodeId`, excluding optimizer kernels outside
the captured model graph.

No other compute process was observed. The desktop remained active. GPU
experiments ran serially, with no builds overlapping timed samples and no clock
or power changes. Absolute times from earlier contended experiments should not
be compared to this run or multiplied into a cumulative speedup.

## Dense GEMMs, including conversion cost

`bf16_gemm_probe` compiles the existing generated Tile IR/PTX backend for actual
current-plan dimensions, transposes, batch layouts and fused bias. It measures
four modes in the same process: TF32x3 with f32 storage, resident BF16, fresh
f32-to-BF16 casts of both operands/bias, and those casts plus output promotion
to f32. All storage is allocated before measurement. Upload, JIT, capture,
warmup, validation and readback are excluded from CUDA-event intervals.
Each sample replays a graph of at least 32 operations; the four modes rotate
and reverse order across 100 samples. All raw times are retained.

One allocation set measures hot reuse. A second regime rotates enough allocation
sets that the BF16 plan payload exceeds twice the device's 64 MiB L2. This avoids
mistaking a cache-resident repeated matmul for a full-model memory workload.
Fresh input conversion in every GEMM can be redundant in a future mixed graph;
conversely, these probes omit cross-operation casts, final split-K sums,
non-bias epilogues and attention's noncompact QKV view maps. They are component
measurements, not end-to-end bounds or an attention benchmark.

Streaming medians, in microseconds:

| Current family (M,K,N; batch) | TF32x3 | Resident BF16 | BF16 input casts | BF16 roundtrip to f32 |
| --- | ---: | ---: | ---: | ---: |
| FFN down wgrad partial (683,1024,256; 4) | 159.81 | 122.53 | 125.92 | 131.90 |
| FFN down forward (4096,683,256) | 164.00 | 117.47 | 120.70 | 129.92 |
| FFN gate/up wgrad partial (256,1024,683; 4) | 166.69 | 122.66 | 124.51 | 130.98 |
| FFN gate/up forward (4096,256,683) | 191.65 | 105.82 | 111.01 | 129.22 |
| FFN gate/up input gradient (4096,683,256) | 153.41 | 95.74 | 107.96 | 117.66 |
| QKV forward (4096,256,768) | 209.28 | 115.71 | 119.65 | 139.68 |
| QKV wgrad partial (256,1024,768; 4) | 173.82 | 123.68 | 136.38 | 142.36 |
| QKV input gradient (4096,768,256) | 138.53 | 46.78 | 74.56 | 83.97 |
| Microbatch-8 FFN wgrad (683,512,256) | 25.72 | 22.94 | 21.97 | 24.24 |
| Microbatch-8 FFN down (512,683,256) | 24.52 | 26.65 | 25.67 | 27.27 |

Large streaming roundtrips save 17.5–39.4% GEMM time. Small-batch results are
mixed: the down projection regresses 11.2%. Cache-hot results are generally
more favorable and are retained separately. Do not apply one headline BF16
GEMM ratio to the whole trainer or blanket-convert small operations.

## Whole-model arithmetic substitution

A separately patched private worker replaces eligible TF32x3 schedules with
BF16 shared operands and f32 accumulation. Global inputs, outputs, gradients,
master parameters and Adam remain f32. In particular, it omits BF16 output and
backward rounding, so it does not implement the original autocast contract or
physical BF16 memory savings. Async raw-f32 staging is disabled for this probe,
and BF16 instructions use K16 rather than TF32's K8. A separate signature
version prevents generated-code cache confusion. The worker explicitly reports
`bf16_compute_probe`; the paired driver checks that label and the baseline's
`tf32x3` label. None of this patch is applied to production source.

Both workers start from the same archive, then evolve independently on the
same repeated fixed batch. Numerical trajectories differ; finite metrics are
checked, but this is not a correctness comparison. Each request includes JSON,
staging, model execution, clipping, Adam and finite checks. Five warmups are
excluded and retained. AB/BA order alternates every pair.

| Microbatch | Pairs | Baseline median | Probe median | Probe wins | Median paired throughput gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 200 | 45.16 ms | 39.02 ms | 200 | 15.9% (15.7–16.3%) |
| 8 | 100 | 65.32 ms | 59.55 ms | 100 | 9.7% (9.1–9.8%) |

Intervals use a circular block bootstrap over paired ratios: 4,000 resamples,
ten adjacent pairs per block, fixed seed. They describe this run, rather than
hardware/workload generality. A 20-pair check of the reusable CLI independently
reproduces the label/protocol checks; it is not pooled into these results.

At microbatch 64 both workers retain 4,055 graph nodes, 34 split-K contractions,
2,239 kernels/full update, one graph launch and zero replay fallbacks. Planned
intermediate output sizes remain 7.319 GB; these are cumulative logical sizes,
not peak VRAM or measured memory traffic. The separate probe trace has the same
9,775 captured kernels, with GEMMs falling 102.48 → 69.98 ms and total captured
kernel time 207.38 → 174.26 ms. Most other categories stay steady. This supports
arithmetic as the cause of the measured gain.

The arithmetic substitution alone does not establish the remaining gap to
PyTorch: there is no fresh matched PyTorch comparison in this experiment.

## Numerical findings are still open

The native BF16 GEMM's declared semantics are BF16 operands, f32 accumulation,
fused bias, then one BF16 output rounding. GPU casts match host BF16 rounding
exactly; replicas and output promotion are checked. Against FP32 GEMM of
BF16-quantized inputs followed by BF16 output rounding, at least 99.8867% of
outputs match exactly in every tested case, and all differences are at most one
BF16 ULP. The worst normalized maximum error is 4.51e-6.

Original PyTorch 2.11.0+cu130 defaults to BF16 reduced-precision reductions.
That reference differs on several shapes. The microbatch-8 down projection has
max error 0.0625, RMSE 0.01809 and 51.99% exact agreement; it fails the probe's
1/128 normalized sanity check against that original default. Disabling reduced
precision removes many discrepancies, but does not make every bias/accumulation
case exact. All three comparisons are retained. The default flag was restored
after diagnostics, and no original-PPO tolerance was changed. The permissive
primitive sanity threshold is not a training acceptance tolerance.

An auxiliary unsplit K4096 TF32x3 contraction also fails the newly selected
2e-5 normalized f32 sanity threshold (4.21e-5 observed). Independent FP64 checking
finds native max error 1.23e-4 versus 4.22e-6 for PyTorch FP32. This unsplit case
is absent from the current large-model split-K path; all current probe families
pass that f32 sanity check. The failure is preserved rather than hidden by a
relaxed threshold or deletion of timings. It is an additional reason to retain
strict f32 and calibrate actual shapes and accumulation orders.

## Implementation decision

Start with one complete linear forward/backward island using physical BF16
buffers and explicit f32/BF16 cast boundaries. Keep f32 master parameters,
gradient accumulation and optimizer state, with exact integer indices. Include
fused-bias output rounding, upstream-gradient casting, BF16 gradient output and
bias reduction, then promotion back to f32. Test shared-producer contributions
and accumulation before integrating the model policy. Preserve the audited
operator boundaries in [the BF16 contract](training-bf16-contract.md).

Measure that complete island including all conversions and replay, on large
and small real shapes. Then expand into the adapter only after named gradients
and multistep optimizer state calibrate against the original unchanged BF16
trainer for both critics, partial batches and clipping/recovery cases. Full PPO
throughput/memory and learning-quality gates remain open. If physical storage
and cast handling erase the measured benefit, revisit pointwise/reduction
fusion rather than enabling a slower policy.

## Reproduction

```sh
cargo build --release --features cuda --example bf16_gemm_probe --example forty_five_worker
/path/to/forty-five/.venv/bin/python integrations/forty_five/profile_bf16_gemms.py \
  --output FRESH_DIRECTORY
```

The driver runs all timed native processes before importing PyTorch for
oracles. `--validate-only --output EXISTING_DIRECTORY` rechecks saved outputs
without replacing measurements. `--include-unsplit-diagnostic` adds the
auxiliary K4096 case; it is excluded from the default current-workload sweep.
The original run included it and retains its failure.

For actual plan/PTX dumps, run the native worker with `TNSR_PROFILE_PPO=1` and
`TNSR_PROFILE_PPO_DUMP=FRESH_DIRECTORY`. Each compiled microbatch/clip bucket
writes once. Existing files are rejected to avoid mixing stale artifacts.
Run Nsight separately from wall-clock measurement:

```sh
TNSR_PROFILE_PPO=1 nsys profile --trace=cuda --cuda-graph-trace=node \
  --sample=none --cpuctxsw=none --output=FRESH_TRACE \
  /path/to/worker ARCHIVE cuda < PROFILE_REQUESTS.jsonl
nsys export --type sqlite --output=FRESH_TRACE.sqlite FRESH_TRACE.nsys-rep
```

To reproduce the private arithmetic experiment, use a separate checkout of this
revision, apply [bf16-compute-probe.patch](benchmarks/training-parity/bf16-compute-probe.patch),
build and preserve its worker, and retain an unpatched baseline worker. This
patch is intentionally research-only and must not be applied to a production
checkout. Generate the archive/fixed batch with the existing
[integration workflow](../integrations/forty_five/README.md), then run:

```sh
TNSR_PROFILE_PPO=1 /path/to/forty-five/.venv/bin/python \
  integrations/forty_five/benchmark_native_pair.py ARCHIVE \
  --batch BATCH.json --baseline-worker BASELINE --worker PATCHED_CANDIDATE \
  --output FRESH_PAIRS --microbatch 64 --warmups 5 --repeats 200 \
  --tf32x3 --tree-reductions --cuda-graphs --bf16-compute-probe
```

Repeat with microbatch 8 and 100 pairs. The precision-probe flag requires the
separately patched candidate; an ordinary production candidate fails its label
check. The production worker ignores the private environment flag, verified
explicitly after restoring and rebuilding. The unpatched primitive rebuild's
SHA-256 exactly matches the originally measured primitive binary.

[bf16-headroom.json](benchmarks/training-parity/bf16-headroom.json) retains every
primitive time, full paired requests/warmups, metrics, replay/compile metadata,
worker/source/checkpoint/batch hashes, original measurement-driver source,
bootstrap settings, trace aggregates, failed numeric checks and artifact hashes.
Complete binary outputs, plan/PTX dumps and Nsight reports remain under ignored
`target/forty_five_bf16_headroom/`.
