# BF16 GEMM schedule experiments

Follow-up: [transposed operand staging](training-bf16-transposed-staging.md) adds
another accepted improvement, 43.00 → 38.61 ms in paired complete updates. Its
matched comparison measures 38.33 ms native BF16 versus 42.14 ms native f32.
The results below retain the register-tile baseline used for that follow-up.

Register tiling, shorter index lifetimes, and the tuned split-K schedule now
have an uncontended complete-update acceptance result on the RTX 4080. The saved
baseline and current candidate measured **47.71/42.77 ms** per full mixed PPO
update in 64 alternating AB/BA pairs: a **1.1128× median paired throughput ratio**,
with 64/64 candidate wins. The tuned mixed path remains opt-in; it is still
slightly slower than current native f32 and substantially slower than compiled
PyTorch BF16 on this workload.

[Uncontended acceptance evidence](benchmarks/training-parity/bf16-gemm-idle-acceptance.json)
retains every request, both five-mode comparisons, all schedule samples, clean
kernel traces, numerical controls, and source/binary hashes. The kernel sources
match the earlier tuning candidate exactly; this follow-up validates that
candidate without adding another kernel change.

[Earlier contended evidence](benchmarks/training-parity/bf16-gemm-tuning.json)
retains the original experiments, compiler resource reports and diagnostic
results. Those timings remain historical diagnostics. This advances
[#124](https://github.com/tninesling/tnsr/issues/124); it does not close it.

## What changed

- Large BF16 matrices (`M >= 2048`, `N >= 256`, `K >= 64`) use a 64×64×32
  block tile when resource budgets permit. Four computing warps each own four
  16×16 accumulators and reuse operand fragments across those accumulators.
  The existing asynchronous pipeline remains available for eligible views.
- The 32×32 split-K tile uses a K extent of 64 and padded shared layouts.
  Partial outputs remain f32, with one BF16 rounding after their final sum.
- Shared index expressions are defined immediately before their first consumer
  in the chosen scope. Epilogue addresses no longer stay live across the entire
  contraction. Large synchronous BF16 register tiles retain normalized index
  expressions without explicitly hoisting all of them; pipelined tiles retain
  sharing. The PTX cache signature changes with the generated code.

All kernels still come from Tile IR. BF16 boundaries, f32 accumulation, exact
indices, and resident Adam are preserved. F32 keeps its existing tile selection.

Blindly enlarging the original tile was a regression. For the QKV weight-gradient
probe, the original 32×32×32 kernel used 56 registers/thread, whereas an 8-warp
32×64×64 tile used 225. Larger variants spilled. Moving index definitions reduced
the latter to 168, but it remained slower. The selected split-K configuration
uses 64 registers; the selected forward and pipelined input-gradient probes use
72 and 84 respectively, with no spills in the compiler reports.
The actual model trace also confirms the selected geometry: the QKV weight
gradient falls from 86 to 72 registers/thread, while its shared allocation grows
from 8,192 to 13,824 bytes. The large QKV forward uses the 35,840-byte double
buffer and four fragments per warp. Resource metadata for all 182 actual BF16
GEMM launches is retained separately from the synthetic probe measurements.

## Isolated GEMM measurements

The sweep preserves the real dimensions and interleaved split-K transpose maps.
It uses synthetic BF16 operands and f32 output/partial storage. It excludes
bias/activation epilogues, attention-specific QKV maps, casts, and Adam.
Each comparison uses preallocated buffers, warmed graphs of eight operations,
48 CUDA-event samples, and rotated/reversed configuration order in one process.
The table below is the uncontended repeat on 2026-10-21.
Every candidate's full f32 output matched the baseline; sampled independent f64
dot products also passed.

Both geometries in this table use the updated index-definition placement.
Forward baselines include padding, and the input-gradient baseline includes
double buffering. These are configuration comparisons within the sweep, not
speedups over the old complete PPO executable.

| Case | Baseline median | Selected median | Median paired speedup |
| --- | ---: | ---: | ---: |
| QKV weight gradient, four K partitions | 129.78 µs | 80.00 µs | 1.622× |
| FFN down weight gradient, four K partitions | 119.52 µs | 74.37 µs | 1.607× |
| FFN gate/up weight gradient, four K partitions | 118.66 µs | 73.34 µs | 1.618× |
| QKV forward | 72.96 µs | 56.32 µs | 1.297× |
| FFN down forward | 68.34 µs | 56.96 µs | 1.201× |
| QKV input gradient | 39.94 µs | 31.74 µs | 1.259× |

No external CUDA compute workload was observed before or after the repeat.
Desktop GPU clients and other host work were left running. This repeat supports
the selected configurations. A 64×128 synchronous tile lowered the isolated FFN
down-forward median to 46.98 µs, but larger geometries were inconsistent across
families. That configuration is not selected in the implementation and has no
complete-model acceptance result.

## Uncontended complete PPO acceptance (2026-10-21)

The saved pre-tuning binary and current build started from the same width-256
shared-critic archive and fixed real-game-derived 64-transition batch. Both used
microbatch 64, tree reductions, TF32x3 for remaining f32 work, graph replay and
resident Adam. Three warmups per worker were excluded; 64 complete requests per
worker alternated AB/BA. Timers include feeds, loss, backward, clipping, Adam,
synchronization, metrics and JSON request handling.

| Paired full mixed update | Median |
| --- | ---: |
| Saved pre-tuning baseline | 47.71 ms |
| Tuned candidate | 42.77 ms |

The median paired throughput ratio is **1.1128×**, with 64/64 candidate wins.
The ratio of aggregate medians is 1.1155×; latency falls 10.35%. These are distinct
summaries of the same retained samples. Every requested replay remained active
with zero fallbacks. An independent 64-pair f32 control measured 42.83/42.03 ms,
a 1.0198× median paired throughput ratio and 53/64 candidate wins. No f32
regression was observed.

A subsequent five-mode comparison uses the **current executable for both native
f32 and mixed BF16**, identical initial weights/batch, reverse mode order per
iteration, three excluded warmups and 32 retained updates per mode. Original
PyTorch uses its existing updater and enabled BF16 reduction policy; compiled
mode wraps the model with `torch.compile(dynamic=True)`.

| Complete update, current comparison | Median |
| --- | ---: |
| Current native f32, tree/replay/TF32x3 | 43.02 ms |
| Tuned native mixed BF16 | 43.75 ms |
| Original eager PyTorch f32 | 30.19 ms |
| Original eager PyTorch BF16 | 34.07 ms |
| Original compiled PyTorch BF16 | 27.41 ms |

Mixed BF16 is 1.7% longer than current native f32 and 1.60× the compiled PyTorch
BF16 median. F32 remains the default. An earlier repeat using saved pre-tuning
native f32 as the comparator is also retained, rather than selecting between
runs. PyTorch samples varied substantially in both runs despite no external
CUDA compute process: compare these modes within their own run, not absolute
medians across dates. All samples and before/after process snapshots are saved.

Two clean Nsight traces verify 32 repetitions of the same 1,414 ordered model
kernels. Excluding three warmups, the captured model span falls from 44.00 to
39.07 ms. Its 4.93 ms reduction agrees closely with the 4.94 ms unprofiled
complete-request median difference, although the spans exclude optimizer,
uploads, request handling and metrics.

| Model kernel group | Baseline median summed GPU time | Candidate |
| --- | ---: | ---: |
| BF16 GEMM | 14.12 ms | 11.82 ms |
| BF16 split-K GEMM | 12.55 ms | 9.92 ms |
| Other f32 pointwise work | 8.52 ms | 8.51 ms |
| Reductions | 5.63 ms | 5.64 ms |
| Other BF16 pointwise work | 1.53 ms | 1.54 ms |
| Cast/copy | 1.43 ms | 1.43 ms |

The two GEMM groups account for essentially all of the measured reduction.
Their combined candidate time is 21.74 ms, about 56% of the captured model span,
so GEMM remains the largest target. Ordered labels match between binaries;
independent graph-node order/count checks pass within each trace. Hardware
counters remain unavailable under the existing permissions.

Fresh three-step controls compare a second baseline run and the current
candidate against an independent first baseline. Every named clipped gradient
and all published weights are checked:

| Step | Baseline vs baseline gradient relative L2 | Candidate vs baseline |
| --- | ---: | ---: |
| 1 | 1.672e-7 | 1.739e-7 |
| 2 | 1.344e-7 | 1.382e-7 |
| 3 | 2.321e-4 | 1.251e-7 |

The candidate's maximum weight discrepancy across all three steps is 7.45e-9;
the second baseline reaches 9.98e-6. Candidate errors remain at the small native
numerical floor. This does not establish learning-quality parity with PyTorch
or remove the earlier BF16 calibration limits.

Acceptance validation passed: 201 CPU tests, 406 tests with CUDA required,
four executable documentation examples, CPU/CUDA builds, formatting, and
all-target/all-feature Clippy. The ignored sweep passed separately for all six
cases, checking every configuration before timing. Evidence integrity checks
verify all expected samples and zero native replay fallbacks.

## Earlier contended diagnostics

The original width-256 shared critic ran complete 64-transition updates from
the same archive and real-game-derived batch. Four balanced AB/BA serial blocks
kept memory bounded alongside the existing job, retaining 64 measured requests
per executable after three warmups per block. Baseline/candidate medians were
96.22/85.22 ms; the median paired throughput ratio was 1.075, with 37/64 candidate
wins. This is a **provisional diagnostic**, not a clean performance acceptance
result: contention changed substantially within blocks.

A separate alternating AB/BA f32 comparison retained 64 requests per executable.
Its baseline/candidate medians were 89.48/84.33 ms, with a 1.050 median paired
throughput ratio and 40/64 candidate wins. No f32 performance regression was
observed under these conditions; this is also contended data and does not
replace the earlier uncontended f32 reference.

Nsight Systems verified 32 repetitions of the same ordered 1,414 model kernels
in each executable. Baseline spans ranged roughly 46–97 ms and candidate spans
40–88 ms as the external job changed phase. Their overall medians must not be
treated as a causal speedup. All launches and kernel groups remain in the raw
evidence. Nsight Compute hardware counters were unavailable (`ERR_NVGPUCTRPERM`),
and administrator access required a password. Driver settings were not changed.

Three real PPO updates also compared a candidate run and a second baseline run
against an independent first baseline run. Aggregate clipped-gradient relative
L2 errors were:

| Step | Baseline vs baseline | Candidate vs baseline |
| --- | ---: | ---: |
| 1 | 1.682e-7 | 1.578e-7 |
| 2 | 1.337e-7 | 1.566e-7 |
| 3 | 2.32087e-4 | 2.32088e-4 |

Both third-step comparisons had the same maximum weight difference, 9.9773e-6.
The candidate remained within observed baseline variability. Per-parameter
errors and the diagnostic driver are retained. This check does not establish
learning-quality parity with PyTorch or remove the earlier BF16 calibration
limits.

Validation passed: 201 CPU tests, 406 CUDA-required tests, four executable
documentation examples, CPU/CUDA builds, formatting, and Clippy across all
targets/features. The opt-in performance sweep is excluded from ordinary tests;
its separate run validates every candidate before timing. The additional new
CUDA test covers register-tiled BF16 with M/N/K edges and replay.

## Reproduce and remaining headroom

Run the opt-in sweep with a fresh output directory:

```bash
TNSR_BF16_TUNE_OUTPUT=target/bf16-sweep \
  cargo test --release --features cuda --lib bf16_schedule_sweep \
  -- --ignored --nocapture
```

`TNSR_BF16_TUNE_CASE=qkv-wgrad` selects a single case. The sweep saves all PTX,
schedule descriptions, correctness checks and samples. `ptxas -arch=sm_89 -v`
can inspect register use and spills without counter permissions.

Reproduce the native mixed acceptance with the saved pre-tuning binary and a
fresh output directory:

```bash
OPENBLAS_NUM_THREADS=1 TNSR_PROFILE_PPO=1 /path/to/forty-five/.venv/bin/python \
  integrations/forty_five/benchmark_native_pair.py /path/to/archive \
  --batch /path/to/fixed-batch.json --baseline-worker /path/to/baseline-worker \
  --worker target/release/examples/forty_five_worker --output target/bf16-paired-repeat \
  --microbatch 64 --warmups 3 --repeats 64 --mixed-bf16 --baseline-mixed-bf16 \
  --tree-reductions --tf32x3 --cuda-graphs
```

Omit both mixed flags for the f32 control. Use the five-mode command in
[actual mixed PPO](training-mixed-ppo.md#reproduce) with the current worker also
passed as `--baseline-worker`, plus `--baseline-tree-reductions`,
`--baseline-tf32x3`, and `--baseline-cuda-graphs`, to include current native f32.
The acceptance evidence records hashes for the archive, batch, workers and
benchmark drivers.

The [transposed staging follow-up](training-bf16-transposed-staging.md) now
implements and accepts layout-aware asynchronous copies for aligned BF16 views.
The [hybrid follow-up](training-bf16-hybrid-staging.md) also accepts unaligned
FFN operand staging and wider split-K gradient tiles. Shape-matched cuBLASLt timings and
hardware counters remain useful for measuring further headroom. The wider
synchronous FFN tile also needs a controlled complete-update ablation before
selection. F32 backward pointwise/reduction regions account for another 14.15 ms
of the current model trace.
