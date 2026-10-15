# Backward fusion and dispatch preparation

Issue [92](https://github.com/tninesling/tnsr/issues/92) preserves legal fusion in
backward DAGs and exposes parallel reduction ordering explicitly. The following
dispatch preparation for [93](https://github.com/tninesling/tnsr/issues/93) removes
repeated plan scans/clones and CUDA function lookups. CUDA Graph replay is still
separate work.

## Planner and numerical behavior

The old quotient-cycle fallback discarded all pointwise/reduction regions. The
planner now splits only cyclic pointwise groups, trims cyclic matmul epilogues,
and finally reduces cyclic reduction regions to their anchors. Independent
regions and explicit schedules survive. Cost alternatives retain legal pointwise
partitions and regroup only freed members. Serial reduction working sets and
lane strides help select materialization; these estimates are profitability
heuristics, not measured cache hit rates.

Recovering regions exposed an existing singleton-axis bug: full and reduced
shapes coincide, so shape-based output classification left some outputs unwritten.
Output phases now follow SSA provenance. Tests cover escaping full/reduced values
and shared singleton inputs in serial and block schedules.

Strict increasing-index model reductions remain the default. CUDA PPO can opt
into `settings.tree_reductions=true`, or `--tree-reductions` in the training,
validation and benchmark scripts. Fixed warp/block trees now include standalone
reductions. This explicitly changes accumulation order; storage, matmul,
gradients and Adam remain f32. The policy cannot change within a trainer.

Plans index physical outputs once and executors share immutable steps and cache
CUDA functions until recompilation. Online rewrites rebuild the output index.
Complete ordered fusion diagnostics share immutable feature allocations;
`FusionAlternative.features` now contains `Arc<FusionFeatures>`. The scorer
interface and feature values remain unchanged. This avoids retaining thousands
of copies of the same kernel metadata for large backward graphs.

## Real-model measurements

Baseline: merged residency revision `2bf49fe`. Measured stack: `7d78366` before
this evidence-only commit. Same v8.4 archive and seeded 64-transition PPO batch,
anchor KL disabled, two excluded warmups, ten timed samples per case, interleaved
with original PyTorch modes on RTX 4080. Existing GPU jobs remain running; these
are contended diagnostic samples, not an exclusive-GPU throughput gate.

| Median seconds/update | Strict micro 8 run | Strict micro 64 run | Tree micro 8 run | Tree micro 64 run |
| --- | ---: | ---: | ---: | ---: |
| Residency baseline | 0.301 | 0.154 | 0.291 | 0.131 |
| Combined stack | 0.285 | 0.150 | 0.207 | 0.141 |
| Original PyTorch eager f32 | 0.212 | 0.0347 | 0.196 | 0.0325 |
| Original PyTorch compiled-policy bf16 | 0.194 | 0.0322 | 0.177 | 0.0317 |

Default strict order improves about **5% / 3%**. Explicit tree ordering improves
microbatch 8 about **1.41x**, while microbatch 64 regresses about **8%** against
its paired baseline. Keep that tradeoff explicit: tree ordering is opt-in and
training parity with PyTorch across the batch envelope has not been reached.
The initial strict fusion layer alone regressed microbatch 8 about 6%; the
dispatch preparation was included before merging and remeasuring the stack.

| Native work/update | Baseline micro 8 | Strict micro 8 | Tree micro 8 | Baseline micro 64 | Strict micro 64 | Tree micro 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU kernel launches | 26,856 | 26,160 | 25,920 | 3,483 | 3,408 | 3,360 |
| Intermediate materialized bytes, decimal GB | 10.308 | 10.292 | 10.106 | 10.138 | 10.123 | 9.869 |

Materialized bytes sum logical value sizes over all executions. They are not
DRAM traffic or peak VRAM: pooling reuses allocations. The baseline byte probe
adds the same optional counters to baseline source without changing arithmetic.
Fragmentation improves modestly; the remaining device work is substantial.

Separate three-update Nsight traces confirm tree-path launches and show GPU
kernel-duration totals of **0.216 -> 0.139 s / 0.138 -> 0.135 s** per update.
Initialization is included. Device-to-host counts remain **10 / 3** per update,
amortized; residency's transfer savings persist. Large scalar-f32 matmuls now
dominate the tree trace. Trace outliers are large, and overlapping profiler
durations must not be added to wall time. Stable buffers/replay (#93) and
operation-selective bf16 (#94) remain necessary follow-ups.

## Correctness and reproduction

All 183 CPU tests and all 363 preexisting CUDA-required tests pass. The added
diagnostics stress fixture passes after correcting its setup, and all 147 CUDA
library tests pass together. Formatting, CPU/CUDA checks and all-feature clippy
pass. Existing regressions cover cache invalidation/recompilation, virtual and
multiple outputs, shared normalization/SiLU backward, masked nonfinite inputs,
singleton axes and strict/tree schedule boundaries including empty reductions.

Both actual shared (140 parameters) and separate critics (229 parameters) pass
two complete PPO updates in each reduction mode, checking losses, every clipped
gradient and every Adam-updated weight, partial microbatches, nonzero bidding KL,
scheduler changes and saved archive restores. Maximum gradient error is below
`3.7e-7`; maximum weight error is below `1.3e-5`, within existing oracle tolerances.
This establishes implementation compatibility, not learning-quality parity.

Preserve the baseline release worker, build the changed worker, and run:

```sh
TNSR_PROFILE_PPO=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
  /path/to/forty-five/.venv/bin/python integrations/forty_five/benchmark_training.py \
  ARCHIVE --forty-five-root /path/to/forty-five \
  --baseline-worker /tmp/baseline-worker --output FRESH_OUTPUT \
  --microbatch 8 --repeats 10 --compile
```

Repeat with microbatch 64 and optionally `--tree-reductions`. Baseline requests
omit the new setting for older workers; profiling uses `baseline-requests.jsonl`
for the baseline and `requests.jsonl` for the new worker. Run `validate_ppo.py`
separately on each actual model with `--backends cuda`, with/without
`--tree-reductions`, then profile separately using the
[Nsight commands](training-parity-research.md#reproducing-this-research).

Full paired samples, hashes, compact correctness reports, baseline byte probes
and trace summaries are in [backward.json](benchmarks/training-parity/backward.json).
Generated full reports and traces are under ignored `target/forty_five_fusion/`.
