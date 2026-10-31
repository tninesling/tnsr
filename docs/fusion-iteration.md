# Fusion iteration: joint serial scheduling and layout readback

This pass builds on `c669048`, using an RTX 4080. It improves strict reductions
and host materialization of virtual output layouts. It does not enable a learned
profitability model or change strict floating-point reduction association.

## Compiler changes

The old cost model assumed 256 threads per serial reduction block, while the
actual legacy CUDA launch used 1024. Serial reduction regions now carry an
explicit validated block size, and the model uses the launch geometry that will
actually execute. Standalone elementwise and materialized-view estimates also
use their actual launch geometry.

Cost-aware fusion considers powers-of-two block sizes from 32 through 1024
alongside producer materialization and epilogue splitting. Candidates exceeding
the next power of two of the output-fiber count are pruned; the existing plan
remains available and wins equal-cost ties. Nonempty F32 standalone sums and
means can enter the same serial-region search. Empty reductions retain their
legacy behavior, including empty-mean NaNs. Each fiber still accumulates in its
original increasing-index order. Cooperative reductions retain their separate
numerical policy. Automatic half-precision repartitioning remains disabled
because extra stores can change rounding.

Cache pressure now accounts for concurrently resident blocks per SM. Compiler
features expose reduction extent/stride/lanes, repeated domain passes, input
index complexity and actual block geometry before candidate execution. An
explicit serial-block override supports controlled experiments and invalidates
cached compilation/replay state when changed.

The layout investigation found a separate bottleneck: end-to-end execution
includes CPU interpretation of virtual output views. Symbolic flattened-address
proofs now recognize ordinary reshapes even when their intermediate coordinates
contain floor/mod. Contiguous full-storage reshapes reuse the downloaded vector;
positive 2D strided copies avoid per-element checked expression evaluation.
Bounds and original arithmetic overflow are proved before accepting a shortcut.
See [readback implementation and correctness](fusion-readback-layout.md).

## Measurements

The reproducible driver is `examples/fusion_probe.rs`. It checks outputs against
CPU, checks all MLP parameter gradients, warms each candidate, and measures
end-to-end execution and isolated output readback separately. The comparison
uses 16 baseline samples of at least 5 ms and 20 current samples of at least
10 ms, with current candidate order reversed. The baseline layout run uses the
same shape-changing reshape as the current driver. Intervals are approximate
95% intervals over sample means, not independent hardware repetitions.

Selected strict-F32, cost-aware results at 256×256:

| Case | Before, µs | After, µs | Speedup |
|---|---:|---:|---:|
| Factored axis-0 reduction | 19.41 | 10.65 | 1.82× |
| Pointwise producer, axis 0 | 22.11 | 13.29 | 1.66× |
| Pointwise producer, axis 1 | 41.66 | 21.89 | 1.90× |
| Reduction chain | 36.03 | 19.14 | 1.88× |
| Full-domain reduction epilogue | 71.14 | 57.28 | 1.24× |
| Reshape + unary | 6386.36 | 61.09 | 104.54× |
| Reshape round trip | 10305.09 | 72.52 | 142.10× |
| Binary then transpose | 395.86 | 123.96 | 3.19× |

The large layout gains are CPU readback improvements, not CUDA kernel speedups.
Binary operations on transposed inputs are essentially unchanged. Benefits are
shape dependent: the 128×192 factored case retains the same kernel and shows a
12.32→13.88 µs timing difference with overlapping intervals. Small cache-resident
cases can retain 1024 threads on model ties, leaving tuning opportunities.

MLP forward/backward shows no established throughput improvement. The initial
256×256 backward comparison was 67.97→70.79 µs, with identical launch count
and PTX source size; a separate 24-sample, 10-ms training repeat measured
67.80→67.43 µs. That repeat replaces the training rows in the results table.
Other training shapes still vary in both directions, so this is a regression
check rather than evidence of a training speedup. Compilation times are diagnostic only: differing driver JIT
cache warmth makes them unsuitable for a before/after speedup claim.
The expanded search does additional compiler work per reduction; large PPO
graphs remain an important compilation-cost limit. The full CUDA run's strict
and graph PPO parity tests each took roughly 110 seconds, including compilation
and execution. This pass establishes no compilation-speed improvement.

The [compact comparison table](benchmarks/fusion-iteration.csv) retains all
cases, variants, intervals, readback times, selected serial block sizes and
fusion-selection overhead. Tree variants are separate numerical-policy controls.
Raw probe dumps and build artifacts stay outside the repository; no tensor
fixtures are saved by this driver.

## Profitability-model experiment

`summarize_fusion.py` exports graph and pre-execution plan features to the
existing trainer. `--feature-mode compiled` excludes runtime execution counters
and records the need to compile candidate plans separately from any need to
execute them. The [66-row pilot](benchmarks/fusion-compiler-training.jsonl) covers
33 cases, 11 families and only three shape groups. Its
[validation report](benchmarks/fusion-compiler-report.json) rejects enablement:
shape coverage is insufficient and separate heuristic-cost labels are absent.
After the matched training repeat, held-out geometric mean regret is 1.007×
versus 1.017× for the current optimizer on these candidates, too small and
narrow a pilot to justify deployment. Identical candidate plans are not fingerprinted in this small pilot;
future collection should add exact candidate identities and many more shapes
before treating it as deployment-quality training data. No model is enabled.

Reproduce collection with:

```sh
cargo run --release --features cuda --example fusion_probe -- \
  --suite all --shapes 128x192,192x192,256x256 --variants auto,greedy,tree \
  --samples 20 --block-ms 10 --reverse-order --output /tmp/fusion.jsonl
```

`integrations/rewrite_cost_model/summarize_fusion.py --help` describes comparison
and compiler-feature export. Baseline files are merged in argument order so a
corrected or repeated subset can replace earlier measurements.

## Verification

- `cargo check` and `cargo check --features cuda` passed.
- `cargo nextest run`: 281 passed.
- `TNSR_REQUIRE_CUDA=1 cargo nextest run --features cuda -j 2`: 568 passed,
  three skipped by the existing test configuration.
- `cargo fmt --all --check` and
  `cargo clippy --all-targets --all-features -- -D warnings` passed.
- Profitability-trainer Python tests: nine passed.

New coverage includes exact F32 serial reduction bits across block sizes and
tails, special values, BF16 storage boundaries, empty-mean exclusion, executor
cache invalidation, compiler feature geometry and host-layout proof properties.

## Parallel follow-up: structural rewrites, search cost and optimizer fusion

Three parallel investigations implemented the following additions. These results
are separate from the reduction/readback measurements above.

### Structural rewrites

The optimizer now visits supported expression islands beneath opaque operators,
retaining casts and unsupported operator boundaries. Identity-keyed memoization
preserves DAG sharing and avoids saturating once per interior node; runner limits
apply separately to each island. Unchanged expressions retain their identity.

Nested narrow composition, exact segment selection from concat, and exhaustive
ordered narrow/concat cancellation are also implemented. They require explicit
`RewriteNumerics::RealAlgebra`: forward storage bits are identical, but removing
zero-filled gradient scaffolding can alter signed-zero gradients or remove an
unused branch's nonfinite gradient. Default-policy regression tests retain both
behaviors. Cast rounding boundaries remain intact under either policy.

Reversed-order acceptance, sizes 256 and 384:

| Case | CPU speedup | CUDA end-to-end speedup | Policy |
|---|---:|---:|---|
| Layout round trip beneath opaque max | 2.73–2.83× | 2.57–3.91× | Default |
| Transpose/unary beneath casts | 2.56–2.76× | Essentially unchanged | Default |
| Concat/narrow reconstruction | 3.00–3.67× | Modest or neutral | RealAlgebra |
| Nested narrow | 2.03–2.12× | 1.07–1.16× | RealAlgebra |
| Select concat segment | 4.57–4.78× | Modest/noisy | RealAlgebra |

The max case drops from three GPU launches to one. Full-output readback dominates
many segment cases, and the 256 concat-selection result includes a noisy apparent
slowdown with overlapping intervals; segment rewrites are not universal CUDA
latency wins. Reshape-under-cast and unchanged controls remain mostly neutral.
[Compact intervals, policies and launch counts](benchmarks/structural-rewrites.json).

```sh
cargo run --release --features cuda --example structural_rewrite_probe -- \
  --backend cuda --sizes 256,384 --samples 24 --block-ms 10 --reverse-order \
  --output /tmp/structural.jsonl
```

The driver validates exact CPU/GPU outputs and finite weighted-loss CPU parameter
gradients before timing. GPU cast cases use MixedExecutor; other CUDA cases use
strict F32. Measurements include synchronous output readback.

### Cheaper fusion search

Serial geometry alternatives now overlay only their changed kernel features,
score the complete ordered feature vector and apply the winning schedule once.
Unchanged kernels are no longer revisited and complete plans are no longer cloned
for every block size. Partition alternatives still rebuild full legal plans.
Arbitrary non-additive scorers retain their original full-plan semantics.

Reference tests compare candidate order, labels, features, scores, tie winners,
selected plans and generated PTX with the original full-plan expansion, including
standalone reductions and an order-sensitive non-additive scorer.

An alternating-order device-free debug benchmark on 131 physical steps and
64 normalizations measured 20 selections at 634.7 ms for full-plan geometry
expansion versus 158.9 ms for overlays, about 4.00× faster. Earlier concurrent
runs varied from 2.21× to 3.96×. This is the local selection path, not an established
whole-PPO compilation/JIT speedup. Reproduce with the ignored
`serial_geometry_overlay_planning_benchmark` library test.

### Bounded resident Adam batching

`ResidentAdam::set_update_batch_size` selects batches of up to 16 small tensors
sharing an optimizer group; the default is eight, and one restores the old path.
Batches sort by size and include only tensors with
at most 4096 elements; larger tensors retain individual kernels. Generated Tile
IR uses tracked pointer parameters and one grid-Y lane per tensor, preserving
each element's original Adam arithmetic and storage rounding. A scalar health
check still completes before publishing any staged weights or moments.

Exact GPU tests compare all weight/moment bits, optimizer groups, tails, clipping,
retained old versions, batch-size changes, rejection of nonfinite staging and
successful retry. Optimized-build timings cover norm calculation, clipping,
updates, health validation and publication; input-gradient preparation and model
forward/backward execution are excluded. First-step lazy batch compilation is
also excluded; each mode has three warm steps followed by 24 measured steps.

Independent reversed-order results with 256 elements per tensor:

| Tensor count | Per-tensor updates, µs | Batched, µs | Speedup | Total step launches |
|---|---:|---:|---:|---:|
| 16 | 75.10 | 40.65 | 1.85× | 20 → 6 |
| 64 | 223.75 | 64.85 | 3.45× | 68 → 12 |
| 128 | 383.18 | 99.03 | 3.87× | 132 → 20 |

The reversed sweep over 16/256/1024 elements and 16/64/128 tensors reproduced
1.76–3.87× optimizer-step speedups. The 4096-element cutoff cases improved
1.81–3.11×; 8192-element fallback controls were essentially unchanged. These
measurements isolate optimizer cost with directly prepared gradients and two
optimizer groups; they do not establish end-to-end model-training speedups.
[Compact measurements and intervals](benchmarks/resident-adam-batching.json).

```sh
TNSR_REQUIRE_CUDA=1 TNSR_ADAM_BENCH_REVERSE=1 cargo nextest run --release \
  --features cuda --lib --run-ignored only --nocapture \
  -E 'test(resident_batched_update_timing)'
```

Set `TNSR_ADAM_BENCH_LARGE=1` to reproduce cutoff/fallback controls. These
environment switches affect only the ignored benchmark, not optimizer behavior.

### ThunderKittens scheduling experiment

Grouped GEMM traversal is implemented as a forced scheduling option, with ordinary
group 1 retained by default. A strict-F32 case improves about 4% in two resident
GPU repeats, while native BF16 cases regress about 3–9% or remain flat. An
expensive general tail mapping was rejected; the final divisor-based fallback
is neutral on the tested odd-tail case. No automatic grouping heuristic is
enabled. See [implementation, measurements and hardware limits](matmul-grid-order.md).

### Follow-up verification

- CPU suite: 294 passed, none skipped.
- Required-CUDA suite with default Adam batching: 587 passed, five intentionally
  ignored manual benchmarks. Both strict/graph PPO parity tests and the tree PPO
  parity test passed.
- CPU/CUDA checks, all-target/all-feature strict Clippy, rustfmt and diff checks
  passed. The nine Python trainer tests also passed.
- Manual Adam discovery, independent reversed acceptance and cutoff/fallback
  controls ran in release builds. Grouped-GEMM controls used resident CUDA events.
- New persisted result summaries total approximately 50 KiB, with no tensor dumps.

The local planning benchmark is also retained in
[fusion-search-overlays.json](benchmarks/fusion-search-overlays.json).
