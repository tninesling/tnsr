# Additional rewrite search and learned profitability experiment

October 29, 2026. This continues the [first rewrite round](egraph-transformations.md).
The goal was at least five additional profitable options. Five concrete rewrites
pass independent acceptance checks on both CPU and RTX 4080 at two sizes; several
other families also save substantial CPU work.

## Accepted options

The table compares the **previous round's optimizer** with the new optimizer in
an independent run with reversed candidate order. These are means at scale 256,
in microseconds. Both scale 256 and 384 require at least 10% speedup and separated
approximate confidence intervals, and must also win in the discovery sweep.

| Rewrite | Numerical policy | CPU before → after | CPU speedup | GPU before → after | GPU speedup |
|---|---|---:|---:|---:|---:|
| `transpose(relu(transpose(x))) → relu(x)` | Conservative | 1,158.25 → 48.27 | 23.99x | 389.15 → 54.76 | 7.11x |
| `reshape(relu(reshape(x))) → relu(x)` when shapes round-trip | Conservative | 91.00 → 50.08 | 1.82x | 5,847.27 → 54.15 | 107.98x |
| `exp(broadcast(x)) → broadcast(exp(x))` | Conservative | 760.87 → 532.16 | 1.43x | 55.27 → 15.47 | 3.57x |
| `mean(broadcast(x), axis) → x` | RealAlgebra | 1,092.93 → 0.38 | 2911.07x | 21.54 → 6.94 | 3.10x |
| `sum(A @ B, columns) → A @ sum(B, columns)` | RealAlgebra | 1,160.56 → 206.90 | 5.61x | 40.08 → 21.37 | 1.88x |

The first two are different concrete cases of a broader pointwise/layout
identity. All five are tested as separate programs. The very large layout and
broadcast-mean wins remove pathological intermediate work; they are not expected
model-level speedups. Sum/mean pushdown supports both noncontracting dimensions
of positive rank-2 matmuls, with the axis/shape checks kept in the e-graph.

Additional implemented families:

- Mixed transpose/permutation composition and cancellation.
- Permutations moving only singleton axes become metadata reshapes or disappear.
- Unsupported operations become identity-preserving opaque leaves. Supported
  parents can simplify without crossing cast, exact-index, convolution, pooling,
  flatten, narrow, concat, or wide-metadata boundaries. Root unsupported nodes
  retain their original structure. This enables optimization around mixed-dtype
  operations while preserving all rounding boundaries and source identities.
- Axis-invariant factors can move outside sum/mean under `RealAlgebra`.

The first three additional families have clear CPU benefits and mostly neutral
CUDA results in these cases. An opaque cast boundary is never algebraically
removed. The general pointwise rules include neg/exp/log/relu; inverse axis maps
include arbitrary permutations and the transpose API representation.

## Confirmed exception

Invariant-factor extraction is profitable at the larger acceptance sizes:
about 2.2x CPU and 1.18x CUDA at scale 256. However, the discovery sweep also found
a confirmed CUDA regression at scale 192: 15.09 us → 19.06 us, **26% slower**.
The manual and extracted alternatives behave alike. It is retained only under
the existing explicit `RealAlgebra` policy, and is not one of the five primary
acceptance options above.

It removes a launch and intermediate storage in that case, but the resulting
reduction kernel still runs slower. A working-set or launch-geometry effect is
plausible; the current counters do not establish the cause. There is no ad hoc
`n == 192` exclusion. The rule is not a universal latency improvement. A future
backend-aware selector needs schedule features such as serial working set,
active threads, block size, registers and reduction layout. The trained models
below do not yet solve this exception.

The previously rejected transpose-after-pointwise alternative remains excluded
from the optimizer. It stays in the dataset as a real negative example, alongside
skinny matmul factoring and unchanged controls.

## Research and implementation

[OpenXLA's current algebraic simplifier](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8357)
compares old/new dot and reduction work before moving reductions through dots;
it also handles reductions through transposes. Our contraction pushdown follows
the same algebraic opportunity, using the existing shape-aware e-graph to retain
and price alternatives. The conservative layout rules prove inverse maps or
unchanged flattened element order, rather than assuming equal shapes imply an
identity map.

Implementation is split between `src/graph/rewrite/layout_rules.rs`,
`reduction_rules.rs`, and the existing `tensor_analysis.rs`. The parent optimizer
now serializes unsupported subtrees as opaque leaves instead of abandoning the
whole supported parent expression. Cached rule tables and runner limits remain
in use. Arithmetic-changing rules still require `RealAlgebra`; the default
preserves operation order, domains and cast boundaries.

## Measurement and provenance

Hardware: AMD Ryzen 7 3700X and NVIDIA RTX 4080, driver 580.196.07.
Discovery: 15 candidate cases × 12 scales × 3 variants = **540 measured rows per
backend**. Scale groups are 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384.
Each case measures original, explicit manual alternative, and optimizer output.
All variants are checked against an independent CPU execution before timing.

Candidate order rotates across family/scale groups. Discovery uses 16 samples,
each calibrated to approximately 5 ms; acceptance and previous-optimizer runs use
20 samples at approximately 10 ms. The reported intervals are block-mean
intervals using 2.2 standard errors, not a claim that systematic clock drift is
modeled. Runs use warmup, precompiled graphs, synchronous host output readback,
and cached parameter uploads. Clocks are not locked. Large wins must reproduce
with reversed order at both acceptance sizes.

Pure f32 CUDA cases use `PtxExecutor` and `StrictF32`. Cast-boundary cases use
`MixedExecutor` to execute real BF16/F32 casts. That executor difference is
recorded in a feature; cast cases do not silently replace their computations.
These are diagnostic graphs with small, cache-resident deterministic parameters,
not Transformer/training throughput benchmarks or kernel-only timings.

The baseline is a saved copy of the previous round's **working tree**, because
those changes had not been committed. Its source/harness archive is
`docs/benchmarks/rewrite-search-previous-source.tar.gz`; the summary records its
SHA-256. It used a separate Cargo target directory so switching source trees
could not reuse a stale same-name executable.

Artifacts under `docs/benchmarks/`:

- `rewrite-search-{cpu,cuda}.jsonl`: complete discovery samples and features.
- `rewrite-search-previous-{cpu,cuda}.jsonl`: previous optimizer, large cases.
- `rewrite-search-accept-{cpu,cuda}.jsonl`: independent reversed-order runs.
- `rewrite-search-summary.json`: acceptance checks, exceptions and data hashes.

Reproduce the search and its acceptance summary:

```sh
cargo run --release --features cuda --example rewrite_search -- \
  --backend cpu --output /tmp/rewrite-cpu.jsonl \
  --sizes 8,12,16,24,32,48,64,96,128,192,256,384
cargo run --release --features cuda --example rewrite_search -- \
  --backend cuda --output /tmp/rewrite-cuda.jsonl \
  --sizes 8,12,16,24,32,48,64,96,128,192,256,384
cargo run --release --features cuda --example rewrite_search -- \
  --backend cuda --output /tmp/rewrite-cuda-reverse.jsonl \
  --sizes 256,384 --samples 20 --block-ms 10 --reverse-order
python3 integrations/rewrite_search/summarize.py
```

The summary command audits the checked-in filenames, rather than the `/tmp`
examples. GPU commands require device access outside the desktop sandbox.

## Learned profitability

We trained CPU static, CUDA static, and CUDA plan-feature log-linear ridge models.
After merging duplicate graph aliases by averaging their timings, each backend
has 372 distinct measured candidate graphs across 180 cases. Identifiers and
family names are excluded from features. Validation holds out each entire scale
across every family, and then each entire rewrite family; all variants stay
together, and fold normalization is fitted only on training data.

None of the models qualifies for deployment. CPU learned regret is approximately
1.09 versus 1.04 for the optimizer. CUDA plan features improve prediction, but
family-held-out regret is about 1.07 versus 1.01 for the optimizer, with much worse
individual misses. Lower regret is better; 1.0 selects the measured best graph.
The model artifacts remain disabled and no Rust runtime model is loaded.

The [learned-cost guide](learned-rewrite-cost.md) contains primary-source research,
all fold results, portable JSON exports, the distinction between manual-candidate
reranking and the actual optimizer, and the static/plan-feature ablation.
The dataset, trainer and gates make a larger follow-on model experiment
reproducible; the current measurements do not justify replacing the heuristic.

## Validation and workspace cleanup

- CPU suite: 279 passed, none skipped.
- CUDA suite: 541 passed; three existing manual GPU timing experiments ignored.
- Python trainer protocol: eight tests passed; trainer and summary scripts compile.
- CPU/CUDA builds, rustfmt, strict all-target/all-feature Clippy and diff checks passed.
- The six model/report fingerprints match their complete datasets; all deployment
  eligibility gates remain disabled.
- Reproducible incremental build caches were cleared to recover roughly 2 GiB.
  Benchmark results and archived source snapshots were preserved.
