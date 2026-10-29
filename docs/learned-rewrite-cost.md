# Learning rewrite profitability

Research and prototype: October 29, 2026.

## Recommendation

Collect a hardware-specific candidate dataset and train a small reranker first.
Keep the current analytical extractor as the baseline and keep the original graph
as a candidate. A learned model should remain opt-in until it wins on held-out
shapes and held-out rewrite families. Eight diagnostic cases, including layout
round trips with extreme wins, cannot establish a general model.

The prototype at `integrations/rewrite_cost_model/train.py` fits ridge regression
from log-transformed graph features to log measured latency. This is deliberately
small: the exported model is numerical JSON, inference is a few logarithms and a
dot product, and no ML dependency is required in Rust. Python training uses NumPy.
It does not enable a model or change optimizer behavior automatically.

## What other systems do

[Ansor (OSDI 2020)](https://www.usenix.org/conference/osdi20/presentation/zheng)
combines a hierarchical tensor-program search space, evolutionary refinement, a
learned cost model and hardware measurements. Its lesson for this work is to
predict relative candidate quality while retaining measurement as the authority.

[TVM MetaSchedule's current documentation](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)
describes learned XGBoost guidance, real builder/runner measurements and a
persistent tuning database. Its
[cost-model API](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html)
uses per-store features and exposes warmup samples and adaptive retraining. We
should similarly persist measured candidates rather than train on a few selected
successes or on static FLOP estimates disguised as labels.

[TenSet (NeurIPS Datasets and Benchmarks 2021)](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/a684eceee76fc522773286a895bc8436-Abstract-round1.html)
provides millions of tensor-program measurements across multiple hardware
platforms. Its [official experiments](https://github.com/tlc-pack/tenset/blob/main/docs/get_started_with_cost_model_experiments.md)
include holding out an entire network's tasks and evaluate model use during
search. This supports separating workload groups during validation instead of
randomly separating timing samples of the same graph.

The [TCL preprint (April 2026)](https://arxiv.org/abs/2604.12891) studies active
sample selection and cross-hardware continual learning. That is a plausible later
direction when measurement coverage is substantial; it does not establish that
CPU labels can predict this RTX 4080 backend without target-specific evaluation.

## Data protocol

JSONL contains one independently measured candidate per row, with all variants
of one semantic case sharing `case_id`, `shape_key`, and `family`:

```json
{"schema_version":1,"backend":"cuda","device":"RTX4080","precision":"strict_f32","family":"reduce_matmul_rows","shape_key":"256x128x256/axis0","case_id":"reduce_matmul_rows/256x128x256","variant":"original","mean_ns":42000,"ci95_ns":[41000,43000],"heuristic_cost":8388608,"features":{"flops":8388608,"read_bytes":524288,"write_bytes":262144,"launch_count":2,"matmul_count":1,"reduction_count":1,"layout_count":0,"output_elements":256,"max_rank":2,"stride_penalty":0,"fused_region_count":2}}
```

Every case needs `original` and one or more alternatives. An optional `graph_key`
identifies structurally identical candidate DAGs within a case; it never enters
the feature matrix. Duplicate graph measurements are merged by averaging their
means, preserving the union of their confidence intervals and all variant aliases.
We do not choose the fastest noisy alias. Conflicting numerical features or costs
for the same key are rejected. Cases with only one distinct graph are excluded
from model evaluation so they cannot inflate apparent ranking accuracy.

The loader rejects
mixed hardware/backend/precision identities, duplicate variants, invalid numeric
features, and cases without an original. Train each hardware/precision policy
separately. Both profitable and harmful rewrites belong in the dataset.

Preferred numerical features are unique-DAG operation counts, FLOPs, element
traffic, materialized-layout volume, output/readback bytes, rank, reduction
extents, contraction dimensions, alignment/tail information, and actual fused
region/kernel counts from the execution plan. A root transpose after a pointwise
region can have a radically different schedule despite fewer graph nodes; basic
operation totals alone may not distinguish that failure. Estimated launch counts
must be identified as estimates and should eventually be replaced with actual
plan metrics. Identifiers, measured timing, family names and case IDs are never
input features. Feature names are derived from the dataset and exported in a
stable sorted order; omitted numerical features mean zero.

Repeated wall timings should be aggregated before writing a row. Preserve
confidence intervals and use the same compilation, synchronization, readback,
precision and power/clock protocol for all variants. Criterion samples from one
candidate must not become independent training and validation rows. Keep a fresh
session or a final locked test corpus for the eventual deployment decision.

## Training and validation

```sh
python3 integrations/rewrite_cost_model/train.py path/to/cpu-candidates.jsonl \
  --output docs/benchmarks/rewrite-model-cpu.json \
  --report docs/benchmarks/rewrite-model-cpu-report.json --feature-mode static
# Repeat the CUDA dataset using --feature-mode static and --feature-mode all.
python3 -m unittest discover -s integrations/rewrite_cost_model -p 'test_*.py'
```

The fixed default ridge penalty is 1.0. No test-set tuning chooses it. All feature
centering/scaling is fitted only on a fold's training rows. Validation performs
both leave-one-shape-key-out and leave-one-family-out folds, keeping candidate
variants together. The report records log-latency error, ranking accuracy,
choice regret relative to the measured oracle, speedup relative to the original,
and choice regret for both the supplied analytical cost reranker and the actual
`optimized` variant. These are different baselines: analytical reranking can see
manually supplied candidates that the optimizer cannot currently generate.
Raw choice regret is accompanied by CI-aware regret that treats overlapping
intervals or differences below 5% as indistinguishable. Eligibility still uses
the more conservative raw regret. Pairwise accuracy
excludes differences below 5% and overlapping confidence intervals.

The model is marked ineligible even for opt-in unless there are at least twelve
shape groups and five families, it beats the supplied heuristic's geometric mean
regret and the actual optimizer output (when supplied) under both splits, and its
worst-case held-out regret is at most 10%.
These counts are conservative initial policy choices, not a statistical guarantee.
The exported reference chooser additionally falls back to the original on a
hardware/precision mismatch, a feature outside the training range, or predicted
improvement below the larger of 5% and twice the held-out 90th-percentile absolute
log error. This margin is exploratory calibration. Reported cross-validation
metrics concern the unguarded ranker; they are not an independent validation of
the final calibrated gate. The prototype remains experimental even if eligible.

The eight unit tests use a synthetic relation with a known ranking. They verify
cross-shape/family evaluation, complete variant grouping, JSON inference
roundtripping, out-of-coverage and hardware fallbacks, and input rejection.
Those tests establish software behavior, not real performance accuracy.

`--feature-mode static` excludes the current post-execution counters and generated
kernel/fused-region counts. `--feature-mode all` includes them and marks the model
as requiring executed-candidate features. These models must be reported as a
separate ablation: today's runtime cannot collect execution counters without
running a candidate, so an all-feature model is not evidence of free online
profitability prediction. The static feature model is the directly portable
baseline; compile-only plan features could later be made available before timing.

## Measured results

The final CPU dataset contains 540 measured rows: fifteen families, twelve
shared scale groups, and three variants per case. Deduplication leaves 372 graph
rows across 180 cases (168 duplicate groups merged). The CPU sweep uses balanced
variant ordering; independent reversed measurements validate final rewrite
acceptance separately. `shape_key` is a shared scale such as `n128`, so every
family at that scale is held out together.

The static ridge model has scale-held-out geometric regret 1.095 and family-held-
out regret 1.091, compared with 1.0048 for analytical candidate reranking and
1.0439 for the actual optimizer output. Its worst held-out choice is 2.13x slower
than the measured oracle, and pairwise accuracy after excluding near ties is
81.25%. It therefore remains **ineligible**. The trained model and complete fold
report are `docs/benchmarks/rewrite-cost-cpu-model.json` and
`docs/benchmarks/rewrite-cost-cpu-report.json`. This is an actual negative result,
not an in-sample success presented as validation. The current analytical policy
is a substantially stronger selector on these CPU cases. Final artifacts were
retrained from the completed balanced-order sweep.

The completed CUDA sweep has the same 540-to-372 deduplication and 180 cases.
Pure f32 cases use compiled `PtxExecutor` with `StrictF32`; the opaque-cast family
uses `MixedExecutor` to preserve real BF16/F32 boundaries and supplies a
`mixed_executor` feature. All timing includes steady-state host readback.
Both ablations retain strict eligibility gates:

| CUDA model | Held-out scale regret | Held-out family regret | Worst scale / family regret |
|---|---:|---:|---:|
| Static graph features | 1.044 | 1.096 | 3.68 / 4.96 |
| Static plus plan/execution features | 1.036 | 1.071 | 3.68 / 4.96 |
| Analytical reranking of all candidates | 1.082 | 1.082 | — |
| Actual optimizer output | 1.011 | 1.011 | — |

Lower regret is better; 1.0 is the measured oracle. Plan/execution features
improve the ranker, including family-held-out pairwise accuracy from 57.8% to
73.3%, but the learned choices still lose to the actual optimizer. Large misses
include leaving invariant reductions or unary-broadcast work in the original
form, missing profitable manual/optimized alternatives. Both trained CUDA models
remain **ineligible**. The artifacts are `rewrite-cost-cuda-{model,report}.json`
and `rewrite-cost-cuda-plan-{model,report}.json` under `docs/benchmarks/`.
The report also gives CI-aware regret; raw regret shown here keeps timing
uncertainty visible rather than treating a near tie as a proven speedup.

Model and report artifacts include a SHA-256 fingerprint of their full input
rows, so replacing measurements requires explicit retraining and can be audited.
This fingerprint is metadata and never a feature.

## Runtime integration

The model predicts whole-graph latency and is **not additive** across e-nodes.
Do not directly substitute its predictions into egg's per-node `CostFunction`:
that would double-count sharing and fail to represent fusion and layout effects.
The first integration should rerank a small explicit candidate set: original,
analytical best, and bounded alternatives produced by the new rewrite families.
Compute plan/DAG features, score only matching backend/precision models, apply
coverage and improvement gates, and retain the original otherwise. Eligibility
never overrides numerical policy; arithmetic-changing rules still require
`RealAlgebra`.

A later top-k or beam extractor can expose more candidates. DAG-aware extraction
and scheduling features should precede larger neural models. A tiny gradient-
boosted tree model is a reasonable next model class after enough real labels
exist; the current ridge prototype provides an interpretable, cheap baseline.
