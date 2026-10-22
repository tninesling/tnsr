# Actual forty-five mixed PPO updates

Latest: [BF16 PPO within 20% of PyTorch](training-bf16-within20.md) records fresh
five-mode comparisons after pointwise launch and aligned split-K operand tuning.

Earlier [packed warp reductions](training-bf16-packed-reductions.md) measured
34.77 → 34.51 ms against the accepted hybrid GEMM worker, a 1.0087×
median paired throughput ratio with 42/64 wins. Earlier [hybrid operand
staging](training-bf16-hybrid-staging.md), [transposed staging](training-bf16-transposed-staging.md)
and [register tiling](training-bf16-gemm-tuning.md) retain their separate
experiments and numerical/learning limits. The original five-mode matrix below
is historical; the latest comparisons appear in the follow-up above.

The experimental `--mixed-bf16` path now executes the actual forty-five shared
and separate critics through graph autograd, exact indices and resident Adam.
F32 remains the default. The original pre-tuning measurements below were slower
end to end; later tuning results appear in the follow-ups above. This integration
does not establish training or learning parity. It advances [#114](https://github.com/tninesling/tnsr/issues/114),
[#115](https://github.com/tninesling/tnsr/issues/115) and
[#94](https://github.com/tninesling/tnsr/issues/94).

[Raw performance/profile evidence](benchmarks/training-parity/mixed-ppo.json)
and [every named calibration error](benchmarks/training-parity/mixed-ppo-calibration.json)
retain samples, source/worker hashes, optimizer checks and measurement limits.
The reference archive is the actual width-256, six-layer, eight-head model with
64 tokens, vocabulary 579 and card-conditioned phase heads. The separate critic
uses the original width-256 checkpoint audited in the BF16 contract. These are
full model updates, rather than standalone projection probes.

## Execution and boundaries

`TensorExpr::linear_bf16` preserves input/weight/bias BF16 casts, f32 GEMM
accumulation and bias addition, and one final BF16 store. Embeddings and indices
remain f32/u32 respectively. The adapter applies the recorded selective policy:
f32 residual promotion, LayerNorm, softmax and forward score sums; BF16 linear,
attention and activation boundaries; f32 policy logits and BF16 values feeding
PPO. Bidding anchors use original CUDA autocast with the same per-microbatch
bidding-row selection. Discard membership and count normalization retain their
BF16 rounding before f32 concatenation.

The resident owner accumulates in f32, clips, checks finiteness and updates
f32 masters/moments with grouped Adam. Published weights refresh stable mixed
bindings on-device. BF16 values can promote through typed loads into f32
registers, including gradient accumulation. Immutable float constants upload
once, exact index feeds validate before uploads, and scatter outputs reset on
every replay. Metrics readback kernels are cached.

A correctness fix was required for contiguous backward reshapes: the homogeneous
executor implements those plan steps as device copies, and their diagnostic
kernels are empty. Fixed bindings now emit real generated copies. Early timings
from the incomplete backward path are excluded from accepted performance claims.

Long BF16 gradient contractions use four f32 partials and a final f32 sum/BF16
store. The halfway-value regression distinguishes this from rounded BF16
partials. The final paired ablation found unsplit execution about 2.2% slower.
All kernels lower through Tile IR and the PTX generator.

## Complete-update performance

The final comparison uses one real-game-derived 64-transition batch, microbatch
64, identical initial weights, three excluded warmups and 24 warm updates per
mode. Execution order reverses each iteration. Timers include feeds, loss,
backward, clipping, Adam, synchronization and normal metrics. Native timing also
includes the JSON worker request. PyTorch uses the original updater; its BF16
reduction setting remains enabled. Compiled mode uses `torch.compile` with
`dynamic=True` for the model, retaining the original PPO orchestration.

No external compute process was observed before or after this final run. Earlier
runs alongside external training/evaluation are separate diagnostics. The user’s
jobs were left untouched and finished naturally.

| Complete update | Median |
| --- | ---: |
| Production native f32, tree/replay/TF32x3 | 44.38 ms |
| Experimental native mixed BF16 | 49.56 ms |
| Original eager PyTorch f32 | 34.46 ms |
| Original eager PyTorch BF16 | 43.70 ms |
| Original compiled PyTorch BF16 | 28.78 ms |

The mixed median is 11.7% longer than production f32 and 1.72 times the compiled
BF16 PyTorch median. F32 remains the preferred path for this workload.

Profiling did produce a measured improvement within the mixed path. BF16 operand
staging was promoting to f32 and narrowing again before shared-memory stores.
The generator now copies the BF16 bits directly. In 64 alternating paired full
updates, this reduced the median from 51.06 to 49.53 ms, with a 1.0312 median
paired throughput ratio and 62/64 wins. Padded shared layouts alone produced a
1.0024 ratio and were reverted. Those experiments use the same batch/initial
weights within each pair; their results are separate from the five-mode table.

## Profile, transfers and memory

Two same-batch Nsight traces verify 32 launches of each of the 1,414 ordered model
kernels. Captured node identities/order are checked before attribution; three
updates are excluded from summaries. The final model replay span is 46.09 ms,
versus 47.18 ms for its staging control. These profiler spans exclude resident
optimizer and upload work and remain separate from unprofiled request timings.

| Final model kernel group | Median summed GPU time |
| --- | ---: |
| BF16 GEMM | 14.70 ms |
| BF16 split-K GEMM | 13.17 ms |
| Other f32 pointwise work | 8.88 ms |
| Reductions | 5.72 ms |
| Other BF16 pointwise work | 1.53 ms |
| Cast/copy | 1.44 ms |

GEMMs account for about 60% of model GPU time. The largest remaining individual
kernels are weight gradients such as `[256,4096] @ [4096,768]` and
`[683,4096] @ [4096,256]`. Larger BF16 tiles and overlapping operand movement
are the next experiments for [#124](https://github.com/tninesling/tnsr/issues/124).
F32 backward pointwise/reduction regions are the next substantial group.

The trace verifies 131,300 bytes of H2D traffic per ordinary warm update,
including 80 bytes of Adam settings; 140 device copies move 24,079,680 master
bytes. Three readbacks total 40 bytes for metrics/health. Ordinary training
downloads no gradients, weights or moments. The calibration diagnostic explicitly
downloads those arrays outside timing.

Lifetime reuse replaces 1,078 temporary allocations. Final retained graph
payload is 1.550 GB, including 0.336 GB of physical BF16 storage, versus 4.390 GB
before reuse. Exact index storage adds 33,664 bytes. This payload excludes
optimizer, module, allocator and readback overhead. Whole worker memory measures
3,518 MiB for mixed and 2,110 MiB for production f32. Mixed has no demonstrated
memory-capacity advantage. Reuse happens after compilation allocations; two mixed
workers ran out of memory alongside the earlier 7,352 MiB external training job.
The evidence retains the balanced serial ablation used under that constraint.

## Numerical calibration and real-loop smoke

The original CUDA `ppo_update` is called unmodified for calibration. Every named
clipped gradient, moment and updated weight is compared across independent
trajectories, with nonzero anchor KL and saved-weight reloads. Native Adam also
has an independent f32 recurrence check driven by its own clipped gradients.
Its largest weight discrepancy is 1.20e-7; timesteps match the requested updates.

| Actual model/shape | Aggregate clipped-gradient relative L2 error |
| --- | --- |
| Shared critic, 8 transitions, microbatches 3/3/2 | 0.377%, 1.516%, 1.203% |
| Separate critic, 8 transitions, microbatches 3/3/2 | 0.552%, 0.790%, 3.992% |
| Shared critic, 64 transitions, one full microbatch | 0.522% |

The full-shape calibration repeats the eight real observations eight times to
exercise long weight-gradient contractions. Some small gradients change sign,
and the three-step maximum weight differences reach 7.29e-4 shared and 9.67e-4
separate. All per-parameter errors remain in the evidence. No existing f32
tolerance was relaxed, and these results do not define accepted BF16 learning
parity. Multi-seed learning and balanced evaluation gates remain open.

A real collector smoke produced 62 transitions and two mixed PPO optimizer
updates with nonzero anchor KL, two captured shapes and zero fallbacks. Scheduler,
archive reload into a fresh native worker, restored PyTorch inference and two
complete evaluation games passed. Restored logits differ by at most 1.15e-5.
The smoke’s 11.95-second PPO wall time includes first-use compilation of unseen
shapes; it is not a warm throughput sample or learning-quality evidence.

Validation includes 199 CPU tests, the foundation’s 412-test CUDA-required full
suite, 33 final focused mixed/dtype/layout tests after the staging change, four
doc tests, both build configurations, all-target/all-feature Clippy, formatting
and Python compilation. Resident tests cover grouped rates, microbatch
accumulation, host mutations, nonfinite rejection and recovery. Preparation and
destruction serialize allocations, uploads and frees with captures on the shared
resident stream; concurrent-capture regressions cover both lifecycle boundaries.
The final lifecycle fix leaves generated update kernels and arithmetic unchanged.

## Reproduce

Use a fresh output directory for every command. First export the desired original
checkpoint with `integrations/forty_five/export.py`, then run:

```bash
cargo build --release --features cuda --example forty_five_worker
OPENBLAS_NUM_THREADS=1 /path/to/forty-five/.venv/bin/python \
  integrations/forty_five/validate_mixed_ppo.py /path/to/archive \
  --forty-five-root /path/to/forty-five \
  --worker target/release/examples/forty_five_worker \
  --output target/mixed-calibration --steps 3

TNSR_PROFILE_PPO=1 /path/to/forty-five/.venv/bin/python \
  integrations/forty_five/benchmark_training.py /path/to/archive \
  --forty-five-root /path/to/forty-five --output target/mixed-benchmark \
  --transitions 64 --microbatch 64 --warmups 3 --repeats 24 \
  --mixed-bf16 --tree-reductions --tf32x3 --cuda-graphs --compile
```

For model-loop execution, add `--mixed-bf16 --tree-reductions --tf32x3` to
`integrations/forty_five/train.py`; replay remains optional. For trace attribution,
set `TNSR_PROFILE_PPO_DUMP` to a fresh directory, capture with Nsight
`--cuda-graph-trace=node`, export SQLite, and run `summarize_mixed_trace.py` with
the saved `mixed-micro-64.json` labels. The script rejects count/order mismatches.
