# BF16 staging with one unaligned operand

Follow-up: [packed existing warp reductions](training-bf16-packed-reductions.md)
reports the subsequent f32 reduction scheduling experiment and its complete-update
acceptance, including rejected numerical variants.

BF16 matmuls can now overlap aligned copies of one operand with predicated scalar
staging of the other. Wider split-K tiles reuse the expensive scalar operand
across twice as many outputs. Against the accepted [transposed-staging
baseline](training-bf16-transposed-staging.md), complete mixed PPO updates improve
from **38.39 to 35.23 ms**: a **1.0925× median paired throughput ratio**, with
63/64 candidate wins. This is an additional within-run gain, not a combination
of medians from different experiments.

[Raw evidence](benchmarks/training-parity/bf16-hybrid-staging.json) retains every
request and sample, the initial diagnostic, final model traces, actual kernel
resources, compiler reports, numerical controls, and source/binary hashes. The
RTX 4080 had no external CUDA compute workload; desktop graphics clients and
unrelated host work were left running. Mixed execution remains opt-in and prior
learning-quality gates remain open. This advances #124.

## Selection and generated code

The scheduler proves eight-value aligned BF16 copy groups independently for each
operand. Each operand has an explicit async flag and its own traversal width:
one aligned operand uses 16-byte asynchronous copies while the unaligned operand
uses bounds-checked scalar loads. A unit-stride proof can preserve column-major
shared storage for a scalar transposed operand without requiring global vector
alignment. Copy groups are never issued to an unproven operand.

The generated stage issues async movement first, then scalar loads. Both operands
still have two shared buffers: the existing wait/publication/retirement barriers
protect the current compute tile and the next stage. Edge prefetches zero-fill
both paths. Layout validation requires the operand flags to agree with the stage
count. F32 and FP16 keep their existing two-operand pipeline selection, and the
compilation signature version changes with this generated code.

Two automatic decisions matter for complete updates:

- Short register-tiled projections with an async LHS and scalar RHS keep their
  existing schedule when K is below 512. Their scalar staging did not amortize
  in the model trace. Explicit double buffering remains available for experiments.
- Eligible split-K matmuls with one scalar operand expand the original 32×32×64
  tile perpendicular to that operand. Scalar LHS uses 32×64×64; scalar RHS uses
  64×32×64. Eight computing warps each own one accumulator fragment. This halves
  repeated scalar-operand loads while keeping f32 partials and the same K order.
  The smaller tile remains selected when two wider stages cannot fit the
  existing two-resident-block resource budget.

Both operands still use the existing scalar fallback if neither has proven
aligned groups. The four split-K partials, final f32 sum, single BF16 output
rounding, graph boundaries, parameter publication and resident Adam are unchanged.
No model tensor buffers, kernel launches or transfer boundaries were added.

## Complete-update acceptance

Both binaries start from the same width-256 shared-critic archive, fixed
real-game-derived 64-transition batch, microbatch 64, tree reductions, TF32x3 for
remaining f32 work, graph replay and resident Adam. Three warmups per worker are
excluded. Sixty-four retained complete requests alternate AB/BA, including feeds,
loss, backward, clipping, Adam, synchronization, ordinary metrics and JSON.

| Complete mixed PPO update | Median |
| --- | ---: |
| Accepted transposed-staging baseline | 38.39 ms |
| Hybrid staging and wider FFN gradients | 35.23 ms |

The median paired throughput ratio is **1.0925×**, with 63/64 wins. Aggregate
median latency falls 8.25%. All requested native replays stayed active with zero
fallbacks. The independent f32 control measured 42.44/42.55 ms and a 1.0007×
median paired throughput ratio, with 33/64 wins: its differences remain small
measurement variation.

A separate five-mode matrix uses the same final worker for native f32 and BF16,
identical initial weights/batch, three excluded warmups and 32 retained
reverse-order iterations. PyTorch uses the original updater and enabled BF16
reduction policy; compiled mode wraps the model with `torch.compile(dynamic=True)`.

| Complete update, matched current comparison | Median |
| --- | ---: |
| Native f32, tree/replay/TF32x3 | 45.87 ms |
| Native mixed BF16 | 38.42 ms |
| Original eager PyTorch f32 | 28.52 ms |
| Original eager PyTorch BF16 | 32.49 ms |
| Original compiled PyTorch BF16 | 27.12 ms |

Native BF16 is 16.2% shorter than native f32 in this matrix, or 1.1939× its
throughput by aggregate medians. Compiled PyTorch remains faster. Matrix request
times differ from the native-only paired run and PyTorch samples vary between
runs; all samples are retained and comparisons remain within each experiment.

## Model attribution and resource use

Two fresh Nsight traces verify identical ordered labels for 1,414 model kernels,
with 32 launches of every node. After three warmups, model span falls from
34.63 to 31.38 ms. This 3.25 ms reduction agrees closely with the 3.17 ms difference
in unprofiled complete-request medians; model spans exclude optimizer, uploads,
request handling and metrics.

| Model kernel group | Baseline median summed GPU time | Candidate |
| --- | ---: | ---: |
| BF16 GEMM | 9.50 ms | 8.72 ms |
| BF16 split-K GEMM | 7.80 ms | 5.32 ms |
| Other f32 pointwise work | 8.51 ms | 8.52 ms |
| Reductions | 5.63 ms | 5.64 ms |
| Other BF16 pointwise work | 1.54 ms | 1.54 ms |
| Cast/copy | 1.43 ms | 1.43 ms |

Across the six FFN down weight-gradient launches, the median of per-kernel
medians falls from 396.56 to 283.83 microseconds. The twelve gate/up weight-gradient
launches fall from 377.67 to 237.28 microseconds. Actual register counts are
78 → 84 and 72 → 80 respectively; static shared allocation grows from 13,824 to
36,864 bytes. The evidence retains metadata for all 182 model BF16 GEMM launches
and verifies consistency across every repetition.

The initial candidate, before wider tiles and the short-projection fallback,
had a 1.0265× paired throughput ratio and a 33.71 ms model span. Its trace exposed
regressions in short width-683 forward projections. Those diagnostics remain
separate from final acceptance and explain the conservative Auto policy.
Hardware counters remain unavailable under existing permissions; driver settings
were unchanged.

## Isolated controls and numerics

The expanded sweep covers eight synthetic shape/layout cases and 512 fixed
configurations, each with 48 rotated/reversed CUDA-event samples of eight captured
operations. Every full f32 output exactly matches its reference; independent
sampled f64 dots pass. The probes preserve dimensions and specified transpose/
split-K maps with BF16 inputs and f32 partial/output storage, but exclude model
view chains, fused epilogues and optimizer work. The final Auto policy chooses
already measured geometries; explicit double-buffer experiments remain in the
sweep, including configurations Auto rejects.

| Selected comparison | Baseline median | Candidate median | Median paired speedup |
| --- | ---: | ---: | ---: |
| FFN down weight gradient | 74.29 µs | 48.75 µs | 1.523× |
| FFN gate/up weight gradient | 73.52 µs | 48.13 µs | 1.529× |
| NN `[4096,683] @ [683,256]` | 61.31 µs | 40.58 µs | 1.511× |

The last comparison retains each path's accepted index-sharing policy. Isolated
compiler reports for the selected wider gradients use 79/75 registers with no
stack frame or spills; these are separate from actual model resource counts.

Three complete native updates compare the candidate and an independent second
baseline with a first baseline. Every named clipped gradient and all published
weights are retained:

| Step | Baseline vs baseline gradient relative L2 | Candidate vs baseline |
| --- | ---: | ---: |
| 1 | 1.474e-7 | 1.745e-7 |
| 2 | 1.273e-7 | 1.591e-7 |
| 3 | 1.287e-7 | 1.261e-7 |

The largest candidate weight discrepancy is 5.96e-8; the independent baseline
control reaches 6.87e-9. Gradient differences remain at the small native numerical
floor. No f32 tolerance or BF16 arithmetic boundary changed. These checks do not
establish original-PyTorch learning parity.

Validation passed: 206 CPU tests, 415 tests with CUDA required, both build
configurations, formatting, all-target/all-feature Clippy and four executable
documentation examples. Expanded tests exercise
left-only and right-only async copies, odd K tails, broadcast batches, changed
feeds, resource fallback, short-projection Auto fallback, and wider split-K tiles
with unaligned M/N edges and resident replay. The ignored performance sweep passed
separately for all 512 configurations.

## Reproduce and next targets

Use the accepted pre-hybrid worker as the baseline and fresh output directories:

```bash
cargo build --release --features cuda --example forty_five_worker
OPENBLAS_NUM_THREADS=1 TNSR_PROFILE_PPO=1 /path/to/forty-five/.venv/bin/python \
  integrations/forty_five/benchmark_native_pair.py /path/to/archive \
  --batch /path/to/fixed-batch.json --baseline-worker /path/to/baseline-worker \
  --worker target/release/examples/forty_five_worker --output target/hybrid-repeat \
  --microbatch 64 --warmups 3 --repeats 64 --mixed-bf16 --baseline-mixed-bf16 \
  --tree-reductions --tf32x3 --cuda-graphs

TNSR_BF16_TUNE_OUTPUT=target/hybrid-sweep-repeat \
  cargo test --release --features cuda --lib bf16_schedule_sweep \
  -- --ignored --nocapture
```

Omit both mixed flags for the f32 control. The five-mode command in
[actual mixed PPO](training-mixed-ppo.md#reproduce) includes current native f32
when the final worker is also passed as `--baseline-worker`, plus
`--baseline-tree-reductions`, `--baseline-tf32x3` and `--baseline-cuda-graphs`.

The candidate's GEMM groups total 14.04 ms, about 45% of model span. F32 pointwise
work and reductions now total 14.15 ms, a similarly large target. Remaining
ragged GEMMs still spend substantial time on scalar operand staging; reducing
that address/load cost and backward pointwise/reduction fragmentation are the
next measured opportunities.
