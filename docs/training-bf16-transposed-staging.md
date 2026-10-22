# BF16 transposed operand staging

Follow-up: [one-unaligned-operand staging](training-bf16-hybrid-staging.md) adds
another accepted improvement: 38.39 → 35.23 ms in paired full updates, a 1.0925×
median paired throughput ratio with 63/64 wins. The results below retain the
transposed-staging baseline used for that follow-up.

Aligned transposed BF16 operands now use column-major shared tiles and
16-byte asynchronous copies instead of scalar loads followed by a shared-memory
transpose. Complete mixed PPO updates improve from **43.00 to 38.61 ms** against
the [accepted register-tile baseline](training-bf16-gemm-tuning.md): a **1.1148×
median paired throughput ratio**, with 64/64 candidate wins. This is an additional
improvement over that baseline, measured in one run rather than by combining
medians from different dates.

[Raw evidence](benchmarks/training-parity/bf16-transposed-staging.json) retains
all requests, both traces, actual model resource metadata, the complete schedule
sweep, numerical controls, and source/binary hashes. All experiments used the
RTX 4080 without an external CUDA compute workload. Desktop graphics clients and
other host work were left running. This advances #124; mixed execution remains
opt-in and the earlier learning-quality gates remain open.

## Compiler change

The scheduler proves aligned groups from each operand's complete physical
access map. It reconstructs flattened batch coordinates, cancels matching
quotient/remainder terms, and accepts affine matrix strides only when the
entire logical domain preserves contiguous, aligned groups of eight BF16 values.
Bounded quotient/remainder analysis handles interleaved split-K layouts. Checked
integer arithmetic rejects unrepresentable proofs; nonlinear wrapping maps,
predicated views, partial groups and misaligned strides retain the existing
staging path. The proof does not infer global contiguity from adjacent samples.

Eligible operands keep their physical minor axis in shared memory. Shared
addressing, allocation sizes, WMMA operand loads and WMMA compute instructions
all carry the same row/column orientation. Shared-memory sizing now takes both
logical dimensions so rectangular column-major tiles are accounted for correctly.
Two stages overlap aligned copies with computation, with the existing waits,
publication/retirement barriers and zero-filled edge prefetches. Automatic
selection still requires target support, enough K tiles, multiple computing
warps, and the existing resource budget for at least two resident blocks.

Both operands must prove valid groups. The common width-683 FFN weight-gradient
families remain on the fallback because at least one physical copy axis or stride
is unaligned. F32 and FP16 retain their prior schedule policies. Split-K still
uses four f32 partials, a final f32 sum, and one BF16 output rounding. No additional
tensor materialization, launches, transfers or optimizer boundaries were added.
The PTX compilation signature version changes with this generated code.

Physical batch axes matter: an initial prototype proved the flattened probe but
rejected equivalent model buffers. Its complete-update throughput ratio was
0.9985×. That diagnostic and its trace remain in the evidence; acceptance uses
the final proof and final worker exclusively.

## Complete updates

The paired run uses the original width-256 shared critic, the same archive and
fixed real-game-derived 64-transition batch, microbatch 64, resident Adam, tree
reductions, TF32x3 for remaining f32 work, and CUDA graph replay. Three warmups
per worker are excluded; 64 retained requests alternate AB/BA. Timers include
feeds, loss, backward, clipping, Adam, synchronization, ordinary metrics and
JSON worker requests.

| Complete mixed PPO update | Median |
| --- | ---: |
| Accepted register-tile baseline | 43.00 ms |
| Transposed async staging | 38.61 ms |

The 1.1148× median paired throughput ratio corresponds to 64/64 wins; aggregate
median latency falls 10.19%. All requested replays remained active with zero
fallbacks. An independent f32 control measured 41.86/41.83 ms, a 1.0020× median
paired throughput ratio and 35/64 wins: no f32 regression was observed.

The subsequent five-mode comparison uses the **same current executable** for
native f32 and mixed BF16, identical initial weights/batch, three excluded
warmups and 32 retained reverse-order iterations. Original PyTorch keeps its
existing updater and enabled BF16 reduction policy; compiled mode wraps its
model with `torch.compile(dynamic=True)`.

| Complete update, matched current comparison | Median |
| --- | ---: |
| Current native f32, tree/replay/TF32x3 | 42.14 ms |
| Current native mixed BF16 | 38.33 ms |
| Original eager PyTorch f32 | 32.65 ms |
| Original eager PyTorch BF16 | 42.21 ms |
| Original compiled PyTorch BF16 | 31.10 ms |

Native BF16 is now 9.0% shorter than native f32, or 1.0993× its throughput by
aggregate medians. It remains 1.23× the compiled PyTorch BF16 latency in this run.
PyTorch request times vary considerably; all samples are retained and these
within-run comparisons should not be mixed with medians from previous dates.

## Actual model trace and resources

Two clean Nsight traces verify the same ordered 1,414 model kernels, each launched
32 times. After excluding three warmups, the captured model span falls from
39.07 to 34.63 ms. The 4.44 ms reduction is consistent with the 4.38 ms difference
in unprofiled complete-request medians; these spans exclude optimizer, uploads,
request handling and metrics.

| Model kernel group | Baseline median summed GPU time | Candidate |
| --- | ---: | ---: |
| BF16 GEMM | 11.81 ms | 9.49 ms |
| BF16 split-K GEMM | 9.92 ms | 7.80 ms |
| Other f32 pointwise work | 8.52 ms | 8.51 ms |
| Reductions | 5.64 ms | 5.63 ms |
| Other BF16 pointwise work | 1.54 ms | 1.54 ms |
| Cast/copy | 1.43 ms | 1.43 ms |

GEMM accounts for essentially all of the measured savings. Each of the six
actual QKV weight-gradient partial kernels drops from approximately 407 to
70 microseconds. Their actual register count falls from 72 to 61 per thread,
while static shared memory grows from 13,824 to 24,576 bytes. The evidence
retains resource metadata for all 182 model BF16 GEMM launches and verifies it
is consistent across every repetition. Hardware counters remain unavailable
under the existing permissions; driver settings were unchanged.

## Isolated schedule controls

The six-case sweep retains 384 configurations with 48 samples each. Every full
f32 output exactly matches its synchronous reference, and sampled independent
f64 dot products pass. Buffers are preallocated, eight operations are captured,
and configuration order rotates/reverses in one process. These synthetic probes
preserve dimensions and split-K interleaving, with f32 partial/output storage;
they exclude model epilogues and optimizer work.

| Same block tile | Synchronous median | Async median | Median paired speedup |
| --- | ---: | ---: | ---: |
| QKV weight gradient, 32×32×64 | 79.49 µs | 35.84 µs | 2.214× |
| QKV forward, 64×64×32 | 56.06 µs | 35.97 µs | 1.560× |

The forward comparison retains each path's existing index-sharing policy:
normalized local expressions for synchronous register tiles, shared expressions
for pipelined tiles. Compiler reports for the weight-gradient probe show
64/52 registers per thread for synchronous/async, with no stack frame or spills
in either. These isolated resource counts are separate from the model counts.
Wider tiles remain experimental; this change keeps the accepted block geometry.

## Numerical checks and validation

Three complete updates compare a candidate and a second baseline against an
independent first baseline. Every named clipped gradient and all published
weights are retained:

| Step | Baseline vs baseline gradient relative L2 | Candidate vs baseline |
| --- | ---: | ---: |
| 1 | 1.530e-7 | 1.719e-7 |
| 2 | 1.268e-7 | 1.326e-7 |
| 3 | 2.32089e-4 | 2.32087e-4 |

Both third-step comparisons have the same maximum weight discrepancy, 9.9773e-6.
The candidate remains within the observed native baseline variability. These
controls preserve the earlier BF16 arithmetic contract; they do not establish
original-PyTorch learning parity.

Validation passed: 205 CPU tests, 412 tests with CUDA required, both build
configurations, formatting, all-target/all-feature Clippy and four executable
documentation examples. The ignored performance sweep passed separately. New tests cover physical
batch/split-K proof recognition; rejection of wraps and unaligned strides;
exhaustive physical-index checks through reshape/permutation combinations;
rectangular column-major allocation; all transpose combinations with M/N/K
edges, broadcast batches and changing feeds; and resident replay with changing
inputs. The ignored performance sweep runs separately from ordinary tests.

## Reproduce and next targets

Use the accepted pre-staging worker as the baseline and a fresh output directory:

```bash
cargo build --release --features cuda --example forty_five_worker
OPENBLAS_NUM_THREADS=1 TNSR_PROFILE_PPO=1 /path/to/forty-five/.venv/bin/python \
  integrations/forty_five/benchmark_native_pair.py /path/to/archive \
  --batch /path/to/fixed-batch.json --baseline-worker /path/to/baseline-worker \
  --worker target/release/examples/forty_five_worker --output target/staging-repeat \
  --microbatch 64 --warmups 3 --repeats 64 --mixed-bf16 --baseline-mixed-bf16 \
  --tree-reductions --tf32x3 --cuda-graphs
```

Omit both mixed flags for the f32 control. The five-mode command in
[actual mixed PPO](training-mixed-ppo.md#reproduce) accepts the current worker
also as `--baseline-worker`, with `--baseline-tree-reductions`,
`--baseline-tf32x3`, and `--baseline-cuda-graphs`, to include current native f32.
The ignored `bf16_schedule_sweep` accepts a fresh `TNSR_BF16_TUNE_OUTPUT` directory
and optional `TNSR_BF16_TUNE_CASE=qkv-wgrad` filter.

GEMMs still account for about half of the current captured model span. The [hybrid staging follow-up](training-bf16-hybrid-staging.md) now implements
and accepts overlap with one unaligned operand for the width-683 FFN gradients. The current f32 pointwise/reduction groups account
for another 14.14 ms. Other wider synchronous FFN tiles and fully vectorized split-K tiles still need
controlled complete-update acceptance before selection.
