# Packed warp reductions in BF16 PPO

Follow-up: [BF16 PPO within 20% of PyTorch](training-bf16-within20.md) adds
aligned split-K operand layouts and smaller pointwise launches, with fresh
original-Adam and fused-Adam PyTorch comparisons.

Complete mixed PPO updates improve from **34.77 to 34.51 ms** against the
accepted [hybrid GEMM baseline](training-bf16-hybrid-staging.md): **1.0087×
median paired throughput**, with 42/64 candidate wins. Aggregate median latency
falls 0.74%. This is a within-run comparison, separate from the earlier
GEMM experiments.

[Raw evidence](benchmarks/training-parity/bf16-packed-reductions.json) retains
all timing samples, model traces, compiled kernel resources, source/binary
hashes, independent numerical controls and rejected experiments. Mixed execution
remains opt-in; the previous original-PyTorch and learning-quality limits remain
open. No new PyTorch timing matrix was run for this scheduling change.

## Accepted implementation

Existing 32-lane warp reductions now pack eight independent output fibers into a
256-thread block. Every lane visits the same input indices in the same order,
then executes the original ascending 1/2/4/8/16 shuffle tree. A register shuffle
broadcasts lane zero's result, replacing the shared float, leader-store branch
and block barrier. Extra fibers in a partial block return as complete warps.

The launcher checks the compiled function's thread limit and occupancy before
choosing the packed size. If required, it halves the number of whole warps until
the kernel fits. Both ordinary PTX execution and fixed-binding mixed programs
use the same launch helper. A kernel with an explicit 128-thread limit tests
that fallback and the resulting output grid.

Serial reductions and all block reductions retain their original schedules,
input traversal and arithmetic. Matmul code, split-K accumulation, BF16 storage
boundaries, optimizer publication and CUDA graph boundaries are unchanged.
Fusion resource estimates account for the actual warp block size. The generated
code signature advances to version 31.

## Complete-update acceptance

Both workers start from the same width-256 shared-critic archive and fixed
real-game-derived 64-transition batch. Microbatch 64, tree reductions, TF32x3,
resident Adam and CUDA replay are enabled in both. Three warmups are excluded;
64 retained requests alternate AB/BA. Times include feeds, loss, backward,
clipping, Adam, synchronization, ordinary metrics and JSON. Every replay remains
active with zero fallback.

| Complete mixed PPO request | Median |
| --- | ---: |
| Accepted hybrid GEMM worker | 34.77 ms |
| Packed existing warp reductions | 34.51 ms |

A 10,000-resample paired bootstrap gives a 95% percentile interval of
1.0008–1.0116× for the median throughput ratio (seed 0). This is a small gain.

The independent f32 control measures 41.96/41.90 ms, a
1.0014× paired ratio with 37/64 wins. All GPU experiments run sequentially
on the RTX 4080; no unrelated CUDA compute job was observed. Desktop graphics and unrelated
host work remain running. Absolute medians from different experiments should
not be combined.

## Model attribution

Two fresh traces verify identical ordered labels for all 1,414 model kernels,
with 32 launches per node and three warmups excluded. Model span falls from
31.39 to 31.19 ms. These spans exclude optimizer, uploads, request handling and metrics.

| Model group | Baseline summed GPU median | Candidate |
| --- | ---: | ---: |
| Reductions | 5.64 ms | 5.50 ms |
| Other f32 pointwise work | 8.52 ms | 8.50 ms |
| BF16 GEMM and split-K GEMM | 14.06 ms | 14.02 ms |

For the 6 matched two-input batch-gradient sums from
`[64,64,256]` to `[64,1,256]`, the median of kernel medians falls from
24.48 to 17.57 microseconds.
Their compiled launches change from 32 to 256 threads and
16384 to 2048 blocks. Static shared allocation changes from
4 to 0 bytes; registers remain at 27 per thread. The raw report checks resource consistency across every captured launch.

## Numerical controls and rejected variants

Six complete backward/update passes per worker hold weights fixed by setting
both learning rates and the scheduler floor to zero. Every saved master weight
is bit-identical to the initial archive across all 18 passes. Against an
independent baseline pass, global gradient relative L2 ranges are:

- Baseline repeat: 1.530e-07–1.923e-07.
- Candidate: 1.532e-07–1.814e-07.

The direct gradient check stays within baseline variation. Ordinary three-update
trajectories remain sensitive. All retained relative L2 errors compare with the
first baseline trajectory:

| Run | Gradient relative L2, steps 1/2/3 | Maximum weight difference |
| --- | --- | ---: |
| Baseline repeat 2 | 1.586e-07, 1.380e-07, 1.273e-07 | 5.006e-09 |
| Baseline repeat 3 | 1.631e-07, 1.688e-07, 2.321e-04 | 9.977e-06 |
| Candidate | 1.622e-07, 1.032e-04, 3.206e-03 | 6.505e-05 |
| Candidate repeat | 1.532e-07, 1.391e-07, 2.321e-04 | 9.977e-06 |

One candidate trajectory diverges more than the observed baseline repeats;
another follows the drifting baseline closely. These short runs do not establish
multi-update parity or learning quality. Fixed-weight controls isolate gradient
arithmetic from the differences in inputs caused by preceding optimizer updates.
All numerical results, including the larger divergence, remain in the evidence.

Tests cover partial warp blocks, middle axes, changed feeds, sum/mean/max,
reduced epilogues, original full-domain block epilogues, deterministic replay and
singleton multi-output domains. An independent cancellation-sensitive reference
verifies the original 128-thread block sum tree bit for bit. CPU tests: 206 passed.
CUDA tests: 419 passed, one opt-in GEMM sweep skipped because matmul lowering is
unchanged. CPU/CUDA builds, formatting, all-target all-feature Clippy and
documentation tests pass (four passed, one ignored).

The first block-to-warp replacement was faster but changed the first update
gradients at identical initial weights by 0.389% relative L2 (up to 2.81% over
three updates). A virtual-lane version passed the isolated exact-tree
test but changed first-update gradients by 0.140% (up to 2.22% over three
updates). Both were rejected for arithmetic differences at matched initial
weights. Retaining intact block thread groups and replacing their shared tree
tail passed numerical controls, but its trace slightly regressed the largest
kernels. Its complete-update paired ratio was 1.0154×. Those block changes
were removed
to keep the retained change limited to existing warp schedules and avoid their
additional shared-memory/barrier logic. The evidence retains these
counterexamples; final acceptance covers only existing warp schedules.

## Reproduce

Preserve the accepted pre-packing worker before rebuilding, and use fresh output
directories:

```bash
cargo build --release --features cuda --example forty_five_worker
OPENBLAS_NUM_THREADS=1 TNSR_PROFILE_PPO=1 /path/to/forty-five/.venv/bin/python \
  integrations/forty_five/benchmark_native_pair.py /path/to/archive \
  --batch /path/to/fixed-batch.json --baseline-worker /path/to/baseline-worker \
  --worker target/release/examples/forty_five_worker --output target/warp-repeat \
  --microbatch 64 --warmups 3 --repeats 64 --mixed-bf16 --baseline-mixed-bf16 \
  --tree-reductions --tf32x3 --cuda-graphs
```

Omit both mixed flags for the f32 control. F32 backward pointwise work and
short block-scheduled reductions remain substantial opportunities. Replacing a
small reduction in isolation requires a complete-update numerical control before
acceptance, as the rejected variants demonstrate.
