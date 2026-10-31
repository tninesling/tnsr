# PPO strict-F32 staging experiment

The subsequent [address-reconstruction experiment](ppo-address-reconstruction.md)
adds a separately measured improvement over this K32 baseline.

The retained compiler change stages 32 reduction elements instead of 16 in
the existing 32×32 scalar-F32 register tile. This halves outer staging/barrier
iterations while retaining the increasing-K multiply-then-add sequence. It
does not introduce split-K, TF32, reassociation, or a different graph partition.
Targets without the conservative shared-memory budget retain K16. Layout
validation accepts both supported staging widths.

## Complete PPO updates on RTX 4080

| Microbatch | Baseline median ms | Candidate median ms | Candidate throughput / baseline | Pair bootstrap 95% interval |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 53.778 | 52.757 | 1.021× | [0.998, 1.036] |
| 8 | 77.305 | 73.851 | 1.049× | [1.047, 1.069] |

Microbatch 8 has a measured gain. The microbatch-64 interval includes parity;
its point estimate does not establish an improvement.

The comparison uses the frozen native worker from the completed competitor
campaign and a new worker built against the candidate compiler. Only
`src/tile/matmul_schedule.rs` and `src/tile/layout.rs` differ in the compiler
source fingerprints. All workload settings and strict-F32 arithmetic policies
remain the same. This is an optimization A/B experiment, not a refresh of
competitor rankings or memory measurements.

Each microbatch has four independent matched process pairs, one cold update,
three warmups, and 32 measured complete updates. Start order and request order
alternate. Each process pair's score divides summed baseline update time by
summed candidate update time; the table reports the median pair score and its
bootstrap interval. Latencies are medians of the per-process warm medians, a
separate statistic. The GPU lock and 616 process audit records exclude foreign
CUDA workloads.

Untimed exact-prestate CPU gradient/metric checks and independent float64 Adam
checks bracket every candidate measurement window. The baseline is also
checked before and after the first pair at each microbatch. All 20 checkpoints
passed with zero violations and the original tolerances. Both workers advance
through identical update counts, including qualification updates.

## Why isolated GEMM gains overstate the result

An eight-shape resident GEMM screen, repeated in reversed process order, found
roughly 1.2–1.4× throughput for contiguous operands, including PPO dimensions and odd
tails. The complete model includes transposed operands and different address
calculations, so these are not predictions of whole-update gains.

Separate Nsight traces include kernels inside CUDA graph replay. GEMMs account
for about 65% of baseline replay-kernel time. Several real GEMMs improve, but
the 683×256×4096 weight-gradient kernel gets slower: its mean replay-kernel
time changes from about 788 to 900 microseconds. Its JIT register count grows
from 60 to 90, with no local-memory spills. This is a concrete resource-pressure
target for future work; the traces alone do not establish a single cause.

## Rejected alternative

Forcing the existing padded shared layout in the actual resident PPO executor
reduces throughput to 0.815–0.823× at microbatch 64 and 0.838–0.840× at
microbatch 8, across two exploratory pairs each. It is not enabled. These
layout screens have no numerical qualification and do not contribute to the
qualified staging A/B estimate.

The first layout probe configured only the inference executor, so those
effectively identical control timings are superseded. The corrected temporary
driver explicitly configures the separate resident executor. An earlier A/B
attempt under `/tmp` stopped before measurement because diagnostic hardlinks
crossed filesystems; the completed experiment stores diagnostics alongside the
benchmark tensor store.

## Evidence and validation

The [compact result](benchmarks/ppo-staging-k.json) points to the complete raw
report in the sibling benchmark project at
`results/ppo-staging-k-20261029/`. That directory retains the A/B driver, exact
changed source files, frozen candidate and baseline hashes, checkpoint checks,
all measured samples, Nsight traces, the contiguous GEMM screen, and the
temporary padding driver and its results. Diagnostic kernel traces are
separate from the uninstrumented A/B timers.

Validation passed:

- 15 scheduling/layout unit tests, including the K16 budget fallback.
- Five CUDA tests with CUDA required: changed feeds, transpose/view layouts,
  tails, cancellation order, nonfinite values, grouped traversal, and BF16.
- `cargo fmt --all --check`.
- `cargo clippy --all-targets --all-features -- -D warnings`.

Reproduction uses the preserved `compare_ppo.py` driver and frozen worker
binaries. Its paths intentionally identify the local source tree, fixtures,
and tensor store; choose a fresh output directory for a new run. Baseline
worker SHA256 is
`21fbb3aefd2da1efe28b46377ec509bbaa049d9e039c19cfcc289ff3a92ade1c`;
candidate SHA256 is
`5dc182632948880a32529792835c336a45d8e51cda2bd852caeb40738d749aba`.
