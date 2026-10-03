# Scheduled integer index optimization (#61)

Measured on 2026-10-03, NVIDIA GeForce RTX 4080 (sm_89), driver 580.178.04.
Baseline: main commit `6fbc618b5f994afa6e254ed9aebfd35d96f51556`.
Candidate: the shared integer e-graph and scheduled index placement in this PR.
Both use the same target-aware matmul schedule, TF32 policy, and fused epilogue.

The integer e-graph is shared by checked signed virtual maps and wrapping unsigned
scheduled addresses. It folds constants and identities, eliminates redundant
quotients/remainders, and strength-reduces proven nonnegative power-of-two division
and modulo. Scheduled expressions are shared and bound outside loops that do not
supply their dependencies. PTX only emits the extracted operations and bindings.
The pass does not reassociate floating-point computations or move memory operations.

## Results

The graph is `relu(A @ W + bias) + residual` with shared weights. Times are
microseconds; values below are medians. See [sweep.csv](sweep.csv) for all 62 rows
(31 paired cases), p10/p90, compile/pass time, launch count, intermediate bytes,
and maximum absolute error against the CPU reference.

| Mode and B/M/K/N | Resident before | Resident after | End-to-end before | End-to-end after |
| --- | ---: | ---: | ---: | ---: |
| Batch 2/128/256/768 (GPT projection) | 66.272 | 22.877 | 402.782 | 355.280 |
| Batch 3/128/256/768 | 85.020 | 31.296 | 504.610 | 453.061 |
| Batch 5/128/256/768 | 134.592 | 50.080 | 712.162 | 629.302 |
| Flat 2/128/256/768 | 23.680 | 20.540 | 347.908 | 348.753 |
| Batch 3/33/37/35 (edge tiles) | 4.576 | 2.976 | 32.900 | 30.099 |

GPT projection resident time falls 65.5% (2.90x faster), and end-to-end time falls
11.8%. All paired cases retain one kernel launch, zero intermediate materialized
bytes, and identical maximum numerical error. The GPT projection error is
0.00048798323. The flat control remains diagnostic; no shape-specific fusion
cutoff is introduced.

Across B = 1/2/8, K = 128/256/512, N = 384/768/1536 at M = 128,
resident time falls 9.2–31.5% for B=1, 59.5–69.0% for B=2, and 63.3–68.5%
for B=8. End-to-end gains are smaller and are not universal: the B=8 sweep ranges
from a 19.3% decrease to a 3.7% increase. These are single paired runs, not
confidence intervals; host copies/allocation and run variance remain relevant.

## Compile time and resource cost

The new scheduled index pass takes 762–1047 us across the sweep, median 823 us.
For the already-JIT-cached GPT projection, total compile time is 553 us before
and 1988 us after, including an 828 us index pass. Other total compile samples
include cold driver JIT work (up to about 39 ms); baseline/candidate JIT cache
states differ, so total compile medians are not a fair isolated pass comparison.
The pass is paid during compilation, not on each resident or compiled execution.
Virtual-map normalization also uses the shared engine outside the separately timed
scheduled pass. Runner limits bound saturation to four iterations and 2048 nodes
per distinct root expression; roots are cached within each kernel.

`ptxas -arch=sm_89 -v` and `nvdisasm` on the GPT projection show:

| Static resource / instruction evidence | Before | After |
| --- | ---: | ---: |
| Registers per thread | 40 | 56 |
| Shared memory bytes | 3072 | 3072 |
| Stack / spill loads / spill stores | 0 / 0 / 0 | 0 / 0 / 0 |
| General u64 division call sites in SASS | 5 | 1 |
| General u64 division call sites inside the K loop | 2 | 0 |

The remaining division decodes an epilogue index with non-power-of-two N=768.
Bindings increase register lifetimes; the measured benefit outweighs that cost
here, but this is not a target-aware register-pressure model. Issue #66 tracks
resource-aware selection. Hardware counters were not used; these conclusions
come from paired timings, generated PTX, and static disassembly.

## Reproduction

Build a release profiler binary from each checkout and copy it before building
the other version. The baseline example can be instrumented with the candidate's
`METRICS` print block, omitting `index_optimization_time` and printing zero instead;
this changes reporting only. The runner also accepts an uninstrumented baseline
and leaves its missing metric fields blank.

```sh
cargo build --release --features cuda --example matmul_profile
cp target/release/examples/matmul_profile /tmp/issue61-after-profile
# Repeat in a detached checkout of the baseline commit, copying to before-profile.
python3 docs/benchmarks/index-issue61/sweep.py \
  /tmp/issue61-before-profile /tmp/issue61-after-profile /tmp/index-sweep.csv

TNSR_PROFILE_PTX=/tmp/index-after.ptx /tmp/issue61-after-profile batch 3 2 128 256 768
ptxas -arch=sm_89 -v /tmp/index-after.ptx -o /tmp/index-after.cubin
nvdisasm /tmp/index-after.cubin > /tmp/index-after.sass
```

The sweep alternates before/after order by case. Run with no other GPU workloads.
The profiler warms resident launches 16 times and measures 20 groups of 32
launches with CUDA events. End-to-end timing warms execution three times and
measures 20 groups of five executions, including upload/allocation/download but
excluding compilation. The CSV percentiles describe these grouped samples.
The added `METRICS` row reports compile us, scheduled index optimization us,
executor kernel launches, and intermediate bytes; the existing `RESULT` row
retains its schema.

## Correctness coverage

Semantic tests cover wrapping boundary values, signed Euclidean division,
checked overflow, invalid divisors, strength reduction, and quotient reconstruction.
A nested-loop interpreter checks placement against reference addresses; sharing
coverage includes commuted expressions and unknown external variables.
GPU regression coverage checks batches 1/2/3/8 with irregular M/K/N under both
strict F32 and TF32. Existing suites cover two-sided broadcasting, aliases,
zero-K/empty outputs, half precision, reductions, and transformer/convolution paths.
Validation: 156 CPU tests, 260 CUDA tests, 275 all-feature tests; format, CPU/CUDA
checks, and all-target/all-feature clippy pass.
