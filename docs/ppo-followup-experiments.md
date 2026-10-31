# PPO follow-up experiments: no additional change retained

This round tested seven paths against the retained address-reconstruction
compiler. None demonstrated a sufficient throughput or resident-memory benefit.
All experimental compiler and test changes were reverted. Every compiler source
hash matches the previously validated baseline; earlier profitable changes remain.

## Exploratory paths

| Experiment | Finding | Decision |
| --- | --- | --- |
| Four-way consecutive-K unrolling | Mostly flat across representative GEMMs; complete outputs match | Rejected |
| Carried shared-memory pointers | No consistent gain over indexed addresses; complete outputs match | Rejected |
| Two-stage strict-F32 asynchronous staging | Helps a small contiguous GEMM, mostly flat on main PPO shapes | Rejected |
| Broad contiguous-reshape copy elision | Complete PPO throughput regresses at both sizes | Rejected |
| Affine contiguous-view address proof | No consistent complete-update improvement in exploratory pairs | Rejected |
| Affine proof plus copy elision | Exploratory updates remain slower | Rejected |
| Canonical dense-alias bases plus copy elision | Small positive estimates remain inconclusive; resident memory unchanged | Rejected |

Kernel screens use resident buffers, CUDA graph repetitions and CUDA events,
with process order reversed between repeats. Full output bits are compared
against the retained compiler. The affine-proof and combined-proof update
screens are exploratory and have no CPU derivative/Adam qualification; they
are not ranking evidence.

## Qualified complete-update comparisons

Both experiments below use four independent matched process pairs per
microbatch, one cold update, three excluded warmups and 32 measured complete
updates. Every candidate window is bracketed by untimed exact-prestate CPU
derivative/metric and independent float64 Adam checks. The first baseline pair
at each size is checked at both boundaries too. All 40 checkpoints across the
two experiments pass the original tolerances with zero violations.

Ratios are candidate throughput / retained-baseline throughput. Each score is
the median pair ratio of summed baseline/candidate update times, with a bootstrap
interval over independent pairs. All samples, including noisy pairs, are kept.

| Experiment | Microbatch | Throughput ratio | Pair bootstrap 95% interval |
| --- | ---: | ---: | ---: |
| Copy elision alone | 64 | 0.939× | [0.933, 0.940] |
| Copy elision alone | 8 | 0.948× | [0.946, 0.952] |
| Canonical dense aliases | 64 | 1.014× | [0.986, 1.016] |
| Canonical dense aliases | 8 | 1.005× | [0.994, 1.040] |

## Less logical work did not establish a practical win

The canonical variant keeps dense reshapes virtual and lets their consumers
index relative to a contiguous alias instead of decoding through earlier
reshape coordinates. Its plan has 61 fewer materializing steps: 1,982 becomes
1,921. These include contiguous-copy steps, so this is not a claim of 61 fewer
executed CUDA kernels.

Logical intermediate materialization falls by 449,750,272 bytes per update at
each microbatch. This sums intermediate extents and is not peak memory or a
measurement of physical traffic. Separate fresh-worker diagnostics show:

| Microbatch | Baseline warm GPU process MiB | Canonical candidate MiB |
| ---: | ---: | ---: |
| 64 | 1,982 | 1,982 |
| 8 | 670 | 670 |

These are repeated warm resident process snapshots after graph replay is active,
including the context and modules, not allocator peaks. Neither memory nor the
timing intervals justified retaining the extra planner complexity.

The result supports retaining the existing materialization cost guards. A dense
logical reshape does not by itself prove that virtual consumer addressing or
the complete graph will be faster.

## Validation and preservation

The final canonical candidate passed 114 CUDA tests, with one manual timing test
intentionally ignored, plus 17 planner tests, formatting and strict all-target,
all-feature Clippy. An initial CUDA assertion assumed that a composed view's
base must be the original tensor; its dense-alias counterpart checked the new
base and full CPU-referenced readback. These experimental test changes were
also reverted with the rejected implementation.

The [compact evidence](benchmarks/ppo-followup-experiments.json) references the
raw qualified comparisons and diagnostic scopes. Drivers, code snapshots,
screen data and decisions are retained in the sibling benchmark project under
`results/ppo-followup-experiments-20261030/`, with qualified cases in
`results/ppo-dense-views-20261030/` and
`results/ppo-dense-canonical-20261030/`. No competitor ranking was refreshed.

The [previous address-reconstruction improvement](ppo-address-reconstruction.md)
remains the production baseline.
