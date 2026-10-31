# PPO reshape-address reconstruction

The retained change reconstructs scheduled unsigned integer addresses before
bounded e-graph extraction. It removes division/remainder round trips created
when a batched tensor is flattened and transposed for a weight-gradient GEMM.

For example, a physical `[64, 64, 683]` tensor generated this reconstruction of
an already-linear index:

```text
(x % 683) + ((x / 683) % 64) * 683 + (x / 43712) * 43712
```

This equals `x` for every unsigned 64-bit value. Previously the generated PTX
still divided by 683 and 43,712. The new prepass expands constant remainders,
combines wrapping integer coefficients, and joins nested positive constant
quotients, then adopts a reconstruction only when its estimated address cost
falls. The existing e-graph normalization/extraction still runs afterward.

Only scheduled wrapping-u64 addresses use the prepass. Checked signed virtual
maps retain their existing rules. Every original divisor and shift is validated
before dead terms can disappear. No memory access or floating-point operation
moves, and the increasing-K accumulation policy is retained. New tests cover
unsigned boundaries, randomized values and an intentionally wrapping reshape
that must not collapse to `x`.

## Complete PPO updates: RTX 4080, strict F32

These results compare against the previously retained K32-staging compiler,
not the older K16 compiler or a competitor framework.

| Microbatch | Baseline warm median ms | Candidate warm median ms | Candidate throughput / baseline | Pair bootstrap 95% interval |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 52.767 | 47.969 | 1.094× | [1.092, 1.098] |
| 8 | 74.101 | 68.784 | 1.078× | [1.049, 1.079] |

Each size uses four independent matched process pairs, one cold update,
three excluded warmups and 32 measured complete PPO updates per process.
Process start order and request order alternate. Throughput uses the median
of each pair's summed-baseline-time / summed-candidate-time ratio; the latency
columns are medians of per-process warm medians, a separate statistic.

Exact-prestate CPU derivative/metric qualification and independent float64
Adam checks bracket every candidate window. The first baseline pair at each
size is also qualified at both boundaries. All 20 checkpoints pass the original
tolerances with zero violations. Measured calls retain strict F32, zero split-K
contractions, active CUDA graph replay, zero replay fallbacks and no diagnostic
timing instrumentation. GPU process audits reject foreign CUDA work.

Only `src/tile/index_egraph.rs` differs in the compiler source fingerprints
between this candidate and the K32 baseline. The transposed-layout probe was
extended separately. Competitor rankings and memory profiles have not been
refreshed for this change.

## Kernel investigation and rejected path

The initial alternative disabled index sharing for synchronous scalar register
tiles. It lowered the plain transposed probe's register count from 72 to 44,
but made the hot contraction slower. That policy was reverted; the existing
BF16 policy is unchanged.

The corrected probe preserves a leading physical batch axis before flattening
and transposing, reproducing the actual PPO address expression. Two process
orders with CUDA-event timing and resident graph repetitions measured:

| GEMM M×N×K | Physical leading batch | Before µs | After µs | Throughput ratio | Registers before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 683×256×4096 | 64 | 843.45 | 383.29 | 2.20× | 90 → 78 |
| 768×256×4096 | 64 | 823.02 | 384.59 | 2.14× | 90 → 78 |
| 683×256×512 | 8 | 100.91 | 49.04 | 2.06× | 90 → 78 |

All output bits match between builds across the ten screened shapes, including
transpose combinations, composed reshapes, odd tails and simple-view controls.
Sampled outputs also check an independent CPU f64 dot product. These exploratory
kernel results exclude transfers and do not replace complete-update timings.
Some simple-view probe cases regress by about 3–5%; the complete PPO estimates
include the cost of the model's other kernels. Register reduction alone is not
the acceptance criterion.

Separate Nsight traces capture kernels inside CUDA graph replay and retain the
rejected and retained candidates. They are diagnostic artifacts, excluded from
the uninstrumented A/B timers. Full PPO is qualified numerically; the probe's
bitwise-equality result is not a full-model bitwise-equivalence claim.

## Validation and evidence

- 50 CPU tile/index tests pass, including unsigned wrapping and checked signed
  index semantics.
- 113 CUDA tests pass across ten binaries: matmul ordering/layouts, scalar
  loops, online regions, indexed loss, convolution, softmax and transformer
  execution. One manual compact-softmax timing test remains intentionally ignored.
- Formatting, strict all-target/all-feature Clippy and `git diff --check` pass.
- All 20 PPO numerical checkpoints pass; sources and frozen binary hashes are
  verified before and after measurement.

The [compact evidence](benchmarks/ppo-address-reconstruction.json) points to
`/home/taylor/code/tnsr-benchmarks/results/ppo-index-reconstruct-20261030/`.
That directory retains raw matched measurements, numerical checkpoints,
source fingerprints, frozen candidate/probe binaries, complete probe outputs,
drivers, GPU audits, Nsight traces and regression-test output. The prior
[staging experiment](ppo-staging-k.md) remains a separate comparison.
