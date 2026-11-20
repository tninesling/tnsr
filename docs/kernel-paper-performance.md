# Kernel paper prototypes: RTX 4080 benchmarks

Measured 2026-10-31. Shared-memory reuse is worth taking into full-workload
qualification for the large BF16 projections: confirmed kernel throughput
improves 1.11-1.19x. Weight sharing does not justify general selection: a large
synthetic normalized contraction improves 1.10x, but multiple smaller cases
regress. Both options remain disabled by default.

[Complete raw samples and provenance (local artifact)](benchmarks/README.md)
retain all 80 independent comparison processes and 4,480 paired event samples,
including controls and regressions. Independent review reproduced every summary
and bootstrap interval, checked the raw stdout against the report, verified
source/binary hashes, and confirmed equal counts of AB and BA pairs.

## Confirmation

Four fresh processes per case, each with 64 paired samples; construction and
first-pair order reverse across processes. Latency columns are medians of the
four per-process medians. The throughput statistic is the median of per-process
`sum(baseline event times) / sum(candidate event times)`; it is not the ratio of
these displayed latency medians. Greater than one favors the candidate.

| Case | Baseline, us | Candidate, us | Throughput ratio | Descriptive 95% interval |
| --- | ---: | ---: | ---: | ---: |
| `bf16_gate_forward` | 35.559 | 32.420 | 1.1122x | 1.1067-1.1197x |
| `bf16_down_forward` | 74.091 | 62.751 | 1.1909x | 1.1815-1.1972x |
| `bf16_qkv_forward` | 39.966 | 35.296 | 1.1255x | 1.1188-1.1363x |
| `bf16_longk_transposed_gradient` | 84.800 | 85.990 | 0.9851x | 0.9809-0.9893x |
| `bf16_aligned` | 5.471 | 5.478 | 0.9987x | 0.9982-1.0047x |
| `online_tiny` | 3.212 | 3.607 | 0.8906x | 0.8903-0.8909x |
| `online_s64_d32_rows4096` | 20.988 | 20.980 | 1.0003x | 0.9997-1.0014x |
| `online_s64_d32_rows32768` | 116.440 | 105.724 | 1.1013x | 1.1010-1.1017x |
| `online_s512_d128` | 71.040 | 80.128 | 0.8864x | 0.8861-0.8868x |
| `online_serial_control` | 40.192 | 40.192 | 1.0001x | 0.9988-1.0013x |

The gate/up-shaped BF16 projection is M4096/N683/K256 with a transposed RHS;
the down projection is M4096/N256/K683 with a transposed RHS; QKV is
M4096/N768/K256 with a transposed RHS. These are ordinary GEMM kernels on
representative dimensions, not complete projection/training timings. The
long-K transposed-gradient control is M256/N683/K4096 with a physical leading
batch before the transposed LHS reshape. It is not the production split-K
training contraction plus its final sum.

The three projection wins correspond to roughly 8.8%, 15.3%, and 11.7%
lower displayed median latency. The transposed-gradient control has a confirmed
small regression, while the small aligned case is effectively flat. Reduced
shared allocation is therefore insufficient as an automatic selection rule.

## Actual driver resources

| Case | Registers/thread, both | Shared bytes, baseline -> candidate | Driver maximum active blocks/SM, baseline -> candidate |
| --- | ---: | ---: | ---: |
| BF16 gate/up and QKV | 78 | 36,864 -> 20,480 | 2 -> 3 |
| BF16 down | 64 | 26,112 -> 16,384 | 3 -> 4 |
| BF16 long-K transposed gradient | 48 | 14,336 -> 10,240 | 5 -> 5 |
| BF16 aligned control | 92 | 35,840 -> 19,456 | 2 -> 2 |

These are queried from the driver-loaded kernels. The occupancy API supplies a
resource-based maximum, not measured realized occupancy or a causal stall
breakdown. All probed variants report zero local-memory bytes. Schedule geometry,
operand layouts, stage count, copy policy and arithmetic are identical within
each pair; only the shared-storage policy differs.

Online weight sharing retains 132 shared bytes but increases registers from 32
to 39/40 in most cooperative cases; driver maximum-block limits remain unchanged.
It also adds one block barrier per score tile. These are observed changes, not
proof of which cost dominates. The large-row case is rows32768/K64/D32;
rows4096/K64/D32 is flat. The tiny case is rows4/K16/D16, while the wider
regressing case is rows128/K512/D128. Width129 is a genuinely unchanged scalar
fallback control and remains flat.

## Discovery matrix

The initial screen used two fresh processes per case and 48 pairs per process.
It selected both wins and losses for the separate confirmation above. The
remaining cases have discovery evidence only; do not pool the two campaigns or
use their time differences as additional speedup.

| Case | Baseline, us | Candidate, us | Throughput ratio | Descriptive 95% interval |
| --- | ---: | ---: | ---: | ---: |
| `bf16_gate_forward` | 35.626 | 32.504 | 1.1144x | 1.1068-1.1231x |
| `bf16_down_forward` | 74.288 | 62.880 | 1.1896x | 1.1797-1.1991x |
| `bf16_qkv_forward` | 40.007 | 35.336 | 1.1246x | 1.1110-1.1370x |
| `bf16_qkv_dgrad` | 31.200 | 30.968 | 1.0075x | 1.0072-1.0078x |
| `bf16_longk_wgrad` | 92.159 | 92.079 | 1.0013x | 0.9960-1.0068x |
| `bf16_longk_transposed_gradient` | 85.044 | 86.208 | 0.9857x | 0.9800-0.9916x |
| `bf16_small_forward` | 6.800 | 6.717 | 1.0130x | 1.0108-1.0153x |
| `bf16_aligned` | 5.472 | 5.480 | 0.9981x | 0.9973-0.9989x |
| `bf16_aligned_sync` | 10.288 | 10.288 | 0.9998x | 0.9991-1.0006x |
| `bf16_tails` | 2.784 | 2.784 | 0.9996x | 0.9984-1.0010x |
| `tf32x3_gate_forward` | 173.540 | 173.220 | 1.0026x | 0.9949-1.0110x |
| `f16_aligned` | 6.563 | 6.551 | 1.0025x | 0.9964-1.0085x |
| `online_tiny` | 3.218 | 3.612 | 0.8907x | 0.8902-0.8912x |
| `online_s64_d32_rows64` | 9.357 | 10.267 | 0.9113x | 0.9110-0.9115x |
| `online_s64_d32_rows4096` | 21.018 | 20.992 | 1.0004x | 0.9994-1.0017x |
| `online_s64_d32_rows32768` | 116.480 | 105.728 | 1.1015x | 1.1010-1.1019x |
| `online_s256_d64` | 38.680 | 42.176 | 0.9171x | 0.9167-0.9176x |
| `online_s512_d128` | 71.016 | 80.126 | 0.8862x | 0.8858-0.8864x |
| `online_width1` | 4.748 | 5.431 | 0.8733x | 0.8727-0.8738x |
| `online_serial_control` | 40.192 | 40.192 | 1.0002x | 0.9985-1.0022x |

## Protocol and scope

- RTX 4080, 16 GB, driver 580.196.07; builds completed before GPU timing.
  GPU commands run outside the restricted sandbox so the device is visible.
- One benchmark process at a time under `/tmp/tnsr-bench.cuda0.lock`. No foreign
  CUDA compute process was observed in pre/post snapshots. Desktop graphics
  stayed active; no clock, power, driver, or permission settings changed.
- Both modes compile and allocate first. Warmup alternates the completed captures
  for at least 200 ms. Timed pairs genuinely alternate AB/BA; the implementation
  avoids the rotate-plus-reverse cancellation in the older two-mode probe.
- CUDA events time captured repetitions over resident repeated buffers. GEMM
  captures contain 64 invocations; online captures contain 4-256 according to
  case size. Upload, compilation/JIT, validation and readback are excluded.
- Every finite output bit matches before and after timing; sampled independent
  f64 sanity checks pass for both variants. The earlier required-device tests
  separately cover nonfinite behavior, masks, tails and changed feeds.
- Inputs are intentionally cache-hot/reused. GEMMs omit fused bias/activation,
  cross-operation casts, training split-K final sums and optimizer work. Online
  inputs are precomputed scores with one KxD values allocation shared across all
  rows. Neither full attention, KV decoding nor complete PPO is measured here.
- Intervals use 2,000 hierarchical process/circular-block bootstrap draws,
  block length four, fixed seed 20261031. They describe this bounded same-session
  dataset. Four immediate confirmation processes and narrow intervals do not
  establish learning quality, across-session behavior or workload generality.

## Reproduce

```sh
cargo build --release --features cuda \
  --example shared_memory_probe --example online_weight_probe
python3 integrations/kernel_research/benchmark_prototypes.py \
  --output target/paper-kernels/new-screen --trials 2 --samples 48
python3 integrations/kernel_research/benchmark_prototypes.py \
  --output target/paper-kernels/new-confirmation --trials 4 --samples 64 \
  --cases bf16_gate_forward,bf16_down_forward,bf16_qkv_forward,bf16_longk_transposed_gradient,bf16_aligned,online_tiny,online_s64_d32_rows4096,online_s64_d32_rows32768,online_s512_d128,online_serial_control
```

Use a fresh output directory and run GPU commands outside the sandbox. Raw
stdout/stderr, commands, snapshots, samples and hashes are also preserved under
`target/paper-kernels/benchmarks/{screen,confirmation}/` locally. The checked-in
JSON embeds both complete reports so cleaning `target/` does not discard the
measurement data. No production selection heuristic changed in this pass.

## Next decision

The [complete transformer training baseline](transformer-training-performance.md)
now measures 15M–211M parameter decoders, including larger batches and an actual
CUDA memory limit. Both experimental kernel options remain disabled in that
baseline; these component gains have not been qualified end to end.

Advance shared-memory reuse through an explicit mixed-program option, first on
large BF16 projections. Include the existing fused epilogues and compare matched
complete updates under unchanged numerical gates before considering automatic
selection. Retain smaller and gradient fallbacks because the current control
shows a regression. Do not multiply component ratios into a predicted full-model
speedup.

Keep online weight sharing experimental. Reducing or eliminating its extra
publication barrier is a separate implementation and correctness experiment;
the current large synthetic win is insufficient to enable it generally or in
the cached decoder.
