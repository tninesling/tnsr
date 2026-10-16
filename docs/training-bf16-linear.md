# Physical BF16 linear forward/backward

`ptx::linear::LinearTraining` implements a complete, fixed-shape linear island:
physical BF16 operands/results, explicit conversion kernels, analytical input,
weight and bias gradients, promotion at the f32 boundary, and resident f32
parameter-gradient accumulation. It is an implementation slice of
[#114](https://github.com/tninesling/tnsr/issues/114), not the full mixed
TensorGraph/autograd foundation or an enabled forty-five training policy.
Production PPO arithmetic is unchanged.

The preceding [headroom study](training-bf16-headroom.md) measured arithmetic
substitution. This path actually allocates BF16 storage and executes the full
forward/backward boundary, so conversion and gradient costs can now be checked.

## Implemented boundaries

Inputs use `x[rows,inputs]`, `weight[outputs,inputs]`, `bias[outputs]`, and one
or two externally supplied upstream gradients `[rows,outputs]`. Master inputs,
weights and bias remain f32. Operand conversions write separate two-byte BF16
allocations. Forward uses f32 accumulation with fused bias and one BF16 output
rounding. Backward computes `g @ weight`, `g.T @ x`, and the bias sum, with BF16
output rounding before promotion into f32 gradients. Accumulation across calls
adds the promoted parameter gradients into f32 buffers.

With two downstream contributions, each seed is converted to BF16 before their
BF16 sum, which then seeds the single linear backward. The original Torch
fixture uses two distinct `y.float()` consumers to observe that same boundary.
This is different from one shared f32 promotion whose consumers sum in f32.
Separate linear uses of a shared master weight/cached cast still require the
larger cast-aware graph and are not represented by this API.

Four ordinary homogeneous Tile IR/PTX subgraphs implement forward, input
gradient, weight gradient and bias reduction; typed boundary kernels connect
them. No static CUDA or handwritten PTX is introduced. Fixed owned bindings
retain every buffer/module through capture. Upload/readback stay outside replay,
changes refresh the existing f32 master addresses, and capture contains the
whole computation including casts, promotions and f32 accumulation. Plain
execution uses the same kernels. Shape/branch/extent errors are checked before
uploads mutate any boundary buffer. This low-level primitive exposes results;
clipping, finite-update acceptance and optimizer publication remain trainer
responsibilities. After a rejected nonfinite input, clear the accumulators
before resuming with finite data.

The TF32x3 comparator uses four K partials plus a final f32 reduction for eligible
weight gradients, with the current split-K eligibility rule. This avoids
comparing BF16 against the superseded unsplit f32 strategy. Both paths use the
same standard `[outputs,inputs]` weight layout; the full model can canonicalize
its contraction orientation differently, so these are isolated linear calls,
not exact substitutes for the complete model plan.

## Correctness and optimizer sensitivity

Independent scalar-oracle GPU tests cover scalar fallback and odd tensor-core
shapes, two separately rounded downstream contributions, f32 accumulation,
changed inputs/weights without recapture, plain/replay equivalence, and invalid
uploads without partial mutation. A cancellation fixture uses BF16-representable
operands whose fused-bias output is exactly `1/16384`; separately rounding GEMM
before bias incorrectly gives zero. Nonfinite propagation and recovery after
clearing accumulation, plus underflowing seeds, are tested.

Validation passes 194 CPU tests, 398 CUDA-required all-feature tests, CPU/CUDA
checks, formatting, all-target/all-feature Clippy and documentation tests.

The original Torch CUDA autocast oracle is separate from timing. All seven
shape/regime cases include forward and all three gradients. The existing f32
sanity threshold (normalized maximum 2e-5) passes every TF32x3 comparison; no
f32/PPO tolerance changes. BF16's quantized-input/f32-accumulation comparison is
reported separately from original default cuBLAS. Large random-input results
are not bitwise identical: the down weight-gradient has 99.7546% exact agreement
with the declared quantized-f32 oracle and 58.2981% with default cuBLAS BF16.
Primitive sanity checks are not original-model acceptance tolerances.

The odd 33×65→35 fixture runs five actual f32 master/moment/variance updates
using native gradients, with CPU-f32 Adam outside timing. An independent original
Torch autocast/Adam trajectory differs by at most 2.24e-8 in weights. These
externally seeded VJPs do not test feedback through an actual PPO loss or the
resident GPU optimizer. Large cases run one optimizer sensitivity step; they
expose a much larger maximum weight difference, about 0.002 at learning rate
0.001, despite small gradient errors.

A focused down-projection diagnostic explains the latter. Among 174,848 weight
gradients, native versus default Torch BF16 has 78 sign differences near zero.
Original Torch with reduced-precision reductions disabled has the same 78 sign
differences from its default. Native has zero sign differences from that
higher-precision diagnostic, quantized FP32, or quantized FP64 in this fixture.
Native's RMSE versus quantized FP64 is 0.000122; versus original default it is
0.001660. For example, one native gradient and the rounded FP64 reference are
+0.0002975464 while default Torch returns -0.0002441406. Adam can turn that sign
change into nearly two learning rates of first-step weight difference.

The default Torch flag is retained for the original oracle and restored after
the explicitly labeled diagnostic. This is evidence for calibrating reference
policy variability and optimizer sensitivity, not for silently changing the
reference, loosening global f32 tolerances or declaring original-BF16 model
parity. Full named-gradient/multistep calibration remains #115.

## Contended performance diagnostics

A separate forty-five Python training process was already using approximately
6.5 GiB and 60–100% GPU utilization during this entire sweep. It was recorded
and left running. Agent-owned GPU work was serialized, and no builds overlapped
the measurements. These are contended component diagnostics, not an exclusive
GPU parity gate or achieved full-model throughput improvement.

The same native process alternates TF32x3/BF16 complete calls for 200 pairs per
case. CUDA events include all three GEMMs, bias reduction, fresh operand/seed
casts, output promotions and f32 accumulation; uploads/readbacks, JIT, allocation
and CPU optimizer diagnostics are outside intervals. Hot tests repeat 32 calls
per sample. Streaming tests rotate four large or 32 small allocation sets, with
each path's buffer payload exceeding twice the 64 MiB L2, and measure one call
per sample. Every owner synchronizes all eight warmups before any timing
starts. Order also reverses across allocation cycles. All replica results
are checked for agreement before timing.

| Shape (rows,inputs,outputs) / regime | TF32x3 | BF16 | Median paired throughput ratio |
| --- | ---: | ---: | ---: |
| 4096,683,256 / streaming | 501.76 µs | 363.52 µs | 1.380 |
| 4096,256,683 / streaming | 549.89 µs | 393.22 µs | 1.398 |
| 512,683,256 / streaming | 81.92 µs | 75.78 µs | 1.095 |
| 33,65,35 / hot, two contributions | 11.17 µs | 15.42 µs | 0.725 |

Large streaming calls win 200/200 pairs, the 512-row case wins 196/200, and the
small fanout case wins 0/200. Circular 32-pair block-bootstrap intervals and all raw hot
and streaming samples are retained. In this shared workload, hot medians can be
slower than streaming and the hot 512-row ratio also differs. The regimes
change both cache reuse and submission duty cycle under external scheduling;
do not compare their absolute medians to infer cache effects or treat tight
within-run intervals as generality across GPU load. Use the streaming cases as
measured diagnostics and repeat on an idle GPU before policy acceptance.

An earlier sweep omitted the explicit per-owner warmup fence. Its trace
shows the first timed graph starting 3.418 ms before the last replica warmup
ended. That entire run is superseded, with all samples retained separately and
excluded from accepted statistics. The corrected trace verifies all warmups
finish before the first timed graph, with a 0.239 ms gap. All 247 saved boundary,
gradient and optimizer-state snapshots are byte-identical across runs. The
small hot ratio changes from 2.325 to 1.218, but external load also varied; this
difference does not isolate the effect of the fence.

A separate Nsight trace of the large down projection confirms 29 complete
launches per path (one validation, eight warmups, twenty samples). TF32x3 has
seven nodes/call; BF16 fourteen. The trace selects non-null `graphNodeId` and
groups by each path's private stream. Summed GEMM time falls 14.564→8.161 ms,
while conversion/accumulation rises 0.093→2.691 ms. Reduction time falls
0.626→0.415 ms because BF16 has no split-K final sum. Total kernel time is
15.283→11.267 ms. This separate, contended trace establishes where costs move;
it is not pooled with CUDA-event samples as a wall-clock speedup.

Across those 29 traced BF16 calls, mean kernel times are 113.93 µs for the
weight gradient, 90.64 µs for forward and 76.86 µs for the input gradient.
Converting the input alone costs 54.60 µs. These diagnostics favor integrating
typed edges that retain BF16 between eligible operations, followed by focused
long-K weight-gradient work; they do not justify blanket BF16 conversion or
extrapolation to full-model throughput.

For the large down case, retained buffer payload is 35.67 MB for TF32x3 and
48.96 MB for the BF16 island. BF16 includes about 16.08 MB of actual BF16 buffers
alongside every f32 master and exposed f32 boundary. This isolation deliberately
retains both representations and intermediates. It does not demonstrate peak
VRAM or whole-model memory savings. `program_temporary_bytes` is a subset of
the dtype totals, not additional memory to add to them.

## Use and reproduce

`LinearTraining::new(shape, LinearPrecision::Bf16, branches, replay)` creates the
fixed owner. `upload(&LinearInputs)` replaces f32 values, `run()` enqueues the
complete forward/backward, `readback()` returns promoted results and accumulated
gradients, and `clear_gradients()` starts a fresh accumulation window.
`info()` reports physical buffer payloads, kernel counts and actual plans.
The TF32x3 variant is an explicit comparator; existing executor defaults are
unchanged. BF16 requires the existing SM80-or-newer backend support.

```sh
cargo build --release --features cuda --example bf16_linear_probe
/path/to/forty-five/.venv/bin/python integrations/forty_five/profile_bf16_linear.py \
  --output FRESH_DIRECTORY
```

`--validate-only --output EXISTING_DIRECTORY` recomputes oracles and the focused
reduction diagnostic without replacing timed samples. The JSON-lines worker
also accepts fixed shape/branch/replay/replica specs and explicit run/bench/info
requests for targeted experiments. The existing Nsight command from the
[headroom report](training-bf16-headroom.md) applies to this worker as well.

[bf16-linear.json](benchmarks/training-parity/bf16-linear.json) preserves complete
samples, warmup protocol, GPU/process observations, numeric/optimizer errors,
reference flags, source/worker hashes, exact original timing-driver source,
bootstrap settings and trace aggregates. Binary boundary/output snapshots,
plan/PTX sources and trace files remain under ignored
`target/forty_five_bf16_linear/`.

Next, integrate explicit dtype/cast values and typed bindings into the general
graph/autograd, including shared master casts across distinct linear uses,
view/liveness/pool/signature behavior and captured ownership. Then encode and
calibrate the actual forty-five policy. This isolated primitive does not close
#114, #115, #94 or the PyTorch parity gates.
