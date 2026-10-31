# Replacing the Ada strict-F32 GEMM architecture

Research proposal, 2026-10-30. No GPU workloads, builds, or production changes
were performed. Resource arithmetic below is analytical; all proposed speedups
remain unmeasured. This extends [the bounded register-tile research](ada-register-tiles.md)
with an architectural experiment rather than another local PTX cleanup.

## Recommendation

Build one new hierarchical SIMT kernel family with **16–32 accumulator values
per thread**, explicitly pipelined register fragments, and a shared-memory
layout designed together with its thread ownership. Start with a 64×64×16 CTA
and 4×4 thread tile, then extend the same framework to 64×128×16 and 8×4.
Select these only when the output grid supplies sufficient parallel work.
Retain the current small tile for long-K weight gradients until measurements
justify replacing it. A single universally larger CTA is the wrong objective.

For suitable large-output GEMMs, **1.5–2× is a plausible experimental target**:
the present kernel computes only four sums per thread and has substantially
less register/block operand reuse. It is not a prediction. Current SASS already
has FFMA, vectorized loads and full K-tile unrolling, so there is no missing
two-to-one FMA shortcut. More reuse can lose to lower occupancy, tails, shared
bank conflicts, register spills or insufficient output parallelism.

## What the implementation already supports

`src/tile/matmul_schedule.rs` selects a 32×32×32 scalar register tile, 256
threads and four sums per thread (K16 fallback). `lower_register_tiled_scalar_matmul`
in `src/ptx/graph/matmul.rs` assigns four rows, separated by eight, to one
column and reuses B across those rows. Shared layouts already include padding,
swizzles and operand orientations. Two-stage global-to-shared asynchronous
copy already exists, including an explicit scalar-F32 option. Larger BF16
tensor-core tiles and operand-fragment reuse also exist; that is not a new
strict-F32 implementation.

The exact retained driver-JIT cubin has 128 FFMA, zero separate FMUL/FADD,
32 LDS.128 and 32 scalar LDS per static K32 body; hot regions94/96 use 78
registers/thread, 8192 shared bytes and no local spills. The matched rejected
stride33 layout instead has 160 scalar LDS and no LDS.128. Preserving vector
alignment is therefore an observed requirement, not cosmetic layout tuning.
Hardware counters remain unavailable (ERR_NVGPUCTRPERM), so static evidence
does not identify the dynamic limiting resource.

Two late division-helper calls per K tile and removable bounds predicates are
real opportunities documented separately. Use their corrected implementation
as a control when judging the architecture; do not credit it with improvements
that an unchanged 4×1 kernel obtains from codegen fixes. Four-way manual
unrolling, carried pointers, a simple asynchronous-F32 toggle and broad reshape
removal were already screened and rejected in
[the follow-up experiments](../ppo-followup-experiments.md).

## Why a different architecture is credible

CUTLASS describes hierarchical CTA/warp/thread blocking, two-dimensional
per-thread outer products, register-fragment double buffering in addition to
shared double buffering, and a separate epilogue exchange when the computation
layout does not coalesce stores. Larger tiles trade reuse against resource
usage and output parallelism. These are the architectural ingredients missing
from the current small scalar thread tile.
([NVIDIA CUTLASS efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html))

This is not solely a tensor-core pattern: NVIDIA's actual SM80 SIMT test
configuration uses a 128×128×16 tile, 256 threads, `UniversalFMA` and
`MainloopSm80CpAsync<3>`. Its load-vectorization permutation is explicit.
This supplies an implementation reference, not evidence that its tile or
three stages are optimal on our Ada shapes.
([CUTLASS SIMT multistage configuration](https://github.com/NVIDIA/cutlass/blob/main/test/unit/gemm/device/default_gemm_configuration.hpp))

Ada has 64K 32-bit registers, a 48-warp limit and up to100 KiB shared memory per
SM; individual threads can use up to255 registers. Static shared allocation
remains limited to48 KiB, and actual shared carveout matters. These limits
permit larger register tiles but make resource scheduling material.
([NVIDIA Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/))

## Ranked prototype 1: hierarchical outer-product family

Use explicit logical lane geometry, separate from the physical 16×16 launch:

| Variant | CTA M×N×K | Logical lane rows×columns | Thread output tile | Accumulators/thread | Input shared/stage, unpadded |
| --- | --- | --- | --- | ---: | ---: |
| First implementation | 64×64×16 | 16×16 | 4×4 | 16 | 8 KiB |
| Large-output extension | 64×128×16 | 8×32 | 8×4 | 32 | 12 KiB |
| Current reference | 32×32×32 | 8×32 | 4×1 | 4 | 8 KiB |

One valid starting ownership is `row = lane_row + i*lane_rows`,
`column = lane_column + j*lane_columns`. At each increasing K position, load
the thread's A row fragment and B column fragment, then update all independent
outer-product sums. Ownership is bijective by construction for these exact
tile extents. This is an initial mapping to evaluate, not an asserted optimal
warp permutation. A CUTLASS-style permutation may preserve vector loads better.

### Reuse and the pipeline

A 4×4 tile requires eight scalar operand values for16 FFMA; 8×4 requires12
for32 FFMA. Current 4×1 requires five for4. Those source-level ratios describe
reuse, **not instruction counts**: the current JIT already combines four K
positions of A into LDS.128. With K16 and retained four-wide K loads, an 8×4
thread tile could use32 vector A loads plus64 scalar B loads for512 FFMA;
if B is also vectorized across K, that becomes32+16 vector loads. These are
conditional static counts, not measured schedules or runtime ratios.

Introduce a small ping-pong register-fragment pipeline: request the next
four-K fragment before finishing the current fragment's FFMA work. Consume
positions in increasing K for every accumulator. Avoid holding every K
fragment live simultaneously. Begin with synchronous staging to isolate
register reuse, then enable the existing two shared stages. Only add a
three-stage ring if the measured gap and generated instruction schedule
justify the extra state and memory. This is different from the rejected
toggle on the existing four-sum kernel.

The larger CTA increases ideal input reuse. Ignoring output traffic and tails,
CTA arithmetic/input-byte ratios are 8,16 and21.33 FLOP/byte for32×32,
64×64 and64×128 respectively; K cancels in this calculation. L2 reuse may make
the actual device-memory benefit smaller. Register tiling chiefly targets
shared-load/issue overhead and independent FFMA chains, not just DRAM bytes.

### Layout and epilogue are part of the architecture

Screen row-major A/B first, and a vectorization-compatible transposed operand
layout second. For K16, a column-major B leading dimension20 preserves
16-byte alignment and costs an extra1 KiB for64 columns, or2 KiB for128
columns. Thus the candidate stages cost9/14 KiB; double buffering18/28 KiB.
This layout can improve K-contiguous register fetches while making staging
writes worse for a different physical source orientation. Keep source
orientation in dispatch and evaluate both operand movement directions.

Do not use a blanket stride17/33 skew: it breaks four-float row alignment.
Also do not declare a whole warp conflict-free just from scalar expressions.
Shared memory has32 banks, broadcasts and wider instructions with multiple
transactions; examine the generated LDS/STS widths and actual transaction
groups. A K-major stride20 arrangement is a hypothesis requiring that audit.
([NVIDIA shared-memory bank rules](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks))

For the 8×32 lane grid, each warp can store one complete32-column stripe per
output-column round. For the16×16 grid, a warp covers two16-column stripes;
check sector utilization. Preserve direct epilogue fusion where efficient.
If a shared exchange is needed, alias dead operand shared storage after its
last barrier rather than blindly allocating another entire C tile. The
epilogue must preserve source-view maps, bias/activation fusion and tail masks.

### Resource budget and shape dispatch

Budget128 registers/thread initially: 256 threads consume32768 registers,
allowing at most two CTAs by registers before allocation rounding. At96,
three are theoretically possible; at160, only one. These are upper bounds,
not occupancy predictions. The existing78-register kernel can theoretically
fit three CTAs by registers. A 32-accumulator tile with two four-K operand
fragments could consume128 floating registers before address state if both
fragments were fully live:32 sums +2×(8+4)×4. Schedule smaller fragments or
shorten lifetimes instead of assuming register pressure will solve itself.
Reject spilling as an initial design; do not cap registers until the resulting
local-memory behavior has been inspected.

| Representative M×N×K | Existing32×32 grid | 64×64 grid | 64×128 grid |
| --- | ---: | ---: | ---: |
| 4096×683×256 forward | 2816 | 704 | 384 |
| 512×683×256 forward | 352 | 88 | 48 |
| 683×256×4096 gradient | 176 | 44 | 22 |
| 768×256×4096 gradient | 192 | 48 | 24 |

These are exact ceiling-divided CTA counts. The large-output forward is a good
first target; the gradient is not. Even the 64×64 tile loses useful grid
parallelism on the512-row forward. Narrow N683 also wastes11% of a128-column
padded extent versus3% with32 columns. Dispatch initially requires at least
two waves relative to queried SM count and sufficient useful tile coverage;
then tune using measured latency, not a universal occupancy threshold.

## Ranked prototype 2: deterministic split-K, only under a new policy

For long-K, small-output gradients, splitting K into4–8 fixed partitions can
restore grid parallelism while using larger microtiles. A separate reduction
kernel combines partial sums in a deterministic order. For683×256, eight
partitions require about5.34 MiB of F32 partial-output workspace; kernels and
workspace must be included in timing. This is a genuine architectural route
to the gradient bottleneck, unlike making an already small output grid larger.

However, it changes summation grouping. **It is incompatible with the current
increasing-K StrictF32 contract**, even with full-precision operands and a
deterministic reduction. The same objection applies to sliced-K across warps.
Treat this as a separately named ordered-policy alternative requiring an
arithmetic decision, not as a StrictF32 implementation detail. Serial split-K
with a carried exact accumulator retains order but removes the useful parallel
reduction and adds traffic/launches. Do not prioritize it.

CUTLASS's documented split-K uses partitioned GEMM plus an additional batched
reduction and workspace; sliced-K similarly reduces warp partial sums.
([CUTLASS parallel reductions](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#parallelized-reductions))

Persistent CTAs over **output tiles** preserve arithmetic order and may reuse
operand tiles or reduce setup overhead, but do not fix an underfilled output
grid. Test persistence only after the new large-output kernel works. Reusing
weights across PPO optimizer steps would require invalidation after updates;
do not cache stale shared/register operands. Ada supports the ordinary SIMT
and asynchronous-copy route; Hopper TMA/WGMMA producer-consumer kernels are
not drop-in implementations for this GPU.

## Implementation footprint

This needs a coherent schedule/lowering change, not a new magic tile constant:

- `src/tile/matmul_schedule.rs`: explicit thread/warp tile geometry, shared
  stage resources and a shape-specific candidate family. Remove the scalar
  resource model's unnecessary C-shared reservation when no exchange is used.
- `src/tile/layout.rs`: replace the hardcoded four-element32×32 validation
  with geometry/ownership validation, retaining conservative defaults.
- `src/ptx/graph/matmul.rs`: outer-product lowering and bounded live register
  fragments, with optional two-stage pipeline support.
- `src/tile/graph.rs`: one authoritative mapping from accumulator slot to
  logical output coordinate, reused for extraction, epilogues and stores.
  Changing math ownership without this change silently permutes outputs.
- `src/tile/shared_layout.rs` and pipeline builder/IR: only extend when the
  selected skew or deeper stage ring cannot use existing representations.
- Ordered-matmul tests plus a frozen resident probe in the benchmark project.

The first prototype is several interacting compiler modules and dedicated
correctness tests; a three-stage ring or shared epilogue is a second phase.
Avoid estimating this as a one-function rewrite.

## Decisive experiment and success gate

First calibrate the opportunity using the identical resident input/view
shapes against cuBLAS pedantic F32 and a CUTLASS SIMT reference, explicitly
disabling TF32. Include packing/materialization for composed views in a second
comparison, and distinguish core-kernel from end-to-end latency. cuBLAS's
pedantic mode uses32-bit arithmetic throughout, but does not promise our exact
increasing-K order; use it as a performance ceiling reference and numerical
comparison, not a bitwise oracle.
([NVIDIA cuBLAS compute types](https://docs.nvidia.com/cuda/cublas/index.html#cublascomputetype-t))

Then implement64×64×16/4×4 behind an explicit candidate selector, extend to
64×128×16/8×4 after correctness, and compare both to the frozen baseline and
the small-tile codegen-corrected control. Collect actual JIT register/shared/
spill data and SASS load widths. Run matched CUDA-event resident repetitions
with process-order reversal, changing inputs between calls and retaining full
outputs. Evaluate actual transpose/reshape access maps, not only contiguous
GEMMs. If neither architecture delivers at least1.5× on4096×683×256 or its
transpose-relevant partner, despite a larger vendor-reference gap, investigate
layout/pipeline before expanding the family. If no such reference gap exists,
lower the target rather than prolonging a speculative implementation.

Correctness gates: accumulator ownership bijection; untouched increasing-K
order and existing contraction behavior; bitwise comparison to retained
kernel on finite ordered cases; cancellation, signed zero and nonfinite
classification tests; K15/16/17/31/32/33/63/64/65/129; M/N tails including683;
transposed A/B, batch/broadcast views, changing feeds, zero-fill of final
prefetch and fused epilogues. Preserve existing tests in
`tests/matmul_register_ordered.rs`. For architecture failures, compare plain
unfused products first to separate math ownership from epilogue errors.

Measure full PPO micro64/micro8 with all numerical checkpoints. Also run the
three primary workload regressions: current GPT-small has M32, AlexNet dense
layers have M1, and LeNet is small. They should exercise fallback, not be used
as evidence that the large-CTA kernel generalizes. Include synthetic128×128,
1024×1024 and4096×1024 controls, with transposes/tails, to distinguish a PPO-only
dispatch from a generally useful kernel family.

## How much PPO improvement could this buy?

For fraction p of **complete measured PPO update time** accelerated by factor
s, Amdahl gives `1 / ((1-p) + p/s)`. The table is a sensitivity calculation,
not a claim that today's retained PPO has exactly these fractions:

| Accelerated fraction p | 1.5× GEMMs | 2× GEMMs | Infinite GEMM speed |
| --- | ---: | ---: | ---: |
| 0.50 | 1.20× PPO | 1.33× PPO | 2.00× PPO |
| 0.65 | 1.28× PPO | 1.48× PPO | 2.86× PPO |
| 0.70 | 1.30× PPO | 1.54× PPO | 3.33× PPO |

A kernel-duration fraction from an instrumented CUDA graph trace is only a
proxy for p: host work, gaps, transfers and changed kernels matter. More
critically, prototype1 initially accelerates only a subset of GEMMs, so use
that subset's time, not the entire GEMM fraction. For example, if eligible
forward GEMMs occupy30% of update time, doubling them yields only1.18× PPO.
Recompute p from the retained trace and verify with uninstrumented full-update
timers. A 1.5–2× dominant-kernel target is ambitious and technically motivated;
a corresponding 1.5–2× complete-PPO claim is not justified.
