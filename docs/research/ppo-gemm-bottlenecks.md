# PPO GEMM bottleneck research

## Strongest next experiment: remove direct divide-by-32 helpers

The retained worker's **actual driver-JIT machine code** calls a 66-instruction
unsigned-64 division helper twice per K tile just to derive thread row/column.
This is a concrete code-generation defect to test before another broad
register-tile redesign. Its performance benefit is **not measured yet**.

In `lower_register_tiled_scalar_matmul` (`src/ptx/graph/matmul.rs`), direct
`lower_expr` calls lower `thread_linear / 32` and `thread_linear % 32` after
the Tile IR index-normalization pass. The frozen PTX for
`matmul_region_96_4891` contains:

```text
div.u64 %rd169, %rd168, 32;
div.u64 %rd174, %rd173, 32;
mul.lo.u64 %rd175, %rd174, 32;
sub.u64 %rd176, %rd173, %rd175;
```

Actual SASS loads divisor `UR18 = 0x20` at offset `0x1260`, calls helper
`0x28a0` at `0x1330` and `0x13d0`, and uses its quotient to construct shared
addresses. The helper occupies `0x28a0–0x2cb0` inclusive: 66 instructions,
including reciprocal/conversion and multiword integer arithmetic. These calls
are inside the outer K-tile loop, immediately after its staging barrier.

For unsigned u64, replacing the quotient with `shr.u64 ..., 5` and remainder
with `and.b64 ..., 31` preserves all input bits, including u64 boundary values.
The existing scheduled normalizer already knows this identity; the direct
lowering bypasses it. Start locally in this register-matmul lowering, or add
constant-power-of-two strength reduction centrally in `lower_expr` while
retaining general and wide-index fallbacks. NVIDIA documents this arithmetic
identity, but its CUDA-C++ optimization guidance is not proof that a particular
PTX driver JIT applied it; the actual code here did not.
[NVIDIA integer-arithmetic guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#integer-arithmetic)

Acceptance sequence: boundary/cancellation tests, inspect new SASS to confirm
both helper calls disappear, compare complete probe output bits, then run the
unchanged PPO derivative/Adam qualification and matched A/B timings at both
microbatch sizes. Do not infer a speedup from instruction counts alone.

## Actual-JIT provenance

The decoded cache entry is:

```text
tnsr-benchmarks/results/ppo-index-reconstruct-20261030/
  m64-pair0/cache/candidate/cuda/7/3/57cfcdbdb41810
```

Its embedded PTX matches **all 926,856 bytes** of the retained baseline's
`results/ppo-dense-canonical-20261030/profiles/m64-baseline/program/`
`micro-64-clip-3e4ccccd.ptx`, starting at cache byte 58. The CUDA ELF begins at
926,916 and has 1,650,784 bytes. Extract the entire remainder, including the
program-header table at its end. `cuobjdump` decodes this ELF as SM89 code.

The qualified retained report is
`/home/taylor/code/tnsr-benchmarks/results/ppo-index-reconstruct-20261030/report.json`;
its candidate binary is the adjacent `ppo-candidate`, SHA256
`17a2a3d0ccc1c7493268859f5788c64467ad4bfc7357e12bbe6915976e6902fa`.
The exact matched archived PTX is
`/home/taylor/code/tnsr-benchmarks/results/ppo-dense-canonical-20261030/profiles/m64-baseline/program/micro-64-clip-3e4ccccd.ptx`,
SHA256 `69e4f6361c24a9b0a7e25e27d9f9f59f689f1ca437eeb7b1b259e546b3cb2242`.

This is cached machine code produced by the driver for that qualified retained
worker, **not an offline ptxas proxy**. The companion JSON also records cache
and ELF SHA256 values and all artifact paths.

## What the code already does

Representative regions 1, 5, 50, 94 and 96 each contain 128 `FFMA`, zero
`FMUL`, and zero `FADD` instructions in their decoded function. The current
PTX `mul.f32`/`add.f32` pairs have therefore already contracted to FMA on
these real kernels. An FMA mode cannot be assumed to unlock a missing FFMA
instruction here; a guaranteed separately rounded mode is an arithmetic-contract
question, addressed separately in [the F32/FMA audit](f32-fma-contract.md).

The 32-position inner K tile is already fully unrolled/vectorized by the JIT.
For region 96, SASS has 32 `LDS.128`, 32 scalar `LDS`, and 128 `FFMA` in one
static compute body. This explains why additional four-way PTX unrolling and
pointer-increment variants could fail to create useful machine-code savings.

Region 96 static segments are:

| Segment | SASS offsets, end exclusive | Static instructions |
| --- | --- | ---: |
| Staging/address/bounds | `0x550–0x1320` | 221 |
| Shared-coordinate setup, including two helper calls | `0x1330–0x1450` | 18 |
| Compute body | `0x1450–0x2060` | 193 |
| Division helper, each invocation | `0x28a0–0x2cc0` | 66 |

The staging segment has eight `LDG.E`, eight `STS`, and 24 `ISETP.GE`
instructions (12 low/high comparison pairs), plus address/control arithmetic.
The compute segment has 128 FFMA, 64 shared-load instructions and one predicate
instruction. These are static instruction counts, not cycle weights, executed
instruction-counter measurements, or a measured stall breakdown.

For 256×683×4096 with 32×32×32 tiles, A's M/K axes and B's K axis are fully
tile-aligned. Their per-axis bounds checks are candidates for static elimination;
B's N tail still needs protection. This is a second, separately measurable
experiment. Do not remove tail checks globally or infer byte-address safety
from sampled coordinates.

## Why blanket padding was a poor experiment

Actual cache code from the earlier matched layout screen confirms a specific
tradeoff, rather than a generic conclusion that padding cannot help:

| Old layout-screen region 96 | Registers | Shared bytes | `LDS.128` | Scalar `LDS` | FFMA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Contiguous stride 32 | 72 | 8,192 | 32 | 32 | 128 |
| Padded stride 33 | 64 | 8,448 | 0 | 160 | 128 |

Padding reduced registers but destroyed vector shared loads. These cached
kernels are from `results/ppo-staging-k-20261029/layout-screen-resident/`
`m64-pair0/cache/{contiguous,padded}/cuda/`, before the retained address rewrite.
They are useful matched static evidence, not a new performance ranking.

For transposed staging into a 32-word row stride, a full warp writes distinct
rows at one fixed column: distinct words collide in one shared bank. Default
compute reads are different: A uses broadcast/vector reads and B uses contiguous
scalar reads. Padding to 33 breaks 16-byte A-row alignment and demonstrably
scalarized the old JIT code. A stride of 36 is worth testing because it preserves
16-byte row alignment while reducing the transposed-store bank multiplicity
from 32 to 4; whether this improves complete updates is unknown. The tile-design
agent is investigating layout-aware alternatives. NVIDIA explains the distinction
between distinct-word conflicts and same-word broadcasts in its
[shared-memory profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

## Measured kernel priority, and unavailable counters

Re-analysis of the retained Nsight Systems trace, restricted to graph-node
kernels, assigns **61.2% of model-graph kernel time to GEMMs**. The largest
groups include region 96 (10.0%, 256×683×4096), region 5 (7.9%,
4096×683×256), and region 97 (5.9%, 4096×256×683). These percentages exclude
optimizer kernels outside the graph and host time. The replay summary records
timings, shapes, register/shared/local storage and launch dimensions. They are
diagnostic trace measurements, not replacement ranking timings.

Nsight Compute 2025.2.1 is installed, but an actual bounded profile failed with
`ERR_NVGPUCTRPERM`. The driver reports `RmProfilingAdminOnly: 1`. No system or
driver permissions were changed. The worker completed and exited, and the GPU
lock was released. Timing printed under this failed instrumentation is not a
usable benchmark.

Consequently issue utilization, eligible warps, achieved occupancy, shared-bank
conflicts, DRAM throughput and stall breakdown remain **unavailable**, represented
as null in the JSON. Registers and static code demonstrate opportunities; they
do not establish that issue, occupancy, banks or DRAM is the limiting resource.
The intended counter sections remain SchedulerStats, WarpStateStats and
MemoryWorkloadAnalysis when authorized counter access becomes available.
[NVIDIA counter-permission explanation](https://developer.nvidia.com/ERR_NVGPUCTRPERM)

## Artifacts

The [compact evidence](ppo-gemm-bottlenecks.json) contains hashes, static counts,
trace summaries and explicit missing-counter reasons. Raw extraction manifests,
cached ELF copies, SASS listings, segment/opcode summaries and the failed ncu
stdout/stderr are under `/tmp/ppo-gemm-research/`. No production or benchmark
source changed, no evidence was deleted, and no GPU work remains active.
