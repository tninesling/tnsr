# Ada strict-F32 GEMM: bounded codegen and register-tile research

Research only. No production source, build, GPU workload, or benchmark was changed
by this investigation. Performance proposals below are unmeasured hypotheses.
The retained baseline is documented in
[the address-reconstruction report](../ppo-address-reconstruction.md); the
[seven rejected follow-ups](../ppo-followup-experiments.md) remain rejected.

## The actual compiled baseline changes the priorities

The current schedule uses a 32×32×32 block tile, 256 threads, and four accumulator
values per thread. The scalar PTX lowering assigns a lane four rows separated
by eight, sharing one B operand across those sums. Its source has an explicit
32-position K loop, but the profiling agent extracted the actual cached JIT
cubin and matched its embedded PTX byte-for-byte to the frozen baseline.

That evidence shows 128 FFMA instructions, 32 LDS.128 instructions and 32 scalar
LDS instructions in the hot mainloop body. The driver already contracts the
multiply/add pairs, unrolls the tile, and vectorizes A loads. Therefore:

- An optional FMA source mode is not evidence of a two-to-one arithmetic
  instruction saving; this baseline already executes FFMA.
- Source-level unrolling and counting scalar load expressions are inadequate
  models of the generated kernel.
- Register count alone does not identify the bottleneck. The retained hot
  kernels use 78 registers and 8 KiB shared memory without local spills, while
  reducing registers in the rejected rematerialization experiment slowed them.

The profiling agent also found two calls per K tile to an unsigned-64 integer
division helper containing approximately 66 static instructions. They implement division and remainder
by 32 for the lane's row/column decode. This late-generated expression bypasses
the scheduled-index optimizer. Hardware counters remain unavailable because
Nsight Compute reports ERR_NVGPUCTRPERM; the static evidence does not establish
whether bank conflicts, issue throughput or another latency dominates.

## First prototype: remove the late power-of-two divisions

This precedes an architectural tile change. In
`src/ptx/graph/matmul.rs::lower_register_tiled_scalar_matmul`, the code directly
passes `FloorDiv(linear_tid, block_tile.n)` and `Mod(linear_tid, block_tile.n)` to
`lower_expr`. In `src/ptx/graph.rs`, those branches emit unsigned division even
for a constant power of two, whereas explicit ShiftRight/BitAnd expressions have
dedicated instruction paths.

For the current validated N tile of 32, use `linear_tid >> 5` and
`linear_tid & 31`. Keep the existing fallback if a future tile width is not a
positive power of two. This is an exact unsigned identity for every possible
input, not a floating-point or workload-specific rewrite. It preserves tile
ownership, accumulation order, staging and epilogues.

The initial experiment should be confined to this direct row/column decode.
Compare the new actual JIT SASS against the exact baseline: confirm that the two
helper calls disappear, inspect register/spill changes and count instructions.
Then compare complete output bits and cancellation/nonfinite tests before
resident-GEMM and complete-PPO timing. Removing two calls to the approximately
66-static-instruction helper per K tile is a strong static lead, not a promised
speedup. Branches may execute only a subset of the helper; this is not a claim
of 132 dynamically executed instructions.

## Second prototype: per-axis exact-tile bounds elision

The staging IR always emits predicated global-to-shared loads. The graph builder
already knows M/N/K, block extents, and the K-tile loop's range. Under the normal
synchronous scalar schedule, it can prove some predicates redundant:

| Operand / coordinate | Proof allowing its logical bound check to disappear |
| --- | --- |
| A row | `M % block_m == 0` |
| A column | `K % block_k == 0`, and this is the current synchronous K tile |
| B row | Same current-tile K proof |
| B column | `N % block_n == 0` |
| Output row / column | The corresponding M / N divisibility proof |

For 256×683×4096 with 32×32×32 tiles, both A checks and B's K check are
redundant; B's N-tail check must remain. For 683×256×4096, keep A's M-tail check,
remove A's K check, and remove both B checks. For 129×257×129, none of these
full-tile proofs holds.

Apply the proof to logical matrix bounds only. It does not authorize dropping
an independent source predicate, a gather-index check, or a source-map validity
check. Keep zero-fill initialization whenever any load predicate remains;
fully proven loads may initialize their destination directly. Register-array
output rounds must use the same per-axis proof as the ordinary output store.

Important caveat: the two-stage path prefetches `k_tile + 1`, including a final
out-of-range prefetch. K divisibility does **not** prove that prefetch valid.
The first bounded experiment should leave asynchronous staging unchanged.
Grouped-M traversal is also a permutation of tile IDs, not permission to assume
an arbitrary block index is valid.

Affected code is the staging/output construction in `src/tile/graph.rs`, the
predicated load/store IR and builder descriptors in `src/tile/ir.rs` and
`builder.rs`, and their PTX lowering in `src/ptx/graph.rs`. Prefer explicit
per-axis proof metadata, defaulting to checked behavior, rather than inferring
validity from a few sampled indices. A generic interval pass is a later project.
The driver may already remove some duplicate checks, so inspect resulting SASS
before interpreting a smaller PTX listing as an improvement.

## Register-tile architecture: the 2×2 trap

The source's 4×1 tile loads four A values and one B value per K, while an outer
product over a 2×2 tile needs two of each. That appears to reduce five loads to
four. It is misleading for this compiled kernel.

If A continues to vectorize in groups of four K positions but B stays scalar,
a 32-position tile changes from approximately `32 vector A + 32 scalar B = 64`
shared-load instructions to `16 vector A + 64 scalar B = 80`. This is a static
count conditional on the same JIT vectorization, not a forecast of cycles.
A successful 2×2 architecture must address B loading and the emitted vector
width, not simply reduce the number of scalar source expressions.

For scalar accesses, a native 16-column lane arrangement puts two different A
rows in a warp. With an unpadded 32-word A row stride, the rows can address the
same bank at different words. But actual LDS.128 accesses are split into
transactions, and broadcasts complicate the count. It is incorrect to apply a
scalar whole-warp bank calculation to a vector instruction and label every
repeated bank a conflict. NVIDIA documents both broadcasts and per-instruction
ideal versus excessive wavefronts. [Bank rules](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks),
[wide-request triage](https://docs.nvidia.com/nsight-compute/ComputeTriage/#symptom-shared-memory-bank-conflicts-are-significant).

The profiling agent confirmed the alignment failure in the rejected blanket
padding experiment using matched cached cubins: contiguous region96 has
32 LDS.128 plus 32 scalar LDS, while stride33 has 160 scalar LDS and no LDS.128;
both retain 128 FFMA. A stride of 33 floats breaks 16-byte row alignment. This
is evidence of scalarization, not a complete dynamic explanation of the old
throughput regression. Provenance is retained by the profiling agent under
`/tmp/ppo-gemm-research/{contiguous,padded}-old-layout.{sass,cubin}`, linked to the
original per-role CUDA caches in
`results/ppo-staging-k-20261029/layout-screen-resident/m64-pair0/`.

Before changing thread ownership, force stride36 under the **existing 4×1
geometry**. Compare contiguous storage with A-only and B-only skew, selecting
the operand whose source is physically transposed. A 36-float leading stride
preserves 16-byte row alignment. For scalar transposed staging, its bank index
advances by four, reducing the nominal multiplicity from 32 distinct words in
one bank to four words per bank across eight banks. Check actual LDS widths,
registers, staging instructions and complete-update timing; no speedup is yet
measured. This isolates alignment-preserving skew from a new thread map.
The existing blanket `Padded` selector uses +1 for scalar layouts, so a +4
per-operand experiment must be explicit.

## Three concrete architecture candidates

### A. Fixed 32×32 tile, 2×2 thread tile, vectorization-aware B layout

Keep 256 threads and four accumulators. Use a logical 16×16 lane arrangement:
`lane_row = tid / 16`, `lane_col = tid % 16`. Accumulator `(i,j)` owns
`(lane_row + 16*i, lane_col + 16*j)`, for `i,j in {0,1}`.

The most interesting B variant is column-major shared storage with physical
leading stride 36 for K32. Each column's four consecutive K values is then
16-byte aligned. In the nominal eight-lane LDS.128 transaction model, vector
starts land at banks `4*lane mod 32`, distributing the 32 words over 32 banks.
By contrast, stride32 column-major storage collapses those vector starts onto
one four-bank group; stride33 breaks vector alignment. Keep A row-major K32 as
the initial control. Actual instruction widths and wavefront behavior must
still be established from the generated kernel.

The physical shared allocation becomes 8.5 KiB instead of 8 KiB. This can fit
without a large capacity increase, but does not guarantee unchanged occupancy.
Global-to-shared staging also matters: this B representation matches a
physically transposed B operand's contiguous K direction. It can create store
conflicts for a physically row-major B operand, so begin with an explicit forced
variant on transposed-B forward projections and retain the legacy kernel for
other orientations. Do not change automatic dispatch first.

Use the successful or rejected 4×1 stride36 result as the layout control before
combining any skew with the 2×2 ownership change.

### B. 32×64 block tile, 2×4 thread tile

Use the same 16×16 logical lane arrangement and eight accumulators. This trades
more accumulator/operand registers for better reuse. At K32, unpadded operands
use 12 KiB shared storage; the column-major-B stride36 variant uses 13 KiB.
Its source-level operand-load count is six values for eight outputs, versus five
for four in the baseline, but the actual vectorized load count must be checked.

The 683×256 gradient output launches 88 CTAs instead of 176; the 512×683 forward
output launches 176 instead of 352. This is a compromise between more reuse
and adequate grid parallelism. N=683 is a required tail case, not a reason to
benchmark aligned square matrices only. Direct epilogue stores now cover two
half-row segments per warp; measure store sectors before adding a shared-memory
exchange.

### C. 64×64 block tile, 4×4 thread tile, large-forward-only experiment

Sixteen accumulators provide substantially more register reuse. Operand storage
is 16 KiB unpadded, or 17 KiB with column-major-B stride36. This is the highest
register and instruction-cache risk of the three candidates.

It is inappropriate as a universal replacement: 683×256 produces only 44 CTAs,
versus 176 now, which can leave much of the GPU unoccupied. Start only on large
forward projections such as 4096×683×256 and 4096×768×256, where the grid remains
large. Keep long-K, small-output weight gradients on the baseline initially.
Do not introduce split-K: it would change the accumulation contract.

These designs follow CUTLASS's distinction between threadblock, warp and
thread-level outer-product tiling, and its warning that epilogue layout is a
separate optimization problem. They are not copied performance predictions.
[Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html),
[SIMT layout/padding implementation](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/default_mma_core_simt.h).

## Representation and implementation requirements

The current four-element array is not enough metadata to describe these maps.
Represent the logical lane arrangement and the per-lane output tile explicitly,
for example `(lane_rows,lane_cols)` and `(thread_rows,thread_cols)`. The existing
map is `(8,32) × (4,1)`; the proposed square map is `(16,16) × (2,2)`.

Share one accumulator-coordinate function between PTX math ownership and Tile
IR output extraction/epilogues. The present math decoder in
`src/ptx/graph/matmul.rs` and output-side `tile_thread_coordinates` in
`src/tile/graph.rs` independently encode the legacy map. Updating just the math
would silently associate accumulators with the wrong outputs.

Update `MatMulSchedule`, `MatMulAccumulatorLayout`/storage layout propagation,
`layout.rs`'s fixed four-element 32×32 validation, register-array allocation,
PTX accumulator indexing, and all epilogue/store rounds together. Preserve the
four-element storage representation where possible, but do not infer ownership
from array length. Include actual shared allocations when checking resources;
register-only scalar accumulators do not require a C shared-memory exchange.

Ada permits 48 warps and 64K 32-bit registers per SM, with up to 100 KiB shared
capacity. The actual carveout and allocation granularity still matter. A rough
register-only ceiling changes from three 256-thread CTAs at 78 registers to two
at 90; this is not measured occupancy. Use the existing driver's actual
register/local/shared/active-block queries for each compiled candidate.
[Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/).

ThunderKittens supplies useful ideas about reuse and tiled ownership, but its
current repository mainly targets Hopper/Blackwell and its architecture selector
does not name SM89. Do not transplant Hopper TMA/WGMMA examples into this Ada
SIMT path. [Repository](https://github.com/HazyResearch/ThunderKittens),
[architecture selector](https://github.com/HazyResearch/ThunderKittens/blob/main/include/kittens.cuh).

## Bounded evaluation matrix

1. For the decode fix, verify division-helper removal in SASS, complete output
   bits, cancellation and nonfinite behavior. Keep arithmetic policy unchanged.
2. For bounds elision, cover every combination of aligned/tail M,N,K, zero K,
   batch/broadcast views and grouped traversal; verify that any tail retains its
   check and zero fill. Keep asynchronous final-prefetch behavior unchanged.
3. For a new thread tile, prove a bijection over all CTA output coordinates on
   CPU, including fragment-slot extraction. Test all A/B transpose combinations,
   composed leading-batch reshapes, bias/ReLU epilogues, changed feeds and replay.
4. Required PPO cases include 683×256×4096, 256×683×4096, 4096×683×256,
   4096×256×683 and the K512 / M512 microbatch-8 equivalents. Preserve physical
   `[microbatch,64,width]` storage before flattening, as the existing probe does.
5. Controls include 129×257×129 tails, 128×256×128, large aligned 2048³, tall/wide
   and small-M cases. Re-run the real primary workloads too: gpt-small is only
   batch2×sequence16 at width32, AlexNet batch1 has 4096-wide FC layers, and LeNet
   batch2 is small. These should retain safe fallbacks rather than be forced
   onto a large tile merely to produce a favorable microbenchmark.
6. Rotate candidate order, use a steady warmed GPU, capture actual JIT resources
   and SASS, then qualify complete PPO updates with unchanged gates and matched
   process pairs. Treat memory counters, source operation counts and occupancy
   estimates as explanatory data, not acceptance criteria.

Recommended order: exact late decode strength reduction; exact-tile bounds
elision; an isolated 4×1 stride36 layout probe; only then one forced
vectorization-aware thread-tile prototype.
