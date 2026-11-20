# ThunderKittens and FlashAttention-2: techniques applicable to tnsr

Review of the downloaded papers against the working tree, 2026-10-31.
Research only: no production edits, builds, GPU runs, or new speed measurements.
Page references below use the PDF page number, which matches the printed page
for both downloaded preprints. Application proposals are inferences from the
papers and source, not results demonstrated by the papers for tnsr.

## Recommendation

The most bounded new experiment is **compute each exponential weight once in
the cooperative online scan**, instead of once per output lane. For tiled
BF16 prefill, the architectural direction is **retain score/probability and
output fragments in registers with query ownership designed around the MMA
layout**. For GEMM, persistent output-tile loops and deeper pipelines remain
possible but should follow the current layout/materialization baseline, not
repeat experiments already rejected.

These opportunities have different coverage. The online scan is inference
infrastructure, not a way to accelerate today's primitive PPO backward.
The existing [packed training proposal](ppo-fused-attention-training.md) finds
attention alone has much less coverage than encoder projections/FFNs. Its
7.70% attention share and 1.083x ideal elimination ceiling belong to an older
retained diagnostic trace, not the current mixed-BF16 worker.
The later [parallel prototypes](../ppo-parallel-prototypes.md) already implement
and qualify opt-in packed FFNs and the larger scalar tile; this review does not
recommend implementing those again. Attention still remains dense in that path.

## What the papers contribute

- [ThunderKittens](../references/thunderkittens-arxiv-2024.pdf), Sections 2-3,
  pp. 3-8: choose physical ownership/layouts and worker schedules together;
  reason about issue, shared-memory, compute and synchronization costs;
  overlap load/compute/store with a staged producer-consumer protocol.
  Section 3.3, p. 8 separates persistent output-tile work from L2-aware grid
  traversal. Appendix B.3, pp. 21-22 shows attention state retained as tiles;
  Appendix C, pp. 25-30 compares six actual shared-memory address maps.
- [FlashAttention-2](../references/flashattention-2-arxiv-2023.pdf), Section
  3.1, pp. 5-7: carry an unnormalized output numerator and divide once at the
  end; recompute probabilities in backward using saved row statistics.
  Section 3.2, pp. 7-8 assigns independent query blocks to CTAs in forward.
  Section 3.3/Figure 3, p. 9 assigns query slices to warps to avoid cross-warp
  partial-output reductions. The backward CTA scheme on p. 8 uses atomic
  additions for dQ; it does not provide our deterministic gradient contract.

Both are useful design sources, but TK's measured kernels here target H100,
and FA2's main experiments target A100/H100 with long sequences. Neither
establishes an Ada/S64 speedup or strict increasing-K equivalence.

## Already implemented: do not count these as new work

| Paper technique | Present implementation/evidence | Remaining distinction |
| --- | --- | --- |
| Unnormalized online numerator; final normalization | `src/tile/normalized_schedule.rs:83`, `:173`, `:308`; BF16 prefill also carries a raw numerator | Already has the central FA2 arithmetic optimization. |
| Query-axis CTA parallelism | `OnlineRegion::block_threads` in `src/tile/online_region.rs:36`; `lower_scan` uses one CTA per row for widths <=128 | General online inference is scalar, not a register-tiled MMA schedule. |
| Tiled tensor-core online inference | `src/nn/kv_cache/prefill.rs:71`, `:158`, `:170` | BF16 causal inference already uses 16/32-query tiles, K32 tiles and fixed key partitions. No custom training backward. |
| Shared layouts and alignment checks | `src/tile/shared_layout.rs:39`, `:75`, `:98`; `with_operand_orientations` in `src/tile/matmul_schedule.rs:238` | TK's byte-address swizzles are different maps, not a missing Boolean swizzle feature. |
| Multi-stage global-to-shared movement | `with_operand_pipeline`, `src/tile/matmul_schedule.rs:329`; `src/tile/graph.rs:786` | Two-stage copies, including independently async/scalar operands, already exist. |
| Register outer products | `src/ptx/graph/matmul.rs:223`; experimental selector `src/tile/matmul_schedule.rs:119` | Current source already has an opt-in 64x64x16 / 4x4-thread tile; the older research proposal predates it. |
| Cache-oriented block order | `with_grid_group_m`, `src/tile/matmul_schedule.rs:212`; `src/tile/graph.rs:1587` | Forced grouping already measured in [grid order](../matmul-grid-order.md); default stays one because BF16 regressed. |
| Repack awkward operands to enable efficient movement | [BF16 within20](../training-bf16-within20.md) | Current BF16 split-K already includes layout-copy cost and aligned 64x64x32 two-stage tiles. Compare future changes against this baseline. |

## Experiment 1: share exponential weights in cooperative online scans

**Source principle:** FA2 Section 3.1, p. 5 reduces scalar work surrounding
matmuls; TK Section 2.2, p. 4 treats exponential and issue throughput separately
from tensor FLOPs. The concrete redundancy below is our source finding.

In `src/tile/normalized_schedule.rs:254`, the first warp produces a 32-score
tile and its maximum. After publication, the loop at `:284` runs
`exp(score - running_max)` separately in every output lane. Every lane also
adds the same denominator contributions. For D=32, D=64 or D=128, this repeats
the weight calculation across all those outputs. It is not necessary for
independent numerator ownership.

First implement a forced alternative that retains the existing score publication
and max merge. After that barrier, the first warp computes one probability per
score lane, overwrites the shared score tile with those probabilities, and
publishes it with one additional block barrier. Consumers load weights and keep
the same increasing-key denominator and numerator operations. This isolates
saved exponential work from denominator reassociation. It trades duplicated
work for an extra barrier: a win is not guaranteed for S64 or D32.

The weight expression must use **the new running maximum**, namely
`max(old_running_max, tile_max)`, with the same operand ordering and -infinity
handling as `merge`. Computing relative to the tile maximum and then multiplying
by a rescale introduces different rounding and is not an exact replacement.
All participating lanes must have initialized, matching carried maximum state.
If the first version wins, investigate computing weights before the existing
publication barrier using a proven broadcast of the exact reference. Today's
`warp_reduce` only establishes the full result in lane zero; simply using each
lane's partial reduction would be incorrect. `warp_all_reduce` uses a butterfly
tree (`src/ptx/graph/scalar.rs:119`), so replacing the tree also needs explicit
NaN/signed-zero comparison or a reference-preserving broadcast primitive.

Do not initially parallelize the denominator sum: that is a second numerical
change. Keep output widths >128 on the existing fallback. The shared allocation
can remain 33 f32 slots if scores are overwritten after their last use.

**Acceptance:** compare full output bits against the old Online schedule where
expressions/order are intended to match, plus the existing materialized
reference tolerances. Cover widths 1/16/32/64/128/129, key tails, leading masks,
empty partials, all-masked rows, finite extremes, signed zero, NaN/Inf and changed
live cache lengths. Reuse `tests/ptx_online.rs` and KV-cache tests. Inspect actual
JIT exponential/shared/barrier instructions and register use. Benchmark attention
and complete cached decoding, including short contexts and partial-merge scans.

## Experiment 2: avoid register/shared round trips in BF16 prefill

**Source:** TK Section 3.1, pp. 5-6 and attention listing, pp. 21-22;
FA2 Section 3.3/Figure 3, p. 9.

The BF16 prefill path currently writes QK fragments to shared memory
(`src/nn/kv_cache/prefill.rs:305`), performs scalar softmax, writes BF16 weights
to shared memory, then reloads them for PV. When running maxima change, it
round-trips the persistent PV accumulator through shared storage to rescale it
(`:428` and `:435`). This is concrete communication the papers suggest reducing.

Start with the smallest change: represent a row-vector scale applied directly
to a register fragment, with an explicit fragment-slot-to-logical-row map. Use
the same f32 multiply and retain current MMA iteration order. This could remove
the accumulator store/scale/reload exchange when alpha differs from one.
The existing `changed` test already skips that exchange when all alphas are one;
measure how often it executes before committing to the feature.

The larger follow-up is a warp-owned query tile: retain scores in registers,
perform row max/sum through ownership-aware shuffles, convert weights to BF16
register operands, and consume them in PV without a full shared probability
tile. This requires new mapped fragment arithmetic and possibly an explicit MMA
lowering beyond the current opaque WMMA loads. It is not achieved by removing
barriers or changing `warp_topology`: QK and PV ownership maps differ.

Keep BF16 conversion points, running f32 state, causal bounds, key-partition
merge, output rounding, and nonfinite behavior explicit. The current prefill
grid already supplies query and key-partition parallelism; do not add more key
splits as a presumed benefit. Test D32/64/128, chunk tails, long prefixes and
short prompts. Compare complete decoder latency and actual registers/spills,
not only the number of source-level shared operations.

## Experiment 3: evaluate richer layouts under the actual load instruction

**Source:** TK Section 3.1/Figure 4, pp. 5-6; Appendix C, pp. 25-30.

The current XOR map is `col ^ (row & xor_mask)`, in elements
(`src/tile/shared_layout.rs:75`). TK's 32/64/128-byte layouts XOR selected bits
of the byte address. They are not interchangeable. More importantly,
`supports_wmma` intentionally requires ordinary aligned 16-column fragments;
an arbitrary TK swizzle cannot be passed to a WMMA instruction that expects
regular stride. The address mapping must match both staging stores and actual
fragment loads. A richer map should be a validated descriptor with bounds,
bijection, inverse/consumer compatibility and contiguous-copy proofs, not raw
XOR constants in codegen.

For today's SIMT path, the bounded control remains alignment-preserving
per-operand padding (for example K16 stride20) and source-orientation-aware
staging. The recorded stride33 experiment lost vector shared loads; scalar
bank counting alone is insufficient. For BF16 WMMA, evaluate existing padded
and fragment-preserving swizzled layouts first against the latest repacked
baseline. More aggressive byte swizzles are an architectural experiment only
with a compatible explicit load path. Do not claim bank conflicts are the
current bottleneck while hardware counters remain unavailable.

## Later experiments and items to defer

1. **Persistent output-tile GEMM loops** (TK Section 3.3, p. 8): use a fixed CTA
   grid that repeatedly owns distinct complete output tiles. This preserves
   each output's increasing-K order and could reduce per-CTA setup or reuse
   stage buffers for the next output. It is distinct from host CUDA graph
   replay and existing grouped traversal. Include ragged tiles, batches,
   epilogues, replay signatures and grid-wave effects. It does not create
   missing parallelism for a small output grid. Prioritize large-output,
   short-K cases only after current source profiling identifies setup costs.
2. **Three-stage rings** (TK Section 3.2/Table 1, p. 7): current Tile IR has
   exactly two stage pointers and an unparameterized wait that completes
   committed copies (`src/tile/ir.rs:93`). A deeper pipeline needs stage
   lifetime and wait-group semantics, plus publication/retirement proofs.
   Do not just set `pipeline_stages=3`. Existing strict-F32 async toggling was
   rejected; current BF16 hybrid and repacked schedules are already accepted.
   Add depth only with evidence that added overlap outweighs shared capacity,
   occupancy and startup/tail costs.
3. **Memory-linear attention training** (FA2 Algorithms 1-2, pp. 6-7): existing
   `OnlineRegion` has one output and no backward/saved-state contract.
   `src/ptx/executor.rs:717` only enables the recognizer under Online policy;
   external normalization consumers keep primitive backward materialized.
   Joint forward/backward regions and deterministic gradient owners are
   prerequisites. The paper's dQ atomics cannot be adopted under deterministic
   execution. Saving only logsumexp and substituting `row_sum(dO*O)` also change
   floating-point algebra. Use the existing packed-training proposal as the
   project plan; avoid presenting it as a small inference-switch change.

Hopper TMA, WGMMA and dynamic register redistribution used in TK's listings
cannot be transplanted to this SM89 path. Borrow the scheduling principles
through the available cp.async/WMMA/SIMT machinery. Neither higher occupancy,
fewer registers, fewer kernels, nor fewer logical bytes is by itself an
acceptance criterion; require numerical qualification and matched end-to-end
measurements on the current baseline.
