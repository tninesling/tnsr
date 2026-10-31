# PPO kernel research: findings and next experiments

## Architectural program: larger end-to-end opportunities

The later research pass targets work removal and a new GEMM architecture,
rather than treating the bounded codegen experiments below as the main project.

- **Packed encoder training:** the real fixture has 2,173 valid tokens, but
  preserving CLS and all card-slot reads requires 2,253 observable query rows
  out of 4,096. Preserve original position IDs and keep those extra padded
  queries excluded as keys. Packing FFNs alone targets a 22.748 ms coverage
  envelope within a 47.233 ms profiled model graph. Ideal linear scaling could
  remove 10.236 ms (1.277× graph-time ceiling), before packing costs, changed
  GEMM efficiency and arithmetic-policy constraints. Extend to projections and
  the rest of the encoder only after the first qualification milestone.
  [Training dataflow proposal](ppo-training-dataflow.md),
  [packed attention/encoder semantics](ppo-fused-attention-training.md),
  [verified fixture counts](ppo-padding-opportunity.json).
- **Hierarchical SIMT GEMM replacement:** investigate 16–32 accumulators per
  thread, register-fragment pipelining and co-designed shared layouts, versus
  the current four sums per thread. Start with a 64×64 CTA / 4×4 thread tile,
  then a 64×128 / 8×4 variant for large outputs. A 1.5–2× kernel target is an
  experimental objective, not a prediction; long-K gradient grids need a
  separate schedule because larger output tiles underfill the GPU.
  [GEMM replacement architecture](ada-gemm-replacement.md).
- **Training-region fusion:** a joint stable SwiGLU forward/backward region
  targets a 5.752 ms non-GEMM envelope. This overlaps the packed-FFN coverage;
  do not add its ceiling to the packing estimate. Forward LayerNorm and
  standalone attention have too little measured coverage to lead this effort.

Standalone attention-core fusion covers about 7.70% of one diagnostic trace;
even ideal elimination is only 1.083× in that trace. Fused attention is an
enabling part of a packed encoder, not evidence of a large PPO speedup by
itself. All larger proposals remain research: original numerical gates,
packing/metadata costs, replay behavior, and no-padding inputs must be checked
in complete updates. No production implementation was changed.

## Earlier bounded research

Three parallel investigations examined the retained compiler without changing
production or benchmark source. Their findings change the order of work.

## First prototype: late power-of-two integer lowering

The actual retained driver-JIT code calls a generic u64 division helper twice
per K tile for thread-row division/modulo by 32. These expressions are created
after scheduled-index normalization. Replace them with unsigned shift/mask
operations, retain non-power-of-two fallbacks, and verify the helper calls
disappear in new machine code. Test u64 boundary semantics, complete GEMM output
bits and numerical PPO qualification before matched timings. The speedup is
currently unknown; the helper's static size is not a dynamic instruction count.

## Subsequent bounded experiments

1. Elide per-axis staging bounds checks only for statically full tiles. Preserve
   tails, independent predicates and asynchronous final-prefetch checks.
2. Test alignment-preserving shared padding under the existing 4×1 ownership.
   Actual earlier stride-33 code lost vector loads: 32 `LDS.128` plus 32 scalar
   `LDS` became 160 scalar `LDS`, despite fewer registers. Stride 36 is a concrete,
   unmeasured way to preserve 16-byte alignment while reducing staging-bank
   multiplicity. Check generated code before changing thread ownership.
3. Only then investigate vectorization-aware 2×2 and larger register tiles.
   Fewer scalar source loads do not necessarily mean fewer issued shared loads.

## Arithmetic contract finding

Hot retained kernels already contain FFMA, and the driver already unrolls the
inner K tile. Adding those features explicitly is not a demonstrated source of
new performance. Unqualified PTX multiply/add permits contraction, so the
intended separately rounded behavior is not guaranteed. Existing tests miss
the distinction. Add the specified register-path rounding discriminator and
decide explicit arithmetic policies separately; preserve the frozen historical
baseline and its labels.

## Reports and evidence

- [Actual-JIT bottleneck evidence](ppo-gemm-bottlenecks.md), including the
  exact cached ELF/PTX/binary hashes and [compact data](ppo-gemm-bottlenecks.json).
- [F32/FMA contract audit](f32-fma-contract.md), including the focused test and
  compatibility-preserving policy proposal.
- [Ada register-tile research](ada-register-tiles.md), including static bank,
  vectorization, occupancy, bounds-proof and correctness analysis.

Nsight Compute hardware counters are blocked by `ERR_NVGPUCTRPERM`; no driver
or system permissions were changed. The actual cached machine code and existing
traces support the findings above, but do not establish a measured stall
breakdown. No new throughput gain is claimed.

Raw cubins, SASS listings, extraction manifests and profiler diagnostics are
preserved with SHA256 verification under
`/home/taylor/code/tnsr-benchmarks/results/ppo-gemm-research-20261030/`.
Original `/tmp/ppo-gemm-research/` paths in the individual reports identify where
the investigation ran; the preserved copies provide durable evidence.
