# PPO training dataflow: remove token work, then fuse training regions

Research only; no compiler changes, builds, or new GPU runs. The largest identified
opportunity is to keep only observable token rows through the six encoder FFNs.
The second is a fused stable SwiGLU forward/backward region that removes the
primitive derivative scaffold. Standalone forward LayerNorm fusion is useful,
but its measured coverage is too small to lead this effort.

## Evidence and attribution

The retained address-reconstruction compiler remains the baseline. Source:
`examples/forty_five/mod.rs`, `examples/forty_five/ppo.rs`,
`src/nn.rs`, `src/nn/activations.rs`, `src/ptx/plan/fusion.rs`,
`src/ptx/resident/`, and `src/ptx/executor/replay.rs`.

Artifacts in the sibling `tnsr-benchmarks` project:

- Retained worker: `results/ppo-index-reconstruct-20261030/ppo-candidate`,
  SHA256 `17a2a3d0ccc1c7493268859f5788c64467ad4bfc7357e12bbe6915976e6902fa`.
- Nsight Systems report:
  `results/ppo-index-reconstruct-20261030/profiles/reconstruct-nodes.nsys-rep`.
  The already-exported SQLite is `/tmp/ppo-registers/reconstruct-nodes.sqlite`.
- Matching baseline plan:
  `results/ppo-dense-canonical-20261030/profiles/m64-baseline/program/micro-64-clip-3e4ccccd.plan`.
  Its PTX is the exact retained cache match documented in
  [the GEMM research note](ppo-gemm-bottlenecks.md).

I joined `CUPTI_ACTIVITY_KIND_KERNEL.demangledName` to `StringIds`, selected
nonzero `graphNodeId`, grouped by graph node, and ordered by first start. All
1,921 resulting kernel names match the physical plan in order, with **zero
mismatches**, after excluding its 61 `contiguous_view` copy actions. Every node
appears in 12 graph replays. Thus shared kernel names can be attributed to actual
physical occurrences, rather than assigning all uses of a generic multiply to
one operation. Per-step diagnostics are in
`/tmp/ppo-gemm-research/dataflow-step-times.json`.

| Scope | Mean summed kernel time per replay |
| --- | ---: |
| Entire captured model graph | 47.233 ms |
| MatMulRegion plus plain MatMul | 28.927 ms (61.24%) |
| Other graph kernels | 18.305 ms (38.76%) |
| Six FFNs: forward, activation-gradient and weight-gradient GEMMs | 16.996 ms |
| All non-GEMM steps with dimension 683 | 5.752 ms, 246 kernels |
| Thirteen width-256 encoder/output LayerNorm forward subgraphs | 1.069 ms, 143 kernels |

These are **diagnostic profiled GPU kernel sums**, not complete-update ranking
timers. They exclude optimizer kernels, graph gaps and host time. The 683 group
is a coverage envelope containing biases, activation/multiply derivatives and
bias-gradient reductions, not a claim that every kernel is fusible. Do not add
it to a LayerNorm estimate or another overlapping region without checking nodes.
Removing all non-GEMM work would yield an unrealistic model-graph ceiling of
1.633x. That makes small individual kernel optimizations a poor primary target.

## Prototype 1: packed observable rows through encoder FFNs

The actual encoder has six layers, width 256 and SwiGLU hidden width 683. Each
FFN computes two `[4096,256] @ [256,683]` projections, the stable SiLU gate,
their product, and a `[4096,683] @ [683,256]` down projection. Backward repeats
these large row domains for activation gradients and reduces over them for
weight gradients. Linear already flattens leading dimensions before GEMM;
that optimization is present.

The companion attention investigation found 2,173 valid positions in the
4,096-position fixture, but valid-mask-only packing is insufficient. Card heads
read encoded slots 10 through 14 regardless of key masking. Their additional
80 padded positions make the observable union **2,253 rows**, or 55.00% of the
dense domain. Preserve original position embeddings, original key eligibility,
CLS/pooling observations, and all five card slots. An observed padded query must
remain a query while remaining excluded as a key. See the companion attention
research for mask and gather semantics; this is an input-dependent opportunity,
not a universal sparsity assumption.

Start with an explicit packed tokenwise FFN boundary while retaining the current
dense attention implementation. Build a stable row map once per batch; gather
live residual rows, run normalization and the complete FFN on packed rows, and
scatter required rows back at the dense attention boundary. The next step can
carry packed storage through QKV/out projections and pooling; those are extra
coverage and are deliberately excluded from the following bound. Keep observed
padded rows alive across all six layers, since their evolving residual affects
their later queries. Unobserved masked rows have no path to a root only after
the model's actual gathers and masks have been checked.

The FFN GEMM attribution is:

| Regions | Role | Time |
| --- | --- | ---: |
| 5, 7 | Gate/up and down forward | 5.800 ms |
| 95, 97 | Activation-gradient GEMMs | 4.104 ms |
| 94, 96 | Weight-gradient GEMMs | 7.092 ms |

At the fixture's 44.995% removable rows, a linear-cost thought experiment gives:

- FFN GEMMs alone: **7.647 ms** removed.
- FFN GEMMs plus the entire 683-shaped non-GEMM envelope: **10.236 ms**
  removed, or a **1.277x** model-graph ceiling.
- If the weight-gradient reduction order must remain the original dense order,
  retain dense weight-gradient GEMMs and unpack their operands/gradients into
  original row slots. Forward plus activation-gradient GEMMs still cover
  9.904 ms, giving **4.456 ms** ideal savings before elementwise savings.

These are coverage-scaled hypotheses, not predicted measured speedups. GEMM
efficiency changes with row count; gather/scatter, index loads and fixed costs
reduce the payoff. Conversely, removing invalid rows from normalization and
additional projections could increase coverage. Padding to a small fixed set of
packed capacities trades some arithmetic for reusable CUDA graph captures.

**Scope:** explicit packed row descriptors and their inverse scatter; gather and
backward scatter; a model-level FFN prototype before any general e-graph rewrite;
fixed capture capacities or a bounded capacity cache; correct parameter-gradient
reduction. This is a substantial training dataflow change rather than reshape
copy elision. It changes how many rows the expensive GEMMs compute.

**Risks:** packing changes floating-point reduction trees for weight/bias
gradients; moving zero terms is mathematically harmless but not a bitwise proof.
Biases and residuals make padded hidden states nonzero. Pool masks, query/key
roles and unconditional card gathers differ. Variable batch occupancy cannot
silently reuse an incompatible captured grid. Do not drop parameter-gradient
contributions or loosen the existing original PPO/Adam tolerances.

**Decisive qualification:** compare original versus packed roots and every
parameter gradient on fixtures with partially padded card slots, different
lengths, an all-valid batch and both microbatch sizes. Use the existing exact
prestate CPU derivative/metric and independent float64 Adam checks. Then run
paired complete-update timing including packing and feed costs. Show a per-kernel
trace that demonstrates fewer FFN rows and separately report any retained dense
weight-gradient path. Reject an implementation that needs many shape captures or
packing overhead erases the multi-millisecond budget.

## Prototype 2: a stable SwiGLU training region

`nn::sigmoid` deliberately uses a sign/magnitude expression, two masked branches
and a shared denominator for numerical stability. `nn::silu` multiplies this by
its input. Primitive autodiff turns that into many independently materialized
683-wide arrays. The physical plan contains backward scaffold nodes interleaved
with forward work: first-layer nodes 3562 and 3568 already run before its up
projection. This is evidence of early derivative materialization, not evidence
that all operations in the forward plan interval are forward operations.

An explicit multi-output training region can consume gate/up preactivations and
produce SwiGLU output; save only the operands needed by the later backward
region. Backward consumes the incoming product gradient and emits gate/up
gradients together, recomputing the inexpensive stable sigmoid locally instead
of saving all primitive branch/quotient derivatives. Include projection biases
at the region boundaries where legal. A second reduction stage can form bias
gradients while preserving deterministic reduction semantics. This can operate
on either dense or packed rows.

The measured 5.752 ms / 246-kernel 683 envelope is the **maximum covered budget**,
not a promised saving. Removing half would save 2.876 ms and yield about 1.065x
on the measured model graph; this is a target for deciding whether the prototype
is worthwhile. A single forward gate fusion alone will cover much less.
One `[4096,683]` f32 temporary is 11,190,272 logical bytes; eliminating one store
and one later read is 22,380,544 logical bytes. These are tensor extents, not
measured DRAM traffic or peak allocation savings.

Current local fusion choices split/materialize pointwise, reduction and matmul
regions. They do not pair forward and backward into a training region with an
explicit saved-state/recompute contract. Generic multi-output pointwise fusion
may supply lowering machinery, but merely increasing a region size cap does not
solve escaping derivative intermediates.

**Scope and qualification:** first implement one stable SwiGLU training operator
or exact recognized subgraph with a dedicated VJP and two-output backward. Test
extreme finite inputs, zero, sign boundaries, masked branches, and parameter
gradients. Preserve the current stable formula before attempting a simplified
algebraic derivative: reassociation, division and contraction can alter values.
Run the original PPO derivative/Adam gates and paired complete-update timing;
inspect that the expected full-size intermediates actually disappear. Do not
claim a win from kernel count or logical-byte reduction alone.

## Supporting work and limits

LayerNorm currently broadcasts variance **before** `(variance + eps).log()` and
`exp(-0.5 * log)`. The first forward norm has eleven physical kernels, including
two row reductions and full-width log/exp. Its thirteen encoder/output instances
sum to only 1.069 ms. A rowwise forward region computing the same row statistic
once, plus a dedicated backward region retaining compact mean/inverse-standard-
deviation, is sensible supporting work. An rsqrt substitution or canonical
closed-form backward changes numerical order and needs separate qualification;
the full backward coverage is not quantified here.

Resident training already retains weights, gradients and Adam state on CUDA,
uses cached model graphs and replays on a private ordered stream. Replacing
per-kernel host dispatch or adding resident optimizer storage is not a missing
architecture. Capture-safe liveness planning and postponing derivative
materialization can lower memory pressure, but no new allocator or DRAM-stall
measurement supports a multi-millisecond time claim. The previous dense-alias
experiment removed 449,750,272 logical bytes yet left warm resident GPU memory
unchanged and did not establish a throughput win. See
[the rejected follow-up experiments](../ppo-followup-experiments.md).

Prioritize **row packing through FFNs**, then **SwiGLU training-region fusion**.
Their bounds overlap on the same activation kernels: savings must not be added
as independent contributions. Qualification should first establish each path
individually, then measure their composition.
