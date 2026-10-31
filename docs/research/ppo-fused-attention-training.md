# PPO fused attention training and packed encoder research

Research only, 2026-10-30. No GPU execution, builds, production edits, or new performance measurements. Standalone FlashAttention is not large enough here: the attention-core group is 7.70% of retained GPU kernel time, giving an ideal elimination bound of 1.083x in that trace. The stronger architectural candidate is **packed encoder training**, retaining 2,253 observable query rows out of 4,096 in the real fixture, with a joint attention forward/backward region as an enabling component.

## What the actual model computes

`examples/forty_five/mod.rs:310` implements six noncausal transformer blocks: width 256, eight heads, 64 token slots, head width 32, SwiGLU hidden width 683. Each block computes QKV projection, QK-transpose, scale, additive key mask, stable softmax, probability-times-V, output projection, then tokenwise normalization and three FFN projections. There is no attention dropout. The shared critic reuses this encoder.

QKV is split with `narrow`, reshape and permute; the retained plan materializes the three narrow outputs and later the context view consumed by the flattened output projection. A fused region should read the interleaved QKV projection through those maps and write the context in projection-friendly token-major order. Merely fusing the softmax misses this layout work.

The attention key mask is **finite `f32::MIN`**, not negative infinity (`Observations::inputs`, around line 540). At least one key is safe, including position zero for an entirely empty observation. Pooling instead uses negative infinity. Position embeddings preserve original slot identities; card-conditioned models additionally assign position embedding index 10 to all slots 10–14. Packing must retain that exact mapping, rather than renumber tokens.

## Existing attention support does not cover this training graph

* PPO uses `with_gradients(loss_node)` (`examples/forty_five/ppo.rs:273`), which selects primitive differentiation. The existing optional `GradientOptions::compact_softmax` is disabled. Its own documentation states that it changes floating-point algebra and still materializes probabilities.
* Matmul differentiation creates both transposed contractions (`src/graph/mod.rs:728`). Softmax's primitive backward keeps forward normalization intermediates alive.
* `PtxExecutor` invokes `fuse_online` only under `PtxReductionMode::Online` (`src/ptx/executor.rs:701`). PPO selects Strict or DeterministicTree, with the retained benchmark using DeterministicTree; it never selects Online.
* `src/ptx/plan/online.rs` refuses a region whose interior has external consumers. The training test `tests/ptx_online.rs:155` explicitly asserts **zero** online regions because backward needs forward intermediates. Changing the reduction-mode switch alone cannot fix this.
* Existing `OnlineRegion` has one output, no saved statistics or custom backward, and lowers to scalar dot-product score production with cooperative streaming normalization (`src/tile/normalized_schedule.rs`). This is inference-style scheduling fusion, not a joint training algorithm.
* `src/nn/kv_cache/prefill.rs` already implements tiled **causal BF16 inference** with F32 running statistics/numerators and BF16 MMA operands/output. Its partitioning is useful reference code, but its precision, causal KV interface and absent backward prevent reuse as strict-F32 PPO training.

The missing pieces are joint forward/backward selection, saved-state/liveness support, deterministic gradient ownership, and layout-aware lowering—not simply another pointwise pattern.

## Measured opportunity bound

I parsed the retained 4,005-node/1,982-kernel microbatch-64 plan at sibling `results/ppo-dense-canonical-20261030/profiles/m64-baseline/program/micro-64-clip-3e4ccccd.plan`, and matched its kernel names against the retained address-reconstruction trace `results/ppo-index-reconstruct-20261030/profiles/reconstruct-kernels.csv`.

| Plan/profile group | Plan kernels | Aggregate trace GPU time | Share |
|---|---:|---:|---:|
| All attention QK/PV and their four backward contractions | 36 | 20.327 ms | 3.00% |
| All score-square output kernels, overlapping QK/dP above | 96 | 27.479 ms | 4.06% |
| Union of those groups plus 18 attention row reductions | 138 | 52.118 ms | 7.70% |
| Whole kernel summary | — | 676.865 ms | 100% |

The four attention GEMM names are `matmul_region_2_4797`, `matmul_region_3_4798`, `matmul_region_102_4897`, and `matmul_region_104_4899`. The last is logical M32/N64/K64. The square group is shape `[64,8,64,64]`; the additional reductions have shape `[64,8,64,1]`. Those selected names are not reused for other output shapes in this plan. Times aggregate multiple updates/warmups in one diagnostic trace; they are neither one-update latency nor a fresh matched benchmark.

One score matrix is 8 MiB at microbatch 64, or 1 MiB at microbatch 8. The 96 square-output kernel steps account for **768 MiB of cumulative logical output extents** at microbatch 64. That is not peak allocation, measured DRAM traffic, or 96 simultaneously live buffers. Six forward probability matrices alone are 48 MiB. The larger seven-gigabyte logical-materialization total includes much more than attention.

Even removing the entire selected attention-core group for free would give only approximately `1/(1-0.077)=1.083x` GPU kernel-time speedup in this trace. Extra layout improvements may help, and the bound is not an end-to-end latency prediction. Nevertheless, attention alone does not explain a plausible 2x full-model improvement. The longer-sequence speedups reported by FlashAttention cannot be transferred to this S=64 workload.

## Higher leverage: pack the whole encoder, retaining observable padded queries

The real microbatch-64 fixture contains 2,173 nonzero tokens out of 4,096 slots (53.05%); lengths range 18–52. Padding occurs inside sequences, followed by valid trailing slots, so prefix trimming is invalid. Summed valid-key/valid-query pairs are 80,837 versus 262,144 padded pairs, before the eight-head multiplier.

However, dropping every zero-token query is also invalid:

* The CLS feature explicitly reads original slot zero.
* The card-conditioned action heads read **all original slots 10–14**, before action-loss selection. These include 80 zero-token slots across this fixture. Their initially masked-zero states become nonzero after attention/residual/FFN, so they cannot simply be replaced with zeros.
* Policy/value pooling masks other padded slots, but its projections and multiplies can still propagate nonfinite values. A finite-value argument is not an unconditional IEEE equivalence proof.

A conservative query keep-set is valid-token slots plus CLS and card slots. It contains **2,253 rows (55.00%)** here. Keep key/value eligibility separate: those 80 observable padded queries remain masked as keys. This gives 84,437 query/key pairs (32.21% of padded pairs). The counts are workload opportunities, not measured speedups or permission to specialize to this fixture.

Under the intended finite-value masking behavior, tokenwise layer norm, linear, SwiGLU and residual operations can run on the kept rows throughout all six blocks, not just within attention. That targets the dominant projection/FFN work. Pack once at encoder entry and retain packed storage until the few final gathers/pools; repacking around each matmul would squander the opportunity.

For correctness, prove that omitted slots cannot reach any observable output through later key participation, pooling, gathers, or another graph consumer. The additive finite mask does not guarantee a zero probability for arbitrary extreme/nonfinite scores; such cases need the original path or an explicitly restricted numerical policy. Weight/embedding-gradient reductions also change when zero-contribution rows disappear, even with stable original row ordering. Preserve the existing path and qualification gates.

## Recommended architectural prototype

Implement an opt-in **packed encoder training region with separate query/key maps**, initially homogeneous F32, noncausal, no dropout, S<=64 and D=32. Treat this as reusable compiler/runtime support, not a fixture-specific benchmark shortcut.

1. Recognize the attention expression before primitive autodiff, or jointly match its forward/backward subgraph afterward. Introduce an internal region descriptor with QKV view maps, scale/mask semantics, explicit saved outputs and dQ/dK/dV results. Decline when probabilities or normalization interiors escape to arbitrary requested outputs. Existing `OnlineRegion` cannot represent this contract.
2. Add stable row packing metadata: original sample/slot, cumulative query/key offsets, and separate key eligibility. Preserve original embedding IDs, including the card-position override. Keep downstream original-slot gathers exact, and pool over the original safe-key set. Gradients invert gathers; shared parameter reductions need deterministic ownership.
3. Keep flattened tokenwise GEMMs/FFN packed through the encoder. Use device-resident counts/maps and fixed-capacity storage so CUDA graph replay does not require host synchronization on the token count. A fixed launch grid with inactive tiles returning, or a small bounded shape-bucket cache, must be measured against padding waste. Include scan/pack/unpack and metadata updates in timed calls.
4. First fuse attention using existing F32 arithmetic and reduction topology as closely as possible, retaining probabilities if needed to preserve primitive backward. At S64, a full probability tile is only 16 KiB per head. This separates layout/packing benefit from an arithmetic rewrite. Preserve the old Strict/DeterministicTree behavior as fallback; verify generated PTX/SASS because today's `StrictF32` name does not guarantee no FMA contraction (see `f32-fma-contract.md`). Packing itself changes reduction placement and needs an explicit numerical-policy opt-in if strict order cannot be retained.
5. Then add an independently labeled memory-linear training mode: forward saves output plus row max/denominator; backward recomputes probabilities and computes dV, dP, dS, dQ and dK in on-chip tiles. Use separate deterministic owners for dQ and dK/dV, or ordered partial buffers followed by fixed reductions; avoid unordered floating-point atomics. Start with small tiles below ordinary shared-memory limits and measured register pressure. A rowwise two-pass max/sum forward is the conservative F32 starting point; online rescaling, compact softmax backward, and `dot(dO,O)` substitution are separate floating-point changes.

The familiar derivative is `dS=P*(dP-row_sum(P*dP))`, with `dP=dO*V^T`, `dQ=scale*dS*K`, `dK=scale*dS^T*Q`, and `dV=P^T*dO`. This is a useful custom VJP but does **not** preserve the current primitive derivative's rounding/overflow/nonfinite behavior. Saving only log-sum-exp can also introduce extra log/exp rounding; save max and denominator first. The old arithmetic policy remains available and historical benchmark labels stay unchanged.

This is a substantial compiler project: region/VJP representation (`src/graph`, `src/ptx/plan`), saved-state memory planning/execution (`src/ptx/executor.rs`), mapped packed contractions and Tile IR schedules (`src/tile`, `src/ptx/graph`), and a narrow example opt-in. A dense fused-attention kernel is an intermediate milestone, not the final high-leverage deliverable.

## Decisive experiment

Use three frozen variants against the retained compiler: dense fused F32 attention; packed encoder with conservative backward; packed encoder plus explicitly labeled recomputation/custom-VJP mode. First qualify masks with interior holes, all-padding safe-token handling, original position IDs, padded card-slot reads, escaping outputs, finite extremes and NaN/Inf fallback. Validate all parameter gradients, not only attention inputs. Compare microbatches 64 and 8 on the unchanged real 64-transition update and additional padding densities; dense/no-padding inputs should expose packing overhead.

Keep the existing exact-prestate CPU derivative and independent float64 Adam checks at both checkpoint boundaries, original tolerances, clip/optimizer behavior, replay policy, process pairing, cold/warm exclusions and summed-inner-time throughput estimator. The PPO gates are metrics `3e-5+5e-4*abs(ref)`, gradients `3e-6+3e-3*abs(ref)`, weights `2e-5+2e-4*abs(ref)`, first moment `1e-9+1e-3*abs(ref)`, variance `1e-12+1e-3*abs(ref)` (sibling `docs/ppo.md`). No tolerance relaxation to admit custom backward.

Report full-update latency/throughput, kernel count, actual packed rows/pairs, pack overhead, replay fallbacks, peak allocated memory separately from logical materialization, and stage/profile attribution. Proceed to a fresh competitor campaign only after the full model passes and matched end-to-end gains exceed noise. No speedup is established by this note.

## Primary sources and what to borrow

* [FlashAttention paper, algorithm and backward appendix](https://arxiv.org/html/2205.14135v2): tile attention on chip and recompute probabilities in backward from saved normalization statistics. It also derives the output-dot-gradient shortcut. These are algorithmic inspirations, not bitwise-F32 guarantees.
* [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691): query-block parallelism, warp work partitioning and reduced communication are useful scheduling ideas; its A100 measurements do not predict S64 Ada performance.
* [Official CUDA implementation requirements](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support): FA2 supports Ada, including forward/backward head dimensions covering32, but CUDA datatypes are FP16/BF16. Using it directly would be an explicit mixed-precision experiment. FA3's Hopper requirements do not fit RTX4080.
* [Official unpadding/autograd source](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py) and [variable-length interface](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py): retain gather indices and cumulative sequence offsets and invert the gather in backward. PPO additionally needs distinct observable-query and eligible-key sets.
* [NVIDIA Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html): SM89 has a 64K-register file and 100 KiB shared memory per SM; static shared allocation remains limited to 48 KiB without the larger dynamic-memory opt-in. Shared/register budgets and occupancy need qualification on the actual RTX 4080.
