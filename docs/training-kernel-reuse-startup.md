# Reuse identical PTX entries to reduce cold training startup

The shared dense training graph introduced in #122 emits 1,982 kernel-producing
plan steps. Only 383 have distinct PTX function bodies after normalizing the
entry symbol. Emitting each body once reduces the microbatch-64 module from
5,939,724 to 1,373,068 bytes. Plan steps retain their own buffer bindings,
allocations and launch configurations; only their function lookup is shared.
The comparison includes the complete emitted ABI, declarations and instructions.
Different indexing, contraction sizes, arithmetic and synchronization therefore
remain distinct. This applies to standalone, pointwise, reduction and matmul
entries. The in-memory compilation signature advances to 19.

Profiling isolates CUDA module JIT as the main cold-start cost: approximately
12.9 seconds of a 15.1-second PTX compilation before entry reuse. Replay buffer
allocation takes about 1 ms and capture about 12 ms in that diagnostic run.
The old PPO `compile_cache_seconds` measures model/autograd setup and does not
include the later PTX compilation inside executor warmup. Its previous
"warmup/capture" bucket therefore included module JIT. The worker now reports
PTX compilation and module-load times, logical kernel steps, unique entries and
PTX bytes separately when `TNSR_PROFILE_PPO=1`.

## Controlled cache measurements

Each baseline/candidate trial starts a fresh worker with a private, initially
empty `CUDA_CACHE_PATH`. A second worker reuses that same private cache. Both
workers use the same initial archive, fixed 64-transition batch, TF32x3,
deterministic tree reductions and CUDA Graph replay. Existing GPU workloads and
the global CUDA cache remain untouched. Three trials cover microbatch 64 and one
covers microbatch 8; each worker performs three updates. The table reports the
median worker-reported first-update total, excluding initial model loading.

| Microbatch / cache | Merged #122 | Entry reuse |
| --- | ---: | ---: |
| 64 / cold | 16.444 s | 6.230 s |
| 64 / warm restart | 2.933 s | 2.897 s |
| 8 / cold | 16.713 s | 6.218 s |
| 8 / warm restart | 2.703 s | 2.757 s |

Cold startup improves about 62%. The microbatch-64 candidate module load takes
3.06–3.29 seconds cold and about 8 ms warm. Every update has one active replay
shape and zero fallbacks. This fixes the cold JIT regression; steady-state GPU
work and launch counts are unchanged by entry reuse.

The earlier 2.92→16.51-second observation compared uncontrolled cache histories.
It does not establish an intrinsic 13.6-second capture regression. The controlled
experiment above separates cold and warm behavior. The original observation is
retained in the dense-contractions report for provenance.

Regression tests check reuse of identical standalone entries, separation of
different operations and contraction sizes, independent bindings, and changed
feeds without recompilation. Full measurements, warmups and executable hashes
are in [kernel-reuse-startup.json](benchmarks/training-parity/kernel-reuse-startup.json).
