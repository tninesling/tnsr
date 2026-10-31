# Parallel PPO optimization prototypes

Three independent implementations explore different limits of the retained strict-F32 PPO baseline. All are opt-in; ordinary execution is unchanged.

## Packed tokenwise FFNs

`Settings.packed_ffn` retains original token order, nonpadding tokens, CLS, and card slots 10–14. The six FFNs gather only retained activation rows, execute normalization and gate/up/down projections, then restore the dense residual layout. Attention remains dense. Each omitted row gets a distinct constant-zero restore slot, avoiding many-to-one gradient accumulation into a sentinel. Runtime capacities are rounded to 128 rows (32 for small batches), and compilation/capture cache keys include capacity.

The real microbatch-64 fixture retains 2,304 of 4,096 allocated rows, including capacity padding. Microbatch 8 uses capacities 288 and 320. This changes weight-gradient reduction length and leaves skipped padded queries with different hidden states. The original finite attention mask is not a universal structural-zero proof for arbitrary extreme/nonfinite parameters; this remains an experimental, model-specific path. Complete original CPU derivative and independent Adam transition gates are required, without relaxed tolerances.

## Larger scalar register tile

`PtxExecutor::set_experimental_register_tiles(true)` and PPO `Settings.experimental_register_tiles` select a 64×64×16 CTA, 256 threads, and a 4×4 output microtile (16 accumulators/thread). Four A and four B operands feed each outer product. Outputs retain increasing-K accumulation order. Resource guards and shape dispatch preserve the existing schedule for smaller outputs, including the important weight gradients. Defaults remain unchanged; compilation signatures include the option and policy changes invalidate replay.

CPU schedule/layout/ownership tests and CUDA bitwise comparisons cover tails, batched/broadcast transpose views, explicit shared layouts, pipeline policies, epilogues, cancellation and nonfinite values. An isolated resident CUDA-graph probe on the RTX 4080 measured approximately 237 µs versus 179 µs for 4096×683×256, with 78 versus 90 registers/thread and no local-memory spills. This single measurement is diagnostic, not a full-model performance claim.

## Stable SwiGLU training kernels

`tile::swiglu::SwiGluTraining` extracts the existing primitive-autodiff scalar dependency closure. Forward consumes gate/up; backward consumes gate/up/incoming gradient and emits both input gradients. It preserves primitive masks, quotients, and gradient-add order rather than substituting an analytic sigmoid derivative. Backward recomputes scalar locals in registers. No automatic graph selection is enabled.

CPU tests compare 400 extreme/sign/Inf/NaN combinations with primitive autograd. The independent GPU probe qualifies all outputs against primitive GPU execution. At 4096×683 it reports 14/20 forward/backward registers with zero local-memory spills. The matched resident CUDA-graph comparison rejects the standalone two-kernel variant: ordinary generic fusion already emits one multi-output kernel when gate/up/incoming gradients are simultaneously available. At 4096×683 the baseline median is 31.74 µs versus 43.39 µs; at 512×683 it is 3.07 µs versus 4.22 µs. Captured-work throughput ratios are 0.705× and 0.728×. Both variants pass all output gates, but the explicit pair is slower. No automatic selection or PPO integration is added. This activation-only baseline can fuse across forward/backward timing boundaries that real PPO separates; the negative result rejects this standalone schedule, rather than ruling out recomputation across the actual projection/backward boundaries.

## Benchmark protocol and status

The frozen baseline is the retained address-reconstruction worker, SHA256 `17a2a3d0ccc1c7493268859f5788c64467ad4bfc7357e12bbe6915976e6902fa`. Experiments use original weights, batch, learning rates and strict-F32 settings. Each candidate is checked against exact-prestate CPU derivatives and float64 Adam transitions before and after timing. GPU jobs are serialized under `/tmp/tnsr-bench.cuda0.lock`, with foreign-process audits and source/binary hashes. CUDA graph capture must remain active with zero fallback.

Initial one-process-pair screens (16 complete updates) passed all numerical checkpoints: register tile ~1.09× at microbatch 64; packing ~1.25× at microbatch 64 and ~1.12× at microbatch 8. Register tiling does not select for the microbatch-8 encoder shapes and its noisy screen is not evidence of improvement. Four independent process pairs confirm the combined path at microbatch 64: median 1.3929× throughput, pair-bootstrap 95% interval [1.3816, 1.4009]. Each pair used 32 measured complete updates and candidate qualification before/after; baseline qualification was repeated at the first pair. Packing alone at microbatch 64 is confirmed at median 1.2697×, pair-bootstrap interval [1.2640, 1.2727]. Packing at microbatch 8 is confirmed at median 1.1200×, pair-bootstrap interval [1.1162, 1.1248]. All 30 untimed qualification checkpoints passed with zero violations; every measured CUDA graph was active with zero fallbacks.

Raw screening evidence: sibling `tnsr-benchmarks/results/ppo-parallel-prototypes-20261030/screen.json`. Confirmation evidence: sibling `tnsr-benchmarks/results/ppo-parallel-confirm-20261030/report.json`.

| Variant | Microbatch | Throughput ratio | Pair-bootstrap 95% interval |
| --- | ---: | ---: | --- |
| Packed FFN + larger register tile | 64 | 1.3929× | 1.3816–1.4009× |
| Packed FFN | 64 | 1.2697× | 1.2640–1.2727× |
| Packed FFN | 8 | 1.1200× | 1.1162–1.1248× |

These are native PPO comparisons against the retained baseline, not refreshed competitor rankings or results for other model workloads. The combined option was qualified at microbatch 64; the register-tile selector intentionally falls back for the principal microbatch-8 encoder shapes.

Validation: CPU `cargo check`; CUDA all-target check; all 164 CPU library tests; two packed-row tests; two CUDA output-bit integration tests. Formatting, whitespace checks, and all-target/all-feature Clippy passed.
