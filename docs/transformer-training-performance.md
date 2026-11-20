# Complete decoder training: performance and memory

## Current opt-in result (V24)

Proven plain BF16 output stores are accepted default-off only for four
W512/S512 cases, with small 0.36–0.81% paired complete-update reductions in both
fresh orders. B16/B32 retain accepted checkpoint-preserving attention fusion.
The other four cases retain V22: regression, mixed, near-neutral or inactive
results do not justify enabling the new option there.

All 48 workers and 32 fixed first-update comparisons pass. Paired scalar
trajectories and first-update gradient/weight hashes are exact. Owned F32/BF16
and queried-pool highwater are unchanged; sampled NVML varies, with no
allocation-saving claim. The [parity report](transformer-training-parity.md)
retains 39 campaigns, 1,018 attempts, 38,848 timed complete updates and 678 passing
first-update comparisons.

The suite remains **two of eight** at target. B32 takes 60.1806 ms, still 2.033x its
strongest qualified PyTorch target. Online-attention and wider-exchange
prototypes are separate from these frozen measurements. Goal active.

## Historical baseline

Measured 2026-10-31 on an RTX 4080, driver 580.196.07, with 16,376 MiB
reported device memory. The 211,147,776-parameter decoder completes training
updates in **40.887 ms / 3,131 tokens/s** in tnsr, versus
**22.549 ms / 5,676 tokens/s** with PyTorch compiled SDPA. Observed whole-process
peaks are **8,886 MiB versus 4,006 MiB**. The larger case exposes a throughput
and VRAM disadvantage that the smaller eager-reference comparisons would miss.

The [local raw evidence](benchmarks/README.md) contains
all 52 process attempts: 50 successful workers with 400 timed updates, and two
native batch-32 CUDA out-of-memory failures. Source hashes, per-update samples,
per-process memory samples, numerical comparisons and
[earlier driver versions](benchmarks/transformer-training-sources/run-scale-screen.py)
are preserved. Multi-gigabyte archives and binary qualification snapshots remain
under `target/transformer-training`; their hashes are recorded in the evidence.

Both experimental kernel options are **off**. These results measure the existing
mixed training path, and establish no end-to-end gain from shared-memory reuse
or online weight sharing. See the separate
[resident kernel measurements](kernel-paper-performance.md) for those options.

## Model and execution policies

Every model is a four-layer pre-LayerNorm causal decoder with learned token and
position embeddings, 64 features per attention head, SwiGLU FFNs, a final
LayerNorm, and a tied token-embedding language-model head. Vocabulary is 4,096;
LayerNorm epsilon is 1e-5. The archive contains f32 master weights, including
biases and affine normalization parameters; the tied head has a constant zero
bias. Parameter counts include all unique trainable tensors.

| Case | Width / heads / FFN hidden | Batch x sequence | Parameters |
| --- | --- | --- | ---: |
| w512_s128 | 512 / 8 / 1,408 | 2 x 128 | 15,038,464 |
| w512_s512 | 512 / 8 / 1,408 | 2 x 512 | 15,235,072 |
| w1024_s128 | 1,024 / 16 / 2,752 | 2 x 128 | 54,980,096 |
| w1024_s512 | 1,024 / 16 / 2,752 | 1 x 512 | 55,373,312 |
| w2048_s128 | 2,048 / 32 / 5,504 | 1 x 128 | 211,147,776 |

Native uses explicit BF16 linear operands/results, BF16 QK and probability/value
matmuls, f32 scaled/masked scores and softmax, BF16 SwiGLU activation/multiply,
and f32 residuals, LayerNorm, masters and Adam. Primitive autograd preserves the
specified storage boundaries. `ResidentAdam::new_with_graph_replay` executes
mixed forward/backward with exact integer feeds; all measured native updates
report active replay with zero fallback. Auxiliary f32 matmuls use `Tf32x3`.
Adam batches up to eight eligible small tensors per update launch.

PyTorch 2.11.0+cu130 uses explicit BF16 casts, f32 masters and fused Adam.
The **matched eager** policy reproduces the primitive normalization,
sign-magnitude SiLU and materialized attention boundaries. **SDPA eager** uses
`F.layer_norm`, `F.silu` and causal scaled-dot-product attention, with different
internal rounding/reduction choices. Both use the same archive and objective.
TF32 and reduced-precision BF16/FP16 reduction flags are disabled in the reference.

**Compiled SDPA** adds `torch.compile` with Inductor, `dynamic=False` and
`fullgraph=False`: the model and automatic backward are compiled; fused Adam
remains outside that compiled model. Compilation and lazy optimizer setup are
excluded from steady samples. The recorded eager backend probe observes Flash
Attention forward/backward operators, but does not prove the compiled model's
actual backend dispatch.

## Complete-update measurements

Each configuration/mode has two fresh processes, with mode order reversed in
trial 1 for the multi-mode campaigns. Each process performs one untimed
qualification update, three excluded warmups and eight synchronized measured
updates, continuing the evolving weights and moments on one fixed batch.

Timings include gradient zeroing, CPU exact-index feed construction/upload,
forward, cross-entropy loss readback, backward, gradient accumulation, global
norm/clipping, Adam, health checks and synchronization through parameter
publication. Archive loading, graph construction/compilation, binary snapshots
and JSON output are excluded. NVML sampling runs concurrently in an external
process. Inputs
are `tokens[j]=(17*j+3)%vocab`; targets use `j+1`. The portable fixture seed is 17.
Adam uses lr 1e-4, betas 0.9/0.999, epsilon 1e-8, zero weight decay and global
norm limit 1.0 with clipping denominator epsilon 1e-6.

The table reports the median of the two process medians. Tokens/s is
`batch * sequence * 1000 / that latency`; memory is the maximum observed
whole-process MiB across those processes.

| Case | Mode | Complete update, ms | Tokens/s | Observed process peak, MiB |
| --- | --- | ---: | ---: | ---: |
| w512_s128 | tnsr | 4.902 | 52,220 | 1,110 |
| w512_s128 | PyTorch matched eager | 18.267 | 14,014 | 618 |
| w512_s128 | PyTorch SDPA eager | 13.668 | 18,730 | 610 |
| w512_s512 | tnsr | 12.216 | 83,821 | 2,416 |
| w512_s512 | PyTorch matched eager | 19.078 | 53,675 | 1,056 |
| w512_s512 | PyTorch SDPA eager | 14.618 | 70,052 | 742 |
| w1024_s128 | tnsr | 12.994 | 19,701 | 2,774 |
| w1024_s128 | PyTorch matched eager | 22.426 | 11,415 | 1,386 |
| w1024_s128 | PyTorch SDPA eager | 16.674 | 15,353 | 1,318 |
| w1024_s512 | tnsr | 19.678 | 26,019 | 3,760 |
| w1024_s512 | PyTorch matched eager | 21.686 | 23,610 | 1,696 |
| w1024_s512 | PyTorch SDPA eager | 16.628 | 30,792 | 1,328 |
| w2048_s128 | tnsr | 40.887 | 3,131 | 8,886 |
| w2048_s128 | PyTorch matched eager | 34.142 | 3,749 | 4,084 |
| w2048_s128 | PyTorch SDPA eager | 27.668 | 4,626 | 3,996 |

The compiled control was a separate campaign on identical fixture bytes:

| Case | Mode | Complete update, ms | Tokens/s | Observed process peak, MiB |
| --- | --- | ---: | ---: | ---: |
| w1024_s512 | PyTorch compiled SDPA | 11.140 | 45,960 | 1,374 |
| w2048_s128 | PyTorch compiled SDPA | 22.549 | 5,676 | 4,006 |

Native loses to compiled SDPA at both measured sizes: 19.678 versus 11.140 ms
at 55.37M parameters, and 40.887 versus 22.549 ms at 211.15M. The former compiled
process medians were 12.002 and 10.278 ms, illustrating the limited precision of
this two-process screen. The compiled campaign was not interleaved with the
native campaign; no confidence interval or universal backend ranking is claimed.

## Batch capacity at width 512, sequence 512

The parameter count remains 15,235,072 while activation work grows with batch.
These runs use the same two-process, three-warmup, eight-update protocol.

| Case | Mode | Complete update, ms | Tokens/s | Observed process peak, MiB |
| --- | --- | ---: | ---: | ---: |
| w512_s512_b8 | tnsr | 40.639 | 100,790 | 7,280 |
| w512_s512_b8 | PyTorch matched eager | 44.206 | 92,657 | 2,606 |
| w512_s512_b8 | PyTorch SDPA eager | 14.952 | 273,945 | 1,284 |
| w512_s512_b16 | tnsr | 83.416 | 98,206 | 13,936 |
| w512_s512_b16 | PyTorch matched eager | 96.490 | 84,900 | 4,810 |
| w512_s512_b16 | PyTorch SDPA eager | 22.497 | 364,135 | 2,012 |
| w512_s512_b32 | tnsr | CUDA OOM, 2/2 attempts | — | 15,406 |
| w512_s512_b32 | PyTorch matched eager | 201.332 | 81,378 | 8,948 |
| w512_s512_b32 | PyTorch SDPA eager | 45.072 | 363,506 | 3,564 |

Native batch 16 succeeds in both attempts; batch 32 fails twice with actual
`CUDA_ERROR_OUT_OF_MEMORY`. The failed attempts reached observed process peaks
of 15,246 and 15,406 MiB. Both PyTorch modes complete batch 32 in both processes.
This establishes a tested native boundary between batches 16 and 32 for this
configuration; batches 17-31 were not tested. It establishes no PyTorch maximum
batch or general maximum parameter count. The 211.15M model is the largest
parameter count successfully tested in this screen.

## Numerical qualification

Before warmups, every successful worker saves all first-step clipped gradients
and post-Adam weights in archive order. Fourteen native/matched comparisons
(two trials across seven fitting configurations) pass the fixed global gates.
Nine additional trial-0 comparisons against eager/compiled SDPA also pass:
the five base eager-SDPA cases, two compiled cases, and batch 8/16 eager SDPA.
No thresholds were relaxed after failures.

| Global metric | Prespecified gate | Largest observed error, all comparisons |
| --- | ---: | ---: |
| Loss normalized absolute error | 0.005 | 0.00013782 |
| Clipped-gradient relative L2 | 0.02 | 0.01661197 |
| Post-weight relative L2 | 0.001 | 0.00063312 |
| Update-vector relative L2 | 0.15 | 0.14242614 |

Loss normalization divides absolute error by `max(1, abs(reference_loss))`.
Finiteness and clipping metadata are checked. An independent float64 first-step
Adam calculation using native's own clipped gradients also passes
`2e-7 + 2e-6 * max(abs(expected_weight))`; its maximum observed absolute error
is about 5.96e-8. This checks the optimizer transition separately from model
arithmetic differences.

These are full-tensor global first-step tolerances, not bitwise parity or
per-parameter error bounds. In particular, the allowed update-vector difference
is 15%, and cross-policy errors reach 14.24%. The experiment does not establish
long-run learning equivalence or training quality; timed losses/norms must remain
finite, but full-gradient snapshots are not taken during timed updates.

## Memory scopes

The headline memory columns use requested 50 ms whole-process NVML polling,
including CUDA context, modules, allocator reservations, setup and qualification.
They are sampled lower bounds on instantaneous process peaks. CUDA work is
serialized under `/tmp/tnsr-bench.cuda0.lock`; the recorded process audits found
no foreign compute process.

Native additionally queries current/default CUDA pool USED/RESERVED current and
high-water values. Both handles agree in these runs. High-water counters are
recorded across setup/qualification, reset only after warmups, and sampled again
after timed updates. These are exact peaks for the queried pool, excluding
context/modules and other pools. PyTorch reports its caching allocator's
allocated/reserved peaks with a corresponding steady reset. The two allocators'
accounting scopes are distinct; neither is an exact whole-process peak.

For the 211.15M model, native's pool used high is 8,323.060 MiB and reserved high
is 8,640 MiB; those values are the same in the lifecycle and steady windows.
Compiled PyTorch's allocator lifecycle peaks are 3,311.526 MiB allocated and
3,696 MiB reserved. Native's 8,886 MiB and compiled PyTorch's 4,006 MiB sampled
process peaks provide the broader comparison. Retained tensor-payload counters
in the evidence are another scope and must not be labeled process peak VRAM.

## Next investigation

The [first steady-update profiles](transformer-training-profile.md) now break
down optimizer, accumulation, materialization and matmul costs for both larger
cases, with bounded improvement targets and candidate changes.

Profile the complete 55M/211M updates before selecting the next optimization.
Attribute latency to BF16 GEMMs, materialized attention forward/backward,
pointwise/reduction work, transfers and optimizer launches. Audit device-buffer
lifetimes across the resident optimizer, staging/health buffers, mixed master
copies and activations, including setup peaks versus steady retention.

The batch-capacity result makes attention materialization and buffer-lifetime
reuse concrete research targets. It does not identify a profiler-proven stall
or prove which allocation dominates. Ordinary GEMM shared-memory reuse changes
per-block scratch storage; it has not been integrated or qualified as a fix for
this full mixed-training path's global VRAM usage. Keep numerical boundaries
and the fixed gates while testing any replacement.

## Reproduction

Use a CUDA-visible host and a Python environment with the recorded PyTorch
2.11.0+cu130 and NumPy 2.4.3. On the measured host, the working interpreter was
`/home/taylor/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12`
with `PYTHONPATH=/home/taylor/code/forty-five/.venv/lib/python3.12/site-packages`;
use that interpreter in place of `python3` below on this host. The runner records
binary/source/fixture hashes, reverses worker
order, checks for foreign compute processes, and saves failures as well as wins.
Output directories must be fresh.

```sh
cargo build --release --features cuda --example transformer_training_bench
python3 integrations/transformer_bench/run.py \
  --output target/transformer-training/reproduce-scale \
  --cases w512_s128,w512_s512,w1024_s128,w1024_s512,w2048_s128 \
  --modes native,pytorch_matched,pytorch_sdpa --trials 2 --warmups 3 --repeats 8
python3 integrations/transformer_bench/run.py \
  --output target/transformer-training/reproduce-compiled \
  --cases w1024_s512,w2048_s128 --modes pytorch_compiled_sdpa \
  --trials 2 --warmups 3 --repeats 8
python3 integrations/transformer_bench/run.py \
  --output target/transformer-training/reproduce-batch \
  --cases w512_s512_b8,w512_s512_b16,w512_s512_b32 \
  --modes native,pytorch_matched,pytorch_sdpa --trials 2 --warmups 3 --repeats 8
```

The current driver extends the original case registry; archived versions and
hashes in the evidence preserve which driver produced each campaign. Run
`integrations/transformer_bench/validate.py` on an archive and two workers'
artifact directories to recompute fixed first-step gates from binary snapshots.
