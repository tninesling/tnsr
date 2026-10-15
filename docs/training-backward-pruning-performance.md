# Pruning unused backward branches

Issue [108](https://github.com/tninesling/tnsr/issues/108) removes generated
derivatives which cannot contribute to a registered parameter gradient. Every
original forward node remains at its public node index, including the loss,
other outputs and disconnected parameters. The retained backward closure keeps
shared expressions; gradient metadata is remapped before dedicated sink views
are pinned. No arithmetic or reduction ordering policy changes.

On the real shared forty-five PPO graph, the node count falls **4,484 -> 3,849**:
all 2,087 original forward nodes and all 140 parameter gradients remain, while
generated backward nodes fall 2,397 -> 1,762. The 635 removed nodes include 354
reductions and one GEMM. A second parameter-gradient reachability probe finds
no remaining unused generated backward nodes. Avoiding their initial construction
is a separate potential cold-build optimization.

## Paired real-model measurements

Baseline is merged `50626f4`, using the preserved `ddf00de` replay worker with
the same kernels; the intervening diagnostic-only internal-copy counter change
does not affect these PPO measurements. Candidate is `34c9fdb`. Each case uses
the same v8.4 archive and seeded 64-transition batch, two excluded warmups and
ten interleaved timed samples on RTX 4080. Existing experiments stay running.
Tests, checkpoint validation, timing and profiling run separately.

| Median ms/update | Before micro 8 | Pruned micro 8 | Before micro 64 | Pruned micro 64 |
| --- | ---: | ---: | ---: | ---: |
| Strict, ordinary launches | 308.9 | 248.4 | 125.6 | 107.5 |
| Strict, replay | 296.6 | 243.0 | 140.7 | 109.2 |
| Tree, ordinary launches | 225.0 | 210.0 | 115.3 | 112.9 |
| Tree, replay | 141.8 | 135.1 | 159.1 | 154.8 |

Strict-order medians improve **14–22%**; tree-order medians improve **2–7%**.
Only compare before/after within a row and microbatch: contention differs
substantially between separately run configurations. These samples do not rank
strict/tree or replay policies and do not establish an exclusive-GPU parity gate.

Original eager-f32 microbatch-64 medians are 32.4–36.0 ms across these paired
runs; compiled-policy bf16 is 29.4–33.2 ms. The native large-batch gap remains
roughly **3–5x**, despite the work removed. Mixed precision #94 remains open.

| Warm GPU work/update | Before micro 8 | Pruned micro 8 | Before micro 64 | Pruned micro 64 |
| --- | ---: | ---: | ---: | ---: |
| Strict kernels | 26,160 | 22,744 | 3,408 | 3,081 |
| Tree kernels | 25,920 | 22,744 | 3,360 | 3,081 |
| Strict logical intermediate GB | 10.292 | 9.778 | 10.123 | 9.597 |
| Tree logical intermediate GB | 10.106 | 9.778 | 9.869 | 9.597 |

Replay changes host submission, with the same warm physical GPU work. Logical
materialization is the sum of intermediate value sizes, not measured DRAM
traffic or peak VRAM. Full/update arithmetic, accumulation and Adam remain f32.

Process GPU memory is stable over the ten samples. Ordinary execution falls
**732 -> 698 MiB** (strict micro 8), **732 -> 696 MiB** (tree micro 8), and
**2,652 -> 2,458 MiB** (both micro 64 cases). Replay falls **786 -> 748/746 MiB**
at micro 8 and **2,706 -> 2,510 MiB** at 64. These process measurements include
modules, workspace and driver graph metadata, sampled outside timing. They
establish savings for one active shape, rather than a full batch-capacity envelope.

Separate three-update node-level Nsight traces confirm the counters exactly:
tree replay GPU instances fall **83,924 -> 73,602** at micro 8 and
**16,232 -> 14,837** at 64. These totals include two pure-model warmup evaluations
before capture; warm work alone is the table above. Host recording and remaining
individual kernel calls fall **13,038 -> 11,847 / 10,080 -> 9,243**; graph launches
stay **24 / 3**, and D2H copies stay **30 / 9** across three updates. Recording
adds host calls but no GPU execution. Node-level graph instrumentation affects
durations heavily, so these traces substantiate counts rather than an additional
timing speedup.

## Validation

All **187 CPU** and **375 CUDA-feature** tests pass, along with formatting and
all-feature clippy. New regression tests cover frozen versus trainable inputs,
shared producers, disconnected parameters, dedicated sink identity, no-parameter
graphs and exact reconstruction through `without_gradients`. The full suites
also exercise masked/nonfinite branches, views, half storage, resident Adam and
replay recovery.

Both actual shared (140 parameters) and separate critics (229 parameters) pass
two PPO updates in strict/tree replay modes. Full and partial shape buckets stay
captured with zero fallbacks. Each named clipped gradient, updated weight,
schedule and saved archive is compared with the original oracle under the
existing tolerances.
Maximum gradient error is below `3.9e-7`; maximum weight error is below `1.3e-5`.

## Reproduction

Preserve baseline and candidate release workers, then use a fresh output directory:

```sh
TNSR_PROFILE_PPO=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
  /path/to/forty-five/.venv/bin/python integrations/forty_five/benchmark_training.py \
  ARCHIVE --forty-five-root /path/to/forty-five \
  --baseline-worker /tmp/before --worker /tmp/after \
  --microbatch 8 --repeats 10 --compile --output FRESH_OUTPUT
```

Repeat at 64. Add `--tree-reductions --baseline-tree-reductions` for matched tree
ordering, and `--cuda-graphs --baseline-cuda-graphs` for replay on both sides.
The baseline must support those explicitly requested options. The benchmark
rejects unexpected replay fallback in either worker. Run actual checkpoint
validation and Nsight profiles separately from timing.

Full generated reports and profiles are under ignored `target/forty_five_pruned/`.
Full paired samples, source/batch/executable hashes, compact correctness reports
and trace summaries are in [pruned.json](benchmarks/training-parity/pruned.json).
