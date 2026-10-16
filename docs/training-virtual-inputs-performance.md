# Indexed inputs for standalone pointwise kernels

Standalone pointwise operations now read valid virtual views through the same
indexed-input lowering used by multi-operation fusion regions. A broadcast,
transpose or permutation no longer needs an expanded intermediate merely because
its consumer has only one arithmetic operation. This is automatic in the PTX
planner and preserves the selected matmul and reduction policies.

## Diagnosis and implementation

A fresh trace of the merged TF32x3 worker attributes 33.7% of captured model GPU
kernel time to matmuls, 22.1% to broadcasts and 20.2% to reductions. The trace
contains five captured executions, separately filtered from eager capture setup.
It uses the real shared 45 model and the fixed 64-transition research batch.
These are profiler duration shares, not throughput measurements.

The planner previously allowed indexed inputs for fused regions but required
contiguous inputs for unfused pointwise kernels. A single multiply or divide
therefore forced its broadcast input to materialize. Give eligible standalone
view consumers a one-operation region. These regions contract no graph edges;
adding them in the common plan-construction path also covers cycle refinement
and cost-model alternatives. Skip meaningless singleton split-cost queries.

The existing fanout, index-complexity and reduction-layout limits still control
view materialization. Opaque consumers keep their existing paths. CPU execution,
the graph's mathematical operations, and the chosen precision policy retain
their contracts. This introduces no new arithmetic approximation; floating-point
results are checked under the existing tolerances rather than promised bitwise
identical. The compilation signature advances to version 17.

## Validation

All 193 CPU tests and 389 CUDA-required all-feature tests pass, as do formatting,
Clippy and doc tests. New CUDA tests read a single aliased source through two
different broadcast/transpose maps, change feeds without recompilation, cover odd
extents and verify that a virtual mask condition suppresses masked NaNs.
Existing half-precision, shared-gradient, layout, residency and replay coverage
also passes.

Two complete PPO updates pass against the unchanged original CPU-f32 oracle:

| Actual model / execution | Parameters | Max gradient error | Max weight error |
| --- | ---: | ---: | ---: |
| Shared / TF32x3, tree reductions + replay | 140 | 1.19e-6 | 1.23e-5 |
| Separate / TF32x3, tree reductions + replay | 229 | 6.85e-7 | 1.44e-5 |
| Shared / strict-f32 matmuls/reductions, no replay | 140 | 3.56e-7 | 2.99e-6 |

The checks include every clipped named gradient and updated weight, weighted
3/3/2 microbatches, nonzero anchor KL, scheduler steps and archive restores. Both
replay buckets stay active with zero fallbacks. Short numerical checks do not
establish learning-quality parity with original BF16 training.

## Measurements

The baseline is merged commit `4f03941`, including exact indices and TF32x3. Both
workers use TF32x3, deterministic-tree reductions and CUDA Graph replay with the
same initial archive, batch, optimizer settings and full update boundaries.
No comparison here combines the earlier precision-policy speedup with this
planner change. Setup/warmup calls are excluded from steady-state timing and
retained separately in the evidence.

Ten interleaved timed samples after two warmups, on RTX 4080:

| Microbatch | Baseline | Candidate | Throughput gain | Worker GPU memory |
| --- | ---: | ---: | ---: | ---: |
| 8 | 89.68 ms | 72.03 ms | 24.5% | 748 -> 708 MiB |
| 64 | 64.34 ms | 52.21 ms | 23.2% | 2510 -> 2214 MiB |

At microbatch 8, physical kernels fall 22752 -> 17352. At microbatch 64 they
fall 3082 -> 2407. Every timed native update uses replay with zero fallbacks.
Memory is whole-worker process GPU usage sampled outside timing, including
modules, workspace and graphs; it is not a precise peak-allocation counter.

Original eager strict-f32 PyTorch measures 237.79 / 33.38 ms at microbatch 8/64;
the original compiled-policy BF16 path measures 206.44 / 30.81 ms. The current
shared-model microbatch-64 gap to compiled BF16 is about **1.7x** in this run.
Policies differ and PyTorch timings vary with host load. This is progress on the
fixed shared workload, not a completed parity gate across workloads and learning
quality. Compare before/after within matched cases; do not multiply speedup
ratios from separate runs.

The initial native-only pilot reverses pair order across 20 timed samples after
two warmups: median update time is **66.64 -> 53.88 ms**, or **23.7% higher
throughput**. At microbatch 64, physical kernels per update decline **3082 ->
2407**, removing 675 launches, while the graph retains all 3851 nodes. Cumulative
logical intermediate outputs decline **9.60 -> 8.13 GB**; these are neither
measured DRAM traffic nor peak VRAM.

The independent post-change trace matches the work reduction: five captured
model executions fall 13990 -> 10615 kernel instances, exactly 675 fewer per
update. Average summed broadcast-kernel duration falls **13.68 -> 2.05 ms per
update**. Total captured kernel duration falls 61.87 -> 51.12 ms; these are
profiler durations, not the wall-clock benchmark. Matmuls now account for 41.6%
and reductions for 25.6% of captured kernel time, making those the next targets.

Full matched measurements and raw provenance are in
[virtual-inputs.json](benchmarks/training-parity/virtual-inputs.json). Generated
reports and traces remain under ignored `target/forty_five_virtual_inputs/` and
`target/forty_five_post_tf32x3/`.

To compare an earlier worker with the candidate using the same arithmetic:

```sh
TNSR_PROFILE_PPO=1 /path/to/forty-five/.venv/bin/python \
  integrations/forty_five/benchmark_training.py ARCHIVE \
  --forty-five-root /path/to/forty-five \
  --baseline-worker /path/to/baseline-worker --worker /path/to/candidate-worker \
  --output target/forty_five_virtual_inputs_repeat --microbatch 64 --repeats 10 \
  --tf32x3 --baseline-tf32x3 --tree-reductions --baseline-tree-reductions \
  --cuda-graphs --baseline-cuda-graphs --compile
```
