# Fusion Baseline

This document records the unfused PTX baseline used to evaluate the general
fusion implementation. Run it with:

```sh
cargo bench --bench fusion --features cuda -- --noplot
```

## Observability

`PtxExecutor` exposes separate compile and execution measurements:

- graph nodes and generated kernels;
- generated PTX bytes and lowering/JIT duration;
- actual CUDA kernel launches, including each batched matmul launch;
- materialized values and logical materialized bytes;
- intermediate materialized bytes; and
- explicit host/device and device/device copies.

`PtxExecutionPlan` records the baseline topological order, input edges, physical
action, materialization intent, release points, and normalized virtual outputs.
`describe_unfused_plan` renders uploads, kernels, materialization copies, and
virtual views for inspection.

## Initial Measurements

Measured on 2026-08-09 with an NVIDIA GeForce RTX 4080 and driver 580.65.06.
Criterion reports 95% confidence intervals.

| Graph | CPU | Static CUDA | Tile PTX |
| --- | ---: | ---: | ---: |
| Pointwise diamond, `[256, 1024]` | 632.04-683.02 us | 343.20-345.18 us | 346.08-347.75 us |
| Reduction chain, `[256, 1024] -> [256]` | 2.5492-2.6306 ms | 275.07-275.34 us | 267.56-270.68 us |

These timings include source upload and final result download, matching current
executor behavior. Later milestones must report both end-to-end time and the
plan metrics so launch or materialization reductions remain visible when copies
dominate small graphs.

## Stage Status

- Stage 0 is complete for the PTX backend: metrics, plan rendering, exact tests,
  pointwise/reduction benchmarks, and existing transformer/convolution coverage.
- Stage 1 has an unfused one-node-per-step `PtxExecutionPlan`. Runtime consumes
  its order, dependencies, and release decisions, establishing the boundary
  used by virtual values and future fused regions.
- Stage 2 implements structural iteration domains, index maps, predicates, and
  normalized virtual tensors. Reshape and flatten alias storage directly.
  Trailing transpose, permutation, and broadcast paths remain virtual through
  host output materialization, while paths feeding opaque kernels conservatively
  retain their existing materialization until consumer fusion is available.

Stage 2 verification on the same GPU measured Tile PTX pointwise at
337.86-340.53 us and reduction at 271.37-272.62 us. Criterion classified the
pointwise change as an improvement and the reduction change as within noise. A
constant -> reshape -> flatten plan now reports zero kernel launches, zero
device-to-device copies, and one materialized value.
