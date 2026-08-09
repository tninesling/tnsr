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
action, materialization intent, and release points. `describe_unfused_plan`
renders uploads, kernels, and view copies for inspection.

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
  its order, dependencies, and release decisions. Virtual tensors and access
  maps are the next structural change before plan steps can become regions.
