# Current accepted B32 training diagnostic

## Durable packages and restoration

The raw archive contains five unique 60,940,288-byte F32 objects (native clipped
gradients/postweights, qualified PyTorch clipped gradients/postweights, and seed17
fixture weights). Nine original paths are SHA256 aliases. Both archives were
fully decompressed and every member's content hash verified. No executable or
compiler/tool binary is included; the native Rust source archive is included.

- `raw-f32.tar.gz`: content-addressed tensor bytes and alias manifest.
- `trace-and-provenance.tar.gz`: original workspace-relative trace/SQLite,
  source archive, frozen controller/plan, validations, labels/resources, audits
  and helper/profiler/runtime provenance; includes `SHA256-MANIFEST.json`.
- `raw-manifest.json`, `metadata-manifest.json`, `evidence.json`: exact hashes.
- `restore-raw.py`: verifies raw content then recreates aliases with hardlinks.

For a fresh scratch directory, run the raw helper first and then extract the
metadata package into that same directory. Paths below describe the extracted
`target/packed-b32-profile-v1` directory. These commands restore evidence, not a
new measurement; rebuilding/reprofiling requires the recorded Rust/CUDA/Nsight
setup and a separate quiet GPU reservation.

```sh
python3 restore-raw.py --archive raw-f32.tar.gz --destination /tmp/tnsr-b32-evidence
tar -xzf trace-and-provenance.tar.gz -C /tmp/tnsr-b32-evidence
```

The fresh profile passes the isolation, unchanged first-update numerical and exact native trajectory checks. GEMMs account for about 45.1% and online attention 21.3% of summed GPU kernel activity. This is a single five-update diagnostic, excluded from benchmark counts; it changes no accepted policy or parity result (still 2/8).

## Scope and correctness

- Accepted `w512_s512_b32`: W512/S512/B32, four layers, vocab 4096, packed online backward, NT ldmatrix 12, small accumulation/norm batches; borrowed/vector Adam and weight shadows disabled.
- Same retained release `125b2bba930b62716cc1009a2d4d12bdc9e650df0d2836154e3873cd31f9a29b`, source archive `c5aecfd0f140b4efaa68fc30af8c7f043051ffdaf013d7d86bddd9010fccda80`.
- Nsight Systems 2026.6.1; five warmups then five CUDA-profiler-API-captured complete updates. Setup, qualification and warmups are outside the capture.
- Both six-gate first-update comparisons pass against existing accepted native and qualified fresh PyTorch references. Native gradient/weight hashes match both native trials (four file pairs); all 66 F64 loss/norm/scale values match the corresponding first 11 optimizer steps.
- Exact native child PID 516999 and its Nsight ancestry were verified; one frozen idle desktop context, no observed monitor violations. The observed process peak is 6,316 MiB, a sampled lower bound including profiler effects, separate from native allocator peaks.

## Semantic mapping and accounting

Five `cuGraphLaunch` correlation IDs identify exactly 298 captured model kernels each. Graph variants alternate `[2,5,2,5,2]`. Every ordered label is matched through actual graph-node/original-node lineage offsets and identical function/grid/block/resource signatures across all five updates. The 298 offsets are identical in both source graph namespaces. Two missing offsets 293/295 are exactly the two embedding-output memsets per update.

There are 59 standalone kernel launches per update: 27 accumulation/norm, one loss-scalar readback materializer, two global-norm kernels, 22 standalone Adam kernels, five batched Adam kernels and two health reductions. Each joins a unique `cuLaunchKernel` correlation. Source-archived sorted parameter registration, parameter ranges and checked grids identify standalone Adam/accumulation parameters; small Adam batch membership follows the source-bound stable length sort and chunks of eight. Fused small accumulation batch members are not inferred from trace names.

Total GPU kernels are 1,785 = 5 × 357. The software counter reports 356 per update because the extra loss-scalar materializer is outside that counter. There are additionally 10 graph memsets and 35 memcpy records. Generic function names are never the sole semantic key.

Fine GEMM categories below describe recorded M/N/K geometry or explicit label kinds. FFN/QKV/dX/tied-head roles are model-context hypotheses until separately joined to gradient/parameter graph provenance; the actual model-position/label/node/function/resource mapping is verified.

## Summed GPU kernel activity

Values are means of the five sums, not unprofiled synchronized latency or critical-path attribution. Raw values for every update and node are retained in `analysis.json`.

| Group | Kernels/update | Mean summed ms/update | Share |
| --- | ---: | ---: | ---: |
| GEMM other | 42 | 15.144408 | 29.07% |
| Online attention | 12 | 11.088338 | 21.28% |
| GEMM split-K partial | 21 | 8.363239 | 16.05% |
| Reduction | 58 | 3.878813 | 7.44% |
| LayerNorm / outer add | 25 | 3.572545 | 6.86% |
| Materialization | 89 | 3.436989 | 6.60% |
| SwiGLU | 8 | 2.416239 | 4.64% |
| Indexed loss | 2 | 1.549115 | 2.97% |
| Pointwise | 16 | 1.198661 | 2.30% |
| Adam / health | 29 | 0.692301 | 1.33% |
| Split-K sum | 21 | 0.321128 | 0.62% |
| Accumulation / norm | 29 | 0.260328 | 0.50% |
| Embedding | 4 | 0.181128 | 0.35% |
| Loss scalar materialization | 1 | 0.000960 | 0.00% |

Total mean summed kernel activity is 52.104192 ms/update. Profiled synchronized elapsed values are preserved separately: 54.565695, 53.816206, 52.234301, 54.534967, 53.980938 ms. Nsight reports legacy software instrumentation; these values must not update the throughput scorecard.

## GEMM geometry detail

| Recorded geometry/kind | Kernels/update | Mean summed ms/update |
| --- | ---: | ---: |
| Split-K partial GEMMs (shape labels) | 21 | 8.363239 |
| NT128 ldmatrix bias (12 nodes) | 12 | 3.907453 |
| MatMul M16384 N512 K1408 (8 nodes) | 8 | 2.895386 |
| WMMA Linear kernels (8 nodes) | 8 | 2.090995 |
| MatMul M16384 N1408 K512 (4 nodes) | 4 | 2.040902 |
| MatMul M16384 N512 K1536 (4 nodes) | 4 | 1.556968 |
| NT128 WMMA bias M16384 N4096 K512 (1 node) | 1 | 1.085168 |
| MatMul M16384 N512 K4096 (1 node) | 1 | 1.000848 |
| MatMul M16384 N512 K512 (4 nodes) | 4 | 0.566687 |

## Online attention and actual resource fields

All 12 attention kernels per update launch 256 threads as `(32,8,1)`, grid `(8,256,1)`. Resources below come from actual traced kernel rows and match across all four corresponding nodes and all five updates.

| Direction | Nodes/update | Mean summed ms/update | Registers/thread | Static shared B | Dynamic shared B | Local/thread B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| forward | 4 | 3.293470 | 56 | 49152 | 0 | 0 |
| packed Q+delta | 4 | 3.414680 | 72 | 49152 | 0 | 0 |
| packed K+V | 4 | 4.380188 | 97 | 49152 | 0 | 0 |

Nsight reports zero local memory per thread for every kernel. Its `localMemoryTotal` field is 67,239,936 bytes for every kernel, regardless of function; that raw constant is retained without claiming spill traffic or a per-node allocation delta. `sharedMemoryExecuted` is a driver setting and differs from static shared bytes. No achieved occupancy, stall, DRAM-traffic or tensor-utilization counters were collected.

## Interpretation and limitations

- Investigate GEMM kernels and online attention next; their combined summed activity is 34.596 ms/update in this trace. This is a prioritization signal, not a predicted end-to-end gain.
- The 21 physical-order weight casts total only 0.160838 ms/update. Adam/health and accumulation/norm total about 0.953 ms/update. Closest-case optimizer/shadow results cannot be extrapolated to this B32 workload.
- Legacy timestamps show 994 small adjacent overlaps on one stream, maximum 1,472 ns. Summed durations are not a critical path. All five values, including slower individual nodes, are retained.
- A single profiler run supplies no confidence interval, causal bottleneck proof or performance comparison. Numerical and replay validity do not remove profiler overhead.
- Inventory samples and approximately 1 s desktop utilization observation cannot exclude sub-sample activity. Process peak and allocator counters remain separate; no new capacity result is claimed.

## Artifacts

Paths below are relative to the parent diagnostic directory.

- `analysis.json`: all 1785 annotated kernel rows, all 298 per-position semantic/resource records with five durations, graph lineage, standalone parameter associations, API/memcpy/memset rows, actual SQLite schema and raw diagnostic events.
- `analyze-trace.py`: deterministic CPU-only analysis with source archive/order/grid checks.
- `run-v1/trace.nsys-rep` and `run-v1/trace.sqlite`: retained raw capture/export.
- `run-v1/validation-summary.json`, two full validators, native artifacts and ordered `kernel-label-map.json`.
- `run-v1/process.json`, complete child ancestry, raw NVML and desktop-monitor evidence.
- `input-audit.json`: six freshly hashed F32 references match original and published evidence despite changed filesystem device/inode/ctime metadata.
- Independent guard/numerical review: `../packed-b32-profile-audit-v1/audit.json`. Independent semantic review: `../packed-b32-profile-audit-v1/semantic-audit.json` (passed).

The initial analysis used inferred role names for some GEMM geometries. That earlier artifact is preserved as `analysis.initial.json`; the current report explicitly limits those associations to geometry.
