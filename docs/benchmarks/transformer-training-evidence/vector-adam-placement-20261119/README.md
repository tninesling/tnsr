# Vector Adam placement experiment

Two fresh processes passed the balanced mode/output-slot protocol: 864 event measurements with 32 captured invocations each. All 432 output comparisons and 60 unchanged-input checks passed. This is a component experiment and does not establish complete-update performance or parity.

| Tensor elements | Process | Scalar 256, us | Vector 256, us | Scalar / vector |
| --- | ---: | ---: | ---: | ---: |
| 2818048 | 0 | 84.1265 | 78.7985 | 1.0676x |
| 3145728 | 0 | 126.2370 | 122.7965 | 1.0280x |
| 3145728 | 1 | 124.6225 | 120.4480 | 1.0347x |
| 2818048 | 1 | 90.3840 | 84.7680 | 1.0663x |

The vector kernel improves against both scalar geometries in both fresh processes for both lengths. The earlier V2 experiment had mixed results; this version balances each mode across all three output allocations and reverses construction order. No causal memory or cache explanation is established.

The first current-environment timing attempt was invalidated by a transient Files CUDA context and is retained under target/recovery-vector-bridge-v4/timing-v1. The successful retry is timing-v2; its complete reports, raw intervals, snapshots and monitors remain under target/recovery-vector-bridge-v4. New owner integration is experimental and requires state, fallback, numerical and complete-model tests before adoption.
