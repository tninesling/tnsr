#!/usr/bin/env python3
"""Convert the eight first-round diagnostic cases into an explicitly small pilot.

Graph features are static estimates, not measured fusion/launch statistics.
The input is docs/benchmarks/graph-rewrites.json from the previous experiment.
"""
import argparse
import json
from pathlib import Path


CASES = {
    "reshape_chain": ("reshape", (256, 256)),
    "permute_chain": ("permutation", (16, 32, 64)),
    "transpose_add": ("transpose_control", (256, 256)),
    "factor_matmul": ("matmul_factor", (64, 32, 64)),
    "matrix_chain": ("matmul_chain", (128, 8, 128, 8)),
    "factor_large": ("matmul_factor", (256, 128, 256)),
    "chain_large": ("matmul_chain", (256, 32, 256, 32)),
    "factor_rejected": ("matmul_factor", (2, 256, 2)),
}


def features(case, variant):
    family, dims = CASES[case]
    optimized = variant == "rewrite"
    result = {"flops": 0, "read_bytes": 0, "write_bytes": 0,
              "matmul_count": 0, "reduction_count": 0, "layout_count": 0,
              "reshape_count": 0, "pointwise_elements": 0, "output_elements": 0,
              "max_rank": 2, "graph_operation_count": 0}
    if family in ("reshape", "permutation"):
        count = 1
        for d in dims:
            count *= d
        ops = 0 if optimized else 2
        result.update(output_elements=count, max_rank=len(dims),
                      read_bytes=4 * count * (1 + ops), write_bytes=4 * count * ops,
                      graph_operation_count=ops)
        result["reshape_count" if family == "reshape" else "layout_count"] = ops
        heuristic = 1 if optimized else (9 if family == "reshape" else 6 * count + 137)
    elif family == "transpose_control":
        count = dims[0] * dims[1]
        result.update(output_elements=count, read_bytes=16 * count, write_bytes=12 * count,
                      layout_count=2, pointwise_elements=count, flops=count, graph_operation_count=3)
        heuristic = 9 * count + 194
    elif family == "matmul_factor":
        m, k, n = dims
        result.update(matmul_count=1 if optimized else 2, output_elements=m * n,
                      pointwise_elements=k * n if optimized else m * n,
                      graph_operation_count=2 if optimized else 3,
                      flops=2 * m * k * n + k * n if optimized else 4 * m * k * n + m * n,
                      read_bytes=4 * (m * k + 3 * k * n + m * n) if optimized else 4 * (m * k + 2 * k * n + 2 * m * n),
                      write_bytes=4 * (k * n + m * n) if optimized else 12 * m * n)
        heuristic = 2 * m * k * n + 3 * k * n + 131 if optimized else 4 * m * k * n + 3 * m * n + 196
    else:
        m, k, n, p = dims
        intermediate = k * p if optimized else m * n
        result.update(matmul_count=2, output_elements=m * p, graph_operation_count=2,
                      flops=2 * (k * n * p + m * k * p) if optimized else 2 * (m * k * n + m * n * p),
                      read_bytes=4 * (m * k + k * n + n * p + intermediate),
                      write_bytes=4 * (intermediate + m * p))
        heuristic = result["flops"] + 131
    return result, heuristic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for backend in ("cpu", "gpu"):
        rows = []
        for case, (family, dims) in CASES.items():
            for variant, timing_key in (("original", f"{backend}_before"), ("rewrite", f"{backend}_after")):
                timing = source[timing_key][f"{case}/optimized"]
                static_features, heuristic = features(case, variant)
                rows.append({"schema_version": 1, "backend": "cuda" if backend == "gpu" else "cpu",
                             "device": source[backend], "precision": source["gpu_precision"] if backend == "gpu" else "f32",
                             "family": family, "shape_key": "x".join(map(str, dims)), "case_id": case,
                             "variant": variant, "mean_ns": timing["mean_ns"], "ci95_ns": timing["ci95_ns"],
                             "heuristic_cost": heuristic, "features": static_features,
                             "feature_provenance": "static graph estimates; not measured plan/fusion/launch statistics",
                             "source": str(args.source)})
        path = args.output_dir / f"previous-{backend}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        print(path)


if __name__ == "__main__":
    main()
