#!/usr/bin/env python3
"""Save compact fusion comparisons and compiler-only profitability examples."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(paths):
    rows = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            rows[(row["case"], row["m"], row["n"], row["variant"])] = row
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs="+", required=True,
                        help="Later files replace matching rows from earlier files")
    parser.add_argument("--current", nargs="+", required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--training-jsonl", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    baseline, current = load(args.baseline), load(args.current)
    fields = ["case", "m", "n", "variant", "before_us", "after_us", "speedup",
              "before_ci_low_us", "before_ci_high_us", "after_ci_low_us", "after_ci_high_us",
              "before_readback_us", "after_readback_us", "launches", "serial_threads",
              "fusion_selection_us"]
    with args.csv.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, row in sorted(current.items()):
            if key not in baseline:
                continue
            before = baseline[key]
            old, new = before["end_to_end"], row["end_to_end"]
            writer.writerow(dict(zip(fields, [
                *key, old["mean_ns"] / 1000, new["mean_ns"] / 1000,
                old["mean_ns"] / new["mean_ns"],
                *(value / 1000 for value in old["ci95_ns"]),
                *(value / 1000 for value in new["ci95_ns"]),
                before["readback_only"]["mean_ns"] / 1000,
                row["readback_only"]["mean_ns"] / 1000, row["launches"],
                ";".join(str(k["threads_per_block"]) for k in row["kernels"]
                         if k["reduction_lanes"] == 1),
                row["fusion_selection_ns"] / 1000])))
    # Precision policies are separate identities. Keep strict candidates only;
    # every feature below is available from the graph or selected plan before execution.
    sums = ("traffic_bytes", "weighted_operations", "address_operations", "intermediate_bytes",
            "active_threads", "full_domain_passes", "input_index_operations", "non_identity_inputs")
    maxima = ("estimated_registers_per_thread", "shared_bytes_per_block", "serial_working_set_bytes",
              "threads_per_block", "reduction_extent", "reduction_stride", "reduction_lanes")
    with args.training_jsonl.open("w") as stream:
        for key, row in sorted(current.items()):
            case, m, n, variant = key
            if variant not in ("auto", "greedy"):
                continue
            if (case, m, n, "auto") not in current or (case, m, n, "greedy") not in current:
                continue
            kernels = row["kernels"]
            features = {"plan_sum_" + name: sum(k[name] for k in kernels) for name in sums}
            features.update({"plan_max_" + name: max((k[name] for k in kernels), default=0)
                             for name in maxima})
            features.update(plan_kernel_count=len(kernels), graph_nodes=row["graph_nodes"],
                            output_elements=row["output_elements"],
                            output_index_operations=row["output_index_operations"],
                            output_non_identity=int(row["output_non_identity"]))
            record = dict(schema_version=1, backend="cuda", device=args.device, precision="strict-f32",
                          family=case, shape_key=f"{m}x{n}", case_id=f"{case}:{m}x{n}",
                          variant="original" if variant == "greedy" else "optimized",
                          mean_ns=row["end_to_end"]["mean_ns"],
                          ci95_ns=row["end_to_end"]["ci95_ns"], features=features)
            stream.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
