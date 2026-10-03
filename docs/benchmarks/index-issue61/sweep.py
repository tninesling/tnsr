#!/usr/bin/env python3
"""Run paired matmul profilers; arguments: before-binary after-binary output.csv."""
import csv
import itertools
import os
from pathlib import Path
import subprocess
import sys


def main():
    before, after, output = sys.argv[1:]
    binaries = {"before": str(Path(before).resolve()), "after": str(Path(after).resolve())}
    cases = [("batch", b, 128, k, n) for b, k, n in itertools.product(
        [1, 2, 8], [128, 256, 512], [384, 768, 1536])]
    cases += [("batch", b, 128, 256, 768) for b in [3, 5]]
    cases += [("flat", 2, 128, 256, 768), ("batch", 3, 33, 37, 35)]
    result_fields = ["mode", "stage", "b", "m", "k", "n", "launches",
                     "resident_p50_us", "resident_p10_us", "resident_p90_us",
                     "e2e_p50_us", "e2e_p10_us", "e2e_p90_us", "max_error"]
    fields = ["variant", *result_fields, "compile_us", "index_optimization_us",
              "intermediate_bytes"]
    env = os.environ.copy()
    env.pop("TNSR_PROFILE_OVERRIDE_PTX", None)
    env.pop("TNSR_PROFILE_PTX", None)
    with Path(output).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for serial, (mode, b, m, k, n) in enumerate(cases):
            variants = ["before", "after"] if serial % 2 == 0 else ["after", "before"]
            for variant in variants:
                text = subprocess.check_output(
                    [binaries[variant], mode, "3", str(b), str(m), str(k), str(n)],
                    text=True, env=env)
                result = next(line for line in text.splitlines()
                              if line.startswith("RESULT,")).split(",")[1:]
                row = dict(zip(result_fields, result, strict=True))
                row["variant"] = variant
                metrics = next((line for line in text.splitlines()
                                if line.startswith("METRICS,")), None)
                if metrics:
                    compile_us, index_us, launches, intermediate = metrics.split(",")[1:]
                    if launches != row["launches"]:
                        raise ValueError("resident and executor launch counts differ")
                    row.update(compile_us=compile_us, index_optimization_us=index_us,
                               intermediate_bytes=intermediate)
                writer.writerow(row)
                file.flush()
            print(f"{mode} B={b} M={m} K={k} N={n}", flush=True)


if __name__ == "__main__":
    main()
