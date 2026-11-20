#!/usr/bin/env python3
"""Serialize resident GPU prototype comparisons and preserve all paired samples."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time


def cases():
    result = []

    def gemm(name, m, n, k, dtype="bf16", precision="native", pipeline="auto", extra=()):
        result.append({
            "name": name, "kind": "gemm", "binary": "shared_memory_probe",
            "args": ["--m", str(m), "--n", str(n), "--k", str(k), "--dtype", dtype,
                     "--precision", precision, "--pipeline", pipeline, *extra],
            "repeats": 64,
        })

    gemm("bf16_gate_forward", 4096, 683, 256, extra=["--transpose-b"])
    gemm("bf16_down_forward", 4096, 256, 683, extra=["--transpose-b"])
    gemm("bf16_qkv_forward", 4096, 768, 256, extra=["--transpose-b"])
    gemm("bf16_qkv_dgrad", 4096, 256, 768)
    gemm("bf16_longk_wgrad", 683, 256, 4096,
         extra=["--transpose-a", "--reshape-batch", "64"])
    gemm("bf16_longk_transposed_gradient", 256, 683, 4096,
         extra=["--transpose-a", "--reshape-batch", "64"])
    gemm("bf16_small_forward", 512, 683, 256, extra=["--transpose-b"])
    gemm("bf16_aligned", 2048, 256, 128)
    gemm("bf16_aligned_sync", 2048, 256, 128, pipeline="sync")
    gemm("bf16_tails", 257, 259, 65)
    gemm("tf32x3_gate_forward", 4096, 683, 256, dtype="f32", precision="tf32x3",
         extra=["--transpose-b"])
    gemm("f16_aligned", 2048, 256, 128, dtype="f16")

    for name, rows, keys, width, repeats in [
        ("online_tiny", 4, 16, 16, 256),
        ("online_s64_d32_rows64", 64, 64, 32, 128),
        ("online_s64_d32_rows4096", 4096, 64, 32, 16),
        ("online_s64_d32_rows32768", 32768, 64, 32, 4),
        ("online_s256_d64", 512, 256, 64, 16),
        ("online_s512_d128", 128, 512, 128, 8),
        ("online_width1", 512, 64, 1, 64),
        ("online_serial_control", 16, 256, 129, 4),
    ]:
        result.append({
            "name": name, "kind": "online", "binary": "online_weight_probe",
            "args": ["--rows", str(rows), "--keys", str(keys), "--width", str(width)],
            "repeats": repeats,
        })
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    result = {}
    for key, query in [
        ("gpu", ["--query-gpu=name,uuid,driver_version,pstate,temperature.gpu,utilization.gpu,power.draw,clocks.sm"]),
        ("compute_processes", ["--query-compute-apps=pid,process_name,used_memory"]),
    ]:
        run = subprocess.run(["nvidia-smi", *query, "--format=csv,noheader"],
                             capture_output=True, text=True, check=True)
        result[key] = run.stdout.strip()
    return result


def summarize(trials):
    baseline = [statistics.median([p["baseline_us"] for p in t["pairs"]]) for t in trials]
    candidate = [statistics.median([p["candidate_us"] for p in t["pairs"]]) for t in trials]
    ratios = [sum(p["baseline_us"] for p in t["pairs"]) /
              sum(p["candidate_us"] for p in t["pairs"]) for t in trials]
    # Hierarchical circular block bootstrap: resample processes, then adjacent
    # timing pairs inside each process. Discovery intervals are descriptive.
    rng = random.Random(20261031)
    boot = []
    for _ in range(2000):
        estimates = []
        for _ in trials:
            pairs = rng.choice(trials)["pairs"]
            sample = []
            while len(sample) < len(pairs):
                start = rng.randrange(len(pairs))
                sample.extend(pairs[(start + offset) % len(pairs)] for offset in range(4))
            sample = sample[:len(pairs)]
            estimates.append(sum(p["baseline_us"] for p in sample) /
                             sum(p["candidate_us"] for p in sample))
        boot.append(statistics.median(estimates))
    boot.sort()
    return {
        "process_trials": len(trials),
        "baseline_median_us": statistics.median(baseline),
        "candidate_median_us": statistics.median(candidate),
        "median_process_throughput_ratio": statistics.median(ratios),
        "process_throughput_ratios": ratios,
        "bootstrap_95_interval": [boot[49], boot[1949]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh result directory")
    parser.add_argument("--bin-dir", type=Path, default=Path("target/release/examples"))
    parser.add_argument("--kind", choices=["all", "gemm", "online"], default="all")
    parser.add_argument("--cases", help="Comma-separated case names; default is the bounded matrix")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--samples", type=int, default=48)
    args = parser.parse_args()
    if args.trials < 1 or args.samples < 8:
        parser.error("positive trials and at least eight samples required")
    wanted = set(args.cases.split(",")) if args.cases else None
    matrix = [c for c in cases() if (args.kind == "all" or c["kind"] == args.kind)
              and (wanted is None or c["name"] in wanted)]
    if not matrix or (wanted and wanted != {c["name"] for c in matrix}):
        parser.error("unknown case names or empty case selection")
    for case in matrix:
        if not (args.bin_dir / case["binary"]).is_file():
            parser.error("build the release probe binaries before benchmarking")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "kind": "resident_kernel_screen", "gpu_executed": True,
        "scope": "resident repeated buffers, unchanged clocks; no transfers/JIT/readback in events; online uses precomputed scores and one shared values matrix, not full attention",
        "protocol": "serial GPU lock; independent processes with reversed construction/first-pair order; warmup >=200ms; alternating AB/BA CUDA graph/event pairs; all samples retained",
        "interval": "2000 hierarchical process/circular-block bootstrap draws, block4, seed20261031; descriptive intervals do not prove cross-workload generality",
        "binary_sha256": {name: digest(args.bin_dir / name)
                          for name in sorted({c["binary"] for c in matrix})},
        "driver_sha256": digest(Path(__file__)),
        "source_sha256": {str(p): digest(p) for p in sorted([
            Path("Cargo.toml"), Path("Cargo.lock"),
            Path("examples/shared_memory_probe.rs"), Path("examples/online_weight_probe.rs"),
            Path("src/ptx/executor.rs"), Path("src/tile/graph.rs"),
            Path("src/tile/matmul_schedule.rs"), Path("src/tile/normalized_schedule.rs"),
            Path("src/ptx/graph.rs"), Path("src/tile/layout.rs"),
        ])},
        "started_unix": time.time(), "cases": [],
    }
    output = args.output / "report.json"
    with open("/tmp/tnsr-bench.cuda0.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report["before"] = snapshot()
        if report["before"]["compute_processes"]:
            raise RuntimeError("another CUDA compute process is present; no timings started")
        for case in matrix:
            entry = {"name": case["name"], "kind": case["kind"], "trials": []}
            report["cases"].append(entry)
            for trial in range(args.trials):
                order = "baseline,candidate" if trial % 2 == 0 else "candidate,baseline"
                command = [str(args.bin_dir / case["binary"]), *case["args"],
                           "--order", order, "--samples", str(args.samples),
                           "--repeats", str(case["repeats"]), "--warmup-ms", "200"]
                before = snapshot()
                if before["compute_processes"]:
                    raise RuntimeError("another CUDA compute process appeared; stopping")
                run = subprocess.run(command, capture_output=True, text=True, timeout=240,
                                     env={**os.environ, "TNSR_REQUIRE_CUDA": "1"})
                stem = f"{case['name']}-trial{trial}"
                (args.output / (stem + ".stdout")).write_text(run.stdout)
                (args.output / (stem + ".stderr")).write_text(run.stderr)
                if run.returncode:
                    report["failure"] = {"case": case["name"], "trial": trial,
                                         "command": command, "exit_code": run.returncode}
                    output.write_text(json.dumps(report, indent=2) + "\n")
                    raise RuntimeError(f"probe failed: {stem}; see saved stderr")
                raw = json.loads(run.stdout)
                pairs = raw["pairs"]
                if len(pairs) != args.samples:
                    raise RuntimeError("incomplete paired samples")
                for i, pair in enumerate(pairs):
                    expected = order.split(",")
                    if i % 2:
                        expected.reverse()
                    if pair.get("execution_order", pair.get("order")) != expected:
                        raise RuntimeError("probe did not alternate the declared order")
                entry["trials"].append({"command": command, "before": before,
                                         "after": snapshot(), **raw})
                entry["summary"] = summarize(entry["trials"])
                output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({"case": case["name"], "trial": trial,
                                  **entry["summary"]}), flush=True)
        report["after"] = snapshot()
        report["finished_unix"] = time.time()
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
