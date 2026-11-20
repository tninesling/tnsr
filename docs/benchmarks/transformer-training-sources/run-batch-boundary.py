#!/usr/bin/env python3
"""Complete-update decoder training sweep, with explicit numerical/capacity status."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).with_name("reference.py")


def configs():
    def make(width, sequence, batch, layers=4, vocab=4096):
        return dict(width=width, sequence=sequence, batch=batch, layers=layers,
                    heads=max(1, width // 64), hidden=math.ceil((8 * width / 3) / 64) * 64,
                    vocab=vocab, epsilon=1e-5, learning_rate=1e-4, max_grad_norm=1.0)
    return {
        "smoke": dict(make(64, 16, 2, 1, 128), heads=4, hidden=192),
        "w512_s128": make(512, 128, 2),
        "w512_s512": make(512, 512, 2),
        "w512_s512_b8": make(512, 512, 8),
        "w512_s512_b16": make(512, 512, 16),
        "w512_s512_b32": make(512, 512, 32),
        "w1024_s128": make(1024, 128, 2),
        "w1024_s512": make(1024, 512, 1),
        "w1536_s128": make(1536, 128, 1),
        "w2048_s128": make(2048, 128, 1),
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def gpu_snapshot():
    result = {}
    for key, option in [
        ("device", "--query-gpu=name,uuid,driver_version,memory.total,memory.free,pstate,temperature.gpu,clocks.sm"),
        ("compute_processes", "--query-compute-apps=pid,process_name,used_gpu_memory"),
    ]:
        run = subprocess.run(["nvidia-smi", option, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True)
        result[key] = run.stdout.strip()
    return result


def host_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def execute(command, directory, timeout):
    """Fresh worker, sampled whole-process NVML memory, and preserved failure logs."""
    directory.mkdir(parents=True, exist_ok=False)
    before = gpu_snapshot()
    if before["compute_processes"]:
        raise RuntimeError("another CUDA compute process is present")
    samples, foreign = [], []
    with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
        started = time.monotonic()
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, cwd=ROOT,
                                   env={**os.environ, "TNSR_REQUIRE_CUDA": "1"})
        sampler = subprocess.Popen([
            "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits", "--loop-ms=50",
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

        def sample_memory():
            for line in sampler.stdout:
                try:
                    pid, memory = [part.strip() for part in line.split(",", 1)]
                    pid, memory = int(pid), int(memory)
                except ValueError:
                    continue
                record = dict(elapsed_seconds=time.monotonic() - started, pid=pid,
                              whole_process_memory_mib=memory)
                if pid == process.pid:
                    samples.append(record)
                else:
                    foreign.append(record)

        reader = threading.Thread(target=sample_memory, daemon=True)
        reader.start()
        timed_out = False
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                (directory / "heartbeat.json").write_text(json.dumps(dict(
                    pid=process.pid, elapsed_seconds=elapsed,
                    latest_memory=samples[-1] if samples else None,
                    observed_peak_mib=max((s["whole_process_memory_mib"] for s in samples), default=None))))
                if elapsed > timeout:
                    timed_out = True
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                time.sleep(1)
            process.wait()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            sampler.terminate()
            try:
                sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait()
            reader.join(timeout=5)
        elapsed = time.monotonic() - started
    log = (directory / "stderr.log").read_text()
    status = "passed" if process.returncode == 0 else "failed"
    if timed_out:
        status = "timeout"
    elif process.returncode != 0 and any(pattern in log.lower() for pattern in [
        "out of memory", "cuda_error_out_of_memory", "cudaerrormemoryallocation",
    ]):
        status = "out_of_memory"
    if foreign:
        status = "contended"
    result = dict(status=status, exit_code=process.returncode, command=command,
                  elapsed_seconds=elapsed, before=before, after=gpu_snapshot(),
                  nvml_samples=samples, foreign_compute_observed=foreign,
                  nvml_observed_process_peak_mib=max(
                      (s["whole_process_memory_mib"] for s in samples), default=None),
                  memory_scope="whole process including context/modules; requested 50ms NVML polling; observed maximum is a lower bound on instantaneous peak")
    if (directory / "artifacts/results.json").exists():
        result["result"] = json.loads((directory / "artifacts/results.json").read_text())
    elif status == "passed":
        result["status"] = status = "missing_results"
    if status != "passed":
        result["stderr_tail"] = log[-6000:]
    (directory / "process.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def summary(runs, config):
    valid = [r["result"] for r in runs if r["status"] == "passed"]
    if not valid:
        return None
    medians = [r["timing"]["median_ms"] if "timing" in r else r["median_wall_ms"] for r in valid]
    milliseconds = statistics.median(medians)
    return dict(successful_processes=len(valid), median_complete_update_ms=milliseconds,
                tokens_per_second=config["batch"] * config["sequence"] * 1000 / milliseconds,
                process_medians_ms=medians,
                nvml_observed_process_peak_mib=max(
                    (r["nvml_observed_process_peak_mib"] for r in runs
                     if r.get("nvml_observed_process_peak_mib") is not None), default=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/examples/transformer_training_bench")
    parser.add_argument("--cases", default="w512_s128,w512_s512,w512_s512_b8,w512_s512_b16,w1024_s128,w1024_s512,w2048_s128")
    parser.add_argument("--modes", default="native,pytorch_matched,pytorch_sdpa")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    matrix = configs()
    modes = args.modes.split(",")
    allowed_modes = {"native", "pytorch_matched", "pytorch_sdpa", "pytorch_compiled_sdpa"}
    if not modes or len(modes) != len(set(modes)) or not set(modes) <= allowed_modes:
        parser.error("unknown or repeated modes")
    names = args.cases.split(",")
    if any(name not in matrix for name in names) or args.trials < 1 or args.repeats < 1:
        parser.error("known cases and positive trials/repeats required")
    if not args.binary.is_file():
        parser.error("build the release training benchmark first")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / "report.json"
    report = dict(kind="complete_decoder_training_sweep", binary_sha256=sha256(args.binary),
                  reference_sha256=sha256(REFERENCE), driver_sha256=sha256(Path(__file__)),
                  python=sys.executable, cases=[], modes=modes,
                  warmups=args.warmups, repeats=args.repeats,
                  source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in [
                      ROOT / "Cargo.toml", ROOT / "Cargo.lock",
                      ROOT / "examples/transformer_training_bench.rs",
                      *sorted((ROOT / "examples/transformer_training_bench").glob("*.rs")),
                      REFERENCE, Path(__file__).resolve(), Path(__file__).with_name("validate.py").resolve(),
                      ROOT / "src/ptx/mixed.rs", ROOT / "src/ptx/resident/mod.rs",
                  ]},
                  initial_numerical_gates=dict(loss_normalized=0.005, clipped_gradient_rel_l2=0.02,
                      post_weight_rel_l2=0.001, update_rel_l2=0.15,
                      native_own_gradient_float64_adam_atol=2e-7, native_own_gradient_float64_adam_rtol=2e-6),
                  scope="fixed synthetic next-token batches; complete forward/backward/clip/Adam and synchronization; first-step full numerical qualification; no learning-quality claim")

    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    with open("/tmp/tnsr-bench.cuda0.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report["before"] = gpu_snapshot()
        for name in names:
            config = matrix[name]
            case_dir = args.output / name
            case_dir.mkdir()
            config_path = case_dir / "config.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            archive = case_dir / "archive"
            subprocess.run([sys.executable, str(REFERENCE), "generate", "--config", str(config_path),
                            "--archive", str(archive)], check=True)
            manifest = json.loads((archive / "manifest.json").read_text())
            count = sum(t["count"] for t in manifest["tensors"])
            entry = dict(name=name, config=config, parameter_elements=count,
                         fixture_sha256=sha256(archive / "weights.f32"),
                         runs={mode: [] for mode in modes}, qualification=[])
            report["cases"].append(entry)
            save()
            for trial in range(args.trials):
                order = list(modes)
                if trial % 2:
                    order.reverse()
                paths = {}
                for mode in order:
                    directory = case_dir / f"trial{trial}-{mode}"
                    paths[mode] = directory / "artifacts"
                    available = host_available()
                    # No OS memory limits or driver settings are changed. Keep
                    # headroom for the existing native all-gradient readback API.
                    required = count * 16 + 512 * 1024 * 1024
                    if mode == "native" and available is not None and available < required:
                        run = dict(status="host_memory_preflight_skipped", host_available_bytes=available,
                                   conservative_required_bytes=required)
                    else:
                        common = ["--archive", str(archive), "--output", str(directory / "artifacts"),
                                  "--warmups", str(args.warmups), "--repeats", str(args.repeats)]
                        if mode == "native":
                            command = [str(args.binary), *common]
                        else:
                            command = [sys.executable, str(REFERENCE), "run", *common,
                                       "--attention", "matched" if mode == "pytorch_matched" else "sdpa",
                                       "--optimizer", "fused"]
                            if mode == "pytorch_compiled_sdpa":
                                command.append("--compile")
                        print(json.dumps(dict(event="start", case=name, trial=trial, mode=mode,
                                              parameters=count)), flush=True)
                        run = execute(command, directory, args.timeout)
                    entry["runs"][mode].append(run)
                    entry["summaries"] = {key: summary(value, config) for key, value in entry["runs"].items()}
                    save()
                    print(json.dumps(dict(event="finish", case=name, trial=trial, mode=mode,
                                          status=run["status"], summary=entry["summaries"][mode])), flush=True)
                if {"native", "pytorch_matched"} <= set(modes) and all(entry["runs"][mode][-1]["status"] == "passed"
                       for mode in ["native", "pytorch_matched"]):
                    from validate import run_validation
                    validation = run_validation(archive, paths["native"], paths["pytorch_matched"],
                                                case_dir / f"trial{trial}-validation.json")
                    entry["qualification"].append(validation)
                    print(json.dumps(dict(event="qualification", case=name, trial=trial,
                                          qualified=validation["qualified"], gates=validation["gates"])), flush=True)
                    save()
            entry["numerically_qualified"] = (
                len(entry["qualification"]) == args.trials
                and all(v["qualified"] for v in entry["qualification"])
            ) if {"native", "pytorch_matched"} <= set(modes) else None
            save()
        report["after"] = gpu_snapshot()
        report["largest_native_tested_fit"] = max(
            (dict(name=e["name"], parameter_elements=e["parameter_elements"], config=e["config"])
             for e in report["cases"] if any(r["status"] == "passed" for r in e["runs"].get("native", []))),
            key=lambda e: e["parameter_elements"], default=None)
        save()


if __name__ == "__main__":
    main()
