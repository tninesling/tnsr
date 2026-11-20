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
import tarfile
import threading
import time

try:
    from . import desktop_contexts
except ImportError:
    import desktop_contexts

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


def implementation_hashes(archive):
    """Keep candidate provenance stable while later experiments edit the tree."""
    if archive is None:
        paths = [ROOT / "Cargo.toml", ROOT / "Cargo.lock",
                 ROOT / "examples/transformer_training_bench.rs",
                 *sorted((ROOT / "examples/transformer_training_bench").glob("*.rs")),
                 *sorted((ROOT / "src").rglob("*.rs"))]
        hashes = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    else:
        hashes = {}
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                name = member.name
                compilation_input = name in {"Cargo.toml", "Cargo.lock"} or (
                    name.endswith(".rs") and (name.startswith("src/") or
                    name == "examples/transformer_training_bench.rs" or
                    name.startswith("examples/transformer_training_bench/")))
                if member.isfile() and compilation_input:
                    with source.extractfile(member) as stream:
                        hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        if not {"Cargo.toml", "Cargo.lock", "examples/transformer_training_bench.rs"} <= hashes.keys():
            raise ValueError("candidate source archive is missing compilation inputs")
    # These three programs actually execute from the current environment.
    for path in [REFERENCE, Path(__file__).resolve(), Path(__file__).with_name("validate.py").resolve(),
                 Path(desktop_contexts.__file__).resolve()]:
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    return hashes


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


def execute(command, directory, timeout, allow_idle_desktop_contexts=False,
            expected_desktop_contexts=None):
    """Fresh worker with fail-closed samplers and cleanup of every started process.

    Explicit desktop mode requires the descriptor set frozen once by the caller
    under the campaign lock. No worker can silently establish a new baseline.
    """
    if allow_idle_desktop_contexts and expected_desktop_contexts is None:
        raise ValueError("explicit desktop mode requires campaign-frozen descriptors")
    directory.mkdir(parents=True, exist_ok=False)
    before = gpu_snapshot()
    allowed = expected_desktop_contexts if allow_idle_desktop_contexts else {}
    desktop_contexts.validate_allowed(allowed)
    if desktop_contexts.classify_csv(before["compute_processes"], allowed):
        raise RuntimeError("another CUDA compute process is present")
    samples, foreign, sampler_errors = [], [], []
    sampler = reader = desktop = process = None
    desktop_report = None
    timed_out = False
    setup_error = None
    started = time.monotonic()
    sampler_started = None
    sampler_records = []
    sampler_stopping = threading.Event()
    with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr, \
            (directory / "nvml-raw.csv").open("w") as raw_nvml, \
            (directory / "nvml-stderr.log").open("w+") as nvml_stderr:
        def sample_memory():
            try:
                for line in sampler.stdout:
                    raw_nvml.write(line)
                    raw_nvml.flush()
                    if not line.strip():
                        continue
                    row = line.strip().split(",")
                    if len(row) != 2:
                        raise ValueError("unrecognized NVML process row")
                    pid, memory = [int(part.strip()) for part in row]
                    if memory < 0:
                        raise ValueError("negative NVML process memory")
                    now = time.monotonic()
                    sampler_records.append(now)
                    record = dict(elapsed_seconds=now - started, pid=pid,
                                  whole_process_memory_mib=memory)
                    own_pid = process.pid if process is not None else None
                    if pid == own_pid:
                        samples.append(record)
                    elif desktop_contexts.classify_csv(line, allowed):
                        foreign.append(record)
                if not sampler_stopping.is_set():
                    sampler_errors.append("NVML stdout closed before cleanup")
            except Exception as error:
                sampler_errors.append(f"NVML reader: {error!r}")

        try:
            # Start monitoring first. Any later startup failure enters this same
            # cleanup region, including sampler/thread/desktop/worker failures.
            sampler = subprocess.Popen([
                "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits", "--loop-ms=50",
            ], stdout=subprocess.PIPE, stderr=nvml_stderr, text=True, bufsize=1)
            sampler_started = time.monotonic()
            reader = threading.Thread(target=sample_memory, daemon=True)
            reader.start()
            if allow_idle_desktop_contexts:
                desktop = desktop_contexts.Monitor(directory, allowed)
                desktop.start()
            if sampler.poll() is not None or sampler_errors or foreign:
                raise RuntimeError("NVML sampler failed or observed contention before worker launch")
            started = time.monotonic()
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, cwd=ROOT,
                                       env={**os.environ, "TNSR_REQUIRE_CUDA": "1"})
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
        except Exception as error:
            setup_error = repr(error)
            sampler_errors.append(f"worker/monitor lifecycle: {error!r}")
        finally:
            if process is not None:
                desktop_contexts.stop_process(process, sampler_errors, "worker")
            if sampler is not None:
                sampler_stopping.set()
                desktop_contexts.stop_process(sampler, sampler_errors, "NVML sampler", True)
            if reader is not None and reader.ident is not None:
                reader.join(timeout=5)
                if reader.is_alive():
                    sampler_errors.append("NVML reader did not close")
            if desktop is not None:
                try:
                    desktop_report = desktop.stop()
                except Exception as error:
                    sampler_errors.append(f"desktop monitor cleanup: {error!r}")
            nvml_stderr.flush()
            nvml_stderr.seek(0)
            monitor_stderr = nvml_stderr.read()
            if monitor_stderr.strip():
                sampler_errors.append(f"NVML stderr: {monitor_stderr}")
            if process is not None and not samples:
                sampler_errors.append("worker had no NVML memory samples")
            if sampler_started is not None and sampler_records:
                boundaries = [sampler_started, *sampler_records, time.monotonic()]
                if any(b - a > desktop_contexts.MAX_SAMPLE_GAP_SECONDS
                       for a, b in zip(boundaries, boundaries[1:])):
                    sampler_errors.append("NVML process inventory coverage gap")
            elif sampler is not None:
                sampler_errors.append("NVML sampler produced no process records")
        elapsed = time.monotonic() - started
    log = (directory / "stderr.log").read_text()
    exit_code = process.returncode if process is not None else None
    status = "passed" if exit_code == 0 else "failed"
    if timed_out:
        status = "timeout"
    elif exit_code != 0 and any(pattern in log.lower() for pattern in [
        "out of memory", "cuda_error_out_of_memory", "cudaerrormemoryallocation",
    ]):
        status = "out_of_memory"
    try:
        after = gpu_snapshot()
        if desktop_contexts.classify_csv(after["compute_processes"], allowed):
            sampler_errors.append("foreign CUDA context present after worker")
    except Exception as error:
        after = dict(error=repr(error))
        sampler_errors.append(f"post-worker GPU inventory: {error!r}")
    if foreign or sampler_errors or (desktop_report and (
            desktop_report["activity_violations"] or desktop_report["monitor_errors"])):
        status = "contended"
    result = dict(status=status, exit_code=exit_code, command=command,
                  elapsed_seconds=elapsed, before=before, after=after,
                  nvml_samples=samples, foreign_compute_observed=foreign,
                  monitor_errors=sampler_errors, lifecycle_error=setup_error,
                  resident_desktop_context_observation=desktop_report,
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
    parser.add_argument("--baseline-binary", type=Path,
                        help="retained native binary for interleaved before/after comparisons")
    parser.add_argument("--source-archive", type=Path,
                        help="immutable source snapshot corresponding to --binary")
    parser.add_argument("--native-arg", action="append", default=[],
                        help="extra candidate-only native argument; use --native-arg=--flag")
    parser.add_argument("--baseline-native-arg", action="append", default=[],
                        help="extra retained-native argument; use --baseline-native-arg=--flag")
    parser.add_argument("--cases", default="w512_s128,w512_s512,w512_s512_b8,w512_s512_b16,w1024_s128,w1024_s512,w2048_s128")
    parser.add_argument("--modes", default="native,pytorch_matched,pytorch_sdpa")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--allow-idle-desktop-contexts", action="store_true",
                        help="explicitly record/monitor tiny idle CUDA+graphics desktop services; reject activity or any other CUDA context")
    args = parser.parse_args()
    matrix = configs()
    modes = args.modes.split(",")
    allowed_modes = {"native", "native_baseline", "pytorch_matched", "pytorch_sdpa", "pytorch_compiled_sdpa", "pytorch_cudagraph_sdpa"}
    if not modes or len(modes) != len(set(modes)) or not set(modes) <= allowed_modes:
        parser.error("unknown or repeated modes")
    names = args.cases.split(",")
    if any(name not in matrix for name in names) or args.trials < 1 or args.repeats < 1:
        parser.error("known cases and positive trials/repeats required")
    if not args.binary.is_file():
        parser.error("build the release training benchmark first")
    if "native_baseline" in modes and (args.baseline_binary is None or not args.baseline_binary.is_file()):
        parser.error("native_baseline requires an existing --baseline-binary")
    if args.source_archive is not None and not args.source_archive.is_file():
        parser.error("--source-archive must be an existing snapshot")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / "report.json"
    report = dict(kind="complete_decoder_training_sweep", binary_sha256=sha256(args.binary),
                  baseline_binary_sha256=sha256(args.baseline_binary) if args.baseline_binary else None,
                  native_arguments=args.native_arg,
                  baseline_native_arguments=args.baseline_native_arg,
                  candidate_source_archive=(dict(path=str(args.source_archive.resolve()),
                      sha256=sha256(args.source_archive)) if args.source_archive else None),
                  reference_sha256=sha256(REFERENCE), driver_sha256=sha256(Path(__file__)),
                  python=sys.executable, cases=[], modes=modes,
                  allow_idle_desktop_contexts=args.allow_idle_desktop_contexts,
                  warmups=args.warmups, repeats=args.repeats,
                  source_sha256=implementation_hashes(args.source_archive),
                  source_hash_scope="candidate compilation inputs from snapshot if supplied; actual orchestration/reference/validator sources always hashed live",
                  initial_numerical_gates=dict(loss_normalized=0.005, clipped_gradient_rel_l2=0.02,
                      post_weight_rel_l2=0.001, update_rel_l2=0.15,
                      native_own_gradient_float64_adam_atol=2e-7, native_own_gradient_float64_adam_rtol=2e-6),
                  scope="fixed synthetic next-token batches; complete forward/backward/clip/Adam and synchronization; first-step full numerical qualification; no learning-quality claim")

    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    with open("/tmp/tnsr-bench.cuda0.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report["before"] = gpu_snapshot()
        expected_desktop_contexts = (desktop_contexts.freeze(args.output)
                                     if args.allow_idle_desktop_contexts else None)
        report["resident_desktop_context_policy"] = (
            dict(policy=desktop_contexts.POLICY, contexts=list(expected_desktop_contexts.values()),
                 max_sample_gap_seconds=desktop_contexts.MAX_SAMPLE_GAP_SECONDS)
            if expected_desktop_contexts is not None else None)
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
                         runs={mode: [] for mode in modes}, qualification=[],
                         qualification_by_reference={mode: [] for mode in modes if mode != "native"})
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
                    if mode in {"native", "native_baseline"} and available is not None and available < required:
                        run = dict(status="host_memory_preflight_skipped", host_available_bytes=available,
                                   conservative_required_bytes=required)
                    else:
                        common = ["--archive", str(archive), "--output", str(directory / "artifacts"),
                                  "--warmups", str(args.warmups), "--repeats", str(args.repeats)]
                        if mode in {"native", "native_baseline"}:
                            binary = args.baseline_binary if mode == "native_baseline" else args.binary
                            command = [str(binary), *common]
                            if mode == "native":
                                command.extend(args.native_arg)
                            else:
                                command.extend(args.baseline_native_arg)
                        else:
                            command = [sys.executable, str(REFERENCE), "run", *common,
                                       "--attention", "matched" if mode == "pytorch_matched" else "sdpa",
                                       "--optimizer", "fused"]
                            if mode in {"pytorch_compiled_sdpa", "pytorch_cudagraph_sdpa"}:
                                command.append("--compile")
                            if mode == "pytorch_cudagraph_sdpa":
                                command.extend(["--compile-mode", "reduce-overhead", "--compile-loss"])
                        print(json.dumps(dict(event="start", case=name, trial=trial, mode=mode,
                                              parameters=count)), flush=True)
                        run = execute(command, directory, args.timeout, args.allow_idle_desktop_contexts,
                                      expected_desktop_contexts)
                    entry["runs"][mode].append(run)
                    entry["summaries"] = {key: summary(value, config) for key, value in entry["runs"].items()}
                    save()
                    print(json.dumps(dict(event="finish", case=name, trial=trial, mode=mode,
                                          status=run["status"], summary=entry["summaries"][mode])), flush=True)
                references = [mode for mode in ["pytorch_matched", "pytorch_cudagraph_sdpa", "pytorch_compiled_sdpa", "pytorch_sdpa", "native_baseline"]
                              if mode in modes]
                if "native" in modes:
                    for reference in references:
                        if all(entry["runs"][mode][-1]["status"] == "passed" for mode in ["native", reference]):
                            from validate import run_validation
                            validation = run_validation(archive, paths["native"], paths[reference],
                                                        case_dir / f"trial{trial}-{reference}-validation.json")
                            entry["qualification_by_reference"][reference].append(validation)
                            if reference == references[0]:
                                entry["qualification"].append(validation)
                            print(json.dumps(dict(event="qualification", case=name, trial=trial,
                                                  reference=reference, qualified=validation["qualified"],
                                                  gates=validation["gates"])), flush=True)
                            save()
            entry["numerically_qualified"] = (
                all(len(entry["qualification_by_reference"][reference]) == args.trials
                    and all(v["qualified"] for v in entry["qualification_by_reference"][reference])
                    for reference in references)
            ) if "native" in modes and references else None
            save()
        report["after"] = gpu_snapshot()
        report["largest_native_tested_fit"] = max(
            (dict(name=e["name"], parameter_elements=e["parameter_elements"], config=e["config"])
             for e in report["cases"] if any(r["status"] == "passed" for r in e["runs"].get("native", []))),
            key=lambda e: e["parameter_elements"], default=None)
        save()


if __name__ == "__main__":
    main()
