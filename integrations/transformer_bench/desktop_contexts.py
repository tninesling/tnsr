"""Explicit, sampled observation of at most two tiny resident desktop contexts.

This is not an empty CUDA inventory and does not prove absence of sub-sample
activity. The default benchmark policy still rejects every foreign CUDA context.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import subprocess
import threading
import time
import xml.etree.ElementTree as ET

NAMES = {"xdg-desktop-portal-gnome", "snapd-desktop-integration"}
MAX_MIB = 16
MAX_SAMPLE_GAP_SECONDS = 3.0
POLICY = "explicit_idle_desktop_C+G_services_v2"


def start_ticks(pid):
    text = Path(f"/proc/{pid}/stat").read_text()
    return int(text[text.rindex(")") + 2:].split()[19])


def admit_descriptor(pid, name, kind, memory, ticks):
    if Path(name).name not in NAMES or kind != "C+G" or not 0 <= memory <= MAX_MIB:
        raise ValueError("context is not a small CUDA/graphics desktop service")
    return dict(pid=int(pid), name=name, type=kind, baseline_mib=memory,
                process_start_ticks=ticks)


def validate_allowed(allowed):
    names = []
    for pid, item in allowed.items():
        if item != admit_descriptor(pid, item["name"], item["type"],
                                    item["baseline_mib"], item["process_start_ticks"]):
            raise ValueError("inconsistent desktop context descriptor")
        names.append(Path(item["name"]).name)
    if len(names) > 2 or len(set(names)) != len(names):
        raise ValueError("desktop policy permits a unique-name subset of at most two services")


def descriptors():
    raw = subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=15)
    gpus = ET.fromstring(raw).findall("gpu")
    if len(gpus) != 1:
        raise ValueError("desktop-context mode requires exactly one visible GPU")
    admitted = {}
    for item in gpus[0].findall("./processes/process_info"):
        kind = item.findtext("type", "").strip()
        if "C" not in kind:
            continue
        pid = int(item.findtext("pid"))
        name = item.findtext("process_name", "").strip()
        memory = int(item.findtext("used_memory").split()[0])
        if pid in admitted:
            raise ValueError("duplicate CUDA process descriptor")
        admitted[pid] = admit_descriptor(pid, name, kind, memory, start_ticks(pid))
    validate_allowed(admitted)
    return admitted, raw


def validate_current(current, expected):
    """Keep original identities and allocation ceilings; never rebaseline a worker."""
    validate_allowed(expected)
    if current.keys() != expected.keys():
        raise ValueError("desktop CUDA context set changed")
    for pid, item in current.items():
        baseline = expected[pid]
        if (any(item[k] != baseline[k] for k in ["pid", "name", "type", "process_start_ticks"])
                or not 0 <= item["baseline_mib"] <= baseline["baseline_mib"]):
            raise ValueError(f"desktop context identity/allocation changed: {pid}")


def freeze(directory):
    """Call once under the campaign's GPU lock; pass this set to every execute()."""
    allowed, xml = descriptors()
    directory.joinpath("desktop-context-campaign.xml").write_text(xml)
    directory.joinpath("desktop-context-campaign.json").write_text(json.dumps(dict(
        policy=POLICY, contexts=list(allowed.values()), max_framebuffer_mib=MAX_MIB,
        max_sample_gap_seconds=MAX_SAMPLE_GAP_SECONDS), indent=2) + "\n")
    return allowed


def classify_csv(raw, allowed, own_pid=None, identity=start_ticks):
    """Return competing processes; malformed data is an error, never an idle sample."""
    foreign = []
    for row in csv.reader(io.StringIO(raw)):
        if not row or not any(x.strip() for x in row):
            continue
        if len(row) not in {2, 3}:
            raise ValueError("unrecognized CUDA process inventory row")
        pid, memory = int(row[0].strip()), int(row[-1].strip())
        if memory < 0:
            raise ValueError("negative CUDA process memory")
        if pid == own_pid:
            continue
        expected = allowed.get(pid)
        try:
            matches = (expected is not None and memory <= expected["baseline_mib"]
                       and identity(pid) == expected["process_start_ticks"])
        except (FileNotFoundError, ProcessLookupError):
            matches = False
        if not matches:
            foreign.append(dict(pid=pid, whole_process_memory_mib=memory))
    return foreign


def pmon_records(raw, allowed):
    rows = []
    for line in raw.splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if len(parts) < 12:
            raise ValueError("unrecognized process-utilization row")
        if parts[1] == "-":
            continue
        gpu, pid = int(parts[0]), int(parts[1])
        if pid not in allowed:
            continue  # The independent 50ms CUDA inventory rejects unknown workers.
        metrics = [0 if p == "-" else int(p) for p in parts[3:9]]
        framebuffer = int(parts[9])
        active = (gpu != 0 or parts[2] != "C+G" or any(x != 0 for x in metrics)
                  or not 0 <= framebuffer <= allowed[pid]["baseline_mib"])
        rows.append(dict(pid=pid, type=parts[2], sm=metrics[0], memory=metrics[1],
                         encoder=metrics[2], decoder=metrics[3], jpeg=metrics[4],
                         optical_flow=metrics[5], framebuffer_mib=framebuffer, active=active))
    return rows


def coverage_errors(samples, allowed, duration):
    errors = []
    for pid in allowed:
        times = [0.0] + [x["elapsed_seconds"] for x in samples if x["pid"] == pid] + [duration]
        if len(times) == 2:
            errors.append(f"desktop context had no utilization samples: {pid}")
        elif any(b - a > MAX_SAMPLE_GAP_SECONDS for a, b in zip(times, times[1:])):
            errors.append(f"desktop context utilization coverage gap: {pid}")
    return errors


def stop_process(process, errors, label, expect_running=False):
    """Best-effort teardown that cannot skip cleanup of a sibling monitor."""
    try:
        code = process.poll()
        if code is not None:
            if expect_running:
                errors.append(f"{label} exited early: {code}")
            return
        try:
            process.terminate()
        except Exception as error:
            errors.append(f"{label} terminate: {error!r}")
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            errors.append(f"{label} required termination")
            process.kill()
            process.wait(timeout=5)
    except Exception as error:
        errors.append(f"{label} cleanup: {error!r}")


class Monitor:
    def __init__(self, directory, expected):
        self.directory = directory
        self.allowed = {pid: dict(item) for pid, item in expected.items()}
        current, xml = descriptors()
        validate_current(current, self.allowed)
        self.directory.joinpath("desktop-context-before.xml").write_text(xml)
        baseline = subprocess.check_output(
            ["nvidia-smi", "pmon", "-s", "um", "-c", "2"], text=True, timeout=15)
        self.directory.joinpath("desktop-context-baseline-pmon.txt").write_text(baseline)
        rows = pmon_records(baseline, self.allowed)
        if any(x["active"] for x in rows) or set(self.allowed) - {x["pid"] for x in rows}:
            raise ValueError("desktop CUDA context is active or unobservable")
        self.samples, self.violations, self.errors = [], [], []
        self.started = None
        self.process = None
        self.reader = None
        self.ready = threading.Event()
        self.stopping = threading.Event()
        self.raw_stream = None
        self.stderr_stream = None

    def start(self):
        self.started = time.monotonic()
        self.raw_stream = self.directory.joinpath("desktop-context-pmon.txt").open("w")
        self.stderr_stream = self.directory.joinpath("desktop-context-pmon-stderr.txt").open("w+")
        self.process = subprocess.Popen(["nvidia-smi", "pmon", "-s", "um"],
                                        stdout=subprocess.PIPE, stderr=self.stderr_stream,
                                        text=True, bufsize=1)

        def read():
            try:
                for line in self.process.stdout:
                    self.raw_stream.write(line)
                    self.raw_stream.flush()
                    for row in pmon_records(line, self.allowed):
                        row["elapsed_seconds"] = time.monotonic() - self.started
                        self.samples.append(row)
                        if row["active"]:
                            self.violations.append(row)
                    if line.strip() and not line.lstrip().startswith("#"):
                        if set(self.allowed) <= {x["pid"] for x in self.samples}:
                            self.ready.set()
                if not self.stopping.is_set():
                    self.errors.append("process-utilization stdout closed before cleanup")
                    self.ready.set()
            except Exception as error:
                self.errors.append(f"process-utilization reader: {error!r}")
                self.ready.set()
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        if not self.ready.wait(timeout=5):
            raise ValueError("process-utilization monitor did not establish coverage")
        if self.errors or self.violations or self.process.poll() is not None:
            raise ValueError("process-utilization monitor failed before worker launch")

    def stop(self):
        self.stopping.set()
        if self.process is not None:
            stop_process(self.process, self.errors, "process-utilization monitor", True)
        if self.reader is not None and self.reader.ident is not None:
            self.reader.join(timeout=5)
            if self.reader.is_alive():
                self.errors.append("process-utilization reader did not close")
        for stream in [self.raw_stream, self.stderr_stream]:
            if stream is not None:
                stream.flush()
                stream.close()
        stderr_path = self.directory / "desktop-context-pmon-stderr.txt"
        if stderr_path.exists() and stderr_path.read_text().strip():
            self.errors.append(stderr_path.read_text())
        try:
            now, xml = descriptors()
            self.directory.joinpath("desktop-context-after.xml").write_text(xml)
            validate_current(now, self.allowed)
        except Exception as error:
            self.errors.append(f"post-worker desktop inventory: {error!r}")
        duration = time.monotonic() - self.started if self.started is not None else 0
        self.errors.extend(coverage_errors(self.samples, self.allowed, duration))
        return dict(policy=POLICY, contexts=list(self.allowed.values()),
                    max_framebuffer_mib=MAX_MIB, max_sample_gap_seconds=MAX_SAMPLE_GAP_SECONDS,
                    process_utilization_samples=self.samples, activity_violations=self.violations,
                    monitor_errors=self.errors,
                    scope="Campaign-frozen desktop PID generations and original allocation ceilings; 50ms CUDA inventory plus approximately 1s pmon sampling, gaps over 3s rejected. Resident desktop contexts are observed, not an empty inventory or proof against sub-sample activity.")
