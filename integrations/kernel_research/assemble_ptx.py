#!/usr/bin/env python3
"""Assemble offline kernel experiments; this does not execute CUDA work."""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory of generated .ptx files")
    parser.add_argument("--ptxas", default=shutil.which("ptxas"))
    parser.add_argument("--arch", default="sm_89")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.ptxas or args.jobs < 1:
        parser.error("an available --ptxas and positive --jobs are required")
    sources = sorted(args.directory.glob("*.ptx"))
    if not sources:
        parser.error("no PTX files found")
    destination = args.output.parent / "assembled"
    destination.mkdir(parents=True, exist_ok=True)
    version = subprocess.run(
        [args.ptxas, "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()

    def assemble(source):
        cubin = destination / (source.stem + ".cubin")
        command = [args.ptxas, "-arch=" + args.arch, "-v", str(source), "-o", str(cubin)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        log = result.stdout + result.stderr
        (destination / (source.stem + ".log")).write_text(log)
        functions = []
        for section in re.split(r"Compiling entry function '", log)[1:]:
            function = {"name": section.split("'", 1)[0]}
            for name, pattern in [
                ("registers", r"Used (\d+) registers"),
                ("shared_bytes", r"(\d+) bytes smem"),
                ("spill_store_bytes", r"(\d+) bytes spill stores"),
                ("spill_load_bytes", r"(\d+) bytes spill loads"),
            ]:
                match = re.search(pattern, section)
                function[name] = int(match[1]) if match else None
            functions.append(function)
        return {
            "file": source.name,
            "ptx_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "exit_code": result.returncode,
            "cubin_sha256": (
                hashlib.sha256(cubin.read_bytes()).hexdigest()
                if result.returncode == 0 else None
            ),
            "functions": functions,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        entries = list(pool.map(assemble, sources))
    failures = sum(entry["exit_code"] != 0 for entry in entries)
    report = {
        "kind": "offline_ptx_assembly",
        "gpu_executed": False,
        "assembler": version,
        "architecture": args.arch,
        "files": len(entries),
        "failures": failures,
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"files": len(entries), "failures": failures, "gpu_executed": False}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
