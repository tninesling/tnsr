#!/usr/bin/env python3
"""Summarize discovery and independent previous-optimizer acceptance runs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/benchmarks"
PRIMARY = ["unary_layout", "unary_reshape", "unary_broadcast", "mean_broadcast", "sum_matmul_cols"]


def load(name):
    rows = [json.loads(line) for line in (DATA / name).read_text().splitlines() if line]
    groups = {}
    for row in rows:
        key = (row["family"], row["shape_key"])
        variants = groups.setdefault(key, {})
        if row["variant"] in variants:
            raise ValueError(f"duplicate case/variant: {key} {row['variant']}")
        variants[row["variant"]] = row
    return groups


def comparison(before, after):
    return {
        "before_ns": before["mean_ns"], "after_ns": after["mean_ns"],
        "before_ci95_ns": before["ci95_ns"], "after_ci95_ns": after["ci95_ns"],
        "speedup": before["mean_ns"] / after["mean_ns"],
        "nonoverlapping_ci": before["ci95_ns"][0] > after["ci95_ns"][1],
    }


def main():
    summary = {"schema_version": 1, "date": "2026-10-29",
               "primary_options": PRIMARY, "scope": "diagnostic graphs, not model-level speedups",
               "minimum_speedup": 1.10, "discovery_samples": 16,
               "acceptance_samples": 20, "acceptance_reverse_order": True,
               "results": {}, "confirmed_discovery_regressions": [], "data_sha256": {}}
    for backend in ("cpu", "cuda"):
        names = [f"rewrite-search-{backend}.jsonl", f"rewrite-search-previous-{backend}.jsonl",
                 f"rewrite-search-accept-{backend}.jsonl"]
        discovery, previous, accepted = [load(name) for name in names]
        for name in names:
            summary["data_sha256"][name] = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        results = {}
        for key, after in accepted.items():
            family, shape = key
            if key not in previous:
                raise ValueError(f"missing baseline {key}")
            result = comparison(previous[key]["optimized"], after["optimized"])
            repeat = comparison(discovery[key]["original"], discovery[key]["optimized"])
            result["discovery_repeat"] = repeat
            result["accepted_profit"] = (result["speedup"] >= 1.10 and result["nonoverlapping_ci"]
                                         and repeat["speedup"] >= 1.10 and repeat["nonoverlapping_ci"])
            results[f"{family}/{shape}"] = result
        summary["results"][backend] = results
        for (family, shape), variants in discovery.items():
            original, optimized = variants["original"], variants["optimized"]
            if optimized["mean_ns"] > 1.10 * original["mean_ns"] and optimized["ci95_ns"][0] > original["ci95_ns"][1]:
                summary["confirmed_discovery_regressions"].append({
                    "backend": backend, "family": family, "shape_key": shape,
                    "slowdown": optimized["mean_ns"] / original["mean_ns"],
                    "original_ns": original["mean_ns"], "optimized_ns": optimized["mean_ns"]})
    summary["five_options_verified_on_both_backends"] = all(
        summary["results"][backend][f"{family}/n{n}"]["accepted_profit"]
        for backend in ("cpu", "cuda") for family in PRIMARY for n in (256, 384))
    if not summary["five_options_verified_on_both_backends"]:
        raise ValueError("five-option acceptance condition not met")
    archive = DATA / "rewrite-search-previous-source.tar.gz"
    if archive.exists():
        summary["previous_source_archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (DATA / "rewrite-search-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for family in PRIMARY:
        print(family, {backend: round(summary["results"][backend][f"{family}/n256"]["speedup"], 3)
                       for backend in ("cpu", "cuda")})
    print("confirmed exceptions:", summary["confirmed_discovery_regressions"])


if __name__ == "__main__":
    main()
