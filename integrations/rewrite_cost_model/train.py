#!/usr/bin/env python3
"""Fit an experimental whole-graph latency model with leakage-resistant validation.

See docs/learned-rewrite-cost.md. Input is JSONL, one measured candidate per row.
This tool does not modify the Rust optimizer or enable a model automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


IDENTITY = ("backend", "device", "precision")
REQUIRED = (*IDENTITY, "family", "shape_key", "case_id", "variant", "mean_ns", "features")
FORBIDDEN_FEATURES = {"mean_ns", "latency_ns", "elapsed_ns", "sample_means_ns", "heuristic_cost",
                      "family", "shape_key", "case_id", "variant", "graph_key", *IDENTITY}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    identities: set[tuple[str, ...]] = set()
    seen = set()
    cases: dict[str, tuple[str, str]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [key for key in REQUIRED if key not in row]
        if missing:
            raise ValueError(f"line {line_number}: missing fields {missing}")
        if row.get("schema_version", 1) != 1:
            raise ValueError(f"line {line_number}: unsupported schema_version")
        if not math.isfinite(row["mean_ns"]) or row["mean_ns"] <= 0:
            raise ValueError(f"line {line_number}: mean_ns must be finite and positive")
        for name, value in row["features"].items():
            if name in FORBIDDEN_FEATURES:
                raise ValueError(f"line {line_number}: feature {name!r} is a label or identifier")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"line {line_number}: feature {name!r} must be finite and nonnegative")
        if "heuristic_cost" in row and (not math.isfinite(row["heuristic_cost"]) or row["heuristic_cost"] < 0):
            raise ValueError(f"line {line_number}: heuristic_cost must be finite and nonnegative")
        key = (row["case_id"], row["variant"])
        if key in seen:
            raise ValueError(f"line {line_number}: duplicate case/variant {key}")
        seen.add(key)
        shape_family = (row["shape_key"], row["family"])
        if row["case_id"] in cases and cases[row["case_id"]] != shape_family:
            raise ValueError(f"line {line_number}: case_id changes family or shape_key")
        cases[row["case_id"]] = shape_family
        identities.add(tuple(row[name] for name in IDENTITY))
        rows.append(row)
    if not rows:
        raise ValueError("dataset is empty")
    if len(identities) != 1:
        raise ValueError("train each backend/device/precision separately; dataset mixes identities")
    for case_id in cases:
        candidates = [row for row in rows if row["case_id"] == case_id]
        if len(candidates) < 2 or not any(row["variant"] == "original" for row in candidates):
            raise ValueError(f"{case_id}: need original and at least one alternative")
    return rows



# Available only after executing a compiled candidate in the current harness.
PLAN_FEATURES = {"launch_count", "materialized_bytes", "intermediate_bytes", "device_copies",
                 "fused_region_count", "generated_kernel_count"}


def deduplicate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge duplicate graph measurements without selecting the fastest alias."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        graph = row.get("graph_key") or f"variant:{row['variant']}"
        groups.setdefault((row["case_id"], graph), []).append(row)
    output, duplicates = [], []
    for (case_id, graph), aliases in groups.items():
        representative = min(aliases, key=lambda row: (row["variant"] != "original",
                                                       row["variant"] != "optimized", row["variant"]))
        merged = dict(representative)
        merged["variant_aliases"] = sorted({row["variant"] for row in aliases})
        merged["mean_ns"] = sum(row["mean_ns"] for row in aliases) / len(aliases)
        intervals = [row["ci95_ns"] for row in aliases if "ci95_ns" in row]
        if len(intervals) == len(aliases):
            # A conservative envelope exposes between-alias/run drift.
            merged["ci95_ns"] = [min(interval[0] for interval in intervals),
                                    max(interval[1] for interval in intervals)]
        samples = [sample for row in aliases for sample in row.get("sample_means_ns", [])]
        if samples:
            merged["sample_means_ns"] = samples
        if len(aliases) > 1:
            names = {key for row in aliases for key in row["features"]}
            for name in names:
                values = [row["features"].get(name, 0.0) for row in aliases]
                if not np.allclose(values, values[0], rtol=1e-9, atol=1e-9):
                    raise ValueError(f"{case_id}: identical graph_key has conflicting feature {name}")
            if all("heuristic_cost" in row for row in aliases):
                costs = [row["heuristic_cost"] for row in aliases]
                if not np.allclose(costs, costs[0], rtol=1e-9, atol=1e-9):
                    raise ValueError(f"{case_id}: identical graph_key has conflicting heuristic_cost")
            duplicates.append({"case_id": case_id, "graph_key": graph,
                               "aliases": merged["variant_aliases"], "merged_mean_ns": merged["mean_ns"]})
        output.append(merged)
    informative = {case for case in {row["case_id"] for row in output}
                   if sum(row["case_id"] == case for row in output) > 1}
    dropped = sorted({row["case_id"] for row in output} - informative)
    output = [row for row in output if row["case_id"] in informative]
    if not output:
        raise ValueError("no cases with distinct candidate graphs remain after deduplication")
    return output, {"raw_rows": len(rows), "deduplicated_rows": len(output),
                    "merged_duplicate_groups": duplicates, "no_choice_cases_dropped": dropped}

def design(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    return np.log1p(np.array([[row["features"].get(name, 0.0) for name in names] for row in rows], dtype=float))


def fit(rows: list[dict[str, Any]], names: list[str], alpha: float) -> dict[str, Any]:
    x = design(rows, names)
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = np.column_stack([np.ones(len(rows)), (x - center) / scale])
    y = np.log([row["mean_ns"] for row in rows])
    penalty = np.eye(z.shape[1]) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.lstsq(z.T @ z + penalty, z.T @ y, rcond=None)[0]
    return {
        "feature_names": names,
        "transform": "log1p",
        "center": center.tolist(),
        "scale": scale.tolist(),
        "intercept": float(weights[0]),
        "weights": weights[1:].tolist(),
        "train_min_log1p": x.min(axis=0).tolist(),
        "train_max_log1p": x.max(axis=0).tolist(),
        "ridge_alpha": alpha,
    }


def predict(model: dict[str, Any], rows: list[dict[str, Any]]) -> np.ndarray:
    x = design(rows, model["feature_names"])
    z = (x - model["center"]) / model["scale"]
    return model["intercept"] + z @ model["weights"]


def within_coverage(model: dict[str, Any], row: dict[str, Any]) -> bool:
    # A small floating-point tolerance, not an extrapolation allowance.
    x = design([row], model["feature_names"])[0]
    return bool(np.all(x >= np.array(model["train_min_log1p"]) - 1e-9)
                and np.all(x <= np.array(model["train_max_log1p"]) + 1e-9))


def grouped_predictions(rows: list[dict[str, Any]], names: list[str], alpha: float,
                        group_field: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    predictions = np.full(len(rows), np.nan)
    folds = []
    groups = sorted({row[group_field] for row in rows})
    for group in groups:
        train_ids = [i for i, row in enumerate(rows) if row[group_field] != group]
        test_ids = [i for i, row in enumerate(rows) if row[group_field] == group]
        if not train_ids:
            continue
        model = fit([rows[i] for i in train_ids], names, alpha)
        predictions[test_ids] = predict(model, [rows[i] for i in test_ids])
        folds.append({"group": group, "train_rows": len(train_ids), "test_rows": len(test_ids)})
    return predictions, folds


def geometrical_mean(values: list[float]) -> float | None:
    return float(math.exp(sum(map(math.log, values)) / len(values))) if values else None



def candidate_priority(row: dict[str, Any]) -> tuple[int, str]:
    variant = row["variant"]
    return ({"original": 0, "optimized": 1}.get(variant, 2), variant)

def evaluate(rows: list[dict[str, Any]], predictions: np.ndarray) -> dict[str, Any]:
    usable = np.isfinite(predictions)
    actual_log = np.log([row["mean_ns"] for row in rows])
    residuals = np.abs(predictions[usable] - actual_log[usable])
    regrets, heuristic_regrets, optimizer_regrets, speedups, selections, ci_aware_regrets = [], [], [], [], [], []
    correct_pairs = pairs = 0
    for case_id in sorted({row["case_id"] for row in rows}):
        ids = [i for i, row in enumerate(rows) if row["case_id"] == case_id]
        if not np.all(usable[ids]):
            continue
        original = next(i for i in ids if rows[i]["variant"] == "original")
        oracle = min(ids, key=lambda i: rows[i]["mean_ns"])
        selected = min(ids, key=lambda i: (predictions[i], candidate_priority(rows[i])))
        oracle_ns = rows[oracle]["mean_ns"]
        selected_ns = rows[selected]["mean_ns"]
        regrets.append(selected_ns / oracle_ns)
        selected_ci = rows[selected].get("ci95_ns")
        oracle_ci = rows[oracle].get("ci95_ns")
        overlap = (selected_ci is not None and oracle_ci is not None
                   and max(selected_ci[0], oracle_ci[0]) <= min(selected_ci[1], oracle_ci[1]))
        near_tie = selected_ns <= oracle_ns * 1.05 or overlap
        ci_aware_regrets.append(1.0 if near_tie else selected_ns / oracle_ns)
        speedups.append(rows[original]["mean_ns"] / selected_ns)
        if all("heuristic_cost" in rows[i] for i in ids):
            heuristic = min(ids, key=lambda i: (rows[i]["heuristic_cost"], candidate_priority(rows[i])))
            heuristic_regrets.append(rows[heuristic]["mean_ns"] / oracle_ns)
        optimized = next((i for i in ids if "optimized" in rows[i].get("variant_aliases", [rows[i]["variant"]])), None)
        if optimized is not None:
            optimizer_regrets.append(rows[optimized]["mean_ns"] / oracle_ns)
        selections.append({"case_id": case_id, "selected": rows[selected]["variant"],
                           "oracle": rows[oracle]["variant"], "regret": selected_ns / oracle_ns,
                           "ci_overlap_with_oracle": overlap, "near_tie_with_oracle": near_tie,
                           "speedup_vs_original": rows[original]["mean_ns"] / selected_ns})
        for at, a in enumerate(ids):
            for b in ids[at + 1:]:
                # Exclude near ties: ranking a few-percent timing noise is not success/failure.
                ratio = rows[a]["mean_ns"] / rows[b]["mean_ns"]
                if 1 / 1.05 <= ratio <= 1.05:
                    continue
                if "ci95_ns" in rows[a] and "ci95_ns" in rows[b]:
                    a_ci, b_ci = rows[a]["ci95_ns"], rows[b]["ci95_ns"]
                    if max(a_ci[0], b_ci[0]) <= min(a_ci[1], b_ci[1]):
                        continue
                pairs += 1
                correct_pairs += int((predictions[a] < predictions[b]) == (ratio < 1))
    return {
        "rows_evaluated": int(usable.sum()), "cases_evaluated": len(regrets),
        "log_latency_rmse": float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) else None,
        "absolute_log_error_p90": float(np.quantile(residuals, 0.90)) if len(residuals) else None,
        "pairwise_accuracy_excluding_ties": correct_pairs / pairs if pairs else None,
        "pairwise_comparisons": pairs,
        "geomean_regret_vs_oracle": geometrical_mean(regrets),
        "ci_aware_geomean_regret_vs_oracle": geometrical_mean(ci_aware_regrets),
        "max_regret_vs_oracle": max(regrets, default=None),
        "optimizer_cases_evaluated": len(optimizer_regrets),
        "optimizer_geomean_regret_vs_oracle": geometrical_mean(optimizer_regrets) if len(optimizer_regrets) == len(regrets) else None,
        "heuristic_cases_evaluated": len(heuristic_regrets),
        "heuristic_geomean_regret_vs_oracle": geometrical_mean(heuristic_regrets) if len(heuristic_regrets) == len(regrets) else None,
        "geomean_speedup_vs_original": geometrical_mean(speedups), "selections": selections,
    }


def choose(model: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    """Conservative portable inference: original outside coverage or margin."""
    original = next(row for row in candidates if row["variant"] == "original")
    wrong_identity = any(row.get(key) != value for row in candidates
                         for key, value in model["identity"].items())
    if wrong_identity or not model.get("eligible_for_opt_in", False) or any(not within_coverage(model, row) for row in candidates):
        return original["variant"]
    scores = predict(model, candidates)
    index = min(range(len(candidates)), key=lambda i: (scores[i], candidate_priority(candidates[i])))
    original_index = next(i for i, row in enumerate(candidates) if row["variant"] == "original")
    if scores[original_index] - scores[index] <= model["decision_margin_log"]:
        return original["variant"]
    return candidates[index]["variant"]


def train(rows: list[dict[str, Any]], alpha: float, min_shapes: int, min_families: int,
          feature_mode: str = "all") -> tuple[dict[str, Any], dict[str, Any]]:
    data_fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"),
                                                allow_nan=False).encode()).hexdigest()
    rows, deduplication = deduplicate_rows(rows)
    # Only numerical program features: identifiers/family/device do not enter x.
    names = sorted({name for row in rows for name in row["features"]}
                   - (PLAN_FEATURES if feature_mode == "static" else set()))
    if set(names) & FORBIDDEN_FEATURES:
        raise ValueError("program features contain labels or identifiers")
    if not names:
        raise ValueError("dataset has no program features")
    validations = {}
    for field in ("shape_key", "family"):
        predictions, folds = grouped_predictions(rows, names, alpha, field)
        validations[field] = {"folds": folds, **evaluate(rows, predictions)}
    shape_count = len({row["shape_key"] for row in rows})
    family_count = len({row["family"] for row in rows})
    reasons = []
    if shape_count < min_shapes:
        reasons.append(f"only {shape_count} shape groups; need >= {min_shapes}")
    if family_count < min_families:
        reasons.append(f"only {family_count} families; need >= {min_families}")
    for field, validation in validations.items():
        regret = validation["geomean_regret_vs_oracle"]
        heuristic = validation["heuristic_geomean_regret_vs_oracle"]
        if regret is None or heuristic is None or regret >= heuristic:
            reasons.append(f"{field} held-out regret does not beat the supplied heuristic")
        current_optimizer = validation["optimizer_geomean_regret_vs_oracle"]
        if regret is not None and current_optimizer is not None and regret >= current_optimizer:
            reasons.append(f"{field} held-out regret does not beat the supplied current optimizer output")
        if validation["max_regret_vs_oracle"] is None or validation["max_regret_vs_oracle"] > 1.10:
            reasons.append(f"{field} held-out worst-case regret exceeds 10% or is unavailable")
    model = fit(rows, names, alpha)
    model.update({"schema_version": 1, "model_kind": "whole_graph_log_latency_ridge",
                  "identity": {name: rows[0][name] for name in IDENTITY},
                  "training_data_sha256": data_fingerprint,
                  "eligible_for_opt_in": not reasons,
                  "decision_margin_log": max([math.log(1.05)] +
                                             [2 * v["absolute_log_error_p90"] for v in validations.values()
                                              if v["absolute_log_error_p90"] is not None]),
                  "fallback": "original", "experimental": True,
                  "feature_mode": feature_mode,
                  "requires_executed_candidate_features": bool(set(names) & PLAN_FEATURES)})
    report = {"schema_version": 1, "rows": len(rows), "deduplication": deduplication,
              "feature_mode": feature_mode, "training_data_sha256": data_fingerprint, "case_count": len({row["case_id"] for row in rows}),
              "shape_group_count": shape_count, "family_count": family_count,
              "validation": validations, "eligible_for_opt_in": not reasons,
              "ineligibility_reasons": reasons,
              "decision_margin_log": model["decision_margin_log"],
              "note": "Held-out results describe the unguarded ranker. The final coverage/margin gate is conservative and is not claimed as independently validated."}
    return model, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--feature-mode", choices=["all", "static"], default="all",
                        help="static excludes current post-execution plan/counter features")
    parser.add_argument("--min-shapes", type=int, default=12)
    parser.add_argument("--min-families", type=int, default=5)
    args = parser.parse_args()
    if not math.isfinite(args.ridge_alpha) or args.ridge_alpha <= 0 or args.min_shapes < 2 or args.min_families < 2:
        parser.error("ridge-alpha must be positive and minimum group counts must be >=2")
    rows = load_rows(args.dataset)
    model, report = train(rows, args.ridge_alpha, args.min_shapes, args.min_families, args.feature_mode)
    for path, payload in ((args.output, model), (args.report, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rows": len(rows), "eligible_for_opt_in": model["eligible_for_opt_in"],
                      "ineligibility_reasons": report["ineligibility_reasons"]}, indent=2))


if __name__ == "__main__":
    main()
