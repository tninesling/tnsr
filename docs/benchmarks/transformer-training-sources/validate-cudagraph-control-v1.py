"""Fixed first-update numerical gates for the shared transformer fixture.

All payloads are memory mapped. Float64 work arrays contain at most 262144
elements; no whole-model Python list or float64 tensor is constructed.
Numerical failures produce an unqualified report. Invalid artifacts raise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CHUNK_ELEMENTS = 262144
THRESHOLDS = {
    "loss_normalized_absolute_error": 0.005,
    "clipped_gradient_relative_l2": 0.02,
    "postweight_relative_l2": 0.001,
    "update_relative_l2": 0.15,
    "native_adam_absolute_floor": 2e-7,
    "native_adam_relative_to_max_expected_weight": 2e-6,
    "norm_denominator_floor": 1e-12,
}


def _json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _filename(directory, name):
    if not isinstance(name, str) or name in {"", ".", ".."} or Path(name).name != name:
        raise ValueError("tensor data path must be a filename")
    return Path(directory) / name


def _integer(value, label, minimum=0):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"invalid integer {label}")
    return value


def _config(value):
    integer_keys = {"width", "layers", "heads", "hidden", "sequence", "batch", "vocab"}
    float_keys = {"epsilon", "learning_rate", "max_grad_norm"}
    if not isinstance(value, dict) or set(value) != integer_keys | float_keys:
        raise ValueError("configuration fields do not match the decoder contract")
    result = {}
    for key in integer_keys:
        result[key] = _integer(value[key], key, 1)
    if result["width"] % result["heads"] or result["vocab"] < 2:
        raise ValueError("invalid head division or vocabulary")
    for key in float_keys:
        raw = value[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"invalid configuration {key}")
        with np.errstate(over="ignore"):
            number = float(np.float32(raw))
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"invalid configuration {key}")
        # Native configuration values are f32. Compare their actual settings,
        # permitting equivalent shortest JSON representations of those values.
        result[key] = number
    return result


def _records(value):
    if not isinstance(value, list) or not value:
        raise ValueError("tensor records must be a nonempty list")
    cursor, names, result = 0, [], []
    fields = {"name", "shape", "offset", "count", "group"}
    for record in value:
        if not isinstance(record, dict) or set(record) != fields:
            raise ValueError("unexpected tensor record fields")
        name = record["name"]
        shape = record["shape"]
        if not isinstance(name, str) or not name:
            raise ValueError("tensor name must be nonempty")
        if not isinstance(shape, list) or not shape:
            raise ValueError(f"invalid tensor shape for {name}")
        count = math.prod(_integer(size, f"{name} shape", 1) for size in shape)
        if _integer(record["offset"], f"{name} offset") != cursor:
            raise ValueError(f"noncontiguous tensor offset for {name}")
        if _integer(record["count"], f"{name} count", 1) != count:
            raise ValueError(f"tensor count/shape mismatch for {name}")
        if record["group"] != "train":
            raise ValueError(f"unexpected tensor group for {name}")
        names.append(name)
        result.append(dict(record))
        cursor += count
    if names != sorted(set(names)):
        raise ValueError("tensor names must be sorted and unique")
    return result, cursor


def _mapped(path, elements):
    if path.stat().st_size != elements * 4:
        raise ValueError(f"tensor file extent mismatch: {path}")
    return np.memmap(path, dtype="<f4", mode="r", shape=(elements,))


def _qualification(directory, config, records, elements):
    metadata = _json(Path(directory) / "qualification.json")
    if _config(metadata.get("config")) != config:
        raise ValueError(f"qualification configuration mismatch: {directory}")
    actual_records, actual_count = _records(metadata.get("tensors"))
    if actual_records != records or actual_count != elements:
        raise ValueError(f"qualification tensor manifest mismatch: {directory}")
    if _integer(metadata.get("optimizer_step"), "optimizer_step", 1) != 1:
        raise ValueError("qualification must describe the first optimizer update")
    if metadata.get("dtype", "f32-le") != "f32-le":
        raise ValueError("qualification payload must be f32-le")
    if metadata.get("offset_units", "f32 elements") != "f32 elements":
        raise ValueError("qualification offsets must count f32 elements")
    for key in ["loss", "gradient_norm_before_clip", "gradient_scale"]:
        if isinstance(metadata.get(key), bool) or not isinstance(metadata.get(key), (int, float)):
            raise ValueError(f"qualification scalar is missing or invalid: {key}")
    gradients = _filename(directory, metadata.get("gradients_file"))
    weights = _filename(directory, metadata.get("weights_file"))
    if gradients.resolve() == weights.resolve():
        raise ValueError("gradient and weight snapshots must be distinct files")
    return metadata, _mapped(gradients, elements), _mapped(weights, elements), [gradients, weights]


class _Comparison:
    def __init__(self):
        self.error_squared = 0.0
        self.reference_squared = 0.0
        self.native_squared = 0.0
        self.max_absolute = 0.0
        self.nonfinite_native = 0
        self.nonfinite_reference = 0

    def add(self, native, reference):
        self.nonfinite_native += int(np.count_nonzero(~np.isfinite(native)))
        self.nonfinite_reference += int(np.count_nonzero(~np.isfinite(reference)))
        if np.isfinite(native).all() and np.isfinite(reference).all():
            difference = native - reference
            self.error_squared += float(np.dot(difference, difference))
            self.reference_squared += float(np.dot(reference, reference))
            self.native_squared += float(np.dot(native, native))
            self.max_absolute = max(self.max_absolute, float(np.max(np.abs(difference))))

    def merge(self, other):
        for name in ["error_squared", "reference_squared", "native_squared",
                     "nonfinite_native", "nonfinite_reference"]:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.max_absolute = max(self.max_absolute, other.max_absolute)

    def metrics(self):
        finite = self.nonfinite_native == self.nonfinite_reference == 0
        return {
            "finite": finite,
            "relative_l2": math.sqrt(self.error_squared) / max(
                math.sqrt(self.reference_squared), THRESHOLDS["norm_denominator_floor"]
            ) if finite else None,
            "absolute_l2": math.sqrt(self.error_squared) if finite else None,
            "max_absolute_error": self.max_absolute if finite else None,
            "native_l2": math.sqrt(self.native_squared) if finite else None,
            "reference_l2": math.sqrt(self.reference_squared) if finite else None,
            "nonfinite_native_elements": self.nonfinite_native,
            "nonfinite_reference_elements": self.nonfinite_reference,
        }


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_validation(archive, native, reference, output):
    archive, native, reference, output = map(Path, (archive, native, reference, output))
    manifest = _json(archive / "manifest.json")
    if manifest.get("version") != 1 or isinstance(manifest.get("version"), bool) or manifest.get("dtype") != "f32-le":
        raise ValueError("unsupported archive version/dtype")
    config = _config(manifest.get("config"))
    records, elements = _records(manifest.get("tensors"))
    initial_path = _filename(archive, manifest.get("data_file"))
    initial = _mapped(initial_path, elements)
    nm, ng, nw, native_paths = _qualification(native, config, records, elements)
    rm, rg, rw, reference_paths = _qualification(reference, config, records, elements)
    paths = [archive / "manifest.json", initial_path, native / "qualification.json",
             reference / "qualification.json", *native_paths, *reference_paths]
    if output.resolve() in {path.resolve() for path in paths}:
        raise ValueError("validation output cannot overwrite an input artifact")
    totals = {name: _Comparison() for name in ["clipped_gradients", "postweights", "updates"]}
    parameters = []
    adam_max_error, adam_max_expected, adam_nonfinite = 0.0, 0.0, 0
    sign_disagreements, small_reference_gradients = 0, 0
    learning_rate = config["learning_rate"]
    epsilon = float(np.float32(1e-8))
    for record in records:
        comparisons = {name: _Comparison() for name in totals}
        local_adam_error, local_adam_expected, local_adam_nonfinite = 0.0, 0.0, 0
        local_signs, local_small = 0, 0
        first, end = record["offset"], record["offset"] + record["count"]
        for start in range(first, end, CHUNK_ELEMENTS):
            stop = min(start + CHUNK_ELEMENTS, end)
            original, native_grad, reference_grad, native_weight, reference_weight = [
                np.asarray(values[start:stop], dtype=np.float64)
                for values in [initial, ng, rg, nw, rw]
            ]
            if not np.isfinite(original).all():
                raise ValueError(f"nonfinite initial fixture weights: {record['name']}")
            comparisons["clipped_gradients"].add(native_grad, reference_grad)
            comparisons["postweights"].add(native_weight, reference_weight)
            comparisons["updates"].add(native_weight - original, reference_weight - original)
            finite = np.isfinite(native_grad) & np.isfinite(reference_grad)
            local_signs += int(np.count_nonzero(finite & (np.signbit(native_grad) != np.signbit(reference_grad))))
            local_small += int(np.count_nonzero(np.isfinite(reference_grad) & (np.abs(reference_grad) <= epsilon)))
            finite_adam = np.isfinite(native_grad) & np.isfinite(native_weight)
            local_adam_nonfinite += int(np.count_nonzero(~finite_adam))
            if finite_adam.all():
                # Independent zero-moment first-step Adam: bias corrections
                # cancel beta1/beta2 in float64. Inputs are already clipped;
                # no additional clipping, weight decay, or BF16 narrowing.
                expected = original - (learning_rate * native_grad) / (np.abs(native_grad) + epsilon)
                local_adam_error = max(local_adam_error, float(np.max(np.abs(native_weight - expected))))
                local_adam_expected = max(local_adam_expected, float(np.max(np.abs(expected))))
        for name, comparison in comparisons.items():
            totals[name].merge(comparison)
        adam_max_error = max(adam_max_error, local_adam_error)
        adam_max_expected = max(adam_max_expected, local_adam_expected)
        adam_nonfinite += local_adam_nonfinite
        sign_disagreements += local_signs
        small_reference_gradients += local_small
        parameters.append({
            "name": record["name"], "shape": record["shape"], "count": record["count"],
            **{name: value.metrics() for name, value in comparisons.items()},
            "native_own_gradient_adam": {
                "finite": local_adam_nonfinite == 0,
                "max_absolute_error": local_adam_error if local_adam_nonfinite == 0 else None,
                "max_absolute_expected_weight": local_adam_expected if local_adam_nonfinite == 0 else None,
                "nonfinite_elements": local_adam_nonfinite,
            },
            "gradient_sign_disagreements": local_signs,
            "reference_gradients_at_most_adam_epsilon": local_small,
        })
    metrics = {name: value.metrics() for name, value in totals.items()}
    scalar_finite = all(math.isfinite(metadata[key]) for metadata in [nm, rm]
                        for key in ["loss", "gradient_norm_before_clip", "gradient_scale"])
    scalar_finite = scalar_finite and all(metadata["gradient_norm_before_clip"] >= 0
                                        and 0 < metadata["gradient_scale"] <= 1 for metadata in [nm, rm])
    loss_finite = math.isfinite(nm["loss"]) and math.isfinite(rm["loss"])
    loss_error = abs(nm["loss"] - rm["loss"]) / max(1.0, abs(rm["loss"])) if loss_finite else None
    adam_limit = (THRESHOLDS["native_adam_absolute_floor"]
                  + THRESHOLDS["native_adam_relative_to_max_expected_weight"] * adam_max_expected)
    metrics["loss"] = {"native": nm["loss"] if math.isfinite(nm["loss"]) else None,
                       "reference": rm["loss"] if math.isfinite(rm["loss"]) else None,
                       "normalized_absolute_error": loss_error}
    metrics["native_own_gradient_adam"] = {
        "finite": adam_nonfinite == 0,
        "max_absolute_error": adam_max_error if adam_nonfinite == 0 else None,
        "max_absolute_expected_weight": adam_max_expected if adam_nonfinite == 0 else None,
        "absolute_error_limit": adam_limit if adam_nonfinite == 0 else None,
        "nonfinite_elements": adam_nonfinite,
        "learning_rate_f32": learning_rate, "epsilon_f32": epsilon,
    }
    metrics["gradient_sign_disagreements"] = sign_disagreements
    metrics["reference_gradients_at_most_adam_epsilon"] = small_reference_gradients
    metrics["gradient_clipping_metadata"] = {
        label: {key: metadata[key] if math.isfinite(metadata[key]) else None
                for key in ["gradient_norm_before_clip", "gradient_scale"]}
        for label, metadata in [("native", nm), ("reference", rm)]
    }
    gates = {
        "finite": scalar_finite and all(value["finite"] for value in metrics.values() if isinstance(value, dict) and "finite" in value),
        "loss": loss_error is not None and loss_error <= THRESHOLDS["loss_normalized_absolute_error"],
        "clipped_gradients": metrics["clipped_gradients"]["finite"] and metrics["clipped_gradients"]["relative_l2"] <= THRESHOLDS["clipped_gradient_relative_l2"],
        "postweights": metrics["postweights"]["finite"] and metrics["postweights"]["relative_l2"] <= THRESHOLDS["postweight_relative_l2"],
        "updates": metrics["updates"]["finite"] and metrics["updates"]["relative_l2"] <= THRESHOLDS["update_relative_l2"],
        "native_own_gradient_adam": adam_nonfinite == 0 and adam_max_error <= adam_limit,
    }
    for parameter in parameters:
        parameter["gates"] = {
            "finite": all(parameter[name]["finite"] for name in totals)
                      and parameter["native_own_gradient_adam"]["finite"],
            **{name: parameter[name]["finite"] and parameter[name]["relative_l2"] <= limit
               for name, limit in [
                   ("clipped_gradients", THRESHOLDS["clipped_gradient_relative_l2"]),
                   ("postweights", THRESHOLDS["postweight_relative_l2"]),
                   ("updates", THRESHOLDS["update_relative_l2"]),
               ]},
            "native_own_gradient_adam": parameter["native_own_gradient_adam"]["finite"]
                and parameter["native_own_gradient_adam"]["max_absolute_error"] <= adam_limit,
        }
        parameter["native_own_gradient_adam"]["absolute_error_limit"] = adam_limit if adam_nonfinite == 0 else None
    result = {
        "schema_version": 1, "qualified": all(gates.values()), "gates": gates,
        "thresholds": dict(THRESHOLDS), "config": manifest["config"],
        "parameter_elements": elements, "parameter_tensors": len(records),
        "optimizer_step": 1, "gradient_convention": "already clipped, not clipped again",
        "global": metrics, "parameters": parameters,
        "per_parameter_gates_are_diagnostic": True,
        "numerical_contract": "first update from the identical archive; all required gates fixed before comparisons; Native-vs-reference weight max error is diagnostic only; optimizer audit uses Native's own clipped gradient",
        "memory_protocol": {"float64_chunk_elements_max": CHUNK_ELEMENTS, "tensor_payloads": "five read-only f32 memory maps; no whole-model float64 copies"},
        "provenance": {str(path): _hash(path) for path in paths},
        "validator_sha256": _hash(Path(__file__)),
        "reference_policy": {key: rm.get(key) for key in ["attention", "optimizer"]},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["archive", "native", "reference", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run_validation(args.archive, args.native, args.reference, args.output)
    print(json.dumps({"qualified": result["qualified"], "gates": result["gates"]}, allow_nan=False))


if __name__ == "__main__":
    main()
