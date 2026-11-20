"""CPU-only artifact and numerical-gate tests; no CUDA/PyTorch required."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SPEC = importlib.util.spec_from_file_location("transformer_validator", Path(__file__).with_name("validate.py"))
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.archive, self.native, self.reference = [self.root / name for name in ["archive", "native", "reference"]]
        for path in [self.archive, self.native, self.reference]:
            path.mkdir()
        self.config = dict(width=2, layers=1, heads=1, hidden=3, sequence=2, batch=1,
                           vocab=3, epsilon=1e-5, learning_rate=1e-4, max_grad_norm=1.0)
        self.original = np.array([0.01, -0.02, 0.03, 1.0, 0.2], dtype="<f4")
        self.gradients = np.array([0.2, -0.4, 0.0, 1e-10, -3e-9], dtype="<f4")
        self.records = [dict(name="a", shape=[2], offset=0, count=2, group="train"),
                        dict(name="b", shape=[3], offset=2, count=3, group="train")]
        self.write_fixture()

    def tearDown(self):
        self.temporary.cleanup()

    def write_fixture(self):
        self.original.tofile(self.archive / "weights.f32")
        manifest = dict(version=1, dtype="f32-le", data_file="weights.f32",
                        config=self.config, tensors=self.records)
        (self.archive / "manifest.json").write_text(json.dumps(manifest))
        gradient64 = self.gradients.astype(np.float64)
        expected = (self.original.astype(np.float64)
                    - float(np.float32(self.config["learning_rate"])) * gradient64
                    / (np.abs(gradient64) + float(np.float32(1e-8)))).astype("<f4")
        self.weights = expected
        for directory in [self.native, self.reference]:
            metadata = dict(config=self.config, tensors=self.records, optimizer_step=1,
                            loss=2.0, gradient_norm_before_clip=1.0, gradient_scale=1.0,
                            gradients_file="qual_gradients.f32", weights_file="qual_weights.f32")
            (directory / "qualification.json").write_text(json.dumps(metadata))
            self.gradients.tofile(directory / "qual_gradients.f32")
            expected.tofile(directory / "qual_weights.f32")

    def run_validation(self):
        result = validator.run_validation(self.archive, self.native, self.reference, self.root / "validation.json")
        self.assertEqual(result, json.loads((self.root / "validation.json").read_text()))
        return result

    def change_metadata(self, directory, function):
        path = directory / "qualification.json"
        value = json.loads(path.read_text())
        function(value)
        path.write_text(json.dumps(value))

    def test_correct_clipped_gradient_and_first_adam_update_qualify(self):
        result = self.run_validation()
        self.assertTrue(result["qualified"])
        self.assertTrue(all(result["gates"].values()))
        self.assertEqual(result["global"]["updates"]["relative_l2"], 0.0)
        self.assertEqual(result["global"]["clipped_gradients"]["relative_l2"], 0.0)
        self.assertLess(result["global"]["native_own_gradient_adam"]["max_absolute_error"], 1e-7)
        self.assertEqual([p["name"] for p in result["parameters"]], ["a", "b"])
        self.assertTrue(all(p["gates"]["native_own_gradient_adam"] for p in result["parameters"]))

    def test_wrong_gradient_fails_without_changing_the_limits(self):
        (self.gradients * np.float32(1.5)).tofile(self.native / "qual_gradients.f32")
        result = self.run_validation()
        self.assertFalse(result["qualified"])
        self.assertFalse(result["gates"]["clipped_gradients"])
        self.assertEqual(result["thresholds"]["clipped_gradient_relative_l2"], 0.02)

    def test_saved_clipped_gradients_are_not_scaled_a_second_time(self):
        for directory in [self.native, self.reference]:
            self.change_metadata(directory, lambda m: m.update(
                gradient_scale=0.25,
                gradient_norm_before_clip=float(np.linalg.norm(self.gradients.astype(np.float64))) / 0.25,
            ))
        result = self.run_validation()
        self.assertTrue(result["qualified"])
        self.assertTrue(result["gates"]["native_own_gradient_adam"])

    def test_wrong_weights_fail_update_and_independent_adam_audit(self):
        (self.weights + np.float32(0.001)).tofile(self.native / "qual_weights.f32")
        result = self.run_validation()
        self.assertFalse(result["qualified"])
        self.assertFalse(result["gates"]["updates"])
        self.assertFalse(result["gates"]["native_own_gradient_adam"])
        self.assertTrue(result["gates"]["clipped_gradients"])

    def test_near_zero_gradient_sign_changes_remain_an_update_failure(self):
        gradients = self.gradients.copy()
        gradients[-1] *= -1
        expected = (self.original.astype(np.float64)
                    - float(np.float32(1e-4)) * gradients.astype(np.float64)
                    / (np.abs(gradients.astype(np.float64)) + float(np.float32(1e-8)))).astype("<f4")
        gradients.tofile(self.native / "qual_gradients.f32")
        expected.tofile(self.native / "qual_weights.f32")
        result = self.run_validation()
        self.assertTrue(result["gates"]["clipped_gradients"])
        self.assertTrue(result["gates"]["native_own_gradient_adam"])
        self.assertFalse(result["gates"]["updates"])
        self.assertFalse(result["qualified"])

    def test_nonfinite_snapshot_is_unqualified_json_not_an_exception(self):
        gradients = self.gradients.copy()
        gradients[0] = np.nan
        gradients.tofile(self.native / "qual_gradients.f32")
        result = self.run_validation()
        self.assertFalse(result["qualified"])
        self.assertFalse(result["gates"]["finite"])
        self.assertIsNone(result["global"]["clipped_gradients"]["relative_l2"])
        self.assertEqual(result["global"]["clipped_gradients"]["nonfinite_native_elements"], 1)

    def test_loss_gate_is_fixed_and_weights_maxabs_is_only_diagnostic(self):
        self.change_metadata(self.reference, lambda m: m.update(loss=2.1))
        result = self.run_validation()
        self.assertFalse(result["gates"]["loss"])
        self.assertNotIn("max_absolute_weight_error", result["gates"])

    def test_malformed_manifests_config_steps_and_filenames_are_rejected(self):
        original_metadata = json.loads((self.native / "qualification.json").read_text())
        mutations = [
            lambda m: m.update(optimizer_step=2),
            lambda m: m.update(optimizer_step=True),
            lambda m: m["config"].update(width=4),
            lambda m: m["tensors"].reverse(),
            lambda m: m["tensors"][1].update(offset=1),
            lambda m: m["tensors"][0].update(count=3),
            lambda m: m["tensors"][0].update(shape=[1, 2]),
            lambda m: m.update(gradients_file="../escape.f32"),
            lambda m: m.update(gradients_file="qual_weights.f32"),
        ]
        for mutation in mutations:
            metadata = copy.deepcopy(original_metadata)
            mutation(metadata)
            (self.native / "qualification.json").write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                self.run_validation()
        (self.native / "qualification.json").write_text(json.dumps(original_metadata))

    def test_payload_extents_and_nonfinite_initial_fixture_are_rejected(self):
        path = self.native / "qual_gradients.f32"
        self.gradients[:-1].tofile(path)
        with self.assertRaises(ValueError):
            self.run_validation()
        np.append(self.gradients, np.float32(0)).astype("<f4").tofile(path)
        with self.assertRaises(ValueError):
            self.run_validation()
        self.gradients.tofile(path)
        original = self.original.copy()
        original[0] = np.inf
        original.tofile(self.archive / "weights.f32")
        with self.assertRaises(ValueError):
            self.run_validation()

    def test_chunk_bound_across_a_large_tensor(self):
        count = validator.CHUNK_ELEMENTS + 7
        self.original = np.full(count, 0.125, dtype="<f4")
        self.gradients = np.full(count, -0.25, dtype="<f4")
        self.records = [dict(name="a", shape=[count], offset=0, count=count, group="train")]
        self.write_fixture()
        original_asarray = validator.np.asarray
        sizes = []

        def bounded_asarray(values, *args, **kwargs):
            if kwargs.get("dtype") == np.float64:
                sizes.append(len(values))
                self.assertLessEqual(len(values), validator.CHUNK_ELEMENTS)
            return original_asarray(values, *args, **kwargs)

        with patch.object(validator.np, "asarray", side_effect=bounded_asarray):
            result = self.run_validation()
        self.assertTrue(result["qualified"])
        self.assertIn(7, sizes)
        self.assertIn(validator.CHUNK_ELEMENTS, sizes)


if __name__ == "__main__":
    unittest.main()
