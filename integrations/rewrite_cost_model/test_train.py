"""Behavioral checks for the offline model protocol; no hardware timing needed."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import train


def fixture():
    rows = []
    for family in range(5):
        for shape in range(1, 13):
            case = f"family{family}/shape{shape}"
            for variant, work in [("original", shape * 100), ("rewrite", shape * 30 + family)]:
                rows.append({"schema_version": 1, "backend": "cpu", "device": "fixture",
                             "precision": "f32", "family": f"family{family}",
                             "shape_key": str(shape), "case_id": case, "variant": variant,
                             "mean_ns": 1000 * (1 + work) ** 0.7,
                             "heuristic_cost": 1 if variant == "original" else 2,
                             "features": {"work": work, "constant_feature": 0}})
    return rows


class TrainingTests(unittest.TestCase):
    def test_held_out_ranker_recovers_known_cross_shape_and_family_relation(self):
        rows = fixture()
        model, report = train.train(rows, 1.0, 12, 5)
        self.assertTrue(model["eligible_for_opt_in"], report["ineligibility_reasons"])
        for validation in report["validation"].values():
            self.assertEqual(validation["cases_evaluated"], 60)
            self.assertEqual(validation["geomean_regret_vs_oracle"], 1.0)
            self.assertEqual(validation["pairwise_accuracy_excluding_ties"], 1.0)
            self.assertGreater(validation["heuristic_geomean_regret_vs_oracle"], 1.0)
        self.assertEqual(train.choose(model, rows[:2]), "rewrite")
        # The serialized numeric protocol preserves inference exactly.
        restored = json.loads(json.dumps(model, allow_nan=False))
        np.testing.assert_allclose(train.predict(model, rows), train.predict(restored, rows))

    def test_out_of_range_features_and_ineligible_models_keep_original(self):
        rows = fixture()
        model, _ = train.train(rows, 1.0, 12, 5)
        candidates = [dict(row, features={"work": 1e15}) for row in rows[:2]]
        self.assertEqual(train.choose(model, candidates), "original")
        wrong_device = [dict(row, device="unseen GPU") for row in rows[:2]]
        self.assertEqual(train.choose(model, wrong_device), "original")
        small_model, report = train.train(rows[:4], 1.0, 12, 5)
        self.assertFalse(small_model["eligible_for_opt_in"])
        self.assertTrue(report["ineligibility_reasons"])
        self.assertEqual(train.choose(small_model, rows[:2]), "original")

    def test_shape_holdout_never_splits_candidate_variants_or_matching_shapes(self):
        rows = fixture()
        predictions, folds = train.grouped_predictions(rows, ["work"], 1.0, "shape_key")
        self.assertTrue(np.all(np.isfinite(predictions)))
        self.assertEqual(len(folds), 12)
        self.assertTrue(all(fold["test_rows"] == 10 for fold in folds))
        self.assertTrue(all(fold["train_rows"] == 110 for fold in folds))

    def test_graph_duplicates_merge_measurements_without_fastest_alias_bias(self):
        base = fixture()[:2]
        original = dict(base[0], graph_key="same", mean_ns=100, ci95_ns=[95, 105])
        manual_alias = dict(original, variant="manual", mean_ns=80, ci95_ns=[75, 85])
        distinct = dict(base[1], graph_key="other", mean_ns=60, ci95_ns=[55, 65])
        rows, metadata = train.deduplicate_rows([original, manual_alias, distinct])
        self.assertEqual(len(rows), 2)
        merged = next(row for row in rows if row["variant"] == "original")
        self.assertEqual(merged["mean_ns"], 90)
        self.assertEqual(merged["ci95_ns"], [75, 105])
        self.assertEqual(merged["variant_aliases"], ["manual", "original"])
        self.assertEqual(len(metadata["merged_duplicate_groups"]), 1)
        conflict = dict(manual_alias, features={"work": 999})
        with self.assertRaisesRegex(ValueError, "conflicting feature"):
            train.deduplicate_rows([original, conflict, distinct])

    def test_no_choice_duplicate_cases_do_not_inflate_validation(self):
        rows = fixture()[:4]
        rows[0]["graph_key"] = rows[1]["graph_key"] = "no-choice"
        rows[1]["features"] = dict(rows[0]["features"])
        rows[1]["heuristic_cost"] = rows[0]["heuristic_cost"]
        deduplicated, metadata = train.deduplicate_rows(rows)
        self.assertEqual(len(deduplicated), 2)
        self.assertEqual(metadata["no_choice_cases_dropped"], [rows[0]["case_id"]])

    def test_static_ablation_excludes_post_execution_counters(self):
        rows = fixture()
        for row in rows:
            row["features"]["launch_count"] = 2 if row["variant"] == "original" else 1
        static, _ = train.train(rows, 1.0, 12, 5, "static")
        all_features, _ = train.train(rows, 1.0, 12, 5, "all")
        self.assertNotIn("launch_count", static["feature_names"])
        self.assertFalse(static["requires_executed_candidate_features"])
        self.assertIn("launch_count", all_features["feature_names"])
        self.assertTrue(all_features["requires_executed_candidate_features"])

    def test_ci_aware_report_does_not_count_indistinguishable_ranking(self):
        rows = fixture()[:2]
        rows[0].update(mean_ns=110, ci95_ns=[90, 130])
        rows[1].update(mean_ns=100, ci95_ns=[90, 120])
        report = train.evaluate(rows, np.array([0.0, 1.0]))
        self.assertEqual(report["pairwise_comparisons"], 0)
        self.assertAlmostEqual(report["geomean_regret_vs_oracle"], 1.1)
        self.assertEqual(report["ci_aware_geomean_regret_vs_oracle"], 1.0)

    def test_loader_rejects_mixed_device_and_missing_original(self):
        rows = fixture()[:2]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            mixed = [rows[0], dict(rows[1], device="another device")]
            path.write_text("\n".join(json.dumps(row) for row in mixed))
            with self.assertRaisesRegex(ValueError, "mixes identities"):
                train.load_rows(path)
            path.write_text(json.dumps(rows[1]))
            with self.assertRaisesRegex(ValueError, "original"):
                train.load_rows(path)


if __name__ == "__main__":
    unittest.main()
