from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from evaluation_protocol import (aggregate_template_scores, evaluate, load_score_rows)


def capture(capture_id, participant, role, expected, session, partition, scores):
    return [{
        "capture_id": capture_id, "participant_id": participant, "role": role,
        "expected_candidate_id": expected, "session_id": session,
        "condition": "baseline", "partition": partition,
        "candidate_id": candidate, "top3_mean": str(score), "mean": str(score - 0.02),
        "maximum": str(min(1.0, score + 0.01)),
    } for candidate, score in scores.items()]


class EvaluationProtocolTests(unittest.TestCase):
    def test_aggregation_strategies_are_distinct_and_quality_requires_metadata(self):
        values = [0.1, 0.2, 0.3, 0.8, 0.9]
        self.assertAlmostEqual(aggregate_template_scores(values, "top3_mean"), (0.3 + 0.8 + 0.9) / 3)
        self.assertAlmostEqual(aggregate_template_scores(values, "mean"), sum(values) / 5)
        self.assertAlmostEqual(aggregate_template_scores(values, "maximum"), 0.9)
        with self.assertRaises(ValueError):
            aggregate_template_scores(values, "quality_weighted")
        self.assertAlmostEqual(aggregate_template_scores(values, "quality_weighted", [1, 1, 1, 1, 2]),
                               (0.1 + 0.2 + 0.3 + 0.8 + 1.8) / 6)

    def test_held_out_evaluation_reports_far_frr_roc_and_incomplete_cohort(self):
        rows = []
        rows += capture("c1", "person_a", "genuine", "candidate_1", "session_cal", "calibration",
                        {"candidate_1": .90, "candidate_2": .30})
        rows += capture("c2", "person_b", "impostor", "", "session_cal", "calibration",
                        {"candidate_1": .70, "candidate_2": .20})
        rows += capture("c3", "person_a", "genuine", "candidate_1", "session_test", "test",
                        {"candidate_1": .85, "candidate_2": .25})
        rows += capture("c4", "person_c", "impostor", "", "session_test", "test",
                        {"candidate_1": .40, "candidate_2": .35})
        report = evaluate(rows, thresholds=(.65,), frozen_threshold=.65)
        calibration = report["partitions"]["calibration"]["strategies"]["top3_mean"]
        self.assertEqual(calibration["operating_point_sweep"][0]["capture_far"], 1.0)
        self.assertEqual(calibration["operating_point_sweep"][0]["capture_frr"], 0.0)
        self.assertIsNotNone(calibration["roc_eer"])
        self.assertEqual(len(calibration["roc_eer"]["roc_points"]), 201)
        self.assertIn("INCOMPLETE", report["partitions"]["calibration"]["threshold_calibration_adequacy"])
        self.assertIsNone(report["threshold_recommendation"])

    def test_rejects_session_leakage_across_partitions(self):
        rows = []
        rows += capture("c1", "person_a", "genuine", "candidate_1", "same", "calibration",
                        {"candidate_1": .9, "candidate_2": .2})
        rows += capture("c2", "person_b", "impostor", "", "same", "test",
                        {"candidate_1": .2, "candidate_2": .1})
        with self.assertRaisesRegex(ValueError, "both calibration and test"):
            load_score_rows_csv_rows(rows)

    def test_rejects_gallery_change_between_capture_groups(self):
        rows = capture("c1", "person_a", "genuine", "candidate_1", "s1", "calibration",
                       {"candidate_1": .9, "candidate_2": .2})
        rows += capture("c2", "person_b", "impostor", "", "s1", "calibration",
                        {"candidate_1": .2})
        with self.assertRaisesRegex(ValueError, "gallery candidate set changed"):
            load_score_rows_csv_rows(rows)

    def test_score_file_loader_validates_schema_and_values(self):
        rows = capture("c1", "person_a", "genuine", "candidate_1", "s1", "calibration",
                       {"candidate_1": .9, "candidate_2": .2})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scores.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            loaded = load_score_rows(path)
        self.assertEqual(len(loaded), 2)


def load_score_rows_csv_rows(rows):
    """Exercise the same row validation without creating a biometric artifact."""
    required = ("capture_id", "participant_id", "role", "expected_candidate_id",
                "session_id", "condition", "partition", "candidate_id",
                "top3_mean", "mean", "maximum")
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "scores.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=required)
            writer.writeheader()
            writer.writerows(rows)
        return load_score_rows(path)


if __name__ == "__main__":
    unittest.main()
