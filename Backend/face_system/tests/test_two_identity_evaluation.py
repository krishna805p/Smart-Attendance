from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from embedding_store import EmbeddingStore
from detector import FaceDetection
from evaluate_two_students import (
    IdentificationCapture, build_identification_report, capture_two_identity_evaluation,
)
from quality import QualityResult
from recognition import RecognitionEngine


def axis(index: int) -> np.ndarray:
    result = np.zeros(128, dtype=np.float32)
    result[index] = 1.0
    return result


class CosineOnlyRecognizer:
    @staticmethod
    def cosine_similarity(first, second):
        a = np.asarray(first, dtype=np.float32)
        b = np.asarray(second, dtype=np.float32)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def __init__(self, active_roll=None):
        self.active_roll = active_roll or {"value": "TEST001"}

    def generate_embedding(self, _aligned):
        return axis(0 if self.active_roll["value"] == "TEST001" else 1)


class TestCamera:
    def __init__(self):
        self.releases = 0

    def isOpened(self):
        return True

    def set(self, *_args):
        return True

    def release(self):
        self.releases += 1


class TestStream:
    def __init__(self, camera, **_kwargs):
        self.camera = camera

    def read(self):
        return np.zeros((160, 160, 3), dtype=np.uint8)

    def close(self):
        self.camera.release()
        return True


class TestDetector:
    def detect(self, _frame):
        return [FaceDetection((20., 20., 100., 100.),
                              ((40., 50.), (80., 50.), (60., 65.),
                               (45., 85.), (75., 85.)), .99)]


class TwoIdentityEvaluationTests(unittest.TestCase):
    def test_two_identity_candidate_ranking_returns_best_second_and_margin(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = EmbeddingStore(Path(temporary) / "templates.json")
            store.save_student("TEST001", [axis(0)])
            store.save_student("TEST002", [axis(1)])
            engine = RecognitionEngine(store=store, detector=object(),
                                       recognizer=CosineOnlyRecognizer(), threshold=0.65,
                                       minimum_margin=None)
            match = engine.compare_embedding(axis(1))
        self.assertEqual(match.candidate_roll_number, "TEST002")
        self.assertEqual(match.roll_number, "TEST002")
        self.assertEqual(match.second_best_roll_number, "TEST001")
        self.assertAlmostEqual(match.similarity, 1.0)
        self.assertAlmostEqual(match.second_best_similarity, 0.0)
        self.assertAlmostEqual(match.score_margin, 1.0)
        self.assertIsNone(match.required_margin)

    def test_capture_session_reports_two_identities_without_a_margin_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = EmbeddingStore(Path(temporary) / "templates.json")
            store.save_student("TEST001", [axis(0)])
            store.save_student("TEST002", [axis(1)])
            camera = TestCamera()
            active = {"value": "TEST001"}
            keys = iter([ord("1"), 32, ord("n"), ord("2"), 32, ord("n"), ord("f")])

            def poll():
                key = next(keys)
                if key in (ord("1"), ord("2")):
                    active["value"] = "TEST001" if key == ord("1") else "TEST002"
                return key

            with patch("evaluate_two_students.assess_face_quality",
                       return_value=QualityResult(True, "accepted", 100., 100.)), \
                 patch("evaluate_two_students.align_face", return_value=np.zeros((112, 112, 3), dtype=np.uint8)):
                report = capture_two_identity_evaluation(
                    ("TEST001", "TEST002"), store=store,
                    camera_factory=lambda *_args: camera,
                    detector_factory=TestDetector,
                    recognizer_factory=lambda: CosineOnlyRecognizer(active),
                    stream_factory=TestStream,
                    display_frame=lambda _frame: None,
                    poll_key=poll,
                )
        self.assertEqual(report["confusion_matrix_rows_expected_columns_predicted"]["TEST001"]["TEST001"], 1)
        self.assertEqual(report["confusion_matrix_rows_expected_columns_predicted"]["TEST002"]["TEST002"], 1)
        self.assertEqual(report["samples"][0]["second_roll"], "TEST002")
        self.assertAlmostEqual(report["samples"][0]["score_margin"], 1.)
        self.assertFalse(report["margin_gate_enabled"])
        self.assertEqual(camera.releases, 1)

    def test_report_groups_per_person_conditions_and_open_set_separately(self):
        captures = [
            IdentificationCapture("TEST001", "baseline", "TEST001", .91,
                                 "TEST002", .4, .51, .65, True, "TEST001", True, True),
            IdentificationCapture("TEST002", "head_rotation", "TEST001", .72,
                                 "TEST002", .7, .02, .65, True, "TEST001", False, False),
            IdentificationCapture(None, "lighting_change", "TEST002", .69,
                                 "TEST001", .5, .19, .65, False, "Unknown", None, None),
        ]
        report = build_identification_report(captures, ("TEST001", "TEST002"), .65)
        matrix = report["confusion_matrix_rows_expected_columns_predicted"]
        self.assertEqual(matrix["TEST001"]["TEST001"], 1)
        self.assertEqual(matrix["TEST002"]["TEST001"], 1)
        self.assertEqual(report["per_person"]["TEST001"]["correct"], 1)
        self.assertEqual(report["per_person"]["TEST002"]["incorrect"], 1)
        self.assertEqual(report["open_set"]["captures"], 1)
        self.assertEqual(report["open_set"]["correctly_rejected_as_unknown"], 1)
        self.assertFalse(report["margin_gate_enabled"])


if __name__ == "__main__":
    unittest.main()
