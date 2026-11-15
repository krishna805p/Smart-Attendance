from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from detector import FaceDetection
from embedding_store import EmbeddingStore, EmbeddingStoreError
from quality import assess_face_quality
from recognition import RecognitionEngine


def unit_vector(index: int) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    vector[index] = 1.0
    return vector


class FakeDetector:
    def __init__(self, faces: list[FaceDetection]):
        self.faces = faces

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        return self.faces


class FakeRecognizer:
    @staticmethod
    def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))

    @staticmethod
    def generate_embedding(aligned: np.ndarray) -> np.ndarray:
        return unit_vector(0)


class Phase3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = EmbeddingStore(Path(self.temp_dir.name) / "students.json")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_store_save_load_normalizes_and_preserves_multiple_samples(self) -> None:
        self.store.save_student("ROLL_01", [unit_vector(0) * 5, unit_vector(1) * 3])
        loaded = self.store.load_all()
        self.assertEqual(len(loaded["ROLL_01"]), 2)
        self.assertTrue(all(np.isclose(np.linalg.norm(item), 1.0) for item in loaded["ROLL_01"]))

    def test_duplicate_requires_explicit_replace(self) -> None:
        self.store.save_student("A1", [unit_vector(0)])
        with self.assertRaises(EmbeddingStoreError):
            self.store.save_student("A1", [unit_vector(1)])
        self.store.save_student("A1", [unit_vector(1)], replace=True)
        self.assertAlmostEqual(float(self.store.load_all()["A1"][0][1]), 1.0)

    def test_failed_atomic_replace_preserves_existing_store(self) -> None:
        self.store.save_student("A1", [unit_vector(0)])
        with patch("embedding_store.os.replace", side_effect=OSError("simulated interrupted write")):
            with self.assertRaises(OSError):
                self.store.save_student("A1", [unit_vector(1)], replace=True)
        loaded = self.store.load_all()
        self.assertAlmostEqual(float(loaded["A1"][0][0]), 1.0)
        self.assertEqual(list(self.store.path.parent.glob(".*.tmp")), [])

    def test_invalid_and_corrupt_store_are_rejected(self) -> None:
        self.store.path.parent.mkdir(parents=True, exist_ok=True)
        self.store.path.write_text("{truncated", encoding="utf-8")
        with self.assertRaises(EmbeddingStoreError):
            self.store.load_all()
        self.store.path.write_text(json.dumps({"schema_version": 1, "students": {"A1": [[0] * 127]}}),
                                   encoding="utf-8")
        with self.assertRaises(EmbeddingStoreError):
            self.store.load_all()

    def test_synthetic_same_vector_matches_and_unregistered_vector_is_unknown(self) -> None:
        self.store.save_student("STU7", [unit_vector(0)])
        engine = RecognitionEngine(self.store, detector=FakeDetector([]), recognizer=FakeRecognizer(),
                                   threshold=0.65)
        same = engine.compare_embedding(unit_vector(0))
        other = engine.compare_embedding(unit_vector(1))
        self.assertTrue(same.accepted)
        self.assertEqual(same.roll_number, "STU7")
        self.assertAlmostEqual(same.similarity, 1.0)
        repeated = [engine.compare_embedding(unit_vector(0)).roll_number for _ in range(5)]
        self.assertEqual(repeated, ["STU7"] * 5)
        self.assertFalse(other.accepted)
        self.assertEqual(other.roll_number, "Unknown")

    def test_duplicate_top_candidate_across_faces_is_not_assigned(self) -> None:
        self.store.save_student("STU7", [unit_vector(0)])
        points = ((40.0, 50.0), (80.0, 50.0), (60.0, 70.0), (45.0, 90.0), (75.0, 90.0))
        faces = [FaceDetection((10.0, 10.0, 100.0, 120.0), points, 0.99),
                 FaceDetection((90.0, 20.0, 100.0, 120.0), points, 0.98)]
        engine = RecognitionEngine(self.store, detector=FakeDetector(faces),
                                   recognizer=FakeRecognizer(), threshold=0.65)
        # Random texture makes the synthetic face crops pass only the mechanical
        # sharpness/lighting gates; no biometric meaning is implied.
        rng = np.random.default_rng(7)
        frame = rng.integers(30, 226, size=(160, 200, 3), dtype=np.uint8)
        detected, results = engine.recognize_frame(frame)
        self.assertEqual(len(detected), 2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(not result.accepted and result.roll_number == "Unknown" for result in results))
        self.assertTrue(all("ambiguous" in result.reason for result in results))

    def test_empty_detection_returns_no_face_results(self) -> None:
        engine = RecognitionEngine(self.store, detector=FakeDetector([]), recognizer=FakeRecognizer())
        faces, results = engine.recognize_frame(np.zeros((120, 120, 3), dtype=np.uint8))
        self.assertEqual(faces, [])
        self.assertEqual(results, [])

    def test_low_quality_face_is_rejected(self) -> None:
        points = ((10.0, 10.0), (20.0, 10.0), (15.0, 15.0), (12.0, 20.0), (18.0, 20.0))
        face = FaceDetection((0.0, 0.0, 24.0, 24.0), points, 0.9)
        result = assess_face_quality(np.zeros((40, 40, 3), dtype=np.uint8), face)
        self.assertFalse(result.accepted)
        self.assertIn("small", result.reason)

    def test_enrollment_sample_count_is_bounded_without_opening_camera(self) -> None:
        # Invalid count must fail before webcam access is attempted.
        from enrollment import enroll_from_camera
        with self.assertRaises(ValueError):
            enroll_from_camera("A1", sample_count=14, store=self.store)


if __name__ == "__main__":
    unittest.main()
