from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from embedding_store import EmbeddingStore
from evaluate_recognition import EvaluationScore, evaluate_thresholds, summarize_scores
from recognition import RecognitionEngine


def basis(index: int) -> np.ndarray:
    value = np.zeros(128, dtype=np.float32)
    value[index] = 1.0
    return value


class Similarity:
    @staticmethod
    def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))


class Phase31Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = EmbeddingStore(Path(self.temp_dir.name) / "store.json")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def engine(self, threshold: float = 0.65, margin: float | None = None) -> RecognitionEngine:
        return RecognitionEngine(self.store, detector=object(), recognizer=Similarity(),
                                 threshold=threshold, minimum_margin=margin)

    def test_genuine_match_and_diagnostics(self) -> None:
        self.store.save_student("A", [basis(0), basis(0)])
        result = self.engine().compare_embedding(basis(0))
        self.assertTrue(result.accepted)
        self.assertEqual(result.roll_number, "A")
        self.assertEqual(result.samples_compared, 2)
        self.assertEqual(result.candidate_samples_compared, 2)
        self.assertTrue(result.threshold_passed)
        self.assertIsNone(result.second_best_similarity)
        self.assertIn("result=MATCH", self.engine().format_diagnostics(result))

    def test_unregistered_and_low_similarity_rejected(self) -> None:
        self.store.save_student("A", [basis(0)])
        result = self.engine(threshold=0.65).compare_embedding(basis(1))
        self.assertFalse(result.accepted)
        self.assertEqual(result.roll_number, "Unknown")
        self.assertEqual(result.similarity, 0.0)
        self.assertFalse(result.threshold_passed)

    def test_correct_ranking_of_multiple_enrolled_identities(self) -> None:
        self.store.save_student("A", [basis(0)])
        self.store.save_student("B", [basis(1), basis(1)])
        result = self.engine().compare_embedding(basis(1))
        self.assertEqual(result.roll_number, "B")
        self.assertEqual(result.second_best_roll_number, "A")
        self.assertAlmostEqual(result.score_margin, 1.0)
        self.assertEqual(result.samples_compared, 3)

    def test_ambiguous_candidates_rejected_by_configured_margin(self) -> None:
        self.store.save_student("A", [basis(0)])
        self.store.save_student("B", [basis(1)])
        query = np.zeros(128, dtype=np.float32)
        query[0:2] = 1 / np.sqrt(2)
        result = self.engine(threshold=0.65, margin=0.1).compare_embedding(query)
        self.assertTrue(result.threshold_passed)
        self.assertFalse(result.margin_passed)
        self.assertFalse(result.accepted)
        self.assertEqual(result.roll_number, "Unknown")

    def test_exact_tie_rejected_even_when_margin_control_disabled(self) -> None:
        self.store.save_student("A", [basis(0)])
        self.store.save_student("B", [basis(1)])
        query = np.zeros(128, dtype=np.float32)
        query[0:2] = 1 / np.sqrt(2)
        result = self.engine(threshold=0.5).compare_embedding(query)
        self.assertFalse(result.accepted)
        self.assertEqual(result.roll_number, "Unknown")
        self.assertIn("ambiguous", result.reason)

    def test_threshold_inclusive_boundary(self) -> None:
        self.store.save_student("A", [basis(0)])
        result = self.engine(threshold=1.0).compare_embedding(basis(0))
        self.assertTrue(result.accepted)  # implementation uses >= threshold
        self.assertEqual(result.similarity, 1.0)

    def test_margin_inclusive_boundary(self) -> None:
        self.store.save_student("A", [basis(0)])
        second = np.zeros(128, dtype=np.float32)
        second[:2] = (0.8, 0.6)
        self.store.save_student("B", [second])
        unguarded = self.engine(threshold=0.7).compare_embedding(basis(0))
        exact_margin = unguarded.score_margin
        guarded = self.engine(threshold=0.7, margin=exact_margin).compare_embedding(basis(0))
        self.assertTrue(guarded.margin_passed)
        self.assertTrue(guarded.accepted)

    def test_threshold_and_margin_ranges_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            self.engine(threshold=1.01)
        with self.assertRaises(ValueError):
            self.engine(margin=2.01)

    def test_invalid_embeddings_rejected(self) -> None:
        self.store.save_student("A", [basis(0)])
        engine = self.engine()
        for value in (np.zeros(128), np.full(128, np.nan), np.zeros((1, 128))):
            with self.subTest(shape=np.asarray(value).shape):
                with self.assertRaises(ValueError):
                    engine.compare_embedding(value)

    def test_single_identity_has_no_margin_information(self) -> None:
        self.store.save_student("A", [basis(0)])
        result = self.engine(threshold=0.65, margin=0.5).compare_embedding(basis(0))
        self.assertTrue(result.accepted)
        self.assertIsNone(result.score_margin)
        self.assertTrue(result.margin_passed)

    def test_evaluation_groups_repeated_frames_by_person(self) -> None:
        samples = [
            EvaluationScore("student", "genuine", 0.8),
            EvaluationScore("student", "genuine", 0.9),
            EvaluationScore("visitor", "impostor", 0.4),
            EvaluationScore("visitor", "impostor", 0.7),
            EvaluationScore("visitor", "impostor", 0.8),
        ]
        summary = summarize_scores(samples)
        self.assertEqual(len(summary), 2)  # two people, not five independent identities
        metrics = evaluate_thresholds(samples, [0.65])[0]
        self.assertEqual(metrics["genuine_people"], 1)
        self.assertEqual(metrics["impostor_people"], 1)
        self.assertEqual(metrics["person_median_far"], 1.0)
        self.assertEqual(metrics["person_any_capture_far"], 1.0)

    def test_person_cannot_be_both_genuine_and_impostor(self) -> None:
        with self.assertRaises(ValueError):
            summarize_scores([EvaluationScore("same", "genuine", 0.9),
                              EvaluationScore("same", "impostor", 0.2)])


if __name__ == "__main__":
    unittest.main()
