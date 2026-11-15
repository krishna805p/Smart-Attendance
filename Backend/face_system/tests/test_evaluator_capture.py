from __future__ import annotations

import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from camera_stream import CameraReadError, CameraStream
from detector import FaceDetection
from embedding_store import EmbeddingStore
from evaluate_recognition import capture_evaluation, validate_participant_label
from quality import QualityResult


def vector(index: int) -> np.ndarray:
    value = np.zeros(128, dtype=np.float32)
    value[index] = 1.0
    return value


class FakeCamera:
    def __init__(self, opened: bool = True):
        self.opened = opened
        self.releases = 0
        self.read_release = threading.Event()

    def isOpened(self):
        return self.opened

    def set(self, *_args):
        return True

    def read(self):
        return True, np.zeros((160, 160, 3), dtype=np.uint8)

    def release(self):
        self.releases += 1
        self.read_release.set()


class FakeStream:
    def __init__(self, camera, **_kwargs):
        self.camera = camera
        self.closed = 0

    def read(self):
        return np.zeros((160, 160, 3), dtype=np.uint8)

    def close(self):
        self.closed += 1
        self.camera.release()
        return True


class FailingStream(FakeStream):
    def read(self):
        raise CameraReadError("test camera read failure")


class FakeDetector:
    def detect(self, _frame):
        points = ((50.0, 55.0), (90.0, 55.0), (70.0, 75.0),
                  (55.0, 95.0), (85.0, 95.0))
        return [FaceDetection((30.0, 25.0, 100.0, 120.0), points, 0.99)]


class SequencedRecognizer:
    calls = 0

    def generate_embedding(self, _aligned):
        type(self).calls += 1
        # First capture is genuine, second impostor, subsequent captures cycle.
        return vector(0 if type(self).calls % 2 else 1)

    @staticmethod
    def cosine_similarity(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class EvaluatorCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = EmbeddingStore(Path(self.temp_dir.name) / "store.json")
        self.store.save_student("T1", [vector(0)])
        self.camera = FakeCamera()
        self.camera_factory_calls = []
        self.stream = None
        SequencedRecognizer.calls = 0

    def tearDown(self):
        self.temp_dir.cleanup()

    def key_sequence(self, keys):
        keys = list(keys)
        def poll():
            if not keys:
                return ord("q")
            return keys.pop(0)
        return poll

    @staticmethod
    def label_keys(label: str):
        return [ord(char) for char in label] + [13]

    def run_capture(self, keys, stream_type=FakeStream):
        def make_camera(*_args):
            self.camera_factory_calls.append(1)
            return self.camera

        def make_stream(camera, **kwargs):
            self.stream = stream_type(camera, **kwargs)
            return self.stream

        with patch("evaluate_recognition.assess_face_quality",
                   return_value=QualityResult(True, "quality accepted", 100.0, 120.0)):
            return capture_evaluation(
                "T1", 0, (0.5, 0.65), self.store,
                camera_factory=make_camera,
                detector_factory=FakeDetector,
                recognizer_factory=SequencedRecognizer,
                stream_factory=make_stream,
                display_frame=lambda _frame: None,
                poll_key=self.key_sequence(keys),
            )

    def test_completes_one_participant_then_proceeds_to_second_without_reopening_camera(self):
        keys = (self.label_keys("P1") + [ord("g"), 32, ord("n")] +
                self.label_keys("P2") + [ord("i"), 32, ord("n"), ord("q")])
        report = self.run_capture(keys)
        self.assertEqual(self.camera_factory_calls, [1])
        self.assertEqual(self.stream.closed, 1)
        self.assertEqual(self.camera.releases, 1)
        self.assertEqual([(r["person_id"], r["role"]) for r in report["samples"]],
                         [("P1", "genuine"), ("P2", "impostor")])

    def test_cancels_current_participant_then_proceeds_to_another(self):
        keys = (self.label_keys("DROP") + [ord("g"), 32, ord("c")] +
                self.label_keys("KEEP") + [ord("g"), 32, ord("n")] +
                self.label_keys("VISITOR") + [ord("i"), 32, ord("n"), ord("q")])
        report = self.run_capture(keys)
        self.assertEqual([row["person_id"] for row in report["samples"]], ["KEEP", "VISITOR"])
        self.assertEqual(self.camera.releases, 1)

    def test_same_participant_is_grouped_across_multiple_capture_conditions(self):
        keys = (self.label_keys("P1") + [ord("g"), 32, ord("n")] +
                self.label_keys("P1") + [ord("g"), 32, ord("n")] +
                self.label_keys("P2") + [ord("i"), 32, ord("n"), ord("q")])
        report = self.run_capture(keys)
        genuine_group = next(s for s in report["person_score_summaries"]
                             if s["person_id"] == "P1")
        self.assertEqual(genuine_group["captures"], 2)
        threshold_row = report["thresholds"][0]
        self.assertEqual(threshold_row["genuine_people"], 1)
        self.assertEqual(threshold_row["impostor_people"], 1)

    def test_q_exits_and_discards_only_the_active_condition(self):
        keys = (self.label_keys("P1") + [ord("g"), 32, ord("n")] +
                self.label_keys("P2") + [ord("i"), 32, ord("n")] +
                self.label_keys("P3") + [ord("g"), 32, ord("q")])
        report = self.run_capture(keys)
        self.assertEqual([row["person_id"] for row in report["samples"]], ["P1", "P2"])
        self.assertEqual(self.camera.releases, 1)

    def test_camera_read_failure_is_reported_and_resources_are_released(self):
        keys = self.label_keys("P1") + [ord("g")]
        with self.assertRaisesRegex(RuntimeError, "Camera frame unavailable: test camera read failure"):
            self.run_capture(keys, stream_type=FailingStream)
        self.assertEqual(self.stream.closed, 1)
        self.assertEqual(self.camera.releases, 1)

    def test_model_initialization_exception_releases_open_camera(self):
        def make_camera(*_args):
            self.camera_factory_calls.append(1)
            return self.camera
        with self.assertRaisesRegex(RuntimeError, "model init failure"):
            capture_evaluation(
                "T1", 0, (0.5, 0.65), self.store,
                camera_factory=make_camera,
                detector_factory=lambda: (_ for _ in ()).throw(RuntimeError("model init failure")),
                stream_factory=FakeStream,
                display_frame=lambda _frame: None,
                poll_key=lambda: ord("q"),
            )
        self.assertEqual(self.camera.releases, 1)

    def test_blank_and_none_labels_do_not_create_participants_and_finish_is_explicit(self):
        for label in ("", "   ", "none", " NONE "):
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_participant_label(label)
        keys = [13] + self.label_keys("none") + [8, 8, 8, 8, ord("f")]
        output = StringIO()
        with redirect_stdout(output), self.assertRaisesRegex(ValueError, "at least one valid genuine"):
            self.run_capture(keys)
        text = output.getvalue()
        self.assertNotIn("Participant label entered:", text)
        self.assertIn("Evaluation finished by operator", text)
        self.assertEqual(self.camera.releases, 1)


class CameraStreamTests(unittest.TestCase):
    def test_repeated_failed_reads_raise_clear_error(self):
        class FailedCamera(FakeCamera):
            def read(self):
                return False, None
        camera = FailedCamera()
        reader = CameraStream(camera, timeout_seconds=0.5,
                              max_consecutive_failures=3, failure_pause=0.01)
        try:
            with self.assertRaisesRegex(CameraReadError, "3 consecutive"):
                reader.read()
        finally:
            reader.close()
        self.assertEqual(camera.releases, 1)

    def test_blocked_camera_read_times_out_and_close_unblocks_it(self):
        class BlockingCamera(FakeCamera):
            def read(self):
                self.read_release.wait(2.0)
                return False, None
        camera = BlockingCamera()
        reader = CameraStream(camera, timeout_seconds=0.05, max_consecutive_failures=1)
        with self.assertRaisesRegex(CameraReadError, "timed out after 0.1s"):
            reader.read()
        stopped = reader.close()
        self.assertTrue(stopped)
        self.assertEqual(camera.releases, 1)


if __name__ == "__main__":
    unittest.main()
