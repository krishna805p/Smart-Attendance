import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import cv2
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main


def run_coroutine_now(coroutine):
    """Drive async test doubles that complete without external I/O or suspension."""
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    raise AssertionError("test coroutine unexpectedly suspended")


class FakeUpload:
    def __init__(self, data):
        self.data = data

    async def read(self, _limit):
        return self.data


class FakeForm:
    def __init__(self, files):
        self.files = files
        self.closed = False

    def getlist(self, name):
        return self.files if name == "image" else []

    def __len__(self):
        return 1 if self.files else 0

    async def close(self):
        self.closed = True


class FakeRequest:
    def __init__(self, form):
        self.upload_form = form

    async def form(self, **_kwargs):
        return self.upload_form


class FaceVerificationTests(unittest.TestCase):
    def setUp(self):
        self.face = SimpleNamespace(landmarks=((1, 1),) * 5)
        class FakeStore:
            def get_student_id_by_roll(_self, roll):
                self.assertEqual(roll, "AUTHENTICATED-ROLL")
                return 91
        self.fake_store = FakeStore()
        self.detector = SimpleNamespace(detect=lambda _image: [self.face])
        self.recognizer = SimpleNamespace(generate_embedding=lambda _aligned: np.ones(128, dtype=np.float32))
        self.components = (self.fake_store, self.detector, self.recognizer)
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)

    def run_image_verification(self, score=0.9, count=15, faces=None,
                                quality_accepted=True, score_error=None):
        detector = self.detector
        if faces is not None:
            detector = SimpleNamespace(detect=lambda _image: faces)
        class Matcher:
            def __init__(self, **kwargs):
                self.store, self.detector, self.recognizer, self.threshold = (
                    kwargs["store"], kwargs["detector"], kwargs["recognizer"], kwargs["threshold"]
                )
            def score_student_id(self, student_id, embedding):
                self.student_id = student_id
                if student_id != 91:
                    raise AssertionError("verification did not use the authenticated student's DB ID")
                if score_error:
                    raise score_error
                return score, count
        with patch.object(main, "_decode_face_image", return_value=self.image), \
             patch.object(main, "_get_face_service", return_value=(detector, self.recognizer)), \
             patch.dict(os.environ, {"FACE_MATCH_THRESHOLD": "0.65",
                                    "FACE_MATCH_THRESHOLD_CALIBRATED": "false"}), \
             patch("face_system.quality.assess_face_quality",
                   return_value=SimpleNamespace(accepted=quality_accepted)), \
             patch("face_system.recognizer.align_face", return_value=self.image), \
             patch("face_system.embedding_store.EmbeddingStore.load_all",
                   side_effect=AssertionError("verification must not fall back to local JSON")), \
             patch("face_system.recognition.RecognitionEngine", Matcher):
            return main.verify_face_image(b"valid-test-image", "AUTHENTICATED-ROLL", self.fake_store)

    def test_authenticated_roll_is_the_only_candidate_and_match_has_no_identity_or_embedding(self):
        result = self.run_image_verification(score=0.91)
        self.assertTrue(result["verified"])
        self.assertEqual(result["score"], 0.91)
        self.assertFalse(result["threshold_calibrated"])
        self.assertNotIn("roll_number", result)
        self.assertNotIn("embedding", result)

    def test_another_students_face_below_threshold_is_rejected(self):
        result = self.run_image_verification(score=0.61)
        self.assertFalse(result["verified"])
        self.assertEqual(result["reason"], "not_matched")

    def test_missing_enrollment_is_reported_without_candidate_search(self):
        with self.assertRaises(HTTPException) as caught:
            self.run_image_verification(score_error=KeyError("not enrolled"))
        self.assertEqual(caught.exception.status_code, 404)

    def test_no_face_and_multiple_faces_are_rejected(self):
        self.assertEqual(self.run_image_verification(faces=[])["reason"], "no_face")
        self.assertEqual(self.run_image_verification(faces=[self.face, self.face])["reason"],
                         "multiple_faces")

    def test_poor_quality_is_rejected_before_embedding(self):
        result = self.run_image_verification(quality_accepted=False)
        self.assertFalse(result["verified"])
        self.assertEqual(result["reason"], "poor_quality")

    def test_invalid_image_is_rejected(self):
        with self.assertRaises(HTTPException) as caught:
            main.verify_face_image(b"not-an-image", "AUTHENTICATED-ROLL", self.fake_store)
        self.assertEqual(caught.exception.status_code, 400)

    def test_small_valid_png_decodes_and_pixel_bomb_dimensions_are_rejected(self):
        ok, encoded = cv2.imencode(".png", np.zeros((8, 9, 3), dtype=np.uint8))
        self.assertTrue(ok)
        decoded = main._decode_face_image(encoded.tobytes())
        self.assertEqual(decoded.shape, (8, 9, 3))

        png_header = (b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" +
                      (4000).to_bytes(4, "big") + (4000).to_bytes(4, "big"))
        with self.assertRaises(ValueError):
            main._decode_face_image(png_header)

    def test_face_image_over_limit_is_rejected_by_route(self):
        form = FakeForm([FakeUpload(b"x" * (main.MAX_FACE_IMAGE_BYTES + 1))])
        with self.assertRaises(HTTPException) as caught:
            run_coroutine_now(main.verify_student_face(FakeRequest(form),
                {"role": "student", "sub": "AUTHENTICATED-ROLL"}))
        self.assertEqual(caught.exception.status_code, 413)
        self.assertTrue(form.closed)

    def test_student_identity_is_taken_from_authenticated_subject(self):
        form = FakeForm([FakeUpload(b"test")])
        with patch.object(main, "verify_face_image", return_value={"verified": True}) as verify:
            response = main.verify_authenticated_face_image(b"test",
                {"role": "student", "sub": "SIGNED-IN-STUDENT"}, self.fake_store)
        self.assertEqual(response, {"verified": True})
        verify.assert_called_once_with(b"test", "SIGNED-IN-STUDENT", self.fake_store)

    def test_database_unavailable_does_not_fall_back_to_local_json(self):
        from face_system.db_embedding_store import EmbeddingStorageUnavailable

        class FailedStore:
            def get_student_id_by_roll(self, _roll):
                raise EmbeddingStorageUnavailable("database unavailable")

        with patch.object(main, "_decode_face_image", return_value=self.image), \
             patch.object(main, "_get_face_service", return_value=(self.detector, self.recognizer)), \
             patch("face_system.quality.assess_face_quality",
                   return_value=SimpleNamespace(accepted=True)), \
             patch("face_system.recognizer.align_face", return_value=self.image), \
             patch("face_system.embedding_store.EmbeddingStore.load_all",
                   side_effect=AssertionError("local JSON fallback attempted")):
            with self.assertRaises(HTTPException) as caught:
                main.verify_face_image(b"valid-test-image", "AUTHENTICATED-ROLL", FailedStore())
        self.assertEqual(caught.exception.status_code, 503)

    def test_non_student_role_is_rejected(self):
        with self.assertRaises(HTTPException) as caught:
            run_coroutine_now(main.verify_student_face(FakeRequest(FakeForm([])),
                {"role": "teacher", "sub": "TEACHER"}))
        self.assertEqual(caught.exception.status_code, 403)

    def test_missing_and_invalid_authentication_are_rejected(self):
        for authorization in (None, "Bearer invalid"):
            with self.assertRaises(HTTPException) as caught:
                main.get_current_user(authorization)
            self.assertEqual(caught.exception.status_code, 401)

    def test_model_loading_failure_returns_service_unavailable(self):
        with patch.object(main, "_face_service", None), \
             patch("face_system.detector.RetinaFaceDetector", side_effect=RuntimeError("model missing")):
            with self.assertRaises(HTTPException) as caught:
                main._get_face_service()
        self.assertEqual(caught.exception.status_code, 503)

    def test_threshold_configuration_is_provisional_by_default_and_validated(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(main._face_threshold_config(), (0.65, False))
        with patch.dict(os.environ, {"FACE_MATCH_THRESHOLD": "1.5"}, clear=False):
            with self.assertRaises(HTTPException) as caught:
                main._face_threshold_config()
        self.assertEqual(caught.exception.status_code, 503)

    def test_openapi_documents_one_multipart_image_field(self):
        operation = main.app.openapi()["paths"]["/api/student/verify-face"]["post"]
        schema = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertEqual(schema["required"], ["image"])
        self.assertEqual(schema["properties"]["image"]["format"], "binary")


class UploadLimitMiddlewareTests(unittest.TestCase):
    def test_oversized_chunked_request_is_rejected_before_application(self):
        middleware = main.FaceUploadSizeLimitMiddleware(lambda *_args: None,
                                                        max_bytes=10)
        incoming = [{"type": "http.request", "body": b"123456", "more_body": True},
                    {"type": "http.request", "body": b"78901", "more_body": False}]
        sent = []
        called = []
        async def receive():
            return incoming.pop(0)
        async def send(message):
            sent.append(message)
        async def app(_scope, _receive, _send):
            called.append(True)
        middleware.application = app
        scope = {"type": "http", "method": "POST", "path": "/api/student/verify-face", "headers": []}
        run_coroutine_now(middleware(scope, receive, send))
        self.assertFalse(called)
        self.assertEqual(sent[0]["status"], 413)


if __name__ == "__main__":
    unittest.main()
