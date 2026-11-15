"""Camera-guided local student enrollment; stores embeddings only."""

from __future__ import annotations

import time

import cv2
import numpy as np

from detector import RetinaFaceDetector
from embedding_store import EmbeddingStore, EmbeddingStoreError
from quality import assess_face_quality
from recognizer import MobileFaceNetRecognizer, align_face


def enroll_from_camera(roll_number: str, *, camera_index: int = 0,
                       sample_count: int = 15, replace: bool = False,
                       store: EmbeddingStore | None = None) -> bool:
    if not 15 <= sample_count <= 20:
        raise ValueError("sample_count must be between 15 and 20")
    store = store or EmbeddingStore()
    roll = store.validate_roll_number(roll_number)
    existing = store.load_all()
    if roll in existing and not replace:
        raise EmbeddingStoreError(
            f"{roll} is already enrolled; rerun with --replace to replace the record"
        )

    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Camera unavailable (index {camera_index}); enrollment was not started")

    embeddings: list[np.ndarray] = []
    detector: RetinaFaceDetector | None = None
    recognizer: MobileFaceNetRecognizer | None = None
    last_status = "Center your face; press SPACE to capture a valid sample. Q cancels."
    last_sample_time = 0.0
    try:
        # Camera access is confirmed before model startup or any sample capture.
        detector = RetinaFaceDetector()
        recognizer = MobileFaceNetRecognizer()
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError("Camera frame read failed; enrollment was not saved")
            faces = detector.detect(frame)
            quality = None
            if len(faces) == 0:
                last_status = "No face detected"
            elif len(faces) > 1:
                last_status = "Multiple faces detected; only one person may be in frame"
            else:
                quality = assess_face_quality(frame, faces[0])
                last_status = quality.reason

            preview = frame.copy()
            if len(faces) == 1:
                face = faces[0]
                x, y, width, height = (int(round(v)) for v in face.box)
                color = (0, 220, 0) if quality and quality.accepted else (0, 0, 255)
                cv2.rectangle(preview, (x, y), (x + width, y + height), color, 2)
                for px, py in face.landmarks:
                    cv2.circle(preview, (int(round(px)), int(round(py))), 2, (0, 180, 255), -1)
                last_status += f" | blur {quality.blur_score:.1f}" if quality and quality.blur_score is not None else ""

            cv2.putText(preview, f"Roll: {roll}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            cv2.putText(preview, f"Samples: {len(embeddings)}/{sample_count}", (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(preview, last_status[:100], (12, 84), cv2.FONT_HERSHEY_SIMPLEX,
                        0.53, (0, 255, 255), 2)
            cv2.putText(preview, "SPACE capture | Q/Esc cancel", (12, 112),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 2)
            cv2.imshow("Student face enrollment", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("Enrollment cancelled; no embeddings saved.")
                return False
            if key != 32:
                continue
            if len(faces) != 1 or quality is None or not quality.accepted:
                continue
            # Small debounce protects against key-repeat creating duplicates.
            now = time.monotonic()
            if now - last_sample_time < 0.35:
                continue
            try:
                aligned = align_face(frame, faces[0].landmarks)
                embedding = recognizer.generate_embedding(aligned)
            except Exception as exc:
                last_status = f"Sample rejected: {type(exc).__name__}"
                continue
            if embedding.shape != (128,) or not np.isfinite(embedding).all():
                last_status = "Invalid embedding; sample rejected"
                continue
            embeddings.append(embedding)
            last_sample_time = now
            last_status = "Sample accepted"
            print(f"Captured valid sample {len(embeddings)}/{sample_count}")
            if len(embeddings) >= sample_count:
                store.save_student(roll, embeddings, replace=replace)
                print(f"Enrollment saved for {roll}: {len(embeddings)} embeddings (no photos).")
                return True
    finally:
        camera.release()
        cv2.destroyAllWindows()
