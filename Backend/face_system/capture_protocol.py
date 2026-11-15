"""Capture consented evaluation faces and write scores only (no images/embeddings)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from camera_stream import CameraReadError, CameraStream
from detector import RetinaFaceDetector
from embedding_store import EmbeddingStore
from db_store_runtime import pooled_embedding_store
from quality import assess_face_quality
from recognizer import MobileFaceNetRecognizer, align_face
from evaluation_protocol import aggregate_template_scores


FACE_SYSTEM_DIR = Path(__file__).resolve().parent
CSV_FIELDS = ["capture_id", "participant_id", "role", "expected_candidate_id", "session_id",
              "condition", "partition", "candidate_id", "top3_mean", "mean", "maximum",
              "query_blur", "query_brightness", "face_confidence", "captured_utc"]


def _dhash(image_bgr: np.ndarray) -> int:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    tiny = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    value = 0
    for row in range(8):
        for col in range(8):
            value = (value << 1) | int(tiny[row, col] > tiny[row, col + 1])
    return value


def _score_against_gallery(query: np.ndarray, templates: dict[str, list[np.ndarray]], recognizer):
    output = {}
    for alias, vectors in templates.items():
        cosine_values = [recognizer.cosine_similarity(query, vector) for vector in vectors]
        output[alias] = {
            "top3_mean": aggregate_template_scores(cosine_values, "top3_mean"),
            "mean": aggregate_template_scores(cosine_values, "mean"),
            "maximum": aggregate_template_scores(cosine_values, "maximum"),
        }
    return output


def capture_session(*, participant_id: str, session_id: str, partition: str,
                    condition: str, expected_roll: str | None, output_path: Path,
                    sample_count: int = 8, camera_index: int = 0, store=None) -> int:
    participant = participant_id.strip()
    session = session_id.strip()
    if not participant.startswith("person_") or not participant[7:] or not participant[7:].replace("_", "").replace("-", "").isalnum():
        raise ValueError("participant_id must be an anonymous label such as person_01")
    if not session or not session.replace("_", "").replace("-", "").isalnum():
        raise ValueError("session_id must be a nonblank anonymous ID using letters, digits, _ or -")
    if partition not in ("calibration", "test"):
        raise ValueError("partition must be calibration or test")
    if sample_count < 1 or sample_count > 30:
        raise ValueError("sample_count must be between 1 and 30")
    output_path = output_path.resolve()
    if not output_path.is_relative_to(FACE_SYSTEM_DIR.resolve()):
        raise ValueError("score output must stay inside Backend/face_system/")

    if store is None:
        # Evaluation is open-set by design, so it reads the database gallery
        # once, then releases the pooled connection before webcam capture.
        with pooled_embedding_store() as live_store:
            enrollments = live_store.get_all_student_embeddings_by_roll()
            validate_roll = live_store.validate_roll_number
    else:
        validate_roll = store.validate_roll_number
        enrollments = (store.get_all_student_embeddings_by_roll()
                       if hasattr(store, "get_all_student_embeddings_by_roll")
                       else store.load_all())
    if not enrollments:
        raise ValueError("no face enrollments are available for evaluation")
    rolls = sorted(enrollments)
    aliases = {roll: f"candidate_{index + 1:04d}" for index, roll in enumerate(rolls)}
    if expected_roll is not None:
        expected_roll = validate_roll(expected_roll)
        if expected_roll not in enrollments:
            raise ValueError("expected student is not enrolled")
        role, expected_alias = "genuine", aliases[expected_roll]
    else:
        role, expected_alias = "impostor", ""
    templates = {aliases[roll]: enrollments[roll] for roll in rolls}

    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Camera unavailable at index {camera_index}")
    stream = None
    try:
        try:
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            timeout_property = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
            if timeout_property is not None:
                camera.set(timeout_property, 2500)
        except (AttributeError, TypeError, cv2.error):
            pass
        detector = RetinaFaceDetector()
        recognizer = MobileFaceNetRecognizer()
        stream = CameraStream(camera, timeout_seconds=2.5)
        accepted = 0
        recent_hashes: list[int] = []
        last_capture = 0.0
        print(f"Evaluation capture ready: participant={participant}, role={role}, "
              f"partition={partition}, condition={condition}. No images or embeddings are saved.")
        while accepted < sample_count:
            try:
                frame = stream.read()
            except CameraReadError as exc:
                raise RuntimeError(f"Camera frame unavailable: {exc}") from exc
            faces = detector.detect(frame)
            quality = assess_face_quality(frame, faces[0]) if len(faces) == 1 else None
            preview = frame.copy()
            status = "No face" if len(faces) == 0 else (
                "Multiple faces" if len(faces) > 1 else quality.reason
            )
            if len(faces) == 1:
                face = faces[0]
                x, y, width, height = (int(round(value)) for value in face.box)
                cv2.rectangle(preview, (x, y), (x + width, y + height),
                              (0, 220, 0) if quality and quality.accepted else (0, 0, 255), 2)
                for px, py in face.landmarks:
                    cv2.circle(preview, (int(round(px)), int(round(py))), 2, (0, 180, 255), -1)
                status += f" | detector {face.confidence:.2f}"
            cv2.putText(preview, f"Samples {accepted}/{sample_count} | {status[:80]}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(preview, "SPACE capture | Q/Esc cancel", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow("Face recognition evaluation capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print(f"Evaluation capture cancelled; accepted samples discarded: {accepted}.")
                return 0
            if key != 32 or len(faces) != 1 or quality is None or not quality.accepted:
                continue
            now = time.monotonic()
            if now - last_capture < 0.8:
                continue
            aligned = align_face(frame, np.asarray(faces[0].landmarks, dtype=np.float32))
            fingerprint = _dhash(aligned)
            if any((fingerprint ^ previous).bit_count() <= 2 for previous in recent_hashes):
                print("Near-duplicate capture ignored; vary pose or wait for a new frame.")
                continue
            embedding = recognizer.generate_embedding(aligned)
            gallery_scores = _score_against_gallery(embedding, templates, recognizer)
            del embedding
            capture_id = uuid.uuid4().hex
            captured = datetime.now(timezone.utc).isoformat()
            rows = [{
                "capture_id": capture_id, "participant_id": participant, "role": role,
                "expected_candidate_id": expected_alias, "session_id": session,
                "condition": condition, "partition": partition,
                "candidate_id": alias, **scores,
                "query_blur": f"{quality.blur_score:.3f}",
                "query_brightness": f"{quality.brightness:.3f}",
                "face_confidence": f"{faces[0].confidence:.5f}", "captured_utc": captured,
            } for alias, scores in gallery_scores.items()]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            exists = output_path.exists()
            if exists:
                with output_path.open("r", newline="", encoding="utf-8-sig") as existing_file:
                    if next(csv.reader(existing_file), []) != CSV_FIELDS:
                        raise ValueError("existing score file has an incompatible header")
            with output_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                if not exists:
                    writer.writeheader()
                writer.writerows(rows)
                handle.flush()
            recent_hashes.append(fingerprint)
            accepted += 1
            last_capture = now
            print(f"Saved score-only capture {accepted}/{sample_count}; no identity scores printed.")
        return accepted
    finally:
        try:
            if stream is not None:
                stream.close()
            else:
                camera.release()
        finally:
            cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", required=True, help="anonymous stable label, e.g. person_01")
    parser.add_argument("--session", required=True, help="new session ID; keep calibration/test sessions distinct")
    parser.add_argument("--partition", choices=("calibration", "test"), required=True)
    parser.add_argument("--condition", choices=("baseline", "left_angle", "right_angle",
                                                  "lighting_change", "different_distance"), required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--expected-roll", help="for a consenting enrolled person; used locally only")
    group.add_argument("--open-set", action="store_true", help="for a consenting person not enrolled in the gallery")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--consent-confirmed", action="store_true", required=True)
    parser.add_argument("--output", type=Path, default=FACE_SYSTEM_DIR / "data" / "evaluation_protocol.csv")
    args = parser.parse_args()
    try:
        count = capture_session(
            participant_id=args.participant, session_id=args.session,
            partition=args.partition, condition=args.condition,
            expected_roll=args.expected_roll, output_path=args.output,
            sample_count=args.samples, camera_index=args.camera,
        )
        return 0 if count == args.samples else 1
    except Exception as exc:
        print(f"Evaluation capture stopped: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
