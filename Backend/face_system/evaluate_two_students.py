"""Consent-based two-identity webcam identification experiment.

This reports candidate ranking and the existing provisional-threshold decision.
It does not change enrollment data, the threshold, or attendance state.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from camera_stream import CameraReadError, CameraStream
from detector import RetinaFaceDetector
from embedding_store import EmbeddingStore
from db_store_runtime import pooled_embedding_store
from quality import assess_face_quality
from recognition import PROVISIONAL_THRESHOLD, RecognitionEngine
from recognizer import MobileFaceNetRecognizer, align_face


FACE_SYSTEM_DIR = Path(__file__).resolve().parent


class _SelectedIdentityStore:
    """Read-only view limiting comparison to the two identities under study."""

    def __init__(self, source: EmbeddingStore, rolls: tuple[str, str]) -> None:
        self.source = source
        self.rolls = rolls

    def _selected(self):
        if hasattr(self.source, "get_all_student_embeddings_by_roll"):
            enrollments = self.source.get_all_student_embeddings_by_roll()
        else:
            enrollments = self.source.load_all()
        return {roll: enrollments[roll] for roll in self.rolls}

    def load_all(self):
        return self._selected()

    def get_all_student_embeddings(self):
        return self._selected()

    def validate_roll_number(self, roll_number: str) -> str:
        return self.source.validate_roll_number(roll_number)


@dataclass(frozen=True)
class IdentificationCapture:
    expected_roll: str | None
    condition: str
    best_roll: str | None
    best_score: float | None
    second_roll: str | None
    second_score: float | None
    score_margin: float | None
    threshold: float
    threshold_accepted: bool
    predicted_roll: str
    top_candidate_correct: bool | None
    identification_correct: bool | None


def build_identification_report(
    captures: list[IdentificationCapture], rolls: tuple[str, str], threshold: float
) -> dict[str, object]:
    """Build a closed-set confusion matrix and separate open-set summary."""
    aliases = {roll: f"candidate_{index + 1:04d}" for index, roll in enumerate(rolls)}
    candidates = tuple(aliases[roll] for roll in rolls)
    matrix = {alias: {column: 0 for column in (*candidates, "Unknown")} for alias in candidates}
    per_person: dict[str, dict[str, int]] = {
        alias: {"captures": 0, "correct": 0, "incorrect": 0} for alias in candidates
    }
    open_set: list[IdentificationCapture] = []
    def safe_capture(item: IdentificationCapture) -> dict[str, object]:
        result = asdict(item)
        result["expected_roll"] = aliases.get(item.expected_roll) if item.expected_roll else None
        result["best_roll"] = aliases.get(item.best_roll) if item.best_roll else None
        result["second_roll"] = aliases.get(item.second_roll) if item.second_roll else None
        result["predicted_roll"] = aliases.get(item.predicted_roll, "Unknown")
        return result

    for sample in captures:
        if sample.expected_roll is None:
            open_set.append(sample)
            continue
        if sample.expected_roll not in matrix:
            raise ValueError(f"unexpected expected identity: {sample.expected_roll}")
        expected = aliases[sample.expected_roll]
        predicted = aliases.get(sample.predicted_roll, "Unknown")
        matrix[expected][predicted] += 1
        person = per_person[expected]
        person["captures"] += 1
        if sample.identification_correct:
            person["correct"] += 1
        else:
            person["incorrect"] += 1
    return {
        "threshold": threshold,
        "threshold_is_provisional": True,
        "margin_gate_enabled": False,
        "identities": list(rolls),
        "confusion_matrix_rows_expected_columns_predicted": matrix,
        "per_person": per_person,
        "conditions": {
            condition: [safe_capture(item) for item in captures if item.condition == condition]
            for condition in ("baseline", "head_rotation", "lighting_change")
        },
        "open_set": {
            "captures": len(open_set),
            "correctly_rejected_as_unknown": sum(not sample.threshold_accepted for sample in open_set),
            "false_accepts": sum(sample.threshold_accepted for sample in open_set),
            "samples": [safe_capture(sample) for sample in open_set],
        },
        "samples": [safe_capture(item) for item in captures],
    }


def capture_two_identity_evaluation(
    rolls: tuple[str, str], camera_index: int = 0, *, threshold: float = PROVISIONAL_THRESHOLD,
    include_unregistered: bool = False, store: EmbeddingStore | None = None,
    camera_factory=None, detector_factory=None, recognizer_factory=None,
    stream_factory=None, display_frame=None, poll_key=None,
) -> dict[str, object]:
    store = store or EmbeddingStore()
    rolls = tuple(store.validate_roll_number(value) for value in rolls)
    if len(set(rolls)) != 2:
        raise ValueError("two distinct enrolled roll numbers are required")
    enrolled = (store.get_all_student_embeddings_by_roll()
                if hasattr(store, "get_all_student_embeddings_by_roll") else store.load_all())
    missing = [roll for roll in rolls if roll not in enrolled]
    if missing:
        raise ValueError(f"enrollment missing for: {', '.join(missing)}")
    if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between -1 and 1")

    make_camera = camera_factory or cv2.VideoCapture
    camera = make_camera(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = make_camera(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Camera unavailable (index {camera_index}); evaluation did not start")
    try:
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        timeout_prop = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
        if timeout_prop is not None:
            camera.set(timeout_prop, 2500)
    except (AttributeError, TypeError, cv2.error):
        pass

    stream = None
    captures: list[IdentificationCapture] = []
    try:
        detector = (detector_factory or RetinaFaceDetector)()
        recognizer = (recognizer_factory or MobileFaceNetRecognizer)()
        engine = RecognitionEngine(store=_SelectedIdentityStore(store, rolls), detector=detector,
                                   recognizer=recognizer, threshold=threshold,
                                   minimum_margin=None)
        stream = (stream_factory or CameraStream)(camera, timeout_seconds=2.5)
        show = display_frame or (lambda frame: cv2.imshow("Two-student identification", frame))
        get_key = poll_key or (lambda: cv2.waitKey(1) & 0xFF)
        selected: str | None = None
        condition = "baseline"
        status = "Select enrolled person: 1/2; F finish; Q/Esc exit."
        finished = False
        valid_in_group = 0
        print(f"Two-identity evaluation ready for {rolls[0]} and {rolls[1]}; threshold={threshold:.3f} (provisional), margin gate disabled.")
        print("Have each consenting person take baseline, modest head-rotation, and lighting-change captures.")
        while not finished:
            try:
                frame = stream.read()
            except CameraReadError as exc:
                raise RuntimeError(f"Camera frame unavailable: {exc}") from exc
            faces = detector.detect(frame) if selected else []
            quality = None
            if selected:
                if len(faces) == 1:
                    quality = assess_face_quality(frame, faces[0])
                    status = quality.reason
                elif len(faces) > 1:
                    status = "Multiple faces; only the selected person may be in frame."
                else:
                    status = "No face detected."
            preview = frame.copy()
            if selected and len(faces) == 1:
                face = faces[0]
                x, y, w, h = (int(round(v)) for v in face.box)
                cv2.rectangle(preview, (x, y), (x + w, y + h),
                              (0, 220, 0) if quality and quality.accepted else (0, 0, 255), 2)
                for px, py in face.landmarks:
                    cv2.circle(preview, (int(round(px)), int(round(py))), 2, (0, 180, 255), -1)
            if selected is None:
                controls = f"1={rolls[0]} | 2={rolls[1]}" + (" | 3=unregistered" if include_unregistered else "") + " | F finish | Q exit"
                title = "Select the consenting person currently in frame"
            else:
                controls = "SPACE capture | B baseline | P head rotation | L lighting | N next | F finish | Q exit"
                title = f"Expected: {selected} | condition: {condition} | captured: {valid_in_group}"
            cv2.putText(preview, title[:100], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            cv2.putText(preview, status[:100], (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(preview, controls[:120], (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2)
            show(preview)
            key = get_key()
            if key < 0:
                continue
            char = chr(key & 0xFF) if 0 <= (key & 0xFF) < 128 else ""
            if char.lower() in ("q",) or key == 27:
                finished = True
                continue
            if selected is None:
                if char.lower() == "f":
                    finished = True
                elif char == "1":
                    selected, condition, valid_in_group = rolls[0], "baseline", 0
                    print(f"Capturing genuine samples for {selected}.")
                elif char == "2":
                    selected, condition, valid_in_group = rolls[1], "baseline", 0
                    print(f"Capturing genuine samples for {selected}.")
                elif char == "3" and include_unregistered:
                    selected, condition, valid_in_group = "__OPEN_SET__", "baseline", 0
                    print("Capturing open-set samples; use an unregistered consenting person only.")
                continue
            if char.lower() == "n":
                selected = None
                valid_in_group = 0
                status = "Select next person/condition or press F to finish."
            elif char.lower() == "f":
                finished = True
            elif char.lower() == "b":
                condition = "baseline"
                status = "Condition set to baseline."
            elif char.lower() == "p":
                condition = "head_rotation"
                status = "Condition set to modest head rotation."
            elif char.lower() == "l":
                condition = "lighting_change"
                status = "Condition set to lighting change."
            elif key == 32:
                if len(faces) != 1 or quality is None or not quality.accepted:
                    print(f"Rejected sample for {selected}: {status}")
                    continue
                try:
                    aligned = align_face(frame, faces[0].landmarks)
                    embedding = recognizer.generate_embedding(aligned)
                    match = engine.compare_embedding(embedding)
                except Exception as exc:
                    print(f"Rejected sample for {selected}: {type(exc).__name__}: {exc}")
                    continue
                expected = None if selected == "__OPEN_SET__" else selected
                best_roll = match.candidate_roll_number
                threshold_accepted = match.accepted
                predicted = match.roll_number if threshold_accepted else "Unknown"
                top_correct = None if expected is None else best_roll == expected
                identified_correct = None if expected is None else predicted == expected
                result = IdentificationCapture(
                    expected, condition, best_roll, match.similarity,
                    match.second_best_roll_number, match.second_best_similarity,
                    match.score_margin, threshold, threshold_accepted, predicted,
                    top_correct, identified_correct,
                )
                captures.append(result)
                valid_in_group += 1
                score_text = "n/a" if match.similarity is None else f"{match.similarity:.4f}"
                second_text = "none" if match.second_best_roll_number is None else f"{match.second_best_roll_number}:{match.second_best_similarity:.4f}"
                margin_text = "n/a" if match.score_margin is None else f"{match.score_margin:.4f}"
                verdict = "open-set false accept" if expected is None and threshold_accepted else (
                    "open-set rejected" if expected is None else
                    "correct identification" if identified_correct else "incorrect identification")
                print(f"capture expected={expected or 'unregistered'} condition={condition} "
                      f"best={best_roll or 'none'}:{score_text} second={second_text} "
                      f"margin={margin_text} predicted={predicted} verdict={verdict}")
                del embedding
        report = build_identification_report(captures, rolls, threshold)
    finally:
        try:
            if stream is not None:
                stopped = stream.close()
                if not stopped:
                    print("Camera shutdown timed out; release was requested.")
            else:
                camera.release()
        finally:
            cv2.destroyAllWindows()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rolls", required=True, help="exactly two enrolled IDs separated by comma")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=PROVISIONAL_THRESHOLD,
                        help="provisional similarity threshold; no margin gate is enabled")
    parser.add_argument("--include-unregistered", action="store_true",
                        help="enable separate open-set capture with key 3")
    parser.add_argument("--consent-confirmed", action="store_true", required=True,
                        help="confirm all captured people have consented")
    parser.add_argument("--output", help="optional report JSON path inside face_system/")
    args = parser.parse_args()
    rolls = tuple(part.strip() for part in args.rolls.split(","))
    if len(rolls) != 2 or any(not part for part in rolls):
        parser.error("--rolls must contain two nonblank comma-separated IDs")
    out_path = Path(args.output).resolve() if args.output else None
    if out_path and not out_path.is_relative_to(FACE_SYSTEM_DIR.resolve()):
        parser.error("--output must be inside Backend/face_system/")
    try:
        with pooled_embedding_store() as store:
            report = capture_two_identity_evaluation(
                (rolls[0], rolls[1]), args.camera, threshold=args.threshold,
                include_unregistered=args.include_unregistered, store=store,
            )
    except Exception as exc:
        print(f"Two-identity evaluation stopped: {exc}")
        return 2
    print(json.dumps(report, indent=2))
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
