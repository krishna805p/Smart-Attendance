"""Consent-based webcam evaluation for one enrolled roll; images/embeddings aren't saved."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from detector import RetinaFaceDetector
from embedding_store import EmbeddingStore, EmbeddingStoreError
from db_store_runtime import pooled_embedding_store
from quality import assess_face_quality
from recognition import RecognitionEngine
from recognizer import MobileFaceNetRecognizer, align_face
from camera_stream import CameraReadError, CameraStream


FACE_SYSTEM_DIR = Path(__file__).resolve().parent
DEFAULT_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


@dataclass(frozen=True)
class EvaluationScore:
    person_id: str
    role: str  # "genuine" or "impostor"
    score: float


def summarize_scores(samples: Iterable[EvaluationScore]) -> dict[str, dict[str, float | int]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    role_by_person: dict[str, str] = {}
    for sample in samples:
        if sample.role not in ("genuine", "impostor"):
            raise ValueError(f"invalid evaluation role: {sample.role}")
        if not sample.person_id.strip():
            raise ValueError("person_id must not be blank")
        if not math.isfinite(sample.score) or not -1.0 <= sample.score <= 1.0:
            raise ValueError("score must be finite and between -1 and 1")
        prior_role = role_by_person.setdefault(sample.person_id, sample.role)
        if prior_role != sample.role:
            raise ValueError(f"person {sample.person_id} cannot be both genuine and impostor")
        grouped[(sample.role, sample.person_id)].append(sample.score)
    output: dict[str, dict[str, float | int]] = {}
    for (role, person_id), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        output[f"{role}:{person_id}"] = {
            "role": role,
            "person_id": person_id,
            "captures": int(len(values)),
            "min": float(np.min(arr)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
        }
    return output


def evaluate_thresholds(samples: Iterable[EvaluationScore],
                        thresholds: Iterable[float]) -> list[dict[str, float | int]]:
    sample_list = list(samples)
    summaries = list(summarize_scores(sample_list).values())
    genuine_people = [s for s in summaries if s["role"] == "genuine"]
    impostor_people = [s for s in summaries if s["role"] == "impostor"]
    if not genuine_people or not impostor_people:
        raise ValueError("evaluation needs at least one genuine and one distinct impostor person")
    results = []
    for threshold in thresholds:
        threshold = float(threshold)
        if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError("candidate thresholds must be finite and between -1 and 1")
        accepted = [item.score >= threshold for item in sample_list]
        genuine_samples = [item for item in sample_list if item.role == "genuine"]
        impostor_samples = [item for item in sample_list if item.role == "impostor"]
        median_far = sum(p["median"] >= threshold for p in impostor_people) / len(impostor_people)
        median_frr = sum(p["median"] < threshold for p in genuine_people) / len(genuine_people)
        # Cluster-aware worst-case rates count each distinct human once: a
        # participant contributes at most one false-accept/reject event even
        # when they supplied many frames or lighting/pose conditions.
        any_frame_far = sum(p["max"] >= threshold for p in impostor_people) / len(impostor_people)
        any_frame_frr = sum(p["min"] < threshold for p in genuine_people) / len(genuine_people)
        genuine_count = len(genuine_samples)
        impostor_count = len(impostor_samples)
        results.append({
            "threshold": threshold,
            "genuine_people": len(genuine_people),
            "impostor_people": len(impostor_people),
            "genuine_captures": genuine_count,
            "impostor_captures": impostor_count,
            "capture_far": (sum(not value for value, item in zip(accepted, sample_list)
                                 if item.role == "impostor") / impostor_count),
            "capture_frr": (sum(not value for value, item in zip(accepted, sample_list)
                                 if item.role == "genuine") / genuine_count),
            "person_median_far": float(median_far),
            "person_median_frr": float(median_frr),
            "person_any_capture_far": float(any_frame_far),
            "person_any_capture_frr": float(any_frame_frr),
        })
    return results


def score_distribution(samples: Iterable[EvaluationScore]) -> dict[str, dict[str, float | int]]:
    sample_list = list(samples)
    result: dict[str, dict[str, float | int]] = {}
    for role in ("genuine", "impostor"):
        values = np.asarray([s.score for s in sample_list if s.role == role], dtype=np.float64)
        if values.size:
            result[role] = {
                "captures": int(values.size),
                "people": len({s.person_id for s in sample_list if s.role == role}),
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
            }
    return result


def validate_participant_label(value: str) -> str:
    """Validate a consented, anonymous evaluator label before creating a group."""
    if not isinstance(value, str):
        raise ValueError("participant label must be text")
    label = value.strip()
    if not label or label.casefold() == "none":
        raise ValueError("participant label cannot be blank or 'none'")
    if len(label) > 40 or any(not (char.isalnum() or char in "_-") for char in label):
        raise ValueError("participant label must be 1-40 letters, digits, underscores, or hyphens")
    return label


def capture_evaluation(roll_number: str, camera_index: int,
                       thresholds: tuple[float, ...],
                       store: EmbeddingStore | None = None, *,
                       camera_factory=None, detector_factory=None,
                       recognizer_factory=None, stream_factory=None,
                       display_frame=None, poll_key=None) -> dict[str, object]:
    """Capture consented samples with keyboard-only participant transitions.

    No terminal input is requested while the OpenCV window is active. Type an
    anonymous participant label in the preview, press Enter, then G/I to choose
    role. Space captures, N completes that condition, C cancels it, and Q/Esc
    cancels the active condition and exits. The same label can be reused across
    conditions so score rows remain grouped by person.
    """
    store = store or EmbeddingStore()
    roll = store.validate_roll_number(roll_number)
    if hasattr(store, "get_student_embeddings_by_roll"):
        enrolled_templates = store.get_student_embeddings_by_roll(roll)
    else:
        enrolled_templates = store.load_all().get(roll, [])
    if not enrolled_templates:
        raise ValueError(f"roll number {roll} has no enrollment templates")

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
        timeout_property = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
        if timeout_property is not None:
            camera.set(timeout_property, 2500)
    except (AttributeError, TypeError, cv2.error):
        # Some camera backends do not support these hints. CameraStream still
        # enforces its own bounded wait around frame delivery.
        pass

    scores: list[EvaluationScore] = []
    stream = None
    try:
        # Models initialize after camera open, but camera remains open across
        # all participant groups and is released once in the outer finally.
        detector = (detector_factory or RetinaFaceDetector)()
        recognizer = (recognizer_factory or MobileFaceNetRecognizer)()
        engine = RecognitionEngine(store=store, detector=detector, recognizer=recognizer,
                                   threshold=0.65)
        stream = ((stream_factory or CameraStream)(camera, timeout_seconds=2.5))
        show = display_frame or (lambda image: cv2.imshow("Consent-based threshold evaluation", image))
        get_key = poll_key or (lambda: cv2.waitKey(1) & 0xFF)

        mode = "label"
        label_buffer = ""
        participant_id: str | None = None
        role: str | None = None
        count_for_group = 0
        group_start = 0
        status = "Type anonymous participant label, Enter confirms; F finishes; Esc exits."
        finished = False
        print("Participant entry is handled in the preview window; no mid-session terminal input is used.")
        print("Type one anonymous label per person; reuse it for that person's other capture conditions.")

        while not finished:
            try:
                frame = stream.read()
            except CameraReadError as exc:
                raise RuntimeError(f"Camera frame unavailable: {exc}") from exc

            faces = []
            quality = None
            quality_message = "Participant selection"
            if mode == "capture":
                faces = detector.detect(frame)
                if len(faces) == 0:
                    quality_message = "No face detected"
                elif len(faces) > 1:
                    quality_message = "Multiple faces; show one person only"
                else:
                    quality = assess_face_quality(frame, faces[0])
                    quality_message = quality.reason

            preview = frame.copy()
            if mode == "capture" and len(faces) == 1:
                face = faces[0]
                x, y, w, h = (int(round(v)) for v in face.box)
                cv2.rectangle(preview, (x, y), (x + w, y + h),
                              (0, 220, 0) if quality and quality.accepted else (0, 0, 255), 2)
                for px, py in face.landmarks:
                    cv2.circle(preview, (int(round(px)), int(round(py))), 2, (0, 180, 255), -1)

            if mode == "label":
                title = f"Participant label: {label_buffer}_"
                controls = "Type label | Enter continue | Backspace erase | F finish | Esc exit"
            elif mode == "role":
                title = f"Participant: {participant_id} | choose role"
                controls = "G = genuine | I = impostor | Backspace edit label | Esc exit"
            else:
                title = f"Participant: {participant_id} | Role: {role} | Valid samples: {count_for_group}"
                controls = "SPACE capture | N next condition/person | C cancel participant | Q/Esc exit"
            cv2.putText(preview, title[:100], (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (255, 255, 255), 2)
            cv2.putText(preview, quality_message[:90], (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(preview, controls[:110], (12, 84), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (255, 255, 255), 2)
            cv2.putText(preview, status[:100], (12, 112), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (200, 255, 200), 1)
            show(preview)
            key = get_key()
            if key < 0:
                continue
            char = chr(key & 0xFF) if 0 <= (key & 0xFF) < 128 else ""

            if mode == "label":
                if char.lower() == "f":
                    if label_buffer.strip():
                        status = "Finish requires an empty label; press Backspace to clear it first."
                    else:
                        print("Evaluation finished by operator; no participant record created.")
                        finished = True
                elif key == 27 or char.lower() == "q":
                    print("Evaluation exited; no active participant was being captured.")
                    finished = True
                elif key in (8, 127):
                    label_buffer = label_buffer[:-1]
                elif key in (10, 13):
                    try:
                        participant_id = validate_participant_label(label_buffer)
                    except ValueError as exc:
                        status = str(exc)
                    else:
                        mode = "role"
                        status = "Choose G for genuine or I for impostor."
                        print(f"Participant label entered: {participant_id}; awaiting role selection.")
                elif char and (char.isalnum() or char in "_-") and len(label_buffer) < 40:
                    label_buffer += char
            elif mode == "role":
                if key == 27 or char in ("q", "Q"):
                    print(f"Participant {participant_id} cancelled before capture; evaluation exiting.")
                    finished = True
                elif key in (8, 127):
                    participant_id = None
                    label_buffer = ""
                    mode = "label"
                elif char.lower() in ("g", "i"):
                    entered_role = "genuine" if char.lower() == "g" else "impostor"
                    previous_roles = {item.role for item in scores
                                      if item.person_id == participant_id}
                    if previous_roles and entered_role not in previous_roles:
                        status = "This label already has the other role; edit it or choose its original role."
                    else:
                        role = entered_role
                        count_for_group = 0
                        group_start = len(scores)
                        mode = "capture"
                        status = "Show exactly one clear face; Space captures a score only."
                        print(f"Processing participant={participant_id} role={role}; valid_samples=0")
            else:  # capture mode
                if key == 27 or char in ("q", "Q"):
                    removed = len(scores) - group_start
                    del scores[group_start:]
                    print(f"Cancelled current participant condition={participant_id}; discarded_samples={removed}; exiting.")
                    finished = True
                elif char.lower() == "c":
                    removed = len(scores) - group_start
                    del scores[group_start:]
                    print(f"Cancelled participant condition={participant_id}; discarded_samples={removed}; returning to label entry.")
                    participant_id = None
                    role = None
                    label_buffer = ""
                    count_for_group = 0
                    mode = "label"
                    status = "Participant condition cancelled; type the next anonymous label."
                elif char.lower() == "n":
                    print(f"Finished participant={participant_id} role={role}; valid_samples={count_for_group}. Ready for next label.")
                    participant_id = None
                    role = None
                    label_buffer = ""
                    count_for_group = 0
                    mode = "label"
                    status = "Type the next anonymous participant label."
                elif key == 32:
                    if len(faces) != 1 or quality is None or not quality.accepted:
                        reason = quality.reason if quality is not None else quality_message
                        print(f"Sample rejected participant={participant_id}; valid_samples={count_for_group}; reason={reason}.")
                        continue
                    try:
                        aligned = align_face(frame, faces[0].landmarks)
                        embedding = recognizer.generate_embedding(aligned)
                        score, _sample_count = engine.score_roll(roll, embedding)
                    except Exception as exc:
                        print(f"Sample rejected participant={participant_id}; valid_samples={count_for_group}; error={type(exc).__name__}.")
                        continue
                    scores.append(EvaluationScore(participant_id, role, score))
                    count_for_group += 1
                    print(f"Captured participant={participant_id} role={role}; valid_samples={count_for_group}; score={score:.4f}.")
                    del embedding
    finally:
        try:
            if stream is not None:
                stopped = stream.close()
                if not stopped:
                    detail = f" ({stream.close_error})" if stream.close_error else ""
                    print("Camera reader/release did not finish within the shutdown timeout; "
                          f"camera release was requested{detail}.")
            else:
                camera.release()
        except Exception as exc:
            print(f"Camera cleanup warning: {type(exc).__name__}: {exc}")
        finally:
            cv2.destroyAllWindows()

    if not any(item.role == "genuine" for item in scores) or not any(item.role == "impostor" for item in scores):
        raise ValueError("evaluation needs at least one valid genuine and one valid impostor capture")
    thresholds_report = evaluate_thresholds(scores, thresholds)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_identity": "authenticated evaluation target (omitted from report)",
        "enrolled_templates_compared_per_capture": len(enrolled_templates),
        "score_policy": "mean of up to top 3 cosine scores among the roll's enrollment templates",
        "threshold_comparison": "accept if score >= threshold",
        "person_grouping": "all captures for the same participant label are one person cluster",
        "score_distribution": score_distribution(scores),
        "person_score_summaries": list(summarize_scores(scores).values()),
        "thresholds": thresholds_report,
        "samples": [asdict(item) for item in scores],
    }


def parse_thresholds(value: str) -> tuple[float, ...]:
    try:
        thresholds = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated numbers") from exc
    if not thresholds or any(not math.isfinite(x) or not -1.0 <= x <= 1.0 for x in thresholds):
        raise argparse.ArgumentTypeError("provide thresholds in the range -1 through 1")
    return thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roll", required=True, help="enrolled roll number to evaluate against")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--consent-confirmed", action="store_true", required=True,
                        help="confirm every recorded participant consented to evaluation capture")
    parser.add_argument("--thresholds", type=parse_thresholds,
                        default=DEFAULT_THRESHOLDS,
                        help="candidate thresholds, comma-separated; default is a broad evaluation grid")
    parser.add_argument("--output", help="optional JSON results path inside face_system/; contains scores, no images")
    args = parser.parse_args()
    out_path: Path | None = None
    if args.output:
        out_path = Path(args.output).resolve()
        if not out_path.is_relative_to(FACE_SYSTEM_DIR.resolve()):
            print("Evaluation output must be inside Backend/face_system/.")
            return 2
    try:
        with pooled_embedding_store() as store:
            report = capture_evaluation(args.roll, args.camera, tuple(args.thresholds), store)
    except Exception as exc:
        print(f"Evaluation stopped: {exc}")
        return 2

    print("\nScore distribution by capture:")
    print(json.dumps(report["score_distribution"], indent=2))
    print("\nGrouped participant summaries (repeated captures grouped by label):")
    print(json.dumps(report["person_score_summaries"], indent=2))
    print("\nThreshold results:")
    print("threshold  person-median FAR/FRR  person-any-capture FAR/FRR  capture FAR/FRR")
    for row in report["thresholds"]:
        print(f"{row['threshold']:.2f}       "
              f"{row['person_median_far']:.3f}/{row['person_median_frr']:.3f}               "
              f"{row['person_any_capture_far']:.3f}/{row['person_any_capture_frr']:.3f}                  "
              f"{row['capture_far']:.3f}/{row['capture_frr']:.3f}")
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Score-only report saved: {out_path}")
    print("Thresholds are evaluation candidates only; do not choose one from this run unless participant coverage is adequate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
