"""Local comparison of webcam embeddings against stored student templates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

if __package__:
    from .detector import FaceDetection, RetinaFaceDetector
    from .embedding_store import EmbeddingStore
    from .quality import assess_face_quality
    from .recognizer import MobileFaceNetRecognizer, align_face
else:  # Preserve the existing standalone CLI execution from face_system/.
    from detector import FaceDetection, RetinaFaceDetector
    from embedding_store import EmbeddingStore
    from quality import assess_face_quality
    from recognizer import MobileFaceNetRecognizer, align_face


PROVISIONAL_THRESHOLD = 0.65


@dataclass(frozen=True)
class FaceMatch:
    roll_number: str
    similarity: float | None
    accepted: bool
    candidate_roll_number: str | None = None
    reason: str = ""
    second_best_roll_number: str | None = None
    second_best_similarity: float | None = None
    score_margin: float | None = None
    samples_compared: int = 0
    candidate_samples_compared: int = 0
    threshold_passed: bool = False
    margin_passed: bool | None = None
    threshold: float | None = None
    required_margin: float | None = None


class RecognitionEngine:
    """Compute per-face matches; threshold is provisional pending calibration."""

    def __init__(self, store: EmbeddingStore | None = None,
                 detector: RetinaFaceDetector | None = None,
                 recognizer: MobileFaceNetRecognizer | None = None,
                 threshold: float = PROVISIONAL_THRESHOLD,
                 minimum_margin: float | None = None) -> None:
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between -1 and 1")
        if minimum_margin is not None and not 0.0 <= minimum_margin <= 2.0:
            raise ValueError("minimum_margin must be between 0 and 2")
        self.store = store or EmbeddingStore()
        self.detector = detector or RetinaFaceDetector()
        self.recognizer = recognizer or MobileFaceNetRecognizer()
        self.threshold = threshold
        self.minimum_margin = minimum_margin
        # Templates are fetched on demand. The authenticated verification path
        # calls score_student_id(), so it never loads the open-set gallery.

    def reload_templates(self) -> None:
        """Compatibility no-op: stores are queried per operation, not cached."""

    def _compare_templates(self, embedding: np.ndarray,
                           templates: dict[object, list[np.ndarray]]) -> FaceMatch:
        if not templates:
            return FaceMatch("Unknown", None, False, reason="no enrollments",
                             samples_compared=0, threshold_passed=False, margin_passed=None,
                             threshold=self.threshold, required_margin=self.minimum_margin)
        candidates: list[tuple[str, float, int]] = []
        total_samples = 0
        for identity, stored in templates.items():
            scores = sorted(
                (self.recognizer.cosine_similarity(embedding, sample) for sample in stored),
                reverse=True,
            )
            total_samples += len(scores)
            aggregate = float(np.mean(scores[:3]))
            candidates.append((str(identity), aggregate, len(scores)))
        candidates.sort(key=lambda item: (-item[1], item[0]))
        best_roll, best_score, best_count = candidates[0]
        second_roll, second_score = (candidates[1][0], candidates[1][1]) if len(candidates) > 1 else (None, None)
        margin = best_score - second_score if second_score is not None else None
        threshold_passed = best_score >= self.threshold
        if margin is None:
            margin_passed: bool | None = True
        elif abs(margin) <= 1e-8:
            margin_passed = False
        elif self.minimum_margin is None:
            margin_passed = True
        else:
            margin_passed = margin >= self.minimum_margin
        accepted = threshold_passed and bool(margin_passed)
        reason = ("below provisional similarity threshold" if not threshold_passed else
                  "ambiguous: best-versus-second-best margin is too small" if not margin_passed else
                  "matched")
        return FaceMatch(
            best_roll if accepted else "Unknown", best_score, accepted,
            candidate_roll_number=best_roll, reason=reason,
            second_best_roll_number=second_roll, second_best_similarity=second_score,
            score_margin=margin, samples_compared=total_samples,
            candidate_samples_compared=best_count, threshold_passed=threshold_passed,
            margin_passed=margin_passed, threshold=self.threshold,
            required_margin=self.minimum_margin,
        )

    def compare_embedding(self, embedding: np.ndarray) -> FaceMatch:
        query = np.asarray(embedding, dtype=np.float32)
        if query.shape != (128,) or not np.isfinite(query).all():
            raise ValueError("query embedding must be a finite 128D vector")
        if float(np.linalg.norm(query)) <= 1e-12:
            raise ValueError("query embedding must have non-zero norm")
        templates = self.store.get_all_student_embeddings() if hasattr(
            self.store, "get_all_student_embeddings") else self.store.load_all()
        return self._compare_templates(query, templates)

    def score_student_id(self, student_id: int, embedding: np.ndarray) -> tuple[float, int]:
        """Compare only to one database student primary key's templates."""
        query = self._validate_query(embedding)
        templates = self.store.get_student_embeddings(student_id)
        if not templates:
            raise KeyError("student has no face enrollment")
        scores = sorted((self.recognizer.cosine_similarity(query, sample)
                         for sample in templates), reverse=True)
        return float(np.mean(scores[:3])), len(scores)

    @staticmethod
    def _validate_query(embedding: np.ndarray) -> np.ndarray:
        query = np.asarray(embedding, dtype=np.float32)
        if query.shape != (128,) or not np.isfinite(query).all():
            raise ValueError("query embedding must be a finite 128D vector")
        if float(np.linalg.norm(query)) <= 1e-12:
            raise ValueError("query embedding must have non-zero norm")
        return query

    def score_roll(self, roll_number: str, embedding: np.ndarray) -> tuple[float, int]:
        """Return current top-three-mean score and template count for one roll."""
        roll = self.store.validate_roll_number(roll_number)
        query = self._validate_query(embedding)
        if hasattr(self.store, "get_student_embeddings_by_roll"):
            stored = self.store.get_student_embeddings_by_roll(roll)
        else:
            stored = self.store.load_all().get(roll)
        if not stored:
            raise KeyError(f"roll number {roll} is not enrolled")
        scores = sorted(
            (self.recognizer.cosine_similarity(query, sample) for sample in stored), reverse=True
        )
        return float(np.mean(scores[:3])), len(scores)

    @staticmethod
    def format_diagnostics(match: FaceMatch) -> str:
        best = match.candidate_roll_number or "none"
        best_score = "n/a" if match.similarity is None else f"{match.similarity:.4f}"
        second = ("none" if match.second_best_roll_number is None else
                  f"{match.second_best_roll_number}:{match.second_best_similarity:.4f}")
        margin = "n/a" if match.score_margin is None else f"{match.score_margin:.4f}"
        threshold = "n/a" if match.threshold is None else f"{match.threshold:.4f}"
        required_margin = ("disabled" if match.required_margin is None
                           else f"{match.required_margin:.4f}")
        return (
            f"best={best}:{best_score} second={second} margin={margin} "
            "aggregation=mean_top3 "
            f"samples_compared={match.samples_compared} "
            f"best_samples={match.candidate_samples_compared} "
            f"threshold={threshold} threshold_passed={match.threshold_passed} "
            f"required_margin={required_margin} "
            f"margin_passed={match.margin_passed} "
            f"result={'MATCH' if match.accepted else 'Unknown'} reason={match.reason}"
        )

    def recognize_frame(self, frame_bgr: np.ndarray) -> tuple[list[FaceDetection], list[FaceMatch]]:
        faces = self.detector.detect(frame_bgr)
        provisional: list[FaceMatch] = []
        for face in faces:
            try:
                quality = assess_face_quality(frame_bgr, face)
                if not quality.accepted:
                    provisional.append(FaceMatch("Unknown", None, False, reason=quality.reason))
                    continue
                aligned = align_face(frame_bgr, face.landmarks)
                embedding = self.recognizer.generate_embedding(aligned)
                provisional.append(self.compare_embedding(embedding))
                del embedding
            except Exception as exc:
                provisional.append(FaceMatch(
                    "Unknown", None, False, reason=f"face processing failed: {type(exc).__name__}"
                ))

        # One physical student cannot be assigned to two detected faces in the
        # same frame. If a roll is the accepted top result for multiple faces,
        # mark every conflicting result Unknown instead of guessing which face.
        occurrences: dict[str, list[int]] = {}
        for idx, match in enumerate(provisional):
            if match.accepted and match.candidate_roll_number:
                occurrences.setdefault(match.candidate_roll_number, []).append(idx)
        for roll, indexes in occurrences.items():
            if len(indexes) > 1:
                for idx in indexes:
                    old = provisional[idx]
                    provisional[idx] = FaceMatch(
                        "Unknown", old.similarity, False, roll,
                        "ambiguous: same student was top candidate for multiple faces",
                        old.second_best_roll_number, old.second_best_similarity,
                        old.score_margin, old.samples_compared,
                        old.candidate_samples_compared, old.threshold_passed,
                        False, old.threshold, old.required_margin,
                    )
        return faces, provisional
