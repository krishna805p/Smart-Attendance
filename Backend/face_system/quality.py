"""Small local image-quality checks for webcam face samples."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

if __package__:
    from .detector import FaceDetection
else:  # Preserve direct imports used by the existing local CLI.
    from detector import FaceDetection


@dataclass(frozen=True)
class QualityResult:
    accepted: bool
    reason: str
    blur_score: float | None = None
    brightness: float | None = None


def assess_face_quality(frame_bgr: np.ndarray, face: FaceDetection,
                        minimum_side: int = 80, minimum_blur_score: float = 45.0,
                        brightness_range: tuple[float, float] = (30.0, 225.0)) -> QualityResult:
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        return QualityResult(False, "invalid frame")
    if len(face.landmarks) != 5 or not np.isfinite(np.asarray(face.landmarks)).all():
        return QualityResult(False, "invalid facial landmarks")
    x, y, width, height = face.box
    if not np.isfinite((x, y, width, height)).all() or width <= 0 or height <= 0:
        return QualityResult(False, "invalid face box")
    if width < minimum_side or height < minimum_side:
        return QualityResult(False, f"face too small (minimum {minimum_side}px per side)")
    frame_h, frame_w = frame_bgr.shape[:2]
    left = max(0, int(np.floor(x)))
    top = max(0, int(np.floor(y)))
    right = min(frame_w, int(np.ceil(x + width)))
    bottom = min(frame_h, int(np.ceil(y + height)))
    roi = frame_bgr[top:bottom, left:right]
    if roi.size == 0 or roi.shape[0] < minimum_side or roi.shape[1] < minimum_side:
        return QualityResult(False, "face crop is outside the frame")
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    if blur_score < minimum_blur_score:
        return QualityResult(False, "face image is blurry", blur_score, brightness)
    if not brightness_range[0] <= brightness <= brightness_range[1]:
        return QualityResult(False, "face lighting is too dark or overexposed", blur_score, brightness)
    return QualityResult(True, "quality accepted", blur_score, brightness)
