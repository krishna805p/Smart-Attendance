"""Qualcomm MobileFaceNet ONNX alignment and embedding inference (CPU)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


MODEL_PATH = Path(__file__).resolve().parent / "models" / "mobilefacenet.onnx"
FACE_SIZE = (112, 112)  # width, height

# Qualcomm's reference template is generated from the original 96x112 template,
# square padding, then 25% inner padding for a 112x112 output. The detector's
# order is right-eye, left-eye, nose, right-mouth, left-mouth, so the coordinates
# below are in that same order (the Qualcomm app documents the opposite order).
_BASE_LEFT_FIRST = np.asarray(
    ((30.29459953, 51.69630051),
     (65.53179932, 51.50139999),
     (48.02519989, 71.73660278),
     (33.54930115, 92.3655014),
     (62.72990036, 92.20410156)),
    dtype=np.float32,
)
_SQUARE_LEFT_FIRST = _BASE_LEFT_FIRST + np.asarray((8.0, 0.0), dtype=np.float32)
_QUALCOMM_TEMPLATE_LEFT_FIRST = (_SQUARE_LEFT_FIRST + 28.0) * (112.0 / 168.0)
REFERENCE_LANDMARKS = _QUALCOMM_TEMPLATE_LEFT_FIRST[[1, 0, 2, 4, 3]].copy()


def align_face(frame_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Warp a BGR frame to Qualcomm's 112x112 five-point reference geometry."""
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be a three-channel BGR image")
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("landmarks must contain five finite (x, y) points")

    # Fit a non-reflecting similarity transform using Umeyama least squares.
    source_mean = points.mean(axis=0)
    target_mean = REFERENCE_LANDMARKS.mean(axis=0)
    source_centered = points - source_mean
    target_centered = REFERENCE_LANDMARKS - target_mean
    variance = float(np.sum(source_centered * source_centered) / len(points))
    if variance < 1e-6:
        raise ValueError("landmarks are degenerate; cannot align face")
    covariance = target_centered.T @ source_centered / len(points)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.eye(2, dtype=np.float32)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1.0
    rotation = u @ sign @ vt
    scale = float(np.sum(singular_values * np.diag(sign)) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("landmarks produced an invalid alignment scale")
    linear = scale * rotation
    translation = target_mean - linear @ source_mean
    transform = np.column_stack((linear, translation)).astype(np.float32)
    aligned = cv2.warpAffine(
        frame_bgr, transform, FACE_SIZE, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    if aligned.shape != (112, 112, 3):
        raise RuntimeError(f"alignment returned unexpected shape {aligned.shape}")
    return aligned


class MobileFaceNetRecognizer:
    """Generate normalized 128D embeddings with Qualcomm's universal float ONNX."""

    def __init__(self, model_path: str | Path = MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MobileFaceNet ONNX model not found: {self.model_path}")
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise RuntimeError(f"Could not load MobileFaceNet ONNX model: {exc}") from exc
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        if not self.inputs or not self.outputs:
            raise RuntimeError("MobileFaceNet graph has no inputs or outputs")
        for info in self.inputs:
            if info.type != "tensor(float)":
                raise RuntimeError(f"Expected float32 input {info.name}; got {info.type}")

    @staticmethod
    def preprocess(aligned_bgr: np.ndarray) -> np.ndarray:
        """Qualcomm app path: BGR->RGB, float [0,1], CHW, batch dimension.

        The exported graph applies the model's documented ImageNet mean/std
        normalization internally; do not normalize a second time here.
        """
        if aligned_bgr is None or aligned_bgr.shape != (112, 112, 3):
            raise ValueError("expected an aligned 112x112 BGR face")
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])

    def generate_embedding(self, aligned_bgr: np.ndarray) -> np.ndarray:
        tensor = self.preprocess(aligned_bgr)
        if len(self.inputs) == 1:
            feed = {self.inputs[0].name: tensor}
        elif len(self.inputs) == 2:
            # Qualcomm's exported verification graph accepts an image pair and
            # returns one embedding per side. Supplying the same face twice
            # exposes the single-image embedding path without inventing data.
            feed = {info.name: tensor for info in self.inputs}
        else:
            raise RuntimeError(
                f"Unsupported MobileFaceNet input count {len(self.inputs)}; expected one or two"
            )

        raw_outputs = self.session.run(None, feed)
        candidates = [np.asarray(value).reshape(-1, 128) for value in raw_outputs
                      if np.asarray(value).size and np.asarray(value).size % 128 == 0]
        if not candidates:
            shapes = [np.asarray(value).shape for value in raw_outputs]
            raise RuntimeError(f"No 128D embedding output found; graph returned {shapes}")
        vector = candidates[0][0].astype(np.float32, copy=False)
        if not np.isfinite(vector).all():
            raise RuntimeError("MobileFaceNet produced a non-finite embedding")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise RuntimeError("MobileFaceNet produced a zero or invalid embedding")
        return np.ascontiguousarray(vector / norm, dtype=np.float32)

    @staticmethod
    def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
        a = np.asarray(first, dtype=np.float32).reshape(-1)
        b = np.asarray(second, dtype=np.float32).reshape(-1)
        if a.shape != (128,) or b.shape != (128,):
            raise ValueError("cosine_similarity expects two 128D embeddings")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("embeddings must be finite")
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-12:
            raise ValueError("cannot compare a zero-norm embedding")
        return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
