"""CPU RetinaFace MobileNetV2 inference for yakhyo/retinaface-pytorch ONNX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort


MODEL_PATH = Path(__file__).resolve().parent / "models" / "retinaface_mv2.onnx"
MEAN_BGR = np.asarray((104.0, 117.0, 123.0), dtype=np.float32)
STEPS = (8, 16, 32)
MIN_SIZES = ((16, 32), (64, 128), (256, 512))
VARIANCES = (0.1, 0.2)


@dataclass(frozen=True)
class FaceDetection:
    """A face box in xywh pixel coordinates and five xy landmark pairs."""

    box: tuple[float, float, float, float]
    landmarks: tuple[tuple[float, float], ...]
    confidence: float


def _anchors(height: int, width: int) -> np.ndarray:
    anchors: list[tuple[float, float, float, float]] = []
    for step, sizes in zip(STEPS, MIN_SIZES):
        map_h = (height + step - 1) // step
        map_w = (width + step - 1) // step
        for row in range(map_h):
            for col in range(map_w):
                for size in sizes:
                    anchors.append(
                        ((col + 0.5) * step / width,
                         (row + 0.5) * step / height,
                         size / width,
                         size / height)
                    )
    return np.asarray(anchors, dtype=np.float32)


def _decode(loc: np.ndarray, landmark_deltas: np.ndarray, priors: np.ndarray,
            width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    center = priors[:, :2] + loc[:, :2] * VARIANCES[0] * priors[:, 2:]
    size = priors[:, 2:] * np.exp(np.clip(loc[:, 2:] * VARIANCES[1], -50, 50))
    boxes = np.concatenate((center - size / 2, center + size / 2), axis=1)
    boxes *= np.asarray((width, height, width, height), dtype=np.float32)

    landmarks = landmark_deltas.reshape(-1, 5, 2)
    landmarks = priors[:, None, :2] + landmarks * VARIANCES[0] * priors[:, None, 2:]
    landmarks *= np.asarray((width, height), dtype=np.float32)
    return boxes, landmarks


class RetinaFaceDetector:
    """Load the selected dynamic-shape ONNX graph and detect faces in BGR frames."""

    def __init__(self, model_path: str | Path = MODEL_PATH,
                 confidence_threshold: float = 0.6,
                 nms_threshold: float = 0.4) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"RetinaFace ONNX model not found: {self.model_path}")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be between 0 and 1")
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise RuntimeError(f"Could not load RetinaFace ONNX model: {exc}") from exc
        self.input = self.session.get_inputs()[0]
        self.output_names = [out.name for out in self.session.get_outputs()]
        required = {"loc", "conf", "landmarks"}
        if not required.issubset(self.output_names):
            raise RuntimeError(
                f"Unexpected RetinaFace outputs {self.output_names}; expected {sorted(required)}"
            )
        if self.input.type != "tensor(float)":
            raise RuntimeError(f"Expected float input tensor, got {self.input.type}")
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    @staticmethod
    def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Expected a non-empty BGR image with three channels")
        image = frame_bgr.astype(np.float32)
        image -= MEAN_BGR
        return np.ascontiguousarray(image.transpose(2, 0, 1)[None, ...])

    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        height, width = frame_bgr.shape[:2]
        tensor = self.preprocess(frame_bgr)
        outputs = self.session.run(self.output_names, {self.input.name: tensor})
        by_name = dict(zip(self.output_names, outputs))
        loc = np.asarray(by_name["loc"])[0]
        conf = np.asarray(by_name["conf"])[0]
        landmark_deltas = np.asarray(by_name["landmarks"])[0]
        priors = _anchors(height, width)
        if not (len(loc) == len(conf) == len(landmark_deltas) == len(priors)):
            raise RuntimeError(
                "Model output count does not match RetinaFace anchors for "
                f"{width}x{height}: {len(loc)}, {len(conf)}, "
                f"{len(landmark_deltas)} vs {len(priors)}"
            )

        scores = conf[:, 1]
        selected = np.flatnonzero(scores >= self.confidence_threshold)
        if not len(selected):
            return []
        boxes, landmarks = _decode(loc[selected], landmark_deltas[selected],
                                    priors[selected], width, height)
        scores = scores[selected]
        # OpenCV NMSBoxes takes x,y,width,height boxes.
        xywh = np.column_stack((boxes[:, 0], boxes[:, 1],
                                boxes[:, 2] - boxes[:, 0],
                                boxes[:, 3] - boxes[:, 1]))
        keep = cv2.dnn.NMSBoxes(
            xywh.tolist(), scores.astype(float).tolist(),
            self.confidence_threshold, self.nms_threshold
        )
        if len(keep) == 0:
            return []
        keep = np.asarray(keep).reshape(-1)
        results: list[FaceDetection] = []
        for idx in keep:
            x1, y1, x2, y2 = boxes[idx]
            x1 = float(np.clip(x1, 0, width - 1))
            y1 = float(np.clip(y1, 0, height - 1))
            x2 = float(np.clip(x2, 0, width - 1))
            y2 = float(np.clip(y2, 0, height - 1))
            points = tuple((float(np.clip(x, 0, width - 1)),
                            float(np.clip(y, 0, height - 1)))
                           for x, y in landmarks[idx])
            results.append(FaceDetection(
                (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)),
                points, float(scores[idx])
            ))
        return results
