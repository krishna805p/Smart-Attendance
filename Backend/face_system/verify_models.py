"""Smoke verification for the two ONNX models; does not test identity recognition."""

from __future__ import annotations

import numpy as np

from detector import RetinaFaceDetector
from recognizer import MobileFaceNetRecognizer, REFERENCE_LANDMARKS, align_face


def main() -> int:
    detector = RetinaFaceDetector()
    recognizer = MobileFaceNetRecognizer()
    assert "CPUExecutionProvider" in detector.session.get_providers()
    assert "CPUExecutionProvider" in recognizer.session.get_providers()
    print("RetinaFace loaded on CPU")
    print("RetinaFace input:", [(x.name, x.shape, x.type) for x in detector.session.get_inputs()])
    print("RetinaFace outputs:", [(x.name, x.shape, x.type) for x in detector.session.get_outputs()])
    print("MobileFaceNet loaded on CPU")
    print("MobileFaceNet inputs:", [(x.name, x.shape, x.type) for x in recognizer.inputs])
    print("MobileFaceNet outputs:", [(x.name, x.shape, x.type) for x in recognizer.outputs])

    # Synthetic landmarks/canvas verify transform plumbing only; they are not
    # a photographed face or identity-recognition sample.
    canvas = np.zeros((180, 180, 3), dtype=np.uint8)
    crop = align_face(canvas, REFERENCE_LANDMARKS + np.asarray((30, 30), dtype=np.float32))
    assert crop.shape == (112, 112, 3)
    print("Synthetic five-point alignment: PASS (112x112)")
    first = recognizer.generate_embedding(crop)
    second = recognizer.generate_embedding(crop)
    assert first.shape == (128,)
    assert np.isfinite(first).all()
    assert np.isclose(np.linalg.norm(first), 1.0, atol=1e-5)
    assert np.isclose(recognizer.cosine_similarity(first, first), 1.0, atol=1e-5)
    repeat_cosine = recognizer.cosine_similarity(first, second)
    assert np.isclose(repeat_cosine, 1.0, atol=1e-4), repeat_cosine
    print("Embedding smoke inference: PASS (128D, finite, L2-normalized)")
    print(f"Self cosine: {recognizer.cosine_similarity(first, first):.6f}")
    print(f"Repeated-input cosine: {repeat_cosine:.6f}")
    print("This confirms model inference only, not real-world identity recognition.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
