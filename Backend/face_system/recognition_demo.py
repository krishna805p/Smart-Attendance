"""Live RetinaFace + MobileFaceNet embedding demo; no identity matching/storage."""

from __future__ import annotations

import argparse
import time

import cv2

from detector import RetinaFaceDetector
from recognizer import MobileFaceNetRecognizer, align_face


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.6)
    args = parser.parse_args()
    try:
        detector = RetinaFaceDetector(confidence_threshold=args.confidence)
        recognizer = MobileFaceNetRecognizer()
    except Exception as exc:
        print(f"Model initialization failed: {exc}")
        return 2
    print("RetinaFace providers:", detector.session.get_providers())
    print("MobileFaceNet providers:", recognizer.session.get_providers())
    print("MobileFaceNet inputs:", [(x.name, x.shape, x.type) for x in recognizer.inputs])
    print("MobileFaceNet outputs:", [(x.name, x.shape, x.type) for x in recognizer.outputs])

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        print(f"Camera unavailable (index {args.camera}).")
        return 3

    frame_count = 0
    fps_started = time.perf_counter()
    fps = 0.0
    print("Embedding demo started. No embeddings are saved. Press Q or Esc to quit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Camera frame read failed; stopping.")
                break
            started = time.perf_counter()
            faces = detector.detect(frame)
            statuses: list[tuple[object, str]] = []
            for face in faces:
                try:
                    aligned = align_face(frame, face.landmarks)
                    embedding = recognizer.generate_embedding(aligned)
                    statuses.append((face, f"EMBEDDING OK ({embedding.shape[0]}D)"))
                    del embedding
                except Exception as exc:
                    statuses.append((face, f"EMBEDDING ERROR: {type(exc).__name__}"))
            inference_ms = (time.perf_counter() - started) * 1000.0
            frame_count += 1
            elapsed = time.perf_counter() - fps_started
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_started = time.perf_counter()

            for face, status in statuses:
                x, y, width, height = (int(round(v)) for v in face.box)
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 220, 0), 2)
                cv2.putText(frame, f"{face.confidence:.2f} {status}",
                            (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (0, 220, 0), 1)
                for px, py in face.landmarks:
                    cv2.circle(frame, (int(round(px)), int(round(py))), 2, (0, 180, 255), -1)
            cv2.putText(frame, f"Faces: {len(faces)}  FPS: {fps:.1f}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Detection + embedding: {inference_ms:.1f} ms", (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("RetinaFace + MobileFaceNet embeddings (CPU)", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
