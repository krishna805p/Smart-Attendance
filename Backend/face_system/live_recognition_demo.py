"""Identify locally enrolled roll numbers from a live webcam (provisional threshold)."""

from __future__ import annotations

import argparse
import time

import cv2

from embedding_store import EmbeddingStoreError
from recognition import PROVISIONAL_THRESHOLD, RecognitionEngine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=PROVISIONAL_THRESHOLD,
                        help="provisional cosine threshold; calibrate with real samples")
    parser.add_argument("--minimum-margin", type=float, default=None,
                        help="optional calibrated best-vs-runner-up margin; not useful with one roll")
    parser.add_argument("--diagnostics", action="store_true",
                        help="print score diagnostics at most once per second")
    args = parser.parse_args()
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        print(f"Camera unavailable (index {args.camera}); recognition did not start.")
        return 3
    try:
        engine = RecognitionEngine(threshold=args.threshold,
                                   minimum_margin=args.minimum_margin)
    except (ValueError, EmbeddingStoreError, RuntimeError, FileNotFoundError) as exc:
        camera.release()
        print(f"Recognition initialization failed: {exc}")
        return 2

    frame_count = 0
    fps_start = time.perf_counter()
    fps = 0.0
    diagnostics_at = 0.0
    print(f"Threshold {args.threshold:.3f} is PROVISIONAL. Press Q or Esc to quit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Camera frame read failed; stopping recognition.")
                break
            started = time.perf_counter()
            faces, results = engine.recognize_frame(frame)
            processing_ms = (time.perf_counter() - started) * 1000.0
            if args.diagnostics and time.monotonic() - diagnostics_at >= 1.0:
                for face_index, result in enumerate(results):
                    print(f"face={face_index + 1} {engine.format_diagnostics(result)}")
                diagnostics_at = time.monotonic()
            frame_count += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.perf_counter()
            for face, result in zip(faces, results):
                x, y, width, height = (int(round(v)) for v in face.box)
                label = result.roll_number
                if result.similarity is not None:
                    label += f" {result.similarity:.3f}"
                color = (0, 220, 0) if result.accepted else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(frame, label[:60], (x, max(18, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
                for px, py in face.landmarks:
                    cv2.circle(frame, (int(round(px)), int(round(py))), 2, (0, 180, 255), -1)
            cv2.putText(frame, f"Faces: {len(faces)}  FPS: {fps:.1f}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Provisional threshold: {args.threshold:.2f} | {processing_ms:.1f} ms",
                        (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow("Local student face recognition", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
