"""Latest-frame reader with bounded waits for camera failures."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass


class CameraReadError(RuntimeError):
    """Raised when a webcam repeatedly fails or times out while reading."""


@dataclass(frozen=True)
class _ReadFailure:
    message: str


class CameraStream:
    """Read webcam frames on a daemon thread; main loop never waits indefinitely."""

    def __init__(self, camera, *, timeout_seconds: float = 2.5,
                 max_consecutive_failures: int = 3, failure_pause: float = 0.1) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be at least one")
        self.camera = camera
        self.timeout_seconds = timeout_seconds
        self.max_consecutive_failures = max_consecutive_failures
        self.failure_pause = failure_pause
        self._queue: queue.Queue[object] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self.close_error: Exception | None = None
        self._thread = threading.Thread(target=self._reader, name="face-eval-camera", daemon=True)
        self._thread.start()

    def _publish_latest(self, item: object) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

    def _reader(self) -> None:
        failures = 0
        while not self._stop.is_set():
            try:
                ok, frame = self.camera.read()
            except Exception as exc:
                self._publish_latest(_ReadFailure(f"camera read raised {type(exc).__name__}: {exc}"))
                return
            if self._stop.is_set():
                return
            if not ok or frame is None:
                failures += 1
                if failures >= self.max_consecutive_failures:
                    self._publish_latest(_ReadFailure(
                        f"camera returned no frame {failures} consecutive times"
                    ))
                    return
                self._stop.wait(self.failure_pause)
                continue
            failures = 0
            self._publish_latest(frame)

    def read(self):
        try:
            item = self._queue.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            raise CameraReadError(
                f"timed out after {self.timeout_seconds:.1f}s waiting for a webcam frame"
            ) from exc
        if isinstance(item, _ReadFailure):
            raise CameraReadError(item.message)
        return item

    def close(self) -> bool:
        """Request stop, release the camera, and join briefly; return if thread stopped."""
        self._stop.set()
        release_errors: list[Exception] = []

        def release_camera() -> None:
            try:
                self.camera.release()
            except Exception as exc:
                release_errors.append(exc)

        # Some camera backends hold an internal lock while read() is stalled.
        # Request release on a separate daemon so shutdown itself stays bounded.
        release_thread = threading.Thread(target=release_camera,
                                          name="face-eval-camera-release", daemon=True)
        release_thread.start()
        join_timeout = max(1.0, self.timeout_seconds)
        self._thread.join(timeout=join_timeout)
        release_thread.join(timeout=join_timeout)
        self.close_error = release_errors[0] if release_errors else None
        return not self._thread.is_alive() and not release_thread.is_alive() and self.close_error is None
