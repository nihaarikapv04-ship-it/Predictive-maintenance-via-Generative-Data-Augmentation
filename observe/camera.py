"""Camera acquisition utilities for the Observe layer.

This module provides a lightweight webcam abstraction that can run against
real hardware or a simulated stream during development.
"""

import os


class CameraCapture:
    """Capture frames from a real camera or a simulated placeholder."""

    def __init__(self, source: int = 0, simulate=None):
        if simulate is None:
            mode = os.getenv("MOTORGUARD_HARDWARE_MODE", "simulate").lower()
            simulate = mode != "real"
        self.source = source
        self.simulate = simulate

    def capture_frame(self):
        """Return a single frame from the camera or a synthetic placeholder."""
        if self.simulate:
            return {"mode": "simulated", "source": self.source}
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment fallback
            raise RuntimeError("Real camera capture requires OpenCV and a connected camera.") from exc

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError("Unable to open camera device")
        try:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Unable to read camera frame")
            return {"mode": "real", "frame": frame}
        finally:
            cap.release()
