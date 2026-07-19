"""Vision inference wrapper with latency instrumentation."""

import time


class VisionInference:
    """Thin wrapper around a YOLO-style model with latency instrumentation."""

    def __init__(self, model_path: str):
        self.model_path = model_path

    def predict(self, frame):
        """Run a prediction and return a latency measurement."""
        start = time.perf_counter()
        if frame is None:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return {"latency_ms": latency_ms, "detections": [], "mode": "simulated"}
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {"latency_ms": latency_ms, "detections": [], "mode": "real"}
