"""End-to-end ODP orchestration for Observe -> Diagnose -> Prescribe."""

from __future__ import annotations

import os

from diagnose.fusion.model import FusionLSTM
from diagnose.vision.inference import VisionInference
from prescribe.parser import PrescriptionParser
from prescribe.llm import run_local_llm
from observe.imu import IMUReader
from observe.camera import CameraCapture
from observe.filtering import butterworth_bandpass


class ODPOrchestrator:
    """Coordinate a full observe -> diagnose -> prescribe loop."""

    def __init__(self, simulate=None):
        if simulate is None:
            mode = os.getenv("MOTORGUARD_HARDWARE_MODE", "simulate").lower()
            simulate = mode != "real"
        self.simulate = simulate
        self.fusion_model = FusionLSTM()
        self.parser = PrescriptionParser()
        self.vision_inference = VisionInference(model_path="")

    def run_once(self):
        """Run one full loop through observe, diagnose, and prescribe stages."""
        imu = IMUReader(simulate=self.simulate)
        camera = CameraCapture(simulate=self.simulate)

        vibration_window = imu.read_window(length=256)
        filtered = butterworth_bandpass(vibration_window, 10.0, 500.0, 1500.0)
        vibration_features = [float(abs(filtered).mean()), float(abs(filtered).std()), 0.0, 0.0, 0.0]

        vision_embedding = [0.0] * 512
        mean_score, variance = self.fusion_model.predict_with_mc_dropout(vision_embedding, vibration_features)

        camera_frame = camera.capture_frame()
        vision_result = self.vision_inference.predict(camera_frame.get("frame") if isinstance(camera_frame, dict) else None)
        llm_output = run_local_llm(
            "Immediate Action: Isolate the motor\nRepair Protocol: Inspect bearings\nPreventive Schedule: Recheck in 30 days",
            os.getenv("MOTORGUARD_LLM_MODEL", ""),
        )
        parsed = self.parser.parse(llm_output)

        return {
            "mode": "real" if not self.simulate else "simulated",
            "camera": camera_frame,
            "vibration_features": vibration_features,
            "fusion_score": mean_score,
            "fusion_variance": variance,
            "vision": vision_result,
            "prescription": parsed,
        }
