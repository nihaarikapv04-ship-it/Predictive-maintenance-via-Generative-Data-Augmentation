"""
MotorGuard AI — Vision Module (Induction Motor Surface Defect Detection)
=========================================================================

This module captures live webcam frames pointed at an induction motor and uses
YOLOv11n to classify the motor's surface condition into one of six classes:

    1. Healthy Baseline       — clean surface, no defects
    2. Mild Oxidation         — slight rust or discoloration
    3. Moderate Corrosion     — visible corrosion spreading
    4. Severe Corrosion       — heavy corrosion, structural concern
    5. Structural Cracking    — visible cracks on motor housing
    6. Contamination          — oil, dust, or foreign material

Based on detection results the module determines:
    • HEALTHY vs FAULTY motor status
    • Overall confidence score
    • Whether the RAG repair pipeline should be triggered

Color coding (BGR for OpenCV):
    Green  = Healthy Baseline
    Yellow = Mild Oxidation
    Orange = Moderate Corrosion
    Red    = Severe Corrosion / Structural Cracking
    Purple = Contamination
"""

import base64
import logging
import math
import time
import random
from typing import Dict, Any, Tuple, List, Optional
from collections import deque

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Optional dependency: ultralytics (YOLO) + PyTorch
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
    import torch
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The six motor surface condition classes
MOTOR_CONDITIONS: List[str] = [
    "Healthy Baseline",
    "Mild Oxidation",
    "Moderate Corrosion",
    "Severe Corrosion",
    "Structural Cracking",
    "Contamination",
]

# Map each condition → overall motor status
_CONDITION_TO_STATUS: Dict[str, str] = {
    "Healthy Baseline":    "HEALTHY",
    "Mild Oxidation":      "FAULTY",
    "Moderate Corrosion":  "FAULTY",
    "Severe Corrosion":    "FAULTY",
    "Structural Cracking": "FAULTY",
    "Contamination":       "FAULTY",
}

# Severity weight (0‑1) used to compute a combined severity score
_SEVERITY_WEIGHT: Dict[str, float] = {
    "Healthy Baseline":    0.00,
    "Mild Oxidation":      0.20,
    "Moderate Corrosion":  0.50,
    "Severe Corrosion":    0.85,
    "Structural Cracking": 0.95,
    "Contamination":       0.40,
}

# BGR colors for OpenCV drawing
_CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "Healthy Baseline":    (0, 255, 0),      # Green
    "Mild Oxidation":      (0, 255, 255),     # Yellow
    "Moderate Corrosion":  (0, 165, 255),     # Orange
    "Severe Corrosion":    (0, 0, 255),       # Red
    "Structural Cracking": (0, 0, 255),       # Red
    "Contamination":       (255, 0, 128),     # Purple
}

# Hex colors for the frontend dashboard
CLASS_COLORS_HEX: Dict[str, str] = {
    "Healthy Baseline":    "#00ff00",
    "Mild Oxidation":      "#ffff00",
    "Moderate Corrosion":  "#ff8c00",
    "Severe Corrosion":    "#ff0000",
    "Structural Cracking": "#ff0000",
    "Contamination":       "#9b30ff",
}

# CLAHE parameters
_CLAHE_CLIP_LIMIT = 2.0
_CLAHE_TILE_SIZE  = (8, 8)


# ═══════════════════════════════════════════════════════════════════════════
# VisionObserver  —  real YOLOv11n inference
# ═══════════════════════════════════════════════════════════════════════════

class VisionObserver:
    """
    Captures frames from a webcam, applies CLAHE pre‑processing, and runs
    YOLOv11n inference to detect & classify surface defects on an induction
    motor.  Falls back to ``SimulatedVisionObserver`` if the model or
    required libraries are unavailable.
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence_threshold: float = 0.25,
        device: str = "auto",
        webcam_index: int = 0,
    ):
        """
        Args:
            model_path:            Path to YOLOv11n weights (.pt).
            confidence_threshold:  Minimum detection confidence.
            device:                'auto' | 'cpu' | 'cuda' | 'mps'.
            webcam_index:          OpenCV VideoCapture device index.
        """
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.model: Optional[Any] = None
        self.is_simulated = False
        self.device = device
        self.webcam_index = webcam_index
        self._cap: Optional[cv2.VideoCapture] = None

        # Rolling history for trend analysis (last 100 readings)
        self.detection_history: deque = deque(maxlen=100)

        # ---- Load model ----
        if not ULTRALYTICS_AVAILABLE:
            logger.warning(
                "ultralytics / torch not installed — falling back to simulation."
            )
            self.is_simulated = True
            return

        try:
            if self.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"

            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            logger.info(
                f"YOLO model loaded from {self.model_path} on device={self.device}"
            )
        except Exception as exc:
            logger.error(f"Failed to load YOLO model: {exc} — using simulation.")
            self.is_simulated = True

    # ------------------------------------------------------------------
    # Webcam helpers
    # ------------------------------------------------------------------

    def open_webcam(self) -> bool:
        """Open the webcam. Returns True on success."""
        if self._cap is not None and self._cap.isOpened():
            return True
        try:
            self._cap = cv2.VideoCapture(self.webcam_index)
            if not self._cap.isOpened():
                logger.error(f"Cannot open webcam index {self.webcam_index}")
                return False
            # Set reasonable resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            logger.info(f"Webcam {self.webcam_index} opened successfully.")
            return True
        except Exception as exc:
            logger.error(f"Webcam open error: {exc}")
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the webcam (BGR)."""
        if self._cap is None or not self._cap.isOpened():
            if not self.open_webcam():
                return None
        ret, frame = self._cap.read()
        if not ret:
            logger.warning("Failed to read frame from webcam.")
            return None
        return frame

    def release_webcam(self) -> None:
        """Release the webcam resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Webcam released.")

    # ------------------------------------------------------------------
    # Pre‑processing
    # ------------------------------------------------------------------

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to the L channel in LAB colour space for low‑light
        enhancement.  Returns the enhanced BGR frame.
        """
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=_CLAHE_CLIP_LIMIT, tileGridSize=_CLAHE_TILE_SIZE
            )
            l_ch = clahe.apply(l_ch)
            enhanced = cv2.merge((l_ch, a_ch, b_ch))
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        except Exception as exc:
            logger.error(f"CLAHE preprocessing failed: {exc}")
            return frame

    # ------------------------------------------------------------------
    # Colour / status helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_fault_color(class_name: str) -> Tuple[int, int, int]:
        """BGR colour for a given motor condition class."""
        return _CLASS_COLORS_BGR.get(class_name, (200, 200, 200))

    @staticmethod
    def get_motor_status(class_name: str) -> str:
        """Return 'HEALTHY' or 'FAULTY' for a condition class."""
        return _CONDITION_TO_STATUS.get(class_name, "FAULTY")

    @staticmethod
    def get_severity(class_name: str) -> float:
        """Severity weight 0‑1 for a condition class."""
        return _SEVERITY_WEIGHT.get(class_name, 0.5)

    # ------------------------------------------------------------------
    # Annotation drawing
    # ------------------------------------------------------------------

    def draw_annotations(
        self, frame: np.ndarray, detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Draw bounding boxes, class labels, confidence bars, and a
        HEALTHY/FAULTY banner on *frame* (in‑place copy).
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            conf = det["confidence"]
            cls_name = det["class_name"]
            color = det["color"]
            status = det["motor_status"]

            # Bounding box (thicker for severe faults)
            thickness = 3 if det["severity"] > 0.5 else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Label background + text
            label = f"{cls_name}  {conf:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale, font_thick = 0.55, 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                annotated, label, (x1 + 3, y1 - 5),
                font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA,
            )

            # Small status tag below box
            status_color = (0, 255, 0) if status == "HEALTHY" else (0, 0, 255)
            cv2.putText(
                annotated, status, (x1, y2 + 18),
                font, 0.5, status_color, 1, cv2.LINE_AA,
            )

        # ------- Top‑left overlay banner -------
        if detections:
            worst = max(detections, key=lambda d: d["severity"])
            overall_status = worst["motor_status"]
            banner_color = (0, 200, 0) if overall_status == "HEALTHY" else (0, 0, 220)
            cv2.rectangle(annotated, (0, 0), (260, 36), banner_color, -1)
            cv2.putText(
                annotated,
                f"Motor: {overall_status}",
                (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
            )

        return annotated

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run YOLO inference on a single BGR frame.

        Returns a dict containing:
            detections        — list of per‑object dicts
            annotated_frame   — base64‑encoded JPEG with overlays
            inference_time_ms — wall‑clock inference time
            frame_shape       — (H, W, C)
            motor_status      — overall 'HEALTHY' or 'FAULTY'
            confidence        — highest detection confidence
            trigger_rag       — bool: True when RAG pipeline should fire
            worst_condition   — name of the most severe condition found
            severity_score    — 0‑1 combined severity
        """
        if self.is_simulated:
            return SimulatedVisionObserver().detect(frame)

        start = time.time()
        try:
            preprocessed = self.preprocess_frame(frame)
            results = self.model(preprocessed, conf=self.confidence_threshold, verbose=False)

            detections: List[Dict[str, Any]] = []
            if results and len(results) > 0:
                result = results[0]
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    # Map YOLO class ID → motor condition.
                    # If the model was trained on our 6 classes the IDs map
                    # directly; otherwise we heuristically remap.
                    if result.names and cls_id in result.names:
                        raw_name = result.names[cls_id]
                    else:
                        raw_name = f"class_{cls_id}"
                    cls_name = self._map_to_motor_condition(raw_name)

                    detections.append({
                        "class_name":   cls_name,
                        "confidence":   conf,
                        "bbox":         [x1, y1, x2, y2],
                        "color":        self.get_fault_color(cls_name),
                        "motor_status": self.get_motor_status(cls_name),
                        "severity":     self.get_severity(cls_name),
                    })

            # ---- Build summary ----
            summary = self._build_summary(detections)

            # ---- Annotate frame ----
            annotated = self.draw_annotations(frame, detections)
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64_img = base64.b64encode(buf).decode("utf-8")

            elapsed_ms = (time.time() - start) * 1000
            logger.info(
                f"Detection: {len(detections)} objects, "
                f"status={summary['motor_status']}, "
                f"severity={summary['severity_score']:.2f}, "
                f"{elapsed_ms:.1f}ms"
            )

            # Store in history
            self.detection_history.append(summary)

            return {
                "detections":       detections,
                "annotated_frame":  b64_img,
                "inference_time_ms": elapsed_ms,
                "frame_shape":      list(frame.shape),
                **summary,
            }

        except Exception as exc:
            logger.error(f"YOLO inference failed: {exc}", exc_info=True)
            return self._empty_result(frame)

    # ------------------------------------------------------------------
    # Base64 convenience
    # ------------------------------------------------------------------

    def detect_from_base64(self, b64_string: str) -> Dict[str, Any]:
        """Decode a base64 image string → BGR frame → detect()."""
        try:
            if "," in b64_string:
                b64_string = b64_string.split(",", 1)[1]
            raw = base64.b64decode(b64_string)
            arr = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("cv2.imdecode returned None")
            return self.detect(frame)
        except Exception as exc:
            logger.error(f"detect_from_base64 failed: {exc}")
            return self._empty_result(None)

    # ------------------------------------------------------------------
    # Live webcam convenience
    # ------------------------------------------------------------------

    def detect_from_webcam(self) -> Dict[str, Any]:
        """Capture one frame from the webcam and run detection."""
        frame = self.read_frame()
        if frame is None:
            logger.warning("No frame from webcam — returning empty result.")
            return self._empty_result(None)
        return self.detect(frame)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_to_motor_condition(raw_name: str) -> str:
        """
        Map an arbitrary YOLO class name to one of the six motor conditions.
        If the model was trained on our custom dataset the names match
        directly.  Otherwise we apply keyword heuristics.
        """
        name_lower = raw_name.lower()

        # Direct match
        for cond in MOTOR_CONDITIONS:
            if cond.lower() == name_lower or cond.lower().replace(" ", "_") == name_lower:
                return cond

        # Keyword fallback
        if any(k in name_lower for k in ("healthy", "normal", "good", "clean")):
            return "Healthy Baseline"
        if any(k in name_lower for k in ("mild", "light", "slight")):
            return "Mild Oxidation"
        if any(k in name_lower for k in ("moderate", "medium")):
            return "Moderate Corrosion"
        if any(k in name_lower for k in ("severe", "heavy", "advanced")):
            return "Severe Corrosion"
        if any(k in name_lower for k in ("crack", "fracture", "break")):
            return "Structural Cracking"
        if any(k in name_lower for k in ("contam", "oil", "dust", "dirt", "foreign")):
            return "Contamination"
        if any(k in name_lower for k in ("rust", "oxid", "discolor")):
            return "Mild Oxidation"
        if any(k in name_lower for k in ("corrosion", "corrode")):
            return "Moderate Corrosion"

        # Default: treat unknown as contamination (caution‑first)
        logger.warning(f"Unmapped YOLO class '{raw_name}' → defaulting to Contamination")
        return "Contamination"

    @staticmethod
    def _build_summary(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute an overall motor status summary from the list of detections.
        """
        if not detections:
            return {
                "motor_status":    "HEALTHY",
                "confidence":      1.0,
                "trigger_rag":     False,
                "worst_condition": "Healthy Baseline",
                "severity_score":  0.0,
            }

        worst = max(detections, key=lambda d: d["severity"])
        max_severity = worst["severity"]
        overall_status = "FAULTY" if max_severity > 0.0 else "HEALTHY"

        # Confidence = the confidence of the most severe detection
        top_conf = worst["confidence"]

        # Trigger RAG pipeline whenever the motor is FAULTY
        trigger_rag = overall_status == "FAULTY"

        return {
            "motor_status":    overall_status,
            "confidence":      round(top_conf, 4),
            "trigger_rag":     trigger_rag,
            "worst_condition": worst["class_name"],
            "severity_score":  round(max_severity, 4),
        }

    @staticmethod
    def _empty_result(frame: Optional[np.ndarray]) -> Dict[str, Any]:
        """Return a safe empty result dict."""
        shape = list(frame.shape) if frame is not None else [0, 0, 0]
        return {
            "detections":       [],
            "annotated_frame":  "",
            "inference_time_ms": 0.0,
            "frame_shape":      shape,
            "motor_status":     "HEALTHY",
            "confidence":       0.0,
            "trigger_rag":      False,
            "worst_condition":  "Healthy Baseline",
            "severity_score":   0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════
# SimulatedVisionObserver  —  synthetic detections (no YOLO / no camera)
# ═══════════════════════════════════════════════════════════════════════════

class SimulatedVisionObserver:
    """
    Generates realistic‑looking synthetic detections for demo / simulation
    mode.  Produces the same output schema as ``VisionObserver.detect()``.

    The simulation cycles through fault scenarios so the dashboard always
    has something interesting to show.
    """

    # Probability distribution for simulation diversity
    _CONDITION_WEIGHTS = [0.30, 0.20, 0.18, 0.10, 0.07, 0.15]
    # Confidence ranges per condition (min, max)
    _CONF_RANGES: Dict[str, Tuple[float, float]] = {
        "Healthy Baseline":    (0.80, 0.99),
        "Mild Oxidation":      (0.55, 0.88),
        "Moderate Corrosion":  (0.50, 0.85),
        "Severe Corrosion":    (0.60, 0.92),
        "Structural Cracking": (0.55, 0.90),
        "Contamination":       (0.50, 0.82),
    }

    def __init__(self) -> None:
        self._call_count = 0

    # ------------------------------------------------------------------
    # Public interface (mirrors VisionObserver)
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Generate synthetic detections on *frame*."""
        start = time.time()
        self._call_count += 1

        # Simulate slight processing delay
        time.sleep(random.uniform(0.015, 0.060))

        h, w = frame.shape[:2]

        # 1‑3 detections per frame
        n = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
        detections: List[Dict[str, Any]] = []

        for _ in range(n):
            cls_name = random.choices(MOTOR_CONDITIONS, weights=self._CONDITION_WEIGHTS)[0]
            lo, hi = self._CONF_RANGES[cls_name]
            conf = round(random.uniform(lo, hi), 4)

            # Generate a plausible bounding box centred around the frame
            cx = random.randint(int(w * 0.2), int(w * 0.8))
            cy = random.randint(int(h * 0.2), int(h * 0.8))
            bw = random.randint(int(w * 0.15), int(w * 0.40))
            bh = random.randint(int(h * 0.15), int(h * 0.40))
            x1 = max(0, cx - bw // 2)
            y1 = max(0, cy - bh // 2)
            x2 = min(w, cx + bw // 2)
            y2 = min(h, cy + bh // 2)

            detections.append({
                "class_name":   cls_name,
                "confidence":   conf,
                "bbox":         [float(x1), float(y1), float(x2), float(y2)],
                "color":        VisionObserver.get_fault_color(cls_name),
                "motor_status": VisionObserver.get_motor_status(cls_name),
                "severity":     VisionObserver.get_severity(cls_name),
            })

        # ---- Summary ----
        summary = VisionObserver._build_summary(detections)

        # ---- Annotate ----
        annotated = self._draw_annotations(frame, detections, summary)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_img = base64.b64encode(buf).decode("utf-8")

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            f"[SIM] {len(detections)} detections, "
            f"status={summary['motor_status']}, "
            f"severity={summary['severity_score']:.2f}, "
            f"{elapsed_ms:.1f}ms"
        )

        return {
            "detections":        detections,
            "annotated_frame":   b64_img,
            "inference_time_ms": round(elapsed_ms, 2),
            "frame_shape":       list(frame.shape),
            **summary,
        }

    def detect_from_base64(self, b64_string: str) -> Dict[str, Any]:
        """Decode base64 → frame → detect()."""
        try:
            if "," in b64_string:
                b64_string = b64_string.split(",", 1)[1]
            raw = base64.b64decode(b64_string)
            arr = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Decode failed")
            return self.detect(frame)
        except Exception as exc:
            logger.error(f"[SIM] detect_from_base64 failed: {exc}")
            return VisionObserver._empty_result(None)

    # ------------------------------------------------------------------
    # Drawing (standalone — no inheritance needed)
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_annotations(
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> np.ndarray:
        """Draw boxes, labels, and status banner."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            color = det["color"]
            conf = det["confidence"]
            cls_name = det["class_name"]
            status = det["motor_status"]
            severity = det["severity"]

            thickness = 3 if severity > 0.5 else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            label = f"{cls_name}  {conf:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                annotated, label, (x1 + 3, y1 - 5),
                font, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )

            status_color = (0, 255, 0) if status == "HEALTHY" else (0, 0, 255)
            cv2.putText(
                annotated, status, (x1, y2 + 18),
                font, 0.5, status_color, 1, cv2.LINE_AA,
            )

        # Banner
        if detections:
            overall = summary.get("motor_status", "HEALTHY")
            banner_c = (0, 200, 0) if overall == "HEALTHY" else (0, 0, 220)
            cv2.rectangle(annotated, (0, 0), (280, 36), banner_c, -1)
            cv2.putText(
                annotated,
                f"Motor: {overall}",
                (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
            )

        # Simulation watermark
        cv2.putText(
            annotated,
            "SIMULATION",
            (w - 150, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA,
        )

        return annotated
