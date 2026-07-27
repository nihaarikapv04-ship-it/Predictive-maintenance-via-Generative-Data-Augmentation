"""
MotorGuard AI — Camera Component (Fast & Optimized for Streamlit)
"""
import cv2
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Resolution presets
RESOLUTIONS = {
    "320x240":  (320, 240),
    "640x480":  (640, 480),
    "1280x720": (1280, 720),
}


def capture_one_frame(source="webcam", url="", resolution="320x240") -> Optional[np.ndarray]:
    """Fast single frame capture - optimized for Streamlit."""
    w, h = RESOLUTIONS.get(resolution, (320, 240))
    
    if source == "webcam":
        device = 0
    elif source == "ip_camera":
        device = url if url else None
        if device is None:
            return None
    else:
        device = 0
    
    cap = None
    try:
        # Use AVFoundation backend on Mac for speed
        cap = cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)
        
        # Force lowest acceptable resolution for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
        
        if not cap.isOpened():
            return None
        
        # Flush buffer - read and discard 2 frames
        for _ in range(2):
            cap.grab()
        
        ret, frame = cap.retrieve()
        if not ret or frame is None or frame.size == 0:
            cap.release()
            return None
        
        # Resize to target resolution
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        return frame_rgb
        
    except Exception as e:
        if cap:
            cap.release()
        return None


def draw_detection_overlay(
    frame_rgb: np.ndarray,
    condition: str,
    confidence: float,
    color_bgr: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Draw a YOLO-style bounding box + label on an RGB frame.
    Returns annotated RGB frame (copy).
    """
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = frame_bgr.shape[:2]

    severity_thick = 3 if confidence > 0.7 else 2
    mx, my = int(w * 0.08), int(h * 0.08)
    x1, y1, x2, y2 = mx, my, w - mx, h - my

    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color_bgr, severity_thick)

    label = f"{condition}  {confidence:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft = 0.5, 1
    (tw, th), _ = cv2.getTextSize(label, font, fs, ft)
    cv2.rectangle(frame_bgr, (x1, y1 - th - 10), (x1 + tw + 8, y1), color_bgr, -1)
    cv2.putText(frame_bgr, label, (x1 + 4, y1 - 4), font, fs, (255, 255, 255), ft, cv2.LINE_AA)

    status = "HEALTHY" if condition == "Healthy Baseline" else "FAULTY"
    sc = (0, 220, 0) if status == "HEALTHY" else (0, 0, 230)
    cv2.putText(frame_bgr, status, (x1, y2 + 20), font, 0.55, sc, 2, cv2.LINE_AA)

    ban = (0, 190, 0) if status == "HEALTHY" else (0, 0, 210)
    cv2.rectangle(frame_bgr, (0, 0), (240, 32), ban, -1)
    cv2.putText(frame_bgr, f"Motor: {status}", (8, 22), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame_bgr, "LIVE", (w - 60, 22), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def generate_sim_motor_frame(
    condition: str = "Healthy Baseline",
    confidence: float = 0.85,
) -> np.ndarray:
    """
    Generate a synthetic 640x480 induction motor image with YOLO-style
    bounding box for simulation mode. Returns RGB numpy array.
    """
    from frontend.components.styles import FAULT_COLORS

    W, H = 640, 480
    frame = np.full((H, W, 3), (35, 30, 30), dtype=np.uint8)

    cv2.rectangle(frame, (120, 100), (520, 380), (80, 80, 90), -1)
    cv2.rectangle(frame, (120, 100), (520, 380), (110, 110, 120), 2)
    cv2.ellipse(frame, (320, 240), (100, 100), 0, 0, 360, (90, 90, 100), -1)
    cv2.ellipse(frame, (320, 240), (100, 100), 0, 0, 360, (130, 130, 140), 2)
    cv2.ellipse(frame, (320, 240), (40, 40), 0, 0, 360, (60, 60, 70), -1)
    cv2.rectangle(frame, (520, 210), (600, 270), (100, 100, 110), -1)
    cv2.rectangle(frame, (520, 210), (600, 270), (140, 140, 150), 2)
    for y in range(120, 370, 30):
        cv2.line(frame, (125, y), (515, y), (70, 70, 78), 1)
    cv2.putText(frame, "INDUCTION MOTOR", (170, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 150), 1, cv2.LINE_AA)

    hex_c = FAULT_COLORS.get(condition, "#ffffff")
    if hex_c.startswith("#"):
        r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
    else:
        r, g, b = 0, 255, 136
    color_bgr = (b, g, r)

    severity = {"Healthy Baseline": 0.0, "Mild Oxidation": 0.2,
                "Moderate Corrosion": 0.5, "Severe Corrosion": 0.85,
                "Structural Cracking": 0.95, "Contamination": 0.4}.get(condition, 0.3)

    jx, jy = np.random.randint(-8, 8), np.random.randint(-6, 6)
    x1, y1 = 110 + jx, 90 + jy
    x2, y2 = 530 + jx, 390 + jy
    thick = 3 if severity > 0.5 else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, thick)

    label = f"{condition}  {confidence:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color_bgr, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 5), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    status = "HEALTHY" if condition == "Healthy Baseline" else "FAULTY"
    scol = (0, 255, 0) if status == "HEALTHY" else (0, 0, 255)
    cv2.putText(frame, status, (x1, y2 + 22), font, 0.6, scol, 2, cv2.LINE_AA)

    ban = (0, 190, 0) if status == "HEALTHY" else (0, 0, 210)
    cv2.rectangle(frame, (0, 0), (290, 36), ban, -1)
    cv2.putText(frame, f"Motor: {status}", (8, 26), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "SIMULATION", (W - 155, H - 12), font, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
