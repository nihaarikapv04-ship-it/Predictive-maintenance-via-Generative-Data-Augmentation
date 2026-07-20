"""
MotorGuard AI — Camera Component (Lag-Free)
=============================================
Single-frame capture per Streamlit refresh cycle.
Never runs a continuous loop. Opens → grabs 1 frame → closes.
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


def capture_one_frame(
    source: str = "webcam",
    url: str = "",
    resolution: str = "640x480",
) -> Optional[np.ndarray]:
    """
    Capture exactly ONE frame from the selected camera source,
    resize it, convert BGR→RGB, and release the camera immediately.

    Args:
        source: "webcam" | "ip_camera" | "external"
        url: IP Webcam URL (only used when source == "ip_camera")
        resolution: key from RESOLUTIONS dict

    Returns:
        RGB numpy array (H, W, 3) or None on failure.
    """
    w, h = RESOLUTIONS.get(resolution, (640, 480))

    # Pick device
    if source == "webcam":
        device = 0
    elif source == "ip_camera":
        if not url:
            logger.warning("IP camera selected but no URL provided.")
            return None
        device = url
    elif source == "external":
        device = 1  # second USB camera
    else:
        device = 0

    cap = None
    try:
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            logger.error(f"Cannot open camera: {device}")
            return None

        # Set resolution hints (camera may ignore)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning("Failed to read frame.")
            return None

        # Resize to target (ensures consistent size)
        frame = cv2.resize(frame, (w, h))
        # BGR → RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    except Exception as exc:
        logger.error(f"Camera capture error: {exc}")
        return None
    finally:
        if cap is not None:
            cap.release()


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
    # Work in BGR for OpenCV drawing, convert back
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = frame_bgr.shape[:2]

    severity_thick = 3 if confidence > 0.7 else 2
    mx, my = int(w * 0.08), int(h * 0.08)
    x1, y1, x2, y2 = mx, my, w - mx, h - my

    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color_bgr, severity_thick)

    label = f"{condition}  {confidence:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft = 0.6, 1
    (tw, th), _ = cv2.getTextSize(label, font, fs, ft)
    cv2.rectangle(frame_bgr, (x1, y1 - th - 14), (x1 + tw + 10, y1), color_bgr, -1)
    cv2.putText(frame_bgr, label, (x1 + 5, y1 - 6), font, fs, (255, 255, 255), ft, cv2.LINE_AA)

    status = "HEALTHY" if condition == "Healthy Baseline" else "FAULTY"
    sc = (0, 220, 0) if status == "HEALTHY" else (0, 0, 230)
    cv2.putText(frame_bgr, status, (x1, y2 + 24), font, 0.65, sc, 2, cv2.LINE_AA)

    # Banner
    ban = (0, 190, 0) if status == "HEALTHY" else (0, 0, 210)
    cv2.rectangle(frame_bgr, (0, 0), (300, 38), ban, -1)
    cv2.putText(frame_bgr, f"Motor: {status}", (10, 28), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # LIVE tag
    cv2.putText(frame_bgr, "LIVE", (w - 70, 28), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def generate_sim_motor_frame(
    condition: str = "Healthy Baseline",
    confidence: float = 0.85,
) -> np.ndarray:
    """
    Generate a synthetic 640×480 induction motor image with YOLO-style
    bounding box for simulation mode. Returns RGB numpy array.
    """
    from frontend.components.styles import FAULT_COLORS

    W, H = 640, 480
    frame = np.full((H, W, 3), (35, 30, 30), dtype=np.uint8)  # BGR dark bg

    # Motor body
    cv2.rectangle(frame, (120, 100), (520, 380), (80, 80, 90), -1)
    cv2.rectangle(frame, (120, 100), (520, 380), (110, 110, 120), 2)
    cv2.ellipse(frame, (320, 240), (100, 100), 0, 0, 360, (90, 90, 100), -1)
    cv2.ellipse(frame, (320, 240), (100, 100), 0, 0, 360, (130, 130, 140), 2)
    cv2.ellipse(frame, (320, 240), (40, 40), 0, 0, 360, (60, 60, 70), -1)
    # Shaft
    cv2.rectangle(frame, (520, 210), (600, 270), (100, 100, 110), -1)
    cv2.rectangle(frame, (520, 210), (600, 270), (140, 140, 150), 2)
    # Fins
    for y in range(120, 370, 30):
        cv2.line(frame, (125, y), (515, y), (70, 70, 78), 1)
    cv2.putText(frame, "INDUCTION MOTOR", (170, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 150), 1, cv2.LINE_AA)

    # Hex → BGR
    hex_c = FAULT_COLORS.get(condition, "#ffffff")
    r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
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
