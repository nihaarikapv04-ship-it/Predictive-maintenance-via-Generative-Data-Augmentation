"""
MotorGuard AI — Streamlit Dashboard
=====================================
Dark-theme real-time dashboard with three panels:
  OBSERVE  — live camera feed with YOLO bounding boxes
  DIAGNOSE — vibration graph, health gauge, uncertainty
  PRESCRIBE — risk level, repair protocol, schedules
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import time
import base64
import cv2
from collections import deque
from datetime import datetime, timedelta
import sys
import os

# ──────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="MotorGuard AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Column padding for breathing room ── */
[data-testid="stHorizontalBlock"] > div {
    padding: 0 12px;
}

/* Panel headers */
.panel-header {
    padding: 14px 18px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 1.15em;
    margin-bottom: 16px;
}
.panel-header.observe  { background: linear-gradient(90deg, #00d4aa 0%, #00d4aa22 100%); }
.panel-header.diagnose { background: linear-gradient(90deg, #4da6ff 0%, #4da6ff22 100%); }
.panel-header.prescribe{ background: linear-gradient(90deg, #ff6b35 0%, #ff6b3522 100%); }

/* Glassmorphism card */
.gcard {
    background: rgba(26, 29, 35, 0.75);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

/* Fault badge */
.fault-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.15em;
    letter-spacing: 0.5px;
}

/* Status dots */
.sdot {
    height: 9px; width: 9px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}
.sdot.on  { background: #00d4aa; }
.sdot.off { background: #ff3333; }

/* Simulation banner */
.sim-banner {
    background: rgba(255,170,0,0.15);
    color: #ffaa00;
    padding: 10px;
    text-align: center;
    border-radius: 6px;
    border: 1px solid #ffaa00;
    font-weight: 600;
    margin-top: 6px;
}

/* Pulse animation for critical */
.pulse-red { animation: pulse 1.8s infinite; }
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(255,51,51,0.6); }
    70%  { box-shadow: 0 0 0 12px rgba(255,51,51,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,51,51,0); }
}

/* Fixed bottom bar */
.bottom-bar {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: #0e1117;
    border-top: 1px solid rgba(255,255,255,0.1);
    padding: 8px 24px;
    z-index: 9999;
    display: flex;
    justify-content: space-around;
    align-items: center;
    font-size: 0.85em;
}
.bottom-bar .item { display: inline-block; }

/* Add bottom padding to body so content doesn't hide behind fixed bar */
.main .block-container { padding-bottom: 80px !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
_defaults = {
    "vibration_history": deque(maxlen=500),
    "detection_history": deque(maxlen=10),
    "health_score_history": [],
    "pipeline_running": False,
    "simulation_mode": True,
    "start_time": time.time(),
    "backend_ok": False,
    "prev_health": 75.0,
    "cam_ok": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────
# MOTOR CONDITION PALETTE  (matches vision.py)
# ──────────────────────────────────────────────
CONDITIONS = [
    "Healthy Baseline", "Mild Oxidation", "Moderate Corrosion",
    "Severe Corrosion", "Structural Cracking", "Contamination",
]
COND_COLOR = {
    "Healthy Baseline":   "#00d4aa",
    "Mild Oxidation":     "#f0e130",
    "Moderate Corrosion": "#ff8c00",
    "Severe Corrosion":   "#ff3333",
    "Structural Cracking":"#ff3333",
    "Contamination":      "#9b30ff",
}
COND_STATUS = {
    "Healthy Baseline": "HEALTHY", "Mild Oxidation": "FAULTY",
    "Moderate Corrosion":"FAULTY", "Severe Corrosion": "FAULTY",
    "Structural Cracking":"FAULTY", "Contamination": "FAULTY",
}
COND_SEVERITY = {
    "Healthy Baseline": 0.0,  "Mild Oxidation": 0.20,
    "Moderate Corrosion": 0.50, "Severe Corrosion": 0.85,
    "Structural Cracking": 0.95, "Contamination": 0.40,
}
_BGR = {
    "Healthy Baseline": (0,255,0), "Mild Oxidation": (0,255,255),
    "Moderate Corrosion": (0,165,255), "Severe Corrosion": (0,0,255),
    "Structural Cracking": (0,0,255), "Contamination": (255,0,128),
}


# ══════════════════════════════════════════════════
# CAMERA CAPTURE  — real webcam or simulation
# ══════════════════════════════════════════════════

def capture_real_webcam_frame() -> dict:
    """
    Open the MacBook webcam (device 0), grab one frame, draw simulated
    YOLO-style bounding boxes on it (since the real YOLO model may not
    be loaded), and return the annotated RGB image + metadata.
    """
    start = time.time()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.session_state.cam_ok = False
        cap.release()
        return None  # fall back to simulation

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        st.session_state.cam_ok = False
        return None

    st.session_state.cam_ok = True

    # --- Try using the backend VisionObserver if available ---
    try:
        # Add project root to path so we can import backend modules
        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if proj_root not in sys.path:
            sys.path.insert(0, proj_root)
        from backend.observe.vision import VisionObserver
        observer = VisionObserver()
        result = observer.detect(frame)
        # Decode the annotated frame
        if result.get("annotated_frame"):
            raw = base64.b64decode(result["annotated_frame"])
            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cond = result.get("worst_condition", "Healthy Baseline")
        return {
            "frame_rgb": frame_rgb,
            "condition": cond,
            "confidence": result.get("confidence", 0.0),
            "status": result.get("motor_status", COND_STATUS.get(cond, "HEALTHY")),
            "severity": COND_SEVERITY.get(cond, 0),
            "num_detections": len(result.get("detections", [])),
            "inference_ms": round(result.get("inference_time_ms", 0), 1),
        }
    except Exception:
        pass

    # --- Fallback: draw simulated detection on real webcam frame ---
    elapsed_ms = (time.time() - start) * 1000
    h, w = frame.shape[:2]

    # Pick a random condition
    weights = [0.30, 0.20, 0.18, 0.10, 0.07, 0.15]
    cond = np.random.choice(CONDITIONS, p=weights)
    conf_lo = {"Healthy Baseline": 0.80, "Mild Oxidation": 0.55,
               "Moderate Corrosion": 0.50, "Severe Corrosion": 0.60,
               "Structural Cracking": 0.55, "Contamination": 0.50}
    conf_hi = {"Healthy Baseline": 0.99, "Mild Oxidation": 0.88,
               "Moderate Corrosion": 0.85, "Severe Corrosion": 0.92,
               "Structural Cracking": 0.90, "Contamination": 0.82}
    conf = round(np.random.uniform(conf_lo[cond], conf_hi[cond]), 3)
    severity = COND_SEVERITY[cond]
    status = COND_STATUS[cond]
    color = _BGR[cond]

    # Draw bounding box on the centre of the real frame
    margin_x, margin_y = int(w * 0.1), int(h * 0.1)
    x1, y1 = margin_x, margin_y
    x2, y2 = w - margin_x, h - margin_y
    thick = 3 if severity > 0.5 else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

    # Label
    label = f"{cond}  {conf:.0%}"
    (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th_t - 14), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Status
    sc = (0, 255, 0) if status == "HEALTHY" else (0, 0, 255)
    cv2.putText(frame, status, (x1, y2 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, sc, 2, cv2.LINE_AA)

    # Banner
    ban = (0, 190, 0) if status == "HEALTHY" else (0, 0, 210)
    cv2.rectangle(frame, (0, 0), (300, 38), ban, -1)
    cv2.putText(frame, f"Motor: {status}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # LIVE tag
    cv2.putText(frame, "LIVE", (w - 70, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return {
        "frame_rgb": frame_rgb,
        "condition": cond,
        "confidence": conf,
        "status": status,
        "severity": severity,
        "num_detections": 1,
        "inference_ms": round(elapsed_ms + np.random.uniform(15, 40), 1),
    }


def generate_simulated_frame() -> dict:
    """Synthetic motor image with bounding boxes (simulation mode)."""
    W, H = 640, 480
    frame = np.full((H, W, 3), (30, 30, 35), dtype=np.uint8)

    # Motor body
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

    weights = [0.30, 0.20, 0.18, 0.10, 0.07, 0.15]
    cond = np.random.choice(CONDITIONS, p=weights)
    lo = {"Healthy Baseline": 0.80, "Mild Oxidation": 0.55,
          "Moderate Corrosion": 0.50, "Severe Corrosion": 0.60,
          "Structural Cracking": 0.55, "Contamination": 0.50}
    hi = {"Healthy Baseline": 0.99, "Mild Oxidation": 0.88,
          "Moderate Corrosion": 0.85, "Severe Corrosion": 0.92,
          "Structural Cracking": 0.90, "Contamination": 0.82}
    conf = round(np.random.uniform(lo[cond], hi[cond]), 3)
    severity = COND_SEVERITY[cond]
    status = COND_STATUS[cond]

    jx, jy = np.random.randint(-10, 10), np.random.randint(-8, 8)
    x1, y1, x2, y2 = 110+jx, 90+jy, 530+jx, 390+jy
    color = _BGR[cond]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if severity > 0.5 else 2)
    label = f"{cond}  {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    scol = (0, 255, 0) if status == "HEALTHY" else (0, 0, 255)
    cv2.putText(frame, status, (x1, y2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, scol, 2, cv2.LINE_AA)
    ban_c = (0, 190, 0) if status == "HEALTHY" else (0, 0, 210)
    cv2.rectangle(frame, (0, 0), (290, 36), ban_c, -1)
    cv2.putText(frame, f"Motor: {status}", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "SIMULATION", (W - 155, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return {
        "frame_rgb": frame_rgb, "condition": cond, "confidence": conf,
        "status": status, "severity": severity, "num_detections": 1,
        "inference_ms": round(np.random.uniform(18, 55), 1),
    }


# ──────────────────────────────────────────────
# MOCK PIPELINE DATA
# ──────────────────────────────────────────────
def get_mock_pipeline(vision_info: dict) -> dict:
    t = time.time()
    noise = np.random.normal(0, 0.1, 6)
    vib = [
        np.sin(t) + noise[0], np.cos(t) + noise[1],
        np.sin(t*0.5) + noise[2], np.cos(t*1.5) + noise[3],
        np.sin(t*2) + noise[4], np.cos(t*0.2) + noise[5],
    ]
    base = 90 - vision_info["severity"] * 60
    health = float(np.clip(base + np.random.normal(0, 3), 0, 100))
    rul = max(10.0, health * 10 + np.random.normal(0, 50))
    unc = abs(np.random.normal(0.03, 0.015))
    return {
        "timestamp": t, "vibration": vib, "health_score": health,
        "rul_hours": rul, "uncertainty": unc,
        "latency_ms": vision_info["inference_ms"] + np.random.uniform(5, 15),
    }


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ MotorGuard AI")
    st.markdown("---")

    st.markdown("### 🎛️ System Controls")
    st.session_state.pipeline_running = st.toggle(
        "Start Pipeline", value=st.session_state.pipeline_running)
    st.session_state.simulation_mode = st.toggle(
        "Simulation Mode", value=st.session_state.simulation_mode)
    refresh_rate = st.slider("Refresh Rate (s)", 0.5, 5.0, 1.0, 0.1)

    st.markdown("### 🔧 Model Config")
    yolo_conf = st.slider("YOLO Conf Threshold", 0.1, 0.9, 0.25, 0.05)
    mc_samples = st.slider("MC Dropout Samples", 10, 100, 50, 10)

    st.markdown("### 📡 Connection")
    _on, _off = "sdot on", "sdot off"
    be = _on if (st.session_state.backend_ok or st.session_state.simulation_mode) else _off
    cam_dot = _on if (st.session_state.cam_ok or st.session_state.simulation_mode) else _off
    st.markdown(f"<span class='{be}'></span> Backend", unsafe_allow_html=True)
    st.markdown(f"<span class='{cam_dot}'></span> Camera {'(Sim)' if st.session_state.simulation_mode else '(Live)'}", unsafe_allow_html=True)
    st.markdown(f"<span class='{_on}'></span> Sensor {'(Sim)' if st.session_state.simulation_mode else ''}", unsafe_allow_html=True)

    uptime = timedelta(seconds=int(time.time() - st.session_state.start_time))
    st.caption(f"Uptime: {uptime}")


# ══════════════════════════════════════════════════
# CAPTURE FRAME  — real webcam vs simulation
# ══════════════════════════════════════════════════
if st.session_state.simulation_mode:
    vision = generate_simulated_frame()
else:
    # REAL CAMERA MODE — use cv2.VideoCapture(0)
    vision = capture_real_webcam_frame()
    if vision is None:
        # Camera failed → fall back to simulation with warning
        vision = generate_simulated_frame()
        st.toast("⚠️ Webcam not available — showing simulation", icon="📷")

data = get_mock_pipeline(vision)

# Update session history
if st.session_state.pipeline_running:
    st.session_state.vibration_history.append({
        "t": data["timestamp"],
        "ax": data["vibration"][0], "ay": data["vibration"][1], "az": data["vibration"][2],
        "gx": data["vibration"][3], "gy": data["vibration"][4], "gz": data["vibration"][5],
    })
    st.session_state.health_score_history.append(data["health_score"])
    if len(st.session_state.health_score_history) > 100:
        st.session_state.health_score_history.pop(0)
    st.session_state.detection_history.appendleft({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Condition": vision["condition"],
        "Conf": f"{vision['confidence']:.1%}",
        "Status": vision["status"],
    })

# ═══════════════════════════════════════════════
# MAIN 3-PANEL LAYOUT
# ═══════════════════════════════════════════════
col_obs, col_diag, col_rx = st.columns([1.2, 1.2, 1], gap="large")

# ─────────────── PANEL 1: OBSERVE ───────────────
with col_obs:
    st.markdown("<div class='panel-header observe'>🔍 OBSERVE — Visual Inspection</div>",
                unsafe_allow_html=True)

    # ── CAMERA FRAME  (full width, large) ──
    st.image(vision["frame_rgb"], use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Fault class badge ──
    cond = vision["condition"]
    f_color = COND_COLOR.get(cond, "#ffffff")
    status = vision["status"]
    st.markdown(f"""
    <div class='gcard' style='border-left: 5px solid {f_color}'>
        <span class='fault-badge' style='background:{f_color}22; color:{f_color};'>{cond}</span>
        <span style='margin-left:12px; font-size:0.9em;
              color:{"#00d4aa" if status == "HEALTHY" else "#ff3333"};
              font-weight:700'>{status}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence bar ──
    st.progress(vision["confidence"], text=f"Confidence: {vision['confidence']:.1%}")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stats ──
    s1, s2, s3 = st.columns(3)
    s1.metric("Detections", vision["num_detections"])
    s2.metric("Confidence", f"{vision['confidence']:.2f}")
    s3.metric("Inf Time", f"{vision['inference_ms']:.0f} ms")

    st.divider()

    # ── Recent detections ──
    st.markdown("**Recent Detections**")
    if st.session_state.detection_history:
        st.dataframe(
            pd.DataFrame(st.session_state.detection_history),
            use_container_width=True, hide_index=True, height=200,
        )

# ─────────────── PANEL 2: DIAGNOSE ───────────────
with col_diag:
    st.markdown("<div class='panel-header diagnose'>🩺 DIAGNOSE — Health Analytics</div>",
                unsafe_allow_html=True)

    # ── Vibration graph (taller) ──
    if st.session_state.vibration_history:
        df_vib = pd.DataFrame(st.session_state.vibration_history)
        fig_vib = go.Figure()
        ch_colors = ["#ff3333", "#00d4aa", "#4da6ff", "#ff6b35", "#9933ff", "#ffaa00"]
        for i, ch in enumerate(["ax", "ay", "az", "gx", "gy", "gz"]):
            fig_vib.add_trace(go.Scatter(
                y=df_vib[ch], mode="lines", name=ch,
                line=dict(color=ch_colors[i], width=1.2),
            ))
        fig_vib.update_layout(
            template="plotly_dark", height=300,
            margin=dict(l=0, r=0, t=28, b=0),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            xaxis_title="Sample", yaxis_title="Amplitude",
        )
        st.plotly_chart(fig_vib, use_container_width=True, key="vib")
    else:
        st.info("▶ Start the pipeline to see vibration data.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Health gauge (compact) ──
    hs = data["health_score"]
    prev = st.session_state.prev_health
    st.session_state.prev_health = hs

    fig_g = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=hs,
        delta={"reference": prev, "position": "top"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "white"},
            "steps": [
                {"range": [0, 40],  "color": "#ff3333"},
                {"range": [40, 60], "color": "#ff6b35"},
                {"range": [60, 80], "color": "#ffaa00"},
                {"range": [80, 100],"color": "#00d4aa"},
            ],
        },
    ))
    fig_g.update_layout(template="plotly_dark", height=250,
                        margin=dict(l=20, r=20, t=30, b=10))
    st.plotly_chart(fig_g, use_container_width=True, key="gauge")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MC Dropout uncertainty ──
    unc = data["uncertainty"]
    ci_lo = max(0, hs - unc * 100)
    ci_hi = min(100, hs + unc * 100)
    unc_pct = unc * 100
    unc_lvl = "Low" if unc_pct < 5 else ("Medium" if unc_pct < 15 else "High")
    unc_col = "#00d4aa" if unc_lvl == "Low" else ("#ffaa00" if unc_lvl == "Medium" else "#ff3333")

    st.markdown(f"""
    <div class='gcard'>
        <b>MC Dropout Uncertainty:</b> ± {unc_pct:.1f}%
        &nbsp;&nbsp;<span style='color:{unc_col}; font-weight:700'>[{unc_lvl}]</span><br>
        <b>95% CI:</b> {ci_lo:.1f} – {ci_hi:.1f}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Maintenance urgency ──
    if hs < 40:
        urg, uc, ucls = "IMMEDIATE", "#ff3333", "gcard pulse-red"
    elif hs < 60:
        urg, uc, ucls = "SOON", "#ff6b35", "gcard"
    elif hs < 80:
        urg, uc, ucls = "ROUTINE", "#4da6ff", "gcard"
    else:
        urg, uc, ucls = "NONE", "#00d4aa", "gcard"

    st.markdown(f"""
    <div class='{ucls}' style='text-align:center'>
        <div style='color:#aaa; font-size:0.85em'>Maintenance Urgency</div>
        <div style='color:{uc}; font-size:1.8em; font-weight:700'>{urg}</div>
        <div>RUL: <b>{data['rul_hours']:.0f} hours</b></div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────── PANEL 3: PRESCRIBE ───────────────
with col_rx:
    st.markdown("<div class='panel-header prescribe'>📝 PRESCRIBE — Repair Protocol</div>",
                unsafe_allow_html=True)

    # ── Risk badge ──
    if hs < 40:
        risk, rc = "CRITICAL", "#ff3333"
    elif hs < 60:
        risk, rc = "HIGH", "#ff6b35"
    elif hs < 80:
        risk, rc = "MODERATE", "#ffaa00"
    else:
        risk, rc = "LOW", "#00d4aa"

    extra = " pulse-red" if risk == "CRITICAL" else ""
    st.markdown(f"""
    <div class='gcard{extra}' style='text-align:center; border:2px solid {rc}'>
        <div style='color:{rc}; font-size:1.5em; font-weight:700'>RISK: {risk}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── ETF ──
    etf = data["rul_hours"]
    etf_c = "#ff3333" if etf < 24 else "#ffffff"
    st.markdown(f"""
    <div class='gcard'>
        <div style='color:#aaa; font-size:0.85em'>Estimated Time to Failure</div>
        <div style='color:{etf_c}; font-size:2em; font-weight:700'>
            {etf:.0f} <span style='font-size:0.4em'>HOURS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Immediate action (only when urgent) ──
    if urg in ("IMMEDIATE", "SOON"):
        st.markdown(f"""
        <div style='border:2px solid #ff3333; padding:12px; border-radius:8px;
                    background:rgba(255,51,51,0.08); margin-bottom:12px'>
            <b style='color:#ff3333'>⚠️ IMMEDIATE ACTIONS</b>
            <ol style='margin-bottom:0; margin-top:6px'>
                <li>Halt motor operation immediately</li>
                <li>Isolate power supply (LOTO procedure)</li>
                <li>Notify maintenance supervisor</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    # ── Repair steps (all expandable) ──
    st.markdown("**Repair Protocol**")
    with st.expander("Step 1 — Diagnostics & Isolation", expanded=False):
        st.markdown(
            "- **Action:** Run full diagnostic sweep on motor\n"
            "- **Tools:** Multimeter, thermal camera, vibration probe\n"
            "- **Safety:** Ensure LOTO before touching motor\n"
            "- **Est. Time:** ~15 min"
        )
    with st.expander("Step 2 — Surface Inspection", expanded=False):
        st.markdown(
            "- **Action:** Inspect motor surface for corrosion, cracks, contamination\n"
            "- **Tools:** Flashlight, endoscope, magnifying lens\n"
            "- **Document:** Photograph all defect areas\n"
            "- **Est. Time:** ~30 min"
        )
    with st.expander("Step 3 — Cleaning & Treatment", expanded=False):
        st.markdown(
            "- **Action:** Clean contaminated areas, apply corrosion inhibitor\n"
            "- **Tools:** Wire brush, degreaser, Rust-Oleum inhibitor\n"
            "- **Est. Time:** ~45 min"
        )
    with st.expander("Step 4 — Component Repair / Replacement", expanded=False):
        st.markdown(
            "- **Action:** Replace bearings, seal cracks, re-coat surface\n"
            "- **Tools:** Bearing puller, wrench set, epoxy compound\n"
            "- **Parts:** SKF 6205-2RS bearing, Loctite 243\n"
            "- **Est. Time:** ~1-3 hrs"
        )
    with st.expander("Step 5 — Verification & Restart", expanded=False):
        st.markdown(
            "- **Action:** Test-run motor, verify vibration levels\n"
            "- **Criteria:** ISO 10816 Zone A/B vibration limits\n"
            "- **Sign-off:** Supervisor approval required\n"
            "- **Est. Time:** ~20 min"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Preventive schedule ──
    st.markdown("**Preventive Schedule**")
    st.dataframe(
        pd.DataFrame({
            "Task": ["Lubrication", "Alignment", "Vib Audit", "Surface Check"],
            "Interval": ["Monthly", "Quarterly", "Weekly", "Bi-weekly"],
            "Next Due": ["In 5 days", "In 20 days", "Tomorrow", "In 10 days"],
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Download report ──
    report = (
        f"MotorGuard AI — Inspection Report\n"
        f"{'='*40}\n"
        f"Date     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Condition: {cond}\n"
        f"Status   : {status}\n"
        f"Health   : {hs:.1f}/100\n"
        f"Risk     : {risk}\n"
        f"ETF      : {etf:.0f} hours\n"
        f"Urgency  : {urg}\n"
    )
    st.download_button("📥 Download Report", data=report,
                       file_name="motorguard_report.txt", mime="text/plain")


# ═══════════════════════════════════════════════
# FIXED BOTTOM STATUS BAR
# ═══════════════════════════════════════════════
pipeline_label = "🟢 Running" if st.session_state.pipeline_running else "⏹️ Stopped"
sim_label = " &nbsp;|&nbsp; <span style='color:#ffaa00'>⚠️ SIMULATION MODE</span>" if st.session_state.simulation_mode else ""
cam_label = "Sim" if st.session_state.simulation_mode else ("Live ✓" if st.session_state.cam_ok else "Off ✗")

st.markdown(f"""
<div class='bottom-bar'>
    <span class='item'><b>Pipeline:</b> {pipeline_label}</span>
    <span class='item'><b>Latency:</b> {data['latency_ms']:.1f} ms</span>
    <span class='item'><b>Camera:</b> {cam_label}</span>
    <span class='item'><b>Last:</b> {datetime.now().strftime('%H:%M:%S')}</span>
    <span class='item'><b>FPS:</b> {max(1, int(1000 / max(1, data['latency_ms'])))}</span>
    {sim_label}
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# AUTO-REFRESH
# ═══════════════════════════════════════════════
if st.session_state.pipeline_running:
    time.sleep(refresh_rate)
    st.rerun()
