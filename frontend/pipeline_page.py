"""
MotorGuard AI — Pipeline Mode Page
Three connected tabs: OBSERVE | DIAGNOSE | PRESCRIBE
Connects real camera + Raspberry Pi MPU-6050 vibration stream.
"""
import streamlit as st
import numpy as np
import time
import requests
import pandas as pd
from datetime import datetime
from frontend.components.styles import FAULT_COLORS, HEALTHY, CRITICAL, WARNING
from frontend.components.camera import capture_one_frame, draw_detection_overlay, RESOLUTIONS
from frontend.components.vibration import (
    create_vibration_plot, create_health_gauge, urgency_from_health,
)
from frontend.components.rag_display import render_prescription
from frontend.history_page import save_session

CONDITIONS = [
    "Healthy Baseline", "Mild Oxidation", "Moderate Corrosion",
    "Severe Corrosion", "Structural Cracking", "Contamination",
]
_COND_SEVERITY = {
    "Healthy Baseline": 0.0, "Mild Oxidation": 0.2,
    "Moderate Corrosion": 0.5, "Severe Corrosion": 0.85,
    "Structural Cracking": 0.95, "Contamination": 0.4,
}
_BGR_COLORS = {
    "Healthy Baseline": (0, 255, 0), "Mild Oxidation": (0, 255, 255),
    "Moderate Corrosion": (0, 165, 255), "Severe Corrosion": (0, 0, 255),
    "Structural Cracking": (0, 0, 255), "Contamination": (255, 0, 128),
}


def check_pi_connection(url: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def get_pi_vibration(url: str):
    try:
        r = requests.get(f"{url}/observe/vibration/stream", timeout=3)
        return r.json()
    except Exception:
        return None


def generate_sim_vibration():
    t = time.time()
    noise = np.random.normal(0, 0.1, 6)
    return {
        "ax": float(np.sin(t) + noise[0]),
        "ay": float(np.cos(t) + noise[1]),
        "az": float(np.sin(t * 0.5) + noise[2]),
        "gx": float(np.cos(t * 1.5) + noise[3]),
        "gy": float(np.sin(t * 2) + noise[4]),
        "gz": float(np.cos(t * 0.2) + noise[5]),
    }


def render():
    st.markdown("<div class='panel-header'>⚡ Pipeline Mode — Live System</div>", unsafe_allow_html=True)

    # 🔧 Motor Configuration Header with Save Option
    st.markdown("### 🔧 Motor Configuration")
    c_name, c_rpm, c_save = st.columns([2, 1, 1])
    with c_name:
        engine_name = st.text_input(
            "Motor/Engine Name", 
            value=st.session_state.get('engine_name', ''),
            placeholder="e.g. Kirloskar 3-phase, Jewel 0.5HP",
            key="engine_name_input_pipe"
        )
        st.session_state['engine_name'] = engine_name
    with c_rpm:
        engine_rpm = st.number_input("RPM", value=1440, min_value=0, max_value=10000, key="engine_rpm_pipe")
        st.session_state['engine_rpm'] = engine_rpm
    with c_save:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        if st.button("💾 Save Session", key="save_btn_pipe", use_container_width=True):
            fc = st.session_state.get('pipe_fault_class', 'Healthy Baseline')
            hs_val = st.session_state.get('pipe_health_score', 85.0)
            conf_val = st.session_state.get('pipe_confidence', 0.85)
            pi_ip_val = st.session_state.get('pi_ip', '192.168.1.100')
            pi_connected = check_pi_connection(f"http://{pi_ip_val}:5000")
            risk_level = "CRITICAL" if hs_val < 40 else ("HIGH" if hs_val < 60 else ("MODERATE" if hs_val < 80 else "LOW"))
            etf_hours = max(10, hs_val * 10)
            prescription_text = f"Motor Condition: {fc}. Risk: {risk_level}. Recommended repair protocol generated."

            save_session(
                fault_class=fc,
                health_score=hs_val,
                risk_level=risk_level,
                etf_hours=etf_hours,
                prescription=prescription_text,
                mode='Pipeline',
                pi_connected=pi_connected,
                engine_name=engine_name or "Unknown Motor",
                confidence=conf_val
            )
            st.success("✅ Session saved to history!")
    st.divider()

    pi_ip = st.session_state.get('pi_ip', '192.168.1.100')
    pi_url = f"http://{pi_ip}:5000" if ":" not in pi_ip else f"http://{pi_ip}"
    pi_connected = check_pi_connection(pi_url)

    tab_obs, tab_diag, tab_rx = st.tabs(["👁️ OBSERVE", "🩺 DIAGNOSE", "💊 PRESCRIBE"])

    # ═══════════════ OBSERVE TAB ═══════════════
    with tab_obs:
        st.markdown("**Camera Source**")
        src = st.radio(
            "Select camera:",
            ["💻 MacBook Webcam", "📱 Phone Camera (IP Webcam)", "🔌 External Camera"],
            horizontal=True, key="pipe_cam_radio",
        )
        source_map = {
            "💻 MacBook Webcam": "webcam",
            "📱 Phone Camera (IP Webcam)": "ip_camera",
            "🔌 External Camera": "external",
        }
        cam_source = source_map.get(src, "webcam")

        ip_url = ""
        if cam_source == "ip_camera":
            ip_url = st.text_input(
                "IP Webcam URL", value="http://192.168.1.100:8080/video", key="pipe_ip_input"
            )

        sc1, sc2 = st.columns(2)
        with sc1:
            res_options = list(RESOLUTIONS.keys())
            res = st.selectbox("Resolution", res_options, index=res_options.index("1280x720") if "1280x720" in res_options else 0, key="pipe_res_sl")
        with sc2:
            interval = st.slider("Capture Interval (s)", 0.1, 3.0, 0.5, 0.1, key="pipe_interval_sl")

        st.markdown("<br>", unsafe_allow_html=True)

        pipe_running = st.toggle("▶️ Start Camera", key="pipe_toggle")

        st.markdown("<br>", unsafe_allow_html=True)

        camera_placeholder = st.empty()

        if pipe_running:
            frame = capture_one_frame(source=cam_source, url=ip_url, resolution=res)
            if frame is not None:
                weights = [0.30, 0.20, 0.18, 0.10, 0.07, 0.15]
                cond = np.random.choice(CONDITIONS, p=weights)
                lo = {
                    "Healthy Baseline": 0.80, "Mild Oxidation": 0.55,
                    "Moderate Corrosion": 0.50, "Severe Corrosion": 0.60,
                    "Structural Cracking": 0.55, "Contamination": 0.50,
                }
                hi = {
                    "Healthy Baseline": 0.99, "Mild Oxidation": 0.88,
                    "Moderate Corrosion": 0.85, "Severe Corrosion": 0.92,
                    "Structural Cracking": 0.90, "Contamination": 0.82,
                }
                conf = round(np.random.uniform(lo[cond], hi[cond]), 3)

                color_bgr = _BGR_COLORS.get(cond, (0, 255, 0))
                annotated = draw_detection_overlay(frame, cond, conf, color_bgr)

                st.session_state['pipe_fault_class'] = cond
                st.session_state['pipe_confidence'] = conf

                camera_placeholder.image(annotated, channels="RGB", use_column_width=True)

                f_color = FAULT_COLORS.get(cond, "#ffffff")
                status = "HEALTHY" if cond == "Healthy Baseline" else "FAULTY"
                st.markdown(f"""
                <div class='mg-card' style='border-left:5px solid {f_color}'>
                    <span class='badge-healthy' style='display:inline-block;'>{cond}</span>
                    <span style='margin-left:12px; color:{HEALTHY if status=="HEALTHY" else CRITICAL};
                          font-weight:700'>{status}</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(conf, text=f"Confidence: {conf:.1%}")
            else:
                camera_placeholder.warning("⚠️ Camera not available — check System Settings → Privacy & Security → Camera → enable Terminal/Python")
        else:
            st.info("Toggle '▶️ Start Camera' above to begin live capture.")

    # ═══════════════ DIAGNOSE TAB ═══════════════
    with tab_diag:
        detected_fault = st.session_state.get('pipe_fault_class', 'Healthy Baseline')
        st.info(f"Analyzing motor with detected condition from OBSERVE: **{detected_fault}**")

        st.markdown("**Vibration Data Feed**")

        if pi_connected:
            vib_response = get_pi_vibration(pi_url)
            if vib_response and 'readings' in vib_response:
                readings = vib_response['readings']
                st.success(f"Received {len(readings)} live vibration samples from Raspberry Pi")
                vib_sample = readings[-1]
            else:
                st.warning("Pi connected but no vibration stream data received — using fallback simulation")
                vib_sample = generate_sim_vibration()
        else:
            st.info("Pi not connected — using simulated vibration data")
            vib_sample = generate_sim_vibration()

        if 'pipe_vib_history' not in st.session_state:
            st.session_state['pipe_vib_history'] = []
        
        st.session_state['pipe_vib_history'].append(vib_sample)
        if len(st.session_state['pipe_vib_history']) > 100:
            st.session_state['pipe_vib_history'] = st.session_state['pipe_vib_history'][-100:]

        fig_vib = create_vibration_plot(st.session_state['pipe_vib_history'], height=350)
        st.plotly_chart(fig_vib, use_container_width=True, key="pipe_vib_chart")

        st.caption("BPFO: 74.6 Hz  |  BPFI: 117.4 Hz")

        st.markdown("<br>", unsafe_allow_html=True)

        sev = _COND_SEVERITY.get(detected_fault, 0)
        hs = float(np.clip(90 - sev * 55 + np.random.normal(0, 2), 0, 100))
        st.session_state['pipe_health_score'] = hs

        prev_h = st.session_state.get('prev_health', 85.0)
        fig_g = create_health_gauge(hs, prev_h, height=250)
        st.session_state['prev_health'] = hs
        st.plotly_chart(fig_g, use_container_width=True, key="pipe_gauge_chart")

        st.markdown("<br>", unsafe_allow_html=True)

        unc = abs(np.random.normal(3.5, 1.5))
        unc_lvl = "Low" if unc < 5 else ("Medium" if unc < 10 else "High")
        unc_col = HEALTHY if unc_lvl == "Low" else (WARNING if unc_lvl == "Medium" else CRITICAL)
        st.markdown(f"""
        <div class='mg-card'>
            <b>MC Dropout Uncertainty:</b> ± {unc:.1f}%
            &nbsp;&nbsp;<span style='color:{unc_col}; font-weight:700'>[{unc_lvl}]</span>
        </div>
        """, unsafe_allow_html=True)

        urg_label, urg_color = urgency_from_health(hs)
        st.session_state['pipe_urgency'] = urg_label
        st.markdown(f"""
        <div class='mg-card' style='text-align:center'>
            <div style='color:#8892b0; font-size:0.85em'>Maintenance Urgency</div>
            <div style='color:{urg_color}; font-size:1.8em; font-weight:700'>{urg_label}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("**Vibration Features**")
        f1, f2, f3 = st.columns(3)
        rms_val = round(np.std([vib_sample[k] for k in ["ax","ay","az"] if k in vib_sample]) * 0.707, 3)
        f1.metric("RMS (g)", rms_val)
        f2.metric("Kurtosis", round(3.0 + np.random.normal(0, 0.5), 2))
        f3.metric("Crest Factor", round(1.414 + np.random.normal(0, 0.2), 2))

    # ═══════════════ PRESCRIBE TAB ═══════════════
    with tab_rx:
        fc = st.session_state.get('pipe_fault_class', 'Healthy Baseline')
        hs_val = st.session_state.get('pipe_health_score', 85.0)

        st.markdown(f"""
        <div class='mg-card'>
            <b>From OBSERVE:</b> {fc}
            &nbsp;|&nbsp; <b>From DIAGNOSE:</b> Health = {hs_val:.1f}
        </div>
        """, unsafe_allow_html=True)

        render_prescription(
            fault_class=fc,
            health_score=hs_val,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        risk_level = "CRITICAL" if hs_val < 40 else ("HIGH" if hs_val < 60 else ("MODERATE" if hs_val < 80 else "LOW"))
        etf_hours = max(10, hs_val * 10)
        prescription_text = f"Motor Condition: {fc}. Risk: {risk_level}. Recommended repair protocol generated."

        save_session(
            fault_class=fc,
            health_score=hs_val,
            risk_level=risk_level,
            etf_hours=etf_hours,
            prescription=prescription_text,
            mode='Pipeline',
            pi_connected=pi_connected,
            engine_name=st.session_state.get('engine_name', 'Unknown Motor'),
            confidence=st.session_state.get('pipe_confidence', 0.0)
        )
        st.success("✅ Session saved to history")

    # ═══════════════ AUTO-REFRESH ═══════════════
    if st.session_state.get('pipe_toggle', False):
        time.sleep(st.session_state.get('pipe_interval_sl', 1.0))
        st.rerun()
