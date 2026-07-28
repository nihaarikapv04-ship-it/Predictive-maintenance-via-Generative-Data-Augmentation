"""
MotorGuard AI — Simulation Mode Page
Three connected tabs: OBSERVE | DIAGNOSE | PRESCRIBE
"""
import streamlit as st
import numpy as np
from datetime import datetime
from frontend.components.styles import (
    FAULT_COLORS, HEALTHY, CRITICAL, WARNING, ACCENT,
)
from frontend.components.camera import generate_sim_motor_frame
from frontend.components.vibration import (
    create_vibration_from_params, create_vibration_plot,
    create_health_gauge, urgency_from_health,
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


def render():
    st.markdown("<div class='panel-header'>🔬 Simulation Mode — Synthetic Telemetry</div>", unsafe_allow_html=True)

    # 🔧 Motor Configuration Header with Save Option
    st.markdown("### 🔧 Motor Configuration")
    c_name, c_rpm, c_save = st.columns([2, 1, 1])
    with c_name:
        engine_name = st.text_input(
            "Motor/Engine Name", 
            value=st.session_state.get('engine_name', ''),
            placeholder="e.g. Kirloskar 3-phase, Jewel 0.5HP",
            key="engine_name_input_sim"
        )
        st.session_state['engine_name'] = engine_name
    with c_rpm:
        engine_rpm = st.number_input("RPM", value=1440, min_value=0, max_value=10000, key="engine_rpm_sim")
        st.session_state['engine_rpm'] = engine_rpm
    with c_save:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        if st.button("💾 Save Session", key="save_btn_sim", use_container_width=True):
            e_name = st.session_state.get('engine_name', '').strip()
            e_rpm = st.session_state.get('engine_rpm', 0)
            if not e_name or e_rpm <= 0:
                st.warning("⚠️ You haven't entered the Motor Name and valid RPM! Please fill in both fields above before saving.")
            else:
                fc = st.session_state.get('sim_fault_class', 'Healthy Baseline')
                hs_val = st.session_state.get('sim_health_score', 85.0)
                conf_val = st.session_state.get('sim_confidence', 0.85)
                risk_level = "CRITICAL" if hs_val < 40 else ("HIGH" if hs_val < 60 else ("MODERATE" if hs_val < 80 else "LOW"))
                etf_hours = max(10, hs_val * 10)
                prescription_text = f"Motor Condition: {fc}. Risk: {risk_level}. Recommended repair protocol generated."

                save_session(
                    fault_class=fc,
                    health_score=hs_val,
                    risk_level=risk_level,
                    etf_hours=etf_hours,
                    prescription=prescription_text,
                    mode='Simulation',
                    pi_connected=False,
                    engine_name=e_name,
                    confidence=conf_val
                )
                st.success(f"✅ Session for '{e_name}' ({e_rpm} RPM) saved to history!")
    st.divider()

    tab_obs, tab_diag, tab_rx = st.tabs(["👁️ OBSERVE", "🩺 DIAGNOSE", "💊 PRESCRIBE"])

    # ═══════════════ OBSERVE TAB ═══════════════
    with tab_obs:
        c1, c2 = st.columns(2)
        with c1:
            cond = st.selectbox("Motor Condition:", CONDITIONS, key="sim_cond_sl")
            st.session_state['sim_fault_class'] = cond
        with c2:
            conf = st.slider("Confidence:", 0.50, 0.99, 0.85, 0.01, key="sim_conf_sl")
            st.session_state['sim_confidence'] = conf

        st.markdown("<br>", unsafe_allow_html=True)
        sim_frame = generate_sim_motor_frame(cond, conf)
        st.image(sim_frame, channels="RGB", use_container_width=True)

        f_color = FAULT_COLORS.get(cond, "#ffffff")
        status = "HEALTHY" if cond == "Healthy Baseline" else "FAULTY"
        st.markdown(f"""
        <div class='mg-card' style='border-left:5px solid {f_color}'>
            <span class='badge-healthy' style='display:inline-block;'>{cond}</span>
            <span style='margin-left:12px; color:{HEALTHY if status=="HEALTHY" else CRITICAL}; font-weight:700'>{status}</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(conf, text=f"Confidence: {conf:.1%}")

    # ═══════════════ DIAGNOSE TAB ═══════════════
    with tab_diag:
        st.markdown("**Vibration Parameters**")
        c1, c2 = st.columns(2)
        with c1:
            vib_amp = st.slider("Amplitude (g):", 0.1, 5.0, 1.2, 0.1, key="sim_amp_sl")
        with c2:
            vib_freq = st.slider("Frequency (Hz):", 5.0, 120.0, 30.0, 1.0, key="sim_freq_sl")

        st.markdown("<br>", unsafe_allow_html=True)
        vib_data = create_vibration_from_params(vib_amp, vib_freq)
        fig_vib = create_vibration_plot(vib_data, height=350)
        st.plotly_chart(fig_vib, use_container_width=True, key="sim_vib_chart")

        st.caption("BPFO: 74.6 Hz  |  BPFI: 117.4 Hz")
        st.markdown("<br>", unsafe_allow_html=True)

        detected_fault = st.session_state.get('sim_fault_class', 'Healthy Baseline')
        sev = _COND_SEVERITY.get(detected_fault, 0)
        hs = float(np.clip(95 - sev * 60 - vib_amp * 5 + np.random.normal(0, 1), 0, 100))
        st.session_state['sim_health_score'] = hs

        fig_g = create_health_gauge(hs, height=250)
        st.plotly_chart(fig_g, use_container_width=True, key="sim_gauge_chart")

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
        st.session_state['sim_urgency'] = urg_label
        st.markdown(f"""
        <div class='mg-card' style='text-align:center'>
            <div style='color:#8892b0; font-size:0.85em'>Maintenance Urgency</div>
            <div style='color:{urg_color}; font-size:1.8em; font-weight:700'>{urg_label}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Vibration Features**")
        f1, f2, f3 = st.columns(3)
        rms_val = round(vib_amp * 0.707, 3)
        f1.metric("RMS (g)", rms_val)
        f2.metric("Kurtosis", round(3.0 + vib_amp * 0.2, 2))
        f3.metric("Crest Factor", round(1.414 + vib_amp * 0.1, 2))

    # ═══════════════ PRESCRIBE TAB ═══════════════
    with tab_rx:
        fc = st.session_state.get('sim_fault_class', 'Healthy Baseline')
        hs_val = st.session_state.get('sim_health_score', 85.0)

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
