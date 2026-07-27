"""
MotorGuard AI — Simulation Mode Page
Three connected tabs: OBSERVE | DIAGNOSE | PRESCRIBE
"""
import streamlit as st
import numpy as np
from frontend.components.styles import FAULT_COLORS, HEALTHY, CRITICAL
from frontend.components.camera import generate_sim_motor_frame
from frontend.components.vibration import (
    create_vibration_plot, create_vibration_from_params,
    create_health_gauge, compute_features, health_from_params, urgency_from_health,
)
from frontend.components.rag_display import render_prescription
from frontend.history_page import save_session

CONDITIONS = [
    "Healthy Baseline", "Mild Oxidation", "Moderate Corrosion",
    "Severe Corrosion", "Structural Cracking", "Contamination",
]


def render():
    st.markdown("<div class='panel-header'>🔬 Simulation Mode</div>", unsafe_allow_html=True)

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
            fc = st.session_state.get('sim_fault_class', CONDITIONS[0])
            hs_val = st.session_state.get('sim_health_score', 85.0)
            conf_val = st.session_state.get('sim_confidence', 0.85)
            risk_level = "CRITICAL" if hs_val < 40 else ("HIGH" if hs_val < 60 else ("MODERATE" if hs_val < 80 else "LOW"))
            etf_hours = max(10, hs_val * 10)
            prescription_text = f"Condition: {fc}. Health: {hs_val:.1f}. Recommended maintenance protocol generated."

            save_session(
                fault_class=fc,
                health_score=hs_val,
                risk_level=risk_level,
                etf_hours=etf_hours,
                prescription=prescription_text,
                mode='Simulation',
                pi_connected=False,
                engine_name=engine_name or "Unknown Motor",
                confidence=conf_val
            )
            st.success("✅ Session saved to history!")
    st.divider()

    tab_obs, tab_diag, tab_rx = st.tabs(["👁️ OBSERVE", "🩺 DIAGNOSE", "💊 PRESCRIBE"])

    # ═══════════════════════ OBSERVE TAB ═══════════════════════
    with tab_obs:
        col_ctrl, col_vis = st.columns([1, 2], gap="large")

        with col_ctrl:
            st.markdown("**Simulation Controls**")
            fault = st.selectbox("Fault Type", CONDITIONS, key="sim_fault_sel")
            conf = st.slider("Confidence Level", 0.50, 1.00, 0.85, 0.01, key="sim_conf_sl")
            noise = st.slider("Noise Level", 0.0, 1.0, 0.2, 0.05, key="sim_noise_sl")

            st.session_state['sim_fault_class'] = fault
            st.session_state['sim_confidence'] = conf

            st.markdown("<br>", unsafe_allow_html=True)

            status = "HEALTHY" if fault == "Healthy Baseline" else "FAULTY"
            f_color = FAULT_COLORS.get(fault, "#ffffff")
            st.markdown(f"""
            <div class='mg-card' style='border-left:5px solid {f_color}'>
                <div class='badge-healthy' style='display:inline-block;'>{fault}</div>
                <div style='margin-top:10px'>
                    <b>Confidence:</b> {conf:.1%}<br>
                    <b>Status:</b> <span style='color:{HEALTHY if status=="HEALTHY" else CRITICAL}; font-weight:700'>{status}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_vis:
            st.markdown("**Motor Visualization**")
            frame = generate_sim_motor_frame(condition=fault, confidence=conf)
            st.image(frame, use_container_width=True)

    # ═══════════════════════ DIAGNOSE TAB ═══════════════════════
    with tab_diag:
        detected_fault = st.session_state.get('sim_fault_class', CONDITIONS[0])
        st.info(f"Analyzing motor with detected condition from OBSERVE: **{detected_fault}**")

        col_sl, col_ch = st.columns([1, 2], gap="large")

        with col_sl:
            st.markdown("**Vibration Parameters**")
            amp = st.slider("Amplitude (g)", 0.1, 5.0, 1.0, 0.1, key="sim_amp_sl")
            freq = st.slider("Frequency (Hz)", 5.0, 200.0, 30.0, 1.0, key="sim_freq_sl")
            load = st.slider("Load Level (%)", 0.0, 100.0, 50.0, 5.0, key="sim_load_sl")
            temp = st.slider("Temperature (°C)", 20.0, 120.0, 55.0, 1.0, key="sim_temp_sl")

            st.markdown("<br>", unsafe_allow_html=True)

            feats = compute_features(amp, freq, load, temp)
            st.markdown("**Extracted Features**")
            for k, v in feats.items():
                st.metric(k, v)

        with col_ch:
            vib_data = create_vibration_from_params(amp, freq)
            fig_vib = create_vibration_plot(vib_data, height=350)
            st.plotly_chart(fig_vib, use_container_width=True, key="sim_vib_chart")

            st.markdown("<br>", unsafe_allow_html=True)

            hs = health_from_params(amp, freq, load, temp)
            if "severe" in detected_fault.lower() or "crack" in detected_fault.lower():
                hs = min(hs, 35.0)
            elif "moderate" in detected_fault.lower() or "contam" in detected_fault.lower():
                hs = min(hs, 55.0)
            elif "mild" in detected_fault.lower():
                hs = min(hs, 75.0)

            st.session_state['sim_health_score'] = hs

            prev_h = st.session_state.get('prev_health', 85.0)
            fig_g = create_health_gauge(hs, prev_h)
            st.session_state['prev_health'] = hs
            st.plotly_chart(fig_g, use_container_width=True, key="sim_gauge_chart")

            st.markdown("<br>", unsafe_allow_html=True)

            unc = abs(np.random.normal(3.0, 1.0))
            unc_lvl = "Low" if unc < 5 else ("Medium" if unc < 10 else "High")
            unc_col = HEALTHY if unc_lvl == "Low" else ("#ffaa00" if unc_lvl == "Medium" else CRITICAL)
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

    # ═══════════════════════ PRESCRIBE TAB ═══════════════════════
    with tab_rx:
        fc = st.session_state.get('sim_fault_class', CONDITIONS[0])
        hs_val = st.session_state.get('sim_health_score', 85.0)
        conf_val = st.session_state.get('sim_confidence', 0.85)

        st.markdown(f"""
        <div class='mg-card'>
            <b>From OBSERVE:</b> {fc}
            &nbsp;|&nbsp; <b>From DIAGNOSE:</b> Health = {hs_val:.1f}
        </div>
        """, unsafe_allow_html=True)

        render_prescription(
            fault_class=fc,
            health_score=hs_val,
        )

        risk_level = "CRITICAL" if hs_val < 40 else ("HIGH" if hs_val < 60 else ("MODERATE" if hs_val < 80 else "LOW"))
        etf_hours = max(10, hs_val * 10)
        prescription_text = f"Condition: {fc}. Health: {hs_val:.1f}. Recommended maintenance protocol generated."

        save_session(
            fault_class=fc,
            health_score=hs_val,
            risk_level=risk_level,
            etf_hours=etf_hours,
            prescription=prescription_text,
            mode='Simulation',
            pi_connected=False,
            engine_name=st.session_state.get('engine_name', 'Unknown Motor'),
            confidence=conf_val
        )
        st.success("✅ Session saved to history")
