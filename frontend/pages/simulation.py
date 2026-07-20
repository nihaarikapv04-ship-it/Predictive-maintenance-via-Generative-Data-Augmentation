"""
MotorGuard AI — Simulation Mode Page
Three tabs: OBSERVE | DIAGNOSE | PRESCRIBE
"""
import streamlit as st
import numpy as np
from datetime import datetime
from frontend.components.styles import ACCENT, FAULT_COLORS, PANEL_BG, HEALTHY, CRITICAL
from frontend.components.camera import generate_sim_motor_frame
from frontend.components.vibration import (
    create_vibration_plot, create_vibration_from_params,
    create_health_gauge, compute_features, health_from_params, urgency_from_health,
)
from frontend.components.rag_display import render_prescription

CONDITIONS = [
    "Healthy Baseline", "Mild Oxidation", "Moderate Corrosion",
    "Severe Corrosion", "Structural Cracking", "Contamination",
]


def render():
    st.markdown("<div class='mg-header observe'>🔬 Simulation Mode</div>", unsafe_allow_html=True)

    tab_obs, tab_diag, tab_rx = st.tabs(["👁️ OBSERVE", "🩺 DIAGNOSE", "💊 PRESCRIBE"])

    # ═══════════════════════ OBSERVE TAB ═══════════════════════
    with tab_obs:
        col_ctrl, col_vis = st.columns([1, 2], gap="large")

        with col_ctrl:
            st.markdown("**Simulation Controls**")
            fault = st.selectbox("Fault Type", CONDITIONS, key="sim_fault_sel")
            st.session_state.sim_fault_class = fault

            conf = st.slider("Confidence Level", 0.50, 1.00, 0.85, 0.01, key="sim_conf_sl")
            st.session_state.sim_confidence = conf

            noise = st.slider("Noise Level", 0.0, 1.0, 0.2, 0.05, key="sim_noise")

            st.markdown("<br>", unsafe_allow_html=True)

            # Detection result card
            status = "HEALTHY" if fault == "Healthy Baseline" else "FAULTY"
            f_color = FAULT_COLORS.get(fault, "#ffffff")
            st.markdown(f"""
            <div class='mg-card' style='border-left:5px solid {f_color}'>
                <div class='mg-badge' style='background:{f_color}22; color:{f_color}'>{fault}</div>
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
        col_sl, col_ch = st.columns([1, 2], gap="large")

        with col_sl:
            st.markdown("**Vibration Parameters**")
            amp = st.slider("Amplitude (g)", 0.1, 5.0, 1.0, 0.1, key="sim_amp")
            st.session_state.sim_amplitude = amp
            freq = st.slider("Frequency (Hz)", 5.0, 200.0, 30.0, 1.0, key="sim_freq")
            st.session_state.sim_frequency = freq
            load = st.slider("Load Level (%)", 0.0, 100.0, 50.0, 5.0, key="sim_load")
            st.session_state.sim_load = load
            temp = st.slider("Temperature (°C)", 20.0, 120.0, 55.0, 1.0, key="sim_temp")
            st.session_state.sim_temperature = temp

            st.markdown("<br>", unsafe_allow_html=True)

            # Features
            feats = compute_features(amp, freq, load, temp)
            st.markdown("**Extracted Features**")
            for k, v in feats.items():
                st.metric(k, v)

        with col_ch:
            # Vibration plot
            vib_data = create_vibration_from_params(amp, freq)
            fig_vib = create_vibration_plot(vib_data, height=350)
            st.plotly_chart(fig_vib, use_container_width=True, key="sim_vib")

            st.markdown("<br>", unsafe_allow_html=True)

            # Health gauge
            hs = health_from_params(amp, freq, load, temp)
            st.session_state.sim_health_score = hs
            fig_g = create_health_gauge(hs, st.session_state.prev_health)
            st.session_state.prev_health = hs
            st.plotly_chart(fig_g, use_container_width=True, key="sim_gauge")

            st.markdown("<br>", unsafe_allow_html=True)

            # Uncertainty
            unc = abs(np.random.normal(3.0, 1.0))
            unc_lvl = "Low" if unc < 5 else ("Medium" if unc < 10 else "High")
            unc_col = HEALTHY if unc_lvl == "Low" else ("#ffaa00" if unc_lvl == "Medium" else CRITICAL)
            st.markdown(f"""
            <div class='mg-card'>
                <b>MC Dropout Uncertainty:</b> ± {unc:.1f}%
                &nbsp;&nbsp;<span style='color:{unc_col}; font-weight:700'>[{unc_lvl}]</span>
            </div>
            """, unsafe_allow_html=True)

            # Urgency
            urg_label, urg_color = urgency_from_health(hs)
            st.markdown(f"""
            <div class='mg-card' style='text-align:center'>
                <div style='color:#888; font-size:0.85em'>Maintenance Urgency</div>
                <div style='color:{urg_color}; font-size:1.8em; font-weight:700'>{urg_label}</div>
            </div>
            """, unsafe_allow_html=True)

    # ═══════════════════════ PRESCRIBE TAB ═══════════════════════
    with tab_rx:
        st.markdown(f"""
        <div class='mg-card'>
            <b>Input from OBSERVE:</b> {st.session_state.sim_fault_class}
            &nbsp;|&nbsp; <b>Input from DIAGNOSE:</b> Health = {st.session_state.sim_health_score:.1f}
        </div>
        """, unsafe_allow_html=True)

        render_prescription(
            fault_class=st.session_state.sim_fault_class,
            health_score=st.session_state.sim_health_score,
        )
