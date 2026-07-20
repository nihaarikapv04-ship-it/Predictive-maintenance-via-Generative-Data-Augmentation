"""
MotorGuard AI — Home Page
"""
import streamlit as st
from datetime import datetime
from frontend.components.styles import ACCENT, PANEL_BG, TEXT_SECONDARY


def render():
    # ── Hero ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='text-align:center; padding:40px 0 20px 0'>
        <div style='font-size:3.2em; font-weight:700; letter-spacing:-1px'>
            ⚙️ Motor<span style='color:{ACCENT}'>Guard</span> AI
        </div>
        <div style='font-size:1.1em; color:{TEXT_SECONDARY}; margin-top:8px'>
            Physics-Aware Generative Maintenance with Edge Deployment
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature Cards ──
    c1, c2, c3 = st.columns(3, gap="large")
    _cards = [
        (c1, "👁️", "Observe", "YOLOv11n visual fault detection on motor surface with 6-class classification"),
        (c2, "🧠", "Diagnose", "Physics-Aware TimeGAN + Late-Fusion LSTM with Monte Carlo Dropout uncertainty"),
        (c3, "💊", "Prescribe", "RAG-powered repair guidance from motor manual with Llama-3 generation"),
    ]
    for col, icon, title, desc in _cards:
        with col:
            st.markdown(f"""
            <div class='feature-card'>
                <div class='icon'>{icon}</div>
                <div class='title'>{title}</div>
                <div class='desc'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ── Action Buttons ──
    b1, _, b2 = st.columns([1, 0.3, 1])
    with b1:
        if st.button("🔬  Enter Simulation Mode", use_container_width=True, type="primary"):
            st.session_state.current_page = "simulation"
            st.rerun()
    with b2:
        if st.button("⚡  Enter Pipeline Mode", use_container_width=True, type="primary"):
            st.session_state.current_page = "pipeline"
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ── System Stats ──
    st.markdown("---")
    st.markdown("**System Information**")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("YOLO Model", "v11n")
    i2.metric("Fusion Model", "LSTM-MC")
    i3.metric("RAG Backend", "Llama-3 8B")
    i4.metric("Last Session", datetime.now().strftime("%H:%M"))

    st.markdown("<br>", unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("<span class='sdot off'></span> Raspberry Pi: <b>Not Connected</b>", unsafe_allow_html=True)
    with s2:
        st.markdown("<span class='sdot on'></span> Simulation Engine: <b>Ready</b>", unsafe_allow_html=True)
