"""
MotorGuard AI — Home Page
"""
import streamlit as st
from datetime import datetime
from frontend.components.styles import ACCENT, TEXT_SECONDARY


def render():
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='text-align:center; padding:30px 0 10px 0'>
        <div style='font-size:3em; font-weight:700; letter-spacing:-1px'>
            ⚙️ Motor<span style='color:{ACCENT}'>Guard</span> AI
        </div>
        <div style='font-size:1.05em; color:{TEXT_SECONDARY}; margin-top:8px'>
            Physics-Aware Generative Maintenance with Edge Deployment
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Cards
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
                <div class='feature-icon'>{icon}</div>
                <div class='feature-title'>{title}</div>
                <div class='feature-desc'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Action Buttons
    b1, _, b2 = st.columns([1, 0.2, 1])
    with b1:
        if st.button("🔬  Enter Simulation Mode", use_container_width=True):
            st.session_state['page'] = "simulation"
            st.rerun()
    with b2:
        if st.button("⚡  Enter Pipeline Mode", use_container_width=True):
            st.session_state['page'] = "pipeline"
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    # 📖 System Parameters & Diagnostic Reference
    with st.expander("📖 System Parameters & Diagnostic Reference", expanded=True):
        st.markdown("""
        ### Industrial Parameter Specifications:

        1. **🏷️ Motor / Engine Designation**:
           - Unique equipment identifier and manufacturer model specification (*e.g., Kirloskar 3-Phase, Jewel 0.5HP*).
           - Maintains traceability across historical maintenance records.

        2. **⚡ Shaft Rotations Per Minute (RPM)**:
           - Rotational frequency of the primary motor shaft (*e.g., 1440 RPM for standard 4-pole induction motor*).
           - Baseline for establishing fundamental rotational frequencies ($f_r$).

        3. **👁️ Surface Defect Classifications (YOLOv11 Visual Path)**:
           - 🟢 **Healthy Baseline**: Clean housing, nominal surface integrity without defects.
           - 🟡 **Mild Oxidation**: Surface discoloration or early-stage oxidative films.
           - 🟧 **Moderate Corrosion**: Localized rust scaling requiring abrasive cleaning and protective sealants.
           - 🔴 **Severe Corrosion**: Advanced oxide penetration compromising structural casing thickness.
           - 🔴 **Structural Cracking**: Physical fractures or mechanical housing cracks.
           - 🟣 **Contamination**: Foreign material accumulation, oil leakage, or particulate deposits.

        4. **🎯 Detection Confidence Index (%)**:
           - Softmax probability associated with the primary YOLOv11 visual bounding box.

        5. **〰️ Vibration Acceleration Amplitude (g)**:
           - Root-Mean-Square (RMS) acceleration amplitude across 3 orthogonal axes ($a_x, a_y, a_z$).

        6. **📈 Characteristic Frequencies (Hz)**:
           - Spectral peak components mapped to bearing defect harmonics (BPFI: 117.4 Hz, BPFO: 74.6 Hz).

        7. **🩺 Multimodal Health Index (0 to 100)**:
           - Fused health metric computed via Late-Fusion LSTM neural network.

        8. **📊 Monte Carlo Dropout Uncertainty (±%)**:
           - Epistemic uncertainty computed via $N=50$ stochastic forward passes ($1.96 \cdot \sigma$).

        9. **⏳ Remaining Useful Life (RUL / ETF in Hours)**:
           - Projected operational lifespan prior to critical degradation limits.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # ⏱️ System Latency Benchmarks (Table VII)
    st.markdown("---")
    st.markdown("### ⚡ System Latency & Performance Benchmarks (Table VII)")
    
    st.markdown("""
    > *"The complete edge-based Observe–Diagnose–Prescribe pipeline executes in approximately **515 ms** on a Raspberry Pi 5, while the fault detection path alone completes in **65 ms**, enabling near real-time monitoring at around **28 FPS**."*
    """)

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Detection Path Latency", "65 ms", "~28 FPS")
    i2.metric("End-to-End Pipeline", "515 ms", "Total O-D-P")
    i3.metric("YOLOv11n Inference", "27 ms", "Surface Defect")
    i4.metric("Llama-3-8B Generation", "340 ms", "Prescriptive RAG")
