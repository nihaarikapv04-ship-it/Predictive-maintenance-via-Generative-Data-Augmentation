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

    # 📖 Simulation Parameters & Beginner's Guide
    with st.expander("📖 Simulation & Pipeline Parameters Guide (Simple Sentences)", expanded=True):
        st.markdown("""
        ### What Each Parameter Means & How It Works:

        1. **🏷️ Motor / Engine Name**:
           - *What it is*: The name or model number of your motor (e.g., *Kirloskar 3-Phase*, *Jewel 0.5HP*).
           - *Why it matters*: Identifies which machine is being inspected so history logs stay organized.

        2. **⚡ RPM (Rotations Per Minute)**:
           - *What it is*: How many complete 360° spins the motor shaft completes in 1 minute (e.g., *1440 RPM*).
           - *Why it matters*: Higher RPMs naturally create higher vibration frequencies and heat.

        3. **👁️ Motor Surface Conditions (The 6 Defect Classes)**:
           - 🟢 **Healthy Baseline**: The motor housing is clean and free of damage.
           - 🟡 **Mild Oxidation**: Slight surface discoloration or early light rust spots.
           - 🟧 **Moderate Corrosion**: Spreading rust patches that require cleaning and protective coating.
           - 🔴 **Severe Corrosion**: Heavy rust scaling threatening the strength of the motor casing.
           - 🔴 **Structural Cracking**: Visible fractures or hairline cracks in the metal housing.
           - 🟣 **Contamination**: Accumulation of oil leaks, heavy dust, or foreign chemicals on the motor.

        4. **🎯 Confidence Score (%)**:
           - *What it is*: How sure the AI vision model (YOLOv11) is about its defect detection (e.g., *88% Confidence*).

        5. **〰️ Vibration Amplitude (g)**:
           - *What it is*: How hard the motor is shaking, measured in G-force acceleration.
           - *Simple meaning*: Low shaking (below 1.0g) is normal; heavy shaking (above 2.5g) means internal damage, loose bolts, or bearing failure.

        6. **📈 Vibration Frequency (Hz)**:
           - *What it is*: How fast the shaking cycles repeat per second.
           - *Simple meaning*: Helps pinpoint exact failing parts inside the motor (like inner bearing rings BPFI or outer rings BPFO).

        7. **🩺 Health Score (0 to 100)**:
           - *What it is*: The overall health condition of your motor.
           - *Simple meaning*: **90–100** = Excellent (Healthy), **60–80** = Moderate Wear, **Below 40** = Critical Danger (Halt Machine!).

        8. **📊 Monte Carlo Dropout Uncertainty (±%)**:
           - *What it is*: The AI's self-checked margin of error. A lower percentage (e.g., $\pm 3\%$) means the AI is very certain of its health diagnosis.

        9. **⏳ Estimated Time to Failure (ETF / RUL in Hours)**:
           - *What it is*: How many working hours the motor can run before it is projected to break down completely.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # System Info
    st.markdown("---")
    st.markdown("**System Information**")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("YOLO Model", "v11n")
    i2.metric("Fusion Model", "LSTM-MC")
    i3.metric("RAG Backend", "Llama-3 8B")
    i4.metric("Last Session", datetime.now().strftime("%H:%M"))
