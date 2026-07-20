"""
MotorGuard AI — Main Dashboard (Navigation Shell)
===================================================
Single entry point: streamlit run frontend/dashboard.py
Routes to pages via st.session_state["current_page"].
"""
import sys
import os

# Ensure the project root (parent of frontend/) is on sys.path
# so that `from frontend.components.X import Y` works correctly.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

# ── Page config (must be first) ──
st.set_page_config(
    page_title="MotorGuard AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ──
from frontend.components.styles import inject_css
inject_css()

# ── Session State Defaults ──
_defaults = {
    "current_page": "home",
    # Simulation state
    "sim_fault_class": "Healthy Baseline",
    "sim_confidence": 0.85,
    "sim_health_score": 85.0,
    "sim_amplitude": 1.0,
    "sim_frequency": 30.0,
    "sim_load": 50.0,
    "sim_temperature": 55.0,
    # Pipeline state
    "pipe_fault_class": "Healthy Baseline",
    "pipe_confidence": 0.0,
    "pipe_health_score": 85.0,
    "pipe_running": False,
    "pipe_camera_source": "webcam",
    "pipe_ip_url": "",
    "pipe_resolution": "640x480",
    "pipe_interval": 1.0,
    "pipe_pi_ip": "192.168.1.100",
    "pipe_vib_history": [],
    "pipe_det_history": [],
    # History
    "history": [],
    # Misc
    "prev_health": 85.0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar Navigation ──
with st.sidebar:
    st.markdown("## ⚙️ MotorGuard AI")
    st.markdown("---")

    pages = [
        ("🏠  Home",            "home"),
        ("🔬  Simulation Mode", "simulation"),
        ("⚡  Pipeline Mode",   "pipeline"),
        ("📊  History",         "history"),
    ]
    for label, key in pages:
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.current_page = key
            st.rerun()

    st.markdown("---")
    st.caption("v2.0 · Physics-Aware Generative Maintenance")

# ── Route to Page ──
page = st.session_state.current_page

if page == "home":
    from frontend.pages.home import render
    render()
elif page == "simulation":
    from frontend.pages.simulation import render
    render()
elif page == "pipeline":
    from frontend.pages.pipeline import render
    render()
elif page == "history":
    from frontend.pages.history import render
    render()
else:
    st.error(f"Unknown page: {page}")
