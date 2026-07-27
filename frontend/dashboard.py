"""
MotorGuard AI — Main Dashboard
================================
Single entry point: PYTHONPATH=. streamlit run frontend/dashboard.py
"""
import sys
import os

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

# ── Inject Inconsistent Theme Fix CSS ──
st.markdown("""
<style>
/* Force dark theme everywhere */
.stApp { background-color: #0e1117; color: #ffffff; }
.stApp > header { background-color: #0e1117; }
section[data-testid="stSidebar"] { 
    background-color: #1a1f2e; 
    border-right: 1px solid #2d3348;
}
section[data-testid="stSidebar"] button {
    background-color: transparent;
    color: #cccccc;
    border: none;
    text-align: left;
    font-size: 15px;
    padding: 10px 16px;
    border-radius: 8px;
    margin-bottom: 4px;
}
section[data-testid="stSidebar"] button:hover {
    background-color: #2d3348;
    color: #00d4ff;
}
.metric-card {
    background-color: #1a1f2e;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    border: 1px solid #2d3348;
}
.panel-header {
    font-size: 18px;
    font-weight: 600;
    color: #00d4ff;
    border-left: 3px solid #00d4ff;
    padding-left: 12px;
    margin-bottom: 20px;
}
div[data-testid="column"] {
    padding: 0 12px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #1a1f2e;
    color: #888888;
    border-radius: 8px 8px 0 0;
}
.stTabs [aria-selected="true"] {
    background-color: #2d3348;
    color: #00d4ff;
}
</style>
""", unsafe_allow_html=True)

# ── Initialize session_state page ──
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

# Also initialize current_page alias for backward compatibility
st.session_state['current_page'] = st.session_state['page']

# ── Sidebar Navigation ──
with st.sidebar:
    st.markdown("# ⚙️ MotorGuard AI")
    st.markdown("---")
    if st.button("🏠 Home", use_container_width=True):
        st.session_state['page'] = 'home'
        st.session_state['current_page'] = 'home'
        st.rerun()
    if st.button("🔬 Simulation Mode", use_container_width=True):
        st.session_state['page'] = 'simulation'
        st.session_state['current_page'] = 'simulation'
        st.rerun()
    if st.button("⚡ Pipeline Mode", use_container_width=True):
        st.session_state['page'] = 'pipeline'
        st.session_state['current_page'] = 'pipeline'
        st.rerun()
    if st.button("📊 History", use_container_width=True):
        st.session_state['page'] = 'history'
        st.session_state['current_page'] = 'history'
        st.rerun()

    st.markdown("---")
    st.caption("v2.0 · Physics-Aware Generative Maintenance")

# ── Route to Page ──
page = st.session_state['page']

if page == 'home':
    from frontend.pages.home import render
    render()
elif page == 'simulation':
    from frontend.pages.simulation import render
    render()
elif page == 'pipeline':
    from frontend.pages.pipeline import render
    render()
elif page == 'history':
    from frontend.pages.history import render
    render()
else:
    st.error(f"Unknown page: {page}")
