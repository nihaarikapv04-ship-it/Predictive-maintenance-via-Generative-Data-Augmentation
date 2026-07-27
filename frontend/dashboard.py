import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import requests

st.set_page_config(
    page_title="MotorGuard AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Must be first st call after set_page_config
from frontend.components.styles import inject_css
inject_css()

from frontend.home_page import render as render_home
from frontend.simulation_page import render as render_simulation
from frontend.pipeline_page import render as render_pipeline
from frontend.history_page import render as render_history

# Initialize session state
defaults = {
    'page': 'home',
    'engine_name': '',
    'engine_rpm': 1440,
    'sim_fault_class': 'Healthy Baseline',
    'sim_health_score': 85.0,
    'sim_confidence': 0.85,
    'pipe_fault_class': 'Healthy Baseline',
    'pipe_health_score': 85.0,
    'pipe_confidence': 0.0,
    'pi_ip': '192.168.1.100',
    'pipeline_running': False,
    'camera_source': 'webcam',
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-text">⚙️ MotorGuard AI</div>
        <div class="sidebar-logo-sub">Physics-Aware Maintenance</div>
    </div>
    """, unsafe_allow_html=True)
    
    pages = {
        'home': '🏠  Home',
        'simulation': '🔬  Simulation Mode',
        'pipeline': '⚡  Pipeline Mode',
        'history': '📊  History',
    }
    
    for key, label in pages.items():
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state['page'] = key
            st.rerun()
    
    st.markdown("---")
    
    # Pi connection in sidebar
    st.markdown("**🔌 Raspberry Pi**")
    pi_ip = st.text_input("IP Address", value=st.session_state.get('pi_ip', '192.168.1.100'), key="pi_ip_input_nav")
    st.session_state['pi_ip'] = pi_ip
    
    def check_pi(ip):
        try:
            r = requests.get(f"http://{ip}:5000/health", timeout=2)
            return r.status_code == 200
        except:
            return False
            
    if check_pi(pi_ip):
        st.markdown("<span class='status-connected'>🟢 Connected</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='status-disconnected'>🔴 Disconnected</span>", unsafe_allow_html=True)

# Route
page = st.session_state.get('page', 'home')
if page == 'home':
    render_home()
elif page == 'simulation':
    render_simulation()
elif page == 'pipeline':
    render_pipeline()
elif page == 'history':
    render_history()
else:
    render_home()
