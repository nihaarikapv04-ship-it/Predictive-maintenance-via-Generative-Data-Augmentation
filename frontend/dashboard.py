import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import requests
from frontend.components.camera import stop_camera_instance

st.set_page_config(
    page_title="MotorGuard AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from frontend.components.styles import inject_css
inject_css()

from frontend.home_page import render as render_home
from frontend.simulation_page import render as render_simulation
from frontend.pipeline_page import render as render_pipeline
from frontend.history_page import render as render_history

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
    'pi_ip': 'rpi.local',
    'pipeline_running': False,
    'camera_source': 'webcam',
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

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
    
    current_page = st.session_state.get('page', 'home')
    for key, label in pages.items():
        is_active = (current_page == key)
        b_type = "primary" if is_active else "secondary"
        if st.button(label, key=f"nav_{key}", type=b_type, use_container_width=True):
            if key != 'pipeline':
                stop_camera_instance()
            st.session_state['page'] = key
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("**🔌 Raspberry Pi**")
    pi_ip = st.text_input("IP Address", value=st.session_state.get('pi_ip', 'rpi.local'), key="pi_ip_input_nav")
    st.session_state['pi_ip'] = pi_ip
    
    @st.cache_data(ttl=5, show_spinner=False)
    def check_pi(ip):
        try:
            ip_str = ip.strip()
            if not ip_str:
                return False
            if not ip_str.startswith("http://") and not ip_str.startswith("https://"):
                if ":" in ip_str and not ip_str.startswith("["):
                    url = f"http://{ip_str}/health"
                else:
                    url = f"http://{ip_str}:5000/health"
            else:
                url = f"{ip_str}/health"
            r = requests.get(url, timeout=0.8)
            return r.status_code == 200
        except Exception:
            return False
            
    if check_pi(pi_ip):
        st.markdown("<span class='status-connected'>🟢 Connected</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='status-disconnected'>🔴 Disconnected</span>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.sidebar.expander("📖 How to connect Pi"):
        st.markdown("""
        **On your Raspberry Pi:**
        ```bash
        python3 app.py
        ```
        **Find your Pi IP address:**
        Run `hostname -I` on your Pi terminal or enter `rpi.local`.
        """)

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
