import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
import time
import json
from collections import deque
from datetime import datetime, timedelta
import base64

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title='MotorGuard AI',
    page_icon='⚙️',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ==========================================
# CUSTOM CSS
# ==========================================
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #0e1117;
            color: #ffffff;
        }
        
        .observe-header {
            background: linear-gradient(90deg, #00d4aa 0%, transparent 100%);
            padding: 10px;
            border-radius: 5px;
            font-weight: 700;
        }
        
        .diagnose-header {
            background: linear-gradient(90deg, #4da6ff 0%, transparent 100%);
            padding: 10px;
            border-radius: 5px;
            font-weight: 700;
        }
        
        .prescribe-header {
            background: linear-gradient(90deg, #ff6b35 0%, transparent 100%);
            padding: 10px;
            border-radius: 5px;
            font-weight: 700;
        }
        
        .metric-card {
            background: rgba(26, 29, 35, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.2);
        }
        
        .critical-alert {
            animation: pulse 2s infinite;
            border: 2px solid #ff3333 !important;
            background: rgba(255, 51, 51, 0.1) !important;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 51, 51, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(255, 51, 51, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 51, 51, 0); }
        }
        
        .status-dot {
            height: 10px;
            width: 10px;
            background-color: #00d4aa;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
        }
        
        .status-dot.offline {
            background-color: #ff3333;
        }
        
        .simulation-banner {
            background-color: rgba(255, 170, 0, 0.2);
            color: #ffaa00;
            padding: 10px;
            text-align: center;
            border-radius: 5px;
            border: 1px solid #ffaa00;
            margin-top: 10px;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ==========================================
# STATE MANAGEMENT
# ==========================================
if 'vibration_history' not in st.session_state:
    st.session_state.vibration_history = deque(maxlen=500)
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = deque(maxlen=10)
if 'health_score_history' not in st.session_state:
    st.session_state.health_score_history = []
if 'pipeline_running' not in st.session_state:
    st.session_state.pipeline_running = False
if 'simulation_mode' not in st.session_state:
    st.session_state.simulation_mode = True
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'last_backend_status' not in st.session_state:
    st.session_state.last_backend_status = False

# ==========================================
# DATA FETCHING / MOCKING
# ==========================================
def get_mock_data():
    t = time.time()
    noise = np.random.normal(0, 0.1, 6)
    vib = [
        np.sin(t) + noise[0], np.cos(t) + noise[1], np.sin(t*0.5) + noise[2],
        np.cos(t*1.5) + noise[3], np.sin(t*2) + noise[4], np.cos(t*0.2) + noise[5]
    ]
    
    classes = ['Normal', 'Inner Race', 'Outer Race', 'Ball', 'Misalignment']
    fault = np.random.choice(classes, p=[0.7, 0.1, 0.1, 0.05, 0.05])
    conf = np.random.uniform(0.6, 0.99)
    
    health = 75 + np.random.normal(0, 2)
    health = max(0, min(100, health))
    
    return {
        'status': 'success',
        'timestamp': t,
        'vibration': vib,
        'detection': {
            'class': fault,
            'confidence': conf,
            'boxes': np.random.randint(0, 5)
        },
        'health_score': health,
        'latency_ms': np.random.uniform(20, 50),
        'uncertainty': np.random.uniform(0.01, 0.1),
        'rul_hours': np.random.uniform(100, 1000)
    }

def fetch_data(sim_mode=False):
    if not sim_mode:
        try:
            response = requests.get('http://localhost:5000/pipeline/run', timeout=2)
            if response.status_code == 200:
                st.session_state.last_backend_status = True
                return response.json()
        except requests.exceptions.RequestException:
            st.session_state.last_backend_status = False
            pass
    return get_mock_data()

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## ⚙️ MotorGuard AI")
    st.markdown("---")
    
    st.markdown("### System Controls")
    st.session_state.pipeline_running = st.toggle("Start Pipeline", value=st.session_state.pipeline_running)
    st.session_state.simulation_mode = st.toggle("Simulation Mode", value=st.session_state.simulation_mode)
    refresh_rate = st.slider("Refresh Rate (s)", 0.5, 5.0, 1.0, 0.1)
    
    st.markdown("### Model Configuration")
    yolo_conf = st.slider("YOLO Conf Threshold", 0.1, 0.9, 0.25, 0.05)
    mc_samples = st.slider("MC Dropout Samples", 10, 100, 50, 10)
    
    st.markdown("### Status")
    backend_status_cls = "status-dot" if st.session_state.last_backend_status or st.session_state.simulation_mode else "status-dot offline"
    st.markdown(f"<div><span class='{backend_status_cls}'></span> Backend</div>", unsafe_allow_html=True)
    st.markdown("<div><span class='status-dot'></span> Camera (Sim)</div>", unsafe_allow_html=True)
    st.markdown("<div><span class='status-dot'></span> Sensor (Sim)</div>", unsafe_allow_html=True)
    
    uptime = timedelta(seconds=int(time.time() - st.session_state.start_time))
    st.markdown(f"<small>Uptime: {uptime}</small>", unsafe_allow_html=True)

# ==========================================
# MAIN LAYOUT
# ==========================================
col1, col2, col3 = st.columns([1, 1, 1])

# Fetch latest data if running
if st.session_state.pipeline_running:
    data = fetch_data(st.session_state.simulation_mode)
    
    st.session_state.vibration_history.append({
        't': data['timestamp'],
        'ax': data['vibration'][0], 'ay': data['vibration'][1], 'az': data['vibration'][2],
        'gx': data['vibration'][3], 'gy': data['vibration'][4], 'gz': data['vibration'][5]
    })
    
    st.session_state.health_score_history.append(data['health_score'])
    if len(st.session_state.health_score_history) > 100:
        st.session_state.health_score_history.pop(0)
        
    st.session_state.detection_history.appendleft({
        'time': datetime.fromtimestamp(data['timestamp']).strftime('%H:%M:%S'),
        'class': data['detection']['class'],
        'conf': data['detection']['confidence']
    })
else:
    data = get_mock_data() # just for UI render when stopped

# ----------------- PANEL 1: OBSERVE -----------------
with col1:
    st.markdown("<div class='observe-header'>🔍 OBSERVE – Visual Inspection</div>", unsafe_allow_html=True)
    
    # Placeholder for webcam feed
    st.image("https://via.placeholder.com/640x480.png?text=Webcam+Feed+(Simulated)", use_container_width=True)
    
    fault_class = data['detection']['class']
    conf = data['detection']['confidence']
    
    color_map = {
        'Normal': '#00d4aa',
        'Inner Race': '#ff3333',
        'Outer Race': '#ff6b35',
        'Ball': '#9933ff',
        'Misalignment': '#ff3399'
    }
    f_color = color_map.get(fault_class, '#ffffff')
    
    st.markdown(f"""
    <div class='metric-card' style='border-left: 5px solid {f_color}'>
        <h3 style='margin:0; color:{f_color}'>{fault_class.upper()}</h3>
        <p style='margin:0; font-size: 0.8em; color:#aaaaaa;'>Current Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.progress(conf, text=f"Confidence: {conf:.1%}")
    
    st.markdown("#### Statistics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Objects", data['detection']['boxes'])
    m2.metric("Avg Conf", f"{conf:.2f}")
    m3.metric("Inf Time", f"{data['latency_ms']:.1f}ms")
    
    st.markdown("#### Recent Detections")
    if len(st.session_state.detection_history) > 0:
        df_det = pd.DataFrame(st.session_state.detection_history)
        st.dataframe(df_det, use_container_width=True, hide_index=True)

# ----------------- PANEL 2: DIAGNOSE -----------------
with col2:
    st.markdown("<div class='diagnose-header'>🩺 DIAGNOSE – Health Analytics</div>", unsafe_allow_html=True)
    
    # Vibration Plot
    if len(st.session_state.vibration_history) > 0:
        df_vib = pd.DataFrame(st.session_state.vibration_history)
        fig_vib = go.Figure()
        colors = ['#ff3333', '#00d4aa', '#4da6ff', '#ff6b35', '#9933ff', '#ffaa00']
        channels = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
        for i, ch in enumerate(channels):
            fig_vib.add_trace(go.Scatter(y=df_vib[ch], mode='lines', name=ch, line=dict(color=colors[i], width=1)))
        
        fig_vib.update_layout(
            template='plotly_dark',
            margin=dict(l=0, r=0, t=20, b=0),
            height=200,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_vib, use_container_width=True, key="vib_plot")
    
    # Health Score Gauge
    hs = data['health_score']
    hs_delta = hs - st.session_state.health_score_history[-2] if len(st.session_state.health_score_history) > 1 else 0
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = hs,
        delta = {'reference': hs - hs_delta, 'position': "top"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "white"},
            'steps': [
                {'range': [0, 40], 'color': "#ff3333"},
                {'range': [40, 60], 'color': "#ff6b35"},
                {'range': [60, 80], 'color': "#ffaa00"},
                {'range': [80, 100], 'color': "#00d4aa"}
            ]
        }
    ))
    fig_gauge.update_layout(template='plotly_dark', height=200, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True, key="gauge")
    
    st.markdown(f"""
    <div class='metric-card'>
        <div><b>Uncertainty (MC Dropout):</b> ± {data['uncertainty']:.3f}</div>
        <div><b>Confidence Interval:</b> {(hs - data['uncertainty']*10):.1f} - {(hs + data['uncertainty']*10):.1f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Urgency Flag
    urgency = "NONE"
    u_color = "#00d4aa"
    u_class = "metric-card"
    if hs < 40:
        urgency = "IMMEDIATE"
        u_color = "#ff3333"
        u_class = "metric-card critical-alert"
    elif hs < 60:
        urgency = "SOON"
        u_color = "#ff6b35"
    elif hs < 80:
        urgency = "ROUTINE"
        u_color = "#4da6ff"
        
    st.markdown(f"""
    <div class='{u_class}' style='text-align:center'>
        <h4 style='margin:0; color:#aaaaaa'>Maintenance Urgency</h4>
        <h2 style='margin:5px 0; color:{u_color}'>{urgency}</h2>
        <p style='margin:0'>Est. Remaining Useful Life: <b>{data['rul_hours']:.1f} hrs</b></p>
    </div>
    """, unsafe_allow_html=True)


# ----------------- PANEL 3: PRESCRIBE -----------------
with col3:
    st.markdown("<div class='prescribe-header'>📝 PRESCRIBE – Repair Protocol</div>", unsafe_allow_html=True)
    
    risk_level = "LOW"
    r_color = "#00d4aa"
    if hs < 40:
        risk_level = "CRITICAL"
        r_color = "#ff3333"
    elif hs < 60:
        risk_level = "HIGH"
        r_color = "#ff6b35"
    elif hs < 80:
        risk_level = "MODERATE"
        r_color = "#ffaa00"
        
    st.markdown(f"""
    <div class='metric-card' style='background-color: {r_color}33; border: 1px solid {r_color}; text-align: center'>
        <h3 style='margin:0; color:{r_color}'>RISK: {risk_level}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    etf_color = "#ff3333" if data['rul_hours'] < 24 else "#ffffff"
    st.markdown(f"""
    <div class='metric-card'>
        <h4 style='margin:0; color:#aaaaaa'>Estimated Time to Failure</h4>
        <h1 style='margin:0; color:{etf_color}'>{data['rul_hours']:.1f} <span style='font-size:0.5em'>Hours</span></h1>
    </div>
    """, unsafe_allow_html=True)
    
    if urgency == "IMMEDIATE":
        st.markdown("""
        <div style='border: 2px solid #ff3333; padding: 10px; border-radius: 5px; margin-bottom: 10px; background: rgba(255, 51, 51, 0.1)'>
            <h4 style='color:#ff3333; margin-top:0'>⚠️ IMMEDIATE ACTION REQUIRED</h4>
            <ol style='margin-bottom:0'>
                <li>Halt machine operation</li>
                <li>Isolate power supply (LOTO)</li>
                <li>Prepare for component replacement</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
    with st.expander("Step 1: Diagnostics & Isolation"):
        st.markdown("- **Action:** Run full diagnostic suite.\n- **Tools:** Multimeter, Diagnostic tablet.\n- **Time:** 15m")
    with st.expander("Step 2: Component Inspection"):
        st.markdown("- **Action:** Visual inspection of bearings.\n- **Tools:** Flashlight, Endoscope.\n- **Time:** 30m")
    with st.expander("Step 3: Replacement"):
        st.markdown("- **Action:** Replace faulty bearing.\n- **Tools:** Wrench set, Puller.\n- **Time:** 2h")
        
    st.markdown("#### Preventive Schedule")
    sched = pd.DataFrame({
        'Task': ['Lubrication', 'Alignment Check', 'Vibration Analysis'],
        'Interval': ['Monthly', 'Quarterly', 'Weekly'],
        'Due': ['In 5 days', 'In 20 days', 'Tomorrow']
    })
    st.dataframe(sched, use_container_width=True, hide_index=True)
    
    # Download report
    report_content = f"MotorGuard AI Report\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nHealth Score: {hs:.1f}\nFault: {fault_class}\nRUL: {data['rul_hours']:.1f}h"
    b64 = base64.b64encode(report_content.encode()).decode()
    st.download_button(label="Download PDF Report", data=report_content, file_name="motorguard_report.txt", mime="text/plain")


# ==========================================
# BOTTOM STATUS BAR
# ==========================================
st.markdown("---")
if st.session_state.simulation_mode:
    st.markdown("<div class='simulation-banner'>⚠️ SIMULATION MODE – Using synthetic data</div>", unsafe_allow_html=True)

b1, b2, b3, b4 = st.columns(4)
b1.markdown(f"**Pipeline:** {'Running' if st.session_state.pipeline_running else 'Stopped'}")
b2.markdown(f"**Latency:** {data['latency_ms']:.1f} ms")
b3.markdown(f"**Last Run:** {datetime.now().strftime('%H:%M:%S')}")
b4.markdown(f"**FPS:** {int(1000/max(1, data['latency_ms']))}")


# ==========================================
# AUTO REFRESH LOOP
# ==========================================
if st.session_state.pipeline_running:
    time.sleep(refresh_rate)
    st.rerun()
