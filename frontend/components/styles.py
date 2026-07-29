"""
MotorGuard AI — Global CSS Design System
Enterprise Industrial Theme for Apple Silicon & Edge Deployment
"""
import streamlit as st

def inject_css():
    st.markdown("""
    <style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global */
    * { font-family: 'Inter', sans-serif; }
    .stApp { 
        background: #0e1117 !important;
        color: #e8eaf6 !important;
    }
    
    /* Hide Streamlit footer & main menu while keeping sidebar toggle button visible */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: transparent !important;
        color: #e8eaf6 !important;
    }
    
    /* Sidebar Layout */
    section[data-testid="stSidebar"] {
        background: #121622 !important;
        border-right: 1px solid #23293a !important;
        padding-top: 10px;
        display: block !important;
        visibility: visible !important;
    }
    section[data-testid="stSidebar"] * {
        color: #ccd6f6 !important;
    }
    
    .sidebar-logo {
        padding: 12px 14px 18px 14px;
        border-bottom: 1px solid #23293a;
        margin-bottom: 18px;
    }
    .sidebar-logo-text {
        font-size: 1.25rem;
        font-weight: 700;
        color: #00d4ff !important;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .sidebar-logo-sub {
        font-size: 0.75rem;
        color: #8892b0 !important;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 500;
    }

    /* INACTIVE SIDEBAR BUTTONS */
    section[data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        color: #a8b2d8 !important;
        border: 1px solid transparent !important;
        width: 100% !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 11px 16px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        margin-bottom: 6px !important;
        transition: all 0.2s ease-in-out !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #1e2536 !important;
        color: #00d4ff !important;
        border-color: rgba(0, 212, 255, 0.2) !important;
        transform: translateX(2px);
    }

    /* ACTIVE HIGHLIGHT SIDEBAR BUTTONS */
    section[data-testid="stSidebar"] .stButton > button[kind="primary"],
    section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
        background: linear-gradient(90deg, rgba(0, 212, 255, 0.18) 0%, rgba(0, 212, 255, 0.04) 100%) !important;
        color: #00d4ff !important;
        border-left: 4px solid #00d4ff !important;
        border-top: 1px solid rgba(0, 212, 255, 0.25) !important;
        border-right: 1px solid rgba(0, 212, 255, 0.25) !important;
        border-bottom: 1px solid rgba(0, 212, 255, 0.25) !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 12px rgba(0, 212, 255, 0.15) !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding-left: 14px !important;
        border-radius: 0 8px 8px 0 !important;
    }

    /* GENERAL MAIN PAGE BUTTON STYLING */
    .main .stButton > button {
        background: #161b27 !important;
        color: #00d4ff !important;
        border: 1px solid #00d4ff !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 22px !important;
        letter-spacing: 0.5px !important;
        transition: all 0.2s ease !important;
    }
    .main .stButton > button:hover {
        background: #00d4ff !important;
        color: #1A1A1A !important;
        border-color: #00d4ff !important;
        box-shadow: 0 4px 14px rgba(0, 212, 255, 0.35) !important;
    }

    /* PRIMARY MAIN PAGE BUTTONS */
    .main .stButton > button[kind="primary"], .main button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #00d4ff 0%, #00a3cc 100%) !important;
        color: #1A1A1A !important;
        border: none !important;
        font-weight: 700 !important;
    }

    /* Cards */
    .mg-card {
        background: #161b27 !important;
        border: 1px solid #2d3348 !important;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 18px;
        color: #e8eaf6 !important;
    }
    .mg-card * {
        color: #e8eaf6 !important;
    }
    
    /* Feature cards on home */
    .feature-card {
        background: #161b27 !important;
        border: 1px solid #2d3348 !important;
        border-radius: 16px;
        padding: 28px 20px;
        text-align: center;
        height: 190px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .feature-card:hover {
        border-color: #00d4ff !important;
        transform: translateY(-2px);
    }
    .feature-icon { font-size: 2.2rem; margin-bottom: 10px; }
    .feature-title { 
        font-size: 1.05rem; 
        font-weight: 600; 
        color: #00d4ff !important;
        margin-bottom: 6px;
    }
    .feature-desc { 
        font-size: 0.82rem; 
        color: #a8b2d8 !important;
        line-height: 1.4;
    }
    
    /* Panel headers */
    .panel-header {
        font-size: 1.05rem;
        font-weight: 600 !important;
        color: #00d4ff !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        padding-bottom: 10px;
        border-bottom: 1px solid #2d3348;
        margin-bottom: 18px;
    }
    
    /* Status badges with explicit WHITE text */
    .badge-healthy {
        background: rgba(0, 255, 136, 0.2) !important;
        color: #ffffff !important;
        border: 1px solid #00ff88 !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700 !important;
        letter-spacing: 1px;
    }
    .badge-faulty {
        background: rgba(255, 68, 68, 0.2) !important;
        color: #ffffff !important;
        border: 1px solid #ff4444 !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700 !important;
        letter-spacing: 1px;
    }
    
    /* Metric display */
    [data-testid="stMetricLabel"] {
        color: #a8b2d8 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-weight: 700 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #161b27 !important;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
        border: 1px solid #2d3348 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #a8b2d8 !important;
        border-radius: 6px;
        padding: 8px 20px;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: #2d3348 !important;
        color: #00d4ff !important;
        font-weight: 700 !important;
    }
    
    /* ALL input fields */
    .stTextInput input, .stNumberInput input, .stSelectbox > div > div {
        background: #1e2535 !important;
        border: 1px solid #3d4663 !important;
        color: #e8eaf6 !important;
        border-radius: 8px !important;
    }
    
    /* History table */
    .stDataFrame {
        border: 1px solid #2d3348 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        background: #161b27 !important;
    }
    .stDataFrame [data-testid="stHeader"] {
        background: #1e2535 !important;
        color: #e8eaf6 !important;
        font-weight: 600 !important;
    }
    
    hr { border-color: #2d3348 !important; }
    
    .main .block-container {
        max-width: 100%;
        padding: 2rem 2.5rem;
        overflow-x: hidden;
    }
    
    .status-connected { color: #00ff88 !important; font-size: 0.8rem; font-weight: 600; }
    .status-disconnected { color: #ff4444 !important; font-size: 0.8rem; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

ACCENT = "#00d4ff"
HEALTHY = "#00ff88"
WARNING = "#ffaa00"
CRITICAL = "#ff4444"
PANEL_BG = "#161b27"
TEXT_PRIMARY = "#e8eaf6"
TEXT_SECONDARY = "#a8b2d8"

FAULT_COLORS = {
    "Healthy Baseline":    HEALTHY,
    "Mild Oxidation":      "#f0e130",
    "Moderate Corrosion":  WARNING,
    "Severe Corrosion":    CRITICAL,
    "Structural Cracking": CRITICAL,
    "Contamination":       "#9b30ff",
}
