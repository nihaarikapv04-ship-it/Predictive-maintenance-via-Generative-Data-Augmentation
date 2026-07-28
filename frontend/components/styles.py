"""
MotorGuard AI — Global CSS Design System
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
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #161b27 !important;
        border-right: 1px solid #2d3348 !important;
        padding-top: 0;
        display: block !important;
        visibility: visible !important;
    }
    section[data-testid="stSidebar"] * {
        color: #ccd6f6 !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        color: #ccd6f6 !important;
        border: none !important;
        width: 100% !important;
        text-align: left !important;
        padding: 12px 20px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        margin-bottom: 4px !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #2d3348 !important;
        color: #00d4ff !important;
    }

    /* GENERAL BUTTON STYLING */
    .stButton > button {
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
    .stButton > button:hover {
        background: #00d4ff !important;
        color: #1A1A1A !important;  /* High-contrast dark charcoal text */
        border-color: #00d4ff !important;
        box-shadow: 0 4px 14px rgba(0, 212, 255, 0.35) !important;
    }

    /* PRIMARY BUTTONS ('Enter Simulation Mode', 'Enter Pipeline Mode') */
    .stButton > button[kind="primary"], button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #00d4ff 0%, #00a3cc 100%) !important;
        color: #1A1A1A !important;  /* Dark charcoal text on bright cyan background */
        border: none !important;
        font-weight: 700 !important;
    }
    .stButton > button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #33ddff 0%, #00b8e6 100%) !important;
        color: #1A1A1A !important;
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.45) !important;
    }

    /* SECONDARY BUTTONS ('Save Session', etc.) */
    .stButton > button[kind="secondary"], button[data-testid="baseButton-secondary"] {
        background: #1e2535 !important;
        color: #e8eaf6 !important;
        border: 1px solid #3d4663 !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="secondary"]:hover, button[data-testid="baseButton-secondary"]:hover {
        background: #2d3348 !important;
        color: #00d4ff !important;
        border-color: #00d4ff !important;
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
    .badge-critical {
        background: rgba(255, 68, 68, 0.3) !important;
        color: #ffffff !important;
        border: 1px solid #ff4444 !important;
        padding: 6px 16px;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 700 !important;
        letter-spacing: 1.5px;
    }
    .badge-moderate {
        background: rgba(255, 170, 0, 0.3) !important;
        color: #ffffff !important;
        border: 1px solid #ffaa00 !important;
        padding: 6px 16px;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 700 !important;
        letter-spacing: 1.5px;
    }
    .badge-low {
        background: rgba(0, 255, 136, 0.3) !important;
        color: #ffffff !important;
        border: 1px solid #00ff88 !important;
        padding: 6px 16px;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 700 !important;
        letter-spacing: 1.5px;
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
    
    /* Tabs: unselected #a8b2d8, selected #00d4ff */
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
    
    /* ALL input fields: background #1e2535, text #e8eaf6, border #3d4663 */
    .stTextInput input, .stNumberInput input, .stSelectbox > div > div {
        background: #1e2535 !important;
        border: 1px solid #3d4663 !important;
        color: #e8eaf6 !important;
        border-radius: 8px !important;
    }
    
    /* History table styling */
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
    .stDataFrame [data-testid="stTable"] tr {
        background: #161b27 !important;
        color: #e8eaf6 !important;
    }
    .stDataFrame [data-testid="stTable"] tr:nth-child(even) {
        background: #1a2040 !important;
    }
    
    /* Divider */
    hr { border-color: #2d3348 !important; }
    
    /* Columns spacing */
    div[data-testid="column"] { padding: 0 10px; }
    
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
