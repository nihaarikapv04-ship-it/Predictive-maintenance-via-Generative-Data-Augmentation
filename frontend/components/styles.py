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
        background: linear-gradient(135deg, #0a0e1a 0%, #0e1117 50%, #0a0e1a 100%);
        color: #e8eaf6;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
        border-right: 1px solid rgba(0, 212, 255, 0.15);
        padding-top: 0;
    }
    section[data-testid="stSidebar"] > div {
        padding: 0;
    }
    
    /* Sidebar buttons */
    section[data-testid="stSidebar"] .stButton > button {
        background: transparent;
        color: #8892b0;
        border: none;
        width: 100%;
        text-align: left;
        padding: 14px 24px;
        font-size: 14px;
        font-weight: 500;
        letter-spacing: 0.3px;
        border-radius: 0;
        border-left: 3px solid transparent;
        transition: all 0.2s ease;
        margin: 0;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(0, 212, 255, 0.08);
        color: #00d4ff;
        border-left: 3px solid #00d4ff;
    }
    
    /* Cards */
    .mg-card {
        background: rgba(22, 27, 39, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 212, 255, 0.12);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        transition: border-color 0.2s ease;
    }
    .mg-card:hover {
        border-color: rgba(0, 212, 255, 0.3);
    }
    
    /* Feature cards on home */
    .feature-card {
        background: rgba(22, 27, 39, 0.6);
        border: 1px solid rgba(0, 212, 255, 0.1);
        border-radius: 20px;
        padding: 32px 24px;
        text-align: center;
        transition: all 0.3s ease;
        height: 200px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .feature-card:hover {
        border-color: rgba(0, 212, 255, 0.4);
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0, 212, 255, 0.1);
    }
    .feature-icon { font-size: 2.5rem; margin-bottom: 12px; }
    .feature-title { 
        font-size: 1.1rem; 
        font-weight: 600; 
        color: #00d4ff;
        margin-bottom: 8px;
    }
    .feature-desc { 
        font-size: 0.85rem; 
        color: #8892b0;
        line-height: 1.5;
    }
    
    /* Panel headers */
    .panel-header {
        font-size: 1rem;
        font-weight: 600;
        color: #ccd6f6;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(0, 212, 255, 0.15);
        margin-bottom: 20px;
    }
    
    /* Status badges */
    .badge-healthy {
        background: rgba(0, 255, 136, 0.15);
        color: #00ff88;
        border: 1px solid rgba(0, 255, 136, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .badge-faulty {
        background: rgba(255, 68, 68, 0.15);
        color: #ff4444;
        border: 1px solid rgba(255, 68, 68, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .badge-critical {
        background: rgba(255, 68, 68, 0.2);
        color: #ff4444;
        border: 1px solid #ff4444;
        padding: 6px 16px;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 2px;
    }
    .badge-moderate {
        background: rgba(255, 170, 0, 0.2);
        color: #ffaa00;
        border: 1px solid #ffaa00;
        padding: 6px 16px;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 2px;
    }
    .badge-low {
        background: rgba(0, 255, 136, 0.2);
        color: #00ff88;
        border: 1px solid #00ff88;
        padding: 6px 16px;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 2px;
    }
    
    /* Metric display */
    .big-metric {
        font-size: 3rem;
        font-weight: 700;
        color: #00d4ff;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 4px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(22, 27, 39, 0.6);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid rgba(0, 212, 255, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #8892b0;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0, 212, 255, 0.15) !important;
        color: #00d4ff !important;
    }
    
    /* Immediate action box */
    .action-box {
        background: rgba(255, 68, 68, 0.08);
        border: 1px solid rgba(255, 68, 68, 0.3);
        border-left: 4px solid #ff4444;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
    }
    
    /* Repair protocol box */
    .repair-box {
        background: rgba(255, 170, 0, 0.06);
        border: 1px solid rgba(255, 170, 0, 0.2);
        border-left: 4px solid #ffaa00;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
    }
    
    /* Schedule box */
    .schedule-box {
        background: rgba(0, 255, 136, 0.06);
        border: 1px solid rgba(0, 255, 136, 0.2);
        border-left: 4px solid #00ff88;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
    }
    
    /* Camera feed */
    .camera-container {
        border: 2px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        overflow: hidden;
        background: #000;
    }
    
    /* Divider */
    hr { border-color: rgba(0, 212, 255, 0.1) !important; }
    
    /* Input fields */
    .stTextInput input, .stNumberInput input {
        background: rgba(22, 27, 39, 0.8) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        color: #e8eaf6 !important;
        border-radius: 8px !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: rgba(22, 27, 39, 0.8) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        color: #e8eaf6 !important;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.2s ease !important;
    }
    
    /* Columns spacing */
    div[data-testid="column"] { padding: 0 10px; }
    
    /* No horizontal scroll */
    .main .block-container {
        max-width: 100%;
        padding: 2rem 2.5rem;
        overflow-x: hidden;
    }
    
    /* Logo area in sidebar */
    .sidebar-logo {
        padding: 24px 20px 16px;
        border-bottom: 1px solid rgba(0, 212, 255, 0.1);
        margin-bottom: 16px;
    }
    .sidebar-logo-text {
        font-size: 1.1rem;
        font-weight: 700;
        color: #00d4ff;
        letter-spacing: 1px;
    }
    .sidebar-logo-sub {
        font-size: 0.7rem;
        color: #8892b0;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }
    
    /* History table */
    .stDataFrame {
        border: 1px solid rgba(0, 212, 255, 0.1) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    /* Connection status */
    .status-connected {
        color: #00ff88;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .status-disconnected {
        color: #ff4444;
        font-size: 0.8rem;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

ACCENT = "#00d4ff"
HEALTHY = "#00ff88"
WARNING = "#ffaa00"
CRITICAL = "#ff4444"
PANEL_BG = "rgba(22, 27, 39, 0.8)"
TEXT_PRIMARY = "#e8eaf6"
TEXT_SECONDARY = "#8892b0"

FAULT_COLORS = {
    "Healthy Baseline":    HEALTHY,
    "Mild Oxidation":      "#f0e130",
    "Moderate Corrosion":  WARNING,
    "Severe Corrosion":    CRITICAL,
    "Structural Cracking": CRITICAL,
    "Contamination":       "#9b30ff",
}
