"""
MotorGuard AI — Global CSS Design System
==========================================
All CSS injected via st.markdown(). Import inject_css() from any page.
"""

# ── Color Palette ──
BG           = "#0e1117"
PANEL_BG     = "#1a1f2e"
ACCENT       = "#00d4ff"
HEALTHY      = "#00ff88"
WARNING      = "#ffaa00"
CRITICAL     = "#ff4444"
TEXT_PRIMARY  = "#ffffff"
TEXT_SECONDARY= "#888888"

# Fault-class colours (hex)
FAULT_COLORS = {
    "Healthy Baseline":    HEALTHY,
    "Mild Oxidation":      "#f0e130",
    "Moderate Corrosion":  "#ff8c00",
    "Severe Corrosion":    CRITICAL,
    "Structural Cracking": CRITICAL,
    "Contamination":       "#9b30ff",
}


def inject_css():
    """Inject the full design-system CSS into the Streamlit page."""
    import streamlit as st
    st.markdown(_CSS, unsafe_allow_html=True)


_CSS = """
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    width: 250px !important;
    background: #0d1017;
}
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    padding: 14px 18px;
    margin-bottom: 6px;
    border-radius: 10px;
    border: none;
    background: transparent;
    color: #ccc;
    font-size: 1.0em;
    font-weight: 500;
    transition: background 0.2s;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(0,212,255,0.12);
    color: #fff;
}

/* ── Column Padding ── */
[data-testid="stHorizontalBlock"] > div { padding: 0 14px; }

/* ── Cards ── */
.mg-card {
    background: """ + PANEL_BG + """;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    transition: transform 0.2s, box-shadow 0.2s;
}
.mg-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.3);
}

/* ── Panel Headers ── */
.mg-header {
    font-size: 18px;
    font-weight: 600;
    padding: 12px 16px;
    margin-bottom: 18px;
    border-radius: 8px;
}
.mg-header.observe   { border-left: 4px solid """ + HEALTHY + """;  background: rgba(0,255,136,0.07); }
.mg-header.diagnose  { border-left: 4px solid """ + ACCENT + """;   background: rgba(0,212,255,0.07); }
.mg-header.prescribe { border-left: 4px solid """ + WARNING + """;  background: rgba(255,170,0,0.07); }

/* ── Badges ── */
.mg-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.05em;
    letter-spacing: 0.5px;
}

/* ── Status dots ── */
.sdot { height:10px; width:10px; border-radius:50%; display:inline-block; margin-right:7px; }
.sdot.on  { background: """ + HEALTHY + """; }
.sdot.off { background: """ + CRITICAL + """; }

/* ── Feature cards (home) ── */
.feature-card {
    background: """ + PANEL_BG + """;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 28px 22px;
    text-align: center;
    min-height: 180px;
    transition: transform 0.25s, box-shadow 0.25s;
}
.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(0,212,255,0.15);
    border-color: """ + ACCENT + """44;
}
.feature-card .icon { font-size: 2.4em; margin-bottom: 12px; }
.feature-card .title { font-size: 1.15em; font-weight: 700; color: """ + ACCENT + """; margin-bottom: 6px; }
.feature-card .desc  { font-size: 0.85em; color: """ + TEXT_SECONDARY + """; }

/* ── Sim banner ── */
.sim-banner {
    background: rgba(255,170,0,0.12);
    color: """ + WARNING + """;
    padding: 10px; text-align:center; border-radius:6px;
    border: 1px solid """ + WARNING + """;
    font-weight:600; margin:8px 0;
}

/* ── Pulse ── */
.pulse-red { animation: pulse 1.8s infinite; }
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(255,68,68,0.55); }
    70%  { box-shadow: 0 0 0 12px rgba(255,68,68,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,68,68,0); }
}

/* ── Fixed bottom bar ── */
.bottom-bar {
    position:fixed; bottom:0; left:0; right:0;
    background: #0a0d12; border-top:1px solid rgba(255,255,255,0.08);
    padding:8px 28px; z-index:9999;
    display:flex; justify-content:space-around; align-items:center;
    font-size:0.82em; color:#999;
}

/* ── Prevent content hiding behind fixed bar ── */
.main .block-container { padding-bottom: 60px !important; }

/* ── Active nav button ── */
section[data-testid="stSidebar"] .active-nav > button {
    background: rgba(0,212,255,0.18) !important;
    color: #00d4ff !important;
    font-weight: 700 !important;
    border-left: 3px solid #00d4ff !important;
}

/* ── Remove default Streamlit padding on top ── */
.block-container { padding-top: 1.5rem !important; }
</style>
"""
