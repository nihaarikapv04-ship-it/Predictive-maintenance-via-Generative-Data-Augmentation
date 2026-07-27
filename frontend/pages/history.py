"""
MotorGuard AI — History Page
Persistent session history saved to data/history.json locally.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import uuid
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from frontend.components.styles import ACCENT, CRITICAL, WARNING, HEALTHY, PANEL_BG

HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "history.json",
)


def load_history() -> list:
    """Load history from JSON file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_session(fault_class: str, health_score: float, risk_level: str, etf_hours: float, mode: str):
    """Save a session entry to history.json."""
    entry = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "fault_class": fault_class,
        "health_score": round(float(health_score), 1),
        "risk_level": risk_level,
        "etf_hours": round(float(etf_hours), 0),
    }
    history = load_history()
    history.append(entry)
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save history: {e}")


def render():
    st.markdown("<div class='mg-header diagnose'>📊 Session History</div>", unsafe_allow_html=True)

    history = load_history()
    if not history:
        st.info("No sessions recorded yet. Run a Simulation or Pipeline session to generate history.")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔬 Go to Simulation", type="primary"):
            st.session_state['page'] = 'simulation'
            st.session_state['current_page'] = 'simulation'
            st.rerun()
        return

    # ── Filters ──
    st.markdown("**Filters**")
    f1, f2, f3 = st.columns(3)
    with f1:
        mode_filter = st.selectbox("Mode", ["All", "Simulation", "Pipeline"], key="hist_mode_sel")
    with f2:
        fault_opts = ["All"] + sorted(list(set(h.get("fault_class", "Unknown") for h in history)))
        fault_filter = st.selectbox("Fault Class", fault_opts, key="hist_fault_sel")
    with f3:
        risk_opts = ["All", "CRITICAL", "HIGH", "MODERATE", "LOW"]
        risk_filter = st.selectbox("Risk Level", risk_opts, key="hist_risk_sel")

    # Apply filters
    filtered = history
    if mode_filter != "All":
        filtered = [h for h in filtered if h.get("mode") == mode_filter]
    if fault_filter != "All":
        filtered = [h for h in filtered if h.get("fault_class") == fault_filter]
    if risk_filter != "All":
        filtered = [h for h in filtered if h.get("risk_level") == risk_filter]

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Table ──
    if filtered:
        rows = []
        for h in filtered:
            rows.append({
                "ID": h.get("id", "N/A"),
                "Time": h.get("timestamp", "N/A"),
                "Mode": h.get("mode", "N/A"),
                "Fault Class": h.get("fault_class", "N/A"),
                "Health Score": h.get("health_score", 0),
                "Risk": h.get("risk_level", "N/A"),
                "ETF (hrs)": h.get("etf_hours", 0),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True, height=300)

        # ── Expand details ──
        for i, h in enumerate(filtered):
            with st.expander(f"Session {h.get('id', i+1)} — {h.get('timestamp', 'N/A')} [{h.get('mode', '')}]"):
                d1, d2, d3 = st.columns(3)
                d1.metric("Fault Class", h.get("fault_class", "N/A"))
                d2.metric("Health Score", f"{h.get('health_score', 0):.1f}")
                d3.metric("Risk", h.get("risk_level", "N/A"))
    else:
        st.warning("No sessions match the current filters.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──
    if len(filtered) >= 1:
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("**Fault Class Distribution**")
            fault_counts = pd.Series(
                [h.get("fault_class", "Unknown") for h in filtered]
            ).value_counts()
            fig_bar = px.bar(
                x=fault_counts.index, y=fault_counts.values,
                labels={"x": "Fault Class", "y": "Count"},
                color_discrete_sequence=[ACCENT],
            )
            fig_bar.update_layout(
                template="plotly_dark", height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="hist_bar_chart")
        with ch2:
            st.markdown("**Health Score Over Time**")
            scores = [h.get("health_score", 0) for h in filtered]
            fig_line = go.Figure(go.Scatter(
                y=scores, mode="lines+markers",
                line=dict(color=ACCENT, width=2),
            ))
            fig_line.update_layout(
                template="plotly_dark", height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="Health", xaxis_title="Session",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_line, use_container_width=True, key="hist_line_chart")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Actions ──
    a1, a2 = st.columns(2)
    with a1:
        if filtered:
            csv = pd.DataFrame(filtered).to_csv(index=False)
            st.download_button(
                "📥 Export History as CSV", data=csv,
                file_name="motorguard_history.csv", mime="text/csv",
            )
    with a2:
        if st.button("🗑️ Clear History", type="secondary"):
            os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
            with open(HISTORY_FILE, "w") as f:
                json.dump([], f)
            st.success("History cleared!")
            st.rerun()
