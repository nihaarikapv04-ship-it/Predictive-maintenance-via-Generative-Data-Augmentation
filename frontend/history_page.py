"""
MotorGuard AI — History Page
Persistent session history saved to data/history.json.
"""
import streamlit as st
import pandas as pd
import json
import os
import uuid
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from frontend.components.styles import ACCENT

HISTORY_FILE = 'data/history.json'


def save_session(fault_class, health_score, risk_level, etf_hours, 
                 prescription, mode, pi_connected, engine_name, confidence):
    os.makedirs('data', exist_ok=True)
    entry = {
        'id': str(uuid.uuid4())[:8],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'engine_name': engine_name or "Unknown Motor",
        'mode': mode,
        'fault_class': fault_class,
        'confidence': round(float(confidence), 3),
        'health_score': round(float(health_score), 1),
        'risk_level': risk_level,
        'etf_hours': round(float(etf_hours), 0),
        'prescription': str(prescription)[:300],
        'pi_connected': pi_connected
    }
    history = load_history()
    history.insert(0, entry)  # newest first
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return []


def render():
    st.markdown("<div class='panel-header'>📊 Session History</div>", unsafe_allow_html=True)

    history = load_history()
    if not history:
        st.info("No sessions recorded yet. Run a Simulation or Pipeline session to generate history.")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔬 Go to Simulation"):
            st.session_state['page'] = 'simulation'
            st.rerun()
        return

    # Filters
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

    filtered = history
    if mode_filter != "All":
        filtered = [h for h in filtered if h.get("mode") == mode_filter]
    if fault_filter != "All":
        filtered = [h for h in filtered if h.get("fault_class") == fault_filter]
    if risk_filter != "All":
        filtered = [h for h in filtered if h.get("risk_level") == risk_filter]

    st.markdown("<br>", unsafe_allow_html=True)

    # Table
    if filtered:
        rows = []
        for h in filtered:
            rows.append({
                "ID": h.get("id", "N/A"),
                "Engine": h.get("engine_name", "N/A"),
                "Time": h.get("timestamp", "N/A"),
                "Mode": h.get("mode", "N/A"),
                "Fault Class": h.get("fault_class", "N/A"),
                "Conf": f"{h.get('confidence', 0):.1%}",
                "Health Score": h.get("health_score", 0),
                "Risk": h.get("risk_level", "N/A"),
                "ETF (hrs)": h.get("etf_hours", 0),
                "Pi": "🟢" if h.get("pi_connected") else "🔴",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True, height=300)

        # Expand details
        for i, h in enumerate(filtered[:10]):
            with st.expander(f"Session {h.get('id', i+1)} — {h.get('engine_name', 'Motor')} [{h.get('timestamp', '')}]"):
                d1, d2, d3 = st.columns(3)
                d1.metric("Fault Class", h.get("fault_class", "N/A"))
                d2.metric("Health Score", f"{h.get('health_score', 0):.1f}")
                d3.metric("Risk Level", h.get("risk_level", "N/A"))
                if h.get("prescription"):
                    st.markdown(f"**Prescription snippet:** {h['prescription']}")
    else:
        st.warning("No sessions match the current filters.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    if len(filtered) >= 1:
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("**Fault Class Distribution**")
            fault_counts = pd.Series([h.get("fault_class", "Unknown") for h in filtered]).value_counts()
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

    # Actions
    a1, a2 = st.columns(2)
    with a1:
        if filtered:
            csv = pd.DataFrame(filtered).to_csv(index=False)
            st.download_button(
                "📥 Export History as CSV", data=csv,
                file_name="motorguard_history.csv", mime="text/csv",
            )
    with a2:
        if st.button("🗑️ Clear History"):
            with open(HISTORY_FILE, "w") as f:
                json.dump([], f)
            st.success("History cleared!")
            st.rerun()
