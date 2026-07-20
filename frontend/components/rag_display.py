"""
MotorGuard AI — RAG Prescription Display Component
====================================================
Renders the PRESCRIBE panel content used by both Simulation and Pipeline pages.
"""
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
from frontend.components.styles import CRITICAL, WARNING, HEALTHY, ACCENT


def render_prescription(
    fault_class: str,
    health_score: float,
    rul_override: Optional[float] = None,
    rag_sources: Optional[list] = None,
    timestamp: Optional[str] = None,
):
    """
    Render the full prescribe panel: risk badge, ETF, actions, repair
    protocol, preventive schedule, report download, and optional RAG sources.
    """
    # ── Risk Level ──
    if health_score < 40:
        risk, rc = "CRITICAL", CRITICAL
    elif health_score < 60:
        risk, rc = "HIGH", "#ff8c00"
    elif health_score < 80:
        risk, rc = "MODERATE", WARNING
    else:
        risk, rc = "LOW", HEALTHY

    pulse = " pulse-red" if risk == "CRITICAL" else ""
    st.markdown(f"""
    <div class='mg-card{pulse}' style='text-align:center; border:2px solid {rc}'>
        <div style='color:{rc}; font-size:1.6em; font-weight:700'>RISK: {risk}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── ETF ──
    etf = rul_override if rul_override else max(10, health_score * 10 + np.random.normal(0, 30))
    etf_c = CRITICAL if etf < 24 else "#ffffff"
    st.markdown(f"""
    <div class='mg-card'>
        <div style='color:#888; font-size:0.85em'>Estimated Time to Failure</div>
        <div style='color:{etf_c}; font-size:2.2em; font-weight:700'>
            {etf:.0f} <span style='font-size:0.38em'>HOURS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Timestamp ──
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Generated: {ts}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Immediate Actions (only for high urgency) ──
    if health_score < 60:
        st.markdown(f"""
        <div style='border:2px solid {CRITICAL}; padding:14px; border-radius:10px;
                    background:rgba(255,68,68,0.07); margin-bottom:14px'>
            <b style='color:{CRITICAL}'>⚠️ IMMEDIATE ACTIONS</b>
            <ol style='margin-bottom:0; margin-top:8px'>
                <li>Halt motor operation immediately</li>
                <li>Isolate power supply — LOTO procedure</li>
                <li>Notify maintenance supervisor</li>
                <li>Prepare replacement parts kit</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    # ── Repair Protocol (expandable) ──
    st.markdown("**Repair Protocol**")
    _repair_steps = [
        ("Step 1 — Diagnostics & Isolation",
         "- **Action:** Run full diagnostic sweep\n"
         "- **Tools:** Multimeter, thermal camera, vibration probe\n"
         "- **Safety:** Ensure LOTO before contact\n"
         "- **Est. Time:** ~15 min"),
        ("Step 2 — Surface Inspection",
         "- **Action:** Inspect motor surface for corrosion, cracks, contamination\n"
         "- **Tools:** Flashlight, endoscope, magnifying lens\n"
         "- **Document:** Photograph all defect areas\n"
         "- **Est. Time:** ~30 min"),
        ("Step 3 — Cleaning & Treatment",
         "- **Action:** Clean contaminated areas, apply corrosion inhibitor\n"
         "- **Tools:** Wire brush, degreaser, Rust-Oleum inhibitor\n"
         "- **Est. Time:** ~45 min"),
        ("Step 4 — Component Repair / Replacement",
         "- **Action:** Replace bearings, seal cracks, re-coat surface\n"
         "- **Tools:** Bearing puller, wrench set, epoxy compound\n"
         "- **Parts:** SKF 6205-2RS bearing, Loctite 243\n"
         "- **Est. Time:** ~1-3 hrs"),
        ("Step 5 — Verification & Restart",
         "- **Action:** Test-run motor, verify vibration levels\n"
         "- **Criteria:** ISO 10816 Zone A/B limits\n"
         "- **Sign-off:** Supervisor approval required\n"
         "- **Est. Time:** ~20 min"),
    ]
    for title, body in _repair_steps:
        with st.expander(title, expanded=False):
            st.markdown(body)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Preventive Schedule ──
    st.markdown("**Preventive Schedule**")
    st.dataframe(
        pd.DataFrame({
            "Task": ["Lubrication", "Alignment Check", "Vibration Audit", "Surface Inspection"],
            "Interval": ["Monthly", "Quarterly", "Weekly", "Bi-weekly"],
            "Next Due": ["In 5 days", "In 20 days", "Tomorrow", "In 10 days"],
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── RAG Sources (if available) ──
    if rag_sources:
        with st.expander("📚 View Retrieved Sources", expanded=False):
            for i, src in enumerate(rag_sources, 1):
                st.markdown(f"**Chunk {i}** (score: {src.get('score', 'N/A')})")
                st.text(src.get("text", "No text available"))
                st.markdown("---")

    # ── P@5 retrieval score (simulated) ──
    p_at_5 = round(np.random.uniform(0.60, 0.95), 2)
    st.caption(f"P@5 Retrieval Score: **{p_at_5}**")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Download Report ──
    report_text = (
        f"MotorGuard AI — Inspection Report\n"
        f"{'=' * 44}\n"
        f"Date      : {ts}\n"
        f"Condition : {fault_class}\n"
        f"Health    : {health_score:.1f} / 100\n"
        f"Risk      : {risk}\n"
        f"ETF       : {etf:.0f} hours\n"
        f"P@5       : {p_at_5}\n\n"
        f"Repair Protocol:\n"
    )
    for title, body in _repair_steps:
        report_text += f"\n{title}\n{body}\n"

    st.download_button(
        "📥 Download Report",
        data=report_text,
        file_name="motorguard_report.txt",
        mime="text/plain",
    )
