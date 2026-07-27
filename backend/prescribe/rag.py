import os
import logging
import json
import uuid
import datetime
import numpy as np
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Resolve PDF_PATH
PDF_PATH = os.path.join(os.path.dirname(__file__), '../../docs/motor_manual.pdf')
if not os.path.exists(PDF_PATH):
    for alt in ['docs/motor_manual.pdf', '../docs/motor_manual.pdf', 'docs/motor_manual.pdf']:
        if os.path.exists(alt):
            PDF_PATH = alt
            break

FAISS_CACHE_PATH = os.path.join(os.path.dirname(__file__), '../../data/faiss_index')

# Check imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from langchain_community.document_loaders import PyPDFLoader
    LANGCHAIN_PDF_AVAILABLE = True
except ImportError:
    LANGCHAIN_PDF_AVAILABLE = False

class RepairKnowledgeBase:
    """Stores and manages repair protocols loaded from motor_manual.pdf or fallback docs."""

    def __init__(self):
        self.documents = self._load_documents()

    def _load_documents(self) -> List[Dict[str, Any]]:
        docs = []
        if os.path.exists(PDF_PATH) and LANGCHAIN_PDF_AVAILABLE:
            try:
                loader = PyPDFLoader(PDF_PATH)
                pdf_pages = loader.load_and_split()
                for idx, page in enumerate(pdf_pages[:30]): # First 30 pages
                    docs.append({
                        "id": f"pdf_page_{idx+1}",
                        "title": f"Motor Manual - Page {idx+1}",
                        "content": page.page_content,
                        "category": "Motor Manual",
                        "severity_level": "MODERATE",
                        "estimated_time_hours": 2.0
                    })
                logger.info(f"Loaded {len(docs)} pages from {PDF_PATH}")
                return docs
            except Exception as e:
                logger.error(f"Error loading PDF via PyPDFLoader: {e}")

        # Fallback knowledge base if PDF or PyPDFLoader fails
        return [
            {
                "id": "doc_01",
                "title": "Healthy Baseline Maintenance",
                "content": "Perform routine visual inspection and vibration check. Ensure cooling fins are clear of dust and debris.",
                "category": "Routine",
                "severity_level": "LOW",
                "estimated_time_hours": 0.5
            },
            {
                "id": "doc_02",
                "title": "Mild Oxidation Treatment",
                "content": "Isolate power. Clean surface with wire brush and solvent. Apply rust converter and anti-corrosive primer coating.",
                "category": "Surface",
                "severity_level": "LOW",
                "estimated_time_hours": 1.5
            },
            {
                "id": "doc_03",
                "title": "Moderate Corrosion Repair",
                "content": "Scrape away corroded layers. Treat housing with zinc-rich primer. Check bearing seals for ingress of corrosive contaminants.",
                "category": "Surface",
                "severity_level": "MODERATE",
                "estimated_time_hours": 2.5
            },
            {
                "id": "doc_04",
                "title": "Severe Corrosion Emergency Repair",
                "content": "Perform LOTO. Inspect structural integrity of motor frame. If frame wall thickness is reduced by >15%, replace housing.",
                "category": "Structural",
                "severity_level": "CRITICAL",
                "estimated_time_hours": 6.0
            },
            {
                "id": "doc_05",
                "title": "Structural Cracking Repair Protocol",
                "content": "Immediate shutdown required. Apply dye penetrant test to check crack depth. Grind out crack and perform stitch welding or epoxy resin injection.",
                "category": "Structural",
                "severity_level": "CRITICAL",
                "estimated_time_hours": 8.0
            },
            {
                "id": "doc_06",
                "title": "Contamination Decontamination",
                "content": "Degrease outer casing. Replace oily seals and gaskets. Flush bearing housing with clean mineral oil and repack with high-temp grease.",
                "category": "Cleaning",
                "severity_level": "MODERATE",
                "estimated_time_hours": 2.0
            }
        ]

class RulePrescriber:
    """Fallback prescriber using rules."""

    def generate_prescription(self, fault_class: str, health_score: float) -> dict:
        fc = fault_class.lower()
        if health_score < 40 or "severe" in fc or "crack" in fc:
            risk = "CRITICAL"
            etf = max(5, int(health_score * 0.4))
        elif health_score < 60 or "moderate" in fc or "contam" in fc:
            risk = "HIGH"
            etf = max(24, int(health_score * 1.5))
        elif health_score < 80 or "mild" in fc:
            risk = "MODERATE"
            etf = max(72, int(health_score * 5))
        else:
            risk = "LOW"
            etf = max(200, int(health_score * 10))

        immediate_actions = []
        if risk in ["CRITICAL", "HIGH"]:
            immediate_actions = [
                "Halt motor operation immediately",
                "Isolate power supply (LOTO procedure)",
                "Notify maintenance supervisor",
                "Prepare replacement parts kit"
            ]

        protocol = [
            {"step_number": 1, "action": "Diagnostics & Isolation", "details": f"Run full diagnostic sweep for {fault_class}.", "tools": ["Multimeter", "Thermal camera"], "time": "15m"},
            {"step_number": 2, "action": "Surface & Component Inspection", "details": "Inspect motor surface and bearings.", "tools": ["Endoscope", "Flashlight"], "time": "30m"},
            {"step_number": 3, "action": "Repair & Treatment", "details": f"Remediate {fault_class} according to motor manual.", "tools": ["Wire brush", "Epoxy compound", "Wrench set"], "time": "1-3 hrs"},
        ]

        return {
            "fault_class": fault_class,
            "health_score": health_score,
            "risk_level": risk,
            "estimated_time_to_failure_hours": etf,
            "immediate_actions": immediate_actions,
            "repair_protocol": protocol,
            "preventive_schedule": [
                {"task": "Lubrication", "interval": "Monthly", "next_due": "In 5 days"},
                {"task": "Alignment Check", "interval": "Quarterly", "next_due": "In 20 days"},
                {"task": "Vibration Audit", "interval": "Weekly", "next_due": "Tomorrow"}
            ]
        }

def get_repair_protocol(fault_class: str = "Healthy Baseline", health_score: float = 85.0) -> dict:
    """Public wrapper to get repair protocol."""
    # Attempt Groq API if available
    api_key = os.environ.get("GROQ_API_KEY")
    if GROQ_AVAILABLE and api_key:
        try:
            client = Groq(api_key=api_key)
            prompt = f"Provide a brief maintenance recommendation for an induction motor with fault condition '{fault_class}' and health score {health_score}/100."
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            llm_text = response.choices[0].message.content
            rx = RulePrescriber().generate_prescription(fault_class, health_score)
            rx["llm_guidance"] = llm_text
            return rx
        except Exception as e:
            logger.warning(f"Groq API call failed: {e}")

    # Default RulePrescriber
    return RulePrescriber().generate_prescription(fault_class, health_score)
