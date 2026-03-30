from flask import Flask, request, jsonify
from groq import Groq
from pydantic import BaseModel, ValidationError
import json


from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from datetime import datetime

app = Flask(__name__)


client = Groq()

# ── Load PDF + Vector DB (runs once on startup) ───────────────────────────────
loader = PyPDFLoader("docs/motor_manual.pdf")
docs = loader.load_and_split()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(docs, embeddings)

# ── Interface Contract (shared with Pujitha's dashboard) ─────────────────────
class MotorDoctorResponse(BaseModel):
    risk_level: str      # Critical | High | Medium | Low
    etf_hours: int       # Estimated Time to Failure in hours
    repair_action: str   # Specific steps from ABB manual
    safety_protocol: str # Lock-out/Tag-out steps

# ── Core RAG + LLM call ───────────────────────────────────────────────────────
def motor_doctor(issue: str) -> MotorDoctorResponse:
    context_docs = vector_db.similarity_search(issue, k=2)
    context = "\n".join(doc.page_content for doc in context_docs)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        response_format={"type": "json_object"},   # locks model to JSON only
        temperature=0.0,
        max_tokens=300,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are MotorGuard AI, an industrial motor maintenance expert. "
                    "You ONLY output a JSON object with exactly these keys: "
                    "risk_level (one of: Critical, High, Medium, Low), "
                    "etf_hours (integer: estimated hours until failure), "
                    "repair_action (string: specific steps from the ABB manual, max 60 words), "
                    "safety_protocol (string: lockout/tagout steps, max 40 words). "
                    "Use ONLY the provided ABB manual context. No other text."
                )
            },
            {
                "role": "user",
                "content": (
                    f"ABB Manual Context:\n{context}\n\n"
                    f"Detected Issue: {issue}\n\n"
                    "Return the JSON report."
                )
            }
        ]
    )

    raw = response.choices[0].message.content

    # Parse and validate against the schema
    try:
        data = json.loads(raw)
        report = MotorDoctorResponse(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        # Schema validation failed — return a safe fallback
        report = MotorDoctorResponse(
            risk_level="Unknown",
            etf_hours=0,
            repair_action="Parsing failed. Refer to ABB manual immediately.",
            safety_protocol="Apply LOTO procedure. Contact qualified engineer."
        )

    return report

# ── Flask route ───────────────────────────────────────────────────────────────
@app.route("/motor-report", methods=["POST"])
def motor_report():
    data = request.json

    defect     = data.get("defect")
    confidence = data.get("confidence")
    size       = data.get("size")
    location   = data.get("location")

    issue = f"{size}mm {defect} detected at {location} with {confidence * 100:.1f}% confidence"

    report = motor_doctor(issue)

    return jsonify({
        "timestamp": datetime.utcnow().isoformat(),
        "issue": issue,
        "report": report.model_dump()   # clean dict from Pydantic model
    })


if __name__ == "__main__":
    app.run(port=5000)