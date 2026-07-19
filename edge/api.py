"""Simple local API entrypoint for the MotorGuard ODP pipeline."""

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template_string

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge.orchestrator import ODPOrchestrator

app = Flask(__name__)
orchestrator = ODPOrchestrator()

DASHBOARD_HTML = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>MotorGuard Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #020617;
      --panel: rgba(15, 23, 42, 0.95);
      --panel-2: rgba(30, 41, 59, 0.9);
      --line: rgba(148, 163, 184, 0.22);
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --ok: #22c55e;
      --warn: #f59e0b;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, 'Segoe UI', Roboto, Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 20% 10%, rgba(56,189,248,0.18), transparent 22%),
        radial-gradient(circle at 80% 0%, rgba(129,140,248,0.14), transparent 18%),
        linear-gradient(135deg, var(--bg), #0f172a 60%, #111827);
      min-height: 100vh;
    }
    .shell { max-width: 1380px; margin: 0 auto; padding: 2rem; }
    .hero {
      display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap;
      padding: 1.25rem 1.4rem; border: 1px solid var(--line); border-radius: 24px;
      background: linear-gradient(120deg, rgba(15,23,42,0.98), rgba(30,41,59,0.95));
      box-shadow: 0 20px 50px rgba(2,8,23,0.45);
      margin-bottom: 1.1rem;
      position: relative; overflow: hidden;
    }
    .hero::after {
      content: ''; position: absolute; inset: 0; pointer-events: none;
      background: linear-gradient(90deg, transparent, rgba(56,189,248,0.08), transparent);
      transform: translateX(-100%); animation: shimmer 4.5s infinite;
    }
    @keyframes shimmer { 100% { transform: translateX(100%); } }
    .hero h1 { margin: 0 0 0.28rem; font-size: 2rem; }
    .hero p { margin: 0; color: var(--muted); }
    .status-pill {
      display: inline-flex; align-items: center; gap: 0.55rem; padding: 0.7rem 0.95rem;
      border-radius: 999px; background: rgba(34,197,94,0.12); color: #bbf7d0; border: 1px solid rgba(34,197,94,0.2);
      font-weight: 700;
    }
    .dot { width: 0.7rem; height: 0.7rem; border-radius: 50%; background: var(--ok); box-shadow: 0 0 12px var(--ok); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 1rem; }
    .card {
      border: 1px solid var(--line); border-radius: 22px; padding: 1rem 1.1rem; background: var(--panel);
      box-shadow: 0 18px 40px rgba(2,8,23,0.35);
      backdrop-filter: blur(16px);
      position: relative;
      overflow: hidden;
    }
    .card::before {
      content: ''; position: absolute; inset: 0; background: linear-gradient(135deg, rgba(56,189,248,0.05), transparent 40%);
      pointer-events: none;
    }
    .card h2 { margin: 0 0 0.75rem; font-size: 1rem; color: #e2e8f0; letter-spacing: 0.02em; }
    .metric { display: flex; justify-content: space-between; align-items: center; margin: 0.62rem 0; }
    .metric .label { color: var(--muted); }
    .metric .value { font-weight: 700; }
    .bar { width: 100%; height: 0.6rem; background: rgba(148,163,184,0.15); border-radius: 999px; overflow: hidden; margin-top: 0.35rem; }
    .bar > span { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), #60a5fa); border-radius: inherit; }
    .btn { padding: 0.72rem 1rem; border: 0; border-radius: 999px; cursor: pointer; background: linear-gradient(135deg, var(--accent), #2563eb); color: white; font-weight: 700; box-shadow: 0 10px 20px rgba(37,99,235,0.2); }
    .btn.secondary { background: linear-gradient(135deg, #475569, #334155); box-shadow: none; }
    .stack { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.7rem; }
    .pill { border-radius: 999px; padding: 0.42rem 0.68rem; background: rgba(56,189,248,0.16); color: #bae6fd; border: 1px solid rgba(56,189,248,0.25); font-size: 0.9rem; }
    pre { white-space: pre-wrap; background: rgba(2, 8, 23, 0.86); padding: 0.95rem; border-radius: 14px; border: 1px solid var(--line); color: #cbd5e1; overflow: auto; max-height: 320px; margin: 0; }
    .kpi { font-size: 1.7rem; font-weight: 800; margin-top: 0.35rem; }
    .subtle { color: var(--muted); font-size: 0.95rem; }
    .two-col { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 1rem; margin-top: 1rem; }
    .three-col { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; margin-top: 1rem; }
    .gauge {
      width: 120px; height: 120px; border-radius: 50%; display: grid; place-items: center;
      background: conic-gradient(var(--accent) 0%, #334155 0%);
      margin: 0.6rem auto 0.2rem;
      box-shadow: inset 0 0 0 10px rgba(15,23,42,0.8), 0 12px 25px rgba(2,8,23,0.25);
    }
    .gauge .inner {
      width: 78px; height: 78px; border-radius: 50%; background: rgba(2,8,23,0.92);
      display: grid; place-items: center; font-weight: 800; font-size: 1.15rem;
      color: #f8fafc;
    }
    .risk-badge { display: inline-block; padding: 0.35rem 0.6rem; border-radius: 999px; font-weight: 700; font-size: 0.9rem; }
    .risk-high { background: rgba(239,68,68,0.16); color: #fecaca; border: 1px solid rgba(239,68,68,0.25); }
    .risk-medium { background: rgba(245,158,11,0.16); color: #fde68a; border: 1px solid rgba(245,158,11,0.25); }
    .risk-low { background: rgba(34,197,94,0.16); color: #bbf7d0; border: 1px solid rgba(34,197,94,0.25); }
    @media (max-width: 900px) { .two-col { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class=\"shell\">
    <div class=\"hero\">
      <div>
        <h1>MotorGuard AI</h1>
        <p>Industrial predictive maintenance • observe, diagnose, prescribe</p>
      </div>
      <div class=\"status-pill\"><span class=\"dot\"></span> Live monitoring</div>
    </div>

    <div class=\"grid\">
      <div class=\"card\">
        <h2>System status</h2>
        <div class=\"kpi\" id=\"healthValue\">--</div>
        <div class=\"subtle\" id=\"healthDetail\">Loading runtime state...</div>
      </div>
      <div class=\"card\">
        <h2>Runtime config</h2>
        <div class=\"metric\"><span class=\"label\">Hardware mode</span><span class=\"value\" id=\"modeValue\">--</span></div>
        <div class=\"metric\"><span class=\"label\">LLM model</span><span class=\"value\" id=\"modelValue\">--</span></div>
        <div class=\"metric\"><span class=\"label\">Port</span><span class=\"value\" id=\"portValue\">--</span></div>
      </div>
      <div class=\"card\">
        <h2>Fusion confidence</h2>
        <div class=\"gauge\" id=\"gauge\"><div class=\"inner\" id=\"gaugeInner\">0%</div></div>
        <div class=\"metric\" style=\"margin-top: 0.6rem;\"><span class=\"label\">Variance</span><span class=\"value\" id=\"varianceValue\">--</span></div>
      </div>
    </div>

    <div class=\"two-col\">
      <div class=\"card\">
        <h2>Pipeline snapshot</h2>
        <div style=\"display:flex; justify-content:space-between; align-items:center; gap:0.7rem; margin-bottom:0.6rem;\">
          <button class=\"btn\" onclick=\"runOdp()\">Run ODP</button>
          <button class=\"btn secondary\" onclick=\"refreshAll()\">Refresh</button>
        </div>
        <pre id=\"odp\">Waiting for the first run...</pre>
      </div>
      <div class=\"card\">
        <h2>Integrated features</h2>
        <div id=\"features\" class=\"stack\"></div>
        <div style=\"margin-top: 0.95rem;\">
          <div class=\"metric\"><span class=\"label\">Recommended action</span><span class=\"value\" id=\"actionValue\">--</span></div>
          <div class=\"metric\"><span class=\"label\">Risk level</span><span class=\"value\" id=\"riskValue\">--</span></div>
        </div>
      </div>
    </div>

    <div class=\"three-col\">
      <div class=\"card\">
        <h2>Observe</h2>
        <pre id=\"observePanel\">Loading...</pre>
      </div>
      <div class=\"card\">
        <h2>Diagnose</h2>
        <pre id=\"diagnosePanel\">Loading...</pre>
      </div>
      <div class=\"card\">
        <h2>Edge & Camera</h2>
        <pre id=\"edgePanel\">Loading...</pre>
      </div>
    </div>
  </div>

  <script>
    function setRisk(risk) {
      const el = document.getElementById('riskValue');
      const value = (risk || 'Medium').toLowerCase();
      el.className = 'value';
      if (value === 'high') { el.classList.add('risk-badge', 'risk-high'); }
      else if (value === 'low') { el.classList.add('risk-badge', 'risk-low'); }
      else { el.classList.add('risk-badge', 'risk-medium'); }
      el.textContent = risk || 'Medium';
    }

    async function refreshAll() {
      const overview = await fetch('/overview').then(r => r.json());
      document.getElementById('healthValue').textContent = (overview.health.status || 'Online').toUpperCase();
      document.getElementById('healthDetail').textContent = 'Service ready · ' + (overview.health.service || 'MotorGuard');
      document.getElementById('modeValue').textContent = overview.config.hardware_mode || 'simulate';
      document.getElementById('modelValue').textContent = overview.config.llm_model || 'local-fallback';
      document.getElementById('portValue').textContent = overview.config.port || '5001';

      const fusionScore = overview.metrics && overview.metrics.fusion_score != null ? overview.metrics.fusion_score : 0;
      const fusionPercent = Math.max(0, Math.min(100, Math.round(Number(fusionScore) * 100)));
      document.getElementById('gauge').style.background = 'conic-gradient(var(--accent) ' + fusionPercent + '%, #334155 0%)';
      document.getElementById('gaugeInner').textContent = fusionPercent + '%';
      document.getElementById('varianceValue').textContent = overview.metrics && overview.metrics.fusion_variance != null ? overview.metrics.fusion_variance : 'n/a';

      document.getElementById('odp').textContent = JSON.stringify(overview.odp, null, 2);
      document.getElementById('features').innerHTML = overview.features.map(f => '<span class=\"pill\">' + f + '</span>').join('');
      document.getElementById('actionValue').textContent = overview.prescription && overview.prescription.immediate_action ? overview.prescription.immediate_action : 'Monitor and inspect';
      setRisk(overview.metrics && overview.metrics.risk_level ? overview.metrics.risk_level : 'Medium');
      document.getElementById('observePanel').textContent = JSON.stringify(overview.observe, null, 2);
      document.getElementById('diagnosePanel').textContent = JSON.stringify(overview.diagnose, null, 2);
      document.getElementById('edgePanel').textContent = JSON.stringify(overview.edge, null, 2);
    }
    async function runOdp() {
      const data = await fetch('/odp').then(r => r.json());
      document.getElementById('odp').textContent = JSON.stringify(data, null, 2);
      document.getElementById('healthValue').textContent = 'RUN COMPLETE';
      document.getElementById('healthDetail').textContent = 'ODP executed successfully';
    }
    refreshAll();
  </script>
</body>
</html>
"""


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/odp", methods=["GET"])
def odp():
    result = orchestrator.run_once()
    return jsonify(result)


@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template_string(DASHBOARD_HTML)


@app.route("/overview", methods=["GET"])
def overview():
    odp_result = orchestrator.run_once()
    prescription = odp_result.get("prescription", {})
    metrics = {
        "fusion_score": odp_result.get("fusion_score", 0.0),
        "fusion_variance": odp_result.get("fusion_variance", 0.0),
        "risk_level": "High" if (odp_result.get("fusion_score", 0.0) > 0.6) else "Medium",
    }
    return jsonify({
        "health": {"status": "ok", "service": "MotorGuard"},
        "config": {
            "hardware_mode": os.getenv("MOTORGUARD_HARDWARE_MODE", "simulate"),
            "llm_model": os.getenv("MOTORGUARD_LLM_MODEL", "local-fallback"),
            "port": 5001,
        },
        "metrics": metrics,
        "odp": odp_result,
        "observe": {
            "mode": odp_result.get("mode", "simulated"),
            "vibration_features": odp_result.get("vibration_features", []),
            "camera": odp_result.get("camera", {}),
        },
        "diagnose": {
            "fusion_score": odp_result.get("fusion_score", 0.0),
            "fusion_variance": odp_result.get("fusion_variance", 0.0),
            "vision": odp_result.get("vision", {}),
        },
        "edge": {
            "prescription": odp_result.get("prescription", {}),
            "camera_status": "active" if odp_result.get("camera") else "idle",
            "mode": odp_result.get("mode", "simulated"),
        },
        "prescription": {
            "immediate_action": prescription.get("Immediate Action", "Inspect and isolate the motor"),
            "repair_protocol": prescription.get("Repair Protocol", "Inspect bearings and verify alignment"),
            "preventive_schedule": prescription.get("Preventive Schedule", "Recheck within 30 days"),
        },
        "features": [
            "Observe layer",
            "Vision inference",
            "Fusion model",
            "Prescription parser",
            "Raspberry Pi deployment",
            "Local dashboard"
        ],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
