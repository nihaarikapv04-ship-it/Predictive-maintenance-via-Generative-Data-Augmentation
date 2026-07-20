"""
MotorGuard AI — Vibration Plotting Component
==============================================
Plotly charts for vibration data + health gauge + feature cards.
"""
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Any


def create_vibration_plot(
    history: List[Dict[str, float]],
    height: int = 350,
) -> go.Figure:
    """
    6-channel vibration line chart (ax, ay, az, gx, gy, gz).
    """
    channels = ["ax", "ay", "az", "gx", "gy", "gz"]
    colors = ["#ff4444", "#00ff88", "#00d4ff", "#ff8c00", "#9b30ff", "#ffaa00"]

    fig = go.Figure()
    for i, ch in enumerate(channels):
        vals = [d.get(ch, 0) for d in history]
        fig.add_trace(go.Scatter(
            y=vals, mode="lines", name=ch,
            line=dict(color=colors[i], width=1.3),
        ))

    fig.update_layout(
        template="plotly_dark", height=height,
        margin=dict(l=0, r=0, t=28, b=0),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        xaxis_title="Sample", yaxis_title="Amplitude (g / °/s)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,31,46,0.5)",
    )
    return fig


def create_vibration_from_params(
    amplitude: float = 1.0,
    frequency: float = 30.0,
    n_samples: int = 200,
    sample_rate: float = 1500.0,
) -> List[Dict[str, float]]:
    """Generate synthetic 6-channel vibration data from sim parameters."""
    t = np.linspace(0, n_samples / sample_rate, n_samples)
    noise = np.random.normal(0, amplitude * 0.1, (n_samples, 6))
    base = amplitude * np.sin(2 * np.pi * frequency * t)
    data = []
    for i in range(n_samples):
        data.append({
            "ax": float(base[i] + noise[i, 0]),
            "ay": float(0.8 * np.cos(2 * np.pi * frequency * t[i]) + noise[i, 1]),
            "az": float(0.5 * np.sin(2 * np.pi * frequency * 0.5 * t[i]) + noise[i, 2]),
            "gx": float(0.3 * np.cos(2 * np.pi * frequency * 1.5 * t[i]) + noise[i, 3]),
            "gy": float(0.4 * np.sin(2 * np.pi * frequency * 2 * t[i]) + noise[i, 4]),
            "gz": float(0.2 * np.cos(2 * np.pi * frequency * 0.3 * t[i]) + noise[i, 5]),
        })
    return data


def create_health_gauge(score: float, prev: float = None, height: int = 250) -> go.Figure:
    """Health score gauge 0-100 with colored bands."""
    delta = {"reference": prev, "position": "top"} if prev is not None else None
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta" if delta else "gauge+number",
        value=score,
        delta=delta,
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "white"},
            "steps": [
                {"range": [0, 40],  "color": "#ff4444"},
                {"range": [40, 60], "color": "#ff8c00"},
                {"range": [60, 80], "color": "#ffaa00"},
                {"range": [80, 100],"color": "#00ff88"},
            ],
        },
    ))
    fig.update_layout(
        template="plotly_dark", height=height,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def compute_features(amplitude: float, frequency: float, load: float, temp: float) -> Dict[str, Any]:
    """Compute vibration features from simulation parameters."""
    rms = amplitude * 0.707 * (1 + load / 200)
    kurtosis = 3.0 + (amplitude - 1.0) * 1.5 + np.random.normal(0, 0.2)
    crest = amplitude * 1.414 / max(rms, 0.01)
    bpfo_ratio = frequency / 74.6
    bpfi_ratio = frequency / 117.4
    return {
        "RMS (g)": round(rms, 3),
        "Kurtosis": round(kurtosis, 2),
        "Crest Factor": round(crest, 2),
        "BPFO Ratio": round(bpfo_ratio, 3),
        "BPFI Ratio": round(bpfi_ratio, 3),
    }


def health_from_params(amplitude: float, frequency: float, load: float, temp: float) -> float:
    """Compute health score from simulation parameters."""
    base = 95
    base -= min(amplitude * 15, 40)
    base -= min((frequency - 30) * 0.1, 15) if frequency > 30 else 0
    base -= min(load * 0.15, 15)
    base -= max(0, (temp - 60) * 0.3)
    return float(np.clip(base + np.random.normal(0, 1.5), 0, 100))


def urgency_from_health(hs: float):
    """Return (label, color) tuple."""
    if hs < 40:
        return "IMMEDIATE", "#ff4444"
    elif hs < 60:
        return "MONITOR", "#ff8c00"
    elif hs < 80:
        return "ROUTINE", "#ffaa00"
    return "NONE", "#00ff88"
