<div align="center">

# ⚙️ MotorGuard AI

### Predictive Maintenance via Generative Data Augmentation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-RPi%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()

**Real-time motor health monitoring combining Computer Vision, Vibration Analysis, Deep Learning Diagnostics, and AI-powered Repair Prescriptions.**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Dashboard](#-dashboard) • [API Reference](#-api-reference)

</div>

---

## 🌟 Features

| Module | Technology | Description |
|--------|-----------|-------------|
| **🔍 Observe** | YOLOv11n + CLAHE | Real-time visual fault detection with low-light enhancement |
| **📈 Observe** | MPU-6050 @ 1500Hz | 6-axis vibration sensing with Butterworth bandpass filtering |
| **🩺 Diagnose** | Late-Fusion LSTM | Multi-modal health scoring with Monte Carlo Dropout uncertainty |
| **📝 Prescribe** | FAISS + Llama-3-8B | RAG-powered repair protocol generation with PDF reports |
| **📊 Dashboard** | Streamlit + Plotly | Real-time dark-theme dashboard with 3-panel layout |
| **🤖 Simulation** | CWRU-style Data | Full system demo without hardware using synthetic data |

## 🏗️ Architecture

```
MotorGuard AI
├── OBSERVE Layer
│   ├── Vision: YOLOv11n webcam inference + CLAHE preprocessing
│   └── Vibration: MPU-6050 I2C @ 1500Hz + Butterworth filter
├── DIAGNOSE Layer  
│   └── Late-Fusion LSTM with Monte Carlo Dropout
│       ├── Health Score (0-100)
│       ├── Fault Classification (5 classes)
│       └── Remaining Useful Life (hours)
├── PRESCRIBE Layer
│   └── FAISS retrieval + Llama-3-8B 4-bit generation
│       ├── Risk Assessment
│       ├── Repair Protocols
│       └── Preventive Schedules
└── DASHBOARD
    └── Streamlit real-time 3-panel interface
```

### System Flow

```
Webcam ────────────────┐
  │ [CLAHE + YOLOv11n]   │
  └────▶ Vision Features ─┐
                         ├─▶ Late-Fusion LSTM ─▶ RAG Prescriber ─▶ Dashboard
MPU-6050 ─────────────┐   │     (MC Dropout)       (FAISS+LLM)
  │ [Butterworth BPF]    │   │
  └────▶ Vib Features ──┘
```

## 📁 Project Structure

```
├── backend/
│   ├── app.py                 # Flask API server (6 endpoints)
│   ├── observe/
│   │   ├── vision.py          # YOLOv11n + CLAHE visual inspection
│   │   └── vibration.py       # MPU-6050 vibration sensing + DSP
│   ├── diagnose/
│   │   └── fusion.py          # Late-Fusion LSTM + MC Dropout
│   └── prescribe/
│       └── rag.py             # FAISS + Llama-3-8B RAG pipeline
├── frontend/
│   └── dashboard.py           # Streamlit dark-theme dashboard
├── scripts/
│   ├── setup_pi.sh            # Raspberry Pi setup automation
│   └── start_system.sh        # System startup script
├── requirements_pi.txt        # Raspberry Pi dependencies
├── requirements_mac.txt       # macOS development dependencies
└── README.md
```

## 🚀 Quick Start

### Option 1: Simulation Mode (No Hardware Required)

```bash
# Clone the repository
git clone https://github.com/nihaarikapv04-ship-it/Predictive-maintenance-via-Generative-Data-Augmentation.git
cd Predictive-maintenance-via-Generative-Data-Augmentation

# Install dependencies
pip install -r requirements_mac.txt

# Start the system in simulation mode
bash scripts/start_system.sh

# Or start components individually:
# Terminal 1 - Backend
SIMULATION_MODE=true python -m backend.app

# Terminal 2 - Dashboard
streamlit run frontend/dashboard.py
```

### Option 2: Raspberry Pi Deployment

```bash
# Run the setup script (requires sudo)
sudo bash scripts/setup_pi.sh

# Reboot to enable I2C
sudo reboot

# Verify MPU-6050 connection
i2cdetect -y 1  # Should show device at address 0x68

# Start the system
SIMULATION_MODE=false bash scripts/start_system.sh
```

## 📊 Dashboard

The dashboard features a professional dark-theme interface with three panels:

| Panel | Function | Key Metrics |
|-------|----------|-------------|
| 🔍 **OBSERVE** | Visual Inspection | Fault class, confidence, bounding boxes |
| 🩺 **DIAGNOSE** | Health Analytics | Health score, vibration graph, uncertainty |
| 📝 **PRESCRIBE** | Repair Protocol | Risk level, ETF, repair steps, schedule |

Features:
- 🌙 Dark theme with glassmorphism design
- 📱 Responsive 3-column layout
- 📈 Real-time Plotly charts
- ⏱️ Pipeline latency monitoring
- 🔄 Start/Stop pipeline toggle
- 📥 PDF report download
- ⚠️ Simulation mode banner

## 🔌 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `POST` | `/observe/vision` | Run visual fault detection |
| `GET` | `/observe/vibration/stream` | Stream vibration data (SSE) |
| `POST` | `/diagnose/fuse` | Run diagnostic fusion |
| `POST` | `/prescribe/repair` | Generate repair prescription |
| `POST` | `/pipeline/run` | Run full pipeline |

### Example: Full Pipeline

```bash
curl -X POST http://localhost:5000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"simulation": true}'
```

Response:
```json
{
  "status": "ok",
  "data": {
    "observe": {
      "vision": {"fault_class": "Inner Race Fault", "confidence": 0.87},
      "vibration": {"severity": "moderate", "rms": 2.34}
    },
    "diagnose": {
      "health_score": 62.5,
      "uncertainty": 4.2,
      "maintenance_urgency": "soon",
      "rul_hours": 156.3
    },
    "prescribe": {
      "risk_level": "MODERATE",
      "etf_hours": 156.3,
      "immediate_actions": ["Schedule bearing inspection within 48 hours"],
      "repair_steps": 8
    }
  },
  "latency_ms": 245.6
}
```

## 🧠 Fault Classes

| Class | Description | Color Code |
|-------|-------------|------------|
| Normal | Healthy motor operation | 🟢 Green |
| Inner Race Fault | Bearing inner race damage | 🔴 Red |
| Outer Race Fault | Bearing outer race damage | 🟠 Orange |
| Ball Fault | Rolling element damage | 🟣 Purple |
| Misalignment | Shaft misalignment | 🟣 Magenta |

## 🛠️ Technology Stack

- **Edge Computing**: Raspberry Pi 4B/5 with MPU-6050 accelerometer
- **Computer Vision**: YOLOv11n (Ultralytics) with CLAHE preprocessing
- **Signal Processing**: SciPy Butterworth bandpass filter (10-500Hz)
- **Deep Learning**: PyTorch Late-Fusion LSTM with Monte Carlo Dropout
- **Knowledge Retrieval**: FAISS vector store with sentence-transformers
- **Language Model**: Llama-3-8B-Instruct with 4-bit quantization
- **Backend**: Flask REST API with SSE streaming
- **Frontend**: Streamlit with Plotly dark-theme visualizations
- **Standards**: ISO 10816 vibration severity classification

## 🧪 Simulation Mode

The system includes a comprehensive simulation mode that generates:
- **CWRU-style vibration data** with realistic fault signatures (BPFI, BPFO, BSF frequencies)
- **Synthetic YOLO detections** with plausible confidence distributions
- **Fluctuating health scores** that demonstrate degradation patterns
- **Full RAG pipeline output** using rule-based fallback

This allows complete system testing without any hardware.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [CWRU Bearing Dataset](https://engineering.case.edu/bearingdatacenter) - Vibration data patterns
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection framework
- [Meta Llama 3](https://llama.meta.com/) - Language model
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search

---

<div align="center">
  <strong>Built with ❤️ for Industrial IoT</strong>
</div>
