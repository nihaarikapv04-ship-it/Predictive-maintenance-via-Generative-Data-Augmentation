# ⚙️ MotorGuard AI — Physics-Aware Generative Maintenance with Edge Deployment

![IEEE Accepted Paper](https://img.shields.io/badge/IEEE%20IRAI-Accepted%202026-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.59%2B-ff4b4b.svg)
![Apple Silicon](https://img.shields.io/badge/Hardware%20Accel-Apple%20M--Series%2010--Core-black.svg)
![Raspberry Pi](https://img.shields.io/badge/Edge-Raspberry%20Pi%205%20%2B%20MPU--6050-red.svg)

**MotorGuard AI** is an industrial predictive maintenance system combining **physics-informed neural models**, **edge telemetry**, **real-time computer vision**, and **RAG-driven prescriptive repair guidance**.

---

## 🌟 Key Features & Capabilities

1. **👁️ Observe (Visual Fault Detection)**
   - High-throughput **YOLOv11n** visual classification for 6 motor surface conditions (*Healthy Baseline*, *Mild Oxidation*, *Moderate Corrosion*, *Severe Corrosion*, *Structural Cracking*, *Contamination*).
   - Real-time AVFoundation camera stream with bounding-box annotations and severity indicators.

2. **🩺 Diagnose (Multi-Modal Late Fusion)**
   - 6-axis **MPU-6050 vibration telemetry** (accelerometer + gyroscope) streamed at 1500Hz from Raspberry Pi 5.
   - Physics-aware **TimeGAN + LSTM** fusion model calculating motor health score (0–100) and **Monte Carlo Dropout** uncertainty estimates (±%).

3. **💊 Prescribe (RAG Repair Guidance)**
   - Retrieval-Augmented Generation (RAG) powered by **FAISS** vector store and **Groq Llama-3.3-70B**.
   - Generates exact step-by-step repair protocols based on motor condition and PDF manuals (`docs/motor_manual.pdf`).

4. **📊 Session History & Analytics**
   - Save session runs directly into `data/history.json` with engine names, RPM, health scores, and timestamps.
   - Interactive analytics dashboard showing fault distribution charts and health trend trajectories.

---

## 🚀 Apple Silicon Hardware Acceleration (32 GB RAM, 10-Core CPU/GPU)

To take full advantage of Apple Silicon M-Series processors (10-core CPU/GPU + 32 GB Unified Memory), MotorGuard AI implements:

1. **Streamlit Isolated Micro-Reruns (`@st.fragment`)**
   - Wraps high-FPS live camera streams and Plotly graphs with `@st.fragment(run_every=0.1)`.
   - Isolates updates to the active component without triggering full-page Streamlit re-renders.

2. **Decoupled Background Threaded Camera (`ThreadedCamera`)**
   - Runs a daemon thread worker continuously fetching OpenCV frames via native macOS `CAP_AVFOUNDATION`.
   - Locks capture resolution to **1280x720 @ 30 FPS** with zero UI thread blocking or stuttering.

3. **OpenCV Multi-Core CPU Parallelization**
   - Configures `cv2.setUseOptimized(True)` and `cv2.setNumThreads(8)` to parallelize image matrix ops across Apple Silicon CPU cores.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MacBook (Host)                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │   Streamlit Frontend Dashboard (Port 8501)       │  │
│  │   - Home | Simulation | Pipeline | History        │  │
│  └──────────────────────────┬────────────────────────┘  │
│                             │ HTTP / SSE                │
└─────────────────────────────┼───────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────┐
│                 Raspberry Pi 5 (Edge)                   │
│  ┌───────────────────────────────────────────────────┐  │
│  │   Flask API Backend Server (Port 5000)            │  │
│  │   - Dual-Stack IPv4/IPv6 (host='::')              │  │
│  │   - MPU-6050 6-Axis Sensor I2C Stream            │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start Guide

### Step 1: Raspberry Pi Edge Setup

1. **SSH into your Raspberry Pi**:
   ```bash
   ssh pi@rpi.local
   ```
2. **Clone & Start Dual-Stack Backend Server**:
   ```bash
   git clone https://github.com/nihaarikapv04-ship-it/Predictive-maintenance-via-Generative-Data-Augmentation.git ~/MotorGuard-AI
   cd ~/MotorGuard-AI/backend
   pip3 install flask flask-cors numpy scipy smbus2 --break-system-packages
   python3 app.py
   ```
   *The server will start listening on port `5000` (`http://0.0.0.0:5000` / `http://[::]:5000`).*

---

### Step 2: MacBook Dashboard Launch

1. **Open Terminal on MacBook**:
   ```bash
   cd "/Users/nihaarikapv/Downloads/Predictive-maintenance-via-Generative-Data-Augmentation-main"
   source venv/bin/activate
   PYTHONPATH=. streamlit run frontend/dashboard.py
   ```
2. **Connect to Raspberry Pi**:
   - In the sidebar under **🔌 Raspberry Pi IP**, type `rpi.local` (or your Pi IP address).
   - The indicator will immediately turn **🟢 Connected**.

---

## 🛠️ Project Structure

```
Predictive-maintenance-via-Generative-Data-Augmentation/
├── backend/
│   ├── app.py                 # Dual-stack Flask Edge API
│   ├── observe/               # Vision & MPU-6050 vibration modules
│   ├── diagnose/              # TimeGAN + LSTM late-fusion model
│   └── prescribe/             # FAISS + Groq RAG motor manual engine
├── frontend/
│   ├── dashboard.py           # Main Streamlit shell & router
│   ├── home_page.py           # Home landing page
│   ├── simulation_page.py     # 3-tab Simulation Mode
│   ├── pipeline_page.py       # Live Pipeline Mode with @st.fragment
│   ├── history_page.py        # Analytics & session history log
│   └── components/
│       ├── camera.py          # ThreadedCamera (1280x720 @ 30 FPS)
│       ├── vibration.py       # Plotly dark theme 6-channel charts
│       ├── styles.py          # High-contrast CSS design tokens
│       └── rag_display.py     # Prescriptive RAG markdown UI
├── docs/                      # Motor manuals and PDF documentation
├── data/                      # Session history & FAISS index storage
├── .streamlit/
│   └── config.toml            # Streamlit theme & performance config
└── README.md
```

---

## 🔧 Troubleshooting

- **Camera permissions on macOS**: Ensure **Terminal** or **Python** has camera access enabled under `System Settings → Privacy & Security → Camera`.
- **Pi Disconnected badge**: Verify `python3 app.py` is active on the Pi. If using dual-stack mDNS on macOS, enter `rpi.local` or the numeric IP from `hostname -I`.
- **Button Visibility**: The dashboard uses explicit `#1A1A1A` dark charcoal button text on cyan gradient backgrounds for maximum contrast.

---

## 📜 Citation & License

Acceptance: **IEEE IRAI 2026** — *Physics-Aware Generative Maintenance with Edge Deployment*.
