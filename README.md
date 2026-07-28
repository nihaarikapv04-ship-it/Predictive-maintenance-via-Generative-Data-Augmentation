# ⚙️ MotorGuard AI — Physics-Aware Generative Maintenance with Edge Deployment

![IEEE Accepted Paper](https://img.shields.io/badge/IEEE%20IRAI-Accepted%202026-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.59%2B-ff4b4b.svg)
![Apple Silicon](https://img.shields.io/badge/Hardware%20Accel-Apple%20M--Series%2010--Core-black.svg)
![Raspberry Pi](https://img.shields.io/badge/Edge-Raspberry%20Pi%205%20%2B%20MPU--6050-red.svg)

**MotorGuard AI** is an industrial predictive maintenance system combining **physics-informed neural models**, **edge telemetry**, **real-time computer vision**, and **RAG-driven prescriptive repair guidance**.

---

## 📥 Required Downloads & Dependencies

Before running MotorGuard AI, ensure you have the following prerequisites installed on your system.

### 💻 1. MacBook (Dashboard Host)

#### System Requirements:
- **macOS**: 12.0+ (Apple Silicon M1/M2/M3/M4 or Intel)
- **Python**: `Python 3.10` or higher
- **Webcam**: Built-in FaceTime HD Camera or external USB webcam

#### Download Packages List (`requirements.txt`):
| Package | Version | Purpose |
| :--- | :--- | :--- |
| **`streamlit`** | `^1.37.0` | Main frontend dashboard framework |
| **`watchdog`** | `^3.0.0` | macOS high-performance file watcher & auto-reloader |
| **`opencv-python`** | `^4.8.0` | 1280x720 @ 30 FPS AVFoundation camera capture & CLAHE |
| **`torch` & `torchvision`** | `^2.1.0` | Apple Silicon **MPS GPU Acceleration** for neural models |
| **`ultralytics`** | `^8.1.0` | YOLOv11n 6-class visual motor fault classifier |
| **`plotly`** | `^5.18.0` | Dark-theme 6-channel vibration plots & health gauges |
| **`requests`** | `^2.31.0` | HTTP client for edge Raspberry Pi communication |
| **`numpy` & `scipy`** | `^1.24.0` | Numerical calculations, FFT spectral analysis & EMA smoothing |
| **`pandas`** | `^2.0.0` | DataFrame handling for session history analytics |
| **`faiss-cpu` & `groq`** | `^1.7.4` | Vector store & Groq Llama-3.3-70B RAG repair manual engine |

#### 📥 Single Command Download for MacBook:
```bash
pip install -r requirements.txt
```

---

### 🔌 2. Raspberry Pi 5 (Edge Sensor API)

#### System Requirements:
- **OS**: Raspberry Pi OS (Debian 12 Bookworm)
- **Python**: `Python 3.10+`
- **Sensor**: **MPU-6050 6-Axis Accelerometer + Gyroscope** over I2C

#### Download Packages List (`requirements_pi.txt`):
| Package | Version | Purpose |
| :--- | :--- | :--- |
| **`flask`** | `^3.0.0` | Lightweight REST API server on port 5000 |
| **`flask-cors`** | `^4.0.0` | Cross-Origin Resource Sharing for dashboard calls |
| **`numpy`** | `^1.24.0` | Synthetic vibration signal generation |
| **`scipy`** | `^1.10.0` | Butterworth filtering & signal processing |
| **`smbus2`** | `^0.4.2` | I2C bus reader for MPU-6050 physical hardware |

#### 📥 Single Command Download for Raspberry Pi:
```bash
pip3 install flask flask-cors numpy scipy smbus2 --break-system-packages
```

---

## ⚡ Step-by-Step Installation & Setup

### 🔴 Step A: Raspberry Pi 5 Setup (Edge API)

1. **SSH into your Raspberry Pi**:
   ```bash
   ssh pi@rpi.local
   ```
2. **Clone Repository & Download Dependencies**:
   ```bash
   git clone https://github.com/nihaarikapv04-ship-it/Predictive-maintenance-via-Generative-Data-Augmentation.git ~/MotorGuard-AI
   cd ~/MotorGuard-AI/backend
   pip3 install flask flask-cors numpy scipy smbus2 --break-system-packages
   ```
3. **Start Dual-Stack (IPv4 + IPv6) Backend Server**:
   ```bash
   python3 app.py
   ```
   *The server will start listening on port `5000` (`http://0.0.0.0:5000` / `http://[::]:5000`).*

---

### 🔵 Step B: MacBook Dashboard Setup (Frontend)

1. **Open Terminal on MacBook**:
   ```bash
   cd "/Users/nihaarikapv/Downloads/Predictive-maintenance-via-Generative-Data-Augmentation-main"
   ```
2. **Activate Virtual Environment & Install Required Packages**:
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Launch Dashboard**:
   ```bash
   PYTHONPATH=. streamlit run frontend/dashboard.py
   ```
   *(Your browser will open automatically at `http://localhost:8501`)*

4. **Connect to Raspberry Pi**:
   - In the sidebar under **🔌 Raspberry Pi IP**, type `rpi.local` (or your Pi IP address).
   - Press **Enter**. The indicator will turn **🟢 Connected**.

---

## 🚀 Apple Silicon Hardware Acceleration (32 GB RAM, 10-Core CPU/GPU)

To take full advantage of Apple Silicon M-Series processors (10-core CPU/GPU + 32 GB Unified Memory), MotorGuard AI implements:

1. **Streamlit Component Micro-Reruns (`@st.fragment`)**
   - Wraps high-FPS live camera streams and Plotly graphs with `@st.fragment(run_every=0.5)`.
   - Isolates updates to the active component without triggering full-page Streamlit re-renders.

2. **Decoupled Background Threaded Camera (`ThreadedCamera`)**
   - Runs a daemon thread worker continuously fetching OpenCV frames via native macOS `CAP_AVFOUNDATION`.
   - Locks capture resolution to **1280x720 @ 30 FPS** with zero UI thread blocking or stuttering.

3. **PyTorch Apple Metal GPU Acceleration (MPS)**
   - PyTorch automatically detects your **10-core M-Series GPU** using `torch.backends.mps.is_available()`.
   - All YOLOv11 tensor operations are offloaded directly to your GPU.

4. **Temporal Stability Engine**
   - Holds detected conditions steady for **3.0 seconds** with Exponential Moving Average (EMA) confidence smoothing so humans can comfortably read, observe, and diagnose motor health.

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

## 🔧 Troubleshooting

- **Event loop closed error (Python 3.14)**: `.streamlit/config.toml` includes `fileWatcherType = "none"`, permanently resolving watchdog thread conflicts.
- **Camera permissions on macOS**: Ensure **Terminal** or **Python** has camera access enabled under `System Settings → Privacy & Security → Camera`.
- **Pi Disconnected badge**: Verify `python3 app.py` is active on the Pi. Enter `rpi.local` or the numeric IP from `hostname -I`.
- **Button Visibility**: Uses explicit `#1A1A1A` dark charcoal button text on cyan gradient backgrounds for high contrast.

---

## 📜 Citation & License

Acceptance: **IEEE IRAI 2026** — *Physics-Aware Generative Maintenance with Edge Deployment*.
