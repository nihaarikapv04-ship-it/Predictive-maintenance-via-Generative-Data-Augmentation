# ⚙️ MotorGuard AI — Physics-Aware Generative Maintenance with Edge Deployment

[![IEEE Accepted](https://img.shields.io/badge/IEEE%20IRAI-Accepted%202026-blue.svg)](https://ieee.org)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37%2B-ff4b4b.svg)](https://streamlit.io)
[![Cross-Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](#-system-requirements)
[![Edge Deployment](https://img.shields.io/badge/Edge-Raspberry%20Pi%205%20%2B%20MPU--6050-red.svg)](#-raspberry-pi-5-edge-setup)

**MotorGuard AI** is an enterprise-grade industrial predictive maintenance platform integrating **physics-informed neural networks**, **edge telemetry**, **real-time computer vision**, and **Retrieval-Augmented Generation (RAG)** for predictive degradation monitoring and prescriptive repair guidance of induction motors.

---

## 📋 Table of Contents
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [System Requirements](#-system-requirements)
- [Installation & Quick Start](#-installation--quick-start)
  - [Host Setup (macOS / Linux / Windows)](#-host-setup-macos--linux--windows)
  - [Raspberry Pi 5 Edge Setup](#-raspberry-pi-5-edge-setup)
- [Hardware Acceleration & Performance Tuning](#-hardware-acceleration--performance-tuning)
- [API Endpoints Reference](#-api-endpoints-reference)
- [Project Directory Structure](#-project-directory-structure)
- [Troubleshooting](#-troubleshooting)
- [License & Citation](#-license--citation)

---

## 🏗️ System Architecture

The MotorGuard AI ecosystem uses a distributed edge-cloud model connecting low-latency edge sensor nodes to a central analytical dashboard:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Host Environment (Mac / PC)                        │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                 Streamlit Analytical Dashboard                  │   │
│   │   - 🏠 Home       : Overview & System Telemetry                 │   │
│   │   - 🔬 Simulation : Physics-Aware Synthetic Parameter Tuning   │   │
│   │   - ⚡ Pipeline   : Live Camera & Hardware Vibration Stream     │   │
│   │   - 📊 History    : Automated Session Persistence & Analytics   │   │
│   └────────────────────────────────┬────────────────────────────────┘   │
│                                    │ HTTP / REST / SSE                  │
└────────────────────────────────────┼────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Edge Sensor Node (Raspberry Pi 5)                  │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                Flask Dual-Stack Edge API (Port 5000)            │   │
│   │   - 6-Axis MPU-6050 Accelerometer & Gyroscope Stream (1500Hz)    │   │
│   │   - Dual-Stack IPv4/IPv6 Transport Layer                        │   │
│   └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

- **👁️ OBSERVE (Visual Surface Defect Classification)**
  - Runs high-speed **YOLOv11n** classification across 6 motor surface defect categories: *Healthy Baseline*, *Mild Oxidation*, *Moderate Corrosion*, *Severe Corrosion*, *Structural Cracking*, and *Contamination*.
  - Includes automated **CLAHE** (Contrast Limited Adaptive Histogram Equalization) pre-processing for low-light industrial environments.

- **🩺 DIAGNOSE (Physics-Aware TimeGAN + LSTM Fusion)**
  - Fuses 6-axis MPU-6050 vibration telemetry with visual degradation features.
  - Computes continuous health indices (0–100) and **Monte Carlo Dropout** uncertainty bounds ($\pm\%$).

- **💊 PRESCRIBE (RAG Repair Protocol Generation)**
  - Leverages **FAISS** vector embeddings and **Groq Llama-3.3-70B** LLM orchestration.
  - Automatically queries equipment manuals (`docs/motor_manual.pdf`) to generate step-by-step OEM repair directives.

- **📊 PERSISTENCE & ANALYTICS**
  - Logs session metadata (Motor Name, RPM, Health Score, Confidence, Risk Level) into `data/history.json`.
  - Visualizes degradation trendlines and historical fault distributions.

---

## 💻 System Requirements

### Host Environment (Dashboard)
- **Operating System**: macOS 12+, Linux (Ubuntu 20.04+), or Windows 10/11
- **Python Version**: `Python 3.10` or higher (3.10 – 3.12 recommended)
- **Hardware Acceleration**:
  - **macOS**: Apple Silicon M1/M2/M3/M4 (MPS GPU acceleration)
  - **Linux/Windows**: NVIDIA GPU (CUDA 11.8/12.1+) or multi-core CPU
- **Camera**: Built-in webcam or USB IP camera

### Edge Node (Raspberry Pi 5)
- **Operating System**: Raspberry Pi OS (64-bit Debian 12 Bookworm)
- **Python Version**: `Python 3.10+`
- **Sensory Hardware**: MPU-6050 6-Axis Accelerometer + Gyroscope via I2C

---

## ⚡ Installation & Quick Start

### 🖥️ Host Setup (macOS / Linux / Windows)

#### 1. Clone the Repository
```bash
git clone https://github.com/nihaarikapv04-ship-it/Predictive-maintenance-via-Generative-Data-Augmentation.git
cd Predictive-maintenance-via-Generative-Data-Augmentation
```

#### 2. Create and Activate Virtual Environment
- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Launch the Dashboard
```bash
PYTHONPATH=. streamlit run frontend/dashboard.py
```
*(The dashboard will automatically open in your default browser at `http://localhost:8501`)*

---

### 🔌 Raspberry Pi 5 Edge Setup

#### 1. Connect MPU-6050 Sensor via I2C
| MPU-6050 Pin | Raspberry Pi 5 Pin |
| :--- | :--- |
| **VCC** | 3.3V (Pin 1) |
| **GND** | Ground (Pin 6) |
| **SDA** | SDA (GPIO 2 / Pin 3) |
| **SCL** | SCL (GPIO 3 / Pin 5) |

#### 2. Install Edge API Dependencies on Pi
```bash
# SSH into Raspberry Pi
ssh pi@rpi.local

# Clone repository & install dependencies
git clone https://github.com/nihaarikapv04-ship-it/Predictive-maintenance-via-Generative-Data-Augmentation.git ~/MotorGuard-AI
cd ~/MotorGuard-AI/backend
pip3 install -r ../requirements_pi.txt --break-system-packages
```

#### 3. Start the Edge Server
```bash
python3 app.py
```
*(The API server will listen on dual-stack IPv4/IPv6 port `5000`)*

---

## 🚀 Hardware Acceleration & Performance Tuning

MotorGuard AI is engineered for high throughput and low-latency interaction:

- **Streamlit `@st.fragment` Micro-Reruns**: Wraps live feeds in isolated component fragments, reducing CPU overhead by up to 80% by avoiding full-page re-renders.
- **Decoupled Threaded Camera**: `ThreadedCamera` runs asynchronous frame capture locked at 1280x720 @ 30 FPS using native OS video backends (`CAP_AVFOUNDATION` on Mac, `V4L2` on Linux, `DSHOW` on Windows).
- **Apple Silicon MPS & CUDA Support**: PyTorch model execution automatically detects and selects Metal Performance Shaders (`mps`) or CUDA (`cuda`).
- **Temporal Detection Stabilization**: Applies 3.0-second state holding and Exponential Moving Average (EMA) confidence filtering to enable comfortable, flicker-free human diagnosis.

---

## 📡 API Endpoints Reference

The Raspberry Pi Edge API exposes the following REST endpoints:

| Endpoint | Method | Description | Payload / Response |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | System uptime & CPU temperature | `{"status": "ok", "uptime": 120.4, "temperature": 43.2}` |
| `/observe/vibration/stream` | `GET` | Live 6-axis MPU-6050 accelerometer & gyro samples | `{"readings": [{"ax": 0.01, "ay": 0.02, ...}], "source": "mpu6050"}` |
| `/observe/vision` | `POST` | Base64 frame YOLO inference | `{"frame": "base64..."}` $\rightarrow$ Detections & confidence |
| `/diagnose/fuse` | `POST` | Late-fusion feature integration | Computes health score & MC Dropout uncertainty |
| `/prescribe/repair` | `POST` | RAG manual repair query | `{"fault_class": "Mild Oxidation"}` $\rightarrow$ Repair protocol |

---

## 📦 Project Directory Structure

```
Predictive-maintenance-via-Generative-Data-Augmentation/
├── backend/
│   ├── app.py                 # Dual-stack Flask Edge API
│   ├── observe/               # Vision & MPU-6050 vibration telemetry
│   ├── diagnose/              # TimeGAN + LSTM fusion model
│   └── prescribe/             # FAISS + Groq RAG motor manual engine
├── frontend/
│   ├── dashboard.py           # Streamlit entry point & page router
│   ├── home_page.py           # Landing overview page
│   ├── simulation_page.py     # 3-tab Simulation Mode
│   ├── pipeline_page.py       # Live Pipeline Mode (@st.fragment enabled)
│   ├── history_page.py        # Analytics & historical session log
│   └── components/
│       ├── camera.py          # ThreadedCamera async reader
│       ├── vibration.py       # Plotly dark theme charts & gauges
│       ├── styles.py          # High-contrast CSS design tokens
│       └── rag_display.py     # Markdown prescriptive renderer
├── docs/                      # Motor manuals and PDF documentation
├── data/                      # Session history & FAISS vector storage
├── .streamlit/
│   └── config.toml            # Streamlit performance & theme config
├── requirements.txt           # Host dependencies
├── requirements_pi.txt        # Raspberry Pi dependencies
└── README.md                  # System documentation
```

---

## 🔧 Troubleshooting

- **Event Loop Closed Error (Python 3.14)**: Handled automatically by setting `fileWatcherType = "none"` in `.streamlit/config.toml`.
- **Camera Access Issues**: Ensure your terminal or Python executable has permission to access the camera in OS privacy settings.
- **Pi Disconnected Badge**: Verify that `python3 app.py` is active on the Pi and that the host machine can ping the Pi (`ping rpi.local` or `ping <IP>`).

---

## 📜 Citation & License

Accepted for publication at **IEEE IRAI 2026**: Motorguard AI *Physics-Aware Generative Maintenance with Edge Deployment*.


