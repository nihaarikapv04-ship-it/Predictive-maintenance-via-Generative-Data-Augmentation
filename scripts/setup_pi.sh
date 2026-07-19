#!/bin/bash
# MotorGuard AI - Raspberry Pi Setup Script
# Configures the complete edge computing environment for predictive maintenance
# Tested on: Raspberry Pi 4B (4GB+), Raspberry Pi 5
# OS: Raspberry Pi OS (64-bit) Bookworm

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# Banner
echo '================================================================='
echo '  ⚙️  MotorGuard AI - Raspberry Pi Setup'
echo '  Predictive Maintenance Edge Computing System'
echo '================================================================='
echo ''

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    log_error 'This script must be run as root (use sudo)'
    exit 1
fi

# Get the actual user (when run with sudo)
ACTUAL_USER=${SUDO_USER:-$USER}
HOME_DIR=$(eval echo ~$ACTUAL_USER)
PROJECT_DIR="${HOME_DIR}/motorguard-ai"

log_step '1/8: Updating system packages...'
apt-get update && apt-get upgrade -y

log_step '2/8: Installing system dependencies...'
apt-get install -y \
    python3-pip python3-venv python3-dev \
    libopencv-dev python3-opencv \
    libatlas-base-dev libhdf5-dev \
    i2c-tools python3-smbus \
    libffi-dev libssl-dev \
    cmake build-essential \
    git wget curl \
    libgstreamer1.0-dev \
    v4l-utils

log_step '3/8: Enabling I2C interface...'
# Enable I2C if not already enabled
if ! grep -q '^dtparam=i2c_arm=on' /boot/config.txt 2>/dev/null && \
   ! grep -q '^dtparam=i2c_arm=on' /boot/firmware/config.txt 2>/dev/null; then
    # Try new location first (Bookworm), then legacy
    if [ -f /boot/firmware/config.txt ]; then
        echo 'dtparam=i2c_arm=on' >> /boot/firmware/config.txt
        echo 'dtparam=i2c_baudrate=400000' >> /boot/firmware/config.txt
    else
        echo 'dtparam=i2c_arm=on' >> /boot/config.txt
        echo 'dtparam=i2c_baudrate=400000' >> /boot/config.txt
    fi
    log_info 'I2C enabled - reboot required after setup'
else
    log_info 'I2C already enabled'
fi

# Load I2C kernel module
modprobe i2c-dev 2>/dev/null || true
if ! grep -q 'i2c-dev' /etc/modules; then
    echo 'i2c-dev' >> /etc/modules
fi

log_step '4/8: Setting up Python virtual environment...'
su - $ACTUAL_USER -c "
    mkdir -p ${PROJECT_DIR}
    cd ${PROJECT_DIR}
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel
"

log_step '5/8: Installing Python dependencies...'
su - $ACTUAL_USER -c "
    cd ${PROJECT_DIR}
    source venv/bin/activate
    pip install -r requirements_pi.txt
"

log_step '6/8: Configuring camera...'
# Enable legacy camera support if needed
if command -v libcamera-hello &> /dev/null; then
    log_info 'libcamera detected - using modern camera stack'
else
    log_warn 'libcamera not found - enabling legacy camera'
    if [ -f /boot/firmware/config.txt ]; then
        echo 'start_x=1' >> /boot/firmware/config.txt
        echo 'gpu_mem=128' >> /boot/firmware/config.txt
    fi
fi

log_step '7/8: Setting up I2C permissions...'
usermod -aG i2c $ACTUAL_USER 2>/dev/null || true
usermod -aG video $ACTUAL_USER 2>/dev/null || true

log_step '8/8: Creating systemd service...'
cat > /etc/systemd/system/motorguard.service << EOF
[Unit]
Description=MotorGuard AI Predictive Maintenance System
After=network.target

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=${PROJECT_DIR}
Environment=SIMULATION_MODE=false
ExecStart=${PROJECT_DIR}/venv/bin/python -m backend.app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable motorguard.service

log_info 'Setup complete!'
echo ''
echo '================================================================='
echo '  ✅ MotorGuard AI setup is complete!'
echo ''
echo '  Next steps:'
echo '    1. Reboot: sudo reboot'
echo '    2. Verify I2C: i2cdetect -y 1'
echo '    3. Start system: sudo systemctl start motorguard'
echo '    4. View logs: journalctl -u motorguard -f'
echo '    5. Open dashboard: streamlit run frontend/dashboard.py'
echo '================================================================='
