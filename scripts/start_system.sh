#!/bin/bash
# MotorGuard AI - System Start Script
# Starts both backend API and frontend dashboard

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
BACKEND_PORT=${BACKEND_PORT:-5000}
FRONTEND_PORT=${FRONTEND_PORT:-8501}
SIMULATION_MODE=${SIMULATION_MODE:-true}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_DIR="${PROJECT_DIR}/.pids"

# Create PID directory
mkdir -p "$PID_DIR"

# Trap for cleanup
cleanup() {
    echo ''
    echo -e "${YELLOW}[INFO]${NC} Shutting down MotorGuard AI..."
    
    if [ -f "${PID_DIR}/backend.pid" ]; then
        kill $(cat "${PID_DIR}/backend.pid") 2>/dev/null || true
        rm -f "${PID_DIR}/backend.pid"
    fi
    
    if [ -f "${PID_DIR}/frontend.pid" ]; then
        kill $(cat "${PID_DIR}/frontend.pid") 2>/dev/null || true
        rm -f "${PID_DIR}/frontend.pid"
    fi
    
    echo -e "${GREEN}[INFO]${NC} System stopped cleanly."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Banner
echo -e "${CYAN}"
echo '  ⚙️  MotorGuard AI - Predictive Maintenance System'
echo '  ================================================'
echo -e "${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python3 not found. Please install Python 3.9+"
    exit 1
fi

# Activate virtual environment if available
if [ -f "${PROJECT_DIR}/venv/bin/activate" ]; then
    source "${PROJECT_DIR}/venv/bin/activate"
    echo -e "${GREEN}[INFO]${NC} Virtual environment activated"
fi

echo -e "${BLUE}[CONFIG]${NC} Simulation Mode: ${SIMULATION_MODE}"
echo -e "${BLUE}[CONFIG]${NC} Backend Port: ${BACKEND_PORT}"
echo -e "${BLUE}[CONFIG]${NC} Frontend Port: ${FRONTEND_PORT}"
echo ''

# Start Backend
echo -e "${GREEN}[START]${NC} Starting Flask backend on port ${BACKEND_PORT}..."
SIMULATION_MODE=${SIMULATION_MODE} \
PYTHONPATH="${PROJECT_DIR}" \
    python3 -m backend.app &
BACKEND_PID=$!
echo $BACKEND_PID > "${PID_DIR}/backend.pid"
echo -e "${GREEN}[INFO]${NC} Backend PID: ${BACKEND_PID}"

# Wait for backend to start
sleep 3

# Health check
if curl -s "http://localhost:${BACKEND_PORT}/health" > /dev/null 2>&1; then
    echo -e "${GREEN}[✓]${NC} Backend is healthy"
else
    echo -e "${YELLOW}[WARN]${NC} Backend health check failed - it may still be starting up"
fi

# Start Frontend
echo -e "${GREEN}[START]${NC} Starting Streamlit dashboard on port ${FRONTEND_PORT}..."
PYTHONPATH="${PROJECT_DIR}" \
    streamlit run "${PROJECT_DIR}/frontend/dashboard.py" \
    --server.port ${FRONTEND_PORT} \
    --server.headless true \
    --theme.base dark \
    --browser.gatherUsageStats false &
FRONTEND_PID=$!
echo $FRONTEND_PID > "${PID_DIR}/frontend.pid"
echo -e "${GREEN}[INFO]${NC} Frontend PID: ${FRONTEND_PID}"

echo ''
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ✅ MotorGuard AI is running!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ''
echo -e "  Backend API:  ${CYAN}http://localhost:${BACKEND_PORT}${NC}"
echo -e "  Dashboard:    ${CYAN}http://localhost:${FRONTEND_PORT}${NC}"
echo -e "  Health Check: ${CYAN}http://localhost:${BACKEND_PORT}/health${NC}"
echo ''
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Wait for background processes
wait
