#!/usr/bin/env bash
# Run MotorGuard RAG service
# Usage: ./scripts/run_rag_service.sh [port]
PORT=${1:-5000}
if [ -z "$GROQ_API_KEY" ]; then
  echo "Warning: GROQ_API_KEY not set. The service will run but AI calls will be disabled."
fi
.venv/bin/python3 rag.py
