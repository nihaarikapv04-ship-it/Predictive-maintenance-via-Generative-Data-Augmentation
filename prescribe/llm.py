"""Local llama.cpp execution wrapper for on-device prescriptions."""

import os
import subprocess


def run_local_llm(prompt: str, model_path: str):
    """Placeholder for local llama.cpp inference."""
    if not model_path or not os.path.exists(model_path):
        return "Immediate Action: Isolate the motor\nRepair Protocol: Inspect bearings\nPreventive Schedule: Recheck in 30 days"

    try:
        completed = subprocess.run([
            "llama-cli",
            "-m",
            model_path,
            "-p",
            prompt,
            "-n",
            "160",
        ], capture_output=True, text=True, timeout=30)
        return completed.stdout.strip() or "Immediate Action: Isolate the motor\nRepair Protocol: Inspect bearings\nPreventive Schedule: Recheck in 30 days"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "Immediate Action: Isolate the motor\nRepair Protocol: Inspect bearings\nPreventive Schedule: Recheck in 30 days"
