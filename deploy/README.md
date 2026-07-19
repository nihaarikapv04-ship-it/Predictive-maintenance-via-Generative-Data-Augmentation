# Raspberry Pi deployment notes

## Environment variables
- MOTORGUARD_HARDWARE_MODE=simulate|real
- MOTORGUARD_LLM_MODEL=/path/to/llama-3-8b-q4.gguf

## Run locally
```bash
cd /home/pi/MotorGuardAI
source /home/pi/.venv/bin/activate
python edge/api.py
```

## Systemd service
Copy deploy/motorguard.service to /etc/systemd/system/ and enable it with:
```bash
sudo systemctl daemon-reload
sudo systemctl enable motorguard.service
sudo systemctl start motorguard.service
```
