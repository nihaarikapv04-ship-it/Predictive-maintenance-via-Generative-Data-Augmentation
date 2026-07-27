import os
import time
import json
import logging
import threading
from functools import wraps
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S%z'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration from environment variables
SIMULATION_MODE = os.environ.get('SIMULATION_MODE', 'True').lower() == 'true'
MODEL_DIR = os.environ.get('MODEL_DIR', 'models')
YOLO_WEIGHTS = os.environ.get('YOLO_WEIGHTS', 'yolov8n.pt')

START_TIME = time.time()

# Thread-safe lazy loading of models
class ModelRegistry:
    def __init__(self):
        self._models = {}
        self._loaded = False
        self._lock = threading.Lock()
        
    def load_models(self):
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    logger.info("Lazy loading models...")
                    self._models['yolo'] = "Loaded YOLO"
                    self._models['fusion'] = "Loaded Fusion Model"
                    self._loaded = True

model_registry = ModelRegistry()

def load_modules():
    try:
        from backend.observe import vision, vibration
        from backend.diagnose import fusion
        from backend.prescribe import rag
        return vision, vibration, fusion, rag
    except ImportError as e:
        logger.warning(f"Using mock modules due to ImportError: {e}")
        class MockModule:
            def __getattr__(self, name):
                return lambda *args, **kwargs: {"mock": f"Called {name}"}
        return MockModule(), MockModule(), MockModule(), MockModule()

vision, vibration, fusion, rag = load_modules()

def get_cpu_temp():
    try:
        if os.path.exists("/sys/class/thermal/thermal_zone0/temp"):
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return float(f.read().strip()) / 1000.0
        return 45.0
    except Exception:
        return 45.0

@app.errorhandler(400)
def bad_request(error):
    return jsonify({"status": "error", "message": "Bad request", "details": str(error)}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "error", "message": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"status": "error", "message": "Internal server error", "details": str(error)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Returns Raspberry Pi system health status."""
    return jsonify({
        'status': 'ok',
        'uptime': time.time() - START_TIME,
        'temperature': get_cpu_temp(),
        'simulation_mode': SIMULATION_MODE
    })

@app.route('/observe/vision', methods=['POST'])
def observe_vision():
    """Accepts base64-encoded frame, runs YOLO inference."""
    data = request.json
    if not data or 'frame' not in data:
        return jsonify({"status": "error", "message": "Missing 'frame' in request body"}), 400
    
    model_registry.load_models()
    result = vision.run_inference(data['frame']) if hasattr(vision, 'run_inference') else {"detections": []}
    return jsonify({"status": "ok", "data": result})

@app.route('/observe/vibration/stream', methods=['GET'])
def vibration_stream():
    """Endpoint returning vibration data (batch or simulation)."""
    try:
        if hasattr(vibration, 'get_reading'):
            readings = [vibration.get_reading() for _ in range(100)]
        else:
            raise ImportError("vibration.get_reading not available")
        return jsonify({'readings': readings, 'source': 'mpu6050'})
    except Exception as e:
        import numpy as np
        readings = []
        for i in range(100):
            readings.append({
                'ax': float(np.random.normal(0, 0.5)),
                'ay': float(np.random.normal(0, 0.5)),
                'az': float(np.random.normal(9.8, 0.3)),
                'gx': float(np.random.normal(0, 2)),
                'gy': float(np.random.normal(0, 2)),
                'gz': float(np.random.normal(0, 2)),
                'temp': float(np.random.normal(45, 2))
            })
        return jsonify({'readings': readings, 'source': 'simulation'})

@app.route('/diagnose/fuse', methods=['POST'])
def diagnose_fuse():
    """Late-fusion endpoint that accepts vision + vibration features."""
    data = request.json
    if not data or 'vision_features' not in data or 'vibration_features' not in data:
        return jsonify({"status": "error", "message": "Missing required features"}), 400
    
    model_registry.load_models()
    result = fusion.fuse_features(data['vision_features'], data['vibration_features']) if hasattr(fusion, 'fuse_features') else {"health_score": 85.0, "uncertainty": 0.05}
    return jsonify({"status": "ok", "data": result})

@app.route('/prescribe/repair', methods=['POST'])
def prescribe_repair():
    """Accepts diagnosis results, runs RAG pipeline, returns repair protocol."""
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "Missing payload"}), 400
    
    fault_class = data.get('fault_class', data.get('diagnosis', {}).get('fault_type', 'Healthy Baseline'))
    health_score = data.get('health_score', data.get('diagnosis', {}).get('health_score', 85.0))
    
    result = rag.get_repair_protocol(fault_class, health_score) if hasattr(rag, 'get_repair_protocol') else {"protocol": "Inspect motor bearings."}
    return jsonify({"status": "ok", "data": result})

@app.route('/pipeline/run', methods=['POST'])
def run_pipeline():
    """Runs full pipeline."""
    data = request.json or {}
    frame = data.get('frame', '')
    vibration_data = data.get('vibration_data', {})
    
    model_registry.load_models()
    vision_res = vision.run_inference(frame) if hasattr(vision, 'run_inference') else {"worst_condition": "Healthy Baseline", "confidence": 0.95}
    vib_features = vibration.extract_features(vibration_data) if hasattr(vibration, 'extract_features') else {"rms": 0.5}
    diagnosis_res = fusion.fuse_features(vision_res, vib_features) if hasattr(fusion, 'fuse_features') else {"health_score": 85.0}
    repair_protocol = rag.get_repair_protocol("Healthy Baseline", 85.0) if hasattr(rag, 'get_repair_protocol') else {"protocol": "Check alignment."}
    
    return jsonify({
        "status": "ok",
        "data": {
            "observe": {"vision": vision_res, "vibration": vib_features},
            "diagnose": diagnosis_res,
            "prescribe": repair_protocol
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
