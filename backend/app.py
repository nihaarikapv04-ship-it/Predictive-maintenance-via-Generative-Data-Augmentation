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
                    # Mock model loading here
                    self._models['yolo'] = "Loaded YOLO"
                    self._models['fusion'] = "Loaded Fusion Model"
                    self._loaded = True

model_registry = ModelRegistry()

def load_modules():
    # Attempt to import real modules, fallback to mock modules if they don't exist
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

def track_latency(f):
    """Decorator to track latency and format JSON response."""
    @wraps(f)
    def decorated(*args, **kwargs):
        start = time.time()
        try:
            result = f(*args, **kwargs)
            latency_ms = (time.time() - start) * 1000
            
            return jsonify({
                "status": "ok",
                "data": result,
                "latency_ms": round(latency_ms, 2)
            })
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            logger.exception(f"Error in {f.__name__}: {str(e)}")
            return jsonify({
                "status": "error",
                "message": str(e),
                "latency_ms": round(latency_ms, 2)
            }), 500
    return decorated

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
@track_latency
def health_check():
    """Returns system health status."""
    uptime = time.time() - START_TIME
    return {
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "simulation_mode": SIMULATION_MODE,
        "models_loaded": model_registry._loaded
    }

@app.route('/observe/vision', methods=['POST'])
@track_latency
def observe_vision():
    """Accepts base64-encoded frame, runs YOLO inference."""
    data = request.json
    if not data or 'frame' not in data:
        raise ValueError("Missing 'frame' in request body")
    
    model_registry.load_models()
    result = vision.run_inference(data['frame']) if hasattr(vision, 'run_inference') else {"detections": []}
    return result

@app.route('/observe/vibration/stream', methods=['GET'])
def stream_vibration():
    """SSE endpoint streaming vibration data."""
    def generate():
        import random
        while True:
            time.sleep(1)
            data = {"x": random.random(), "y": random.random(), "z": random.random()}
            yield f"data: {json.dumps(data)}\n\n"
            
    return Response(generate(), mimetype='text/event-stream')

@app.route('/diagnose/fuse', methods=['POST'])
@track_latency
def diagnose_fuse():
    """Late-fusion endpoint that accepts vision + vibration features."""
    data = request.json
    if not data or 'vision_features' not in data or 'vibration_features' not in data:
        raise ValueError("Missing required features in request")
    
    model_registry.load_models()
    result = fusion.fuse_features(data['vision_features'], data['vibration_features']) if hasattr(fusion, 'fuse_features') else {"health_score": 0.95, "uncertainty": 0.05}
    return result

@app.route('/prescribe/repair', methods=['POST'])
@track_latency
def prescribe_repair():
    """Accepts diagnosis results, runs RAG pipeline, returns repair protocol."""
    data = request.json
    if not data or 'diagnosis' not in data:
        raise ValueError("Missing 'diagnosis' in request")
    
    result = rag.get_repair_protocol(data['diagnosis']) if hasattr(rag, 'get_repair_protocol') else {"protocol": "Inspect motor bearings."}
    return result

@app.route('/pipeline/run', methods=['POST'])
@track_latency
def run_pipeline():
    """Runs the full Observe->Diagnose->Prescribe pipeline end-to-end."""
    data = request.json
    if not data:
        raise ValueError("Empty request payload")
        
    # Example orchestrated flow
    frame = data.get('frame', '')
    vibration_data = data.get('vibration_data', {})
    
    # 1. Observe
    model_registry.load_models()
    vision_res = vision.run_inference(frame) if hasattr(vision, 'run_inference') else {"features": [0.1, 0.2]}
    vib_features = vibration.extract_features(vibration_data) if hasattr(vibration, 'extract_features') else {"features": [0.3, 0.4]}
    
    # 2. Diagnose
    diagnosis_res = fusion.fuse_features(vision_res, vib_features) if hasattr(fusion, 'fuse_features') else {"health_score": 0.7}
    
    # 3. Prescribe
    repair_protocol = rag.get_repair_protocol(diagnosis_res) if hasattr(rag, 'get_repair_protocol') else {"protocol": "Replace belt."}
    
    return {
        "observe": {
            "vision": vision_res,
            "vibration": vib_features
        },
        "diagnose": diagnosis_res,
        "prescribe": repair_protocol
    }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
