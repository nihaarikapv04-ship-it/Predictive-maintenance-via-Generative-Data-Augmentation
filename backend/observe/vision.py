import base64
import logging
import time
import random
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import cv2

try:
    from ultralytics import YOLO
    import torch
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

logger = logging.getLogger(__name__)

class VisionObserver:
    """
    VisionObserver for YOLOv11n webcam inference with CLAHE preprocessing.
    """
    def __init__(self, model_path: str = 'yolo11n.pt', confidence_threshold: float = 0.25, device: str = 'auto'):
        """
        Initializes the VisionObserver, loading the YOLO model and configuring the device.
        
        Args:
            model_path: Path to the YOLO weights.
            confidence_threshold: Minimum confidence threshold for detections.
            device: 'auto', 'cpu', 'cuda', or 'mps'.
        """
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.model = None
        self.is_simulated = False
        self.device = device

        if not ULTRALYTICS_AVAILABLE:
            logger.warning("Ultralytics package not available. Falling back to simulation.")
            self.is_simulated = True
            return

        try:
            if self.device == 'auto':
                if torch.cuda.is_available():
                    self.device = 'cuda'
                elif torch.backends.mps.is_available():
                    self.device = 'mps'
                else:
                    self.device = 'cpu'
            
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            logger.info(f"Successfully loaded YOLO model from {self.model_path} on device {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {self.model_path}: {e}. Falling back to simulation.")
            self.is_simulated = True

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for low-light enhancement.
        Converts to LAB, applies CLAHE to L channel, and converts back to BGR.
        
        Args:
            frame: Input BGR image as numpy array.
            
        Returns:
            Preprocessed BGR image.
        """
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return enhanced_frame
        except Exception as e:
            logger.error(f"Error in preprocessing frame: {e}")
            return frame

    def get_fault_color(self, class_name: str) -> Tuple[int, int, int]:
        """
        Returns BGR color for fault classes.
        
        Args:
            class_name: Name of the fault class.
            
        Returns:
            Tuple of (B, G, R) color values.
        """
        colors = {
            'Normal': (0, 255, 0),
            'Inner Race Fault': (0, 0, 255),
            'Outer Race Fault': (255, 0, 0),
            'Ball Fault': (0, 165, 255),
            'Misalignment': (255, 0, 255)
        }
        return colors.get(class_name, (200, 200, 200))

    def draw_annotations(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draws bounding boxes with class labels and confidence on the frame.
        
        Args:
            frame: BGR image.
            detections: List of detection dictionaries.
            
        Returns:
            Annotated BGR image.
        """
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls_name = det['class_name']
            color = det['color']
            
            # Draw bounding box
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label background
            label = f"{cls_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (int(x1), int(y1) - 20), (int(x1) + w, int(y1)), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return annotated

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Runs YOLO inference on a frame.
        
        Args:
            frame: Input BGR image.
            
        Returns:
            Dict containing detections, annotated frame (base64), inference time, and frame shape.
        """
        if self.is_simulated:
            logger.debug("Falling back to SimulatedVisionObserver due to initialization failure.")
            return SimulatedVisionObserver().detect(frame)

        start_time = time.time()
        try:
            preprocessed = self.preprocess_frame(frame)
            results = self.model(preprocessed, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            if len(results) > 0:
                result = results[0]
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id] if result.names else f"class_{cls_id}"
                    
                    detections.append({
                        'class_name': cls_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2],
                        'color': self.get_fault_color(cls_name)
                    })
            
            annotated_frame = self.draw_annotations(frame, detections)
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            b64_img = base64.b64encode(buffer).decode('utf-8')
            
            inference_time_ms = (time.time() - start_time) * 1000
            logger.info(f"Detected {len(detections)} objects in {inference_time_ms:.2f} ms")
            
            return {
                'detections': detections,
                'annotated_frame': b64_img,
                'inference_time_ms': inference_time_ms,
                'frame_shape': frame.shape
            }
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {
                'detections': [],
                'annotated_frame': "",
                'inference_time_ms': 0.0,
                'frame_shape': frame.shape if frame is not None else (0, 0, 0)
            }

    def detect_from_base64(self, b64_string: str) -> Dict[str, Any]:
        """
        Decodes base64 image and runs detect().
        
        Args:
            b64_string: Base64 encoded image string (with or without data URI prefix).
            
        Returns:
            Dict containing detections, annotated frame (base64), inference time, and frame shape.
        """
        try:
            if ',' in b64_string:
                b64_string = b64_string.split(',')[1]
            img_data = base64.b64decode(b64_string)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Failed to decode image from base64 string.")
            return self.detect(frame)
        except Exception as e:
            logger.error(f"Failed to detect from base64: {e}")
            return {
                'detections': [],
                'annotated_frame': "",
                'inference_time_ms': 0.0,
                'frame_shape': (0, 0, 0)
            }


class SimulatedVisionObserver(VisionObserver):
    """
    Simulation mode that generates synthetic detections without a real YOLO model.
    Useful for testing or when the model/ultralytics is unavailable.
    """
    def __init__(self, *args, **kwargs):
        """Initializes the simulated observer."""
        self.is_simulated = True
        self.classes = ['Normal', 'Inner Race Fault', 'Outer Race Fault', 'Ball Fault', 'Misalignment']
        
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Generates synthetic detections and bounding boxes.
        
        Args:
            frame: Input BGR image.
            
        Returns:
            Dict matching VisionObserver output.
        """
        start_time = time.time()
        
        # Simulate processing time
        time.sleep(random.uniform(0.02, 0.08))
        
        h, w = frame.shape[:2]
        
        # Randomly generate 0 to 3 detections
        num_detections = random.randint(0, 3)
        detections = []
        for _ in range(num_detections):
            cls_name = random.choice(self.classes)
            # Realistic confidence distributions
            conf = random.uniform(0.45, 0.99)
            
            # Generate synthetic bounding box
            x1 = random.randint(0, int(w * 0.7))
            y1 = random.randint(0, int(h * 0.7))
            x2 = random.randint(x1 + 30, w)
            y2 = random.randint(y1 + 30, h)
            
            detections.append({
                'class_name': cls_name,
                'confidence': conf,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'color': self.get_fault_color(cls_name)
            })
            
        annotated_frame = self.draw_annotations(frame, detections)
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        b64_img = base64.b64encode(buffer).decode('utf-8')
        
        inference_time_ms = (time.time() - start_time) * 1000
        logger.info(f"[SIMULATED] Detected {len(detections)} objects in {inference_time_ms:.2f} ms")
        
        return {
            'detections': detections,
            'annotated_frame': b64_img,
            'inference_time_ms': inference_time_ms,
            'frame_shape': frame.shape
        }
