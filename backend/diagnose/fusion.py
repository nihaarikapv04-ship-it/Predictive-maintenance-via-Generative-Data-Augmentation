import logging
import math
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FAULT_CLASSES = [
    'Healthy Baseline',
    'Mild Oxidation',
    'Moderate Corrosion',
    'Severe Corrosion',
    'Structural Cracking',
    'Contamination',
]

class FusionDiagnostics(nn.Module):
    """
    Late-Fusion LSTM with Monte Carlo Dropout for the MotorGuard AI diagnostic system.
    Combines vision features (from YOLO) and vibration features.
    """
    def __init__(self, vision_dim: int, vibration_dim: int, num_classes: int = len(FAULT_CLASSES)):
        super(FusionDiagnostics, self).__init__()
        
        # Vision branch
        self.vision_branch = nn.Sequential(
            nn.Linear(vision_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Vibration branch
        self.vibration_lstm = nn.LSTM(
            input_size=vibration_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(64 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output heads
        self.health_head = nn.Linear(64, 1)
        self.fault_head = nn.Linear(64, num_classes)
        self.rul_head = nn.Linear(64, 1)

    def forward(self, vision_features: torch.Tensor, vibration_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Vision features processing
        vis_out = self.vision_branch(vision_features)
        
        # Vibration features processing
        # vibration_features shape should be (batch, seq_len, vibration_dim)
        lstm_out, (hn, cn) = self.vibration_lstm(vibration_features)
        # Take the last hidden state from the top layer of LSTM
        vib_out = hn[-1] # shape (batch, 128)
        
        # Fusion
        fused = torch.cat((vis_out, vib_out), dim=1)
        fused_out = self.fusion_layer(fused)
        
        # Output heads
        health_score = torch.sigmoid(self.health_head(fused_out)) * 100.0  # Scale 0-100
        fault_logits = self.fault_head(fused_out)
        fault_probs = F.softmax(fault_logits, dim=1)
        rul = F.relu(self.rul_head(fused_out))
        
        return health_score, fault_probs, rul


class MCDropoutPredictor:
    """
    Monte Carlo Dropout Predictor for uncertainty estimation.
    """
    def __init__(self, model: nn.Module, num_samples: int = 50):
        self.model = model
        self.num_samples = num_samples
        self.device = next(model.parameters()).device

    def predict_with_uncertainty(self, vision_features: torch.Tensor, vibration_features: torch.Tensor) -> Dict[str, Any]:
        self.model.train()  # Keep dropout enabled
        
        health_scores = []
        fault_probs = []
        ruls = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                h, f, r = self.model(vision_features.to(self.device), vibration_features.to(self.device))
                health_scores.append(h.cpu().numpy())
                fault_probs.append(f.cpu().numpy())
                ruls.append(r.cpu().numpy())
                
        # Stack results
        health_scores = np.stack(health_scores, axis=0)  # (num_samples, batch, 1)
        fault_probs = np.stack(fault_probs, axis=0)      # (num_samples, batch, num_classes)
        ruls = np.stack(ruls, axis=0)                    # (num_samples, batch, 1)
        
        # Calculate statistics (assuming batch size 1 for simplicity in output dict, take index 0)
        h_mean = float(np.mean(health_scores[:, 0, 0]))
        h_std = float(np.std(health_scores[:, 0, 0]))
        
        f_mean = np.mean(fault_probs[:, 0, :], axis=0)
        predicted_fault_idx = int(np.argmax(f_mean))
        
        r_mean = float(np.mean(ruls[:, 0, 0]))
        r_std = float(np.std(ruls[:, 0, 0]))
        
        # Compute epistemic uncertainty (variance of the predictions)
        # For classification, we use the predictive entropy as epistemic uncertainty approximation
        entropy = float(-np.sum(f_mean * np.log(f_mean + 1e-8)))
        epistemic_uncertainty = entropy
        
        # Aleatoric uncertainty (expected entropy of the individual predictions)
        individual_entropies = -np.sum(fault_probs[:, 0, :] * np.log(fault_probs[:, 0, :] + 1e-8), axis=1)
        aleatoric_uncertainty = float(np.mean(individual_entropies))
        
        # Health score uncertainty level (relative standard deviation)
        cv = (h_std / h_mean) if h_mean > 0 else 0
        if cv < 0.05:
            uncertainty_level = 'low'
        elif cv < 0.15:
            uncertainty_level = 'medium'
        else:
            uncertainty_level = 'high'
            
        # Maintenance urgency
        if h_mean > 80:
            urgency = 'none'
        elif h_mean > 60:
            urgency = 'routine'
        elif h_mean > 40:
            urgency = 'soon'
        else:
            urgency = 'immediate'
            
        return {
            'health_score_mean': h_mean,
            'health_score_std': h_std,
            'fault_class_probabilities': f_mean.tolist(),
            'predicted_fault': FAULT_CLASSES[predicted_fault_idx],
            'rul_mean_hours': r_mean,
            'rul_std_hours': r_std,
            'confidence_interval_95': {
                'health': (max(0, h_mean - 1.96 * h_std), min(100, h_mean + 1.96 * h_std)),
                'rul': (max(0, r_mean - 1.96 * r_std), r_mean + 1.96 * r_std)
            },
            'uncertainty_level': uncertainty_level,
            'maintenance_urgency': urgency,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty
        }


class DiagnosticEngine:
    """
    Engine for managing features, diagnostic pipeline, and simulation mode.
    """
    def __init__(self, model_path: Optional[str] = None, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        
        # Feature dimensions
        self.vision_dim = 16
        self.vibration_dim = 32
        
        if not self.simulation_mode:
            self.model = FusionDiagnostics(vision_dim=self.vision_dim, vibration_dim=self.vibration_dim).to(self.device)
            if model_path:
                try:
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    logger.info(f"Loaded model from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load model from {model_path}: {e}")
                    raise
            self.predictor = MCDropoutPredictor(self.model)
        else:
            self.model = None
            self.predictor = None
            logger.info("Initializing DiagnosticEngine in Simulation Mode")

    def prepare_vision_features(self, detections: Dict[str, Any]) -> torch.Tensor:
        """
        Converts YOLO detections to feature tensor.
        Expected detections format: {'confidence': list, 'class_id': list, 'bbox': list_of_lists}
        """
        try:
            # Simulated feature extraction: average confidence, bounding box area, etc.
            # Here we just create a dummy tensor of size (1, vision_dim) for the model
            features = np.zeros((1, self.vision_dim), dtype=np.float32)
            if 'confidence' in detections and detections['confidence']:
                features[0, 0] = np.mean(detections['confidence'])
                features[0, 1] = len(detections['confidence'])
            return torch.tensor(features)
        except Exception as e:
            logger.error(f"Error preparing vision features: {e}")
            return torch.zeros((1, self.vision_dim))

    def prepare_vibration_features(self, vib_data: Dict[str, Any]) -> torch.Tensor:
        """
        Converts vibration features to LSTM-ready tensor.
        Expected vib_data format: {'time_series': array_like_sequence}
        """
        try:
            # Create dummy sequence tensor of size (1, seq_len, vibration_dim)
            seq_len = 10
            features = np.zeros((1, seq_len, self.vibration_dim), dtype=np.float32)
            if 'rms' in vib_data:
                features[0, :, 0] = vib_data['rms']
            return torch.tensor(features)
        except Exception as e:
            logger.error(f"Error preparing vibration features: {e}")
            return torch.zeros((1, 10, self.vibration_dim))

    def _simulate_diagnosis(self) -> Dict[str, Any]:
        """
        Generates realistic simulation data.
        """
        base_health = np.random.normal(75, 5)
        h_mean = max(0, min(100, base_health))
        h_std = abs(np.random.normal(2.0, 0.5))
        
        # Derive probabilities
        probs = np.random.dirichlet(np.ones(len(FAULT_CLASSES)))
        if h_mean > 80:
            probs[0] = 0.9 + np.random.uniform(0, 0.05)  # High prob for 'Healthy Baseline'
            probs = probs / np.sum(probs)
        predicted_fault_idx = int(np.argmax(probs))
        
        r_mean = h_mean * 10 + np.random.normal(0, 100) # Roughly 10 hours per health point
        r_mean = max(10, r_mean)
        r_std = r_mean * 0.1
        
        cv = (h_std / h_mean) if h_mean > 0 else 0
        uncertainty_level = 'low' if cv < 0.05 else ('medium' if cv < 0.15 else 'high')
        
        urgency = 'none' if h_mean > 80 else ('routine' if h_mean > 60 else ('soon' if h_mean > 40 else 'immediate'))
        
        return {
            'health_score_mean': h_mean,
            'health_score_std': h_std,
            'fault_class_probabilities': probs.tolist(),
            'predicted_fault': FAULT_CLASSES[predicted_fault_idx],
            'rul_mean_hours': r_mean,
            'rul_std_hours': r_std,
            'confidence_interval_95': {
                'health': (max(0, h_mean - 1.96 * h_std), min(100, h_mean + 1.96 * h_std)),
                'rul': (max(0, r_mean - 1.96 * r_std), r_mean + 1.96 * r_std)
            },
            'uncertainty_level': uncertainty_level,
            'maintenance_urgency': urgency,
            'epistemic_uncertainty': np.random.uniform(0.1, 0.5),
            'aleatoric_uncertainty': np.random.uniform(0.05, 0.3)
        }

    def diagnose(self, vision_data: Dict[str, Any], vibration_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full diagnostic pipeline.
        """
        try:
            if self.simulation_mode:
                return self._simulate_diagnosis()
                
            vision_tensor = self.prepare_vision_features(vision_data)
            vibration_tensor = self.prepare_vibration_features(vibration_data)
            
            return self.predictor.predict_with_uncertainty(vision_tensor, vibration_tensor)
        except Exception as e:
            logger.error(f"Error during diagnosis: {e}")
            raise

    def get_trend(self, history: List[float]) -> str:
        """
        Analyze health score trend from historical data.
        """
        if not history or len(history) < 2:
            return 'stable'
            
        recent = history[-min(5, len(history)):]
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if slope > 2.0:
            return 'improving'
        elif slope < -5.0:
            return 'critical_decline'
        elif slope < -1.0:
            return 'degrading'
        else:
            return 'stable'
