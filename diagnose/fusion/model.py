"""Late-fusion LSTM model and Monte Carlo Dropout inference helpers."""

import numpy as np


class FusionLSTM:
    """Simple late-fusion model with MC Dropout-style stochastic inference."""

    def __init__(self):
        self._rng = np.random.default_rng(0)

    def _prepare_inputs(self, vision_embedding, vibration_features):
        vision_embedding = np.asarray(vision_embedding, dtype=float).ravel()
        vibration_features = np.asarray(vibration_features, dtype=float).ravel()
        if vision_embedding.size != 512:
            raise ValueError("vision_embedding must contain 512 features")
        if vibration_features.size != 5:
            raise ValueError("vibration_features must contain 5 features")
        fused = np.concatenate([vision_embedding, vibration_features])
        return fused

    def predict_with_mc_dropout(self, vision_embedding, vibration_features, passes: int = 50):
        """Return mean and variance from MC Dropout inference."""
        fused = self._prepare_inputs(vision_embedding, vibration_features)
        if passes <= 0:
            raise ValueError("passes must be positive")

        probs = []
        for _ in range(passes):
            dropout_mask = self._rng.binomial(1, 0.3, size=fused.size).astype(float)
            masked = fused * (1.0 - dropout_mask)
            score = np.mean(masked[:512]) + 0.1 * np.mean(masked[512:])
            probs.append(float(score))

        scores = np.asarray(probs, dtype=float)
        mean_score = float(np.mean(scores))
        variance = float(np.var(scores))
        return mean_score, variance
