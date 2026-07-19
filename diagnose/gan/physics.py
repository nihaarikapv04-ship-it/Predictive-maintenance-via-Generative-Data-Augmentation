"""Bearing kinematics helpers for the physics-aware GAN loss.

The deployment motor constants are kept separate from the CWRU training-time
constants that appear in the prototype code.
"""

import math

DEPLOYMENT_BPFO = 74.6
DEPLOYMENT_BPFI = 117.4
DEPLOYMENT_BSF = 51.2

TRAINING_BPFO = 107.4
TRAINING_BPFI = 162.2
TRAINING_BSF = 68.5


def bearing_defect_frequencies(nb: float, fr: float, bd: float, pd: float, alpha: float):
    """Return BPFO, BPFI, and BSF using the provided bearing formulas."""
    alpha_rad = math.radians(alpha)
    cos_alpha = math.cos(alpha_rad)

    bpfo = (nb / 2.0) * fr * (1 - (bd / pd) * cos_alpha)
    bpfi = (nb / 2.0) * fr * (1 + (bd / pd) * cos_alpha)
    bsf = (pd / (2.0 * bd)) * fr * (1 - ((bd / pd) ** 2) * (cos_alpha ** 2))
    return bpfo, bpfi, bsf
