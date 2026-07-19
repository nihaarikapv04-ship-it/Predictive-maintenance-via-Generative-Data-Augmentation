"""IMU acquisition utilities for the Observe layer.

This module wraps MPU-6050 access behind a hardware abstraction so the
same code can run against a simulator or a real sensor if one is attached.
"""

import os
import numpy as np


class IMUReader:
    """Read a window of vibration samples from a simulator or hardware."""

    def __init__(self, simulate=None):
        if simulate is None:
            mode = os.getenv("MOTORGUARD_HARDWARE_MODE", "simulate").lower()
            simulate = mode != "real"
        self.simulate = simulate

    def read_window(self, length: int = 1024):
        """Return a synthetic or real vibration window."""
        if self.simulate:
            fs = 1500.0
            t = np.arange(length) / fs
            return np.sin(2 * np.pi * 74.6 * t) + 0.2 * np.sin(2 * np.pi * 117.4 * t)
        try:
            import smbus2  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment fallback
            raise RuntimeError("Real IMU access requires smbus2 and a connected sensor.") from exc

        bus = smbus2.SMBus(1)
        try:
            # Read a tiny synthetic window as a hardware placeholder.
            return np.linspace(0.0, 1.0, length)
        finally:
            bus.close()
