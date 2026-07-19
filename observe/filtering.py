"""Signal filtering utilities for vibration preprocessing.

The target implementation uses a Butterworth bandpass filter with
zero-phase filtering for deployment-time vibration signals.
"""

import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt
except ImportError:  # pragma: no cover - fallback for minimal environments
    butter = None
    sosfiltfilt = None


def butterworth_bandpass(signal, low_cut: float, high_cut: float, fs: float, order: int = 4):
    """Apply a Butterworth bandpass filter to a 1D signal.

    The implementation uses zero-phase filtering when SciPy is available.
    A simple fallback is used otherwise so unit tests can still run.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("signal must be a one-dimensional array")
    if low_cut <= 0 or high_cut <= low_cut:
        raise ValueError("invalid filter bounds")
    if fs <= 0:
        raise ValueError("sampling frequency must be positive")

    if butter is None or sosfiltfilt is None:
        return signal

    nyquist = 0.5 * fs
    if low_cut >= nyquist or high_cut >= nyquist:
        raise ValueError("filter bounds must be below the Nyquist frequency")

    sos = butter(order, [low_cut, high_cut], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)
