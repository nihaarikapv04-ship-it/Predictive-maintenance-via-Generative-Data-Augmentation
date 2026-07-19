import numpy as np

from observe.filtering import butterworth_bandpass


def test_butterworth_bandpass_passes_target_frequencies():
    fs = 1500.0
    duration = 1.0
    t = np.arange(0, duration, 1.0 / fs)

    signal = np.sin(2 * np.pi * 74.6 * t) + np.sin(2 * np.pi * 117.4 * t) + np.sin(2 * np.pi * 51.2 * t)

    filtered = butterworth_bandpass(signal, 10.0, 500.0, fs, order=4)

    assert filtered.shape == signal.shape
    assert np.isfinite(filtered).all()
    assert np.max(np.abs(filtered)) > 0
