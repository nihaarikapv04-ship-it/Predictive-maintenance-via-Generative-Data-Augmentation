import logging
import time
import json
import struct
import math
from typing import Dict, Any, Generator, Optional, Tuple
from enum import Enum

import numpy as np
from scipy import signal
from scipy.stats import kurtosis

try:
    import smbus2
    SMBUS_AVAILABLE = True
except ImportError:
    smbus2 = None
    SMBUS_AVAILABLE = False

logger = logging.getLogger(__name__)

class FaultType(Enum):
    NORMAL = "normal"
    INNER_RACE = "inner_race"
    OUTER_RACE = "outer_race"
    BALL_FAULT = "ball_fault"
    RANDOM = "random"


class SimulatedVibrationSource:
    """Generates synthetic vibration data styled after the CWRU bearing dataset."""
    
    def __init__(self, sample_rate: int = 1500, window_size: int = 2048):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.motor_freq = 30.0  # 1797 RPM ~ 30Hz
        
        # Characteristic frequencies (typical values for 6205 bearing)
        self.bpfi = 162.19
        self.bpfo = 107.36
        self.bsf = 141.17
        
    def _generate_time_vector(self) -> np.ndarray:
        return np.linspace(0, self.window_size / self.sample_rate, self.window_size, endpoint=False)
        
    def _add_noise_and_drift(self, sig: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        # Add white noise
        sig += np.random.normal(0, noise_level, self.window_size)
        # Add slight low frequency drift
        drift = np.sin(2 * np.pi * 0.5 * self._generate_time_vector()) * 0.02
        return sig + drift
        
    def generate_normal(self) -> np.ndarray:
        t = self._generate_time_vector()
        # Motor running speed harmonic
        sig = 0.1 * np.sin(2 * np.pi * self.motor_freq * t)
        return self._add_noise_and_drift(sig, noise_level=0.05)
        
    def generate_inner_race_fault(self) -> np.ndarray:
        t = self._generate_time_vector()
        # BPFI modulated by running speed
        carrier = np.sin(2 * np.pi * 3000 * t) # High freq resonance
        modulator = 0.5 * (1 + np.sin(2 * np.pi * self.motor_freq * t))
        impacts = np.zeros(self.window_size)
        
        # Add impacts at BPFI
        impact_interval = int(self.sample_rate / self.bpfi)
        for i in range(0, self.window_size, impact_interval):
            impacts[i:min(i+10, self.window_size)] = np.exp(-np.linspace(0, 5, min(10, self.window_size-i)))
            
        sig = 0.1 * np.sin(2 * np.pi * self.motor_freq * t) + 1.5 * impacts * carrier * modulator
        return self._add_noise_and_drift(sig, noise_level=0.1)
        
    def generate_outer_race_fault(self) -> np.ndarray:
        t = self._generate_time_vector()
        carrier = np.sin(2 * np.pi * 2500 * t) 
        impacts = np.zeros(self.window_size)
        
        impact_interval = int(self.sample_rate / self.bpfo)
        for i in range(0, self.window_size, impact_interval):
            impacts[i:min(i+10, self.window_size)] = np.exp(-np.linspace(0, 5, min(10, self.window_size-i)))
            
        sig = 0.1 * np.sin(2 * np.pi * self.motor_freq * t) + 1.2 * impacts * carrier
        return self._add_noise_and_drift(sig, noise_level=0.1)
        
    def generate_ball_fault(self) -> np.ndarray:
        t = self._generate_time_vector()
        carrier = np.sin(2 * np.pi * 4000 * t) 
        modulator = 0.5 * (1 + np.sin(2 * np.pi * (self.motor_freq / 2.0) * t)) # FTF modulation
        impacts = np.zeros(self.window_size)
        
        impact_interval = int(self.sample_rate / self.bsf)
        for i in range(0, self.window_size, impact_interval):
            impacts[i:min(i+10, self.window_size)] = np.exp(-np.linspace(0, 5, min(10, self.window_size-i)))
            
        sig = 0.1 * np.sin(2 * np.pi * self.motor_freq * t) + 1.0 * impacts * carrier * modulator
        return self._add_noise_and_drift(sig, noise_level=0.08)

    def generate_sample(self, fault_type: str = 'random') -> np.ndarray:
        """Returns (window_size, 6) synthetic data for ax, ay, az, gx, gy, gz"""
        if fault_type == 'random':
            fault_type = np.random.choice([f.value for f in FaultType])
            
        if fault_type == FaultType.NORMAL.value:
            base_sig = self.generate_normal()
        elif fault_type == FaultType.INNER_RACE.value:
            base_sig = self.generate_inner_race_fault()
        elif fault_type == FaultType.OUTER_RACE.value:
            base_sig = self.generate_outer_race_fault()
        elif fault_type == FaultType.BALL_FAULT.value:
            base_sig = self.generate_ball_fault()
        else:
            base_sig = self.generate_normal()
            
        # Create 6 channels with some cross-talk and phase shifts
        data = np.zeros((self.window_size, 6))
        for i in range(6):
            if i < 3: # Accel
                data[:, i] = base_sig * (1.0 - i*0.2) + np.random.normal(0, 0.02, self.window_size)
            else: # Gyro
                data[:, i] = base_sig * 50 * (1.0 - (i-3)*0.2) + np.random.normal(0, 1.0, self.window_size)
                
        return data


class VibrationObserver:
    # MPU-6050 Registers
    PWR_MGMT_1 = 0x6B
    SMPLRT_DIV = 0x19
    CONFIG = 0x1A
    GYRO_CONFIG = 0x1B
    ACCEL_CONFIG = 0x1C
    INT_ENABLE = 0x38
    ACCEL_XOUT_H = 0x3B
    
    MPU6050_ADDR = 0x68
    
    def __init__(self, i2c_bus: int = 1, sample_rate: int = 1500, window_size: int = 2048, simulation_mode: bool = True):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.simulation_mode = simulation_mode
        self.i2c_bus_num = i2c_bus
        self.bus = None
        
        if not self.simulation_mode and not SMBUS_AVAILABLE:
            logger.warning("smbus2 not available. Forcing simulation mode.")
            self.simulation_mode = True
            
        if self.simulation_mode:
            logger.info("Initializing VibrationObserver in Simulation Mode.")
            self.sim_source = SimulatedVibrationSource(sample_rate=self.sample_rate, window_size=self.window_size)
        else:
            logger.info(f"Initializing VibrationObserver with I2C bus {self.i2c_bus_num}.")
            try:
                self.bus = smbus2.SMBus(self.i2c_bus_num)
                self._init_mpu6050()
            except Exception as e:
                logger.error(f"Failed to initialize MPU6050 on I2C bus {self.i2c_bus_num}: {e}")
                logger.warning("Falling back to simulation mode.")
                self.simulation_mode = True
                self.sim_source = SimulatedVibrationSource(sample_rate=self.sample_rate, window_size=self.window_size)

        # Pre-compute filter coefficients
        self.sos = signal.butter(5, [10.0, 500.0], btype='bandpass', fs=self.sample_rate, output='sos')
                
    def _init_mpu6050(self):
        """Initialize MPU-6050."""
        try:
            # Wake up
            self.bus.write_byte_data(self.MPU6050_ADDR, self.PWR_MGMT_1, 0x00)
            time.sleep(0.1)
            # DLPF to 260Hz (CONFIG = 0x00)
            self.bus.write_byte_data(self.MPU6050_ADDR, self.CONFIG, 0x00)
            # Gyro config: +/- 250 deg/s
            self.bus.write_byte_data(self.MPU6050_ADDR, self.GYRO_CONFIG, 0x00)
            # Accel config: +/- 2g
            self.bus.write_byte_data(self.MPU6050_ADDR, self.ACCEL_CONFIG, 0x00)
            # Sample rate divider
            self.bus.write_byte_data(self.MPU6050_ADDR, self.SMPLRT_DIV, 0x00)
            logger.info("MPU-6050 initialized successfully.")
        except Exception as e:
            logger.error(f"Error writing to MPU6050 registers: {e}")
            raise
            
    def _read_raw_data(self, addr: int) -> int:
        """Read 16-bit signed value from I2C register."""
        try:
            high = self.bus.read_byte_data(self.MPU6050_ADDR, addr)
            low = self.bus.read_byte_data(self.MPU6050_ADDR, addr+1)
            value = (high << 8) | low
            if value > 32768:
                value = value - 65536
            return value
        except Exception as e:
            logger.error(f"Failed to read from address {addr}: {e}")
            return 0

    def read_sample(self) -> Dict[str, float]:
        """Read one 6-axis sample."""
        if self.simulation_mode:
            sim_data = self.sim_source.generate_sample('normal')[0]
            return {
                "ax": float(sim_data[0]), "ay": float(sim_data[1]), "az": float(sim_data[2]),
                "gx": float(sim_data[3]), "gy": float(sim_data[4]), "gz": float(sim_data[5])
            }
            
        try:
            acc_x = self._read_raw_data(self.ACCEL_XOUT_H)
            acc_y = self._read_raw_data(self.ACCEL_XOUT_H + 2)
            acc_z = self._read_raw_data(self.ACCEL_XOUT_H + 4)
            gyro_x = self._read_raw_data(self.ACCEL_XOUT_H + 8)
            gyro_y = self._read_raw_data(self.ACCEL_XOUT_H + 10)
            gyro_z = self._read_raw_data(self.ACCEL_XOUT_H + 12)
            
            # Scale factors for +/- 2g and +/- 250 deg/s
            return {
                "ax": float(acc_x) / 16384.0,
                "ay": float(acc_y) / 16384.0,
                "az": float(acc_z) / 16384.0,
                "gx": float(gyro_x) / 131.0,
                "gy": float(gyro_y) / 131.0,
                "gz": float(gyro_z) / 131.0
            }
        except Exception as e:
            logger.error(f"Error reading sample: {e}")
            return {"ax": 0.0, "ay": 0.0, "az": 0.0, "gx": 0.0, "gy": 0.0, "gz": 0.0}

    def read_window(self) -> np.ndarray:
        """Read window_size samples."""
        if self.simulation_mode:
            return self.sim_source.generate_sample()
            
        data = np.zeros((self.window_size, 6))
        interval = 1.0 / self.sample_rate
        for i in range(self.window_size):
            start = time.perf_counter()
            s = self.read_sample()
            data[i] = [s['ax'], s['ay'], s['az'], s['gx'], s['gy'], s['gz']]
            
            elapsed = time.perf_counter() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)
        return data

    def apply_butterworth_filter(self, data: np.ndarray, lowcut: float = 10.0, highcut: float = 500.0, order: int = 5) -> np.ndarray:
        """Apply Butterworth bandpass filter to each channel."""
        try:
            filtered = np.zeros_like(data)
            for i in range(data.shape[1]):
                filtered[:, i] = signal.sosfilt(self.sos, data[:, i])
            return filtered
        except Exception as e:
            logger.error(f"Error applying filter: {e}")
            return data

    def compute_features(self, filtered_data: np.ndarray) -> Dict[str, Any]:
        """Compute time and frequency domain features."""
        try:
            features = {}
            channels = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
            
            ovs = 0.0
            
            for i, ch in enumerate(channels):
                ch_data = filtered_data[:, i]
                
                # Time domain features
                rms = np.sqrt(np.mean(ch_data**2))
                peak = np.max(np.abs(ch_data))
                p2p = np.max(ch_data) - np.min(ch_data)
                crest_factor = peak / rms if rms > 1e-6 else 0.0
                kurt = float(kurtosis(ch_data))
                
                if i < 3: # Accel channels contribute to OVS
                    ovs += rms**2
                
                # Frequency domain features
                f, Pxx = signal.periodogram(ch_data, self.sample_rate)
                dom_freq_idx = np.argmax(Pxx)
                dom_freq = f[dom_freq_idx]
                
                # Spectral entropy
                Pxx_norm = Pxx / np.sum(Pxx) if np.sum(Pxx) > 0 else np.ones_like(Pxx) / len(Pxx)
                spectral_entropy = -np.sum(Pxx_norm * np.log2(Pxx_norm + 1e-12))
                
                features[f"{ch}_rms"] = float(rms)
                features[f"{ch}_p2p"] = float(p2p)
                features[f"{ch}_crest"] = float(crest_factor)
                features[f"{ch}_kurtosis"] = float(kurt)
                features[f"{ch}_dom_freq"] = float(dom_freq)
                features[f"{ch}_spectral_entropy"] = float(spectral_entropy)
                
            features["overall_vibration_severity"] = float(np.sqrt(ovs))
            return features
        except Exception as e:
            logger.error(f"Error computing features: {e}")
            return {}

    def get_health_indicators(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Maps features to health indicators based on ISO 10816."""
        try:
            ovs = features.get("overall_vibration_severity", 0.0)
            
            # ISO 10816 typical levels (Class I machines - small machines < 15kW)
            # mm/s RMS (approximated here assuming unit is g, conversion roughly * 9800 / (2*pi*f))
            # Just using arbitrary thresholds for demonstration
            if ovs < 0.71:
                severity = "GOOD"
            elif ovs < 1.8:
                severity = "SATISFACTORY"
            elif ovs < 4.5:
                severity = "UNSATISFACTORY"
            else:
                severity = "UNACCEPTABLE"
                
            # Bearing condition from kurtosis (normal is ~3)
            max_kurt = max([features.get(f"{ch}_kurtosis", 3.0) for ch in ['ax', 'ay', 'az']])
            if max_kurt > 5.0:
                bearing_cond = "FAULTY"
            elif max_kurt > 4.0:
                bearing_cond = "WARNING"
            else:
                bearing_cond = "NORMAL"
                
            # Imbalance from dominant frequency (if at 1X running speed ~ 30Hz)
            # Check ax/ay dominant frequency
            ax_dom = features.get("ax_dom_freq", 0.0)
            if 28.0 <= ax_dom <= 32.0 and features.get("ax_rms", 0.0) > 0.5:
                imbalance = "HIGH"
            else:
                imbalance = "LOW"
                
            return {
                "vibration_severity": severity,
                "bearing_condition": bearing_cond,
                "imbalance_indicator": imbalance
            }
        except Exception as e:
            logger.error(f"Error getting health indicators: {e}")
            return {}

    def stream_generator(self) -> Generator[Dict[str, Any], None, None]:
        """Yields JSON-serializable dicts with timestamp, data, and features."""
        while True:
            try:
                timestamp = time.time()
                raw_window = self.read_window()
                filtered_window = self.apply_butterworth_filter(raw_window)
                features = self.compute_features(filtered_window)
                health = self.get_health_indicators(features)
                
                yield {
                    "timestamp": timestamp,
                    "features": features,
                    "health": health
                }
            except Exception as e:
                logger.error(f"Error in stream generator: {e}")
                time.sleep(1)
