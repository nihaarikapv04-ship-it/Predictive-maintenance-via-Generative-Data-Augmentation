import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

def prepare_data(file_path, key, seq_len=24):
    # 1. Load MATLAB file
    data = loadmat(file_path)
    signals = data[key].flatten()
    
    # 2. Scale data (Mandatory for GAN stability)
    scaler = MinMaxScaler()
    signals_scaled = scaler.fit_transform(signals.reshape(-1, 1))
    
    # 3. Slice into sequences (Windowing)
    # This turns 1 long line of data into many small chunks of 24 points
    n_sequences = len(signals_scaled) // seq_len
    data_reshaped = signals_scaled[:n_sequences * seq_len].reshape(n_sequences, seq_len, 1)
    
    return data_reshaped, scaler

# Test it with your Healthy Baseline
# Replace './data/raw/97.mat' with your actual path
X_train, motor_scaler = prepare_data('./Data/raw/97.mat.txt', 'X097_DE_time')
print(f"Prepared {X_train.shape[0]} sequences for training!")