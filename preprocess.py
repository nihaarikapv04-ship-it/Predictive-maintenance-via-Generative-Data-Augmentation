import os
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import joblib  # To save your scaler for later use

def preprocess_bearing_data(input_folder, output_folder, seq_len=24):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. Find all .mat files (including .mat.txt files)
    files = [f for f in os.listdir(input_folder) if f.endswith('.mat') or f.endswith('.mat.txt')]
    
    for file_name in files:
        print(f"Processing: {file_name}...")
        file_path = os.path.join(input_folder, file_name)
        
        # 2. Load the MATLAB file
        raw_data = loadmat(file_path)
        
        # 3. Find the Drive End (DE) vibration key automatically
        # CWRU keys usually look like 'X097_DE_time' or 'X105_DE_time'
        de_key = [k for k in raw_data.keys() if 'DE_time' in k][0]
        signals = raw_data[de_key].flatten()
        
        # 4. Scale the data (0 to 1)
        # This is mandatory for GAN stability!
        scaler = MinMaxScaler(feature_range=(0, 1))
        signals_scaled = scaler.fit_transform(signals.reshape(-1, 1))
        
        # 5. Slice into sequences (Windowing)
        # TimeGAN needs "windows" of data, not one long line
        n_sequences = len(signals_scaled) // seq_len
        processed_data = signals_scaled[:n_sequences * seq_len].reshape(n_sequences, seq_len, 1)
        
        # 6. Save the processed data and the scaler
        # Saving the scaler is important so you can "un-scale" your GenAI data later
        base_name = file_name.replace('.mat', '')
        np.save(os.path.join(output_folder, f"{base_name}_processed.npy"), processed_data)
        joblib.dump(scaler, os.path.join(output_folder, f"{base_name}_scaler.pkl"))
        
        print(f"✅ Saved {processed_data.shape[0]} sequences to {output_folder}")

if __name__ == "__main__":
    # Define your paths based on the structure we built
    RAW_DIR = "./Data/raw/"
    PROCESSED_DIR = "./Data/processed/"
    
    preprocess_bearing_data(RAW_DIR, PROCESSED_DIR)
