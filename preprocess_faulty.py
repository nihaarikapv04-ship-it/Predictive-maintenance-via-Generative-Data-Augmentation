import os
import numpy as np
import joblib
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

# 1. SETUP PATHS (Matches your diagnostic result)
RAW_DIR = "./Data/raw/"
PROCESSED_DIR = "./Data/processed/"
FILE_NAME = "105.mat.txt"  # <--- Updated to match your folder!

if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

def preprocess_faulty():
    file_path = os.path.join(RAW_DIR, FILE_NAME)
    
    if not os.path.exists(file_path):
        print(f"❌ Error: Still can't find {FILE_NAME} in {RAW_DIR}")
        print(f"Found these instead: {os.listdir(RAW_DIR)}")
        return

    # 2. LOAD MATLAB FILE
    print(f"📂 Loading Faulty Data: {FILE_NAME}")
    raw_data = loadmat(file_path)
    
    # 3. AUTOMATIC KEY FINDER
    # For file 105, the key is usually 'X105_DE_time'
    de_key = [k for k in raw_data.keys() if 'DE_time' in k][0]
    signals = raw_data[de_key].flatten()
    print(f"✅ Found Drive End (DE) signals using key: {de_key}")

    # 4. CREATE THE SPECIFIC FAULTY SCALER
    scaler = MinMaxScaler(feature_range=(0, 1))
    signals_scaled = scaler.fit_transform(signals.reshape(-1, 1))
    
    # 5. WINDOWING (24 steps for TimeGAN)
    seq_len = 24
    n_sequences = len(signals_scaled) // seq_len
    processed_data = signals_scaled[:n_sequences * seq_len].reshape(n_sequences, seq_len, 1)

    # 6. SAVE EVERYTHING
    output_npy = os.path.join(PROCESSED_DIR, "IR007_0_processed.npy")
    output_scaler = os.path.join(PROCESSED_DIR, "IR007_0_scaler.pkl")
    
    np.save(output_npy, processed_data)
    joblib.dump(scaler, output_scaler)

    print(f"\n--- Saturday Morning Stats ---")
    print(f"📊 Total Sequences: {processed_data.shape[0]}")
    print(f"💾 Scaler Saved: {output_scaler}")
    print(f"💾 Data Saved: {output_npy}")
    print(f"🚀 Status: Ready for Faulty GAN Training!")

if __name__ == "__main__":
    preprocess_faulty()