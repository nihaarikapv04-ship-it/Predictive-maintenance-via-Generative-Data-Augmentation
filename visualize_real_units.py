import joblib
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the "Scaler" we saved during preprocessing
scaler = joblib.load('./data/processed/97.txt_scaler.pkl')

# 2. Load your synthetic data (assuming you saved it as a .npy)
# For now, let's just use the synthetic sample from your plot
synthetic_scaled = np.array([[0.51, 0.48, 0.55, ...]]) # Your GenAI output

# 3. THE MAGIC: Inverse Transform
# This turns 0.5 back into real-world vibration values
real_units_data = scaler.inverse_transform(synthetic_scaled.reshape(-1, 1))

plt.plot(real_units_data)
plt.ylabel('Vibration (m/s²)')
plt.title('GenAI Generated Vibration in Real-World Units')
plt.show()