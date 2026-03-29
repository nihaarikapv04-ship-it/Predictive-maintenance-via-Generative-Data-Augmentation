import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the SPECIFIC Faulty Scaler
scaler = joblib.load('./Data/processed/IR007_0_scaler.pkl')

# 2. Load the trained Faulty Generator
# (Make sure your Generator class is defined above this in your script)
# checkpoint = torch.load("./models/timegan_inner_race_fault.pth")
# generator.load_state_dict(checkpoint['generator'])

# 3. Generate and Unscale
# with torch.no_grad():
#     z = torch.randn(1, 10).to(device)
#     fake_scaled = generator(z).cpu().numpy().reshape(-1, 1)
#     real_units = scaler.inverse_transform(fake_scaled)

# print(f"Maximum Synthetic Vibration: {np.max(real_units):.4f} g")