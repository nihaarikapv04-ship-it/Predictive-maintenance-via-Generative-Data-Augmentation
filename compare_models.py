import os
import torch
import torch.nn as nn
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. DEFINE THE GENERATOR
class Generator(nn.Module):
    def __init__(self, noise_dim=10, hidden_dim=32, seq_len=24, feature_dim=1):
        super().__init__()
        self.seq_len = seq_len
        self.fc1 = nn.Linear(noise_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, seq_len * feature_dim)
        self.relu = nn.ReLU()
        
    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.seq_len, 1)
        return torch.sigmoid(x)

# 2. PATH CHECKING (Solves the FileNotFoundError)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Let's define the paths based on your actual folder screenshot
h_scaler_path = './Data/processed/97.txt_scaler.pkl'
f_scaler_path = './Data/processed/IR007_0_scaler.pkl'
h_model_path  = "./models/timegan_healthy.pth"
f_model_path  = "./models/timegan_inner_race_fault.pth"

# Quick Diagnostic Print
print("🔍 Checking for files...")
for p in [h_scaler_path, f_scaler_path, h_model_path, f_model_path]:
    if os.path.exists(p):
        print(f"✅ Found: {p}")
    else:
        print(f"❌ MISSING: {p}")

# Load Healthy Scaler and Model
scaler_h = joblib.load(h_scaler_path)
gen_h = Generator().to(device)
ckpt_h = torch.load(h_model_path, map_location=device)
gen_h.load_state_dict(ckpt_h['generator'])

# Load Faulty Scaler and Model
scaler_f = joblib.load(f_scaler_path)
gen_f = Generator().to(device)
ckpt_f = torch.load(f_model_path, map_location=device)
gen_f.load_state_dict(ckpt_f['generator'])

# 3. GENERATE SYNTHETIC SAMPLES
gen_h.eval()
gen_f.eval()

with torch.no_grad():
    z = torch.randn(1, 10).to(device)
    # Healthy Generation
    synth_h = gen_h(z).cpu().numpy().reshape(-1, 1)
    real_units_h = scaler_h.inverse_transform(synth_h)
    
    # Faulty Generation
    synth_f = gen_f(z).cpu().numpy().reshape(-1, 1)
    real_units_f = scaler_f.inverse_transform(synth_f)

# 4. PLOT THE COMPARISON
plt.figure(figsize=(14, 6))

# Healthy Plot (Blue)
plt.subplot(1, 2, 1)
plt.plot(real_units_h, color='blue', linewidth=2)
plt.title('GenAI: Healthy Motor Baseline', fontsize=14)
plt.ylabel('Vibration (g)')
plt.grid(True, alpha=0.3)
plt.ylim([-0.6, 0.6])

# Faulty Plot (Red)
plt.subplot(1, 2, 2)
plt.plot(real_units_f, color='red', linewidth=2)
plt.title('GenAI: Inner Race Fault (IR007)', fontsize=14)
plt.ylabel('Vibration (g)')
plt.grid(True, alpha=0.3)
plt.ylim([-0.6, 0.6])

plt.suptitle('Predictive Maintenance: Healthy vs. Faulty Synthetic Vibration', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("./models/final_comparison_plot.png", dpi=300)
print("\n🎉 Victory! Plot saved to ./models/final_comparison_plot.png")
plt.show()