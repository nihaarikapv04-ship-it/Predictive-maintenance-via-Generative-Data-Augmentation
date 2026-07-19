import torch
import torch.nn as nn
import numpy as np
import os

# 1. THE BRAIN ARCHITECTURE
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

# 2. SETUP & LOADING
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_dim = 10
samples_per_class = 5000  # Total 10,000 samples

gen_h = Generator().to(device)
gen_f = Generator().to(device)

# Load the trained brains
gen_h.load_state_dict(torch.load("./models/timegan_healthy.pth", map_location=device)['generator'])
gen_f.load_state_dict(torch.load("./models/timegan_inner_race_fault.pth", map_location=device)['generator'])

gen_h.eval()
gen_f.eval()

# 3. MASS GENERATION
print(f"🚀 Generating {samples_per_class * 2} synthetic samples...")

with torch.no_grad():
    # Generate Healthy (Label 0)
    z_h = torch.randn(samples_per_class, noise_dim).to(device)
    healthy_data = gen_h(z_h).cpu().numpy()
    healthy_labels = np.zeros(samples_per_class) # Label 0
    
    # Generate Faulty (Label 1)
    z_f = torch.randn(samples_per_class, noise_dim).to(device)
    faulty_data = gen_f(z_f).cpu().numpy()
    faulty_labels = np.ones(samples_per_class) # Label 1

# 4. COMBINE AND SHUFFLE
X = np.concatenate([healthy_data, faulty_data], axis=0)
y = np.concatenate([healthy_labels, faulty_labels], axis=0)

# Shuffle so the AI doesn't just learn "all healthy then all faulty"
indices = np.random.permutation(len(X))
X, y = X[indices], y[indices]

# 5. SAVE THE MASTER DATASET
os.makedirs("./Data/master/", exist_ok=True)
np.save("./Data/master/X_synthetic.npy", X)
np.save("./Data/master/y_synthetic.npy", y)

print(f"✅ Success! Master dataset saved to ./Data/master/")
print(f"📊 Data Shape: {X.shape} | Labels Shape: {y.shape}")