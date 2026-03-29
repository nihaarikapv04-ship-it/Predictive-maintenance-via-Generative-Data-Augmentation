import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

# 1. LOAD PROCESSED DATA
data_path = "./Data/processed/IR007_0_processed.npy"
if not os.path.exists(data_path):
    print("❌ Error: Processed data not found. Run preprocess.py first!")
    exit()

X_train = np.load(data_path)
print(f"✅ Loaded {X_train.shape[0]} sequences. Shape: {X_train.shape}")
print(f"Ready to train TimeGAN.")

# 2. DEVICE & HYPERPARAMETERS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

seq_len = X_train.shape[1]   # 24
feature_dim = X_train.shape[2]   # 1
hidden_dim = 32
noise_dim = 10
batch_size = 128
epochs = 2000  
lr_g = 0.001       # Aggressive Generator
lr_d = 0.00001     # Slow Discriminator for stability

# 3. TIMEGAN COMPONENTS
class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, seq_len, feature_dim):
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

class Discriminator(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.fc1 = nn.Linear(seq_len, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.shape[0], -1) 
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        prediction = torch.sigmoid(self.fc3(x))
        return prediction

# 4. INITIALIZE MODELS
generator = Generator(noise_dim, hidden_dim, seq_len, feature_dim).to(device)
discriminator = Discriminator(seq_len).to(device)

g_optimizer = Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
d_optimizer = Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

criterion = nn.BCELoss()

# --- CHECKPOINT LOADER ---
checkpoint_path = "./models/timegan_faulty_checkpoint.pth"
start_epoch = 0
if os.path.exists(checkpoint_path):
    print(f"📂 Found checkpoint! Resuming from previous state...")
    ckpt = torch.load(checkpoint_path)
    generator.load_state_dict(ckpt['gen_state'])
    discriminator.load_state_dict(ckpt['disc_state'])
    g_optimizer.load_state_dict(ckpt['g_opt'])
    d_optimizer.load_state_dict(ckpt['d_opt'])
    start_epoch = ckpt['epoch']

# 5. PREPARE DATA
X_train_tensor = torch.FloatTensor(X_train).to(device)
dataset = TensorDataset(X_train_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 6. STABILIZED TRAINING LOOP (3-to-1 RULE)
print(f"\n🚀 Training for {epochs} total epochs. Starting at {start_epoch}...")

for epoch in range(start_epoch, epochs):
    for step, (real_data,) in enumerate(dataloader):
        batch_curr = real_data.shape[0]

        # --- STEP A: TRAIN GENERATOR (3-to-1 Rule) ---
        # Generator gets 3 practice rounds for every 1 round the Discriminator gets
        for _ in range(3):
            z = torch.randn(batch_curr, noise_dim).to(device)
            fake_data = generator(z)
            d_fake = discriminator(fake_data)
            
            # Goal: Fool discriminator (Target = 1.0)
            g_loss = criterion(d_fake, torch.ones_like(d_fake))
            
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        # --- STEP B: TRAIN DISCRIMINATOR (With Label Smoothing) ---
        d_real = discriminator(real_data)
        d_fake = discriminator(fake_data.detach())

        # Label Smoothing: 0.9 and 0.1 prevents the D_Loss from hitting 0.0000
        real_labels = torch.full_like(d_real, 0.9)
        fake_labels = torch.full_like(d_fake, 0.1)

        d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

    # Log Progress & Save Checkpoint every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
        if not os.path.exists("./models"):
            os.makedirs("./models")
        torch.save({
    'generator': generator.state_dict(),
    'discriminator': discriminator.state_dict(),
    'seq_len': seq_len,
    'noise_dim': noise_dim,
    'hidden_dim': hidden_dim,
    'feature_dim': feature_dim
}, "./models/timegan_inner_race_fault.pth")

print("\n✅ Training Complete!")

# 7. SAVE FINAL MODEL
torch.save({
    'generator': generator.state_dict(),
    'discriminator': discriminator.state_dict(),
    'seq_len': seq_len,
    'noise_dim': noise_dim,
    'hidden_dim': hidden_dim,
    'feature_dim': feature_dim
}, "./models/timegan_inner_race_fault.pth")
print("✅ Final Model saved to ./models/timegan_inner_race_fault.pth")

# 8. GENERATE & PREVIEW
print("\n📊 Generating synthetic data for preview...")
generator.eval()
with torch.no_grad():
    z = torch.randn(5, noise_dim).to(device)
    synthetic_data = generator(z).cpu().numpy()

# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(X_train[0, :, 0], label='Real Vibration (CWRU)', color='blue', linewidth=2)
plt.title('Real Healthy Motor Vibration')
plt.xlabel('Time Step')
plt.ylabel('Normalized Amplitude')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
for i in range(3):
    plt.plot(synthetic_data[i, :, 0], label=f'GenAI Synthetic {i+1}', linestyle='--', alpha=0.7)
plt.plot(X_train[0, :, 0], label='Real (Reference)', color='black', linewidth=2)
plt.title('GenAI Generated vs Real')
plt.xlabel('Time Step')
plt.ylabel('Normalized Amplitude')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./models/timegan_preview.png", dpi=100)
print("✅ Preview saved to ./models/timegan_preview.png")
plt.show()