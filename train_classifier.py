import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split

print("🟢 [STATUS]: Script Started. Initializing AI Brain...")

# 1. LOAD THE MASTER DATASET
data_path = "./Data/master/X_synthetic.npy"
labels_path = "./Data/master/y_synthetic.npy"

if not os.path.exists(data_path):
    print(f"❌ Error: Cannot find {data_path}. Run generate_master_data.py first!")
    exit()

X = np.load(data_path)
y = np.load(labels_path)

# Convert to PyTorch Tensors
X_tensor = torch.FloatTensor(X).transpose(1, 2) 
y_tensor = torch.LongTensor(y)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# 2. THE CNN ARCHITECTURE
class FaultClassifier(nn.Module):
    def __init__(self):
        super(FaultClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(32 * 6, 64) 
        self.fc2 = nn.Linear(64, 2)      
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# 3. TRAINING
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaultClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"🚀 Training on {device}...")
for epoch in range(5): # Doing 5 epochs for a quick victory
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"✅ Epoch {epoch+1}/5 | Loss: {loss.item():.4f}")

# 4. FINAL ACCURACY
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

print(f"\n🎯 FINAL ACCURACY: {100 * correct / total:.2f}%")

# 5. SAVE
torch.save(model.state_dict(), "./models/motor_fault_classifier.pth")
print("💾 Brain saved as: ./models/motor_fault_classifier.pth")