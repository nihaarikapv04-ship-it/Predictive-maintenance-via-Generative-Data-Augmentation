from ultralytics import YOLO
import os

# 1. LOAD THE BASE MODEL
# 'yolov8n.pt' is the 'Nano' version—it's small, fast, and perfect for the Pi.
print("🧠 Loading the YOLO base model...")
model = YOLO('yolo11n.pt') 

# 2. DEFINE THE DATA PATH
# This points to the data.yaml file that Roboflow created for you
dataset_yaml = os.path.join(os.getcwd(), "Data", "vision", "data.yaml")

# 3. START TRAINING
# We use 25 epochs so it finishes in about 15-20 minutes on a standard laptop.
print("🚀 Starting Team Compile Error Vision Training...")
results = model.train(
    data=dataset_yaml,
    epochs=10,        # 25 'laps' through the data
    imgsz=320,       # Standard image size
    device='cpu',    # Change to '0' if you have an NVIDIA GPU!
    project='runs/detect',
    name='motor_vision_fast'
)

print("\n🎉 VICTORY! Training is complete.")
print("Check this folder for your results: runs/detect/motor_vision_fast/weights/best.pt")