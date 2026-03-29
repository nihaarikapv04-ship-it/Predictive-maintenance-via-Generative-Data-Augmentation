import os
from roboflow import Roboflow

# 1. SET THE DESTINATION
# This ensures images go to Induction_Motor_Project/Data/vision
target_path = os.path.join(os.getcwd(), "Data", "vision")

# 2. INITIALIZE WITH YOUR KEY
# PASTE THE KEY YOU FOUND IN STEP 1 BELOW:
my_key = "8Qpc6WcPpDkO4x5eId9j"

rf = Roboflow(api_key=my_key)
project = rf.workspace("steel-defects-sdxqu").project("steel-loose-bolt")
version = project.version(2)

# 3. THE DOWNLOAD
print(f"🚚 Starting download to: {target_path}")
dataset = version.download(model_format="yolov11", location=target_path)

print(f"✅ Success! Your 'Vision Ingredients' are ready in Data/vision.")