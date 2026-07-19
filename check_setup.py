import os
import glob

# This script checks if your files are in the right place
base_path = "./data/raw/"
files = glob.glob(os.path.join(base_path, "*.mat"))

print("--- Compile Error Workspace Check ---")
if len(files) == 0:
    print("❌ ERROR: No .mat files found in data/raw/ folder.")
else:
    print(f"✅ SUCCESS: Found {len(files)} dataset files.")
    for f in files:
        print(f"   - {os.path.basename(f)}")