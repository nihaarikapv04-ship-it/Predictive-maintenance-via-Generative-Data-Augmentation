from ultralytics import YOLO
import cv2
import os
model_path= './runs/detect/runs/detect/motor_vision_fast/weights/best.pt'
model= YOLO(model_path)
test_image_path = './Data/vision/test/images/'
images = os.listdir(test_image_path)
sample_img = os.path.join(test_image_path, images[0])
results = model.predict(source=sample_img, conf=0.25, save=True)
print(f"Prediction done! Check 'runs/detect/predict' to see the image with boxes")