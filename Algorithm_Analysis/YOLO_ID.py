from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

# Load trained model
model = YOLO(
    "/home/robot/2d-lidar-identification/training_outputs/"
    "120_DO_simple_Fused/webots_model_assets_simpel_fused_fulltrain_120/"
    "yolov8n_lidar.pt"
)

print("✅ Model loaded successfully")

while True:
    # Load RGB LiDAR image
    rgb_image = cv2.imread(
        "/home/robot/scan_logs/rgb_output/current_rgb.png"
    )

    if rgb_image is None:
        print("Waiting for image...")
        time.sleep(0.5)
        continue

    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # Run inference
    results = model(
        rgb_image,
        imgsz=[64, 384],
        device="cpu",
        half=False,
        verbose=False
    )

    annotated = results[0].plot()

    temp_file = "/home/robot/scan_logs/latest_detection.tmp.jpg"
    final_file = "/home/robot/scan_logs/latest_detection.jpg"
    cv2.imwrite(temp_file, annotated)
    os.replace(temp_file, final_file)
    print("Detection image updated")

    time.sleep(0.5)
