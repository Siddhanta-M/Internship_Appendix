from ultralytics import YOLO   # YOLO object detection model
import cv2                     # OpenCV for image loading and saving
import numpy as np             # Numerical operations (not directly used here)
import time                    # Pause between inference cycles
import os                      # Atomic file replacement

# ---------------------------------------------------------
# Load the trained YOLO model
# ---------------------------------------------------------

# Create a YOLO object using the trained model weights.
#
# These weights were produced during the training stage and
# have learned to recognize objects from RGB-encoded LiDAR images.
model = YOLO(
    "/home/robot/2d-lidar-identification/training_outputs/"
    "120_DO_simple_Fused/webots_model_assets_simpel_fused_fulltrain_120/"
    "yolov8n_lidar.pt"
)

print("✅ Model loaded successfully")

# ---------------------------------------------------------
# Continuous inference loop
# ---------------------------------------------------------

# Keep running until the program is manually stopped.
while True:

    # -----------------------------------------------------
    # Step 1: Load the latest RGB LiDAR image
    # -----------------------------------------------------

    # Read the most recently generated RGB image.
    #
    # This image is continuously updated by the RGB generation
    # script and represents the three most recent LiDAR scans.
    rgb_image = cv2.imread(
        "/home/robot/scan_logs/rgb_output/current_rgb.png"
    )

    # If no image exists yet, wait and try again.
    if rgb_image is None:
        print("Waiting for image...")
        time.sleep(0.5)
        continue

    # -----------------------------------------------------
    # Step 2: Convert image colour format
    # -----------------------------------------------------

    # OpenCV loads images using BGR colour ordering.
    #
    # YOLO expects images in standard RGB format, so the
    # channels must be swapped before inference.
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # -----------------------------------------------------
    # Step 3: Run object detection
    # -----------------------------------------------------

    # Perform inference on the RGB LiDAR image.
    #
    # imgsz=[64,384]
    #     Matches the dimensions of the generated RGB images.
    #
    # device="cpu"
    #     Run inference on the CPU rather than a GPU.
    #
    # half=False
    #     Use full 32-bit precision instead of FP16.
    #
    # verbose=False
    #     Disable detailed console output.
    results = model(
        rgb_image,
        imgsz=[64, 384],
        device="cpu",
        half=False,
        verbose=False
    )

    # -----------------------------------------------------
    # Step 4: Draw detection results
    # -----------------------------------------------------

    # The first Results object contains all detections for
    # this image. The plot() function automatically draws:
    #
    # - Bounding boxes
    # - Class labels
    # - Confidence scores
    #
    # onto a copy of the input image.
    annotated = results[0].plot()

    # -----------------------------------------------------
    # Step 5: Save the annotated image
    # -----------------------------------------------------

    # Save to a temporary file first.
    temp_file = "/home/robot/scan_logs/latest_detection.tmp.jpg"

    # Final output image viewed by other applications.
    final_file = "/home/robot/scan_logs/latest_detection.jpg"

    # Write the annotated image.
    cv2.imwrite(temp_file, annotated)

    # Atomically replace the old image with the new one.
    #
    # This prevents another process from reading a partially
    # written image while it is still being saved.
    os.replace(temp_file, final_file)

    print("Detection image updated")

    # -----------------------------------------------------
    # Step 6: Wait before processing the next image
    # -----------------------------------------------------

    # The RGB generation script updates periodically, so a
    # short delay avoids unnecessary repeated processing of
    # the same image.
    time.sleep(0.5)
