import cv2
from ultralytics import YOLO
import os

# 1. Load your model (ensure best.pt is in the same folder)
model = YOLO('best.pt')

# 2. Path to your local video
input_video_path = 'fire.mp4' 
output_video_path = 'detected_fire_output.mp4'

# 3. Run Inference
# 'save=True' handles the video writing automatically
# 'device=0' ensures your RTX 3050 is doing the work
results = model.predict(
    source=input_video_path,
    save=True,
    project="local_runs",  # Creates a folder named local_runs
    name="fire_test",      # Subfolder name
    device=0,              # Uses your NVIDIA GPU
    conf=0.4               # Confidence threshold
)

print(f"Done! Check the 'local_runs/fire_test' folder for your output.")