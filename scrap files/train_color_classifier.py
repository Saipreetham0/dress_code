# collect_and_label_crops.py
# Run this to create crop images and a CSV you can manually label (color labels).
# Edit MODEL_PATH to your trained YOLOv8 (local) or set use_roboflow=True and fill RF_MODEL_ID.

import os
from ultralytics import YOLO
import cv2
import csv
from tqdm import tqdm

# CONFIG
MODEL_PATH = "runs/your_yolov8_model/best.pt"  # or yolov8s.pt if testing
USE_ROBOFLOW = False
ROBofLOW_MODEL_ID = "PUT_ROBOFLOW_MODEL_ID_HERE"  # if using Roboflow SDK route
INPUT_IMAGES_DIR = "images_to_label"   # put your 43 images here
CROPS_OUT_DIR = "crops_dataset"
CSV_OUT = "crops_labels.csv"           # will contain one row per crop: crop_path,class,orig_image

os.makedirs(CROPS_OUT_DIR, exist_ok=True)

if not USE_ROBOFLOW:
    model = YOLO(MODEL_PATH)

rows = []
idx = 0
for fname in tqdm(sorted(os.listdir(INPUT_IMAGES_DIR))):
    if not fname.lower().endswith(('.jpg','.png','.jpeg')): 
        continue
    path = os.path.join(INPUT_IMAGES_DIR, fname)
    img = cv2.imread(path)
    if img is None:
        print("Failed to read", path); continue

    # Run inference (Ultralytics)
    if not USE_ROBOFLOW:
        results = model.predict(path, imgsz=640, conf=0.1, verbose=False)
        res = results[0]
        boxes = res.boxes
        for b in boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_name = f"crop_{idx}_{fname}"
            crop_path = os.path.join(CROPS_OUT_DIR, crop_name)
            cv2.imwrite(crop_path, crop)
            rows.append([crop_path, cls, fname, conf])
            idx += 1
    else:
        # Placeholder: use Roboflow SDK inference here and append crops in same format
        raise NotImplementedError("Roboflow path not implemented in this script; set USE_ROBOFLOW=False or implement RF inference.")

# Write CSV for manual labeling
with open(CSV_OUT, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['crop_path', 'class_id', 'orig_image', 'conf'])
    writer.writerows(rows)

print("Done. Annotate CSV", CSV_OUT, "with a color label column 'color' manually (e.g. gray, black, blue, red, none').")
