"""Quick test of trained model on a single validation image"""
import cv2
from ultralytics import YOLO
import os

MODEL_PATH = 'runs/segment/sitam_uniform_detector/weights/best.pt'
TEST_IMAGE = 'dataset/valid/images/Complete_Uniform1_jpeg.rf.c619ea0eebda4ec08c51e3b5cf53bfbd.jpg'
DETECTION_CONF = 0.01  # Threshold to capture trained model detections

if not os.path.exists(MODEL_PATH):
    print(f"Model not found: {MODEL_PATH}")
    exit(1)

if not os.path.exists(TEST_IMAGE):
    print(f"Test image not found: {TEST_IMAGE}")
    exit(1)

print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

print(f"Running inference on: {TEST_IMAGE}")
results = model(TEST_IMAGE, verbose=False, conf=DETECTION_CONF)

# Get detections
names = results[0].names
boxes = results[0].boxes
classes = [names[int(c.item())] for c in boxes.cls] if len(boxes.cls) > 0 else []
confs = [float(cf.item()) for cf in boxes.conf] if len(boxes.conf) > 0 else []

print(f"\nDetections: {len(classes)}")
for cls, conf in zip(classes, confs):
    print(f"  - {cls}: {conf:.4f}")

# Apply uniform rule
has_shirt = 'shirt' in classes
has_shoes = 'shoes' in classes
uniform_ok = has_shirt and has_shoes

print(f"\nUniform Check:")
print(f"  Shirt: {'✓' if has_shirt else '✗'}")
print(f"  Shoes: {'✓' if has_shoes else '✗'}")
print(f"  Result: {'1 (OK)' if uniform_ok else '0 (INCOMPLETE)'}")

# Show annotated image
annotated = results[0].plot()
cv2.imshow('Detection Test', annotated)
print("\nPress any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()
