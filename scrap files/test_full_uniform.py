"""Test full uniform detection on a single image"""
import cv2
from ultralytics import YOLO
import os

MODEL_PATH = 'runs/segment/sitam_uniform_detector_v2/weights/best.pt'
TEST_IMAGE = 'dataset/Sitam_Uniform_Detection1.v2-dataset.yolov8/valid/images/Complete_Uniform2_jpeg.rf.d24b00660b57d930e721f25f85429402.jpg'
DETECTION_CONF = 0.3

if not os.path.exists(MODEL_PATH):
    print(f"Model not found: {MODEL_PATH}")
    exit(1)

if not os.path.exists(TEST_IMAGE):
    print(f"Test image not found: {TEST_IMAGE}")
    exit(1)

print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

print(f"Testing: {os.path.basename(TEST_IMAGE)}\n")
results = model(TEST_IMAGE, verbose=False, conf=DETECTION_CONF)

# Get detections
names = results[0].names
boxes = results[0].boxes
classes = [names[int(c.item())] for c in boxes.cls] if len(boxes.cls) > 0 else []
confs = [float(cf.item()) for cf in boxes.conf] if len(boxes.conf) > 0 else []

print(f"Detections found: {len(classes)}")
for cls, conf in zip(classes, confs):
    print(f"  • {cls}: {conf:.2%}")

# Check for all 4 uniform components
has_shirt = any('shirt' in c.lower() for c in classes)
has_pant = any('pant' in c.lower() for c in classes)
has_shoes = any('shoe' in c.lower() for c in classes)
has_id_card = any('id' in c.lower() or 'card' in c.lower() for c in classes)

# Full uniform = ALL 4 components
uniform_complete = has_shirt and has_pant and has_shoes and has_id_card

print(f"\n{'='*50}")
print("FULL UNIFORM CHECK")
print('='*50)
print(f"  ✓ Shirt   : {'YES ✓' if has_shirt else 'NO ✗'}")
print(f"  ✓ Pant    : {'YES ✓' if has_pant else 'NO ✗'}")
print(f"  ✓ Shoes   : {'YES ✓' if has_shoes else 'NO ✗'}")
print(f"  ✓ ID Card : {'YES ✓' if has_id_card else 'NO ✗'}")
print('='*50)

if uniform_complete:
    print("  OUTPUT: 1 (FULL UNIFORM) ✅")
else:
    print("  OUTPUT: 0 (NOT IN UNIFORM) ❌")
print('='*50)

# Show annotated image
annotated = results[0].plot()
cv2.imshow('Full Uniform Detection Test', annotated)
print("\nPress any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()
