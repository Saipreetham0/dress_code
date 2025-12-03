from ultralytics import YOLO
import os
from pathlib import Path

# New dataset with 4 classes: id card, pant, shirt, shoes
DATASET_PATH = 'dataset/Sitam_Uniform_Detection1.v2-dataset.yolov8/data.yaml'

def is_segmentation_dataset(data_yaml_path: str) -> bool:
    """Auto-detect if dataset is for segmentation (vs detection)"""
    try:
        base = Path(data_yaml_path).parent
        train_labels = base / 'train' / 'labels'
        for p in train_labels.glob('*.txt'):
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    # Segmentation has >5 tokens (class + many x,y pairs)
                    if len(parts) > 5:
                        return True
                    if len(parts) == 5:
                        return False
        return False
    except Exception:
        return False

print("="*60)
print("TRAINING UNIFORM DETECTOR - 4 CLASSES")
print("="*60)
print(f"Dataset: {DATASET_PATH}")

if not os.path.exists(DATASET_PATH):
    print(f"\n❌ ERROR: Dataset not found at {DATASET_PATH}")
    exit(1)

print("✓ Dataset found")

seg = is_segmentation_dataset(DATASET_PATH)
base_model = 'yolov8n-seg.pt' if seg else 'yolov8n.pt'
project_dir = 'runs/segment' if seg else 'runs/detect'

print(f"Dataset type: {'Segmentation' if seg else 'Detection'}")
print(f"Base model: {base_model}")
print(f"Output: {project_dir}/sitam_uniform_detector_v2/")
print("\n🚀 Starting training...")
print("="*60)

model = YOLO(base_model)

results = model.train(
    data=DATASET_PATH,
    epochs=50,
    imgsz=640,
    batch=8,
    name='sitam_uniform_detector_v2',
    patience=10,
    save=True,
    plots=True,
    device='cpu',
    workers=2,
    project=project_dir
)

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"\nBest model: {project_dir}/sitam_uniform_detector_v2/weights/best.pt")
print("\nClasses trained: id card, pant, shirt, shoes")
print("="*60)
