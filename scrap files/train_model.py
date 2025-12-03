from ultralytics import YOLO
import os
from pathlib import Path

"""
SITAM COLLEGE UNIFORM DETECTION - TRAINING SCRIPT

Uniform Requirements (Complete Uniform):
- Gray shirt (tucked in)
- Black pants  
- Shoes (any color)
- ID card (red/yellow/green/pink on chest)

Missing ANY item = No Uniform
"""

def train_uniform_detector():
    print("=" * 60)
    print("SITAM COLLEGE UNIFORM DETECTOR - TRAINING")
    print("=" * 60)
    
    # Check if you have the dataset from Roboflow
    print("\n⚠️  BEFORE RUNNING THIS SCRIPT:")
    print("1. Complete Phase 2: Label all images in Roboflow")
    print("2. Export dataset in YOLOv8 format")
    print("3. Download and extract to: D:/Smart_Vechicle/dataset/")
    print("\nDataset structure should be:")
    print("  dataset/")
    print("    ├── train/")
    print("    ├── valid/")
    print("    └── data.yaml")
    
    dataset_path = input("\nEnter path to data.yaml file (or press Enter for default): ").strip()
    
    if not dataset_path:
        dataset_path = "D:/Smart_Vechicle/dataset/data.yaml"
    
    if not os.path.exists(dataset_path):
        print(f"\n❌ ERROR: Dataset not found at {dataset_path}")
        print("\nPlease:")
        print("1. Download dataset from Roboflow")
        print("2. Extract to D:/Smart_Vechicle/dataset/")
        print("3. Run this script again")
        return
    
    print(f"\n✓ Dataset found: {dataset_path}")
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print("Model: YOLOv8 Nano (fastest)")
    print("Epochs: 50 (can increase for better accuracy)")
    print("Image Size: 640x640")
    print("Batch Size: 8")
    print("\nEstimated Training Time: 30 minutes - 2 hours")
    print("=" * 60)
    
    start = input("\nStart training? (y/n): ").strip().lower()
    
    if start != 'y':
        print("Training cancelled.")
        return
    
    print("\n🚀 Starting training...")
    print("=" * 60)
    
    # Choose base model: detection vs segmentation (auto-detect from label format)
    def is_segmentation_dataset(data_yaml_path: str) -> bool:
        try:
            base = Path(data_yaml_path).parent
            train_labels = base / 'train' / 'labels'
            # pick any .txt file
            for p in train_labels.glob('*.txt'):
                with open(p, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        # YOLO detection has 5 items per line (class cx cy w h)
                        # Segmentation has class followed by many x y pairs (>5 tokens)
                        if len(parts) > 5:
                            return True
                        if len(parts) == 5:
                            return False
            return False
        except Exception:
            return False

    seg = is_segmentation_dataset(dataset_path)
    base_model = 'yolov8n-seg.pt' if seg else 'yolov8n.pt'
    project_dir = 'runs/segment' if seg else 'runs/detect'

    print(f"\nDetected dataset type: {'Segmentation' if seg else 'Detection'}")
    print(f"Using base model: {base_model}")

    model = YOLO(base_model)
    
    # Train the model
    results = model.train(
        data=dataset_path,
        epochs=50,           # Number of training epochs
        imgsz=640,           # Image size
        batch=8,             # Batch size
        name='sitam_uniform_detector',  # Model name
        patience=10,         # Early stopping patience
        save=True,           # Save checkpoints
        plots=True,          # Generate training plots
        device='cpu',        # Use CPU (change to 0 for GPU)
        workers=2,           # Number of workers
        project=project_dir  # Output directory
    )
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved to: {project_dir}/sitam_uniform_detector/weights/best.pt")
    print("\nNext Steps:")
    print("1. Check training results in: runs/detect/sitam_uniform_detector/")
    print("2. View plots: results.png, confusion_matrix.png")
    print("3. Test the model with: python test_trained_model.py")
    print("=" * 60)


if __name__ == '__main__':
    train_uniform_detector()
