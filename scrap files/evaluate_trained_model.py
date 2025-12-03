from ultralytics import YOLO
import os
import glob

# Use new segmentation model with 4 classes
MODEL_PATH = 'runs/segment/sitam_uniform_detector_v2/weights/best.pt'
DATASET_IMAGES = 'dataset/valid/images/*.jpg'

# All classes present in new trained model
REQUIRED_CLASSES = {
    'id_card': ['id card', 'id-card', 'idcard'],
    'pant': ['pant', 'pants', 'trousers'],
    'shirt': ['shirt'],
    'shoes': ['shoes', 'shoe']
}


def load_model():
    if not os.path.exists(MODEL_PATH):
        print('Model not found at', MODEL_PATH)
        print('Train the model first (python train_model.py).')
        return None
    print('Loading model:', MODEL_PATH)
    return YOLO(MODEL_PATH)


def passes_uniform_rules(preds):
    labels = [p.cls if hasattr(p, 'cls') else None for p in preds]
    # Ultralytics result objects typically have .names mapping; use that
    return None


def evaluate_on_images(model):
    # Test with validation, test, and train images from new dataset
    base_path = 'dataset/Sitam_Uniform_Detection1.v2-dataset.yolov8'
    val_images = glob.glob(f'{base_path}/valid/images/*.jpg')
    test_images_list = glob.glob(f'{base_path}/test/images/*.jpg')
    train_images = glob.glob(f'{base_path}/train/images/*.jpg')
    
    print("=== Testing on VALIDATION images ===")
    val_acc = test_images(model, val_images)
    
    print("\n=== Testing on TEST images ===")
    test_acc = test_images(model, test_images_list)
    
    print("\n=== Testing on TRAIN images (sanity check) ===")
    train_acc = test_images(model, train_images[:10])  # first 10 only
    
    print("\n" + "="*60)
    print("📊 OVERALL ACCURACY SUMMARY")
    print("="*60)
    print(f"Validation Set: {val_acc:.1f}%")
    print(f"Test Set:       {test_acc:.1f}%")
    print(f"Train Set:      {train_acc:.1f}%")
    print("="*60)


def test_images(model, image_paths):
    if not image_paths:
        print('No images found.')
        return 0.0
    ok, total = 0, 0
    
    # Track detection stats for each class
    class_stats = {
        'id_card': {'detected': 0, 'total': 0},
        'pant': {'detected': 0, 'total': 0},
        'shirt': {'detected': 0, 'total': 0},
        'shoes': {'detected': 0, 'total': 0}
    }
    
    for img in image_paths:  # evaluate all images
        total += 1
        results = model(img, verbose=False, conf=0.01)
        names = results[0].names
        detections = results[0].boxes
        classes = [names[int(c.item())] for c in detections.cls] if len(detections.cls) > 0 else []
        confs = [float(cf.item()) for cf in detections.conf] if len(detections.conf) > 0 else []
        
        # Check each component
        has_id_card = any(c in REQUIRED_CLASSES['id_card'] for c in classes)
        has_pant = any(c in REQUIRED_CLASSES['pant'] for c in classes)
        has_shirt = any(c in REQUIRED_CLASSES['shirt'] for c in classes)
        has_shoes = any(c in REQUIRED_CLASSES['shoes'] for c in classes)
        
        # Update stats
        class_stats['id_card']['total'] += 1
        class_stats['pant']['total'] += 1
        class_stats['shirt']['total'] += 1
        class_stats['shoes']['total'] += 1
        
        if has_id_card: class_stats['id_card']['detected'] += 1
        if has_pant: class_stats['pant']['detected'] += 1
        if has_shirt: class_stats['shirt']['detected'] += 1
        if has_shoes: class_stats['shoes']['detected'] += 1
        
        # Complete uniform = ALL 4 components
        uniform = has_id_card and has_pant and has_shirt and has_shoes
        ok += 1 if uniform else 0
        
        # Get top detections for display
        top_detections = list(zip(classes, confs))[:6]
        status_icons = f"{'✓' if has_shirt else '✗'}shirt {'✓' if has_pant else '✗'}pant {'✓' if has_shoes else '✗'}shoes {'✓' if has_id_card else '✗'}id"
        print(f"{os.path.basename(img):60} -> {'1' if uniform else '0'} | {status_icons} | {top_detections[:3]}")
    
    accuracy = (ok / total * 100) if total > 0 else 0
    print(f"\nSummary: {ok}/{total} images passed (all 4 components detected) - Accuracy: {accuracy:.1f}%")
    
    # Print per-class accuracy
    print("\nPer-Class Detection Rates:")
    for cls_name, stats in class_stats.items():
        rate = (stats['detected'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {cls_name.ljust(10)}: {stats['detected']}/{stats['total']} = {rate:.1f}%")
    
    return accuracy


if __name__ == '__main__':
    model = load_model()
    if model:
        evaluate_on_images(model)
