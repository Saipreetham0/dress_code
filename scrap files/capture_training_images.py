import cv2
import os
from datetime import datetime

# Training data folders
BASE_DIR = os.path.join(os.path.dirname(__file__), 'training_data', 'images')
CATEGORIES = {
    '1': ('uniform_complete', 'Complete Uniform (shirt, pants, shoes, ID card, tucked)'),
    '2': ('uniform_partial', 'Partial Uniform (missing some items)'),
    '3': ('no_uniform', 'No Uniform (casual clothes)')
}

def capture_training_images():
    """Interactive tool to capture training images"""
    print("=" * 60)
    print("TRAINING IMAGE CAPTURE TOOL")
    print("=" * 60)
    print("\nThis tool helps you capture images for training the uniform detector.")
    print("\nImage Categories:")
    for key, (folder, desc) in CATEGORIES.items():
        print(f"  {key}. {desc}")
    print("\nControls:")
    print("  Press '1', '2', or '3' - Select category and capture image")
    print("  Press 'q' - Quit")
    print("\n" + "=" * 60)
    
    # Ensure folders exist
    for folder, _ in CATEGORIES.values():
        os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return
    
    # Image counters
    counters = {folder: len(os.listdir(os.path.join(BASE_DIR, folder))) 
                for folder, _ in CATEGORIES.values()}
    
    current_category = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display frame with instructions
        display = frame.copy()
        
        # Show current stats
        y_offset = 30
        cv2.putText(display, "Training Image Capture", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 40
        
        for key, (folder, desc) in CATEGORIES.items():
            count = counters[folder]
            text = f"{key}. {folder}: {count} images"
            color = (0, 255, 255) if current_category == folder else (255, 255, 255)
            cv2.putText(display, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30
        
        cv2.putText(display, "Press 1/2/3 to capture, Q to quit", 
                   (10, display.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Training Image Capture', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif chr(key) in CATEGORIES:
            # Capture image
            category_key = chr(key)
            folder, desc = CATEGORIES[category_key]
            current_category = folder
            
            # Save image with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{folder}_{counters[folder]:04d}_{timestamp}.jpg"
            filepath = os.path.join(BASE_DIR, folder, filename)
            
            cv2.imwrite(filepath, frame)
            counters[folder] += 1
            
            print(f"✓ Captured: {filename} -> {desc}")
            
            # Flash effect
            flash = frame.copy()
            cv2.rectangle(flash, (0, 0), (flash.shape[1], flash.shape[0]), (255, 255, 255), -1)
            cv2.imshow('Training Image Capture', flash)
            cv2.waitKey(100)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "=" * 60)
    print("CAPTURE SUMMARY")
    print("=" * 60)
    total = 0
    for folder, desc in CATEGORIES.values():
        count = counters[folder]
        total += count
        print(f"  {folder}: {count} images")
    print(f"\nTotal images captured: {total}")
    print("\nImages saved to: training_data/images/")
    print("=" * 60)


def list_training_images():
    """Show summary of collected training images"""
    print("\n" + "=" * 60)
    print("TRAINING DATA SUMMARY")
    print("=" * 60)
    
    if not os.path.exists(BASE_DIR):
        print("No training data folder found.")
        return
    
    total = 0
    for folder, desc in CATEGORIES.values():
        folder_path = os.path.join(BASE_DIR, folder)
        if os.path.exists(folder_path):
            count = len([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            total += count
            print(f"\n{folder}:")
            print(f"  Description: {desc}")
            print(f"  Images: {count}")
            print(f"  Path: {folder_path}")
        else:
            print(f"\n{folder}: Folder not found")
    
    print(f"\n{'=' * 60}")
    print(f"TOTAL IMAGES: {total}")
    
    if total < 50:
        print(f"⚠️  Recommendation: Collect at least 50 images total (you have {total})")
        print("   - 20+ complete uniform images")
        print("   - 15+ partial uniform images")
        print("   - 15+ no uniform images")
    else:
        print("✓ Good amount of training data!")
    
    print("=" * 60)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        list_training_images()
    else:
        print("\n📸 Starting camera capture tool...")
        print("Make sure you have:")
        print("  1. Complete uniform ready (shirt, pants, shoes, ID card, tucked)")
        print("  2. Partial uniform variations")
        print("  3. Casual/non-uniform clothes")
        print("\nPress Enter to start...")
        input()
        capture_training_images()
        print("\n")
        list_training_images()
