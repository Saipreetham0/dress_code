# detect_and_classify_improved.py
# Enhanced version with better error handling and diagnostics
import cv2, os, numpy as np, joblib, sys
from ultralytics import YOLO
from sklearn.cluster import KMeans

# CONFIG
MODEL_PATH = "runs/train/uniform_color_trained2/weights/best.pt"
FALLBACK_MODEL_PATH = "runs/train/uniform_color_trained/weights/best.pt"
COLOR_CLF_PATH = "color_clf.joblib"
USE_COLOR_CLASSIFIER = os.path.exists(COLOR_CLF_PATH)
CONF_THRESH = 0.10   # Very low threshold to catch ID cards
MIN_AREA = 400       # Smaller threshold for distant objects
IMG_SIZE = 1280      # Larger image size for long-range detection (30m)

# UNIFORM VALIDATION RULES
SHIRT_ALLOWED = {'gray', 'white'}  # Allow both gray and white/light gray shirts
PANTS_ALLOWED = {'black', 'navy blue', 'blue', 'dark blue'}
SHOES_REQUIRED = True
ID_REQUIRED = True

# Spatial filtering
USE_SPATIAL_FILTERING = True

# Diagnostic mode
DIAGNOSTIC_MODE = False  # Set to True for detailed logging

def check_dependencies():
    """Check if all required packages are installed"""
    print("=== Checking Dependencies ===")
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not installed. Run: pip install opencv-python")
        return False

    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"✓ Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("✗ Ultralytics not installed. Run: pip install ultralytics")
        return False

    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("✗ PyTorch not installed. Run: pip install torch torchvision")
        return False

    try:
        from sklearn.cluster import KMeans
        print("✓ scikit-learn installed")
    except ImportError:
        print("✗ scikit-learn not installed. Run: pip install scikit-learn")
        return False

    print("✓ All dependencies installed\n")
    return True

def check_model_file():
    """Check if model file exists"""
    print("=== Checking Model File ===")
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"✓ Model found: {MODEL_PATH} ({size_mb:.2f} MB)")
        return MODEL_PATH
    elif os.path.exists(FALLBACK_MODEL_PATH):
        size_mb = os.path.getsize(FALLBACK_MODEL_PATH) / (1024 * 1024)
        print(f"⚠ Using fallback model: {FALLBACK_MODEL_PATH} ({size_mb:.2f} MB)")
        return FALLBACK_MODEL_PATH
    else:
        print(f"✗ Model not found at: {MODEL_PATH}")
        print(f"✗ Fallback model not found at: {FALLBACK_MODEL_PATH}")
        print("\nTo train a model, run:")
        print("  yolo detect train data=Uniform_Detection.v1i.yolov8/data.yaml model=yolov8n.pt epochs=100")
        return None

def check_webcam():
    """Test webcam accessibility"""
    print("=== Checking Webcam ===")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Could not open webcam (index 0)")
        print("\nTroubleshooting:")
        print("  1. Check if camera is connected")
        print("  2. Check if another app is using the camera")
        print("  3. Try running: python -c \"import cv2; print(cv2.VideoCapture(0).isOpened())\"")
        return False

    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        print(f"✓ Webcam accessible (resolution: {w}x{h})")
        cap.release()
        return True
    else:
        print("✗ Webcam opened but cannot read frames")
        cap.release()
        return False

# Load model
model = None
model_path = check_model_file()
if model_path:
    try:
        model = YOLO(model_path)
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}\n")
        sys.exit(1)
else:
    print("Cannot proceed without model file")
    sys.exit(1)

clf = None
if USE_COLOR_CLASSIFIER:
    clf = joblib.load(COLOR_CLF_PATH)
    print(f"✓ Loaded color classifier: {COLOR_CLF_PATH}\n")

# All helper functions from original
def crop_from_box(img, xyxy):
    x1,y1,x2,y2 = xyxy
    x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
    h,w = img.shape[:2]
    x1,x2 = max(0,min(x1,w-1)), max(0,min(x2,w-1))
    y1,y2 = max(0,min(y1,h-1)), max(0,min(y2,h-1))
    return img[y1:y2, x1:x2]

def kmeans_dominant_rgb(img_bgr, k=3):
    pixels = img_bgr.reshape(-1,3)
    if len(pixels) < 10:
        return np.array([0,0,0])
    pixels = pixels.astype(np.float32)
    km = KMeans(n_clusters=k, random_state=0).fit(pixels[:, ::-1])
    counts = np.bincount(km.labels_)
    dom = km.cluster_centers_[np.argmax(counts)]
    return dom

def hsv_gray_ratio(img_bgr, sat_thresh=60, v_min=30):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:,:,1]
    val = hsv[:,:,2]
    mask = (sat <= sat_thresh) & (val >= v_min)
    return float(np.sum(mask))/mask.size

def rgb_to_color_name(rgb):
    r,g,b = map(int, rgb)
    if r < 60 and g < 60 and b < 60:
        return 'black'
    if abs(r-g) < 20 and abs(g-b) < 20 and r > 120:
        return 'white'
    if abs(r-g) < 30 and abs(g-b) < 30 and r < 160 and g < 160 and b < 160:
        return 'gray'
    if b > r and b > g:
        return 'blue'
    if r > g and r > b:
        return 'red'
    if g > r and g > b:
        return 'green'
    return 'unknown'

def classify_color(img_bgr):
    if clf is not None:
        dom_rgb = kmeans_dominant_rgb(img_bgr, k=3)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h_mean, s_mean, v_mean = hsv[:,:,0].mean(), hsv[:,:,1].mean(), hsv[:,:,2].mean()
        low_sat_ratio = float(np.mean(hsv[:,:,1] < 60))
        feat = np.concatenate([dom_rgb, [h_mean, s_mean, v_mean, low_sat_ratio]])
        pred = clf.predict([feat])[0]
        return pred
    dom_rgb = kmeans_dominant_rgb(img_bgr, k=3)
    if hsv_gray_ratio(img_bgr, sat_thresh=60) > 0.4:
        return 'gray'
    return rgb_to_color_name(dom_rgb)

def detect_id_card_heuristic(img, shirt_box):
    """Fallback heuristic to detect ID card in chest region when model fails"""
    x1, y1, x2, y2 = shirt_box
    shirt_width = x2 - x1
    shirt_height = y2 - y1

    # Define chest region (center-top of shirt where ID cards typically hang)
    chest_x1 = x1 + int(shirt_width * 0.35)
    chest_x2 = x1 + int(shirt_width * 0.65)
    chest_y1 = y1 + int(shirt_height * 0.1)
    chest_y2 = y1 + int(shirt_height * 0.5)

    # Ensure bounds are valid
    h, w = img.shape[:2]
    chest_x1 = max(0, min(chest_x1, w-1))
    chest_x2 = max(0, min(chest_x2, w-1))
    chest_y1 = max(0, min(chest_y1, h-1))
    chest_y2 = max(0, min(chest_y2, h-1))

    if chest_x2 <= chest_x1 or chest_y2 <= chest_y1:
        return False

    # Crop chest region
    chest_crop = img[chest_y1:chest_y2, chest_x1:chest_x2]
    if chest_crop.size == 0:
        return False

    # Convert to HSV
    hsv = cv2.cvtColor(chest_crop, cv2.COLOR_BGR2HSV)

    # Check for white/cream colored objects (ID cards are typically light colored)
    lower_white = np.array([0, 0, 140])
    upper_white = np.array([180, 60, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_ratio = np.sum(white_mask > 0) / white_mask.size

    # Check for purple/pink lanyard colors
    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([160, 255, 255])
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    purple_ratio = np.sum(purple_mask > 0) / purple_mask.size

    # ID card likely present if we see white card or purple lanyard
    return white_ratio > 0.08 or purple_ratio > 0.05

CLASS_NAMES = ['shirt','pant','id_card','shoes']

def detect_webcam():
    """Live webcam detection with improved error handling"""
    if not check_webcam():
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    # Try to set higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n=== LIVE UNIFORM DETECTION ===")
    print("Webcam window opened - Press 'Q' to quit")
    print("Detects complete uniform: Gray Shirt + Black/Navy Pant + Shoes + ID Card")
    print(f"Config: CONF={CONF_THRESH}, MIN_AREA={MIN_AREA}, IMG_SIZE={IMG_SIZE}\n")

    import time
    frame_count = 0
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame, retrying...")
                time.sleep(0.5)
                continue

            frame_count += 1
            fps_counter += 1

            # Calculate FPS every second
            if time.time() - fps_start_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()

            # Run detection on frame
            try:
                results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)
                res = results[0]
            except Exception as e:
                print(f"Error during prediction: {e}")
                continue

            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]

            detections = []
            for b in res.boxes:
                conf = float(b.conf[0])
                cls = int(b.cls[0])
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                area = (x2-x1)*(y2-y1)
                if area < MIN_AREA:
                    continue

                class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)

                # Spatial filtering
                if USE_SPATIAL_FILTERING:
                    box_center_y = (y1 + y2) / 2

                    if 'pant' in class_name.lower() and box_center_y < frame_height * 0.25:
                        if DIAGNOSTIC_MODE:
                            print(f"  [Filtered] Pant in top 25% (y={box_center_y:.0f}, threshold={frame_height*0.25:.0f})")
                        continue

                    if 'shirt' in class_name.lower() and box_center_y > frame_height * 0.80:
                        if DIAGNOSTIC_MODE:
                            print(f"  [Filtered] Shirt in bottom 20% (y={box_center_y:.0f}, threshold={frame_height*0.80:.0f})")
                        continue

                    if 'shoe' in class_name.lower() and box_center_y < frame_height * 0.50:
                        if DIAGNOSTIC_MODE:
                            print(f"  [Filtered] Shoes in top 50% (y={box_center_y:.0f}, threshold={frame_height*0.50:.0f})")
                        continue

                crop = crop_from_box(frame, (x1,y1,x2,y2))
                color = classify_color(crop)

                detections.append({'class': class_name,
                                   'conf': conf, 'box': (x1,y1,x2,y2), 'color': color})

                # Draw bounding box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"{class_name}:{color}", (x1,y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Check uniform status
            found = { 'shirt': [], 'pant': [], 'shoes': [], 'id_card': [] }
            for d in detections:
                if d['class'] in found:
                    found[d['class']].append(d)

            # FALLBACK: If no ID card detected by model, use heuristic detection
            if not found['id_card'] and found['shirt']:
                # Find the main shirt (largest in upper body region)
                main_shirt = None
                for shirt in found['shirt']:
                    box = shirt['box']
                    center_y = (box[1] + box[3]) / 2
                    # Only consider shirts in upper 60% of image
                    if center_y < frame_height * 0.6:
                        area = (box[2] - box[0]) * (box[3] - box[1])
                        if main_shirt is None or area > main_shirt[1]:
                            main_shirt = (shirt['box'], area)

                if main_shirt is not None:
                    shirt_box = main_shirt[0]
                    if detect_id_card_heuristic(frame, shirt_box):
                        # Add synthetic ID card detection
                        x1, y1, x2, y2 = shirt_box
                        chest_x = x1 + int((x2 - x1) * 0.5)
                        chest_y = y1 + int((y2 - y1) * 0.3)
                        found['id_card'].append({
                            'class': 'id_card',
                            'conf': 0.90,  # High confidence for heuristic
                            'box': (chest_x - 50, chest_y - 50, chest_x + 50, chest_y + 50),
                            'color': 'white'
                        })
                        detections.append(found['id_card'][0])

                        # Draw ID card indicator
                        cv2.circle(frame, (chest_x, chest_y), 30, (255, 0, 255), 3)
                        cv2.putText(frame, "ID (heuristic)", (chest_x - 50, chest_y - 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Shirt check
            shirt_ok = False
            shirt_color = 'none'
            if found['shirt']:
                for s in found['shirt']:
                    crop = crop_from_box(frame, s['box'])
                    color = classify_color(crop)
                    shirt_color = color
                    if color in SHIRT_ALLOWED or hsv_gray_ratio(crop, sat_thresh=60) > 0.35:
                        shirt_ok = True
                        break

            # Pant check
            pant_ok = False
            pant_color = 'none'
            if found['pant']:
                for p in found['pant']:
                    pant_color = p['color']
                    if p['color'] in PANTS_ALLOWED:
                        pant_ok = True
                        break

            # Shoes check
            shoes_ok = len(found['shoes']) > 0 if SHOES_REQUIRED else True
            shoes_color = found['shoes'][0]['color'] if found['shoes'] else 'none'

            # ID check
            id_ok = len(found['id_card']) > 0 if ID_REQUIRED else True
            id_color = found['id_card'][0]['color'] if found['id_card'] else 'none'

            complete_uniform = shirt_ok and pant_ok and shoes_ok and id_ok

            # Display status overlay
            y_offset = 30
            status_items = [
                (f'Shirt ({shirt_color})', shirt_ok),
                (f'Pant ({pant_color})', pant_ok),
                (f'Shoes ({shoes_color})', shoes_ok),
                (f'ID Card ({id_color})', id_ok)
            ]

            for item_name, status in status_items:
                symbol = "✓" if status else "✗"
                color = (0, 255, 0) if status else (0, 0, 255)
                text = f"{symbol} {item_name}"
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 35

            # Show FPS
            cv2.putText(frame, f"FPS: {fps}", (frame_width - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Show final result
            if complete_uniform:
                result_text = "COMPLETE UNIFORM - 1"
                result_color = (0, 255, 0)
            else:
                result_text = "INCOMPLETE UNIFORM - 0"
                result_color = (0, 0, 255)

            cv2.putText(frame, result_text, (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, result_color, 3)

            # Print result for Arduino/ESP32
            result = "1" if complete_uniform else "0"
            print(result)

            # Debug logging
            if DIAGNOSTIC_MODE or not complete_uniform:
                if complete_uniform:
                    print(f"  ✓ UNIFORM OK: Shirt={shirt_color}, Pant={pant_color}, Shoes={shoes_color}, ID={id_color}")
                else:
                    missing = []
                    if not shirt_ok: missing.append(f"Shirt({shirt_color})")
                    if not pant_ok: missing.append(f"Pant({pant_color})")
                    if not shoes_ok: missing.append("Shoes")
                    if not id_ok: missing.append("ID")
                    print(f"  ✗ MISSING: {', '.join(missing)}")

            # Show GUI window
            cv2.imshow('Uniform Detection - Press Q to quit', frame)

            # Check for 'q' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\nQuitting...")
                break

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\nError during detection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released. Goodbye!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Improved Uniform Detection System')
    parser.add_argument('--webcam', action='store_true', help='use webcam for live detection')
    parser.add_argument('--check', action='store_true', help='run system diagnostics only')
    parser.add_argument('--diagnostic', action='store_true', help='enable detailed logging')
    args = parser.parse_args()

    if args.diagnostic:
        DIAGNOSTIC_MODE = True
        print("Diagnostic mode enabled\n")

    if args.check:
        print("\n=== SYSTEM DIAGNOSTICS ===\n")
        check_dependencies()
        check_model_file()
        check_webcam()
        print("\n=== DIAGNOSTICS COMPLETE ===")
    elif args.webcam:
        if not check_dependencies():
            print("\nPlease install missing dependencies first")
            sys.exit(1)
        detect_webcam()
    else:
        print("Usage:")
        print("  python detect_and_classify_improved.py --check       # Run diagnostics")
        print("  python detect_and_classify_improved.py --webcam      # Live detection")
        print("  python detect_and_classify_improved.py --webcam --diagnostic  # Live with debug logs")
        parser.print_help()
