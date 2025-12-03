# detect_and_classify.py
import cv2, os, numpy as np, joblib
from ultralytics import YOLO
from sklearn.cluster import KMeans

# CONFIG
MODEL_PATH = "runs/train/uniform_color_trained2/weights/best.pt"  # newly trained detector
COLOR_CLF_PATH = "color_clf.joblib"            # optional trained color classifier
USE_COLOR_CLASSIFIER = os.path.exists(COLOR_CLF_PATH)
CONF_THRESH = 0.25   # Lowered for long-range detection (30m)
MIN_AREA = 800       # Smaller threshold for distant objects
IMG_SIZE = 1280      # Larger image size for long-range detection (30m)

# Trained on 8 classes from Uniform_Detection.v1i.yolov8 dataset:
# Classes: ['Shirt', 'civilShirt', 'civilpant', 'identity card', 'pant', 'shirt', 'shoes', 'slippers']

# COLORS LEARNED FROM TRAINING DATASET:
# Based on analysis of Uniform_Detection.v1i.yolov8 training images:
# - SHIRT: Gray, Light Gray, White (college uniform shirts)
# - PANT: Navy Blue, Dark Blue, Black (college uniform pants)
# - SHOES: White, Black, Brown (various colors allowed)
# - ID CARD: Red, Yellow, Blue, Green (lanyard/card colors)

# UNIFORM VALIDATION RULES (Student Uniform Analysis System)
# For Hardware Integration (Arduino/ESP32): Prints 1 or 0 via serial
# Complete Uniform = shirt (gray only) + pant (navy/black) + shoes + id_card
SHIRT_ALLOWED = {'gray'}
PANTS_ALLOWED = {'black', 'navy blue', 'blue'}
SHOES_REQUIRED = True
ID_REQUIRED = True

# Spatial filtering to reject false positives (face/head detected as pant)
USE_SPATIAL_FILTERING = True

model = YOLO(MODEL_PATH)
clf = None
if USE_COLOR_CLASSIFIER:
    clf = joblib.load(COLOR_CLF_PATH)
    print("Loaded color classifier:", COLOR_CLF_PATH)

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
    km = KMeans(n_clusters=k, random_state=0).fit(pixels[:, ::-1])  # cluster in RGB
    counts = np.bincount(km.labels_)
    dom = km.cluster_centers_[np.argmax(counts)]
    return dom  # RGB

def hsv_gray_ratio(img_bgr, sat_thresh=60, v_min=30):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:,:,1]
    val = hsv[:,:,2]
    mask = (sat <= sat_thresh) & (val >= v_min)
    return float(np.sum(mask))/mask.size

def rgb_to_color_name(rgb):
    # Simple mapping from RGB -> coarse color names
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
    # If classifier exists, use it
    if clf is not None:
        # extract same features as training
        # dominant RGB
        dom_rgb = kmeans_dominant_rgb(img_bgr, k=3)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h_mean, s_mean, v_mean = hsv[:,:,0].mean(), hsv[:,:,1].mean(), hsv[:,:,2].mean()
        low_sat_ratio = float(np.mean(hsv[:,:,1] < 60))
        feat = np.concatenate([dom_rgb, [h_mean, s_mean, v_mean, low_sat_ratio]])
        pred = clf.predict([feat])[0]
        return pred
    # fallback: kmeans + simple mapping
    dom_rgb = kmeans_dominant_rgb(img_bgr, k=3)
    # check gray via hsv ratio
    if hsv_gray_ratio(img_bgr, sat_thresh=60) > 0.4:
        return 'gray'
    # otherwise map rgb to coarse color
    return rgb_to_color_name(dom_rgb)

# Map class indices to names (use same order as in training)
CLASS_NAMES = ['shirt','pant','id_card','shoes']  # adjust if different

def evaluate_image(image_path, visualize=False):
    img = cv2.imread(image_path)
    if img is None:
        print("Could not read", image_path); return None
    results = model.predict(image_path, imgsz=640, conf=CONF_THRESH, verbose=False)
    res = results[0]
    detections = []
    for b in res.boxes:
        conf = float(b.conf[0])
        cls = int(b.cls[0])
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        area = (x2-x1)*(y2-y1)
        if area < MIN_AREA:
            continue
        crop = crop_from_box(img, (x1,y1,x2,y2))
        color = classify_color(crop)
        detections.append({'class': CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls),
                           'conf': conf, 'box': (x1,y1,x2,y2), 'color': color})
        if visualize:
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, f"{CLASS_NAMES[cls]}:{color} {conf:.2f}", (x1,y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    # apply uniform logic
    # Must find at least one of each (shirt, pant, shoes, id_card) with correct colors
    found = { 'shirt': [], 'pant': [], 'shoes': [], 'id_card': [] }
    for d in detections:
        if d['class'] in found:
            found[d['class']].append(d)

    # Shirt check: at least one shirt and color gray and sat ratio check
    shirt_ok = False
    if found['shirt']:
        for s in found['shirt']:
            # additional check: ensure hsv gray ratio high
            x1,y1,x2,y2 = s['box']
            crop = crop_from_box(img, s['box'])
            if hsv_gray_ratio(crop, sat_thresh=60) > 0.35:
                shirt_ok = True
                break

    # Pant check: presence and color in PANTS_ALLOWED
    pant_ok = False
    if found['pant']:
        for p in found['pant']:
            if p['color'] in PANTS_ALLOWED:
                pant_ok = True
                break

    # Shoes check: presence
    shoes_ok = len(found['shoes']) > 0 if SHOES_REQUIRED else True

    # ID check: presence and id_color in allowed set
    id_ok = False
    if found['id_card']:
        for idc in found['id_card']:
            # id color mapping could be id_red, id_yellow etc. We normalize
            clr = idc['color']
            # try map 'red' etc
            base = clr.replace('id_','') if clr.startswith('id_') else clr
            if base in ID_ALLOWED:
                id_ok = True
                break

    complete_uniform = shirt_ok and pant_ok and shoes_ok and id_ok
    result = {
        'detections': detections,
        'shirt_ok': shirt_ok,
        'pant_ok': pant_ok,
        'shoes_ok': shoes_ok,
        'id_ok': id_ok,
        'complete_uniform': complete_uniform
    }
    if visualize:
        cv2.imshow('vis', cv2.resize(img, (800, int(img.shape[0]*800/img.shape[1]))))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return result

def detect_webcam():
    """Live webcam detection with GUI window - no face detection."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return
    
    print("\n=== LIVE UNIFORM DETECTION ===")
    print("Webcam window opened - Press 'Q' to quit")
    print("Detects complete uniform: Gray Shirt + Black/Navy Pant + Shoes + ID Card\n")
    
    import time
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.5)
                continue
            
            frame_count += 1
            
            # Run detection on frame (optimized for 30m range)
            results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)
            res = results[0]
            
            # Get frame dimensions for spatial filtering
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
                
                # Spatial filtering: reject false positives
                if USE_SPATIAL_FILTERING:
                    box_center_y = (y1 + y2) / 2
                    
                    # Reject PANT in top 25% (likely head/face misdetection)
                    if 'pant' in class_name.lower() and box_center_y < frame_height * 0.25:
                        continue
                    
                    # Reject SHIRT in bottom 20% (likely legs)
                    if 'shirt' in class_name.lower() and box_center_y > frame_height * 0.80:
                        continue
                    
                    # SHOES must be in bottom half
                    if 'shoe' in class_name.lower() and box_center_y < frame_height * 0.50:
                        continue
                
                crop = crop_from_box(frame, (x1,y1,x2,y2))
                color = classify_color(crop)
                
                detections.append({'class': class_name,
                                   'conf': conf, 'box': (x1,y1,x2,y2), 'color': color})
                
                # Draw bounding box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"{class_name}:{color}", (x1,y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
            # Check uniform status directly from detections
            found = { 'shirt': [], 'pant': [], 'shoes': [], 'id_card': [] }
            for d in detections:
                if d['class'] in found:
                    found[d['class']].append(d)
            
            # Shirt check: must be gray
            shirt_ok = False
            shirt_color = 'none'
            if found['shirt']:
                for s in found['shirt']:
                    crop = crop_from_box(frame, s['box'])
                    color = classify_color(crop)
                    shirt_color = color
                    # Check if color is gray
                    if color in SHIRT_ALLOWED or hsv_gray_ratio(crop, sat_thresh=60) > 0.35:
                        shirt_ok = True
                        break
            
            # Pant check: must be black or navy blue
            pant_ok = False
            pant_color = 'none'
            if found['pant']:
                for p in found['pant']:
                    pant_color = p['color']
                    if p['color'] in PANTS_ALLOWED:
                        pant_ok = True
                        break
            
            # Shoes check: presence required
            shoes_ok = len(found['shoes']) > 0 if SHOES_REQUIRED else True
            shoes_color = found['shoes'][0]['color'] if found['shoes'] else 'none'
            
            # ID check: presence required (any color accepted)
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
            
            # Show final result
            if complete_uniform:
                result_text = "COMPLETE UNIFORM - 1"
                result_color = (0, 255, 0)
            else:
                result_text = "INCOMPLETE UNIFORM - 0"
                result_color = (0, 0, 255)
            
            cv2.putText(frame, result_text, (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, result_color, 3)
            
            # Print result for Arduino/ESP32 Serial Communication
            # Hardware reads: 1 = Complete Uniform, 0 = Incomplete
            result = "1" if complete_uniform else "0"
            print(result)
            
            # Debug logging (comment out for hardware deployment)
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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='path to image to process')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--webcam', action='store_true', help='use webcam for live detection')
    args = parser.parse_args()
    
    if args.webcam:
        detect_webcam()
    elif args.image:
        r = evaluate_image(args.image, visualize=args.vis)
        print("Result:", r)
    else:
        print("Please specify --image <path> or --webcam")
        parser.print_help()
