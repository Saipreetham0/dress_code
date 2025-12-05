"""
Web-based Uniform Detection System
Upload images to test detection before going live
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import os
import numpy as np
import base64
from datetime import datetime
from ultralytics import YOLO
from sklearn.cluster import KMeans
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TRAINING_FOLDER'] = 'training_data'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['TRAINING_FOLDER'], 'uniform'), exist_ok=True)
os.makedirs(os.path.join(app.config['TRAINING_FOLDER'], 'no_uniform'), exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load model
MODEL_PATH = "runs/train/uniform_color_trained2/weights/best.pt"
FALLBACK_MODEL_PATH = "runs/train/uniform_color_trained/weights/best.pt"

model = None
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
elif os.path.exists(FALLBACK_MODEL_PATH):
    model = YOLO(FALLBACK_MODEL_PATH)

# Configuration
CONF_THRESH = 0.10   # Very low threshold to catch ID cards
MIN_AREA = 400       # Lower minimum area
IMG_SIZE = 640       # Smaller image size for faster processing
USE_SPATIAL_FILTERING = True

SHIRT_ALLOWED = {'gray', 'white'}  # Allow both gray and white/light gray shirts
PANTS_ALLOWED = {'black', 'navy blue', 'blue', 'dark blue'}
SHOES_REQUIRED = True
ID_REQUIRED = True

COLOR_CLF_PATH = "color_clf.joblib"
clf = None
if os.path.exists(COLOR_CLF_PATH):
    clf = joblib.load(COLOR_CLF_PATH)


# Helper functions
def crop_from_box(img, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    h, w = img.shape[:2]
    x1, x2 = max(0, min(x1, w-1)), max(0, min(x2, w-1))
    y1, y2 = max(0, min(y1, h-1)), max(0, min(y2, h-1))
    return img[y1:y2, x1:x2]

def kmeans_dominant_rgb(img_bgr, k=3):
    pixels = img_bgr.reshape(-1, 3)
    if len(pixels) < 10:
        return np.array([0, 0, 0])
    pixels = pixels.astype(np.float32)
    km = KMeans(n_clusters=k, random_state=0).fit(pixels[:, ::-1])
    counts = np.bincount(km.labels_)
    dom = km.cluster_centers_[np.argmax(counts)]
    return dom

def hsv_gray_ratio(img_bgr, sat_thresh=60, v_min=30):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (sat <= sat_thresh) & (val >= v_min)
    return float(np.sum(mask)) / mask.size

def rgb_to_color_name(rgb):
    r, g, b = map(int, rgb)
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
        h_mean = hsv[:, :, 0].mean()
        s_mean = hsv[:, :, 1].mean()
        v_mean = hsv[:, :, 2].mean()
        low_sat_ratio = float(np.mean(hsv[:, :, 1] < 60))
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

# Actual class names from data.yaml (8 classes)
# ['Shirt', 'civilShirt', 'civilpant', 'identity card', 'pant', 'shirt', 'shoes', 'slippers']
CLASS_NAMES_RAW = ['Shirt', 'civilShirt', 'civilpant', 'identity card', 'pant', 'shirt', 'shoes', 'slippers']

# Normalize to our detection categories
def normalize_class_name(raw_name):
    raw_lower = raw_name.lower()
    if 'shirt' in raw_lower:
        return 'shirt'
    elif 'pant' in raw_lower:
        return 'pant'
    elif 'shoe' in raw_lower or 'slipper' in raw_lower:
        return 'shoes'
    elif 'identity' in raw_lower or 'id' in raw_lower or 'card' in raw_lower:
        return 'id_card'
    return raw_lower

def process_image(image_path):
    """Process image and return detection results"""
    if model is None:
        return {
            'error': 'Model not loaded. Please train the model first.',
            'complete_uniform': False
        }

    img = cv2.imread(image_path)
    if img is None:
        return {'error': 'Could not read image', 'complete_uniform': False}

    # Run detection
    results = model.predict(image_path, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)
    res = results[0]

    frame_height, frame_width = img.shape[:2]
    detections = []

    for b in res.boxes:
        conf = float(b.conf[0])
        cls = int(b.cls[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        area = (x2 - x1) * (y2 - y1)

        if area < MIN_AREA:
            continue

        # Get raw class name and normalize it
        raw_class_name = CLASS_NAMES_RAW[cls] if cls < len(CLASS_NAMES_RAW) else f"class_{cls}"
        class_name = normalize_class_name(raw_class_name)

        # Spatial filtering
        if USE_SPATIAL_FILTERING:
            box_center_y = (y1 + y2) / 2

            if 'pant' in class_name.lower() and box_center_y < frame_height * 0.25:
                continue
            if 'shirt' in class_name.lower() and box_center_y > frame_height * 0.80:
                continue
            if 'shoe' in class_name.lower() and box_center_y < frame_height * 0.50:
                continue

        crop = crop_from_box(img, (x1, y1, x2, y2))
        color = classify_color(crop)

        detections.append({
            'class': class_name,
            'conf': float(conf),
            'box': [int(x1), int(y1), int(x2), int(y2)],
            'color': color
        })

        # Draw on image
        color_bgr = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)
        cv2.putText(img, f"{class_name}:{color}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

    # Check uniform status
    found = {'shirt': [], 'pant': [], 'shoes': [], 'id_card': []}
    for d in detections:
        if d['class'] in found:
            found[d['class']].append(d)

    # FALLBACK: If no ID card detected by model, use heuristic detection
    id_card_heuristic_detected = False
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
            if detect_id_card_heuristic(img, shirt_box):
                # Add synthetic ID card detection
                id_card_heuristic_detected = True
                x1, y1, x2, y2 = shirt_box
                chest_x = x1 + int((x2 - x1) * 0.5)
                chest_y = y1 + int((y2 - y1) * 0.3)
                found['id_card'].append({
                    'class': 'id_card',
                    'conf': 0.90,  # High confidence for heuristic
                    'box': [chest_x - 50, chest_y - 50, chest_x + 50, chest_y + 50],
                    'color': 'white'
                })
                detections.append(found['id_card'][0])

                # Draw ID card indicator
                cv2.circle(img, (chest_x, chest_y), 30, (255, 0, 255), 3)
                cv2.putText(img, "ID (heuristic)", (chest_x - 50, chest_y - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Validate uniform
    shirt_ok = False
    shirt_color = 'none'
    if found['shirt']:
        for s in found['shirt']:
            crop = crop_from_box(img, s['box'])
            color = classify_color(crop)
            shirt_color = color
            # Accept gray, white, or if gray ratio check passes
            if color in SHIRT_ALLOWED or hsv_gray_ratio(crop, sat_thresh=60) > 0.25:
                shirt_ok = True
                break

    pant_ok = False
    pant_color = 'none'
    if found['pant']:
        for p in found['pant']:
            pant_color = p['color']
            if p['color'] in PANTS_ALLOWED:
                pant_ok = True
                break

    shoes_ok = len(found['shoes']) > 0 if SHOES_REQUIRED else True
    shoes_color = found['shoes'][0]['color'] if found['shoes'] else 'none'

    id_ok = len(found['id_card']) > 0 if ID_REQUIRED else True
    id_color = found['id_card'][0]['color'] if found['id_card'] else 'none'

    complete_uniform = shirt_ok and pant_ok and shoes_ok and id_ok

    # Add status overlay
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
        cv2.putText(img, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 35

    # Final result overlay
    if complete_uniform:
        result_text = "COMPLETE UNIFORM - 1"
        result_color = (0, 255, 0)
    else:
        result_text = "INCOMPLETE UNIFORM - 0"
        result_color = (0, 0, 255)

    cv2.putText(img, result_text, (10, img.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, result_color, 3)

    # Save processed image
    output_path = os.path.join('static', 'result_' + os.path.basename(image_path))
    cv2.imwrite(output_path, img)

    return {
        'complete_uniform': complete_uniform,
        'result': 1 if complete_uniform else 0,
        'detections': detections,
        'status': {
            'shirt': {'ok': shirt_ok, 'color': shirt_color},
            'pant': {'ok': pant_ok, 'color': pant_color},
            'shoes': {'ok': shoes_ok, 'color': shoes_color},
            'id_card': {'ok': id_ok, 'color': id_color}
        },
        'output_image': output_path
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process image
    result = process_image(filepath)

    if 'output_image' in result:
        result['output_image'] = '/' + result['output_image']

    return jsonify(result)

@app.route('/add_training_image', methods=['POST'])
def add_training_image():
    """Add image to training dataset"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    label = request.form.get('label', 'uniform')  # 'uniform' or 'no_uniform'

    if label not in ['uniform', 'no_uniform']:
        return jsonify({'error': 'Invalid label'}), 400

    # Save to training folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(app.config['TRAINING_FOLDER'], label, filename)
    file.save(filepath)

    # Count training images
    uniform_count = len(os.listdir(os.path.join(app.config['TRAINING_FOLDER'], 'uniform')))
    no_uniform_count = len(os.listdir(os.path.join(app.config['TRAINING_FOLDER'], 'no_uniform')))

    return jsonify({
        'success': True,
        'label': label,
        'filename': filename,
        'counts': {
            'uniform': uniform_count,
            'no_uniform': no_uniform_count
        }
    })

@app.route('/training_stats')
def training_stats():
    """Get training dataset statistics"""
    uniform_count = len(os.listdir(os.path.join(app.config['TRAINING_FOLDER'], 'uniform')))
    no_uniform_count = len(os.listdir(os.path.join(app.config['TRAINING_FOLDER'], 'no_uniform')))

    return jsonify({
        'uniform': uniform_count,
        'no_uniform': no_uniform_count,
        'total': uniform_count + no_uniform_count
    })

@app.route('/config')
def get_config():
    """Get current configuration"""
    return jsonify({
        'CONF_THRESH': CONF_THRESH,
        'MIN_AREA': MIN_AREA,
        'IMG_SIZE': IMG_SIZE,
        'SHIRT_ALLOWED': list(SHIRT_ALLOWED),
        'PANTS_ALLOWED': list(PANTS_ALLOWED),
        'SHOES_REQUIRED': SHOES_REQUIRED,
        'ID_REQUIRED': ID_REQUIRED,
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  WEB UNIFORM DETECTION SYSTEM")
    print("="*60)
    print("\n📸 Test with images before going live")
    print("🎯 Upload uniform/non-uniform images to build training dataset")
    print("🚀 Run live detection after testing\n")
    print("Server starting at: http://localhost:8080")
    print("Press Ctrl+C to stop\n")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=8080)
