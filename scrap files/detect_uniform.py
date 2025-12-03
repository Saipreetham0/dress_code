import cv2
import numpy as np
import json
import os
import sys

# Config file to store sampled HSV range
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'uniform_config.json')
UNIFORM_IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'uniform_reference.jpg')

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Default messages (polite and configurable)
MSG_FULL = "Very good — you are fully uniformed."  # shown when matched
MSG_NOT = "Please wear your full uniform."       # shown when not matched

# Thresholds
MATCH_RATIO_THRESHOLD = 0.35  # fraction of pixels in shirt ROI that must match uniform color
USE_NUMERIC_OUTPUT = True  # Set to True to print 1/0 instead of messages


def save_config(hsv_median, tol=(10, 60, 60)):
    cfg = {'h': int(hsv_median[0]), 's': int(hsv_median[1]), 'v': int(hsv_median[2]),
           'tol_h': int(tol[0]), 'tol_s': int(tol[1]), 'tol_v': int(tol[2])}
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f)
    print(f"Saved uniform config to {CONFIG_PATH}")


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return None
    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)
    return cfg


def get_hsv_bounds(cfg):
    h, s, v = cfg['h'], cfg['s'], cfg['v']
    th, ts, tv = cfg.get('tol_h', 10), cfg.get('tol_s', 60), cfg.get('tol_v', 60)
    lower = np.array([max(0, h - th), max(0, s - ts), max(0, v - tv)], dtype=np.uint8)
    upper = np.array([min(179, h + th), min(255, s + ts), min(255, v + tv)], dtype=np.uint8)
    return lower, upper


def sample_from_image(image_path):
    """Sample uniform color from an uploaded image file."""
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found: {image_path}")
        return None
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Could not read image: {image_path}")
        return None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    
    if len(faces) == 0:
        print("No face detected in the image. Please provide an image with a visible face.")
        return None
    
    # Pick largest face
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    (x, y, w, h) = faces[0]
    
    # Define shirt ROI
    shirt_y1 = y + h
    shirt_y2 = min(frame.shape[0], y + h + int(1.2*h))
    shirt_x1 = max(0, x - int(0.15*w))
    shirt_x2 = min(frame.shape[1], x + w + int(0.15*w))
    
    if shirt_y1 >= shirt_y2 or shirt_x1 >= shirt_x2:
        print("Could not detect shirt area. Please provide a full-body or upper-body image.")
        return None
    
    shirt_roi = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
    hsv = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2HSV)
    
    # Compute median HSV
    h_med = int(np.median(hsv[:,:,0]))
    s_med = int(np.median(hsv[:,:,1]))
    v_med = int(np.median(hsv[:,:,2]))
    
    cfg = {'h': h_med, 's': s_med, 'v': v_med, 'tol_h': 12, 'tol_s': 65, 'tol_v': 65}
    
    # Show preview
    frame_display = frame.copy()
    cv2.rectangle(frame_display, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.rectangle(frame_display, (shirt_x1, shirt_y1), (shirt_x2, shirt_y2), (0, 255, 0), 2)
    cv2.putText(frame_display, f"Sampled: H={h_med} S={s_med} V={v_med}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Uniform Sample Preview', frame_display)
    print(f"Sampled HSV from image: H={h_med}, S={s_med}, V={v_med}")
    print("Press any key to close preview...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return cfg


def register_mode():
    """Interactive mode to register uniform - choose between camera or image file."""
    print("\n=== UNIFORM REGISTRATION MODE ===")
    print("Choose registration method:")
    print("  1. Capture from camera (live)")
    print("  2. Upload image file")
    print("  3. Set reference uniform image (for visual comparison)")
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == '2':
        image_path = input("Enter full path to uniform image (e.g., D:/uniform.jpg): ").strip()
        cfg = sample_from_image(image_path)
        if cfg:
            save_config((cfg['h'], cfg['s'], cfg['v']), (cfg.get('tol_h',12), cfg.get('tol_s',65), cfg.get('tol_v',65)))
            print("✓ Uniform registered successfully from image!")
        return
    
    elif choice == '1':
        print("\nStarting camera capture...")
        print("Instructions:")
        print("  - Wear your full uniform")
        print("  - Face the camera")
        print("  - Press 'r' to sample shirt color")
        print("  - Press 's' to save")
        print("  - Press 'q' to cancel")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open camera.")
            return
        
        cfg = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            
            if len(faces) > 0:
                faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame_display, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                shirt_y1 = y + h
                shirt_y2 = min(frame.shape[0], y + h + int(1.2*h))
                shirt_x1 = max(0, x - int(0.15*w))
                shirt_x2 = min(frame.shape[1], x + w + int(0.15*w))
                
                if shirt_y1 < shirt_y2 and shirt_x1 < shirt_x2:
                    cv2.rectangle(frame_display, (shirt_x1, shirt_y1), (shirt_x2, shirt_y2), (0, 255, 0), 2)
                
                cv2.putText(frame_display, "Press 'r' to sample uniform", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame_display, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow('Register Uniform', frame_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    shirt_y1 = y + h
                    shirt_y2 = min(frame.shape[0], y + h + int(1.2*h))
                    shirt_x1 = max(0, x - int(0.15*w))
                    shirt_x2 = min(frame.shape[1], x + w + int(0.15*w))
                    if shirt_y1 < shirt_y2 and shirt_x1 < shirt_x2:
                        shirt_roi = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
                        hsv = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2HSV)
                        h_med = int(np.median(hsv[:,:,0]))
                        s_med = int(np.median(hsv[:,:,1]))
                        v_med = int(np.median(hsv[:,:,2]))
                        cfg = {'h': h_med, 's': s_med, 'v': v_med, 'tol_h': 12, 'tol_s': 65, 'tol_v': 65}
                        print(f"Sampled HSV: H={h_med}, S={s_med}, V={v_med}")
                        print("Press 's' to save or 'r' to re-sample.")
                else:
                    print("No face detected.")
            elif key == ord('s'):
                if cfg:
                    save_config((cfg['h'], cfg['s'], cfg['v']), (cfg.get('tol_h',12), cfg.get('tol_s',65), cfg.get('tol_v',65)))
                    print("✓ Uniform registered successfully from camera!")
                    break
                else:
                    print("No sample to save. Press 'r' first.")
        
        cap.release()
        cv2.destroyAllWindows()
    
    elif choice == '3':
        image_path = input("Enter full path to uniform reference image: ").strip()
        if os.path.exists(image_path):
            import shutil
            shutil.copy(image_path, UNIFORM_IMAGE_PATH)
            print(f"✓ Reference uniform image saved to {UNIFORM_IMAGE_PATH}")
            print("This image will be displayed during detection for comparison.")
        else:
            print(f"ERROR: Image file not found: {image_path}")
    
    else:
        print("Invalid choice.")


def detection_mode():
    """Live detection mode - checks if student is wearing uniform."""
    cfg = load_config()
    if not cfg:
        print("ERROR: No uniform configuration found!")
        print("Please run registration mode first: python detect_uniform.py --register")
        return
    
    lower, upper = get_hsv_bounds(cfg)
    print("Starting live uniform detection. Press 'q' to quit.")
    if USE_NUMERIC_OUTPUT:
        print("Output: 1 = Wearing uniform, 0 = Not wearing uniform")
    
    # Load reference uniform image if available
    reference_img = None
    if os.path.exists(UNIFORM_IMAGE_PATH):
        reference_img = cv2.imread(UNIFORM_IMAGE_PATH)
        if reference_img is not None:
            # Resize reference image to 200x300 for display
            reference_img = cv2.resize(reference_img, (200, 300))
            print("Reference uniform image loaded for comparison.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        frame_display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        message = MSG_NOT
        color = (0, 0, 255)  # red for not matched

        if len(faces) > 0:
            faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame_display, (x, y), (x+w, y+h), (255, 0, 0), 2)

            shirt_y1 = y + h
            shirt_y2 = min(frame.shape[0], y + h + int(1.2*h))
            shirt_x1 = max(0, x - int(0.15*w))
            shirt_x2 = min(frame.shape[1], x + w + int(0.15*w))

            if shirt_y1 < shirt_y2 and shirt_x1 < shirt_x2:
                shirt_roi = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
                cv2.rectangle(frame_display, (shirt_x1, shirt_y1), (shirt_x2, shirt_y2), (0, 255, 0), 2)
                hsv = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower, upper)
                match_ratio = (cv2.countNonZero(mask) / mask.size) if mask.size > 0 else 0

                cv2.putText(frame_display, f"Match: {match_ratio:.2f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if match_ratio >= MATCH_RATIO_THRESHOLD:
                    if USE_NUMERIC_OUTPUT:
                        print("1")  # Wearing uniform
                    message = MSG_FULL
                    color = (0, 200, 0)
                else:
                    if USE_NUMERIC_OUTPUT:
                        print("0")  # Not wearing uniform
        else:
            cv2.putText(frame_display, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame_display, message, (10, frame_display.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
        
        # Display reference uniform image alongside if available
        if reference_img is not None:
            # Create combined view
            h, w = frame_display.shape[:2]
            ref_h, ref_w = reference_img.shape[:2]
            combined = np.zeros((max(h, ref_h), w + ref_w + 10, 3), dtype=np.uint8)
            combined[:h, :w] = frame_display
            combined[:ref_h, w+10:w+10+ref_w] = reference_img
            cv2.putText(combined, "Reference Uniform", (w+15, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow('Uniform Detection', combined)
        else:
            cv2.imshow('Uniform Detection', frame_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ['--register', '-r', 'register']:
        register_mode()
    else:
        detection_mode()
