import cv2
import numpy as np
import os
from pathlib import Path

# Training data paths
UNIFORM_COMPLETE_PATH = os.path.join(os.path.dirname(__file__), 'training_data', 'images', 'uniform_complete')
NO_UNIFORM_PATH = os.path.join(os.path.dirname(__file__), 'training_data', 'images', 'no_uniform')

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detection settings
SIMILARITY_THRESHOLD = 0.55  # Adjust this: higher = stricter matching (lowered for better detection)
USE_NUMERIC_OUTPUT = True
MOBILE_CAMERA_MODE = True  # Optimized for mobile camera


def load_training_images(folder_path):
    """Load all training images from a folder"""
    images = []
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        return images
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # Resize to standard size for comparison
                img = cv2.resize(img, (224, 224))
                images.append(img)
    
    return images


def extract_features(image):
    """Extract color histogram features from image"""
    # Convert to HSV for better color representation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate histogram with more bins for better accuracy
    hist_h = cv2.calcHist([hsv], [0], None, [60], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [64], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [64], [0, 256])
    
    # Normalize
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    
    # Concatenate histograms
    features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
    
    # Also extract edge features for better matching
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_hist = cv2.calcHist([edges], [0], None, [32], [0, 256])
    cv2.normalize(edge_hist, edge_hist)
    
    # Combine color and edge features
    features = np.concatenate([features, edge_hist.flatten()])
    
    return features


def compare_with_training_images(current_img, training_images):
    """Compare current image with all training images and return best match score"""
    if len(training_images) == 0:
        return 0.0
    
    # Extract features from current image
    current_features = extract_features(current_img)
    
    # Compare with each training image
    max_similarity = 0.0
    
    for train_img in training_images:
        train_features = extract_features(train_img)
        
        # Calculate correlation (similarity)
        similarity = cv2.compareHist(
            current_features.reshape(-1, 1),
            train_features.reshape(-1, 1),
            cv2.HISTCMP_CORREL
        )
        
        max_similarity = max(max_similarity, similarity)
    
    return max_similarity


def main():
    print("=" * 60)
    print("UNIFORM DETECTION - IMAGE SIMILARITY MATCHING")
    print("=" * 60)
    
    # Load training images
    print("\nLoading training images...")
    uniform_complete_images = load_training_images(UNIFORM_COMPLETE_PATH)
    no_uniform_images = load_training_images(NO_UNIFORM_PATH)
    
    print(f"✓ Complete Uniform images: {len(uniform_complete_images)}")
    print(f"✓ No Uniform images: {len(no_uniform_images)}")
    
    if len(uniform_complete_images) == 0 and len(no_uniform_images) == 0:
        print("\nERROR: No training images found!")
        print("Please add images to:")
        print(f"  - {UNIFORM_COMPLETE_PATH}")
        print(f"  - {NO_UNIFORM_PATH}")
        return
    
    print("\nOutput: 1 = Complete Uniform, 0 = No Uniform")
    print("Press 'q' to quit\n")
    print("=" * 60)
    
    # Get mobile camera IP
    print("\n📱 Mobile Camera Setup:")
    print("Make sure IP Webcam app is running on your phone")
    print("Default IP: http://10.252.44.248:8080")
    
    use_default = input("\nUse default IP? (y/n): ").strip().lower()
    
    if use_default == 'y' or use_default == '':
        mobile_ip = "http://10.252.44.248:8080/video"
    else:
        ip_input = input("Enter IP from IP Webcam app (e.g., 192.168.1.100:8080): ").strip()
        # Clean up input
        ip_input = ip_input.replace('http://', '').replace('https://', '')
        if '/video' in ip_input:
            mobile_ip = f"http://{ip_input}"
        else:
            mobile_ip = f"http://{ip_input}/video"
    
    print(f"\nConnecting to: {mobile_ip}")
    print("Please wait...")
    
    cap = cv2.VideoCapture(mobile_ip)
    
    if not cap.isOpened():
        print("\n❌ ERROR: Could not connect to mobile camera!")
        print("\nTroubleshooting:")
        print("  1. Open IP Webcam app on your phone")
        print("  2. Tap 'Start Server' at the bottom")
        print("  3. Note the IP address shown (e.g., http://192.168.1.100:8080)")
        print("  4. Make sure phone and laptop are on SAME WiFi")
        print("  5. Try opening the IP in your browser first to test")
        print(f"\n  Test URL in browser: {mobile_ip.replace('/video', '')}")
        return
    
    frame_count = 0
    last_result = ("NO UNIFORM", (0, 0, 255), 0)  # Store last result
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame if too large (for better performance)
        if MOBILE_CAMERA_MODE and frame.shape[1] > 1280:
            scale = 1280 / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        frame_display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Improved face detection for mobile camera
        # Try multiple detection passes with different parameters
        faces = FACE_CASCADE.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # Smaller scale for better detection
            minNeighbors=3,     # Lower neighbor count for easier detection
            minSize=(40, 40),   # Smaller minimum size
            maxSize=(300, 300)  # Maximum size limit
        )
        
        result = 0  # Default: no uniform
        
        if len(faces) > 0:
            # Get largest face
            faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
            x, y, w, h = faces[0]
            cv2.rectangle(frame_display, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract person region (full body or upper body)
            # Expand region below face to capture shirt/uniform
            person_y1 = max(0, y - int(0.2*h))
            person_y2 = min(frame.shape[0], y + h + int(2.0*h))
            person_x1 = max(0, x - int(0.5*w))
            person_x2 = min(frame.shape[1], x + w + int(0.5*w))
            
            person_roi = frame[person_y1:person_y2, person_x1:person_x2]
            
            if person_roi.size > 0:
                # Resize for comparison
                person_roi_resized = cv2.resize(person_roi, (224, 224))
                
                # Draw person ROI
                cv2.rectangle(frame_display, (person_x1, person_y1), (person_x2, person_y2), (0, 255, 0), 2)
                
                # Compare with training images (every 3 frames for more responsive detection)
                if frame_count % 3 == 0:
                    # Calculate similarity with both categories
                    uniform_similarity = compare_with_training_images(person_roi_resized, uniform_complete_images)
                    no_uniform_similarity = compare_with_training_images(person_roi_resized, no_uniform_images)
                    
                    # Display similarity scores
                    cv2.putText(frame_display, f"Uniform: {uniform_similarity:.3f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame_display, f"No Uniform: {no_uniform_similarity:.3f}", (10, 65), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame_display, f"Threshold: {SIMILARITY_THRESHOLD}", (10, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    # Improved decision logic with confidence margin
                    confidence_margin = 0.05
                    if uniform_similarity > no_uniform_similarity + confidence_margin and uniform_similarity > SIMILARITY_THRESHOLD:
                        result = 1  # Complete uniform
                        msg = "COMPLETE UNIFORM"
                        color = (0, 255, 0)
                    else:
                        result = 0  # No uniform
                        msg = "NO UNIFORM"
                        color = (0, 0, 255)
                    
                    # Print result
                    if USE_NUMERIC_OUTPUT:
                        print(result)
                    
                    # Display result
                    cv2.putText(frame_display, msg, (10, frame_display.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    
                    # Store for display in non-processing frames
                    last_result = (msg, color, result)
                else:
                    # Use last result for display
                    msg, color, result = last_result
                    cv2.putText(frame_display, msg, (10, frame_display.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        else:
            cv2.putText(frame_display, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if USE_NUMERIC_OUTPUT and frame_count % 10 == 0:
                print(0)
        
        cv2.imshow('Uniform Detection - Image Matching', frame_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("Detection stopped.")
    print("=" * 60)


if __name__ == '__main__':
    main()
