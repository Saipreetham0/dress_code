import cv2
import numpy as np
import os
from pathlib import Path

# Training data paths
UNIFORM_COMPLETE_PATH = os.path.join(os.path.dirname(__file__), 'training_data', 'images', 'uniform_complete')
NO_UNIFORM_PATH = os.path.join(os.path.dirname(__file__), 'training_data', 'images', 'no_uniform')

# Detection settings
SIMILARITY_THRESHOLD = 0.50  # Threshold for uniform detection
USE_NUMERIC_OUTPUT = True
MOBILE_CAMERA_MODE = True


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
                img = cv2.resize(img, (320, 320))
                images.append(img)
    
    return images


def extract_features(image):
    """Extract comprehensive features from image for matching"""
    # Resize to standard size
    image = cv2.resize(image, (320, 320))
    
    # 1. Color histogram in HSV space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    hist_h = cv2.calcHist([hsv], [0], None, [60], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [64], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [64], [0, 256])
    
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    
    color_features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
    
    # 2. Edge features for structure
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_hist = cv2.calcHist([edges], [0], None, [32], [0, 256])
    cv2.normalize(edge_hist, edge_hist)
    
    # 3. Texture features using Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_hist = cv2.calcHist([np.uint8(np.absolute(laplacian))], [0], None, [32], [0, 256])
    cv2.normalize(texture_hist, texture_hist)
    
    # Combine all features
    features = np.concatenate([color_features, edge_hist.flatten(), texture_hist.flatten()])
    
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
    print("UNIFORM DETECTION - NO FACE DETECTION MODE")
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
    
    print("\nDetection Mode: Full Frame Matching (No Face Detection)")
    print("Output: 1 = Complete Uniform, 0 = No Uniform")
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
    
    print("✓ Camera connected!\n")
    
    frame_count = 0
    last_result = ("NO UNIFORM", (0, 0, 255), 0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame if too large (for better performance)
        if MOBILE_CAMERA_MODE and frame.shape[1] > 1280:
            scale = 1280 / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        frame_display = frame.copy()
        
        result = 0  # Default: no uniform
        
        # Compare with training images (every 3 frames for responsive detection)
        if frame_count % 3 == 0:
            # Calculate similarity with both categories
            uniform_similarity = compare_with_training_images(frame, uniform_complete_images)
            no_uniform_similarity = compare_with_training_images(frame, no_uniform_images)
            
            # Display similarity scores
            cv2.putText(frame_display, f"Uniform Match: {uniform_similarity:.3f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame_display, f"No Uniform Match: {no_uniform_similarity:.3f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame_display, f"Threshold: {SIMILARITY_THRESHOLD:.2f}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
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
            
            # Display result message
            cv2.putText(frame_display, msg, (10, frame_display.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
            
            # Display detection info
            cv2.putText(frame_display, "Full Frame Detection - No Face Required", (10, frame_display.shape[0] - 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Store for display in non-processing frames
            last_result = (msg, color, result)
        else:
            # Use last result for display
            msg, color, result = last_result
            cv2.putText(frame_display, msg, (10, frame_display.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
            cv2.putText(frame_display, "Full Frame Detection - No Face Required", (10, frame_display.shape[0] - 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Uniform Detection - Full Frame Mode', frame_display)
        
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
