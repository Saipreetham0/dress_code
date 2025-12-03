import cv2
import os
import argparse
from ultralytics import YOLO

MODEL_PATH = 'runs/segment/sitam_uniform_detector_v2/weights/best.pt'
DEFAULT_CONF = 0.45


def main():
    parser = argparse.ArgumentParser(description='Capture one frame and run uniform detection')
    parser.add_argument('--camera', '-c', type=int, default=0, help='Camera index (default 0)')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF, help='Detection confidence threshold')
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print('Model not found:', MODEL_PATH)
        return

    print('Loading model:', MODEL_PATH)
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'ERROR: Cannot open camera {args.camera}. Try a different index with --camera')
        return

    print('Capturing one frame...')
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print('ERROR: Failed to capture frame')
        return

    # Run inference
    results = model(frame, verbose=False, conf=args.conf)
    res = results[0]

    names = res.names
    boxes = res.boxes
    classes = [names[int(c.item())] for c in boxes.cls] if len(boxes.cls) > 0 else []
    confs = [float(cf.item()) for cf in boxes.conf] if len(boxes.conf) > 0 else []

    print('\nDetections:')
    if classes:
        for cls, cf in zip(classes, confs):
            print(f"  - {cls}: {cf:.2f}")
    else:
        print('  (none)')

    # Check full uniform
    has_shirt = any('shirt' in c.lower() for c in classes)
    has_pant = any('pant' in c.lower() for c in classes)
    has_shoes = any('shoe' in c.lower() for c in classes)
    has_id = any('id' in c.lower() or 'card' in c.lower() for c in classes)

    uniform = has_shirt and has_pant and has_shoes and has_id
    print(f"\nFull uniform: {'YES (1)' if uniform else 'NO (0)'}")

    # Save annotated image for inspection
    annotated = res.plot() if hasattr(res, 'plot') else frame
    out_dir = os.path.join('runs', 'debug')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'annotated_once.jpg')
    cv2.imwrite(out_path, annotated)
    print('Annotated image saved to', out_path)

    # Show annotated image briefly
    cv2.imshow('Check Uniform - One Shot', annotated)
    print('Press any key in the image window to close...')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
