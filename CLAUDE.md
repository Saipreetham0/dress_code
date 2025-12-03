# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Student Uniform Detection System** that uses YOLOv8 object detection and color classification to verify if students are wearing complete college uniforms. The system outputs binary results (1 = complete uniform, 0 = incomplete) for hardware integration with Arduino/ESP32 microcontrollers.

## Architecture

### Detection Pipeline
1. **YOLOv8 Object Detection**: Detects 4 uniform components (shirt, pant, shoes, ID card)
2. **Color Classification**: Uses K-means clustering + HSV analysis to verify component colors
3. **Spatial Filtering**: Rejects false positives (e.g., head misdetected as pant) based on bounding box position
4. **Validation Logic**: Checks all components against allowed color sets

### Key Files
- **[detect_and_classify.py](detect_and_classify.py)**: Main detection script with webcam/image processing
- **[Uniform_Detection.v1i.yolov8/](Uniform_Detection.v1i.yolov8/)**: YOLOv8 training dataset (8 classes, Roboflow export)
- **[runs/train/uniform_color_trained2/weights/best.pt](runs/train/uniform_color_trained2/weights/best.pt)**: Trained model weights
- **[HARDWARE_INTEGRATION.md](HARDWARE_INTEGRATION.md)**: Arduino/ESP32 integration guide

## Environment Setup

### Virtual Environment
The project uses a Python virtual environment located at `.venv/`:
```bash
# Activate (Windows)
.venv/Scripts/Activate.ps1

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
Core: `opencv-python`, `ultralytics` (YOLOv8), `torch`, `torchvision`, `numpy`, `pillow`, `scikit-learn`

## Running the System

### Live Webcam Detection
```bash
python detect_and_classify.py --webcam
```
- Press `q` to quit
- Outputs `1` (complete uniform) or `0` (incomplete) to stdout for serial communication

### Single Image Detection
```bash
python detect_and_classify.py --image <path> [--vis]
```
- `--vis` flag shows visualization window

## Training the Model

### Dataset Structure
The dataset follows YOLOv8 format in `Uniform_Detection.v1i.yolov8/`:
- `train/images/`: Training images
- `train/labels/`: YOLO format annotations (.txt files)
- `data.yaml`: Dataset configuration with 8 classes

### Classes (from data.yaml)
`['Shirt', 'civilShirt', 'civilpant', 'identity card', 'pant', 'shirt', 'shoes', 'slippers']`

Note: Classes have some duplicates/case variations (e.g., 'Shirt' vs 'shirt'). The detection script normalizes to lowercase.

### Training Command
```bash
# Train YOLOv8 model
yolo detect train data=Uniform_Detection.v1i.yolov8/data.yaml model=yolov8n.pt epochs=100 imgsz=640

# Results saved to runs/train/<exp_name>/
```

## Uniform Validation Rules

The system enforces these rules (defined in [detect_and_classify.py:24-30](detect_and_classify.py#L24-L30)):

- **Shirt**: Must be gray (checked via HSV saturation ratio > 0.35)
- **Pant**: Must be black, navy blue, or blue
- **Shoes**: Required (any color accepted)
- **ID Card**: Required (any color accepted)

Complete uniform = all 4 components detected with correct colors → outputs `1`

## Configuration Parameters

Key tunable parameters in [detect_and_classify.py](detect_and_classify.py):

```python
CONF_THRESH = 0.25      # Detection confidence threshold (lowered for 30m range)
MIN_AREA = 800          # Minimum bounding box area (pixels²)
IMG_SIZE = 1280         # Input image size for detection (larger = better long-range)
USE_SPATIAL_FILTERING = True  # Enable false positive rejection
```

### Spatial Filtering Logic
Rejects detections based on vertical position in frame:
- Pant in top 25% → Rejected (likely head)
- Shirt in bottom 20% → Rejected (likely legs)
- Shoes must be in bottom 50%

## Hardware Integration

The system outputs to serial port for Arduino/ESP32:
- Prints `1` for complete uniform
- Prints `0` for incomplete uniform

See [HARDWARE_INTEGRATION.md](HARDWARE_INTEGRATION.md) for wiring diagrams and Arduino code examples.

## Long-Range Detection (30m)

The system is optimized for 30-meter detection:
1. Use high-resolution camera (1080p or 4K)
2. Use optical zoom lens (3x-5x recommended)
3. Increase `IMG_SIZE` to 1280 or higher
4. Lower `CONF_THRESH` to 0.25 or below
5. Ensure good lighting (add IR illumination if needed)

## Color Classification

Two methods available:
1. **K-means + HSV heuristic** (default): No training required
2. **Trained sklearn classifier** (optional): Requires `color_clf.joblib` file

The system automatically uses the trained classifier if `color_clf.joblib` exists in the root directory.

## Scrap Files

The `scrap files/` directory contains:
- Previous implementations (older uniform detection approaches)
- Documentation drafts
- Training guides
- Uniform configuration JSONs

These are reference materials and not used by the main detection script.
