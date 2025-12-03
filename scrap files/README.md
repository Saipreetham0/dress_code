# Smart Rover - Student Uniform Detection

This project uses OpenCV and your laptop camera to detect whether a student is wearing the full uniform.

## Features
- **Face detection** to locate the student
- **Color-based uniform detection** using HSV color matching
- **Real-time feedback** with audio-like messages:
  - ✅ "Very good — you are fully uniformed."
  - ❌ "Please wear your full uniform."

## Setup Instructions

### 1. Install Python
Make sure Python 3.8+ is installed. Check with:
```powershell
python --version
```

### 2. Create Virtual Environment
```powershell
cd d:\Smart_Vechicle
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

## Usage

### Step 1: Register Uniform Color (Choose ONE Method)

#### **Method 1: Capture from Camera**
1. **Wear your full uniform**
2. Run registration mode:
   ```powershell
   D:/Smart_Vechicle/.venv/Scripts/python.exe detect_uniform.py --register
   ```
3. Choose option **1** (Capture from camera)
4. Face the camera
5. Press **`r`** to sample the uniform shirt color
6. Press **`s`** to save
7. Press **`q`** to exit

#### **Method 2: Upload Image File** 
1. Take or find a photo showing someone wearing the full uniform (face visible)
2. Run registration mode:
   ```powershell
   D:/Smart_Vechicle/.venv/Scripts/python.exe detect_uniform.py --register
   ```
3. Choose option **2** (Upload image file)
4. Enter the full path to your image (e.g., `D:/uniform_photo.jpg`)
5. The system will automatically detect and save the uniform color

#### **Method 3: Set Reference Uniform Image** (Optional)
1. Run registration mode:
   ```powershell
   D:/Smart_Vechicle/.venv/Scripts/python.exe detect_uniform.py --register
   ```
2. Choose option **3** (Set reference uniform image)
3. Enter path to a uniform photo (e.g., `D:/uniform_reference.jpg`)
4. This image will be displayed alongside the camera for visual comparison

### Step 2: Run Live Detection
1. Run the script:
   ```powershell
   D:/Smart_Vechicle/.venv/Scripts/python.exe detect_uniform.py
   ```
2. The camera detects your face and checks if you're wearing the uniform
3. **Console Output:**
   - Prints **`1`** = Wearing uniform ✅
   - Prints **`0`** = Not wearing uniform ❌
4. **Visual Display:**
   - Green message for uniform match
   - Red message for no match
   - Reference uniform image shown on the right (if set)
5. Press **`q`** to quit

## Customization

### Adjust Messages
Edit `detect_uniform.py` lines 14-15:
```python
MSG_FULL = "Very good — you are fully uniformed."  # change this
MSG_NOT = "Please wear your full uniform."         # change this
```

### Adjust Sensitivity
- **`MATCH_RATIO_THRESHOLD`** (line 18): Lower = more lenient, Higher = stricter
- **HSV tolerances** in `uniform_config.json`: Increase `tol_h`, `tol_s`, `tol_v` for wider color range

## Troubleshooting
- **Camera not opening**: Check if another app is using the camera
- **No face detected**: Ensure good lighting and face the camera directly
- **Color not matching**: Re-sample with `r` in good lighting conditions

## Future Improvements
- Add text-to-speech for audio feedback
- Use deep learning (YOLO, Detectron2) for detecting specific uniform items (tie, badge, etc.)
- Support multiple uniform color variations
q