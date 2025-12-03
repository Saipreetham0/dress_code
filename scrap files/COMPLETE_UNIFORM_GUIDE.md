# Complete Uniform Detection - Training Guide

## 🎯 Current Status

### ✅ What Works Now (Basic Detection):
- **Shirt color matching** - Detects if wearing registered uniform shirt color
- **Tucked shirt check** - Basic heuristic to check if shirt appears tucked
- **Face detection** - Identifies person in frame
- **Output**: Prints `1` for complete uniform, `0` for incomplete

### ⚠️ Limitations (Needs Custom Training):
The current system uses a pre-trained YOLO model that **cannot detect**:
- Specific pants/trousers
- Shoes
- ID cards

**To detect these items, you need to train a custom AI model.**

---

## 📋 Step-by-Step Guide to Complete Detection

### **Phase 1: Collect Training Data** (Current Step)

#### What You Need:
- **50-100 images** of students in your college
- Images should include:
  - ✅ Students wearing **complete uniform** (shirt, pants, shoes, ID card, tucked)
  - ❌ Students **not in uniform** or partially uniformed
  - Different angles, lighting conditions, backgrounds

#### How to Collect:
1. Take photos with your phone or laptop camera
2. Save all images to folder: `D:/Smart_Vechicle/training_data/images/`
3. Organize into subfolders:
   ```
   training_data/
   ├── images/
   │   ├── uniform_complete/    (complete uniform images)
   │   ├── uniform_partial/     (missing items)
   │   └── no_uniform/          (not in uniform)
   ```

---

### **Phase 2: Label/Annotate Images**

Use **Roboflow** (easiest) or **LabelImg** to draw boxes around:

#### Items to Label:
1. **shirt** - Draw box around shirt area
2. **pants** - Draw box around pants/trousers
3. **shoes** - Draw box around shoes
4. **id-card** - Draw box around ID card (if visible)
5. **face** - Draw box around face

#### Roboflow Method (Recommended):
1. Go to https://roboflow.com/
2. Create free account
3. Create new project: "Sitam College Uniform Detection"
4. Upload your images
5. Annotate each item (draw boxes and label)
6. Export dataset in **YOLOv8 format**

---

### **Phase 3: Train Custom YOLO Model**

Once you have labeled data, train your model:

```python
from ultralytics import YOLO

# Create a new YOLOv8 nano model
model = YOLO('yolov8n.pt')

# Train on your custom dataset
model.train(
    data='path/to/your/dataset.yaml',  # From Roboflow export
    epochs=50,
    imgsz=640,
    batch=8,
    name='sitam_uniform_detector'
)
```

**Training time**: 30 minutes to 2 hours (depending on your computer)

---

### **Phase 4: Use Trained Model**

After training, update `detect_complete_uniform.py`:

```python
# Replace this line:
model = YOLO('yolov8n.pt')

# With your trained model:
model = YOLO('runs/detect/sitam_uniform_detector/weights/best.pt')
```

---

## 🚀 Quick Start (Test Current System)

Run the basic complete uniform detector now:

```powershell
D:/Smart_Vechicle/.venv/Scripts/python.exe detect_complete_uniform.py
```

**What it checks now:**
- ✅ Shirt color (accurate)
- ✅ Tucked shirt (basic heuristic)
- ⚠️ Pants (assumes detected if person present - not accurate)
- ❌ Shoes (not detected - needs training)
- ❌ ID card (not detected - needs training)

**Output**: `1` = Pass, `0` = Fail

---

## 📊 Accuracy Expectations

| Detection Item | Current Accuracy | After Custom Training |
|---|---|---|
| Shirt Color | 85-90% | 90-95% |
| Tucked Shirt | 60-70% | 85-90% |
| Pants | 40% (placeholder) | 85-95% |
| Shoes | 0% (not detected) | 80-90% |
| ID Card | 0% (not detected) | 75-85% |

---

## 💡 Alternative: Simpler Approach

If you don't want to train a custom model, you can:

### Option 1: Focus on Shirt + Manual Checks
- Use current system for **shirt color only**
- Manually verify other items
- Print `1` only if shirt matches

### Option 2: Use Multiple Cameras
- Camera 1: Check upper body (shirt, ID card)
- Camera 2: Check lower body (pants, shoes)

### Option 3: Rule-Based Heuristics
- **Pants**: Check if lower body has distinct color from background
- **Shoes**: Check bottom 10% of image for dark regions
- **ID Card**: Check chest area for rectangular object

---

## ⚙️ Configuration

Edit `detect_complete_uniform.py` to adjust requirements:

```python
UNIFORM_REQUIREMENTS = {
    'shirt_color': True,      # Set False to skip
    'pants': True,
    'shoes': True,
    'id_card': True,
    'tucked_shirt': True
}
```

---

## 🎓 Next Steps

1. **Test current system** → Run `detect_complete_uniform.py`
2. **Decide approach**:
   - **Full AI**: Collect 50-100 images → Label → Train custom model
   - **Simple**: Stick with shirt color detection only
   - **Hybrid**: Use shirt color + manual verification
3. **Deploy** → Integrate with your smart rover project

---

## 📞 Need Help?

- Labeling tutorial: https://roboflow.com/annotate
- YOLO training guide: https://docs.ultralytics.com/modes/train/
- Custom model examples: https://github.com/ultralytics/ultralytics

**Time estimate for full system**: 1-2 days (including data collection and training)
