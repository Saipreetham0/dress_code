# Training Images Collection Guide

## 📸 **Two Ways to Add Training Images**

---

## **Method 1: Capture Live from Laptop Camera** (Recommended)

Run this command:
```powershell
D:/Smart_Vechicle/.venv/Scripts/python.exe capture_training_images.py
```

### What to do:
1. Camera window opens
2. **Wear complete uniform** → Press **`1`** to capture
3. **Remove ID card or untuck shirt** → Press **`2`** to capture
4. **Change to casual clothes** → Press **`3`** to capture
5. Repeat 20-30 times for each category
6. Press **`q`** when done

### Tips:
- Change your position/angle between captures
- Try different lighting
- Move closer/farther from camera
- Include different people if possible

---

## **Method 2: Copy Images from Phone/USB**

If you already have photos on your phone or camera:

### Step 1: Transfer photos to your computer
- Connect phone via USB or use cloud (Google Drive, etc.)
- Save images to a folder like `D:/uniform_photos/`

### Step 2: Organize images
Manually copy images to these folders:

**Complete Uniform** (shirt + pants + shoes + ID card + tucked):
```
D:/Smart_Vechicle/training_data/images/uniform_complete/
```

**Partial Uniform** (missing items or untucked):
```
D:/Smart_Vechicle/training_data/images/uniform_partial/
```

**No Uniform** (casual clothes):
```
D:/Smart_Vechicle/training_data/images/no_uniform/
```

### Step 3: Rename images (optional)
Give meaningful names like:
- `complete_001.jpg`
- `partial_untucked_005.jpg`
- `casual_002.jpg`

---

## 📊 **Check Your Progress**

See how many images you've collected:
```powershell
D:/Smart_Vechicle/.venv/Scripts/python.exe capture_training_images.py --list
```

### Target:
- **20+ Complete Uniform** images
- **15+ Partial Uniform** images
- **15+ No Uniform** images
- **Total: 50+ images minimum**

---

## 🎯 **What Makes Good Training Images?**

### ✅ Good Examples:
- Person facing camera (face visible)
- Full body or upper body visible
- Clear lighting
- Different angles (front, slight side)
- Various backgrounds

### ❌ Avoid:
- Blurry images
- Too dark or overexposed
- Only face visible (need body)
- Duplicate/nearly identical images

---

## 📋 **Image Checklist for Each Category**

### **Category 1: Complete Uniform** ✅
- [ ] Correct uniform shirt (color matching)
- [ ] Uniform pants/trousers
- [ ] Proper shoes (not sneakers)
- [ ] ID card visible on chest
- [ ] Shirt tucked in

### **Category 2: Partial Uniform** ⚠️
Examples to capture:
- [ ] Uniform shirt but wrong pants
- [ ] Correct uniform but shirt untucked
- [ ] Missing ID card
- [ ] Wrong shoes (sneakers instead of formal)
- [ ] Missing one or more items

### **Category 3: No Uniform** ❌
- [ ] T-shirt and jeans
- [ ] Casual shirt
- [ ] Traditional clothes
- [ ] Sports wear
- [ ] Any non-uniform clothes

---

## 🚀 **Quick Start**

1. **Capture images now:**
   ```powershell
   D:/Smart_Vechicle/.venv/Scripts/python.exe capture_training_images.py
   ```

2. **Check progress:**
   ```powershell
   D:/Smart_Vechicle/.venv/Scripts/python.exe capture_training_images.py --list
   ```

3. **When you have 50+ images, proceed to labeling!**

---

## ⏱️ **Time Estimate**
- Camera capture: 10-15 minutes (20 images per category)
- Manual copy: 5-10 minutes (if images already exist)

**Target: Get at least 50 images total before proceeding to next step!**
