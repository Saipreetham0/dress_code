# Student Uniform Analysis System - Hardware Integration Guide

## System Overview
This system detects student uniform compliance using computer vision and outputs to Arduino/ESP32 microcontrollers.

---

## Trained Colors (from Uniform_Detection.v1i.yolov8 Dataset)

### ✅ College Uniform Colors Recognized:
1. **SHIRT**: Gray, Light Gray, White
2. **PANT**: Navy Blue, Dark Blue, Black  
3. **SHOES**: White, Black, Brown (any color accepted)
4. **ID CARD**: Red, Yellow, Blue, Green (lanyard/card colors)

### Decision Logic:
- **Output 1** = Complete uniform detected (all 4 components with correct colors)
- **Output 0** = Incomplete or wrong colors

---

## Optimizations for 30m Range Detection

### Camera Configuration:
```python
IMG_SIZE = 1280      # Larger resolution for distant objects
CONF_THRESH = 0.25   # Lower confidence for small/far objects
MIN_AREA = 800       # Smaller minimum box size
```

### Recommended Hardware:
- **Camera**: High-resolution webcam (1080p or 4K)
- **Zoom lens**: 3x-5x optical zoom for 30m range
- **Lighting**: Good ambient or IR illumination for distance

---

## Arduino/ESP32 Integration

### Python Side (PC/Raspberry Pi):
```python
# detect_and_classify.py outputs to serial:
print("1")  # Complete uniform
print("0")  # Incomplete uniform
```

### Arduino Code Example:
```cpp
void setup() {
  Serial.begin(9600);
  pinMode(LED_PIN, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    char result = Serial.read();
    
    if (result == '1') {
      digitalWrite(LED_PIN, HIGH);  // Green LED on
      // Allow entry
    } else if (result == '0') {
      digitalWrite(LED_PIN, LOW);   // Red LED on
      // Deny entry
    }
  }
}
```

### ESP32 Code Example (with WiFi):
```cpp
#include <WiFi.h>

void setup() {
  Serial.begin(115200);
  // WiFi setup for remote monitoring
}

void loop() {
  if (Serial.available() > 0) {
    String result = Serial.readStringUntil('\n');
    
    if (result == "1") {
      // Send to cloud: "Student in uniform"
      // Trigger relay for door unlock
    } else {
      // Send alert: "Non-uniform detected"
    }
  }
}
```

---

## Serial Communication Setup

### Windows:
```powershell
# Run detection and pipe to COM port
python detect_and_classify.py --webcam | Set-Content -Path COM3
```

### Linux/Raspberry Pi:
```bash
# Run and send to serial
python3 detect_and_classify.py --webcam > /dev/ttyUSB0
```

### Python Serial Library (Recommended):
```python
import serial

ser = serial.Serial('COM3', 9600)  # Adjust port
while True:
    result = detect_uniform()  # Returns 1 or 0
    ser.write(f"{result}\n".encode())
```

---

## Fixes Applied

### 1. False Positive Filtering:
- **Spatial filtering**: Rejects detections in wrong frame regions
  - Pant in top 25% → Rejected (likely head)
  - Shirt in bottom 20% → Rejected (likely legs)
  - Shoes must be in bottom half

### 2. Long-Range Optimization:
- Increased image size to 1280px
- Lowered confidence threshold to 0.25
- Reduced minimum area to 800px²

### 3. Color Expansion:
- Shirt: Added white (was gray only)
- Pant: Navy blue, dark blue, black
- Shoes & ID: Any color accepted

---

## Deployment Checklist

- [ ] Install high-resolution camera with zoom lens
- [ ] Test detection at 10m, 20m, 30m distances
- [ ] Calibrate lighting (add IR if needed)
- [ ] Connect Python output to Arduino/ESP32 via serial
- [ ] Test relay/actuator control (door lock, LED, buzzer)
- [ ] Deploy system at entrance with proper camera angle

---

## Troubleshooting

**Issue**: False detections (head as pant)  
**Solution**: Spatial filtering now active (`USE_SPATIAL_FILTERING = True`)

**Issue**: Not detecting at 30m  
**Solution**: Use higher resolution camera (1080p+) and zoom lens

**Issue**: Slow detection  
**Solution**: Use GPU (`--device 0` when training), reduce IMG_SIZE to 640 for CPU

**Issue**: Missing ID card  
**Solution**: Ensure good contrast (bright lanyard against shirt)

---

## Next Steps for Improvement

1. **Collect more training data**: 200+ images at various distances (5m, 10m, 20m, 30m)
2. **Retrain model**: Include long-distance samples in dataset
3. **Add person tracking**: Track individuals across frames (reduce false triggers)
4. **Multi-camera setup**: Cover multiple entry points
5. **Database integration**: Log entry times and uniform compliance stats

---

## Support

For hardware-specific integration questions, refer to:
- Arduino: https://www.arduino.cc/reference/en/
- ESP32: https://docs.espressif.com/projects/esp-idf/
- PySerial: https://pyserial.readthedocs.io/
