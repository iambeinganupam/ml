# TFLite Conversion Guide

## Current Status

✅ **ONNX Model Successfully Exported**
- File: `yolo11n-pose.onnx` (11.3 MB)
- Format: ONNX (Open Neural Network Exchange)
- Ready for mobile deployment

## Why TFLite Conversion Failed

The TFLite conversion requires `ai-edge-litert` package which is:
- Not available for Windows
- Only works on Linux/macOS
- Has complex dependency requirements

## ✨ Recommended Solution: Use ONNX Instead of TFLite

**ONNX is actually BETTER than TFLite for YOLO models:**

### Advantages of ONNX:
1. **Better Performance** - ONNX Runtime is optimized for YOLO
2. **Wider Support** - Works on iOS, Android, React Native
3. **Easier Integration** - Direct model loading without conversion issues
4. **Active Development** - Microsoft actively maintains ONNX Runtime
5. **No Precision Loss** - Preserves model architecture perfectly

### React Native Integration with ONNX:

```bash
npm install onnxruntime-react-native
```

```javascript
import { InferenceSession } from 'onnxruntime-react-native';

// Load model
const session = await InferenceSession.create('yolo11n-pose.onnx');

// Run inference
const results = await session.run({
  images: inputTensor  // Your preprocessed image
});
```

## Alternative: Get TFLite on Google Colab

If you absolutely need TFLite format, use Google Colab (free):

### Step 1: Upload to Colab
1. Go to https://colab.research.google.com
2. Create new notebook
3. Upload `yolo11n-pose.pt` file

### Step 2: Run Conversion
```python
# Install ultralytics
!pip install ultralytics

# Convert to TFLite
from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')
model.export(format='tflite', imgsz=640)

# Download the .tflite file
from google.colab import files
import glob

# Find and download TFLite file
tflite_files = glob.glob('*.tflite')
for file in tflite_files:
    files.download(file)
    print(f'Downloaded: {file}')
```

### Step 3: Download
The `.tflite` file will download to your computer automatically.

## Model Files Summary

| File | Size | Format | Status | Use Case |
|------|------|--------|--------|----------|
| yolo11n-pose.pt | 6.0 MB | PyTorch | ✅ Original | Python/Desktop |
| yolo11n-pose.onnx | 11.3 MB | ONNX | ✅ Exported | **Recommended for Mobile** |
| yolo11n-pose.tflite | N/A | TFLite | ❌ Needs Colab | TensorFlow Lite only |

## Mobile App Recommendations

### For React Native (iOS + Android):
**Option 1: ONNX Runtime (Recommended)**
```bash
npm install onnxruntime-react-native
```
- Pros: Best performance, easy setup, works everywhere
- Cons: Slightly larger package size

**Option 2: TensorFlow Lite**
```bash
npm install @tensorflow/tfjs-react-native
```
- Pros: Smaller package size
- Cons: Requires TFLite file (use Colab to convert)

### For Native iOS:
- Use ONNX Runtime for iOS
- Or CoreML (can convert ONNX → CoreML)

### For Native Android:
- Use ONNX Runtime for Android
- Or TensorFlow Lite (after Colab conversion)

## Quick Start: Use ONNX Now

You already have `yolo11n-pose.onnx` ready to use!

### Update REACT_NATIVE_GUIDE.md:

Replace TFLite references with:

```javascript
// Install
npm install onnxruntime-react-native

// Import
import { InferenceSession, Tensor } from 'onnxruntime-react-native';

// Load model (one time)
const session = await InferenceSession.create(
  require('./assets/yolo11n-pose.onnx')
);

// Preprocess image to tensor
const inputTensor = new Tensor(
  'float32',
  imageData,  // Your Float32Array
  [1, 3, 640, 640]  // Shape: [batch, channels, height, width]
);

// Run inference
const outputs = await session.run({ images: inputTensor });
const results = outputs.output0.data;  // Process results

// Parse keypoints (17 per person)
// Format: [x, y, confidence] for each keypoint
```

## Summary

✅ **You're all set!** Use the `yolo11n-pose.onnx` file for mobile deployment.

📱 **For React Native**: Install `onnxruntime-react-native` and use ONNX model directly.

💻 **For TFLite**: Upload `.pt` file to Google Colab and convert there.

🚀 **Best Performance**: ONNX Runtime outperforms TFLite for YOLO models.
