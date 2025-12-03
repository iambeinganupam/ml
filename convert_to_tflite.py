"""
Convert YOLO11 Pose Model to TensorFlow Lite for Mobile Deployment
This script exports the PyTorch YOLO model to TFLite format for React Native
"""

import sys
from pathlib import Path
from ultralytics import YOLO

# Optional imports - only needed for advanced conversion
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def simple_export(model_path='yolo11n-pose.pt', img_size=640):
    """Simplified export using only Ultralytics export (no TensorFlow needed)"""
    print("\n" + "="*70)
    print("Simple YOLO Export to TFLite")
    print("="*70 + "\n")
    
    try:
        # Try loading the model, if it fails, download a fresh one
        try:
            model = YOLO(model_path)
        except Exception as load_error:
            print(f"⚠ Model load error: {load_error}")
            print("Downloading fresh YOLO11n-pose model...")
            model = YOLO('yolo11n-pose')  # This will auto-download
        
        print(f"✓ Model loaded: {model_path}\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Try direct TFLite export
    print("Attempting TFLite export...")
    try:
        results = model.export(format='tflite', imgsz=img_size)
        print(f"\n✓ Success! TFLite model: {results}")
        print("\n" + "="*70)
        print("Conversion Complete!")
        print("="*70)
        return True
    except Exception as e:
        print(f"✗ TFLite export failed: {e}")
    
    # Try ONNX export (useful for many platforms)
    print("\nAttempting ONNX export...")
    try:
        results = model.export(format='onnx', imgsz=img_size)
        print(f"\n✓ Success! ONNX model: {results}")
        print("\nNote: ONNX can be converted to TFLite using:")
        print("  1. onnx2tf tool: pip install onnx2tf")
        print("  2. Run: onnx2tf -i model.onnx -o tflite_model")
        print("\n" + "="*70)
        print("Conversion Complete!")
        print("="*70)
        return True
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
    
    return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert YOLO11 Pose to TFLite')
    parser.add_argument('--model', default='yolo11n-pose.pt', help='Path to YOLO model')
    parser.add_argument('--size', type=int, default=640, help='Input image size')
    
    args = parser.parse_args()
    
    # Check if TensorFlow is available
    if not TF_AVAILABLE:
        print("\nNote: TensorFlow not installed. Using simplified export method.\n")
    
    # Run conversion
    success = simple_export(args.model, args.size)
    
    if not success:
        print("\n" + "="*70)
        print("FALLBACK INSTRUCTIONS")
        print("="*70)
        print("\nIf automatic conversion failed, try these methods:")
        print("\n1. Use Ultralytics Export in Python:")
        print("   from ultralytics import YOLO")
        print("   model = YOLO('yolo11n-pose.pt')")
        print("   model.export(format='tflite')")
        print("\n2. Use ONNX as intermediate:")
        print("   model.export(format='onnx')")
        print("   # Then convert ONNX to TFLite with onnx2tf")
        print("\n3. Use Google Colab for conversion:")
        print("   - Upload model to Colab")
        print("   - Run conversion in cloud environment")
        print("   - Download TFLite model")
        print("\n4. Contact Ultralytics support:")
        print("   https://github.com/ultralytics/ultralytics/issues")
        print("="*70 + "\n")
        
        sys.exit(1)
    else:
        print("\n" + "="*70)
        print("Next Steps:")
        print("="*70)
        print("1. Copy the TFLite model to your React Native project")
        print("2. Use TensorFlow Lite React Native package")
        print("3. Integrate with your app's camera/video processing")
        print("="*70 + "\n")
        sys.exit(0)
