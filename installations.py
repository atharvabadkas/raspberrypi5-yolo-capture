import sys
import platform

print("🔍 Python Version:", sys.version)
print("🖥️ Platform:", platform.platform())

try:
    import numpy as np
    print("✅ NumPy:", np.__version__)
except ImportError:
    print("❌ NumPy not installed")

try:
    import requests
    print("✅ Requests:", requests.__version__)
except ImportError:
    print("❌ Requests not installed")

try:
    import cv2
    print("✅ OpenCV:", cv2.__version__)
except ImportError:
    print("❌ OpenCV not installed")

try:
    import torch
    print("✅ PyTorch:", torch.__version__)
    print("  ┗ CUDA Available:", torch.cuda.is_available())
except ImportError:
    print("❌ PyTorch not installed")

try:
    import torchvision
    print("✅ TorchVision:", torchvision.__version__)
except ImportError:
    print("❌ TorchVision not installed")

try:
    from ultralytics import YOLO
    print("✅ Ultralytics YOLO installed")
except ImportError:
    print("❌ Ultralytics not installed")

try:
    from picamera2 import Picamera2
    print("✅ Picamera2 available")
except ImportError as e:
    print("❌ Picamera2 not available:", str(e))

try:
    import libcamera
    print("✅ libcamera Python bindings available")
except ImportError:
    print("❌ libcamera not available (but usually not needed directly)")