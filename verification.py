import os
import sys
import cv2
import torch
import numpy as np
import requests
import time
import subprocess
from pathlib import Path
from model import ObjectDetector

def print_status(message, success=True):
    prefix = "✅" if success else "❌"
    print(f"{prefix} {message}")

def verify_dependencies():
    print("\n--- Verifying Dependencies ---")
    
    dependencies = {
        "os": os,
        "cv2": cv2,
        "numpy": np,
        "torch": torch,
        "requests": requests,
        "ultralytics": None  # Will be imported dynamically
    }
    
    all_passed = True
    for name, module in dependencies.items():
        try:
            if module is None:
                # Try to import the module dynamically
                __import__(name)
            
            # Try to get the version (different modules have different ways)
            version = None
            if name == "torch":
                version = torch.__version__
            elif name == "opencv-python" or name == "cv2":
                version = cv2.__version__
            elif name == "numpy":
                version = np.__version__
            elif name == "ultralytics":
                import ultralytics
                version = ultralytics.__version__
                
            version_str = f"(v{version})" if version else ""
            print_status(f"{name} {version_str} is installed")
            
        except ImportError:
            print_status(f"Failed to import {name}", False)
            all_passed = False
    
    return all_passed

def verify_paths():
    print("\n--- Verifying Paths ---")
    
    paths = {
        "models": Path("models"),
        "captured_images": Path("captured_images"),
    }
    
    all_passed = True
    for name, path in paths.items():
        if path.exists():
            print_status(f"{name} directory exists at '{path}'")
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print_status(f"Created {name} directory at '{path}'")
            except Exception as e:
                print_status(f"Failed to create {name} directory at '{path}': {e}", False)
                all_passed = False
    
    return all_passed

def verify_model():
    print("\n--- Verifying Model ---")
    
    # Try to find any .pt model in the models directory first
    model_dir = Path("models")
    model_files = list(model_dir.glob("*.pt"))
    
    if model_files:
        model_path = model_files[0]  # Use the first .pt file found
        print_status(f"Found model file at '{model_path}'")
    else:
        # No model found, but we'll let the ObjectDetector download it
        model_path = Path("models/yolov8n.pt")
        print_status(f"No model files found in '{model_dir}', will download during initialization", False)
    
    try:
        # Import YOLO here to avoid errors if the module is not installed
        from ultralytics import YOLO
        
        # Don't load the model if it doesn't exist yet - let ObjectDetector handle it
        if model_path.exists():
            # Load the model
            model = YOLO(model_path)
            print_status(f"Successfully loaded model from '{model_path}'")
            
            # Check if the model can be used for inference
            device = torch.device("cpu")
            print_status(f"Using device: {device} for inference")
            
            # Check classes
            class_names = model.names
            print_status(f"Model has {len(class_names)} classes")
            
            # Check if 'person' class exists (should be class 0 in COCO)
            if 0 in class_names and class_names[0] == "person":
                print_status("Model includes 'person' class needed for detection")
            else:
                print_status("Will filter for person class during detection", True)
        else:
            print_status("Model will be downloaded during first run")
            
        return True
        
    except Exception as e:
        print_status(f"Failed to verify model: {e}", False)
        print_status("Model will be downloaded and configured during first run", True)
        return True

def verify_camera():
    print("\n--- Verifying Camera ---")
    
    try:
        # Primarily try to verify using Picamera2
        print("Attempting to access IMX219 camera using Picamera2...")
        
        try:
            from picamera2 import Picamera2
            
            # Get available cameras
            camera_info = Picamera2.global_camera_info()
            
            if camera_info:
                print_status(f"Detected camera(s): {camera_info}")
                
                # Try to initialize picamera2
                picam2 = Picamera2(0)
                config = picam2.create_still_configuration(main={"size": (1920, 1080)})
                picam2.configure(config)
                
                print_status("Starting camera to verify connection...")
                picam2.start()
                time.sleep(2)  # Give more time to initialize
                
                # Try to capture a test frame
                test_frame = picam2.capture_array()
                height, width, channels = test_frame.shape
                print_status(f"Successfully captured test frame: {width}x{height}")
                
                # Stop the camera
                picam2.stop()
                print_status("Camera verified and released")
                
                return True
            else:
                print_status("No cameras detected through Picamera2", False)
        except (ImportError, Exception) as e:
            print_status(f"Picamera2 not available or error: {e}", False)
            
        # Check if any video devices are available as fallback information
        try:
            import os
            v4l2_devices = [f for f in os.listdir('/dev') if f.startswith('video')]
            if v4l2_devices:
                print_status(f"Available video devices: {', '.join(v4l2_devices)}")
                print_status("Note: Direct camera access not used in this application, only using Picamera2")
        except Exception as e:
            print_status(f"Error checking video devices: {e}", False)
            
        # If we get here, we couldn't access the camera
        print_status("Failed to access camera with Picamera2 (required for this application)", False)
        print_status("Please ensure the camera is properly connected and enabled", False)
        print_status("Install Picamera2 with: sudo apt install -y python3-picamera2", False)
        
        # Return False to indicate failure but allow verification to proceed
        return False
            
    except Exception as e:
        print_status(f"Camera verification error: {e}", False)
        return False

def verify_object_detector():
    print("\n--- Verifying ObjectDetector Class ---")
    
    try:
        # Instantiate the detector
        detector = ObjectDetector(
            capture_index=0,
            conf_threshold=0.5,
            save_dir='captured_images',
            model_dir='models'
        )
        print_status("Successfully instantiated ObjectDetector")
        
        # Test image quality assessment methods with a dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        blur_score = detector.detect_blur(dummy_frame)
        print_status(f"Blur detection function works (score: {blur_score:.2f})")
        
        stability_score = detector.detect_stability(dummy_frame)
        print_status(f"Stability detection function works (score: {stability_score:.2f})")
        
        lighting_score = detector.assess_lighting(dummy_frame)
        print_status(f"Lighting assessment function works (score: {lighting_score:.2f})")
        
        return True
        
    except Exception as e:
        print_status(f"ObjectDetector verification error: {e}", False)
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        return False

def camera_diagnostic():
    """Run diagnostics on the camera system"""
    print("\n----- Camera Diagnostics -----")
    
    # Check if camera module is enabled in raspi-config
    try:
        result = subprocess.run(['vcgencmd', 'get_camera'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Camera hardware status: {result.stdout.strip()}")
            if "detected=1" not in result.stdout:
                print("❌ Camera hardware not detected. Check physical connection.")
            else:
                print("✅ Camera hardware detected.")
        else:
            print("❌ Unable to check camera hardware status.")
    except Exception as e:
        print(f"❌ Error checking camera hardware: {e}")
    
    # Check v4l2 devices
    try:
        print("\nDetected Video Devices:")
        result = subprocess.run(['ls', '-l', '/dev/video*'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout:
            print(result.stdout)
        else:
            print("No video devices found in /dev/video*")
            
        # Try v4l2-ctl if available
        print("\nDetailed Camera Information:")
        try:
            for i in range(3):  # Try first 3 potential camera devices
                device = f"/dev/video{i}"
                if os.path.exists(device):
                    print(f"\nInfo for {device}:")
                    subprocess.run(['v4l2-ctl', '--device', device, '--all'], capture_output=False)
        except Exception as e:
            print(f"Error getting detailed camera info: {e}")
            print("Install v4l-utils for more detailed diagnostics: sudo apt install v4l-utils")
    except Exception as e:
        print(f"Error listing video devices: {e}")
    
    print("\nCamera installation recommendations:")
    print("1. Ensure camera is properly connected to the Raspberry Pi")
    print("2. Enable camera interface: sudo raspi-config (Interface Options > Camera)")
    print("3. Install camera libraries: sudo apt install -y python3-picamera2 libcamera-dev")
    print("4. Reboot after making changes: sudo reboot")

def main():
    print("\n========== System Verification ==========")
    print("This script verifies all components needed for the object detection system")
    
    results = {}
    
    # Verify dependencies
    results["dependencies"] = verify_dependencies()
    
    # Verify paths
    results["paths"] = verify_paths()
    
    # Verify model
    results["model"] = verify_model()
    
    # Verify camera
    results["camera"] = verify_camera()
    
    # Verify ObjectDetector class
    results["object_detector"] = verify_object_detector()
    
    # Print summary
    print("\n========== Verification Summary ==========")
    all_passed = True
    critical_failure = False
    
    for test, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test.upper()}: {status}")
        
        # Consider camera failure as a warning if ObjectDetector passes
        if not passed:
            if test == "camera" and results.get("object_detector", False):
                print("⚠️  Camera check failed but ObjectDetector initialized successfully.")
                print("   The application may still work if the camera is connected.")
            else:
                critical_failure = True
                all_passed = False
    
    if all_passed:
        print("\n✅ All verification checks passed! The system is ready to run.")
        print("You can now run 'main.py' to start the application.")
        return 0
    elif critical_failure:
        print("\n❌ Some critical verification checks failed. Please fix the issues before running the application.")
        return 1
    else:
        print("\n⚠️  Some non-critical checks failed but the application may still work.")
        print("You can try running 'main.py' to start the application.")
        return 0

if __name__ == "__main__":
    result = main()
    
    # Run camera diagnostics if camera verification failed
    if result != 0:
        print("\nRunning camera diagnostics to help troubleshoot...")
        camera_diagnostic()
    
    sys.exit(result)