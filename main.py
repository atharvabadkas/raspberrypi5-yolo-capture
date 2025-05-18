from model import ObjectDetector
import os
import time
import sys

def check_dependencies():
    try:
        import psutil
        print("psutil is installed for memory tracking")
    except ImportError:
        print("WARNING: psutil package not found")
        print("Installing psutil for memory usage tracking...")
        
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
            print("Successfully installed psutil")
        except Exception as e:
            print(f"Could not install psutil: {e}")
            print("Memory tracking will be limited")

def main():
    print("Starting Object Detection System on Raspberry Pi...")
    print("Using IMX219 Sony camera with Picamera2 module (1920x1080)")
    print("Press 'q' to quit")
    
    check_dependencies()
    
    models_dir = 'models'
    save_dir = 'captured_images'
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    detector = ObjectDetector(
        capture_index=0,  # For Picamera2, we use camera index 0
        conf_threshold=0.5,
        save_dir=save_dir,
        model_dir=models_dir
    )
    
    try:
        detector()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected, shutting down...")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        print("Object Detection System stopped")

if __name__ == "__main__":
    main() 