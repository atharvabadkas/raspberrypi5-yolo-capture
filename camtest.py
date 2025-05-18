import cv2
import time
import os
import subprocess
import tempfile
import numpy as np

def test_picamera2():
    print("\nTesting connection using picamera2 module (RECOMMENDED METHOD)...")
    
    try:
        from picamera2 import Picamera2
        
        # Get info about available cameras
        camera_info = Picamera2.global_camera_info()
        print(f"Available cameras: {camera_info}")
        
        # Initialize camera
        picam2 = Picamera2(0)  # Use camera index 0
        
        # Create and set configuration - using still configuration for better quality
        config = picam2.create_still_configuration(main={"size": (1920, 1080)})
        picam2.configure(config)
        
        # Start camera
        print("Starting camera...")
        picam2.start()
        time.sleep(2)  # Give time to initialize
        
        # Capture frame
        print("Capturing frame...")
        frame = picam2.capture_array()
        
        # Get frame dimensions
        height, width, channels = frame.shape
        print(f"Successfully captured frame: {width}x{height}, {channels} channels")
        
        # Convert to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Save test image
        cv2.imwrite("test_imx219_picamera2.jpg", frame_bgr)
        print("Saved test image as test_imx219_picamera2.jpg")
        
        # Show frame if display available
        try:
            if "DISPLAY" in os.environ and os.environ["DISPLAY"]:
                cv2.imshow("IMX219 Test (picamera2)", frame_bgr)
                print("Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception as e:
            print(f"Could not display image: {e}")
        
        # Test continuous capture
        print("\nTesting continuous capture with Picamera2...")
        frames_captured = 0
        frame_times = []
        
        for i in range(10):  # Capture 10 frames
            start_time = time.time()
            
            # Capture frame
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Calculate and print FPS
            end_time = time.time()
            frame_time = end_time - start_time
            fps = 1 / frame_time
            frame_times.append(frame_time)
            
            print(f"Frame {i+1} captured in {frame_time:.4f} seconds ({fps:.2f} FPS)")
            frames_captured += 1
            
        # Calculate average FPS
        if frame_times:
            avg_time = sum(frame_times) / len(frame_times)
            avg_fps = 1 / avg_time
            print(f"Average frame time: {avg_time:.4f} seconds")
            print(f"Average FPS: {avg_fps:.2f}")
        
        # Stop camera
        picam2.stop()
        print("Camera stopped")
        return True
    except ImportError:
        print("picamera2 module not installed. Install with: sudo apt install -y python3-picamera2")
        return False
    except Exception as e:
        print(f"Error with picamera2: {e}")
        if 'picam2' in locals():
            try:
                picam2.stop()
            except:
                pass
        return False

def test_direct_camera():
    print("\nTesting direct connection to IMX219 camera via OpenCV (optional method)...")
    print("Note: This method is NOT expected to work and is kept for reference only.")
    
    # Try different camera indices that might work with IMX219
    for device in ["/dev/video2", "/dev/video0", "/dev/video3"]:
        try:
            print(f"Trying camera at {device}...")
            cap = cv2.VideoCapture(device)
            
            if not cap.isOpened():
                print(f"Failed to open {device}")
                cap.release()
                continue
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            # Try to read a frame
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame from {device}")
                cap.release()
                continue
            
            # Get frame dimensions
            height, width, channels = frame.shape
            print(f"Successfully captured frame: {width}x{height}, {channels} channels")
            
            # Save test image
            cv2.imwrite(f"test_imx219_{device.split('/')[-1]}.jpg", frame)
            print(f"Saved test image as test_imx219_{device.split('/')[-1]}.jpg")
            
            # Record camera properties
            print("\nCamera Properties:")
            print(f"Frames per second: {cap.get(cv2.CAP_PROP_FPS)}")
            print(f"Frame width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            print(f"Frame height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
            
            cap.release()
            return True
        except Exception as e:
            print(f"Error with {device}: {e}")
            if 'cap' in locals():
                cap.release()
    
    print("Direct camera connection failed for all devices (expected)")
    return False

def test_libcamera():
    print("\nTesting connection using libcamera-still command (optional method)...")
    print("Note: This method is NOT expected to work and is kept for reference only.")
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg') as temp:
            # Capture image using libcamera-still
            print(f"Capturing image to {temp.name}...")
            cmd = ["libcamera-still", "-n", "-o", temp.name, "--width", "1920", "--height", "1080", "--immediate"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                print(f"libcamera-still command failed: {result.stderr}")
                return False
            
            # Read the image with OpenCV
            frame = cv2.imread(temp.name)
            if frame is None:
                print(f"Failed to read captured image from {temp.name}")
                return False
            
            # Get frame dimensions
            height, width, channels = frame.shape
            print(f"Successfully captured frame: {width}x{height}, {channels} channels")
            
            # Save a copy of the test image
            cv2.imwrite("test_imx219_libcamera.jpg", frame)
            print("Saved test image as test_imx219_libcamera.jpg")
            
            return True
    except FileNotFoundError:
        print("libcamera-still command not found. Install with: sudo apt install -y libcamera-apps")
        return False
    except Exception as e:
        print(f"Error with libcamera: {e}")
        return False

def analyze_image(image_path):
    print(f"\nAnalyzing image quality for {image_path}...")
    
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return False
        
        # Get basic image properties
        height, width, channels = img.shape
        print(f"Image dimensions: {width}x{height}, {channels} channels")
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate blur measure (variance of Laplacian)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"Blur score: {blur_score:.2f} (higher is better, less blurry)")
        
        # Calculate brightness
        brightness = np.mean(gray)
        print(f"Brightness: {brightness:.2f}/255")
        
        # Calculate contrast (standard deviation of pixel values)
        contrast = np.std(gray)
        print(f"Contrast: {contrast:.2f}")
        
        # Calculate histogram and entropy
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        non_zero_vals = hist_norm[hist_norm > 0]
        entropy = -np.sum(non_zero_vals * np.log2(non_zero_vals))
        print(f"Entropy: {entropy:.2f}/8.0 (8.0 is max for 8-bit image)")
        
        # Overall quality assessment
        quality_score = (min(blur_score/100, 10) * 0.4 +  # Normalized blur score (40%)
                       (1 - abs(brightness - 128) / 128) * 0.3 +  # Normalized brightness score (30%)
                       (contrast / 128) * 0.15 +  # Normalized contrast score (15%)
                       (entropy / 8) * 0.15)  # Normalized entropy score (15%)
        
        quality_score = min(max(quality_score, 0), 1) * 10  # Scale to 0-10
        print(f"Overall quality score: {quality_score:.1f}/10")
        
        return True
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return False

def main():
    print("===== IMX219 Camera Test (Picamera2 Focus) =====")
    
    # First check what video devices are available
    try:
        video_devices = [f for f in os.listdir('/dev') if f.startswith('video')]
        print(f"Available video devices: {', '.join(video_devices)}")
    except Exception as e:
        print(f"Error checking video devices: {e}")
    
    # Run the recommended test first
    print("\n----- Testing Picamera2 Module (RECOMMENDED) -----")
    picamera2_result = test_picamera2()
    
    # If successful, analyze the image
    if picamera2_result and os.path.exists("test_imx219_picamera2.jpg"):
        analyze_image("test_imx219_picamera2.jpg")
    
    # Ask if the user wants to run optional tests
    try:
        run_optional = input("\nDo you want to run optional camera tests? (y/n): ").lower().strip() == 'y'
    except:
        run_optional = False
    
    if run_optional:
        # Try other methods if requested
        print("\n----- Testing Optional Methods -----")
        direct_result = test_direct_camera()
        libcamera_result = test_libcamera()
        
        # Print summary of all methods
        print("\n===== Test Results Summary =====")
        print(f"Picamera2 Module (RECOMMENDED): {'✅ SUCCESS' if picamera2_result else '❌ FAILED'}")
        print(f"Direct Camera Access (optional): {'✅ SUCCESS' if direct_result else '❌ FAILED (expected)'}")
        print(f"Libcamera Command (optional): {'✅ SUCCESS' if libcamera_result else '❌ FAILED (expected)'}")
    else:
        # Just print Picamera2 result
        print("\n===== Test Results Summary =====")
        print(f"Picamera2 Module (RECOMMENDED): {'✅ SUCCESS' if picamera2_result else '❌ FAILED'}")
    
    if picamera2_result:
        print("\n✅ Picamera2 test was successful!")
        print("You can now run the main application with 'python main.py'")
    else:
        print("\n❌ Picamera2 test failed. Recommendations:")
        print("1. Ensure camera is properly connected")
        print("2. Make sure the camera interface is enabled: sudo raspi-config")
        print("3. Install or reinstall picamera2: sudo apt install -y python3-picamera2")
        print("4. Reboot the Raspberry Pi: sudo reboot")

if __name__ == "__main__":
    main() 