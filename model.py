import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from datetime import datetime
import psutil 

class ObjectDetector:
    def __init__(self, capture_index=0, conf_threshold=0.5, save_dir='captured_images', model_dir='models'):
        # Initialize camera device
        self.capture_index = capture_index
        self.conf_threshold = conf_threshold
        
        # Create directory for saving images - ensure absolute path
        if not os.path.isabs(save_dir):
            self.save_dir = os.path.abspath(save_dir)
        else:
            self.save_dir = save_dir
        
        print(f"Images will be saved to: {self.save_dir}")
        
        # Make sure directory exists with correct permissions
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            import stat
            os.chmod(self.save_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO) 
            print(f"Set full permissions on {self.save_dir}")
            # Test if directory is writable
            test_file = os.path.join(self.save_dir, "test_write.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"Confirmed {self.save_dir} is writable")
        except Exception as e:
            print(f"WARNING: Directory permission issue: {e}")
            print(f"Will attempt to continue, but image saving may fail.")
        
        # Create directory for models
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Set model path - using standard YOLOv8n
        self.model_path = os.path.join(self.model_dir, 'yolov8n.pt')
        
        # Initialize device
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        self._load_model()
        
        # Get class names
        self.class_names = self.model.names
        print(f"Available classes: {self.class_names}")
        
        # Person class ID (is 0 in COCO dataset)
        self.person_class_id = 0
        
        # Variables for image quality assessment
        self.prev_frame = None
        
        # Performance metrics tracking
        self.detection_times = []
        self.capture_times = []
        self.frame_times = []
        self.latency_times = []
        self.vram_usage = []
        self.total_detections = 0
        self.total_captures = 0
        self.ram_usage = []
        self.disk_usage = []
        self.cpu_usage = []
        self.cpu_temp = [] 
        
        # Capture delay tracking
        self.last_capture_time = 0
        self.capture_delay = 3 
    
    def _load_model(self):
        try:
            if os.path.exists(self.model_path):
                print(f"Loading model from {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                print(f"Model not found at {self.model_path}, downloading...")
                self.model = YOLO('yolov8n.pt')
                # Save the model for future use
                self.model.save(self.model_path)
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting to download model...")
            try:
                self.model = YOLO('yolov8n.pt')
                self.model.save(self.model_path)
                return True
            except Exception as e2:
                print(f"Failed to download model: {e2}")
                raise
    
    def detect_blur(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def detect_stability(self, frame):
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return 100.0 
        
        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference between frames
        frame_diff = cv2.absdiff(gray, self.prev_frame)
        self.prev_frame = gray
        
        # Return the mean difference as stability score
        return np.mean(frame_diff)
    
    def assess_lighting(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean brightness
        mean_brightness = np.mean(gray)
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Normalize histogram to get PDF
        hist_norm = hist / hist.sum()
        
        # Calculate entropy as a measure of balanced exposure
        non_zero_vals = hist_norm[hist_norm > 0]
        entropy = -np.sum(non_zero_vals * np.log2(non_zero_vals))
        
        # Return a composite score
        brightness_score = 1.0 - 2.0 * abs(mean_brightness - 128) / 255
        
        return 0.5 * brightness_score + 0.5 * (entropy / 8.0)  # 8 is max entropy for 8-bit image
    
    def capture_best_frame(self, frame, detection_info):
        # Start timer for image capture
        capture_start_time = time.time()
        
        # Check if we're still in the delay period after last capture
        current_time = time.time()
        if current_time - self.last_capture_time < self.capture_delay:
            # still in the delay period, don't capture
            time_left = self.capture_delay - (current_time - self.last_capture_time)
            print(f"In capture delay period, {time_left:.1f} seconds left before next capture")
            return False
        
        # Assess frame quality
        blur_score = self.detect_blur(frame)
        stability_score = self.detect_stability(frame)
        lighting_score = self.assess_lighting(frame)
        
        # thresholds for quality metrics - relaxed for easier capture
        blur_threshold = 50.0  # Lowered from 100
        stability_threshold = 100.0  # Increased from 5
        lighting_threshold = 0.3  # Lowered from 0.5
        
        print(f"Quality scores - Blur: {blur_score:.2f}, Stability: {stability_score:.2f}, Lighting: {lighting_score:.2f}")
        
        # Get class name
        class_name = "person"
        
        # Check if we should classify this as a hand
        is_hand = detection_info.get("is_hand", False)
        if is_hand:
            class_name = "hand"
        
        # We capture frames even if quality isn't perfect
        # At least one quality metric should pass
        capture_success = False
        if (blur_score > blur_threshold or 
            stability_score < stability_threshold or 
            lighting_score > lighting_threshold):
            
            # Get detection metrics for filename
            detection_time = detection_info.get("detection_time", 0) * 1000

            # Get latest FPS from frame times (or 0 if no frames yet)
            fps = 1.0 / self.frame_times[-1] if self.frame_times else 0

            # Get latest latency (or 0 if no frames yet)
            latency = self.latency_times[-1] * 1000 if self.latency_times else 0

            # Get latest VRAM usage (or 0 if no data yet)
            vram = self.vram_usage[-1][0] if self.vram_usage else 0  # in MB
            
            # Format the metrics for the filename (clean values for filesystem safety)
            det_time_str = f"{detection_time:.1f}ms"
            fps_str = f"{fps:.1f}fps"
            latency_str = f"{latency:.1f}ms"
            vram_str = f"{vram:.1f}MB"
            
            # Create filename with metrics
            # Format: WMSV4AI_Object Detected Time_Image Captured Time_FPS_Latency_VRAM Consumption
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save the frame
            try:
                # After capture, get the capture time 
                # (initialized here, will be updated after successful capture)
                capture_time = 0
                
                # Create base filename without capture time (will be added after capture)
                filename_base = f"WMSV4AI_{timestamp}_{det_time_str}_{fps_str}_{latency_str}_{vram_str}"
                
                # Temporary filename for saving
                temp_filename = f"{self.save_dir}/{filename_base}_capturing.jpg"
                
                print(f"Attempting to save image to temporary file...")
                success = cv2.imwrite(temp_filename, frame)
                
                if success:
                    # Calculate capture time
                    capture_time = time.time() - capture_start_time
                    capture_time_str = f"{capture_time*1000:.1f}ms"
                    
                    # Now create the final filename with capture time included
                    final_filename = f"{self.save_dir}/{filename_base}_{capture_time_str}.jpg"
                    
                    # Rename the temporary file to the final filename
                    os.rename(temp_filename, final_filename)
                    
                    print(f"Frame captured and saved as {final_filename}")
                    capture_success = True
                    self.total_captures += 1
                    
                    # Set the last capture time to enforce delay
                    self.last_capture_time = time.time()
                    print(f"Next capture will be allowed after a {self.capture_delay} second delay")
                else:
                    print(f"ERROR: cv2.imwrite failed to save image")
            except Exception as e:
                print(f"ERROR: Exception while saving image: {e}")
                import traceback
                traceback.print_exc()
            
        else:
            print(f"Frame quality insufficient for {class_name} - not saving")
        
        # Record capture time only if successful
        if capture_success:
            capture_time = time.time() - capture_start_time
            self.capture_times.append(capture_time)
            print(f"Image capture took {capture_time:.4f} seconds")
            
        return capture_success
    
    def plot_box(self, frame, box, label, color=(0, 255, 0)):
        x1, y1, x2, y2 = map(int, box)
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1-20), (x1+label_width, y1), color, -1)
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return frame
    
    def get_vram_usage(self):
        # Get current GPU memory usage in MB (if available)
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            return allocated, reserved
        else:
            # Return CPU memory usage as fallback on Raspberry Pi
            try:
                import psutil
                # Get process memory info in MB
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                return memory_info.rss / 1024 / 1024, 0
            except:
                return 0, 0
    
    def get_system_resources(self):
        try:
            # Get RAM usage (in MB)
            ram = psutil.virtual_memory()
            ram_used_mb = ram.used / 1024 / 1024
            ram_total_mb = ram.total / 1024 / 1024
            ram_percent = ram.percent
            
            # Get disk usage (in MB)
            disk = psutil.disk_usage('/')
            disk_used_mb = disk.used / 1024 / 1024
            disk_total_mb = disk.total / 1024 / 1024
            disk_percent = disk.percent
            
            # Get CPU usage (percent)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get CPU temperature (Raspberry Pi specific)
            cpu_temp = 0
            try:
                # Try to get temperature from thermal_zone0
                if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
                    with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                        temp = int(f.read().strip())
                        cpu_temp = temp / 1000.0  # Convert to Celsius
                # Fallback to vcgencmd for Raspberry Pi
                elif os.path.exists('/usr/bin/vcgencmd'):
                    import subprocess
                    temp = subprocess.check_output(['/usr/bin/vcgencmd', 'measure_temp'])
                    temp = temp.decode('utf-8')
                    # Extract temperature value (format is "temp=XX.X'C")
                    cpu_temp = float(temp.replace('temp=', '').replace('\'C', ''))
            except Exception as e:
                print(f"Could not get CPU temperature: {e}")
                cpu_temp = 0
            
            return {
                'ram': (ram_used_mb, ram_total_mb, ram_percent),
                'disk': (disk_used_mb, disk_total_mb, disk_percent),
                'cpu': cpu_percent,
                'cpu_temp': cpu_temp
            }
        except Exception as e:
            print(f"Error getting system resources: {e}")
            return {
                'ram': (0, 0, 0),
                'disk': (0, 0, 0),
                'cpu': 0,
                'cpu_temp': 0
            }
    
    def print_performance_metrics(self):
        print("\n======== Performance Metrics Summary ========")
        
        # VRAM Consumption
        avg_vram_allocated = np.mean([v[0] for v in self.vram_usage]) if self.vram_usage else 0
        avg_vram_reserved = np.mean([v[1] for v in self.vram_usage]) if self.vram_usage else 0
        print(f"1. VRAM Consumption:")
        print(f"   - Average Allocated: {avg_vram_allocated:.2f} MB")
        if avg_vram_reserved > 0:
            print(f"   - Average Reserved: {avg_vram_reserved:.2f} MB")
        
        # Object Detection Time
        avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0
        print(f"2. Object Detection Time:")
        print(f"   - Average: {avg_detection_time*1000:.2f} ms")
        print(f"   - Total Detections: {self.total_detections}")
        
        # Image Capture Time
        avg_capture_time = np.mean(self.capture_times) if self.capture_times else 0
        print(f"3. Image Capture Time:")
        print(f"   - Average: {avg_capture_time*1000:.2f} ms")
        print(f"   - Total Captures: {self.total_captures}")
        
        # Average FPS
        avg_fps = 1.0 / np.mean(self.frame_times) if self.frame_times else 0
        print(f"4. Average FPS: {avg_fps:.2f}")
        
        # Average Latency
        avg_latency = np.mean(self.latency_times) if self.latency_times else 0
        print(f"5. Average Latency: {avg_latency*1000:.2f} ms")
        
        # New system resource metrics
        if self.ram_usage:
            avg_ram_percent = np.mean([r[2] for r in self.ram_usage])
            avg_ram_used_mb = np.mean([r[0] for r in self.ram_usage])
            avg_ram_total_mb = np.mean([r[1] for r in self.ram_usage])
            print(f"6. RAM Usage:")
            print(f"   - Average: {avg_ram_percent:.2f}% ({avg_ram_used_mb:.2f} MB / {avg_ram_total_mb:.2f} MB)")
        
        if self.disk_usage:
            avg_disk_percent = np.mean([d[2] for d in self.disk_usage])
            avg_disk_used_mb = np.mean([d[0] for d in self.disk_usage])
            avg_disk_total_mb = np.mean([d[1] for d in self.disk_usage])
            print(f"7. Disk Usage:")
            print(f"   - Average: {avg_disk_percent:.2f}% ({avg_disk_used_mb:.2f} MB / {avg_disk_total_mb:.2f} MB)")
        
        if self.cpu_usage:
            avg_cpu = np.mean(self.cpu_usage)
            print(f"8. CPU Usage:")
            print(f"   - Average: {avg_cpu:.2f}%")
        
        # CPU Temperature metrics
        if self.cpu_temp:
            avg_temp = np.mean(self.cpu_temp)
            max_temp = max(self.cpu_temp)
            print(f"9. CPU Temperature:")
            print(f"   - Average: {avg_temp:.2f}°C")
            print(f"   - Maximum: {max_temp:.2f}°C")
            
            # Provide thermal status assessment
            if max_temp >= 80:
                print(f"   - Status: CRITICAL - CPU reached potentially damaging temperatures")
            elif max_temp >= 70:
                print(f"   - Status: WARNING - CPU running very hot, performance throttling likely")
            elif max_temp >= 60:
                print(f"   - Status: CAUTION - CPU running hot but within operational limits")
            else:
                print(f"   - Status: NORMAL - CPU temperature within safe operating range")
        
        print("============================================")
    
    def __call__(self):
        # Initialize video capture using Picamera2 only since it's working
        print("Initializing IMX219 Sony camera using Picamera2...")
        
        try:
            from picamera2 import Picamera2
            
            # Get a list of available cameras
            camera_info = Picamera2.global_camera_info()
            print(f"Available cameras: {camera_info}")
            
            # Initialize picamera2
            picam2 = Picamera2(0)  # Use camera index 0
            
            # Create a still configuration for better quality
            config = picam2.create_still_configuration(main={"size": (1920, 1080)})
            picam2.configure(config)
            
            # Start with longer timeout
            print("Starting camera with still configuration...")
            picam2.start()
            time.sleep(2)  # Give more time to initialize
            
            # Create a custom capture class that mimics OpenCV's VideoCapture
            class PiCameraCapture:
                def __init__(self, picam):
                    self.picam = picam
                    # Test frame capture to ensure camera is working
                    self._test_frame = self.picam.capture_array()
                    print(f"PiCamera initialized with resolution: {self._test_frame.shape[1]}x{self._test_frame.shape[0]}")
                
                def read(self):
                    try:
                        # Capture a frame
                        frame = self.picam.capture_array()
                        # Convert from RGB to BGR (OpenCV format)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        return True, frame
                    except Exception as e:
                        print(f"Error capturing frame: {e}")
                        return False, None
                
                def isOpened(self):
                    return True
                
                def release(self):
                    try:
                        self.picam.stop()
                        print("PiCamera released")
                    except Exception as e:
                        print(f"Error releasing PiCamera: {e}")
            
            # Create our custom capture object
            cap = PiCameraCapture(picam2)
            
            # Test frame capture
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                raise RuntimeError("Failed to read frame with picamera2")
            
            print("Successfully connected using picamera2")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize IMX219 camera with Picamera2: {e}")
        
        print("IMX219 Camera initialized successfully using Picamera2")
        
        # Main detection loop
        total_frames = 0
        fps_array = []
        capture_count = 0
        
        try:
            # For headless operation (no display), create a dummy window first
            # This prevents Qt/xcb display errors
            use_display = False
            try:
                # Check if DISPLAY is set
                if "DISPLAY" in os.environ and os.environ["DISPLAY"]:
                    # Try to create a tiny window to see if display works
                    cv2.namedWindow("Person Detection", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Person Detection", 320, 240)
                    use_display = True
            except Exception as e:
                print(f"Display not available: {e}")
                print("Running in headless mode (no window display)")
                use_display = False
                
            while True:
                # Measure start time for FPS calculation and total latency
                start_time = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame, retrying...")
                    # Try a few times before giving up
                    retry_count = 0
                    max_retries = 3
                    while retry_count < max_retries:
                        time.sleep(0.5)  # Wait a bit before retrying
                        retry_count += 1
                        ret, frame = cap.read()
                        if ret:
                            print("Successfully recovered frame")
                            break
                    
                    if not ret:
                        print(f"Failed to grab frame after {max_retries} retries, exiting...")
                        break
                
                # Start timer for detection
                detection_start_time = time.time()
                
                # Run detection
                results = self.model(frame)
                
                # Record detection time
                detection_time = time.time() - detection_start_time
                self.detection_times.append(detection_time)
                self.total_detections += 1
                
                # Record VRAM usage
                self.vram_usage.append(self.get_vram_usage())
                
                # Record system resource usage
                resources = self.get_system_resources()
                self.ram_usage.append(resources['ram'])
                self.disk_usage.append(resources['disk'])
                self.cpu_usage.append(resources['cpu'])
                self.cpu_temp.append(resources['cpu_temp'])
                
                # Process detections
                person_detected = False
                hand_detected = False
                
                # Extract detection results
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Get class id and confidence
                        cls_id = int(box.cls.item())
                        conf = box.conf.item()
                        
                        # Only process person class (class ID 0 in COCO dataset)
                        if cls_id == self.person_class_id and conf > self.conf_threshold:
                            # Person detected
                            person_detected = True
                            
                            # Get bounding box
                            xyxy = box.xyxy[0].cpu().numpy()
                            
                            # Get coordinates for hand detection
                            x1, y1, x2, y2 = map(int, xyxy)
                            box_width = x2 - x1
                            box_height = y2 - y1
                            
                            # Get frame dimensions
                            frame_height, frame_width = frame.shape[:2]
                            
                            # Check if box is near edge of frame (potentially a hand)
                            edge_margin = 100  # pixels
                            is_at_edge = (x1 < edge_margin or 
                                          y1 < edge_margin or 
                                          x2 > frame_width - edge_margin or 
                                          y2 > frame_height - edge_margin)
                            
                            # Check if box is small (relative to frame)
                            is_small = (box_width * box_height) < (frame_width * frame_height * 0.25)
                            
                            # If both conditions met, label as hand
                            is_hand = False
                            if is_small and is_at_edge:
                                hand_detected = True
                                is_hand = True
                                # Special highlight and label for hand
                                label = f"Hand {conf:.2f}"
                                color = (0, 0, 255)  # Red for hand
                            else:
                                # Regular person box
                                label = f"Person {conf:.2f}"
                                color = (0, 255, 0)  # Green for person
                            
                            # Draw the box with the appropriate label and color
                            self.plot_box(frame, xyxy, label, color)
                            
                            # Attempt to capture if it's a good frame
                            detection_info = {
                                "confidence": conf,
                                "box": xyxy,
                                "class_id": cls_id,
                                "is_hand": is_hand,
                                "detection_time": detection_time  # Add detection time to info dict
                            }
                            
                            # Try to save the frame
                            captured = self.capture_best_frame(frame, detection_info)
                            if captured:
                                # Draw capture indicator
                                cv2.putText(frame, "CAPTURED!", (50, 50), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                capture_count += 1
                                print(f"Total captures: {capture_count}")
                
                # Calculate total processing time (latency)
                end_time = time.time()
                frame_time = end_time - start_time
                self.frame_times.append(frame_time)
                
                # Record total latency (from frame read to complete processing)
                latency = frame_time
                self.latency_times.append(latency)
                
                # Calculate FPS
                fps = 1 / frame_time
                
                # Keep track of fps for averaging
                fps_array.append(fps)
                if len(fps_array) > 30:  # Average over the last 30 frames
                    fps_array.pop(0)
                avg_fps = sum(fps_array) / len(fps_array)
                
                # Display FPS on frame
                fps_text = f"FPS: {fps:.2f} (avg: {avg_fps:.2f})"
                cv2.putText(frame, fps_text, (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display detection metrics
                latency_text = f"Latency: {latency*1000:.1f}ms Det: {detection_time*1000:.1f}ms"
                cv2.putText(frame, latency_text, (20, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display memory usage 
                vram_allocated, vram_reserved = self.vram_usage[-1]
                memory_text = f"Memory: {vram_allocated:.1f}MB"
                cv2.putText(frame, memory_text, (20, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display system resource usage
                if self.ram_usage:
                    ram_used_mb = self.ram_usage[-1][0] 
                    ram_percent = self.ram_usage[-1][2]
                    ram_text = f"RAM: {ram_percent:.1f}% ({ram_used_mb:.0f} MB)"
                    cv2.putText(frame, ram_text, (20, 190),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if self.cpu_usage:
                    cpu_percent = self.cpu_usage[-1]
                    cpu_text = f"CPU: {cpu_percent:.1f}%"
                    cv2.putText(frame, cpu_text, (20, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if self.disk_usage:
                    disk_used_mb = self.disk_usage[-1][0]
                    disk_percent = self.disk_usage[-1][2]
                    disk_text = f"Disk: {disk_percent:.1f}% ({disk_used_mb:.0f} MB)"
                    cv2.putText(frame, disk_text, (20, 270),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display CPU temperature with color coding
                if self.cpu_temp:
                    temp = self.cpu_temp[-1]
                    # Choose color based on temperature
                    if temp >= 80:
                        color = (0, 0, 255)  # Red (danger)
                    elif temp >= 70:
                        color = (0, 165, 255)  # Orange (warning)
                    elif temp >= 60:
                        color = (0, 255, 255)  # Yellow (caution)
                    else:
                        color = (0, 255, 0)  # Green (normal)
                    
                    temp_text = f"CPU Temp: {temp:.1f}°C"
                    cv2.putText(frame, temp_text, (20, 310),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Display status message
                status = "No detections"
                if person_detected:
                    status = "Person Detected"
                if hand_detected:
                    status = "Hand Detected"
                if person_detected and hand_detected:
                    status = "Person & Hand Detected"
                
                cv2.putText(frame, status, (20, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Only try to show frame if display is available
                if use_display:
                    # Show frame
                    cv2.imshow("Person Detection", frame)
                    
                    # Break loop if 'q' pressed
                    if cv2.waitKey(1) == ord('q'):
                        break
                else:
                    # In headless mode, provide a way to exit with Ctrl+C
                    # Print status every 100 frames instead of showing window
                    if total_frames % 100 == 0:
                        print(f"Running in headless mode - FPS: {avg_fps:.2f}, Detected: {status}")
                
                # Increment frame counter
                total_frames += 1
                
                # Print performance stats every 100 frames
                if total_frames % 100 == 0:
                    print(f"Processed {total_frames} frames, current FPS: {fps:.2f}, avg FPS: {avg_fps:.2f}")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user. Shutting down...")
        except Exception as e:
            print(f"\nError in detection loop: {e}")
        finally:
            # Release resources
            cap.release()
            if use_display:
                cv2.destroyAllWindows()
            print("Camera released and windows closed.")
            print(f"Total images captured: {capture_count}")
            
            # Print performance metrics summary
            self.print_performance_metrics()

