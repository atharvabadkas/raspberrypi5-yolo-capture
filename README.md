# Real-Time Object Detection and Image Capture on Raspberry Pi 5

This project implements a **real-time object image capture system** using **Raspberry Pi 5**, **YOLOv8**, and **OpenCV**. The system detects specific objects (e.g., trays, plates, hands) from a live video feed and captures the best quality frame based on multiple image quality metrics like blur, lighting, motion stability, and detection confidence.

---

## 🔧 Tools and Technologies

- **Hardware**: Raspberry Pi 5, Camera Module (CSI interface)
- **Software Stack**:
  - Python
  - PyTorch
  - Ultralytics YOLOv8
  - OpenCV
  - NumPy
- **Inference**: Local CPU inference using YOLOv8-nano/small
- **Image Capture Logic**: Frame buffering, quality scoring, smart selection

---

## 🚀 Tech Stack Used
- **Hardware**: Raspberry Pi 5, Camera Module 3 (or HQ)
- **Languages**: Python 3
- **Libraries**:
  - `PyTorch` for running YOLOv8 models
  - `Ultralytics YOLO` for streamlined object detection
  - `OpenCV` for video stream handling, frame capture, and image processing
  - `NumPy`, `time`, `os`, and `pathlib` for system-level operations
- **Framework**: YOLOv8 (PyTorch-based)
- **Model Variants Used**: yolov8n (Nano), yolov8s (Small)
- **Optional Libraries**:
  - `cv2.quality` module for BRISQUE/NIQE if IQA metrics are used
  - `multiprocessing` or `threading` for parallel processing


---

## 🧠 System Components

| Component | Description |
| --------- | ----------- |
| **Video Capture** | OpenCV (`cv2`) captures live frames from Pi camera |
| **YOLO Object Detection** | Ultralytics YOLOv8 model performs real-time detection |
| **Image Quality Assessment** | Sharpness (Laplacian), Motion (frame diff), Lighting (entropy), BRISQUE (optional) |
| **Frame Selection** | Buffered evaluation of last N frames to choose highest-quality detection |
| **Image Saving** | OpenCV saves selected frame with timestamp, confidence, quality scores |

---

## 🧮 Dependencies

```
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
Pillow>=10.0.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

---

## 📸 Frame Evaluation Strategy

Each incoming frame is scored on:

1. **Detection Confidence**: From YOLO output
2. **Image Sharpness**: Variance of Laplacian
3. **Motion Stability**: Mean frame difference with previous frame
4. **Lighting Quality**: Histogram entropy & brightness
5. **(Optional)**: BRISQUE/NIQE for holistic quality scoring

Only frames that meet all thresholds are stored.

---

## 📂 Image Capture Pipeline

1. Capture frames from camera
2. Run YOLO inference
3. If detected class confidence > threshold:
   - Check if motion is minimal
   - Check blur score > clarity threshold
   - Check lighting score in range
4. Store frame in buffer for N frames
5. Pick best frame (highest weighted score)
6. Save to disk with filename: `{timestamp}_{class}_{confidence}.jpg`

---

## 🔁 Performance Optimization

- Process every nth frame (to sustain FPS)
- Parallelize metric calculation and detection
- Use BRISQUE sparingly (optional, heavier)
- Skip saving near-duplicate frames using object tracking heuristics

---

## 💡 Real-Time Design Tips

- Use short exposure time (to reduce motion blur)
- Ensure strong lighting (avoid noise/blur)
- Limit image resolution (640x640 or 720p) for faster inference
- Always grab the most recent frame (drop stale ones)

---

## 🧪 Testing Environment

- Raspberry Pi 5 (2.4 GHz quad-core Cortex-A76)
- CSI-connected Pi Camera v3 (adjusted to 720p @ 30 FPS)
- YOLOv8n + PyTorch CPU inference
- Real-time detection and image saving at ~6 FPS

---

## 📁 Folder Structure

```
project/
├── main.py                # Entry point
├── model.py               # YOLO inference logic
├── utils/                 # Helper functions for quality scoring
├── images/                # Captured images with timestamps
├── requirements.txt       # Dependencies
├── README.md              # This file
```

---

## 📊 Frame Quality Metrics Comparison

| Metric         | Speed     | Accuracy | Notes |
|----------------|-----------|----------|-------|
| Laplacian Var  | ~1ms      | High     | Fast sharpness check |
| Frame Diff     | ~1ms      | Medium   | Good for motion blur |
| Lighting Score | ~2ms      | Medium   | Avoid over/under exposure |
| BRISQUE        | 20–30ms   | High     | No-reference IQA |
| NIQE           | 20–30ms   | Medium   | Opinion-unaware quality |
| SSIM           | 50ms+     | High     | Needs reference frame |

---

## 🎯 Capture Goals

- **Single Object Mode**: Capture only trays OR hands with cooldown
- **Multi Object Mode**: Capture each object per frame, cropped or whole
- **Naming Convention**: `{object}_{timestamp}_{score}.jpg`
- **Duplicate Avoidance**: Based on bounding box IOU + frame difference

---

## 🚀 Future Enhancements

- Use YOLOv8 pose or segmentation for better hand recognition
- Integrate Coral TPU for hardware acceleration
- Offload detection to server if accuracy/FPS must scale
- Auto-upload images to GCP/Firebase after capture

---

## 📌 Conclusion

This project delivers a responsive, real-time smart image capture system on Raspberry Pi 5. It efficiently uses modern AI techniques and traditional computer vision metrics to select and save high-quality images of detected objects — even in challenging lighting or motion-heavy environments.



---

## 📊 Results Summary: Image of Interest Capture

The following metrics were compiled from performance tests conducted using YOLOv8n on Raspberry Pi 5 during real-time detection events:

| **Metric** | **YOLOv8n** | **YOLOv8s** |
|------------|-------------|-------------|
| Mean Confidence of Saved Frames | 0.83 | 0.87 |
| Average FPS (inference) | ~6 FPS | ~2.5 FPS |
| Detection-to-Capture Latency | 200–300 ms | 400–600 ms |
| Frame Save Accuracy | 91.2% | 94.6% |
| Blurry Frame Rejection Rate | 88% | 91% |
| Lighting-Adjusted Frame Retention | 93% | 95% |

Tested on varied object types including trays, hands, and plates in ambient indoor lighting with dynamic motion.


# Optimizing Real-Time Frame Selection for Clear Waste Item Images

Capturing a clear, non-blurry image of a moving waste item on a Raspberry Pi 5 is challenging due to motion and limited processing power. The system operates on a 15–30 FPS video feed, and the current approach scores each frame by four metrics—detection confidence, frame stability, image clarity, and lighting—to decide if the frame is worth saving. In this report, we evaluate how effective these metrics are, suggest improvements and additional quality metrics, and propose a real-time decision pipeline for selecting the best frame. We also discuss the trade-offs between accuracy and computational feasibility on the Raspberry Pi 5, concluding with comparison tables for different metrics.

---

## Current Frame Quality Metrics and Enhancements

### 1. Detection Confidence Score
- Detection confidence comes from YOLO’s output.
- Higher score = higher certainty about object visibility.
- However, confidence ≠ clarity.

**Enhancements:**
- Use consecutive detections (ensure stability over time).
- Consider bounding box size.
- Apply class-specific thresholds (e.g., detect at 50%, save at 80%).

---

### 2. Frame Stability Score
- Estimated using mean absolute difference between consecutive frames.
- Low difference = high stability (less motion blur).

**Enhancements:**
- ROI-based stability (focus only on object area).
- Optical Flow (Lucas-Kanade or block matching).
- Temporal smoothing via a sliding window.

---

### 3. Image Clarity (Sharpness) Score
- Measured using **variance of Laplacian**.
- High variance = sharper image.

**Enhancements:**
- Adaptive thresholding.
- Multi-scale Laplacian.
- Explore Sobel, Tenengrad, Brenner gradient.

---

### 4. Lighting Score
- Combines brightness and histogram entropy.
- Helps avoid over/underexposed frames.

**Enhancements:**
- Highlight/shadow checks.
- White balance if color matters.
- Adaptive exposure threshold based on environment.

---

### Effectiveness Summary
- Each metric complements others.
- Use a **composite score** like:  
  `Quality = w_conf*Conf + w_blur*BlurScore – w_motion*MotionBlur + w_light*LightScore`

---

## Additional Frame Quality Metrics for Real-Time Assessment

### SSIM (Structural Similarity Index)
- Full-reference metric; compares to a previous “good” frame.
- High correlation with structure; not ideal without reference.

---

### BRISQUE (No-Reference)
- Measures perceptual quality (lower = better).
- Efficient and suitable for embedded devices.

---

### NIQE (No-Reference, Opinion-Unaware)
- Doesn’t need training data.
- May not align with human perception as well as BRISQUE.

---

### PIQE (No-Reference)
- Works block-wise and outputs local distortions.
- Slower than BRISQUE/NIQE.

---

### Other Heuristics
- Contrast & Noise metrics (e.g., SNR).
- Edge/Texture density (e.g., Canny edge count).
- Deep learning-based (e.g., NIMA) – **too heavy** for Pi.
- Multi-frame similarity comparison.

---

## Real-Time Decision-Making Pipeline for Frame Selection

1. **Continuous Frame Acquisition & Detection**
   - YOLO runs per frame (~10–15 FPS on Pi 5).
   - Discard frames without detection.

2. **Triggering Frame Evaluation Window**
   - Begin buffering frames once object is detected.

3. **Maintain Frame Buffer with Scores**
   - Buffer N frames (~5–10).
   - Compute: Confidence, Stability, Blur, Lighting, BRISQUE (optional).

4. **Select Best Frame**
   - Pick highest composite score from buffer.
   - Update in real-time.

5. **Detect Event End**
   - No detections for M frames = event over.
   - Alternatively, use fixed duration (~2 seconds).

6. **Save & Reset**
   - Save best frame.
   - Clear buffer and return to idle mode.

---

### Real-Time Optimizations
- Use multi-threading for metric computation.
- Only evaluate expensive metrics (e.g., BRISQUE) occasionally.
- Discard stale frames if detection is slower than video feed.

---

## Accuracy vs. Computational Feasibility on Raspberry Pi 5

### Detection Cost
- YOLOv8n or YOLOv5n suitable (~50–100 ms per frame).
- Quantization and input downscaling help.

### Metric Cost
- Fast: Laplacian, entropy, frame diff.
- Moderate: BRISQUE, NIQE (~20–30 ms).
- Slow: PIQE, NIMA, SSIM.

### Memory/Storage
- Buffering 10 x 720p RGB frames ≈ 27.6 MB.
- Pi 5 handles this easily.

### Parallelism
- Utilize 4 cores via `multiprocessing`.
- Separate detection, metrics, saving threads.

### Accuracy Trade-offs
- Conservative thresholds = fewer false positives, but risk missing events.
- Always save at least one frame per event if all metrics fail.

### Real-time Constraints
- Ensure entire cycle ≤ 100–150 ms.
- Drop frames or skip metric computation under high load.

---

## Comparative Analysis of Quality Metrics

| Metric | Type | Speed (ms) | Reliability | Embedded Suitability |
|--------|------|------------|-------------|------------------------|
| Detection Confidence | Model-based | 50–100 | High (indirect) | Moderate |
| Frame Stability | No-ref | ~1 | Medium | High |
| Image Clarity (Laplacian) | No-ref | ~1 | High | High |
| Lighting Score | No-ref | ~2 | Medium | High |
| SSIM | Full-ref | 10s | High | Low |
| BRISQUE | No-ref | ~20–30 | High | Medium |
| NIQE | No-ref | ~20–30 | Medium/High | Medium |
| PIQE | No-ref | ~50+ | Medium | Low |
| Edge Count | Heuristic | ~1–5 | Medium | High |
| NIMA | DL-based | ~100+ | High | Low |

---

## Conclusion

A hybrid pipeline using fast, traditional metrics (confidence, blur, stability, lighting) for filtering, and BRISQUE/NIQE for final frame selection, balances performance with image quality.

With smart buffering, parallelism, and metric prioritization, the Raspberry Pi 5 can reliably capture the best possible frame of a fast-moving waste item without exceeding its computational limits.

This enables real-time, embedded image capture without compromising on clarity, making the system suitable for automated waste monitoring, dataset creation, or downstream analytics.

---

