Here is your **updated README**, modified to reflect:

* ✅ SCRFD + ArcFace pipeline
* ✅ Picamera2 switch via `IS_RASPBERRY_PI`
* ✅ Graph generation (`People vs Time`)
* ✅ New CSV summary format
* ✅ Integer-only Y-axis in plot
* ✅ Embedding-based ID system (not Haar + distance tracker anymore)

---

# Human Proximity Attention Tracker (SCRFD + ArcFace)

Lightweight AI-powered attention analytics system that:

* Detects faces using **SCRFD**
* Identifies unique people using **ArcFace embeddings**
* Tracks per-person attention duration
* Generates:

  * 📊 Attention CSV Report
  * 📈 People vs Time graph
* Supports:

  * 💻 Desktop (cv2)
  * 🍓 Raspberry Pi (Picamera2 – low latency)

---

# Installation

## 1️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Required Libraries

* `opencv-python`
* `onnxruntime`
* `numpy`
* `matplotlib`
* `python-vlc`
* `picamera2` (Raspberry Pi only)

---

## 2️⃣ Install System VLC (Required for Audio)

### Windows

Download from:
[https://www.videolan.org/vlc/](https://www.videolan.org/vlc/)

### macOS

```bash
brew install vlc
```

### Linux / Raspberry Pi

```bash
sudo apt update
sudo apt install vlc
```

---

## 3️⃣ Install Picamera2 (Raspberry Pi Only)

If using:

```python
IS_RASPBERRY_PI = True
```

Install:

```bash
sudo apt install -y python3-picamera2
```

---

# Configuration

Edit `config.py`:

```python
# Display
DEBUG = True

# Tracking
STALE_FACE_TIMEOUT = 3.0

# Platform
IS_RASPBERRY_PI = False  # True = Picamera2 | False = cv2

# Audio
Campaign_name = "default"
AUDIO_FILE = None
```

---

## Configuration Examples

### Desktop with GUI

```python
DEBUG = True
IS_RASPBERRY_PI = False
```

### Raspberry Pi Headless (Recommended)

```python
DEBUG = False
IS_RASPBERRY_PI = True
```

---

# Camera Backend Logic

The system automatically switches capture backend:

### Desktop

```python
cv2.VideoCapture(0)
```

### Raspberry Pi

```python
Picamera2.capture_array()
```

Picamera2 advantages:

* Lower latency
* Stable FPS
* Native Raspberry Pi camera stack
* No extra frame conversion

---

# Running

```bash
python main.py
```

### GUI Mode

* `DEBUG = True`
* Press **Q** to exit

### Headless Mode

* `DEBUG = False`
* Press **Ctrl+C** to exit

---

# System Pipeline (Actual Architecture)

```
Frame Capture (cv2 or Picamera2)
        ↓
SCRFD Face Detection
        ↓
Head Pose Filtering (Yaw + Pitch)
        ↓
ArcFace Embedding Extraction
        ↓
Cosine Similarity Matching
        ↓
Assign / Reuse Face ID
        ↓
Update Attention Time
        ↓
Cleanup Stale Faces
        ↓
Generate:
   • CSV Report
   • Summary
   • People vs Time Graph
```

---

# Terminal Output Example

```
FPS: 31.2 | Detected: 2 | Tracked: 1
FPS: 30.8 | Detected: 1 | Tracked: 1
FPS: 29.9 | Detected: 0 | Tracked: 1
```

### Meaning

* **Detected** → Faces detected in current frame
* **Tracked** → Unique active people being tracked
* **FPS** → Performance metric

---

# CSV Output

Auto-saved to:

```
Human_proximity_Results/attention_report_<timestamp>.csv
```

---

## Example CSV

```csv
Face_ID,Attention_Time_s,start_time,end_time,Total_Time_s
1,8.0,2.1,10.1,8.0
2,5.5,12.2,18.4,6.2
3,3.2,20.5,25.0,4.5

Summary
Total_People_Watched,3
Total_Attention_Time_s,16.7
Average_Attention_Time_s,5.57
Campaign_Duration_s,45.3
```

---

## Metrics Explained

| Metric               | Meaning                         |
| -------------------- | ------------------------------- |
| Face_ID              | Unique embedding-based identity |
| Attention_Time_s     | Seconds looking at camera       |
| Total_Time_s         | Total seconds on screen         |
| Total_People_Watched | Unique identities detected      |
| Campaign_Duration_s  | Total runtime                   |

---

# Graph Output

Also auto-saved:

```
Human_proximity_Results/people_vs_time_<timestamp>.png
```

### Graph Details

* X-axis → Time (seconds)
* Y-axis → Number of People (Integer only)
* Automatically enforced using:

```python
MaxNLocator(integer=True)
```

---

# Identity System (ArcFace Based)

Instead of simple distance tracking:

* Extract 512-D normalized embedding
* Compute cosine similarity
* If similarity > threshold → reuse ID
* Else → create new ID

```python
sim = np.dot(a, b)
```

Default similarity threshold:

```python
0.45
```

This allows:

* Re-identification after short disappearance
* More stable identity tracking
* Person-level analytics

---

# Performance Notes (Important)

For Raspberry Pi:

* Use 640x480 resolution
* Consider computing embedding every N frames
* Avoid running GUI (`DEBUG = False`)
* Use Picamera2 backend

---

# Output Artifacts Per Session

Each run generates:

* 📄 CSV Report
* 📊 Summary printed in terminal
* 📈 People vs Time graph

No data is overwritten (timestamped).

---

# Campaign Audio

Place audio in:

```
Campain_Audio/<Campaign_name>/
```

System automatically:

* Searches folder
* Plays first audio file
* Stops when no one is present

---

# Hardware Requirements

### Minimum (Desktop)

* CPU with AVX support
* 4GB RAM

### Raspberry Pi Recommended

* Raspberry Pi 4 (4GB or 8GB)
* Official Pi Camera Module
* 64-bit OS

---

# Use Cases

* Retail attention analytics
* Campaign performance tracking
* Exhibition engagement measurement
* Smart advertising display systems


## Resources & Tools

### Audio Generation
Need to generate advertisement audio files for your campaigns? Use:
- **[Wondercraft AI Studio](https://www.wondercraft.ai/studio)** – AI-powered audio/voiceover generation
  - Generate natural-sounding ads and announcements
  - Download as MP3 and place in `Campain_Audio/<Campaign_name>/` folder
  - Supports multiple languages and voice styles