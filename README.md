# Human Proximity Attention Tracker

Lightweight camera system that detects nearby people, identifies frontal faces (people looking at the camera), and tracks per-face attention duration. Designed to run on desktop/laptop and on Raspberry Pi (with Picamera2) for low-latency capture.

---

## Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies:**
- `opencv-python` – Video capture & face detection
- `python-vlc` – Audio playback

### 2. Install System VLC (Required for Audio)

**Windows:**
- Download from [https://www.videolan.org/vlc/](https://www.videolan.org/vlc/) and install

**macOS:**
```bash
brew install vlc
```

**Linux / Raspberry Pi:**
```bash
sudo apt update
sudo apt install vlc
```

### 3. (Optional) Install Picamera2 on Raspberry Pi
If using Raspberry Pi with `IS_RASPBERRY_PI = True`:
```bash
sudo apt install -y python3-picamera2
```

---

## Configuration

Edit `config.py` to customize behavior:

```python
# 1. Display Settings
DEBUG = True              # True: Show GUI with face boxes | False: Print to terminal only

# 2. Tracking Settings
STALE_FACE_TIMEOUT = 3.0  # Remove tracked face if not seen for > 3 seconds

# 3. Platform Settings
IS_RASPBERRY_PI = False   # True: Use Picamera2 on Pi | False: Use cv2.VideoCapture

# 4. Audio Settings
Campaign_name = "default"       # Campaign folder name (searches Campain_Audio/<Campaign_name>/)
AUDIO_FILE = None               # Fallback audio file if campaign folder not found
```

### Configuration Examples

**Desktop/Laptop with GUI:**
```python
DEBUG = True
IS_RASPBERRY_PI = False
Campaign_name = "default"
```

**Raspberry Pi Headless (No Display):**
```python
DEBUG = False
IS_RASPBERRY_PI = True
Campaign_name = "default"
```

**Custom Audio Campaign:**
```python
Campaign_name = "my_campaign"
# Create folder: Campain_Audio/my_campaign/
# Add MP3 files inside (first one will be played)
```

---

## Running the Code

### Option 1: GUI Mode (See Live Video)
```bash
python face_detection.py
```
**Requirements:**
- `config.DEBUG = True`
- Display/monitor connected
- Press **Q** to exit

### Option 2: Headless Mode (Terminal Output)
```bash
python face_detection.py
```
**Requirements:**
- `config.DEBUG = False`
- No display needed
- Press **Ctrl+C** to exit and show final report

### Option 3: Raspberry Pi (Headless, No GUI)
```bash
# config.py settings:
# IS_RASPBERRY_PI = True
# DEBUG = False

python face_detection.py
```

---

## Sample Output

### Live Terminal Output (GUI or Headless)
```
FPS: 32.1 | Detected: 0 | Tracked: 1
FPS: 30.9 | Detected: 0 | Tracked: 1
FPS: 31.1 | Detected: 0 | Tracked: 1
FPS: 30.4 | Detected: 0 | Tracked: 1
FPS: 32.3 | Detected: 1 | Tracked: 1
FPS: 29.8 | Detected: 0 | Tracked: 1
FPS: 32.9 | Detected: 1 | Tracked: 1
FPS: 20.5 | Detected: 0 | Tracked: 1
FPS: 32.2 | Detected: 1 | Tracked: 1
FPS: 31.0 | Detected: 1 | Tracked: 1
FPS: 32.3 | Detected: 1 | Tracked: 1
FPS: 31.2 | Detected: 1 | Tracked: 1

[!] Interrupted by user
```

**Output Explanation:**
- **FPS**: Frames per second (30 FPS typical for webcam)
- **Detected**: Number of frontal faces found in current frame
- **Tracked**: Number of faces actively being tracked across frames

### Final Report (On Exit)
```
============================================================
FINAL REPORT - Face Attention Times
============================================================
Face 1:
  Attention Time:  2.45s
  Total Time:      2.45s
  Attention %:     100.0%

Campaign Duration (Start → Stop):  2.45s
============================================================
```

**Report Explanation:**
- **Attention Time**: Duration face was detected in frame (seconds)
- **Total Time**: Duration face was being tracked (seconds)
- **Attention %**: (Attention Time / Total Time) × 100 = % of time looking at camera
- **Campaign Duration**: Overall time from script start to stop

### CSV Output (Auto-saved)
Each session automatically saves results to `Human_proximity_Results/attention_report_<timestamp>.csv`

**Example CSV Output:**
```csv
Face_ID,Attention_Time_s,Total_Time_s
1,8.0,8.0
2,5.5,6.2
3,3.2,4.5

Summary
Total_People_Watched,3
Total_Attention_Time_s,16.7
Average_Attention_Time_s,5.57
Campaign_Duration_s,45.3
```

**CSV Explanation:**
- **Face_ID**: Unique person identifier (each ID = one person)
- **Attention_Time_s**: Time that person looked at camera (frontal face detected)
- **Total_Time_s**: Total time that person was tracked on screen
- **Total_People_Watched**: Number of unique people detected
- **Total_Attention_Time_s**: Sum of all attention times across all people
- **Average_Attention_Time_s**: Average attention time per person
- **Campaign_Duration_s**: Overall time from script start to stop

---

## Pipeline Flow

```
Frame arrives → read_frame()
    ↓
detect_faces(frame)
    ↓
build_detections(faces)
    ↓
tracker.update(detections, curr_time, dt)
    ├─→ _compute_distances()      [All pairwise distances]
    ├─→ _greedy_assignment()      [One-to-one matching]
    ├─→ _assign_ids()             [Reuse or create face IDs]
    ├─→ _update_tracking()        [Update position & time]
    └─→ _cleanup_stale()          [Remove old faces]
    ↓
Display/Print Results
    ↓
Exit on Q or Ctrl+C → Print Final Report
```

---

## CSV Report Format

The system automatically saves detailed reports in CSV format after each session.

**Location:** `Human_proximity_Results/attention_report_<YYYYMMDD_HHMMSS>.csv`

**Example Report (3 people watched):**

| Face_ID | Attention_Time_s | Total_Time_s |
|---------|-----------------|--------------|
| 1 | 8.0 | 8.0 |
| 2 | 5.5 | 6.2 |
| 3 | 3.2 | 4.5 |
| | | |
| **Summary** | | |
| Total_People_Watched | 3 | |
| Total_Attention_Time_s | 16.7 | |
| Average_Attention_Time_s | 5.57 | |
| Campaign_Duration_s | 45.3 | |

**Key Metrics:**
- **Face_ID**: Unique person identifier (each ID = one person visit)
- **Attention_Time_s**: Seconds person looked at camera
- **Total_Time_s**: Total seconds person was on screen
- **Total_People_Watched**: Number of unique individuals tracked
- **Campaign_Duration_s**: Total runtime (script start → stop)

See [MODULE_STRUCTURE.md](MODULE_STRUCTURE.md) for detailed module documentation.

**Key Files:**
- `face_detection.py` – Main orchestrator
- `detector.py` – Face detection (camera + Haar cascade)
- `tracker.py` – Face tracking & ID assignment
- `audio_player.py` – Audio playback
- `config.py` – Configuration flags
- `requirements.txt` – Python dependencies

---

## Resources & Tools

### Audio Generation
Need to generate advertisement audio files for your campaigns? Use:
- **[Wondercraft AI Studio](https://www.wondercraft.ai/studio)** – AI-powered audio/voiceover generation
  - Generate natural-sounding ads and announcements
  - Download as MP3 and place in `Campain_Audio/<Campaign_name>/` folder
  - Supports multiple languages and voice styles