# Face Detection & Attention Tracking System

## Project Structure

```
Human_proximity_sensor/
├── face_detection.py       ← Main script (clean, easy to read)
├── detector.py             ← Face detection logic (isolated module)
├── tracker.py              ← Face tracking & ID assignment (isolated module)
├── audio_player.py         ← Audio playback & campaign management
├── config.py               ← Configuration (DEBUG, Campaign, IS_RASPBERRY_PI)
├── requirements.txt        ← Dependencies
├── .gitignore              ← Git ignore rules
├── haarcascade_frontalface_default.xml  ← Haar cascade file
├── yolov8n.pt              ← YOLOv8 model
├── Campain_Audio/          ← Campaign audio folder (optional)
└── Human_proximity_Results/← Results output folder
```

---

## Module Overview

### 1. `face_detection.py` (Main Script)
**Purpose:** Orchestrates the entire pipeline. Clean, readable, easy to understand.

**Key Functions:**
- `main()` – Main execution loop
- `_visualize_debug()` – Display video with face boxes and IDs
- `_print_headless()` – Print stats to terminal (no GUI)
- `_print_final_report()` – Print final attention report

**Flow:**
```
Initialize Camera & Detector
    ↓
Loop:
    Read Frame → Detect Faces → Build Detections
    ↓
    Update Tracker (match IDs, accumulate time)
    ↓
    Display/Print Results
    ↓
    Exit on 'Q' or failure
    ↓
Print Final Report
```

---

### 2. `detector.py` (Detection Module)
**Purpose:** Encapsulates all camera and face detection logic.

**Class: FaceDetector**
| Method | Purpose |
|--------|---------|
| `__init__(cascade_path)` | Load cascade, open camera (cv2 or Picamera2 on Pi) |
| `read_frame()` | Get next video frame |
| `detect_faces(frame)` | Find all frontal faces in frame |
| `build_detections(faces)` | Convert face rectangles to detection dicts |
| `get_fps(dt)` | Calculate FPS |
| `release()` | Close camera & windows |

**Key Points:**
- Haar cascade **only detects frontal faces**, so detected = looking at camera
- Uses `cv2.VideoCapture` on desktop/laptop
- Uses `Picamera2` on Raspberry Pi when `IS_RASPBERRY_PI = True` (falls back to cv2 if unavailable)

---

### 3. `tracker.py` (Tracking Module)
**Purpose:** Encapsulates all face ID assignment and tracking logic.

**Class: FaceTracker**
| Method | Purpose |
|--------|---------|
| `update(detections, time, dt)` | Main update: match → assign IDs → track time |
| `_compute_distances(detections)` | Calculate all pairwise centroid distances |
| `_greedy_assignment(detections, pairs)` | One-to-one matching (greedy) |
| `_assign_ids(detections, assignments, time)` | Reuse matched IDs, create new ones |
| `_update_tracking(detections, time, dt)` | Update positions and times |
| `_cleanup_stale(time, timeout)` | Remove faces not seen for > timeout |
| `get_summary()` | Return dict of {face_id → attention_s, total_s} |

**Greedy Nearest-Neighbor Algorithm:**
1. Compute distances between all current and previous face centroids
2. Sort by distance (smallest first)
3. Iterate and assign: if both current and previous face unassigned AND distance < threshold, match them
4. Each face used at most once per frame = **no ID collisions**

---

### 4. `audio_player.py` (Audio Module)
**Purpose:** Encapsulates audio playback and campaign-based audio selection.

**Class: AudioPlayer**
| Method | Purpose |
|--------|---------|
| `find_campaign_audio(campaign_name, base_dir)` | Search `Campain_Audio/<Campaign_name>/` for first MP3 |
| `play(file_path)` | Start playing MP3 file |
| `stop()` | Stop audio playback |
| `is_playing()` | Check if audio is currently playing |
| `time_remaining()` | Get remaining time in current track |
| `manage(num_tracked)` | Auto-play while people present; let track finish naturally |

**Key Features:**
- Searches `Campain_Audio/<Campaign_name>/` for first `.mp3` file
- Falls back to `AUDIO_FILE` config if campaign not found
- Plays continuously while faces tracked; restarts automatically if people remain
- Requires system VLC (install via `apt install vlc` on Raspberry Pi)

---

## Config (`config.py`)

```python
DEBUG = True               # Show GUI window with face boxes and IDs
DEBUG = False              # Print stats to terminal only

Campaign_name = "default"  # Campaign folder to search for audio
IS_RASPBERRY_PI = False    # Use Picamera2 on Raspberry Pi when True
STALE_FACE_TIMEOUT = 3.0   # Remove faces not seen for > 3 seconds
AUDIO_FILE = None          # Fallback audio file if campaign not found
```

---

## Running the Script

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install system VLC (required for audio)
**Windows/macOS:** Download from https://www.videolan.org/vlc/

**Raspberry Pi:**
```bash
sudo apt update
sudo apt install vlc
```

### 3. Set up Campaign Audio (optional)
Create folder structure:
```
Campain_Audio/
└── <Campaign_name>/
    └── audio_file.mp3
```

Update `config.py`:
```python
Campaign_name = "<Campaign_name>"
```

### 4. Run with GUI (see video in real-time)
```bash
python face_detection.py
# config.DEBUG must be True
```

### 5. Run headless (print stats only)
```bash
python face_detection.py
# config.DEBUG must be False
```

### 6. Exit
- GUI mode: Press **Q**
- Terminal prints final report automatically

---

## Output Example

**Debug Mode (GUI):**
```
FPS: 30.1 | Detected: 2 | Tracked: 2
```

**Headless Mode (Terminal):**
```python
{'FPS': 29.85, 
 'faces_detected': 2, 
 'faces_tracked': 2, 
 'attention_mapping': {
   1: {'attention_s': 5.23, 'total_s': 5.23},
   2: {'attention_s': 3.87, 'total_s': 3.87}
 }}
```

**Final Report:**
```
============================================================
FINAL REPORT - Face Attention Times
============================================================
Face 1:
  Attention Time:  45.32s
  Total Time:      45.32s
  Attention %:     100.0%

Face 2:
  Attention Time:  28.15s
  Total Time:      28.15s
  Attention %:     100.0%

Campaign Duration (Start → Stop):  60.00s
============================================================
```

**CSV Output Example** (Auto-saved to `Human_proximity_Results/`):
```csv
Face_ID,Attention_Time_s,Total_Time_s
1,45.32,45.32
2,28.15,28.15

Summary
Total_People_Watched,2
Total_Attention_Time_s,73.47
Average_Attention_Time_s,36.74
Campaign_Duration_s,60.0
```

---

## How Face Tracking Works

### Example: Two people enter frame

**Frame 1:**
- Detect: Face_A at (100, 150), Face_B at (400, 200)
- No previous faces → Create ID 1 (Face_A), ID 2 (Face_B)
- Attention: 0.033s each (1 frame at 30 FPS)

**Frame 2:**
- Detect: Face at (102, 152), Face at (398, 205)
- Compute distances to previous: (100→102=2px), (400→398=2px)
- Both < threshold (50px) → Match Face 1 to detected[0], Face 2 to detected[1]
- Reuse IDs 1 and 2
- Attention: +0.033s each

**Frame 3:**
- Detect: Only 1 face at (105, 154)
- Compute distances: Face 1 = 5px (match!), Face 2 = 300px (too far)
- Match to ID 1 only
- Face 2 still tracked but not seen this frame
- Attention: ID 1 +0.033s, ID 2 no change

**Frame 10 (3+ seconds later, Face 2 not seen):**
- Cleanup removes Face 2 from tracking
- Final report shows Face 2: 9 frames at 30 FPS = 0.3 seconds attention

---

## Why This Structure is Better

| Aspect | Before | After |
|--------|--------|-------|
| **Lines** | 160 (monolithic) | 50+60+100 (modular) |
| **Readability** | Hard to follow | Clear separation of concerns |
| **Testing** | Can't test parts | Each module testable |
| **Reuse** | Can't reuse logic | Use FaceTracker in other scripts |
| **Debugging** | Where's the bug? | Check detector, tracker, or main |
| **Changes** | Touch everything | Change only relevant module |

---

## Next Steps

1. **Test locally:** `python face_detection.py`
2. **Integrate with `human_proximity.py`:** Import and use the modules
3. **Add export:** Save attention data to Excel/CSV
4. **Upgrade tracking:** Replace greedy with Hungarian algorithm or DeepSORT for more robust matching
