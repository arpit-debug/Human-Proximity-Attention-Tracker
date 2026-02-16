Here is the **fully updated module documentation**, aligned with your new architecture:

* ❌ No more Haar cascade
* ❌ No more centroid greedy tracker
* ❌ No YOLO
* ✅ SCRFD (ONNX) detector
* ✅ ArcFace embeddings for identity
* ✅ Cosine similarity matching
* ✅ CSV + Summary + Graph output
* ✅ Picamera2 backend switch
* ✅ Attention tracking based on head pose

---

# Face Detection & Attention Tracking System (SCRFD + ArcFace)

## Updated Project Structure

```
Human_proximity_sensor/
├── main.py                         ← Main orchestrator (SCRFD + ArcFace)
├── scrfd.py                        ← SCRFD face detection wrapper
├── audio_player.py                 ← Audio playback manager
├── config.py                       ← Configuration flags
├── Models/
│   ├── scrfd_500m_bnkps.onnx
│   └── arcface.onnx
├── Campain_Audio/
├── Human_proximity_Results/
├── requirements.txt
└── README.md
```

---

# Architecture Overview

This system performs:

1. Face Detection → SCRFD (ONNX)
2. Face Identification → ArcFace embeddings
3. Head Pose Filtering → Only count frontal faces
4. Identity Matching → Cosine similarity
5. Attention Time Tracking
6. CSV + Summary Export
7. People vs Time Graph Generation

---

# Module Breakdown

---

## 1️⃣ `main.py` (Core Pipeline)

### Purpose

Controls entire attention analytics pipeline.

### Responsibilities

| Component        | Purpose                 |
| ---------------- | ----------------------- |
| Camera Backend   | cv2 or Picamera2        |
| SCRFD            | Face detection          |
| ArcFace          | Embedding extraction    |
| FaceMemory       | Identity assignment     |
| Head Pose Filter | Frontal face validation |
| Timeline Logger  | Graph data collection   |
| CSV Export       | Save analytics          |
| Graph Export     | Save PNG plot           |

---

### Runtime Flow

```
Initialize Camera (cv2 or Picamera2)
        ↓
Load SCRFD ONNX
        ↓
Load ArcFace ONNX
        ↓
Loop:
    Capture Frame
        ↓
    Detect Faces (SCRFD)
        ↓
    Filter: is_facing_camera()
        ↓
    Extract Embedding (ArcFace)
        ↓
    Identity Matching (Cosine Similarity)
        ↓
    Update Tracking + Attention Time
        ↓
    Cleanup Stale Faces
        ↓
    Store Timeline Data
        ↓
    Display / Print
        ↓
Exit → Save CSV → Save Graph
```

---

# Identity System (Embedding-Based)

### Class: `FaceMemory`

Instead of centroid tracking, we use embedding similarity.

### Matching Logic

```python
sim = np.dot(embedding, stored_embedding)
```

If:

```
similarity > threshold (default 0.45)
```

→ Reuse existing ID
Else → Create new ID

### Benefits

* Re-identification after disappearance
* More stable ID assignment
* Person-level analytics
* Scalable to vector database (future FAISS upgrade)

---

# Head Pose Filtering

### Function: `is_facing_camera(landmarks)`

Uses 5 keypoints:

* Left eye
* Right eye
* Nose
* Left mouth
* Right mouth

Computes:

* Yaw ratio
* Pitch ratio

Only if:

```
yaw < 0.25 AND pitch < 0.7
```

→ Count as attention

This prevents side profiles from being counted.

---

# Camera Backend

Controlled via:

```python
IS_RASPBERRY_PI = True / False
```

### Desktop

```python
cv2.VideoCapture(0)
```

### Raspberry Pi

```python
Picamera2.capture_array()
```

Advantages of Picamera2:

* Lower latency
* Stable FPS
* Native Pi camera stack
* No extra conversion overhead

---

# Audio System

## `audio_player.py`

### Responsibilities

* Auto-search campaign folder:

```
Campain_Audio/<Campaign_name>/
```

* Play first MP3 found
* Stop when no faces present
* Restart if people reappear

Requires:

* System VLC installed

---

# Configuration (`config.py`)

```python
DEBUG = True
STALE_FACE_TIMEOUT = 3.0
IS_RASPBERRY_PI = False

Campaign_name = "default"
AUDIO_FILE = None
```

---

# Output System

Each session produces:

### 1️⃣ CSV Report

Saved to:

```
Human_proximity_Results/attention_report_<timestamp>.csv
```

### CSV Format

```csv
Face_ID,Attention_Time_s,start_time,end_time,Total_Time_s
1,8.0,2.1,10.1,8.0
2,5.5,12.2,18.4,6.2

Summary
Total_People_Watched,2
Total_Attention_Time_s,13.5
Average_Attention_Time_s,6.75
Campaign_Duration_s,45.3
```

---

### Metrics

| Metric               | Meaning                         |
| -------------------- | ------------------------------- |
| Face_ID              | Unique embedding-based identity |
| Attention_Time_s     | Time looking at camera          |
| Total_Time_s         | Time present in frame           |
| Total_People_Watched | Unique visitors                 |
| Campaign_Duration_s  | Script runtime                  |

---

### 2️⃣ People vs Time Graph

Saved to:

```
Human_proximity_Results/people_vs_time_<timestamp>.png
```

Graph details:

* X-axis → Time (seconds)
* Y-axis → Number of people
* Integer-only Y-axis enforced:

```python
MaxNLocator(integer=True)
```

---

# Performance Optimization Notes

For Raspberry Pi:

* Use 640x480 resolution
* Disable GUI (`DEBUG = False`)
* Use Picamera2 backend
* Consider embedding every N frames

Current complexity:

```
O(N) per face for ID matching
```

Can be upgraded to FAISS for:

```
O(log N) similarity search
```

---


# Example Final Summary (Terminal)

```
===== Campaign Summary =====
Total_People_Watched     : 3
Total_Attention_Time_s   : 16.7
Average_Attention_Time_s : 5.57
Campaign_Duration_s      : 45.3
```

---


