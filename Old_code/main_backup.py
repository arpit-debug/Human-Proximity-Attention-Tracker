import cv2
import time
import numpy as np
import onnxruntime as ort
from scrfd import SCRFD

# ------------------ ArcFace ------------------

class ArcFace:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, face):
        face = cv2.resize(face, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32)
        face = (face / 255.0 - 0.5) / 0.5
        return np.expand_dims(face, axis=0)

    def get_embedding(self, face):
        blob = self.preprocess(face)
        emb = self.session.run(None, {self.input_name: blob})[0]
        emb = emb.flatten()
        emb = emb / np.linalg.norm(emb)
        return emb

# ------------------ Face Memory ------------------

class FaceMemory:
    def __init__(self, similarity_thresh=0.45):
        self.db = {}          # id -> embedding
        self.last_seen = {}  # id -> timestamp
        self.next_id = 1
        self.thresh = similarity_thresh

    def cosine(self, a, b):
        return np.dot(a, b)

    def get_id(self, emb):
        best_id = None
        best_sim = 0

        for fid, stored_emb in self.db.items():
            sim = self.cosine(emb, stored_emb)
            if sim > best_sim:
                best_sim = sim
                best_id = fid

        if best_sim > self.thresh:
            self.last_seen[best_id] = time.time()
            return best_id

        # new face
        fid = self.next_id
        self.db[fid] = emb
        self.last_seen[fid] = time.time()
        self.next_id += 1
        return fid

# ------------------ Head Pose Filter ------------------

def is_facing_camera(landmarks):
    le, re, nose, lm, rm = landmarks

    # horizontal symmetry (yaw)
    eye_mid_x = (le[0] + re[0]) / 2
    nose_offset = abs(nose[0] - eye_mid_x)

    eye_dist = abs(le[0] - re[0])
    yaw_ratio = nose_offset / eye_dist

    # vertical symmetry (pitch)
    eye_mid_y = (le[1] + re[1]) / 2
    mouth_mid_y = (lm[1] + rm[1]) / 2
    pitch_ratio = abs(nose[1] - eye_mid_y) / (mouth_mid_y - eye_mid_y)

    return yaw_ratio < 0.25 and pitch_ratio < 0.7 and eye_dist < 40 #For more strict frontal face, use pitch_ratio < 0.7 and eye_dist < 30


det_sess = ort.InferenceSession("Models/scrfd_500m_bnkps.onnx")
detector = SCRFD(det_sess)

arcface = ArcFace("Models/arcface.onnx")
memory = FaceMemory()

cap = cv2.VideoCapture(0)

prev_time = time.time()

# ------------------ Main Loop ------------------
frame_count = 0
face_id = None
while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes, landmarks = detector.detect(frame, 0.6)

    for (x1, y1, x2, y2), lm in zip(boxes, landmarks):

        if not is_facing_camera(lm):
            continue

        # Run ArcFace only every 10 frames
        if frame_count % 10 == 0 or face_id is None:
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            emb = arcface.get_embedding(face_crop)
            face_id = memory.get_id(emb)
            

        # Draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        for (lx, ly) in lm.astype(int):
            cv2.circle(frame, (lx, ly), 2, (0,0,255), -1)
        cv2.putText(frame, f"ID: {face_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2)

    # FPS
    curr = time.time()
    fps = 1 / (curr - prev_time)
    prev_time = curr

    cv2.putText(frame, f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("SCRFD + ArcFace Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
