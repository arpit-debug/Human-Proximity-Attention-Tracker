import cv2
import time
import os
import csv
import numpy as np
from datetime import datetime
import onnxruntime as ort
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator


from scrfd import SCRFD
from audio_player import AudioPlayer
import config


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


# ------------------ Identity Memory ------------------

class FaceMemory:
    def __init__(self, similarity_thresh=0.45):
        self.db = {}
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
            return best_id

        fid = self.next_id
        self.db[fid] = emb
        self.next_id += 1
        return fid


# ------------------ Head Pose ------------------

def is_facing_camera(landmarks):
    le, re, nose, lm, rm = landmarks

    eye_mid_x = (le[0] + re[0]) / 2
    nose_offset = abs(nose[0] - eye_mid_x)
    eye_dist = abs(le[0] - re[0])
    yaw_ratio = nose_offset / eye_dist if eye_dist > 0 else 1

    eye_mid_y = (le[1] + re[1]) / 2
    mouth_mid_y = (lm[1] + rm[1]) / 2
    pitch_ratio = abs(nose[1] - eye_mid_y) / (mouth_mid_y - eye_mid_y + 1e-6)

    return yaw_ratio < 0.25 and pitch_ratio < 0.7


# ------------------ MAIN ------------------

def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))

    det_sess = ort.InferenceSession(
        os.path.join(script_dir, "Models/scrfd_500m_bnkps.onnx")
    )
    detector = SCRFD(det_sess)

    arcface = ArcFace(os.path.join(script_dir, "Models/arcface.onnx"))
    memory = FaceMemory()

    face_tracking = {}
    face_history = {}

    # ✅ Timeline storage for graph
    timeline = []
    time_axis = []

    campaign_name = getattr(config, "Campaign_name", None)
    audio_path = ""
    if campaign_name:
        base_dir = os.path.join(script_dir, "Campain_Audio")
        audio_path = AudioPlayer.find_campaign_audio(campaign_name, base_dir) or ""
    audio = AudioPlayer(audio_path)

    if config.IS_RASPBERRY_PI:
        from picamera2 import Picamera2

        picam2 = Picamera2()
        picam2.configure(
            picam2.create_preview_configuration(
                main={"format": "BGR888", "size": (640, 480)}
            )
        )
        picam2.start()
    else:
        cap = cv2.VideoCapture(0)

    campaign_start_time = time.time()
    prev_time = 0
    frame_count = 0

    print("✓ SCRFD + ArcFace tracking started")

    try:
        while True:

            if config.IS_RASPBERRY_PI:
                frame = picam2.capture_array()
                if frame is None:
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            curr_time = time.time()
            elapsed = curr_time - campaign_start_time

            dt = curr_time - prev_time if prev_time != 0 else 0
            prev_time = curr_time
            fps = 1/dt if dt > 0 else 0

            boxes, landmarks = detector.detect(frame, 0.6)

            active_ids = set()

            for (x1, y1, x2, y2), lm in zip(boxes, landmarks):

                if not is_facing_camera(lm):
                    continue

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                if frame_count % 10 == 0:
                    emb = arcface.get_embedding(face_crop)
                    fid = memory.get_id(emb)

                active_ids.add(fid)

                if fid not in face_tracking:
                    face_tracking[fid] = {
                        "attention_time": 0.0,
                        "total_time": 0.0,
                        "start_time": elapsed,   # ✅ added
                        "end_time": elapsed,     # ✅ added
                        "last_seen": curr_time
                    }

                face_tracking[fid]["last_seen"] = curr_time
                face_tracking[fid]["end_time"] = elapsed  # ✅ update end_time
                face_tracking[fid]["total_time"] += dt
                face_tracking[fid]["attention_time"] += dt

                if config.DEBUG:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    label = f"ID:{fid} Attn:{face_tracking[fid]['attention_time']:.1f}s"
                    cv2.putText(frame,label,(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

            stale = [fid for fid in face_tracking
                     if curr_time - face_tracking[fid]["last_seen"]
                     > config.STALE_FACE_TIMEOUT]

            for fid in stale:
                face_history[fid] = face_tracking[fid]
                del face_tracking[fid]

            audio.manage(len(face_tracking))

            # ✅ Store graph data
            timeline.append(len(face_tracking))
            time_axis.append(elapsed)

            if config.DEBUG:
                cv2.putText(frame,f"FPS:{int(fps)}",
                            (10,30),cv2.FONT_HERSHEY_SIMPLEX,
                            1,(0,255,0),2)
                cv2.imshow("SCRFD ArcFace Attention",frame)
                if cv2.waitKey(1)&0xFF==ord('q'):
                    raise KeyboardInterrupt
                print(f"FPS: {fps:.1f} | Detected: {len(boxes)} | Tracked: {len(face_tracking)}")

            frame_count += 1

    except KeyboardInterrupt:
        print("\nStopped")

    finally:
        if config.IS_RASPBERRY_PI:
            picam2.stop()
        else:
            cap.release()
            cv2.destroyAllWindows()
        audio.stop()

        campaign_duration = time.time() - campaign_start_time
        save_report(face_tracking, face_history,
                    script_dir, campaign_duration)

        plot_graph(time_axis, timeline, script_dir)  # ✅ added


# ------------------ CSV ------------------

def save_report(face_tracking, face_history,
                script_dir, campaign_duration):

    all_faces = {}
    all_faces.update(face_history)
    all_faces.update(face_tracking)

    results_dir = os.path.join(script_dir,"Human_proximity_Results")
    os.makedirs(results_dir,exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir,
                            f"attention_report_{timestamp}.csv")

    total_attention = 0.0

    for fid in all_faces:
        total_attention += all_faces[fid]["attention_time"]

    total_people = len(all_faces)
    avg_attention = (total_attention / total_people) if total_people > 0 else 0

    with open(csv_path,"w",newline="") as f:
        writer = csv.writer(f)

        # ---- Detailed Per Face ----
        writer.writerow(["Face_ID","Attention_Time_s","start_time","end_time","Total_Time_s"])

        for fid in sorted(all_faces.keys()):
            data = all_faces[fid]
            writer.writerow([
                fid,
                round(data["attention_time"],2),
                round(data["start_time"],2),
                round(data["end_time"],2),
                round(data["total_time"],2)
            ])

        # ---- Summary Section ----
        writer.writerow([])
        writer.writerow(["Summary"])
        writer.writerow(["Total_People_Watched", total_people])
        writer.writerow(["Total_Attention_Time_s", round(total_attention,2)])
        writer.writerow(["Average_Attention_Time_s", round(avg_attention,2)])
        writer.writerow(["Campaign_Duration_s", round(campaign_duration,2)])

    # ---- Print Summary in Terminal ----
    print("\n===== Campaign Summary =====")
    print(f"Total_People_Watched     : {total_people}")
    print(f"Total_Attention_Time_s   : {round(total_attention,2)}")
    print(f"Average_Attention_Time_s : {round(avg_attention,2)}")
    print(f"Campaign_Duration_s      : {round(campaign_duration,2)}")

    print(f"\n✓ Report saved: {csv_path}")



# ------------------ GRAPH ------------------

def plot_graph(time_axis, timeline, script_dir):

    results_dir = os.path.join(script_dir,"Human_proximity_Results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure()
    plt.plot(time_axis, timeline)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Number of People")
    plt.title("People vs Time")
    plt.grid()
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    graph_path = os.path.join(results_dir,
                               f"people_vs_time_{timestamp}.png")

    plt.savefig(graph_path)
    plt.close()

    print(f"✓ Graph saved: {graph_path}")


if __name__ == "__main__":
    main()
