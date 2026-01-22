import cv2
import time
import numpy as np
from ultralytics import YOLO  # Using YOLOv8 for human detection
import pandas as pd

# Load models
face_cascade = cv2.CascadeClassifier(r"D:\Project\Human_proximity_sensor\haarcascade_frontalface_default.xml")
yolo_model = YOLO(r"D:\Project\Human_proximity_sensor\yolov8n.pt")  # Load YOLOv8 model

# Initialize tracking variables
trigger_count = 0
face_look_time = {}
human_timestamps = []
face_tracking = {}
face_look_frames = {}
face_first_instance = {}

# Ask user for input source
input_source = r"D:\Project\Human_proximity_sensor\mall_road.mp4"
# input_source = input("Enter video file path or type 'cam' for webcam: ")
if input_source.lower() == "cam":
    cap = cv2.VideoCapture(0)  # Use webcam
    video_name = f"camera_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
else:
    cap = cv2.VideoCapture(input_source)  # Load video file
    video_name = input_source.split("/")[-1].replace(".mp4", "") + "_processed.mp4"

fps = cap.get(cv2.CAP_PROP_FPS) 
out = None  # Initialize VideoWriter as None

start_time = time.time()
Frame_count = 0
face_records = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: # or (time.time() - start_time > 5):  # Stop after 5 seconds
        break

    # Resize while maintaining aspect ratio
    height, width, _ = frame.shape
    aspect_ratio = width / height
    if height > width:
        new_height = 480        
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = 480
        new_height = int(new_width * (1 /aspect_ratio))

    # Initialize VideoWriter only on first valid frame
    if out is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_name, fourcc, fps, (new_width, new_height))

    frame = cv2.resize(frame, (new_width, new_height))
    results = yolo_model(frame)  # Run YOLO detection
    
    for result in results:
        print("Frame count",Frame_count)
        Frame_count += 1
        Time_in_video = round(Frame_count/fps,2)
        # import pdb
        # pdb.set_trace()
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            if int(cls) == 0:  # Class 0 is 'person'
                center_x = (x1 + x2) / 2
                distance = (width / 2) / (x2 - x1) * 2  # Approximate distance
                person_box_height = int(y2 - y1)
                person_box_width = int(x2 - x1)
                print(f"Time: {(Time_in_video)} s","person box height",person_box_height,"  Width",person_box_width)

                
                if person_box_height > 200 or person_box_width > 70:  # If within 2 meters
                # if True:  # If within 2 meters
                    trigger_count += 1
                    human_timestamps.append(time.time())
                    cv2.putText(frame, f"ALERT: Human in Proximity! {Time_in_video}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Draw bounding box on human
                    
                    # Crop the detected human region for face detection
                    human_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    gray_crop = cv2.cvtColor(human_crop, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray_crop, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    
                    detected_faces = []
                    for (x, y, w, h) in faces:
                        x += int(x1)  # Adjust coordinates to original frame
                        y += int(y1)
                        face_id = None
                        for prev_face in face_tracking:
                            prev_x, prev_y = face_tracking[prev_face]
                            if abs(prev_x - x) < 50 and abs(prev_y - y) < 50:
                                face_id = prev_face  # Re-map face
                                break
                        
                        if face_id is None:
                            face_id = f"face_{len(face_tracking)}"
                            face_tracking[face_id] = (x, y)
                        
                        if face_id not in face_look_frames:
                            face_look_frames[face_id] = 0
                        face_look_frames[face_id] += 1  # Count frames
                        if face_id not in face_first_instance:    
                            face_first_instance[face_id] = Time_in_video
                        
                        detected_faces.append((face_id, x, y, w, h))
                    
                    # Update tracked face positions
                    for face_id, x, y, w, h in detected_faces:
                        face_tracking[face_id] = (x, y)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw bounding box on face
                        look_time = face_look_frames[face_id] / fps  # Convert frames to seconds
                        cv2.putText(frame, f"Looking: {look_time:.1f}s", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        face_records.append([face_id, face_first_instance[face_id],face_first_instance[face_id] + look_time, look_time])
    out.write(frame)
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Generate final report
print(f"Total triggers: {trigger_count}")
print(f"Total faces detected: {len(face_look_frames.keys())}")
for face, frames in face_look_frames.items():
    duration = frames / fps  # Convert frames to seconds
    print(f"{face} looked at the camera for {duration:.1f} seconds")

# Save to Excel
excel_name = video_name.replace(".mp4", ".xlsx")
df = pd.DataFrame(face_records, columns=["face", "Start time (s)", "End time(s)", "look towards camera (s)"])

# Keep only the last instance of each face
df = df.groupby("face").last().reset_index()

df.to_excel(excel_name, index=False)
print(f"Report saved as {excel_name}")
