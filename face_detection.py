"""
Face Detection & Attention Tracking
Main script using modular face detection and tracking.

Flow:
    1. Initialize camera and face detector
    2. Loop: read frame → detect faces → track IDs → accumulate attention time
    3. Display results (GUI if DEBUG=True, print if DEBUG=False)
    4. Exit on 'Q' or camera failure
"""

import time
import os
import csv
from datetime import datetime
from pprint import pprint

from detector import FaceDetector
from tracker import FaceTracker
import config
from audio_player import AudioPlayer


def main():
    """Main execution loop."""
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cascade_path = os.path.join(script_dir,"Models", "haarcascade_frontalface_default.xml")
    
    # Initialize detector and tracker    
    detector = FaceDetector(cascade_path)
    tracker = FaceTracker()
    # Initialize audio player using campaign name (preferred) or AUDIO_FILE fallback
    audio_path = ""
    campaign_name = getattr(config, "Campaign_name", None)
    if campaign_name:
        base_dir = os.path.join(script_dir, "Campain_Audio")
        audio_path = AudioPlayer.find_campaign_audio(campaign_name, base_dir) or ""
        if not audio_path:
            expected_dir = os.path.join(base_dir, campaign_name)
            print(f"No audio file found for campaign '{campaign_name}'.")
            print(f"Create the folder and add an .mp3 file: {expected_dir}")
            print("Audio playback disabled for this run.")

    audio = AudioPlayer(audio_path)
    
    print("✓ Camera initialized")
    print("✓ Haar cascade loaded (frontal faces only)")
    print("✓ Face tracker ready")
    print("-" * 60)
    print("Press 'Q' to exit and print final report.")
    print("-" * 60)
    
    # Track campaign start and end times
    campaign_start_time = time.time()
    prev_time = 0
    
    try:
        while True:
            # Read frame from camera
            success, frame = detector.read_frame()
            if not success:
                print("Failed to read frame from camera")
                break
            
            # Get timing info
            curr_time = time.time()
            dt = curr_time - prev_time if prev_time != 0 else 0
            fps = detector.get_fps(dt)
            prev_time = curr_time
            
            # Detect faces in current frame
            faces = detector.detect_faces(frame)
            
            # Build detection objects with centroids
            curr_detections = detector.build_detections(faces)
            
            # Update tracker (match faces, assign IDs, accumulate time)
            curr_detections = tracker.update(curr_detections, curr_time, dt)

            # Audio management: play/loop while any faces are tracked
            num_tracked = len(tracker.face_tracking)
            audio.manage(num_tracked)

            # Visualization & Output
            if config.DEBUG:
                _visualize_debug(frame, curr_detections, tracker, fps)
            else:
                _print_headless(fps, faces, tracker)
    
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
    finally:
        # Calculate campaign duration
        campaign_end_time = time.time()
        campaign_duration = campaign_end_time - campaign_start_time
        
        # stop audio and release resources
        try:
            audio.stop()
        except Exception:
            pass
        detector.release()
        _print_final_report(tracker, campaign_duration)
        _save_report_csv(tracker, script_dir, campaign_duration)


def _visualize_debug(frame, curr_detections, tracker, fps):
    """
    Display video with face boxes, IDs, and attention times.
    
    Args:
        frame: Current video frame (BGR)
        curr_detections: List of detected faces with IDs
        tracker: FaceTracker object
        fps: Frames per second
    """
    import cv2
    
    # Draw face boxes and labels
    for detection in curr_detections:
        x, y, w, h = detection["rect"]
        face_id = detection["face_id"]
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw label with ID and attention time
        attn_time = tracker.face_tracking[face_id]["attention_time"]
        label = f"ID:{face_id} Attn:{attn_time:.1f}s"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Display FPS and tracked count
    num_tracked = len(tracker.face_tracking)
    cv2.putText(
        frame,
        f"FPS: {int(fps)} | Tracked: {num_tracked}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    # Show on screen
    cv2.imshow("Face Detection & Attention Tracking", frame)
    
    # Print to console
    print(f"FPS: {fps:.1f} | Detected: {len(curr_detections)} | Tracked: {num_tracked}")
    
    # Exit on 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        raise KeyboardInterrupt


def _print_headless(fps, faces, tracker):
    """
    Print statistics in headless mode (no GUI).
    
    Args:
        fps: Frames per second
        faces: List of detected faces
        tracker: FaceTracker object
    """
    summary = tracker.get_summary()
    pprint({
        "FPS": round(fps, 2),
        "faces_detected": len(faces),
        "faces_tracked": len(tracker.face_tracking),
        "attention_mapping": summary
    })


def _print_final_report(tracker, campaign_duration):
    """
    Print final report with all faces and their attention times.
    
    Args:
        tracker: FaceTracker object
        campaign_duration: Total campaign duration in seconds
    """
    print("\n" + "=" * 60)
    print("FINAL REPORT - Face Attention Times")
    print("=" * 60)
    
    # Get all faces (active + history)
    all_faces = tracker.get_all_faces_history()
    
    if not all_faces:
        print("No faces detected during recording.")
    else:
        for face_id in sorted(all_faces.keys()):
            info = all_faces[face_id]
            attn = info["attention_time"]
            total = info["total_time"]
            percent = (attn / total * 100) if total > 0 else 0
            
            print(f"Face {face_id}:")
            print(f"  Attention Time:  {attn:.2f}s")
            print(f"  Total Time:      {total:.2f}s")
            print(f"  Attention %:     {percent:.1f}%")
            print()
    
    print(f"Campaign Duration (Start → Stop):  {campaign_duration:.2f}s")
    print("=" * 60)


def _save_report_csv(tracker, script_dir, campaign_duration):
    """
    Save face attention data to CSV file.
    
    CSV format:
    - Face_ID: Unique person identifier (each ID = one person visit)
    - Attention_Time_s: Time person was looking at camera (frontal face detected)
    - Total_Time_s: Total time person was tracked on screen
    - Campaign_Duration_s: Overall time from script start to stop
    
    Includes ALL faces ever tracked (active + stale/left).
    
    Args:
        tracker: FaceTracker object
        script_dir: Script directory path
        campaign_duration: Total campaign duration in seconds (start → stop)
    """
    # Create results directory if it doesn't exist
    results_dir = os.path.join(script_dir, "Human_proximity_Results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(results_dir, f"attention_report_{timestamp}.csv")
    
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(["Face_ID", "Attention_Time_s", "Total_Time_s"])
            
            # Get ALL faces (active + stale/history)
            all_faces = tracker.get_all_faces_history()
            
            # Write data for each face (including ones that left)
            if all_faces:
                for face_id in sorted(all_faces.keys()):
                    info = all_faces[face_id]
                    attn = info["attention_time"]
                    total = info["total_time"]
                    
                    writer.writerow([face_id, round(attn, 2), round(total, 2)])
                
                # Write summary section
                writer.writerow([])  # Empty row for separation
                writer.writerow(["Summary"])
                
                # Number of unique Face_IDs = Number of people watched
                total_people = len(all_faces)
                writer.writerow(["Total_People_Watched", total_people])
                
                # Calculate statistics
                total_attention = sum(info["attention_time"] for info in all_faces.values())
                avg_attention = total_attention / total_people if total_people > 0 else 0
                
                writer.writerow(["Total_Attention_Time_s", round(total_attention, 2)])
                writer.writerow(["Average_Attention_Time_s", round(avg_attention, 2)])
                writer.writerow(["Campaign_Duration_s", round(campaign_duration, 2)])
        
        print(f"\n✓ Report saved to: {csv_filename}")
        print(f"✓ Campaign duration: {campaign_duration:.2f}s")
        print(f"✓ Total unique people tracked: {len(all_faces)}")
    
    except Exception as e:
        print(f"\n✗ Error saving CSV report: {e}")


if __name__ == "__main__":
    main()
