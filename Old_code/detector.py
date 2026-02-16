"""
Face Detection Module
Handles camera initialization, frame capture, and face detection.
Supports Picamera2 on Raspberry Pi when `config.IS_RASPBERRY_PI` is True.
"""
import cv2
import os
import config

try:
    if config.IS_RASPBERRY_PI:
        from picamera2 import Picamera2
    else:
        Picamera2 = None
except Exception:
    Picamera2 = None


class FaceDetector:
    """
    Detects frontal faces in video frames using Haar Cascade.
    
    Attributes:
        face_cascade: OpenCV Haar cascade classifier
        cap: Video capture object (webcam)
    """
    
    def __init__(self, cascade_path):
        """
        Initialize the face detector.
        
        Args:
            cascade_path (str): Path to Haar cascade XML file
        
        Raises:
            RuntimeError: If cascade file cannot be loaded or camera cannot be opened
        """
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Could not load cascade from {cascade_path}")
        
        # If configured for Raspberry Pi and Picamera2 is available, use it for capture
        self.use_picamera = False
        self.picam2 = None
        if Picamera2 is not None:
            try:
                self.picam2 = Picamera2()
                config_preview = self.picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)})
                self.picam2.configure(config_preview)
                self.picam2.start()
                self.use_picamera = True
            except Exception:
                # fallback to cv2 VideoCapture
                self.picam2 = None

        if not self.use_picamera:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open video device 0. Try a different camera index.")
    
    def read_frame(self):
        """
        Read a frame from the camera.
        
        Returns:
            tuple: (success, frame) where frame is BGR image array or None on failure
        """
        if self.use_picamera and self.picam2 is not None:
            try:
                frame = self.picam2.capture_array()
                # Picamera2 returns RGB-like arrays; convert to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return True, frame
            except Exception:
                return False, None
        else:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return False, None
            return True, frame
    
    def detect_faces(self, frame):
        """
        Detect frontal faces in a frame.
        
        Note: This detector ONLY finds frontal faces (looking at camera).
        Side profiles or angles are not detected.
        
        Args:
            frame: BGR image (from cv2.read())
        
        Returns:
            list: [(x, y, w, h), ...] bounding boxes of detected faces
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,      # Image pyramid scale
            minNeighbors=5,       # Minimum neighbors for detection
            minSize=(60, 60)      # Minimum face size
        )
        return faces
    
    def get_fps(self, dt):
        """
        Calculate frames per second.
        
        Args:
            dt (float): Time delta since last frame (in seconds)
        
        Returns:
            float: FPS (returns 0 if dt is 0)
        """
        return 1 / dt if dt > 0 else 0
    
    def build_detections(self, faces):
        """
        Build detection dictionaries with centroid information.
        
        Args:
            faces: Array of (x, y, w, h) tuples from face_cascade.detectMultiScale()
        
        Returns:
            list: [{rect, center, face_id}, ...] with centroids calculated
        """
        detections = []
        for (x, y, w, h) in faces:
            cx = x + w / 2
            cy = y + h / 2
            detections.append({
                "rect": (x, y, w, h),
                "center": (cx, cy),
                "face_id": None
            })
        return detections
    
    def release(self):
        """Release camera and close windows."""
        try:
            if self.use_picamera and self.picam2 is not None:
                try:
                    self.picam2.stop()
                except Exception:
                    pass
            else:
                try:
                    self.cap.release()
                except Exception:
                    pass
        except Exception:
            pass
        cv2.destroyAllWindows()
