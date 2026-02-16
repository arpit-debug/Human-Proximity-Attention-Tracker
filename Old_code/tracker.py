"""
Face Tracking Module
Handles face ID assignment, remapping, and attention span tracking.
"""

import config


class FaceTracker:
    """
    Tracks faces across frames using greedy nearest-neighbor assignment.
    
    Attributes:
        face_tracking (dict): Maps face_id -> {center, attention_time, total_time, last_seen}
        face_history (dict): Maps face_id -> {center, attention_time, total_time} (all faces ever seen)
        next_face_id (int): Counter for generating unique face IDs
    """
    
    def __init__(self):
        self.face_tracking = {}      # Currently active faces
        self.face_history = {}       # All faces ever tracked (including stale ones)
        self.next_face_id = 1
    
    def update(self, curr_detections, curr_time, dt):
        """
        Update face tracking with current detections.
        
        Args:
            curr_detections (list): List of dicts with {rect, center, face_id}
            curr_time (float): Current timestamp
            dt (float): Time delta since last frame
        
        Returns:
            list: Updated curr_detections with assigned face_ids
        """
        # Step 1: Compute all pairwise distances
        pairs = self._compute_distances(curr_detections)
        
        # Step 2: Greedy nearest-neighbor assignment
        assignments = self._greedy_assignment(curr_detections, pairs)
        
        # Step 3: Assign IDs (reuse or create new)
        self._assign_ids(curr_detections, assignments, curr_time)
        
        # Step 4: Update tracking info (position, time)
        self._update_tracking(curr_detections, curr_time, dt)
        
        # Step 5: Remove stale faces
        self._cleanup_stale(curr_time)
        
        return curr_detections
    
    def _compute_distances(self, curr_detections):
        """
        Compute Euclidean distances between all current and previous face centers.
        
        Returns:
            list: [(distance, curr_idx, prev_id), ...]
        """
        pairs = []
        for i, curr in enumerate(curr_detections):
            cx, cy = curr["center"]
            for fid in self.face_tracking:
                px, py = self.face_tracking[fid]["center"]
                dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
                pairs.append((dist, i, fid))
        return pairs
    
    def _greedy_assignment(self, curr_detections, pairs):
        """
        Greedy nearest-neighbor assignment: one-to-one matching.
        
        Ensures each previous face ID is used at most once per frame,
        and each current detection matches at most one previous face.
        
        Returns:
            dict: {curr_idx -> face_id}
        """
        pairs.sort(key=lambda t: t[0])  # Sort by distance (smallest first)
        assigned_curr = set()
        assigned_prev = set()
        assignments = {}
        
        for dist, i, fid in pairs:
            # Skip if either face already assigned
            if i in assigned_curr or fid in assigned_prev:
                continue
            
            # Threshold: allow movement up to 50px or 60% of face size
            _, _, w, h = curr_detections[i]["rect"]
            threshold = max(50, max(w, h) * 0.6)
            
            if dist <= threshold:
                assignments[i] = fid
                assigned_curr.add(i)
                assigned_prev.add(fid)
        
        return assignments
    
    def _assign_ids(self, curr_detections, assignments, curr_time):
        """
        Assign face IDs: reuse matched IDs, create new IDs for unmatched faces.
        """
        for i, curr in enumerate(curr_detections):
            x, y, w, h = curr["rect"]
            cx, cy = curr["center"]
            
            if i in assignments:
                # Reuse matched ID
                face_id = assignments[i]
            else:
                # Create new ID
                face_id = self.next_face_id
                self.next_face_id += 1
                self.face_tracking[face_id] = {
                    "center": (cx, cy),
                    "attention_time": 0.0,
                    "total_time": 0.0,
                    "last_seen": curr_time
                }
            
            curr["face_id"] = face_id
    
    def _update_tracking(self, curr_detections, curr_time, dt):
        """
        Update tracking info: position, total_time, attention_time.
        """
        for curr in curr_detections:
            face_id = curr["face_id"]
            cx, cy = curr["center"]
            
            # self.face_tracking is getting modified here
            tracked = self.face_tracking[face_id]
            tracked["center"] = (cx, cy)
            tracked["last_seen"] = curr_time
            tracked["total_time"] += dt
            
            # Since Haar cascade only detects frontal faces, detected = looking at camera
            tracked["attention_time"] += dt
    
    def _cleanup_stale(self, curr_time):
        """
        Remove faces not seen for more than `config.STALE_FACE_TIMEOUT` seconds.
        Also saves them to face_history before deletion.
        """
        timeout = config.STALE_FACE_TIMEOUT
        stale = [fid for fid in self.face_tracking 
                 if curr_time - self.face_tracking[fid].get("last_seen", 0) > timeout]
        for fid in stale:
            # Save to history before deletion
            self.face_history[fid] = self.face_tracking[fid].copy()
            del self.face_tracking[fid]
    
    def get_summary(self):
        """
        Get a summary of all tracked faces and their attention times.
        
        Returns:
            dict: {face_id -> {attention_s, total_s}, ...}
        """
        return {
            fid: {
                "attention_s": round(self.face_tracking[fid]["attention_time"], 2),
                "total_s": round(self.face_tracking[fid]["total_time"], 2)
            }
            for fid in self.face_tracking
        }
    
    def get_all_faces_history(self):
        """
        Get ALL faces ever tracked (active + stale).
        Used for final report and CSV export.
        
        Returns:
            dict: {face_id -> {attention_time, total_time}, ...}
        """
        # Combine active faces and history
        all_faces = {}
        
        # Add currently active faces
        for fid in self.face_tracking:
            all_faces[fid] = {
                "attention_time": self.face_tracking[fid]["attention_time"],
                "total_time": self.face_tracking[fid]["total_time"]
            }
        
        # Add stale faces from history
        for fid in self.face_history:
            all_faces[fid] = {
                "attention_time": self.face_history[fid]["attention_time"],
                "total_time": self.face_history[fid]["total_time"]
            }
        
        return all_faces
