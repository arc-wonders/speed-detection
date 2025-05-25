"""
Simple multi-object tracker for vehicle tracking.
"""

import math
import numpy as np
from typing import Dict, List, Tuple
from data_structures import Detection


class SimpleTracker:
    """Simple multi-object tracker using IoU and centroid distance."""
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100):
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def update(self, detections: List[Detection]) -> List[Tuple[int, Detection]]:
        """
        Update tracker with new detections.
        
        Returns:
            List of (track_id, detection) pairs
        """
        if not detections:
            # Mark all tracks as disappeared
            to_remove = []
            for track_id in self.tracks:
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    to_remove.append(track_id)
            
            for track_id in to_remove:
                del self.tracks[track_id]
            
            return []
        
        # If no existing tracks, create new ones
        if not self.tracks:
            results = []
            for detection in detections:
                track_id = self._create_new_track(detection)
                results.append((track_id, detection))
            return results
        
        # Calculate distances between existing tracks and new detections
        track_ids = list(self.tracks.keys())
        track_centers = [self.tracks[tid]['center'] for tid in track_ids]
        detection_centers = [det.center for det in detections]
        
        # Create distance matrix
        distances = np.zeros((len(track_ids), len(detections)))
        for i, track_center in enumerate(track_centers):
            for j, det_center in enumerate(detection_centers):
                distances[i, j] = math.sqrt(
                    (track_center[0] - det_center[0])**2 + 
                    (track_center[1] - det_center[1])**2
                )
        
        # Assign detections to tracks using simple greedy approach
        used_detections = set()
        used_tracks = set()
        results = []
        
        # Sort by distance and assign
        assignments = []
        for i in range(len(track_ids)):
            for j in range(len(detections)):
                if distances[i, j] < self.max_distance:
                    assignments.append((distances[i, j], i, j))
        
        assignments.sort()  # Sort by distance
        
        for distance, track_idx, det_idx in assignments:
            if track_idx not in used_tracks and det_idx not in used_detections:
                track_id = track_ids[track_idx]
                detection = detections[det_idx]
                
                # Update track
                self.tracks[track_id]['center'] = detection.center
                self.tracks[track_id]['disappeared'] = 0
                
                results.append((track_id, detection))
                used_tracks.add(track_idx)
                used_detections.add(det_idx)
        
        # Create new tracks for unassigned detections
        for j, detection in enumerate(detections):
            if j not in used_detections:
                track_id = self._create_new_track(detection)
                results.append((track_id, detection))
        
        # Mark unassigned tracks as disappeared
        for i, track_id in enumerate(track_ids):
            if i not in used_tracks:
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
        
        return results
    
    def _create_new_track(self, detection: Detection) -> int:
        """Create a new track."""
        track_id = self.next_id
        self.next_id += 1
        
        self.tracks[track_id] = {
            'center': detection.center,
            'disappeared': 0
        }
        
        return track_id