"""
Speed estimation using tracking and perspective transformation.
"""

import math
import numpy as np
from typing import Dict, Optional, Any
from data_structures import Detection, TrackPoint, VehicleTrack
from perspective_transformer import PerspectiveTransformer

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class SpeedEstimator:
    """Estimates vehicle speeds using tracking and perspective transformation."""
    
    def __init__(self, transformer: PerspectiveTransformer, 
                 min_track_points: int = 3, speed_smoothing_window: int = 5):
        self.transformer = transformer
        self.vehicle_tracks: Dict[int, VehicleTrack] = {}
        self.min_track_points = min_track_points
        self.speed_smoothing_window = speed_smoothing_window
        
        # Statistics
        self.total_vehicles = 0
        self.speed_measurements = []
    
    def update(self, track_id: int, detection: Detection, timestamp: float) -> Optional[float]:
        """
        Update tracking for a vehicle and estimate speed.
        
        Args:
            track_id: Unique identifier for the track
            detection: Vehicle detection
            timestamp: Current timestamp in seconds
            
        Returns:
            Estimated speed in km/h, or None if not enough data
        """
        # Get world coordinates
        world_pos = self.transformer.image_to_world([detection.center])[0]
        
        # Create track point
        track_point = TrackPoint(
            timestamp=timestamp,
            image_pos=detection.center,
            world_pos=world_pos
        )
        
        # Initialize or update track
        if track_id not in self.vehicle_tracks:
            self.vehicle_tracks[track_id] = VehicleTrack(track_id=track_id)
            self.total_vehicles += 1
        
        track = self.vehicle_tracks[track_id]
        track.add_point(track_point)
        
        # Calculate speed if we have enough points
        if len(track.points) >= self.min_track_points:
            speed = self._calculate_speed(track)
            if speed is not None:
                track.add_speed(speed)
                self.speed_measurements.append(speed)
                return track.get_average_speed()
        
        return None
    
    def _calculate_speed(self, track: VehicleTrack) -> Optional[float]:
        """Calculate instantaneous speed for a track."""
        if len(track.points) < 2:
            return None
        
        # Use recent points for speed calculation
        recent_points = list(track.points)[-self.min_track_points:]
        
        if len(recent_points) < 2:
            return None
        
        # Calculate speed using linear regression or simple average
        total_distance = 0
        total_time = 0
        
        for i in range(1, len(recent_points)):
            prev_point = recent_points[i-1]
            curr_point = recent_points[i]
            
            # Distance in meters
            dx = curr_point.world_pos[0] - prev_point.world_pos[0]
            dy = curr_point.world_pos[1] - prev_point.world_pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Time in seconds
            time_diff = curr_point.timestamp - prev_point.timestamp
            
            if time_diff > 0:
                total_distance += distance
                total_time += time_diff
        
        if total_time > 0:
            # Speed in m/s, convert to km/h
            speed_ms = total_distance / total_time
            speed_kmh = speed_ms * 3.6
            
            # Filter out unrealistic speeds
            if 0 < speed_kmh < 200:  # Reasonable speed range
                return speed_kmh
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get speed estimation statistics."""
        if not self.speed_measurements:
            return {
                'total_vehicles': self.total_vehicles,
                'measurements': 0,
                'average_speed': None,
                'max_speed': None,
                'min_speed': None,
                'std_speed': None
            }
        
        speeds = np.array(self.speed_measurements)
        return {
            'total_vehicles': self.total_vehicles,
            'measurements': len(speeds),
            'average_speed': float(np.mean(speeds)),
            'max_speed': float(np.max(speeds)),
            'min_speed': float(np.min(speeds)),
            'std_speed': float(np.std(speeds))
        }
    
    def cleanup_old_tracks(self, current_time: float, max_age: float = 10.0):
        """Remove old tracks that haven't been updated recently."""
        to_remove = []
        for track_id, track in self.vehicle_tracks.items():
            if current_time - track.last_update > max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.vehicle_tracks[track_id]