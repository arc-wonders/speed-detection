"""
Data structures for the vehicle speed estimation system.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Detection:
    """Represents a vehicle detection."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str = ""
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of detection."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        """Get area of detection."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class TrackPoint:
    """Represents a point in a vehicle's trajectory."""
    timestamp: float
    image_pos: Tuple[float, float]
    world_pos: Tuple[float, float]
    speed_kmh: Optional[float] = None


@dataclass
class VehicleTrack:
    """Represents a tracked vehicle with its trajectory and speed history."""
    track_id: int
    points: deque = field(default_factory=lambda: deque(maxlen=30))
    speeds: deque = field(default_factory=lambda: deque(maxlen=10))
    last_update: float = 0
    total_distance: float = 0
    frames_tracked: int = 0
    
    def add_point(self, point: TrackPoint):
        """Add a new tracking point."""
        self.points.append(point)
        self.last_update = point.timestamp
        self.frames_tracked += 1
        
        # Calculate distance from previous point
        if len(self.points) >= 2:
            prev_world = self.points[-2].world_pos
            curr_world = point.world_pos
            distance = math.sqrt(
                (curr_world[0] - prev_world[0])**2 + 
                (curr_world[1] - prev_world[1])**2
            )
            self.total_distance += distance
    
    def add_speed(self, speed: float):
        """Add a speed measurement."""
        if speed > 0:  # Only add valid speeds
            self.speeds.append(speed)
    
    def get_average_speed(self) -> Optional[float]:
        """Get smoothed average speed."""
        if not self.speeds:
            return None
        
        # Remove outliers (speeds that are too different from median)
        speeds_list = list(self.speeds)
        if len(speeds_list) >= 3:
            median_speed = np.median(speeds_list)
            filtered_speeds = [s for s in speeds_list 
                             if abs(s - median_speed) < median_speed * 0.5]
            if filtered_speeds:
                return np.mean(filtered_speeds)
        
        return np.mean(speeds_list)