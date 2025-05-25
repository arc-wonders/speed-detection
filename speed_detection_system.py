"""
Complete vehicle speed detection system that integrates all components.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from vehicle_detector import VehicleDetector
from tracker import SimpleTracker
from perspective_transformer import PerspectiveTransformer
from speed_estimator import SpeedEstimator


class SpeedDetectionSystem:
    """Complete vehicle speed detection system."""
    
    def __init__(self, model_path: str = "yolov8x.pt", confidence_threshold: float = 0.5):
        # Initialize components
        self.detector = VehicleDetector(model_path, confidence_threshold)
        self.tracker = SimpleTracker()
        self.transformer = PerspectiveTransformer()
        self.speed_estimator = None  # Will be initialized after calibration
        
        # Display settings
        self.show_trajectories = True
        self.show_detection_boxes = True
        self.show_speed_info = True
        
    def calibrate_perspective(self, image_points: List[Tuple[float, float]], 
                            world_points: List[Tuple[float, float]]) -> bool:
        """Calibrate perspective transformation."""
        success = self.transformer.calibrate(image_points, world_points)
        if success:
            self.speed_estimator = SpeedEstimator(self.transformer)
        return success
    
    def process_frame(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        """Process a single frame and return annotated result."""
        if self.speed_estimator is None:
            raise ValueError("System not calibrated. Call calibrate_perspective() first.")
        
        # Detect vehicles
        detections = self.detector.detect(frame)
        
        # Update tracker
        tracked_objects = self.tracker.update(detections)
        
        # Update speed estimation and draw annotations
        annotated_frame = frame.copy()
        
        for track_id, detection in tracked_objects:
            # Update speed estimation
            speed = self.speed_estimator.update(track_id, detection, timestamp)
            
            # Draw detection box
            if self.show_detection_boxes:
                color = self._get_speed_color(speed)
                cv2.rectangle(annotated_frame, 
                            (int(detection.x1), int(detection.y1)),
                            (int(detection.x2), int(detection.y2)),
                            color, 2)
            
            # Draw speed information
            if self.show_speed_info:
                speed_text = f"ID:{track_id}"
                if speed is not None:
                    speed_text += f" {speed:.1f} km/h"
                else:
                    speed_text += " Calculating..."
                
                # Text background
                text_size = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_frame,
                            (int(detection.x1), int(detection.y1) - 30),
                            (int(detection.x1) + text_size[0] + 10, int(detection.y1) - 5),
                            (0, 0, 0), -1)
                
                color = self._get_speed_color(speed)
                cv2.putText(annotated_frame, speed_text,
                          (int(detection.x1) + 5, int(detection.y1) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw trajectory
            if self.show_trajectories and track_id in self.speed_estimator.vehicle_tracks:
                track = self.speed_estimator.vehicle_tracks[track_id]
                if len(track.points) > 1:
                    points = [pt.image_pos for pt in track.points]
                    for i in range(1, len(points)):
                        cv2.line(annotated_frame,
                               (int(points[i-1][0]), int(points[i-1][1])),
                               (int(points[i][0]), int(points[i][1])),
                               (255, 255, 0), 2)
        
        # Add frame statistics
        stats = self.speed_estimator.get_statistics()
        info_lines = [
            f"Vehicles: {len(tracked_objects)} active, {stats['total_vehicles']} total",
            f"Speed measurements: {stats['measurements']}"
        ]
        
        if stats['average_speed'] is not None:
            info_lines.append(f"Avg speed: {stats['average_speed']:.1f} km/h")
        
        for i, line in enumerate(info_lines):
            cv2.putText(annotated_frame, line, (10, 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Cleanup old tracks periodically
        if int(timestamp) % 10 == 0:  # Every 10 seconds
            self.speed_estimator.cleanup_old_tracks(timestamp)
        
        return annotated_frame
    
    def _get_speed_color(self, speed: Optional[float]) -> Tuple[int, int, int]:
        """Get color based on speed value."""
        if speed is None:
            return (255, 255, 0)  # Yellow for calculating
        elif speed < 30:
            return (0, 255, 0)    # Green for slow
        elif speed < 60:
            return (0, 255, 255)  # Yellow for medium
        elif speed < 80:
            return (0, 165, 255)  # Orange for fast
        else:
            return (0, 0, 255)    # Red for very fast