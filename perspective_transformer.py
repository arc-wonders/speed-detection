"""
Perspective transformation for converting image coordinates to world coordinates.
"""

import cv2
import numpy as np
from typing import List, Tuple


class PerspectiveTransformer:
    """Handles perspective transformation from image to world coordinates."""
    
    def __init__(self):
        self.matrix = None
        self.inverse_matrix = None
    
    def calibrate(self, image_points: List[Tuple[float, float]], 
                  world_points: List[Tuple[float, float]]) -> bool:
        """
        Calibrate the perspective transformation.
        
        Args:
            image_points: Points in image coordinates [(x,y), ...]
            world_points: Corresponding points in world coordinates [(x,y), ...]
            
        Returns:
            True if calibration successful, False otherwise
        """
        if len(image_points) != len(world_points) or len(image_points) < 4:
            print("Error: Need at least 4 corresponding points for calibration")
            return False
        
        try:
            img_pts = np.float32(image_points).reshape(-1, 1, 2)
            world_pts = np.float32(world_points).reshape(-1, 1, 2)
            
            self.matrix = cv2.getPerspectiveTransform(img_pts, world_pts)
            self.inverse_matrix = cv2.getPerspectiveTransform(world_pts, img_pts)
            
            print("✓ Perspective transformation calibrated successfully")
            return True
            
        except Exception as e:
            print(f"✗ Perspective transformation calibration failed: {e}")
            return False
    
    def image_to_world(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Transform points from image to world coordinates."""
        if self.matrix is None:
            raise ValueError("Transformer not calibrated")
        
        points_array = np.float32(points).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points_array, self.matrix)
        return [tuple(pt[0]) for pt in transformed]
    
    def world_to_image(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Transform points from world to image coordinates."""
        if self.inverse_matrix is None:
            raise ValueError("Transformer not calibrated")
        
        points_array = np.float32(points).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points_array, self.inverse_matrix)
        return [tuple(pt[0]) for pt in transformed]