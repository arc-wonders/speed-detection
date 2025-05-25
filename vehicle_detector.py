"""
Vehicle detection using YOLOv8.
"""

import numpy as np
from typing import List
from data_structures import Detection

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False


class VehicleDetector:
    """Vehicle detection using YOLOv8."""
    
    # Vehicle class IDs in COCO dataset
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle', 
        5: 'bus',
        7: 'truck'
    }
    
    def __init__(self, model_path: str = "yolov8x.pt", confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package not available. Install with: pip install ultralytics")
        
        try:
            print(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            print("✓ YOLO model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load YOLO model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect vehicles in frame."""
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filter for vehicles with sufficient confidence
                    if (class_id in self.VEHICLE_CLASSES and 
                        confidence >= self.confidence_threshold):
                        
                        detection = Detection(
                            x1=float(x1), y1=float(y1),
                            x2=float(x2), y2=float(y2),
                            confidence=confidence,
                            class_id=class_id,
                            class_name=self.VEHICLE_CLASSES[class_id]
                        )
                        detections.append(detection)
        
        return detections