"""
Configuration settings for the vehicle speed estimation system.
"""

# File paths
VIDEO_PATH = "data/videos/traffic_video.mp4"
MODEL_PATH = "yolov8x.pt"  # Will be downloaded automatically if not present

# Detection settings
CONFIDENCE_THRESHOLD = 0.5

# Tracking settings
MAX_DISAPPEARED_FRAMES = 30
MAX_TRACKING_DISTANCE = 100

# Speed estimation settings
MIN_TRACK_POINTS = 3
SPEED_SMOOTHING_WINDOW = 5

# Perspective transformation points
# These points define the transformation from image coordinates to world coordinates
IMAGE_POINTS = [
    (800, 410),   # Top-left corner in image
    (1125, 410),  # Top-right corner in image  
    (1920, 850),  # Bottom-right corner in image
    (0, 850)      # Bottom-left corner in image
]

WORLD_POINTS = [
    (0, 0),       # Top-left corner in world coordinates (meters)
    (32, 0),      # Top-right corner in world coordinates
    (32, 140),    # Bottom-right corner in world coordinates  
    (0, 140)      # Bottom-left corner in world coordinates
]

# Display settings
DEFAULT_SHOW_TRAJECTORIES = True
DEFAULT_SHOW_DETECTION_BOXES = True
DEFAULT_SHOW_SPEED_INFO = True

# Video processing settings
DEFAULT_FPS = 25.0  # Fallback FPS if video FPS cannot be determined

# Speed validation settings
MIN_REALISTIC_SPEED = 0    # km/h
MAX_REALISTIC_SPEED = 200  # km/h