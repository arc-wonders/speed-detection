# Vehicle Speed Estimation System

A comprehensive computer vision system for detecting and tracking vehicles in video footage and estimating their speeds using YOLOv8 detection, multi-object tracking, and perspective transformation.

## Features

- **YOLOv8 Vehicle Detection**: Accurate detection of cars, trucks, buses, and motorcycles
- **Multi-Object Tracking**: Simple but effective tracking using centroid distance
- **Perspective Transformation**: Convert image coordinates to real-world coordinates
- **Speed Estimation**: Calculate vehicle speeds with smoothing and filtering
- **Real-time Visualization**: Live display with trajectories, speed info, and statistics
- **Modular Architecture**: Clean, maintainable code structure

## Project Structure

```
vehicle_speed_estimation/
├── main.py                     # Main execution script
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── data_structures.py          # Core data classes
├── perspective_transformer.py  # Coordinate transformation
├── vehicle_detector.py         # YOLOv8 detection
├── tracker.py                  # Multi-object tracking
├── speed_estimator.py          # Speed calculation
├── speed_detection_system.py   # Main system integration
└── README.md                   # This file
```

## Installation

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your video file:**
   - Place your traffic video in `data/videos/traffic_video.mp4`
   - Or update the `VIDEO_PATH` in `config.py`

4. **Configure perspective transformation:**
   - Update `IMAGE_POINTS` and `WORLD_POINTS` in `config.py`
   - These define the transformation from image to real-world coordinates

## Usage

### Basic Usage

```bash
python main.py
```

### Controls

- **'q'**: Quit the application
- **'s'**: Toggle trajectory display on/off
- **'d'**: Toggle detection boxes on/off

### Configuration

Edit `config.py` to modify:

- **Video settings**: Path, FPS fallback
- **Detection settings**: Confidence threshold, model path
- **Tracking settings**: Max disappeared frames, tracking distance
- **Perspective points**: Image and world coordinate mappings
- **Display settings**: What to show/hide by default

## Perspective Calibration

The system requires calibration to map image pixels to real-world distances. You need to identify:

1. **Four points in your video** that form a quadrilateral
2. **The real-world dimensions** of that area

Example configuration in `config.py`:
```python
IMAGE_POINTS = [
    (800, 410),   # Top-left corner in pixels
    (1125, 410),  # Top-right corner in pixels  
    (1920, 850),  # Bottom-right corner in pixels
    (0, 850)      # Bottom-left corner in pixels
]

WORLD_POINTS = [
    (0, 0),       # Top-left in meters
    (32, 0),      # Top-right in meters (32m width)
    (32, 140),    # Bottom-right in meters (140m length)  
    (0, 140)      # Bottom-left in meters
]
```

## Components

### Core Classes

- **`Detection`**: Represents a vehicle detection with bounding box and confidence
- **`TrackPoint`**: Point in a vehicle's trajectory with timestamp and coordinates
- **`VehicleTrack`**: Complete track history with speed measurements

### Main Modules

- **`VehicleDetector`**: YOLOv8-based vehicle detection
- **`SimpleTracker`**: Multi-object tracking using centroid distance
- **`PerspectiveTransformer`**: Image to world coordinate conversion
- **`SpeedEstimator`**: Speed calculation with smoothing and filtering
- **`SpeedDetectionSystem`**: Integrates all components

## Output

The system displays:

- **Bounding boxes** around detected vehicles (color-coded by speed)
- **Vehicle IDs** and estimated speeds
- **Trajectory paths** showing vehicle movement
- **Real-time statistics**: vehicle count, average speed, etc.

Final statistics are printed including:
- Total processing time and FPS
- Total vehicles detected
- Speed measurements and statistics

## Speed Estimation Method

1. **Detection**: YOLOv8 identifies vehicles in each frame
2. **Tracking**: Vehicles are tracked across frames using centroid distance
3. **Coordinate transformation**: Image pixels converted to real-world meters
4. **Speed calculation**: Distance/time between track points
5. **Smoothing**: Outlier removal and averaging for stable readings

## Requirements

- Python 3.7+
- OpenCV
- Ultralytics YOLOv8
- NumPy
- SciPy (optional, for advanced filtering)
- PyTorch (for YOLOv8)

## Limitations

- Requires manual perspective calibration
- Simple tracking may lose vehicles in crowded scenes
- Speed accuracy depends on calibration quality
- Works best with overhead/angled camera views

## Customization

The modular structure makes it easy to:

- **Replace the detector**: Swap YOLOv8 for other detection models
- **Improve tracking**: Implement more sophisticated tracking algorithms
- **Add features**: Speed limit warnings, vehicle counting, etc.
- **Optimize performance**: Adjust detection frequency, tracking parameters

## Troubleshooting

**"ultralytics not installed"**: Run `pip install ultralytics`

**"Could not open video file"**: Check the video path in `config.py`

**Inaccurate speeds**: Verify perspective calibration points

**Poor tracking**: Adjust `MAX_TRACKING_DISTANCE` in `config.py`

## License

This project is provided as-is for educational and research purposes.