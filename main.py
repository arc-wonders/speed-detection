"""
Main script to run the vehicle speed estimation system.

Usage:
    python main.py
    
Controls:
    - 'q': Quit
    - 's': Toggle trajectory display
    - 'd': Toggle detection boxes
"""

import os
import cv2
import time

# Suppress OpenMP warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from speed_detection_system import SpeedDetectionSystem
import config


def main():
    """Main function to run the speed detection system."""
    
    try:
        # Initialize system
        print("Initializing speed detection system...")
        system = SpeedDetectionSystem(
            model_path=config.MODEL_PATH, 
            confidence_threshold=config.CONFIDENCE_THRESHOLD
        )
        
        # Set display preferences
        system.show_trajectories = config.DEFAULT_SHOW_TRAJECTORIES
        system.show_detection_boxes = config.DEFAULT_SHOW_DETECTION_BOXES
        system.show_speed_info = config.DEFAULT_SHOW_SPEED_INFO
        
        # Calibrate perspective transformation
        print("Calibrating perspective transformation...")
        if not system.calibrate_perspective(config.IMAGE_POINTS, config.WORLD_POINTS):
            print("Failed to calibrate perspective transformation")
            return
        
        # Open video
        print(f"Opening video: {config.VIDEO_PATH}")
        cap = cv2.VideoCapture(config.VIDEO_PATH)
        if not cap.isOpened():
            print(f"Error: Could not open video file {config.VIDEO_PATH}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 1000:
            fps = config.DEFAULT_FPS
        print(f"Video FPS: {fps}")
        
        frame_count = 0
        start_time = time.time()
        
        print("Starting video processing...")
        print("Controls: 'q' to quit, 's' to toggle trajectories, 'd' to toggle detection boxes")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps
            
            # Process frame
            try:
                processed_frame = system.process_frame(frame, timestamp)
                
                # Display frame
                cv2.imshow('Vehicle Speed Detection', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    system.show_trajectories = not system.show_trajectories
                    print(f"Trajectories: {'ON' if system.show_trajectories else 'OFF'}")
                elif key == ord('d'):
                    system.show_detection_boxes = not system.show_detection_boxes
                    print(f"Detection boxes: {'ON' if system.show_detection_boxes else 'OFF'}")
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                continue
        
        # Print final statistics
        elapsed_time = time.time() - start_time
        stats = system.speed_estimator.get_statistics()
        
        print("\n" + "="*50)
        print("FINAL STATISTICS")
        print("="*50)
        print(f"Processing time: {elapsed_time:.1f} seconds")
        print(f"Frames processed: {frame_count}")
        print(f"Average FPS: {frame_count/elapsed_time:.1f}")
        print(f"Total vehicles detected: {stats['total_vehicles']}")
        print(f"Speed measurements: {stats['measurements']}")
        
        if stats['average_speed'] is not None:
            print(f"Average speed: {stats['average_speed']:.1f} ± {stats['std_speed']:.1f} km/h")
            print(f"Speed range: {stats['min_speed']:.1f} - {stats['max_speed']:.1f} km/h")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("Cleanup completed")


if __name__ == "__main__":
    main()