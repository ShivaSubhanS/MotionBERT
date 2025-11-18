import cv2
import mediapipe as mp
import time
import os
import json
import argparse
from pathlib import Path
from mediapipe.framework.formats import landmark_pb2

# --- Configuration ---
MODEL_PATH = 'hand_landmarker.task'
# ---------------------

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process video for 3D hand tracking and save results to JSON')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output_dir', type=str, default='hand_tracking_output', help='Directory to save JSON output files')
    parser.add_argument('--display_width', type=int, default=640, help='Width of display window (default: 640)')
    parser.add_argument('--min_detection_confidence', type=float, default=0.2, help='Minimum detection confidence (default: 0.2)')
    parser.add_argument('--min_tracking_confidence', type=float, default=0.2, help='Minimum tracking confidence (default: 0.2)')
    parser.add_argument('--use_last_known', action='store_true', help='Use last known detection when hands not detected')
    parser.add_argument('--max_interpolate_frames', type=int, default=5, help='Maximum frames to interpolate from last detection (default: 5)')
    return parser.parse_args()

def save_landmarks_to_json(frame_number, hand_world_landmarks_list, output_dir, detection_status="detected"):
    """
    Save the 3D world landmarks to a JSON file for a given frame.
    
    Args:
        frame_number: The frame number
        hand_world_landmarks_list: List of hand world landmarks from MediaPipe
        output_dir: Directory to save the JSON file
        detection_status: Status of detection ("detected", "interpolated", "last_known")
    """
    output_data = {
        'frame_number': frame_number,
        'timestamp': time.time(),
        'detection_status': detection_status,
        'hands': []
    }
    
    if hand_world_landmarks_list:
        for hand_idx, hand_world_landmarks in enumerate(hand_world_landmarks_list):
            hand_data = {
                'hand_index': hand_idx,
                'landmarks': []
            }
            
            for landmark_idx, landmark in enumerate(hand_world_landmarks):
                landmark_data = {
                    'landmark_id': landmark_idx,
                    'x': float(landmark.x),
                    'y': float(landmark.y),
                    'z': float(landmark.z)
                }
                hand_data['landmarks'].append(landmark_data)
            
            output_data['hands'].append(hand_data)
    
    # Save to JSON file
    output_file = os.path.join(output_dir, f'frame_{frame_number:06d}.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

def main():
    args = parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found at '{args.video}'")
        exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please download the 'hand_landmarker.task' model from:")
        print("https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index#models")
        exit(1)

    # MediaPipe setup
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # Create HandLandmarker options for VIDEO mode with GPU delegate
    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=BaseOptions.Delegate.GPU  # Enable GPU acceleration
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=2,  # Detect up to 2 hands
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_tracking_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )

    # Main processing loop
    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                print(f"Error: Could not open video file '{args.video}'")
                exit(1)
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate display dimensions maintaining aspect ratio
            display_width = args.display_width
            display_height = int(orig_height * (display_width / orig_width))
            
            print(f"Processing video: {args.video}")
            print(f"Original size: {orig_width}x{orig_height}")
            print(f"Display size: {display_width}x{display_height}")
            print(f"Total frames: {total_frames}")
            print(f"FPS: {fps}")
            print(f"Using GPU acceleration")
            print(f"Detection confidence: {args.min_detection_confidence}")
            print(f"Tracking confidence: {args.min_tracking_confidence}")
            if args.use_last_known:
                print(f"Using last known detection (max {args.max_interpolate_frames} frames)")
            print("-" * 50)
            
            frame_number = 0
            last_known_world_landmarks = None
            last_known_hand_landmarks = None
            frames_since_detection = 0
            total_detected = 0
            total_interpolated = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR (OpenCV) to RGB (MediaPipe)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create MediaPipe Image object
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Get a timestamp (required for VIDEO mode)
                # Use frame number to calculate timestamp based on FPS
                frame_timestamp_ms = int((frame_number / fps) * 1000)
                
                # Run detection
                detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                
                # Determine which landmarks to use
                world_landmarks_to_use = detection_result.hand_world_landmarks
                hand_landmarks_to_use = detection_result.hand_landmarks
                detection_status = "detected"
                
                # Handle missing detections
                if not detection_result.hand_landmarks or len(detection_result.hand_landmarks) == 0:
                    frames_since_detection += 1
                    
                    # Use last known detection if enabled and within max frames
                    if (args.use_last_known and 
                        last_known_world_landmarks is not None and 
                        frames_since_detection <= args.max_interpolate_frames):
                        world_landmarks_to_use = last_known_world_landmarks
                        hand_landmarks_to_use = last_known_hand_landmarks
                        detection_status = "interpolated"
                        total_interpolated += 1
                else:
                    # Update last known detections
                    last_known_world_landmarks = detection_result.hand_world_landmarks
                    last_known_hand_landmarks = detection_result.hand_landmarks
                    frames_since_detection = 0
                    total_detected += 1
                
                # Create annotated frame for display
                annotated_frame = frame.copy()
                
                # Draw 2D hand landmarks on the frame
                if hand_landmarks_to_use:
                    for hand_landmarks_list in hand_landmarks_to_use:
                        # Convert to protobuf format for drawing
                        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                        hand_landmarks_proto.landmark.extend([
                            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                            for landmark in hand_landmarks_list
                        ])
                        
                        # Draw the hand landmarks and connections
                        # Use different color for interpolated detections
                        if detection_status == "interpolated":
                            # Draw in yellow for interpolated
                            mp_drawing.draw_landmarks(
                                annotated_frame,
                                hand_landmarks_proto,
                                mp_hands.HAND_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                            )
                        else:
                            # Draw in default colors for detected
                            mp_drawing.draw_landmarks(
                                annotated_frame,
                                hand_landmarks_proto,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style()
                            )
                
                # Add frame info and progress to display
                progress = (frame_number + 1) / total_frames * 100
                status_color = (0, 255, 255) if detection_status == "interpolated" else (0, 255, 0)
                cv2.putText(annotated_frame, f"Frame: {frame_number + 1}/{total_frames} ({progress:.1f}%)", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, f"Hands: {len(hand_landmarks_to_use) if hand_landmarks_to_use else 0} [{detection_status}]", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, "Press 'q' to quit", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Resize frame for display
                display_frame = cv2.resize(annotated_frame, (display_width, display_height))
                
                # Display the annotated frame
                cv2.imshow('Hand Tracking - Processing', display_frame)
                
                # Save landmarks to JSON
                save_landmarks_to_json(
                    frame_number, 
                    world_landmarks_to_use, 
                    str(output_dir),
                    detection_status
                )
                
                # Print progress
                if frame_number % 30 == 0 or frame_number == total_frames - 1:
                    print(f"Progress: {frame_number + 1}/{total_frames} frames ({progress:.1f}%)")
                
                frame_number += 1
                
                # Check for 'q' key to quit early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing interrupted by user")
                    break

    finally:
        # Clean up
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("-" * 50)
        print(f"Processing complete!")
        print(f"Processed {frame_number} frames")
        if 'total_detected' in locals():
            print(f"Frames with detection: {total_detected} ({total_detected/frame_number*100:.1f}%)")
            if args.use_last_known:
                print(f"Frames interpolated: {total_interpolated} ({total_interpolated/frame_number*100:.1f}%)")
        print(f"JSON files saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()

