import cv2
import mediapipe as mp
import time
import os
import json
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

# --- Configuration ---
MODEL_PATH = 'hand_landmarker.task'
# ---------------------

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process video for 3D hand tracking and save results to JSON (Parallel Processing)')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output_dir', type=str, default='hand_tracking_output', help='Directory to save JSON output files')
    parser.add_argument('--display_width', type=int, default=640, help='Width of display window (default: 640)')
    parser.add_argument('--min_detection_confidence', type=float, default=0.2, help='Minimum detection confidence (default: 0.2)')
    parser.add_argument('--min_tracking_confidence', type=float, default=0.2, help='Minimum tracking confidence (default: 0.2)')
    parser.add_argument('--use_last_known', action='store_true', help='Use last known detection when hands not detected')
    parser.add_argument('--max_interpolate_frames', type=int, default=5, help='Maximum frames to interpolate from last detection (default: 5)')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--chunk_size', type=int, default=100, help='Number of frames per chunk (default: 100)')
    return parser.parse_args()

def save_landmarks_to_json(frame_number, hand_world_landmarks_list, hand_landmarks_list, output_dir, detection_status="detected"):
    """
    Save the 3D world landmarks and 2D normalized landmarks to a JSON file for a given frame.
    
    Args:
        frame_number: The frame number
        hand_world_landmarks_list: List of hand world landmarks from MediaPipe (3D coordinates)
        hand_landmarks_list: List of hand landmarks from MediaPipe (2D normalized coordinates)
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
                'world_landmarks': [],
                'landmarks': []
            }
            
            # Save 3D world landmarks
            for landmark_idx, landmark in enumerate(hand_world_landmarks):
                landmark_data = {
                    'landmark_id': landmark_idx,
                    'x': float(landmark.x),
                    'y': float(landmark.y),
                    'z': float(landmark.z)
                }
                hand_data['world_landmarks'].append(landmark_data)
            
            # Save 2D normalized landmarks (for visualization)
            if hand_landmarks_list and hand_idx < len(hand_landmarks_list):
                for landmark_idx, landmark in enumerate(hand_landmarks_list[hand_idx]):
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

def process_chunk(args_tuple):
    """
    Process a chunk of video frames.
    
    Args:
        args_tuple: Tuple containing (video_path, start_frame, end_frame, output_dir, config)
    
    Returns:
        Tuple of (chunk_id, frames_processed, detected_count, interpolated_count)
    """
    video_path, start_frame, end_frame, output_dir, config = args_tuple
    
    # MediaPipe setup
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    
    # Create HandLandmarker options for VIDEO mode with CPU delegate
    # Using CPU for multiprocessing stability and better parallelism
    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=BaseOptions.Delegate.CPU  # Use CPU for parallel processing
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=config['min_detection_confidence'],
        min_hand_presence_confidence=config['min_tracking_confidence'],
        min_tracking_confidence=config['min_tracking_confidence']
    )
    
    frames_processed = 0
    detected_count = 0
    interpolated_count = 0
    
    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file '{video_path}' in worker")
                return (start_frame, 0, 0, 0)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            last_known_world_landmarks = None
            last_known_hand_landmarks = None
            frames_since_detection = 0
            
            for frame_number in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create MediaPipe Image object
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Calculate timestamp
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
                    if (config['use_last_known'] and 
                        last_known_world_landmarks is not None and 
                        frames_since_detection <= config['max_interpolate_frames']):
                        world_landmarks_to_use = last_known_world_landmarks
                        hand_landmarks_to_use = last_known_hand_landmarks
                        detection_status = "interpolated"
                        interpolated_count += 1
                else:
                    last_known_world_landmarks = detection_result.hand_world_landmarks
                    last_known_hand_landmarks = detection_result.hand_landmarks
                    frames_since_detection = 0
                    detected_count += 1
                
                # Save landmarks to JSON
                save_landmarks_to_json(
                    frame_number, 
                    world_landmarks_to_use,
                    hand_landmarks_to_use,
                    output_dir,
                    detection_status
                )
                
                frames_processed += 1
            
            cap.release()
    
    except Exception as e:
        print(f"Error processing chunk {start_frame}-{end_frame}: {str(e)}")
    
    return (start_frame, frames_processed, detected_count, interpolated_count)

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
    
    # Get video properties
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{args.video}'")
        exit(1)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Determine number of workers
    num_workers = args.num_workers if args.num_workers else min(10, cpu_count())
    chunk_size = args.chunk_size
    
    print(f"Processing video: {args.video}")
    print(f"Video size: {orig_width}x{orig_height}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Using CPU parallel processing (10 workers)")
    print(f"Detection confidence: {args.min_detection_confidence}")
    print(f"Tracking confidence: {args.min_tracking_confidence}")
    print(f"Number of workers: {num_workers}")
    print(f"Chunk size: {chunk_size} frames")
    if args.use_last_known:
        print(f"Using last known detection (max {args.max_interpolate_frames} frames)")
    print("-" * 50)
    
    # Create chunks
    chunks = []
    config = {
        'min_detection_confidence': args.min_detection_confidence,
        'min_tracking_confidence': args.min_tracking_confidence,
        'use_last_known': args.use_last_known,
        'max_interpolate_frames': args.max_interpolate_frames
    }
    
    for start_frame in range(0, total_frames, chunk_size):
        end_frame = min(start_frame + chunk_size, total_frames)
        chunks.append((args.video, start_frame, end_frame, str(output_dir), config))
    
    print(f"Created {len(chunks)} chunks for parallel processing")
    print("-" * 50)
    
    # Process chunks in parallel
    start_time = time.time()
    
    with Pool(processes=num_workers) as pool:
        results = []
        for i, result in enumerate(pool.imap(process_chunk, chunks)):
            results.append(result)
            chunk_start, frames_processed, detected, interpolated = result
            progress = (i + 1) / len(chunks) * 100
            print(f"Chunk {i+1}/{len(chunks)} complete ({progress:.1f}%) - Frames: {chunk_start}-{chunk_start+frames_processed}, Detected: {detected}, Interpolated: {interpolated}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate statistics
    total_processed = sum(r[1] for r in results)
    total_detected = sum(r[2] for r in results)
    total_interpolated = sum(r[3] for r in results)
    
    print("-" * 50)
    print(f"Processing complete!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Processing speed: {total_processed / elapsed_time:.2f} frames/second")
    print(f"Processed {total_processed} frames")
    print(f"Frames with detection: {total_detected} ({total_detected/total_processed*100:.1f}%)")
    if args.use_last_known:
        print(f"Frames interpolated: {total_interpolated} ({total_interpolated/total_processed*100:.1f}%)")
    print(f"JSON files saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
