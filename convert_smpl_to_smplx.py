#!/usr/bin/env python3
"""
Convert MotionBERT SMPL parameters to SMPL-X format for Blender addon.

MotionBERT outputs SMPL format:
- body_pose: (N, 69) - 23 joints × 3 (axis-angle)
- global_orient: (N, 3) - root orientation
- betas: (N, 10) - shape parameters
- transl: (N, 3) - translation

SMPL-X Blender addon expects:
- body_pose: (1, 63) - 21 joints × 3 (axis-angle)
- global_orient: (1, 3)
- betas: (1, 10)
- jaw_pose: (1, 3)
- left_hand_pose: (1, 45) - 15 joints × 3
- right_hand_pose: (1, 45) - 15 joints × 3
- expression: (1, 10)
- transl: (3,) optional
"""

import numpy as np
import pickle
import argparse
import os
import sys
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

# Add paths for Homogenus
sys.path.insert(0, '/home/sss/project/pose_3d/homogenus')
try:
    from homogenus.tf.homogenus_infer import Homogenus_infer
    HOMOGENUS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Homogenus: {e}")
    HOMOGENUS_AVAILABLE = False

# SMPL joint order (24 joints, but body_pose has 23 excluding root):
# 0: pelvis (root - stored in global_orient)
# 1: left_hip, 2: right_hip, 3: spine1, 4: left_knee, 5: right_knee
# 6: spine2, 7: left_ankle, 8: right_ankle, 9: spine3, 10: left_foot
# 11: right_foot, 12: neck, 13: left_collar, 14: right_collar, 15: head
# 16: left_shoulder, 17: right_shoulder, 18: left_elbow, 19: right_elbow
# 20: left_wrist, 21: right_wrist, 22: left_hand, 23: right_hand

# SMPL-X body joint order (21 joints):
# Similar to SMPL but excludes hand joints (22, 23)


def smooth_rotations(rotations, sigma=1.0, method='gaussian'):
    """
    Smooth rotation sequences using quaternion interpolation.
    
    Args:
        rotations: (N, 3) axis-angle rotations or (N, M) for multiple joints
        sigma: smoothing strength (higher = smoother)
        method: 'gaussian' or 'savgol'
    
    Returns:
        smoothed: (N, 3) or (N, M) smoothed rotations
    """
    if rotations.ndim == 1:
        rotations = rotations.reshape(-1, 3)
    
    num_frames = rotations.shape[0]
    num_joints = rotations.shape[1] // 3
    
    smoothed = np.zeros_like(rotations)
    
    for j in range(num_joints):
        # Extract this joint's rotations
        joint_rot = rotations[:, j*3:(j+1)*3]
        
        # Convert to quaternions for proper interpolation
        quats = np.zeros((num_frames, 4))
        for i in range(num_frames):
            r = R.from_rotvec(joint_rot[i])
            quats[i] = r.as_quat()  # [x, y, z, w]
        
        # Handle quaternion sign flips (q and -q represent same rotation)
        for i in range(1, num_frames):
            if np.dot(quats[i], quats[i-1]) < 0:
                quats[i] = -quats[i]
        
        # Smooth each quaternion component
        if method == 'gaussian':
            smoothed_quats = np.zeros_like(quats)
            for k in range(4):
                smoothed_quats[:, k] = gaussian_filter1d(quats[:, k], sigma=sigma)
        elif method == 'savgol':
            window = min(int(sigma * 4) * 2 + 1, num_frames)
            if window < 5:
                window = min(5, num_frames)
            if window % 2 == 0:
                window += 1
            polyorder = min(3, window - 1)
            smoothed_quats = np.zeros_like(quats)
            for k in range(4):
                smoothed_quats[:, k] = savgol_filter(quats[:, k], window, polyorder)
        else:
            smoothed_quats = quats
        
        # Normalize quaternions
        norms = np.linalg.norm(smoothed_quats, axis=1, keepdims=True)
        smoothed_quats = smoothed_quats / norms
        
        # Convert back to axis-angle
        for i in range(num_frames):
            r = R.from_quat(smoothed_quats[i])
            smoothed[i, j*3:(j+1)*3] = r.as_rotvec()
    
    return smoothed


def smooth_translations(translations, sigma=1.0, method='gaussian'):
    """
    Smooth translation sequences.
    
    Args:
        translations: (N, 3) translations
        sigma: smoothing strength
        method: 'gaussian' or 'savgol'
    
    Returns:
        smoothed: (N, 3) smoothed translations
    """
    smoothed = np.zeros_like(translations)
    
    if method == 'gaussian':
        for i in range(3):
            smoothed[:, i] = gaussian_filter1d(translations[:, i], sigma=sigma)
    elif method == 'savgol':
        num_frames = translations.shape[0]
        window = min(int(sigma * 4) * 2 + 1, num_frames)
        if window < 5:
            window = min(5, num_frames)
        if window % 2 == 0:
            window += 1
        polyorder = min(3, window - 1)
        for i in range(3):
            smoothed[:, i] = savgol_filter(translations[:, i], window, polyorder)
    else:
        smoothed = translations.copy()
    
    return smoothed


def fix_orientation_flips(global_orient, threshold_deg=120):
    """
    Fix sudden 180-degree flips in global orientation.
    
    The MotionBERT output sometimes has the character flip orientation
    where X euler angle jumps from ~±180° to ~0° (or vice versa).
    When this happens, the Y rotation also becomes incorrect.
    
    Strategy:
    1. Find frames where there's a sudden large rotation change (>threshold)
    2. Check if this is a flip (X near ±180° jumps to X near 0° or vice versa)
    3. For flip segments, interpolate Y and Z from surrounding good frames
    4. Apply 180° X rotation to correct the flip
    
    Args:
        global_orient: (N, 3) axis-angle rotations
        threshold_deg: angle threshold to detect flips (default 120°)
    
    Returns:
        fixed_orient: (N, 3) corrected axis-angle rotations
    """
    fixed_orient = global_orient.copy()
    num_frames = len(global_orient)
    threshold_rad = np.deg2rad(threshold_deg)
    
    # Convert all to euler angles for analysis
    eulers = np.zeros((num_frames, 3))
    for i in range(num_frames):
        r = R.from_rotvec(global_orient[i])
        eulers[i] = r.as_euler('XYZ', degrees=True)
    
    # Determine reference state from frame 0
    ref_x_near_180 = abs(eulers[0, 0]) > 90
    print(f"  Reference frame 0: X={eulers[0,0]:.1f}°, Y={eulers[0,1]:.1f}°, Z={eulers[0,2]:.1f}°")
    print(f"  Reference X is {'near ±180°' if ref_x_near_180 else 'near 0°'}")
    
    # Find flipped frames based on:
    # 1. Large angular change from previous frame (>threshold)
    # 2. X angle crossed the 90° boundary (true flip, not smooth rotation)
    is_flipped = np.zeros(num_frames, dtype=bool)
    
    for i in range(1, num_frames):
        r_prev = R.from_rotvec(fixed_orient[i-1])
        r_curr = R.from_rotvec(global_orient[i])
        
        angle_diff = (r_prev.inv() * r_curr).magnitude()
        
        # Check if X crossed boundary
        prev_x_near_180 = abs(eulers[i-1, 0]) > 90
        curr_x_near_180 = abs(eulers[i, 0]) > 90
        
        # Only mark as flipped if:
        # - Large angular change (>threshold)
        # - X boundary was crossed (flip, not smooth rotation)
        # - The flip goes AWAY from reference state
        if angle_diff > threshold_rad and prev_x_near_180 != curr_x_near_180:
            # Check if this frame is in the "wrong" state relative to reference
            if curr_x_near_180 != ref_x_near_180:
                is_flipped[i] = True
    
    # Propagate flip state forward until we return to normal
    for i in range(1, num_frames):
        if is_flipped[i-1]:
            curr_x_near_180 = abs(eulers[i, 0]) > 90
            if curr_x_near_180 != ref_x_near_180:
                is_flipped[i] = True
    
    # Find contiguous flipped segments
    flip_segments = []
    in_segment = False
    seg_start = 0
    
    for i in range(num_frames):
        if is_flipped[i] and not in_segment:
            seg_start = i
            in_segment = True
        elif not is_flipped[i] and in_segment:
            flip_segments.append((seg_start, i - 1))
            in_segment = False
    
    if in_segment:
        flip_segments.append((seg_start, num_frames - 1))
    
    print(f"  Found {len(flip_segments)} flipped segments: {flip_segments}")
    
    # Fix each segment
    for seg_start, seg_end in flip_segments:
        print(f"  Fixing segment frames {seg_start}-{seg_end}")
        
        # Get Y and Z rotation from surrounding good frames for interpolation
        y_before = eulers[seg_start - 1, 1] if seg_start > 0 else eulers[0, 1]
        y_after = eulers[seg_end + 1, 1] if seg_end < num_frames - 1 else y_before
        
        z_before = eulers[seg_start - 1, 2] if seg_start > 0 else eulers[0, 2]
        z_after = eulers[seg_end + 1, 2] if seg_end < num_frames - 1 else z_before
        
        seg_len = seg_end - seg_start + 1
        
        for i in range(seg_start, seg_end + 1):
            curr_euler = eulers[i].copy()
            
            # Flip X by adding/subtracting 180°
            if curr_euler[0] > 0:
                new_x = curr_euler[0] - 180
            else:
                new_x = curr_euler[0] + 180
            
            # Interpolate Y and Z from surrounding good frames
            t = (i - seg_start) / max(seg_len - 1, 1) if seg_len > 1 else 0.5
            new_y = y_before + t * (y_after - y_before)
            new_z = z_before + t * (z_after - z_before)
            
            new_euler = np.array([new_x, new_y, new_z])
            
            # Convert back to rotation
            r_new = R.from_euler('XYZ', new_euler, degrees=True)
            fixed_orient[i] = r_new.as_rotvec()
    
    total_fixed = sum(seg_end - seg_start + 1 for seg_start, seg_end in flip_segments)
    print(f"  Total frames fixed: {total_fixed}")
    
    return fixed_orient


def predict_gender_with_homogenus(video_file, keypoint_file, fallback_gender='male'):
    """
    Predict gender using Homogenus with fallback to specified gender.
    
    Args:
        video_file: Path to the video file
        keypoint_file: Path to AlphaPose keypoint JSON file  
        fallback_gender: Gender to use if prediction fails (default 'male')
    
    Returns:
        str: Predicted gender ('male' or 'female')
    """
    if not HOMOGENUS_AVAILABLE:
        print(f"  Homogenus not available, using fallback gender: {fallback_gender}")
        return fallback_gender
        
    try:
        print(f"  Predicting gender using Homogenus...")
        print(f"    Video file: {video_file}")
        print(f"    Keypoint file: {keypoint_file}")
        
        # Load the keypoint file to check what image_ids exist
        import json
        with open(keypoint_file, 'r') as f:
            keypoint_data = json.load(f)
        
        # Find the first available detection to use as reference
        if keypoint_data and len(keypoint_data) > 0:
            # Get first detection's image_id to use as frame reference
            first_detection = keypoint_data[0]
            target_image_id = first_detection['image_id']
            print(f"    Using keypoint data for frame: {target_image_id}")
            
            # Create a subset keypoint file with only the first frame's detections
            import tempfile
            import os
            
            # Filter detections for the target frame
            target_detections = [d for d in keypoint_data if d['image_id'] == target_image_id]
            
            if len(target_detections) == 0:
                print(f"    No detections found for {target_image_id}, using fallback: {fallback_gender}")
                return fallback_gender
                
            # Create temporary keypoint file with just this frame's data
            temp_keypoint_dir = tempfile.mkdtemp(prefix="homogenus_keypoints_")
            temp_keypoint_file = os.path.join(temp_keypoint_dir, 'alphapose-results.json')
            
            with open(temp_keypoint_file, 'w') as f:
                json.dump(target_detections, f)
            
            print(f"    Created temporary keypoint file with {len(target_detections)} detections")
            
            # Initialize Homogenus
            homogenus_model_dir = "/home/sss/project/pose_3d/homogenus/homogenus/trained_models/tf"
            hg = Homogenus_infer(trained_model_dir=homogenus_model_dir)
            
            # Predict gender - override the frame naming to match the keypoint data
            # Extract frame from video and save with the correct name
            import cv2
            import shutil
            
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                print(f"    Could not open video file, using fallback: {fallback_gender}")
                return fallback_gender
                
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print(f"    Could not read video frame, using fallback: {fallback_gender}")
                return fallback_gender
            
            # Save frame with the correct filename to match keypoint data
            temp_image_dir = tempfile.mkdtemp(prefix="homogenus_images_")
            temp_image_file = os.path.join(temp_image_dir, target_image_id)
            cv2.imwrite(temp_image_file, frame)
            
            # Predict gender
            results = hg.predict_genders(
                images_indir=temp_image_dir,
                openpose_indir=temp_keypoint_dir,
                pose_format='alphapose',
                video_file=None,  # Don't use video since we've extracted the frame
                images_outdir=None,  # Don't save images
                openpose_outdir=None  # Don't save augmented keypoints
            )
            
            # Cleanup temporary directories
            shutil.rmtree(temp_keypoint_dir)
            shutil.rmtree(temp_image_dir)
            
            # Extract gender from results
            for image_name, genders in results.items():
                if genders and len(genders) > 0:
                    predicted_gender = genders[0]['gender']
                    print(f"    Homogenus prediction: {predicted_gender}")
                    return predicted_gender
                    
        print(f"    No valid detections found, using fallback: {fallback_gender}")
        return fallback_gender
        
    except Exception as e:
        print(f"    Error during gender prediction: {e}")
        print(f"    Using fallback gender: {fallback_gender}")
        return fallback_gender


def convert_smpl_to_smplx_single_frame(body_pose_smpl, global_orient, betas, transl=None):
    """
    Convert a single frame of SMPL params to SMPL-X format.
    
    SMPL body_pose has 23 joints (69 params), SMPL-X has 21 (63 params).
    The last 2 joints in SMPL (left_hand, right_hand) are not in SMPL-X body_pose.
    """
    # body_pose_smpl: (69,) -> body_pose_smplx: (63,)
    # Remove joints 22 and 23 (left_hand, right_hand) - indices 21*3:23*3 = 63:69
    body_pose_smplx = body_pose_smpl[:63].reshape(1, 63)
    
    # global_orient stays the same
    global_orient_smplx = global_orient.reshape(1, 3)
    
    # betas stays the same (but we only use first 10)
    betas_smplx = betas[:10].reshape(1, 10)
    
    # Create default values for SMPL-X specific parameters
    jaw_pose = np.zeros((1, 3))
    left_hand_pose = np.zeros((1, 45))  # 15 joints × 3
    right_hand_pose = np.zeros((1, 45))  # 15 joints × 3
    expression = np.zeros((1, 10))
    
    result = {
        'body_pose': body_pose_smplx,
        'global_orient': global_orient_smplx,
        'betas': betas_smplx,
        'jaw_pose': jaw_pose,
        'left_hand_pose': left_hand_pose,
        'right_hand_pose': right_hand_pose,
        'expression': expression,
    }
    
    if transl is not None:
        result['transl'] = transl.reshape(3)
    
    return result


def convert_smpl_to_smplx_animation(smpl_params, gender='male'):
    """
    Convert full animation sequence for use with SMPL-X Blender addon's 
    "Add Animation" feature (SMPL-X format .npz).
    
    Returns data in AMASS/SMPL-X animation format.
    
    Args:
        smpl_params: Dictionary with SMPL parameters
        gender: Gender for the animation ('male' or 'female')
    """
    body_pose = smpl_params['body_pose']  # (N, 69)
    global_orient = smpl_params['global_orient']  # (N, 3)
    betas = smpl_params['betas']  # (N, 10)
    transl = smpl_params['transl']  # (N, 3)
    
    num_frames = body_pose.shape[0]
    
    # SMPL-X full pose: 55 joints × 3 = 165
    # pelvis(3) + body(63) + jaw(3) + leye(3) + reye(3) + left_hand(45) + right_hand(45) = 165
    poses = np.zeros((num_frames, 165))
    
    for i in range(num_frames):
        # Global orient (pelvis) - first 3 values
        poses[i, 0:3] = global_orient[i]
        
        # Body pose (21 joints) - next 63 values
        # SMPL has 23 body joints, SMPL-X has 21 (excluding hand base joints)
        poses[i, 3:66] = body_pose[i, :63]
        
        # jaw_pose, leye_pose, reye_pose - 9 values (zeros)
        # left_hand_pose - 45 values (zeros) 
        # right_hand_pose - 45 values (zeros)
        # These remain zero
    
    # Use average betas across all frames
    avg_betas = np.mean(betas, axis=0)[:10]
    
    result = {
        'trans': transl,
        'gender': gender,  # Use predicted gender
        'mocap_framerate': 30,  # Default framerate
        'betas': avg_betas,
        'poses': poses,
    }
    
    return result


def load_smplx_betas(smplx_pkl_path):
    """Load betas from SMPLify-X output to get accurate body shape."""
    print(f"Loading body shape from SMPLify-X: {smplx_pkl_path}")
    with open(smplx_pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    betas = np.array(data['betas']).flatten()
    print(f"  SMPLify-X betas shape: {betas.shape}")
    print(f"  SMPLify-X betas[:5]: {betas[:5]}")
    return betas


def main():
    parser = argparse.ArgumentParser(description='Convert MotionBERT SMPL to SMPL-X format')
    parser.add_argument('-i', '--input', required=True, help='Input pkl file from MotionBERT')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    parser.add_argument('--fps', type=int, default=30, help='Framerate for animation')
    parser.add_argument('--gender', default=None, choices=['female', 'male', 'neutral'], 
                        help='Gender for SMPL-X model (if not specified, will use Homogenus to predict)')
    parser.add_argument('--video-file', type=str, default=None,
                        help='Path to video file for gender prediction (auto-detected if not provided)')
    parser.add_argument('--keypoint-file', type=str, default=None,
                        help='Path to AlphaPose keypoint JSON file for gender prediction (auto-detected if not provided)')
    parser.add_argument('--fallback-gender', default='male', choices=['female', 'male', 'neutral'],
                        help='Fallback gender if Homogenus prediction fails (default: male)')
    parser.add_argument('--smplx-betas', type=str, default=None,
                        help='Path to SMPLify-X pkl file to use its betas for accurate body shape')
    parser.add_argument('--smooth', type=float, default=0.0,
                        help='Smoothing strength (0=none, 1=light, 2=medium, 3=strong). Recommended: 1-2 for sparse data')
    parser.add_argument('--smooth-method', default='gaussian', choices=['gaussian', 'savgol'],
                        help='Smoothing method: gaussian (default) or savgol (Savitzky-Golay)')
    args = parser.parse_args()
    
    # Load body shape from SMPLify-X if provided
    smplx_betas = None
    if args.smplx_betas:
        smplx_betas = load_smplx_betas(args.smplx_betas)
    
    # Load MotionBERT output
    print(f"\nLoading SMPL params from: {args.input}")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    
    smpl_params = data['pred_smpl_params']
    
    print(f"Input data:")
    print(f"  - body_pose: {smpl_params['body_pose'].shape}")
    print(f"  - global_orient: {smpl_params['global_orient'].shape}")
    print(f"  - betas: {smpl_params['betas'].shape}")
    print(f"  - transl: {smpl_params['transl'].shape}")
    
    num_frames = smpl_params['body_pose'].shape[0]
    print(f"  - Total frames: {num_frames}")
    
    # Determine gender using Homogenus if not manually specified
    if args.gender is None:
        print(f"\nPredicting gender using Homogenus...")
        
        # Auto-detect video file if not provided
        video_file = args.video_file
        if video_file is None:
            # Look for the original high-res video first (for AlphaPose keypoints), then fallback to others
            input_dir = os.path.dirname(args.input)
            for video_name in ['mapping*.mp4', 'mesh.mp4', 'X3D.mp4']:
                video_path = os.path.join(input_dir, video_name)
                if '*' in video_name:
                    # Use glob for pattern matching
                    import glob
                    matches = glob.glob(video_path)
                    if matches:
                        video_file = matches[0]
                        break
                elif os.path.exists(video_path):
                    video_file = video_path
                    break
        
        # Auto-detect keypoint file if not provided  
        keypoint_file = args.keypoint_file
        if keypoint_file is None:
            # Look for AlphaPose results
            possible_paths = [
                '/home/sss/project/pose_3d/AlphaPose/results_video/alphapose-results.json',
                os.path.join(os.path.dirname(args.input), 'alphapose-results.json'),
                os.path.join(os.path.dirname(args.input), '..', 'AlphaPose', 'results_video', 'alphapose-results.json')
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    keypoint_file = path
                    break
        
        # Predict gender
        if video_file and keypoint_file:
            gender = predict_gender_with_homogenus(video_file, keypoint_file, args.fallback_gender)
        else:
            print(f"  Could not find video or keypoint files for gender prediction")
            print(f"    Video file: {video_file}")
            print(f"    Keypoint file: {keypoint_file}")
            print(f"  Using fallback gender: {args.fallback_gender}")
            gender = args.fallback_gender
    else:
        gender = args.gender
        print(f"\nUsing manually specified gender: {gender}")

    print(f"Final gender selection: {gender}")
    
    # Override betas with SMPLify-X betas if provided
    if smplx_betas is not None:
        print(f"\n*** Using SMPLify-X betas for accurate body shape ***")
        # Broadcast SMPLify-X betas to all frames (first 10 components)
        smpl_params['betas'] = np.tile(smplx_betas[:10], (num_frames, 1))
        print(f"  Overriding betas with SMPLify-X shape: {smpl_params['betas'].shape}")
    
    # Fix orientation flips (180° sudden rotations)
    print(f"\nFixing orientation flips...")
    smpl_params['global_orient'] = fix_orientation_flips(smpl_params['global_orient'], threshold_deg=120)
    
    # Apply smoothing if requested
    if args.smooth > 0:
        print(f"\nApplying smoothing (strength={args.smooth}, method={args.smooth_method})...")
        
        # Smooth global orientation
        print(f"  Smoothing global orientation...")
        smpl_params['global_orient'] = smooth_rotations(
            smpl_params['global_orient'], 
            sigma=args.smooth, 
            method=args.smooth_method
        )
        
        # Smooth body pose (all 23 joints)
        print(f"  Smoothing body pose ({smpl_params['body_pose'].shape[1]//3} joints)...")
        smpl_params['body_pose'] = smooth_rotations(
            smpl_params['body_pose'], 
            sigma=args.smooth, 
            method=args.smooth_method
        )
        
        # Smooth translation
        print(f"  Smoothing translation...")
        smpl_params['transl'] = smooth_translations(
            smpl_params['transl'], 
            sigma=args.smooth, 
            method=args.smooth_method
        )
        
        print(f"  Smoothing complete!")
    
    os.makedirs(args.output, exist_ok=True)
    
    # Export single-frame PKL files (for Load Pose feature)
    print(f"\nExporting single-frame PKL files...")
    for i in range(num_frames):
        frame_data = convert_smpl_to_smplx_single_frame(
            smpl_params['body_pose'][i],
            smpl_params['global_orient'][i],
            smpl_params['betas'][i],
            smpl_params['transl'][i]
        )
        
    #     pkl_path = os.path.join(args.output, f'frame_{i:04d}.pkl')
    #     with open(pkl_path, 'wb') as f:
    #         pickle.dump(frame_data, f)
    
    # print(f"  Saved {num_frames} PKL files to {args.output}/frame_XXXX.pkl")
    
    # Export animation NPZ file (for Add Animation feature)
    print(f"\nExporting animation NPZ file...")
    anim_data = convert_smpl_to_smplx_animation(smpl_params, gender)
    anim_data['mocap_framerate'] = args.fps
    
    npz_path = os.path.join(args.output, 'animation.npz')
    np.savez(npz_path, **anim_data)
    print(f"  Saved animation to {npz_path}")
    
    print(f"\nDone! Files saved to: {args.output}")
    print(f"\nTo use in Blender:")
    print(f"  1. Install the SMPL-X addon")
    print(f"  2. Add a SMPL-X model ({gender})")
    print(f"  3. Use 'Add Animation' and select {npz_path}")
    print(f"  OR")
    print(f"  3. Use 'Load Pose' with any frame_XXXX.pkl file")


if __name__ == '__main__':
    main()
