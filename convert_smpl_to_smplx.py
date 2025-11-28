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
from scipy.spatial.transform import Rotation as R

# SMPL joint order (24 joints, but body_pose has 23 excluding root):
# 0: pelvis (root - stored in global_orient)
# 1: left_hip, 2: right_hip, 3: spine1, 4: left_knee, 5: right_knee
# 6: spine2, 7: left_ankle, 8: right_ankle, 9: spine3, 10: left_foot
# 11: right_foot, 12: neck, 13: left_collar, 14: right_collar, 15: head
# 16: left_shoulder, 17: right_shoulder, 18: left_elbow, 19: right_elbow
# 20: left_wrist, 21: right_wrist, 22: left_hand, 23: right_hand

# SMPL-X body joint order (21 joints):
# Similar to SMPL but excludes hand joints (22, 23)


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


def convert_smpl_to_smplx_animation(smpl_params):
    """
    Convert full animation sequence for use with SMPL-X Blender addon's 
    "Add Animation" feature (SMPL-X format .npz).
    
    Returns data in AMASS/SMPL-X animation format.
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
        'gender': 'female',  # Using female as requested
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
    parser.add_argument('--gender', default='female', choices=['female', 'male', 'neutral'], 
                        help='Gender for SMPL-X model')
    parser.add_argument('--smplx-betas', type=str, default=None,
                        help='Path to SMPLify-X pkl file to use its betas for accurate body shape')
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
    
    # Override betas with SMPLify-X betas if provided
    if smplx_betas is not None:
        print(f"\n*** Using SMPLify-X betas for accurate body shape ***")
        # Broadcast SMPLify-X betas to all frames (first 10 components)
        smpl_params['betas'] = np.tile(smplx_betas[:10], (num_frames, 1))
        print(f"  Overriding betas with SMPLify-X shape: {smpl_params['betas'].shape}")
    
    # Fix orientation flips (180° sudden rotations)
    print(f"\nFixing orientation flips...")
    smpl_params['global_orient'] = fix_orientation_flips(smpl_params['global_orient'], threshold_deg=120)
    
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
        
        pkl_path = os.path.join(args.output, f'frame_{i:04d}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(frame_data, f)
    
    print(f"  Saved {num_frames} PKL files to {args.output}/frame_XXXX.pkl")
    
    # Export animation NPZ file (for Add Animation feature)
    print(f"\nExporting animation NPZ file...")
    anim_data = convert_smpl_to_smplx_animation(smpl_params)
    anim_data['gender'] = args.gender
    anim_data['mocap_framerate'] = args.fps
    
    npz_path = os.path.join(args.output, 'animation.npz')
    np.savez(npz_path, **anim_data)
    print(f"  Saved animation to {npz_path}")
    
    print(f"\nDone! Files saved to: {args.output}")
    print(f"\nTo use in Blender:")
    print(f"  1. Install the SMPL-X addon")
    print(f"  2. Add a SMPL-X model (female)")
    print(f"  3. Use 'Add Animation' and select {npz_path}")
    print(f"  OR")
    print(f"  3. Use 'Load Pose' with any frame_XXXX.pkl file")


if __name__ == '__main__':
    main()
