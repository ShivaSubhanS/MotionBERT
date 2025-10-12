import bpy
import numpy as np
import mathutils
from mathutils import Vector
import os

# File paths
blend_file_path = "/home/sss/project/pose_3d/rigged_character_with_ik.blend"
pose_data_path = "/home/sss/project/pose_3d/MotionBERT/results/X3D.npy"

def load_blend_file():
    """Load the Blender file"""
    if os.path.exists(blend_file_path):
        bpy.ops.wm.open_mainfile(filepath=blend_file_path)
        print(f"✓ Successfully loaded: {blend_file_path}\n")
        return True
    else:
        print(f"✗ Error: File not found at {blend_file_path}")
        return False

def load_pose_data():
    """Load original pose data"""
    if os.path.exists(pose_data_path):
        pose_data = np.load(pose_data_path)
        print(f"✓ Loaded pose data: {pose_data.shape}")
        return pose_data
    else:
        print(f"✗ Error: Pose data not found at {pose_data_path}")
        return None

def transform_pose_to_y_axis(pose_data):
    """Transform to Y-axis forward (same as original code)"""
    num_frames, num_joints, _ = pose_data.shape
    transformed = np.zeros_like(pose_data)
    
    for frame in range(num_frames):
        for joint in range(num_joints):
            x, y, z = pose_data[frame, joint]
            x_rot, y_rot, z_rot = -x, -y, z
            transformed[frame, joint] = [x_rot, -z_rot, y_rot]
    
    return transformed

def get_bone_world_position(armature_obj, bone_name, frame):
    """Get bone head position in world space for a specific frame"""
    bpy.context.scene.frame_set(frame)
    
    if bone_name not in armature_obj.pose.bones:
        return None
    
    pose_bone = armature_obj.pose.bones[bone_name]
    bone_matrix = armature_obj.matrix_world @ pose_bone.matrix
    head_pos = bone_matrix.translation
    
    return np.array([head_pos.x, head_pos.y, head_pos.z])

def get_bone_tail_position(armature_obj, bone_name, frame):
    """Get bone tail position in world space for a specific frame"""
    bpy.context.scene.frame_set(frame)
    
    if bone_name not in armature_obj.pose.bones:
        return None
    
    pose_bone = armature_obj.pose.bones[bone_name]
    bone = armature_obj.data.bones[bone_name]
    bone_matrix = armature_obj.matrix_world @ pose_bone.matrix
    
    # Calculate tail position
    tail_offset = Vector((0, bone.length, 0))
    tail_pos = bone_matrix @ tail_offset
    
    return np.array([tail_pos.x, tail_pos.y, tail_pos.z])

def calculate_error_rate(armature_obj, pose_data):
    """Calculate error between armature bones and original pose data"""
    
    # Joint indices from original code
    ROOT, RHIP, RKNE, RANK = 0, 1, 2, 3
    LHIP, LKNE, LANK = 4, 5, 6
    BELLY, NECK, NOSE, HEAD = 7, 8, 9, 10
    LSHO, LELB, LWRI = 11, 12, 13
    RSHO, RELB, RWRI = 14, 15, 16
    
    # Mapping bone names to joint indices
    bone_to_joint_mapping = {
        'Root': (ROOT, 'head'),
        'Spine': (BELLY, 'head'),
        'Neck': (NECK, 'head'),
        'Head': (NOSE, 'head'),
        # Head end point
        'Head_end': (HEAD, 'tail'),
        
        # Left arm
        'Clavicle.L': (NECK, 'head'),
        'UpperArm.L': (LSHO, 'head'),
        'ForeArm.L': (LELB, 'head'),
        'ForeArm.L_end': (LWRI, 'tail'),
        
        # Right arm
        'Clavicle.R': (NECK, 'head'),
        'UpperArm.R': (RSHO, 'head'),
        'ForeArm.R': (RELB, 'head'),
        'ForeArm.R_end': (RWRI, 'tail'),
        
        # Left leg
        'Thigh.L': (LHIP, 'head'),
        'Shin.L': (LKNE, 'head'),
        'Shin.L_end': (LANK, 'tail'),
        
        # Right leg
        'Thigh.R': (RHIP, 'head'),
        'Shin.R': (RKNE, 'head'),
        'Shin.R_end': (RANK, 'tail'),
    }
    
    num_frames = min(pose_data.shape[0], bpy.context.scene.frame_end)
    
    print("\n" + "="*80)
    print("ERROR ANALYSIS: Armature Bones vs Original Pose Data")
    print("="*80 + "\n")
    
    total_error = 0
    total_comparisons = 0
    per_bone_errors = {}
    per_frame_errors = []
    
    # Analyze each frame
    for frame_idx in range(num_frames):
        frame_num = frame_idx + 1
        frame_error = 0
        frame_comparisons = 0
        
        for bone_key, (joint_idx, point_type) in bone_to_joint_mapping.items():
            # Get actual bone name (remove _end suffix)
            if bone_key.endswith('_end'):
                bone_name = bone_key[:-4]
                bone_pos = get_bone_tail_position(armature_obj, bone_name, frame_num)
            else:
                bone_name = bone_key
                bone_pos = get_bone_world_position(armature_obj, bone_name, frame_num)
            
            if bone_pos is None:
                continue
            
            # Get corresponding pose point
            pose_point = pose_data[frame_idx, joint_idx]
            
            # Calculate Euclidean distance (error)
            error = np.linalg.norm(bone_pos - pose_point)
            
            # Accumulate statistics
            total_error += error
            total_comparisons += 1
            frame_error += error
            frame_comparisons += 1
            
            if bone_key not in per_bone_errors:
                per_bone_errors[bone_key] = []
            per_bone_errors[bone_key].append(error)
        
        if frame_comparisons > 0:
            avg_frame_error = frame_error / frame_comparisons
            per_frame_errors.append(avg_frame_error)
    
    # Calculate overall statistics
    avg_error = total_error / total_comparisons if total_comparisons > 0 else 0
    
    print(f"Total Frames Analyzed: {num_frames}")
    print(f"Total Comparisons: {total_comparisons}")
    print(f"\n{'─'*80}")
    print(f"OVERALL AVERAGE ERROR: {avg_error:.6f} units")
    print(f"{'─'*80}\n")
    
    # Per-bone error statistics
    print("\nPER-BONE ERROR STATISTICS:")
    print(f"{'─'*80}")
    print(f"{'Bone Name':<25} {'Avg Error':>12} {'Min Error':>12} {'Max Error':>12}")
    print(f"{'─'*80}")
    
    sorted_bones = sorted(per_bone_errors.items(), key=lambda x: np.mean(x[1]), reverse=True)
    
    for bone_name, errors in sorted_bones:
        avg = np.mean(errors)
        min_err = np.min(errors)
        max_err = np.max(errors)
        print(f"{bone_name:<25} {avg:>12.6f} {min_err:>12.6f} {max_err:>12.6f}")
    
    # Per-frame error trend
    print(f"\n{'─'*80}")
    print("PER-FRAME ERROR TREND (first 10 and last 10 frames):")
    print(f"{'─'*80}")
    
    for i in range(min(10, len(per_frame_errors))):
        print(f"Frame {i+1:3d}: {per_frame_errors[i]:.6f} units")
    
    if len(per_frame_errors) > 20:
        print("...")
        for i in range(len(per_frame_errors)-10, len(per_frame_errors)):
            print(f"Frame {i+1:3d}: {per_frame_errors[i]:.6f} units")
    
    # Error distribution
    print(f"\n{'─'*80}")
    print("ERROR DISTRIBUTION:")
    print(f"{'─'*80}")
    
    all_errors = [err for errors in per_bone_errors.values() for err in errors]
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    
    for p in percentiles:
        val = np.percentile(all_errors, p)
        print(f"{p}th percentile: {val:.6f} units")
    
    print("\n" + "="*80 + "\n")
    
    return {
        'average_error': avg_error,
        'per_bone_errors': per_bone_errors,
        'per_frame_errors': per_frame_errors,
        'total_comparisons': total_comparisons
    }

def main():
    # Load Blender file
    if not load_blend_file():
        return
    
    # Load pose data
    pose_data = load_pose_data()
    if pose_data is None:
        return
    
    # Transform pose data (same transformation as original)
    pose_data = transform_pose_to_y_axis(pose_data)
    print(f"✓ Transformed pose data shape: {pose_data.shape}\n")
    
    # Find armature in scene
    armatures = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']
    
    if not armatures:
        print("✗ Error: No armature found in the scene!")
        return
    
    armature_obj = armatures[0]
    print(f"✓ Found armature: {armature_obj.name}")
    print(f"  Total bones: {len(armature_obj.data.bones)}")
    
    # Calculate error
    results = calculate_error_rate(armature_obj, pose_data)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Average positional error: {results['average_error']:.6f} units")
    print(f"Total comparisons made: {results['total_comparisons']}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()