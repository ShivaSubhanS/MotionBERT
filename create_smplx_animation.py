#!/usr/bin/env python3
"""
Blender script to create animated SMPL-X mesh from MotionBERT output.

This script is meant to be run from Blender:
    blender --background --python create_smplx_animation.py -- <args>
    
Or interactively in Blender's Python console.
"""

import bpy
import sys
import os
import numpy as np
from math import radians
from mathutils import Vector, Quaternion

# Get arguments passed after '--'
argv = sys.argv
if '--' in argv:
    argv = argv[argv.index('--') + 1:]
else:
    argv = []

# Default paths (can be overridden by command line args)
DEFAULT_NPZ_PATH = "smplx_output/animation.npz"
DEFAULT_OUTPUT_PATH = "smplx_output/animation.blend"
DEFAULT_GENDER = "female"
DEFAULT_FPS = 30


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Create SMPL-X animation in Blender')
    parser.add_argument('-i', '--input', default=DEFAULT_NPZ_PATH, 
                        help='Input NPZ animation file')
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT_PATH,
                        help='Output .blend file path')
    parser.add_argument('--gender', default=DEFAULT_GENDER,
                        choices=['female', 'male', 'neutral'],
                        help='SMPL-X model gender')
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS,
                        help='Output framerate')
    parser.add_argument('--export-glb', action='store_true',
                        help='Also export as GLB for web viewing')
    return parser.parse_args(argv)


# SMPL-X joint names (55 joints)
SMPLX_JOINT_NAMES = [
    'pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2',
    'left_ankle','right_ankle','spine3','left_foot','right_foot','neck',
    'left_collar','right_collar','head','left_shoulder','right_shoulder',
    'left_elbow','right_elbow','left_wrist','right_wrist',
    'jaw','left_eye_smplhf','right_eye_smplhf',
    'left_index1','left_index2','left_index3','left_middle1','left_middle2',
    'left_middle3','left_pinky1','left_pinky2','left_pinky3','left_ring1',
    'left_ring2','left_ring3','left_thumb1','left_thumb2','left_thumb3',
    'right_index1','right_index2','right_index3','right_middle1','right_middle2',
    'right_middle3','right_pinky1','right_pinky2','right_pinky3','right_ring1',
    'right_ring2','right_ring3','right_thumb1','right_thumb2','right_thumb3'
]

NUM_SMPLX_JOINTS = len(SMPLX_JOINT_NAMES)
NUM_SMPLX_BODYJOINTS = 21
NUM_SMPLX_HANDJOINTS = 15


def set_pose_from_rodrigues(armature, bone_name, rodrigues):
    """Set bone rotation from rodrigues (axis-angle) vector."""
    rod = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
    angle_rad = rod.length
    
    if angle_rad < 1e-8:
        axis = Vector((1, 0, 0))
        angle_rad = 0
    else:
        axis = rod.normalized()
    
    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'
    
    quat = Quaternion(axis, angle_rad)
    armature.pose.bones[bone_name].rotation_quaternion = quat


def create_smplx_animation(npz_path, output_path, gender='female', fps=30, export_glb=False):
    """Create SMPL-X animated mesh in Blender from NPZ file."""
    
    print(f"Loading animation from: {npz_path}")
    
    # Load animation data
    with np.load(npz_path) as data:
        trans = data['trans']
        betas = data['betas']
        poses = data['poses']
        mocap_fps = int(data['mocap_framerate']) if 'mocap_framerate' in data else fps
        gender = str(data['gender']) if 'gender' in data else gender
    
    num_frames = poses.shape[0]
    print(f"Loaded {num_frames} frames at {mocap_fps} fps, gender: {gender}")
    
    # Clear existing scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Check if SMPL-X addon is available
    if not hasattr(bpy.ops.scene, 'smplx_add_gender'):
        print("ERROR: SMPL-X addon not installed!")
        print("Please install the SMPL-X Blender addon from:")
        print("  https://smpl-x.is.tue.mpg.de/")
        
        # Create a simple fallback armature for testing
        print("\nCreating fallback skeleton armature...")
        create_fallback_armature(poses, trans, fps, num_frames)
        bpy.ops.wm.save_as_mainfile(filepath=output_path)
        print(f"Saved (fallback) to: {output_path}")
        return
    
    # Set up scene
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = num_frames
    
    # Add SMPL-X model
    print(f"Adding SMPL-X {gender} model...")
    bpy.context.window_manager.smplx_tool.smplx_gender = gender
    bpy.context.window_manager.smplx_tool.smplx_handpose = "flat"
    bpy.ops.scene.smplx_add_gender()
    
    obj = bpy.context.view_layer.objects.active
    armature = obj.parent
    
    # Set shape parameters
    print("Setting shape parameters...")
    bpy.ops.object.mode_set(mode='OBJECT')
    for index, beta in enumerate(betas[:10]):
        key_block_name = f"Shape{index:03}"
        if key_block_name in obj.data.shape_keys.key_blocks:
            obj.data.shape_keys.key_blocks[key_block_name].value = float(beta)
    
    # Update joint locations for shape
    bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')
    
    # Keyframe animation
    print(f"Creating keyframes for {num_frames} frames...")
    
    for frame_idx in range(num_frames):
        if frame_idx % 50 == 0:
            print(f"  Processing frame {frame_idx}/{num_frames}")
        
        current_frame = frame_idx + 1
        current_pose = poses[frame_idx].reshape(-1, 3)
        current_trans = trans[frame_idx]
        
        for bone_index, bone_name in enumerate(SMPLX_JOINT_NAMES):
            if bone_index >= current_pose.shape[0]:
                break
                
            if bone_name == "pelvis":
                # Keyframe pelvis location (translation)
                armature.pose.bones[bone_name].location = Vector(
                    (current_trans[0], current_trans[1], current_trans[2])
                )
                armature.pose.bones[bone_name].keyframe_insert('location', frame=current_frame)
            
            # Keyframe bone rotation
            pose_rodrigues = current_pose[bone_index]
            set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)
            armature.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=current_frame)
    
    # Apply AMASS ground plane correction
    print("Applying ground plane correction...")
    bone_name = "root"
    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'
    armature.pose.bones[bone_name].rotation_quaternion = Quaternion((1.0, 0.0, 0.0), radians(-90))
    armature.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=1)
    
    # Add a simple light and camera
    print("Setting up scene lighting and camera...")
    setup_scene_lighting()
    
    # Save .blend file
    print(f"Saving to: {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output_path)
    
    # Export GLB if requested
    if export_glb:
        glb_path = output_path.replace('.blend', '.glb')
        print(f"Exporting GLB to: {glb_path}")
        bpy.ops.export_scene.gltf(
            filepath=glb_path,
            export_format='GLB',
            export_animations=True
        )
    
    print("Done!")


def create_fallback_armature(poses, trans, fps, num_frames):
    """Create a simple armature animation when SMPL-X addon is not available."""
    
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = num_frames
    
    # Create armature
    bpy.ops.object.armature_add(enter_editmode=True)
    armature = bpy.context.object
    armature.name = "SMPLX_Fallback"
    
    # Add bones for main joints
    arm_data = armature.data
    
    # Create a simple spine
    bones_to_create = ['pelvis', 'spine', 'neck', 'head']
    parent_bone = None
    
    for i, bone_name in enumerate(bones_to_create):
        if i == 0:
            bone = arm_data.edit_bones[0]
            bone.name = bone_name
        else:
            bone = arm_data.edit_bones.new(bone_name)
            bone.parent = parent_bone
            bone.head = parent_bone.tail
        
        bone.tail = bone.head + Vector((0, 0, 0.2))
        parent_bone = bone
    
    bpy.ops.object.mode_set(mode='POSE')
    
    # Keyframe pelvis location
    pelvis = armature.pose.bones['pelvis']
    for frame_idx in range(num_frames):
        current_frame = frame_idx + 1
        pelvis.location = Vector((trans[frame_idx][0], trans[frame_idx][1], trans[frame_idx][2]))
        pelvis.keyframe_insert('location', frame=current_frame)


def setup_scene_lighting():
    """Add basic lighting to the scene."""
    # Add sun light
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.object
    sun.data.energy = 3.0
    
    # Add camera
    bpy.ops.object.camera_add(location=(0, -5, 1.5))
    camera = bpy.context.object
    camera.rotation_euler = (radians(80), 0, 0)
    bpy.context.scene.camera = camera


def main():
    args = parse_args()
    
    # Make paths absolute
    if not os.path.isabs(args.input):
        args.input = os.path.abspath(args.input)
    if not os.path.isabs(args.output):
        args.output = os.path.abspath(args.output)
    
    print("=" * 60)
    print("SMPL-X Animation Creator for Blender")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Gender: {args.gender}")
    print(f"FPS:    {args.fps}")
    print("=" * 60)
    
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    create_smplx_animation(
        args.input, 
        args.output, 
        gender=args.gender,
        fps=args.fps,
        export_glb=args.export_glb
    )


if __name__ == "__main__":
    main()
