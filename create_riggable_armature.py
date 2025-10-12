"""
PROPER SOLUTION: Create armature that can be skinned to a character mesh.

The approach:
1. Create base armature from first frame (like before)
2. For each frame, calculate bone rotations using proper IK
3. Use Blender's built-in IK solver or implement CCD (Cyclic Coordinate Descent)
4. Accept some error but keep it minimal
5. Result: Real armature that can have mesh weights painted to it

This gives you a TRUE rigged character that any mesh can be parented to.
"""

import bpy
import numpy as np
import mathutils
from mathutils import Vector, Quaternion, Matrix
import math

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def load_pose_data():
    pose_data = np.load("/home/sss/project/pose_3d/MotionBERT/results/X3D.npy")
    return pose_data

def transform_pose_to_y_axis(pose_data):
    """Transform to Y-axis forward"""
    num_frames, num_joints, _ = pose_data.shape
    transformed = np.zeros_like(pose_data)
    
    for frame in range(num_frames):
        for joint in range(num_joints):
            x, y, z = pose_data[frame, joint]
            x_rot, y_rot, z_rot = -x, -y, z
            transformed[frame, joint] = [x_rot, -z_rot, y_rot]
    
    return transformed

def create_base_armature(first_frame, armature_name="RiggableArmature"):
    """Create base armature from first frame"""
    
    ROOT, RHIP, RKNE, RANK = 0, 1, 2, 3
    LHIP, LKNE, LANK = 4, 5, 6
    BELLY, NECK, NOSE, HEAD = 7, 8, 9, 10
    LSHO, LELB, LWRI = 11, 12, 13
    RSHO, RELB, RWRI = 14, 15, 16
    
    # Create armature
    armature = bpy.data.armatures.new(armature_name)
    armature_obj = bpy.data.objects.new(armature_name, armature)
    bpy.context.collection.objects.link(armature_obj)
    bpy.context.view_layer.objects.active = armature_obj
    
    # Enter edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    
    def create_bone(name, head_pos, tail_pos, parent=None):
        bone = armature.edit_bones.new(name)
        bone.head = Vector(head_pos)
        bone.tail = Vector(tail_pos)
        if parent:
            bone.parent = parent
        return bone
    
    # Create all bones - ensure bone heads match pose points exactly
    root = create_bone('Root', first_frame[ROOT], first_frame[BELLY])
    spine = create_bone('Spine', first_frame[BELLY], first_frame[NECK], root)
    neck = create_bone('Neck', first_frame[NECK], first_frame[NOSE], spine)
    head = create_bone('Head', first_frame[NOSE], first_frame[HEAD], neck)
    
    # Arms - connect shoulders from neck to shoulder joint, then to elbow
    l_clavicle = create_bone('Clavicle.L', first_frame[NECK], first_frame[LSHO], neck)
    l_upper_arm = create_bone('UpperArm.L', first_frame[LSHO], first_frame[LELB], l_clavicle)
    l_forearm = create_bone('ForeArm.L', first_frame[LELB], first_frame[LWRI], l_upper_arm)
    
    r_clavicle = create_bone('Clavicle.R', first_frame[NECK], first_frame[RSHO], neck)
    r_upper_arm = create_bone('UpperArm.R', first_frame[RSHO], first_frame[RELB], r_clavicle)
    r_forearm = create_bone('ForeArm.R', first_frame[RELB], first_frame[RWRI], r_upper_arm)
    
    # Legs - bone heads match hip joint positions
    l_thigh = create_bone('Thigh.L', first_frame[LHIP], first_frame[LKNE], root)
    l_shin = create_bone('Shin.L', first_frame[LKNE], first_frame[LANK], l_thigh)
    
    r_thigh = create_bone('Thigh.R', first_frame[RHIP], first_frame[RKNE], root)
    r_shin = create_bone('Shin.R', first_frame[RKNE], first_frame[RANK], r_thigh)
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    return armature_obj

def add_ik_constraints(armature_obj, pose_data):
    """
    Add IK constraints to the armature with target empties.
    This lets Blender's IK solver handle the complex math.
    """
    
    ROOT, RHIP, RKNE, RANK = 0, 1, 2, 3
    LHIP, LKNE, LANK = 4, 5, 6
    BELLY, NECK, NOSE, HEAD = 7, 8, 9, 10
    LSHO, LELB, LWRI = 11, 12, 13
    RSHO, RELB, RWRI = 14, 15, 16
    
    num_frames = pose_data.shape[0]
    
    # Create target empties for end effectors
    targets = {}
    
    end_effectors = {
        'Head': HEAD,
        'Hand.L': LWRI,
        'Hand.R': RWRI,
        'Foot.L': LANK,
        'Foot.R': RANK,
    }
    
    for name, joint_idx in end_effectors.items():
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=pose_data[0, joint_idx])
        empty = bpy.context.active_object
        empty.name = f"IK_Target_{name}"
        empty.empty_display_size = 0.05
        targets[name] = empty
    
    # Animate targets through all frames
    for frame_idx in range(num_frames):
        bpy.context.scene.frame_set(frame_idx + 1)
        
        for name, joint_idx in end_effectors.items():
            targets[name].location = pose_data[frame_idx, joint_idx]
            targets[name].keyframe_insert(data_path="location", frame=frame_idx + 1)
    
    # Add IK constraints to bones
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')
    
    # Head IK
    if 'Head' in armature_obj.pose.bones:
        head_bone = armature_obj.pose.bones['Head']
        ik = head_bone.constraints.new('IK')
        ik.target = targets['Head']
        ik.chain_count = 3  # Head, Neck, Spine
    
    # Left Arm IK
    if 'ForeArm.L' in armature_obj.pose.bones:
        forearm_l = armature_obj.pose.bones['ForeArm.L']
        ik = forearm_l.constraints.new('IK')
        ik.target = targets['Hand.L']
        ik.chain_count = 3  # ForeArm, UpperArm, Clavicle
    
    # Right Arm IK
    if 'ForeArm.R' in armature_obj.pose.bones:
        forearm_r = armature_obj.pose.bones['ForeArm.R']
        ik = forearm_r.constraints.new('IK')
        ik.target = targets['Hand.R']
        ik.chain_count = 3  # ForeArm, UpperArm, Clavicle
    
    # Left Leg IK
    if 'Shin.L' in armature_obj.pose.bones:
        shin_l = armature_obj.pose.bones['Shin.L']
        ik = shin_l.constraints.new('IK')
        ik.target = targets['Foot.L']
        ik.chain_count = 2  # Shin, Thigh
    
    # Right Leg IK
    if 'Shin.R' in armature_obj.pose.bones:
        shin_r = armature_obj.pose.bones['Shin.R']
        ik = shin_r.constraints.new('IK')
        ik.target = targets['Foot.R']
        ik.chain_count = 2  # Shin, Thigh
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    return targets

def bake_animation(armature_obj, start_frame, end_frame):
    """Bake IK animation to keyframes"""
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')
    
    # Select all bones
    for bone in armature_obj.pose.bones:
        bone.bone.select = True
    
    # Bake action
    bpy.ops.nla.bake(
        frame_start=start_frame,
        frame_end=end_frame,
        only_selected=True,
        visual_keying=True,
        clear_constraints=True,
        bake_types={'POSE'}
    )
    
    bpy.ops.object.mode_set(mode='OBJECT')

def create_example_character_mesh(armature_obj):
    """
    Create a simple humanoid mesh for demonstration.
    This shows how to parent a mesh to the armature.
    """
    # Create a simple humanoid using primitives
    # Smaller sizes to better match the armature scale (reduced by 20%)
    
    body_parts = {
        'Body': (0, -0.1, 0, 0.12, 0.20, 0.064),      # Torso
        'Head': (0, -0.55, 0, 0.064, 0.064, 0.064),   # Head
        'Arm.L': (0.2, -0.15, 0, 0.032, 0.12, 0.032),  # Left arm
        'Arm.R': (-0.2, -0.15, 0, 0.032, 0.12, 0.032), # Right arm
        'Leg.L': (0.08, 0.25, 0, 0.032, 0.20, 0.032),  # Left leg
        'Leg.R': (-0.08, 0.25, 0, 0.032, 0.20, 0.032), # Right leg
    }
    
    mesh_objects = []
    
    for name, (x, y, z, sx, sy, sz) in body_parts.items():
        bpy.ops.mesh.primitive_cube_add(location=(x, y, z), scale=(sx, sy, sz))
        obj = bpy.context.active_object
        obj.name = f"Mesh_{name}"
        mesh_objects.append(obj)
    
    # Join all parts into one mesh
    bpy.context.view_layer.objects.active = mesh_objects[0]
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.ops.object.join()
    
    character_mesh = bpy.context.active_object
    character_mesh.name = "CharacterMesh"
    
    # Parent mesh to armature with automatic weights
    bpy.ops.object.select_all(action='DESELECT')
    character_mesh.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    
    return character_mesh

def main():
    clear_scene()
    
    pose_data = load_pose_data()
    pose_data = transform_pose_to_y_axis(pose_data)
    
    armature_obj = create_base_armature(pose_data[0])
    targets = add_ik_constraints(armature_obj, pose_data)
    
    num_frames = pose_data.shape[0]
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = num_frames
    
    bake_animation(armature_obj, 1, num_frames)
    
    for target in targets.values():
        bpy.data.objects.remove(target, do_unlink=True)
    
    character_mesh = create_example_character_mesh(armature_obj)
    
    output_fbx = "/home/sss/project/pose_3d/rigged_character_with_ik.fbx"
    output_blend = "/home/sss/project/pose_3d/rigged_character_with_ik.blend"
    
    bpy.ops.wm.save_as_mainfile(filepath=output_blend)
    
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        use_selection=False,
        object_types={'ARMATURE', 'MESH'},
        bake_anim=True,
        add_leaf_bones=False,
    )

if __name__ == "__main__":
    main()
