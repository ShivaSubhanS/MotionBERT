import bpy
import mathutils
import os

# Load the Blender file
blend_file_path = "/home/sss/project/pose_3d/rigged_character_with_ik.blend"

if os.path.exists(blend_file_path):
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)
    print(f"Successfully loaded: {blend_file_path}\n")
else:
    print(f"Error: File not found at {blend_file_path}")
    exit()

def get_bone_world_position(armature_obj, bone):
    """Get the world space position of a bone's head and tail"""
    # Get the bone's matrix in world space
    if armature_obj.mode == 'EDIT':
        bone_matrix = armature_obj.matrix_world @ bone.matrix
        head = armature_obj.matrix_world @ bone.head
        tail = armature_obj.matrix_world @ bone.tail
    else:  # POSE or OBJECT mode
        if bone.name in armature_obj.pose.bones:
            pose_bone = armature_obj.pose.bones[bone.name]
            bone_matrix = armature_obj.matrix_world @ pose_bone.matrix
            # For pose bones, use the bone's length to calculate tail
            head = bone_matrix.translation
            tail = head + (bone_matrix.to_quaternion() @ mathutils.Vector((0, bone.length, 0)))
        else:
            bone_matrix = armature_obj.matrix_world @ bone.matrix_local
            head = armature_obj.matrix_world @ bone.head_local
            tail = armature_obj.matrix_world @ bone.tail_local
    
    return head, tail

def print_bone_hierarchy(armature_obj, bone, level=0):
    """Recursively print bone hierarchy with 3D positions"""
    indent = "  " * level
    
    head, tail = get_bone_world_position(armature_obj, bone)
    
    print(f"{indent}Bone: {bone.name}")
    print(f"{indent}  Head: ({head.x:.4f}, {head.y:.4f}, {head.z:.4f})")
    print(f"{indent}  Tail: ({tail.x:.4f}, {tail.y:.4f}, {tail.z:.4f})")
    
    if bone.parent:
        print(f"{indent}  Parent: {bone.parent.name}")
    else:
        print(f"{indent}  Parent: None (Root)")
    
    print(f"{indent}  Children: {len(bone.children)}")
    print()
    
    # Recursively print children
    for child in bone.children:
        print_bone_hierarchy(armature_obj, child, level + 1)

def display_armature_bones():
    """Main function to display all bones in all armatures"""
    # Find all armature objects in the scene
    armatures = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']
    
    if not armatures:
        print("Error: No armature objects found in the scene!")
        return
    
    # Process each armature
    for obj in armatures:
        process_armature(obj)

def process_armature(obj):
    """Process a single armature object"""
    
    print("=" * 60)
    print(f"ARMATURE: {obj.name}")
    print("=" * 60)
    print()
    
    # Get all root bones (bones without parents)
    root_bones = [bone for bone in obj.data.bones if bone.parent is None]
    
    print(f"Total bones: {len(obj.data.bones)}")
    print(f"Root bones: {len(root_bones)}")
    print()
    print("-" * 60)
    print()
    
    # Print hierarchy starting from each root bone
    for root_bone in root_bones:
        print_bone_hierarchy(obj, root_bone)

# Run the function
display_armature_bones()