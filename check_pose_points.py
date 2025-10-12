import numpy as np

# Load pose data
pose_data = np.load("/home/sss/project/pose_3d/MotionBERT/results/X3D.npy")
first_frame = pose_data[0]

# Transform to Y-axis (same as in create_riggable_armature.py)
def transform_pose_to_y_axis(pose_data):
    num_frames, num_joints, _ = pose_data.shape
    transformed = np.zeros_like(pose_data)
    
    for frame in range(num_frames):
        for joint in range(num_joints):
            x, y, z = pose_data[frame, joint]
            x_rot, y_rot, z_rot = -x, -y, z
            transformed[frame, joint] = [x_rot, -z_rot, y_rot]
    
    return transformed

transformed = transform_pose_to_y_axis(pose_data)
first_frame_transformed = transformed[0]

# Joint indices
ROOT, RHIP, RKNE, RANK = 0, 1, 2, 3
LHIP, LKNE, LANK = 4, 5, 6
BELLY, NECK, NOSE, HEAD = 7, 8, 9, 10
LSHO, LELB, LWRI = 11, 12, 13
RSHO, RELB, RWRI = 14, 15, 16

joint_names = ['ROOT', 'RHIP', 'RKNE', 'RANK', 'LHIP', 'LKNE', 'LANK',
               'BELLY', 'NECK', 'NOSE', 'HEAD', 'LSHO', 'LELB', 'LWRI',
               'RSHO', 'RELB', 'RWRI']

print("="*60)
print("POSE DATA - First Frame (Transformed)")
print("="*60)
for i, name in enumerate(joint_names):
    pos = first_frame_transformed[i]
    print(f"{i:2d}. {name:6s}: ({pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f})")

print("\n" + "="*60)
print("EXPECTED BONE STRUCTURE")
print("="*60)
print(f"Root:       head={first_frame_transformed[ROOT]}, tail={first_frame_transformed[BELLY]}")
print(f"Spine:      head={first_frame_transformed[BELLY]}, tail={first_frame_transformed[NECK]}")
print(f"Neck:       head={first_frame_transformed[NECK]}, tail={first_frame_transformed[NOSE]}")
print(f"Head:       head={first_frame_transformed[NOSE]}, tail={first_frame_transformed[HEAD]}")
print(f"\nShoulder.L: head={first_frame_transformed[LSHO]}, tail={first_frame_transformed[LELB]}")
print(f"ForeArm.L:  head={first_frame_transformed[LELB]}, tail={first_frame_transformed[LWRI]}")
print(f"\nShoulder.R: head={first_frame_transformed[RSHO]}, tail={first_frame_transformed[RELB]}")
print(f"ForeArm.R:  head={first_frame_transformed[RELB]}, tail={first_frame_transformed[RWRI]}")
print(f"\nThigh.L:    head={first_frame_transformed[LHIP]}, tail={first_frame_transformed[LKNE]}")
print(f"Shin.L:     head={first_frame_transformed[LKNE]}, tail={first_frame_transformed[LANK]}")
print(f"\nThigh.R:    head={first_frame_transformed[RHIP]}, tail={first_frame_transformed[RKNE]}")
print(f"Shin.R:     head={first_frame_transformed[RKNE]}, tail={first_frame_transformed[RANK]}")

print("\n" + "="*60)
print("COMPARISON WITH ARMATURE")
print("="*60)
print("\nArmature Root head:       (-0.0007, -0.0000, -0.0035)")
print(f"Expected ROOT:            {first_frame_transformed[ROOT]}")
print(f"\nArmature Shoulder.L head: (-0.1816, 0.0840, 0.4750)")
print(f"Expected LSHO:            {first_frame_transformed[LSHO]}")
print(f"\nArmature Shoulder.R head: (0.1303, 0.0717, 0.5127)")
print(f"Expected RSHO:            {first_frame_transformed[RSHO]}")
