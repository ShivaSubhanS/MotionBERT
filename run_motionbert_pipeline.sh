#!/bin/bash
# Simple script to create riggable armature from MotionBERT pose data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSE_DATA="$SCRIPT_DIR/results/X3D.npy"

# Find Blender
BLENDER=""
if command -v blender &> /dev/null; then
    BLENDER=$(which blender)
elif [ -f "$HOME/Downloads/blender-4.5.2-linux-x64/blender" ]; then
    BLENDER="$HOME/Downloads/blender-4.5.2-linux-x64/blender"
elif [ -n "$(find $HOME/Downloads -maxdepth 3 -name "blender" -type f -executable 2>/dev/null | head -1)" ]; then
    BLENDER=$(find $HOME/Downloads -maxdepth 3 -name "blender" -type f -executable 2>/dev/null | head -1)
fi

if [ -z "$BLENDER" ]; then
    echo "Error: Blender not found"
    exit 1
fi

if [ ! -f "$POSE_DATA" ]; then
    echo "Error: Pose data not found at $POSE_DATA"
    exit 1
fi

# Create riggable armature
"$BLENDER" --background --python "$SCRIPT_DIR/create_riggable_armature.py" > /dev/null 2>&1

exit 0
