#!/bin/bash
# Script to create animated SMPL-X mesh from MotionBERT output
#
# Usage:
#   ./run_smplx_animation.sh [options]
#
# Options:
#   -i, --input       Input smpl_params.pkl from MotionBERT (default: results/smpl_params.pkl)
#   -o, --output      Output directory (default: smplx_output)
#   --fps             Animation framerate (default: 30)
#   --gender          SMPL-X gender: female, male, neutral (default: female)
#   --smplx-betas     Path to SMPLify-X pkl file to use its body shape (betas)
#   --export-glb      Also export GLB file for web viewing
#   --blender         Path to Blender executable
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
INPUT_PKL="$SCRIPT_DIR/results/smpl_params.pkl"
OUTPUT_DIR="$SCRIPT_DIR/smplx_output"
FPS=30
GENDER="female"
SMPLX_BETAS=""
EXPORT_GLB=""
BLENDER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_PKL="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --gender)
            GENDER="$2"
            shift 2
            ;;
        --smplx-betas)
            SMPLX_BETAS="$2"
            shift 2
            ;;
        --export-glb)
            EXPORT_GLB="--export-glb"
            shift
            ;;
        --blender)
            BLENDER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -i, --input       Input smpl_params.pkl from MotionBERT"
            echo "  -o, --output      Output directory"
            echo "  --fps             Animation framerate (default: 30)"
            echo "  --gender          SMPL-X gender: female, male, neutral (default: female)"
            echo "  --smplx-betas     Path to SMPLify-X pkl to use its body shape (weight/height)"
            echo "  --export-glb      Also export GLB file for web viewing"
            echo "  --blender         Path to Blender executable"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Find Blender if not specified
if [ -z "$BLENDER" ]; then
    if command -v blender &> /dev/null; then
        BLENDER=$(which blender)
    elif [ -f "$HOME/Downloads/blender-4.5.2-linux-x64/blender" ]; then
        BLENDER="$HOME/Downloads/blender-4.5.2-linux-x64/blender"
    elif [ -n "$(find $HOME/Downloads -maxdepth 3 -name "blender" -type f -executable 2>/dev/null | head -1)" ]; then
        BLENDER=$(find $HOME/Downloads -maxdepth 3 -name "blender" -type f -executable 2>/dev/null | head -1)
    elif [ -f "/usr/bin/blender" ]; then
        BLENDER="/usr/bin/blender"
    elif [ -f "/snap/bin/blender" ]; then
        BLENDER="/snap/bin/blender"
    fi
fi

if [ -z "$BLENDER" ]; then
    echo "Error: Blender not found. Please specify path with --blender"
    exit 1
fi

echo "Using Blender: $BLENDER"

# Check input file
if [ ! -f "$INPUT_PKL" ]; then
    echo "Error: Input file not found: $INPUT_PKL"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "MotionBERT to SMPL-X Animation Pipeline"
echo "=========================================="
echo "Input:  $INPUT_PKL"
echo "Output: $OUTPUT_DIR"
echo "Gender: $GENDER"
echo "FPS:    $FPS"
if [ -n "$SMPLX_BETAS" ]; then
    echo "Body Shape: $SMPLX_BETAS"
fi
echo "=========================================="

# Step 1: Convert SMPL to SMPL-X format
echo ""
echo "Step 1: Converting SMPL to SMPL-X format..."

BETAS_ARG=""
if [ -n "$SMPLX_BETAS" ]; then
    BETAS_ARG="--smplx-betas $SMPLX_BETAS"
fi

python3 "$SCRIPT_DIR/convert_smpl_to_smplx.py" \
    --input "$INPUT_PKL" \
    --output "$OUTPUT_DIR" \
    --fps "$FPS" \
    --gender "$GENDER" \
    $BETAS_ARG

# Step 2: Create Blender animation
echo ""
echo "Step 2: Creating Blender animation..."
NPZ_FILE="$OUTPUT_DIR/animation.npz"
BLEND_FILE="$OUTPUT_DIR/animation.blend"

"$BLENDER" --background --python "$SCRIPT_DIR/create_smplx_animation.py" -- \
    --input "$NPZ_FILE" \
    --output "$BLEND_FILE" \
    --gender "$GENDER" \
    --fps "$FPS" \
    $EXPORT_GLB

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - Animation NPZ:  $OUTPUT_DIR/animation.npz"
echo "  - Blender file:   $OUTPUT_DIR/animation.blend"
if [ -n "$EXPORT_GLB" ]; then
    echo "  - GLB file:       $OUTPUT_DIR/animation.glb"
fi
echo "  - Frame PKLs:     $OUTPUT_DIR/frame_XXXX.pkl"
echo ""
echo "To view in Blender:"
echo "  $BLENDER $BLEND_FILE"
echo ""
echo "Or use the SMPL-X addon manually:"
echo "  1. Open Blender"
echo "  2. Add SMPL-X model (gender: $GENDER)"
echo "  3. Use 'Add Animation' with: $NPZ_FILE"
