#!/bin/bash

# Build script for AlphaPose + MotionBERT Docker image

set -e

echo "Building AlphaPose + MotionBERT Docker image..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if nvidia-docker is available
if ! docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-docker runtime not detected. GPU support may not work."
    echo "Install nvidia-docker2 if you haven't already:"
    echo "  sudo apt-get install nvidia-docker2"
    echo "  sudo systemctl restart docker"
    echo ""
fi

# Build the image
docker build -t alphapose-motionbert:cuda11.6 -f Dockerfile ..

echo "Build completed successfully!"
