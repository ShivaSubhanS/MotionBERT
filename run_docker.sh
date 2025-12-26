#!/bin/bash

# Run script for AlphaPose + MotionBERT Docker container

set -e

IMAGE_NAME="alphapose-motionbert:cuda11.6"
CONTAINER_NAME="alphapose-motionbert"

# Check if image exists
if ! docker image inspect $IMAGE_NAME &> /dev/null; then
    echo "Error: Docker image '$IMAGE_NAME' not found."
    echo "Please build the image first using:"
    echo "  ./build_docker.sh"
    exit 1
fi

# Check if container is already running
if docker ps | grep -q $CONTAINER_NAME; then
    echo "Container is already running. Attaching to it..."
    docker exec -it $CONTAINER_NAME bash
    exit 0
fi

# Check if container exists but is stopped
if docker ps -a | grep -q $CONTAINER_NAME; then
    echo "Starting existing container..."
    docker start $CONTAINER_NAME
    docker exec -it $CONTAINER_NAME bash
    exit 0
fi

echo "Starting new container..."
docker run --gpus all -it \
  --name $CONTAINER_NAME \
  -v $(pwd)/checkpoint:/workspace/MotionBERT/checkpoint:ro \
  -v ~/project/pose_3d/AlphaPose/detector/yolo/data:/workspace/AlphaPose/detector/yolo/data:ro \
  -v ~/project/pose_3d/AlphaPose/pretrained_models:/workspace/AlphaPose/pretrained_models:ro \
  -v $(pwd)/results:/workspace/MotionBERT/results \
  -v $(pwd)/smplx_output:/workspace/MotionBERT/smplx_output \
  --shm-size 8g \
  $IMAGE_NAME \
  bash
