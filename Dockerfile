# Use CUDA 11.6 base image with Ubuntu 20.04 (Python 3.8)
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies and Python 3.8 (default for Ubuntu 20.04)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-tk \
    gcc-10 \
    g++-10 \
    libyaml-dev \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set up Python 3 as default python command
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# ==================== ARUN mkdir -p detector/yolo/data pretrained_modelslphaPose Setup ====================
# Clone AlphaPose repository
RUN git clone https://github.com/ShivaSubhanS/AlphaPose.git /workspace/AlphaPose

WORKDIR /workspace/AlphaPose

# Install PyTorch first (needed by AlphaPose)
RUN pip install --no-cache-dir \
    torch==1.13.1+cu116 \
    torchvision==0.14.1+cu116 \
    torchaudio==0.13.1+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116

# Install Cython and other AlphaPose build dependencies
RUN pip install --no-cache-dir \
    Cython==3.1.4 \
    numpy==1.24.3 \
    cython-bbox

# Build AlphaPose with specific gcc/g++ versions
# Set TORCH_CUDA_ARCH_LIST to common GPU architectures (no GPU detection during build)
RUN TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6" CXX=g++-10 CC=gcc-10 python setup.py build_ext --inplace

# Create directories for models
RUN mkdir -p detector/yolo/data pretrained_models

# Copy AlphaPose models (these will be mounted or copied)
# YOLO weights should be at: detector/yolo/data/yolov3-spp.weights
# Pretrained models should be at: pretrained_models/fast_res50_256x192.pth

# ==================== MotionBERT Setup ====================
# Clone MotionBERT repository
RUN git clone -b smplx-test https://github.com/ShivaSubhanS/MotionBERT.git /workspace/MotionBERT

WORKDIR /workspace/MotionBERT

# Install MotionBERT requirements (minimal set for SMPLX)
COPY MotionBERT/requirements_minimal.txt /workspace/MotionBERT/

# Install PyTorch with CUDA 11.6 from specific index, then other requirements
RUN pip install --no-cache-dir -r requirements_minimal.txt

# Create checkpoint directory (will be mounted or copied)
RUN mkdir -p checkpoint

# Set default working directory
WORKDIR /workspace

# ==================== Install vim ====================
RUN apt-get update && apt-get install -y vim && rm -rf /var/lib/apt/lists/*

# ==================== Copy Models into Image ====================
# Copy MotionBERT checkpoint
COPY MotionBERT/checkpoint /workspace/MotionBERT/checkpoint

# Copy AlphaPose models (using relative paths from build context)
# YOLO weights
COPY AlphaPose/detector/yolo/data /workspace/AlphaPose/detector/yolo/data
# Pretrained models
COPY AlphaPose/pretrained_models /workspace/AlphaPose/pretrained_models

# Default command
CMD ["/bin/bash"]
