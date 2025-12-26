#!/usr/bin/env python3
"""
Pipeline to run AlphaPose + MotionBERT for 3D pose estimation with SMPLX output
Similar to pipelined_run2.py but for the unified AlphaPose-MotionBERT container
Uses RAM-based temporary storage for efficiency
"""

import os
import subprocess
import argparse
import tempfile
import shutil

PROJECT_ROOT = os.path.abspath(".")

def run_combined_pipeline(video_file, gpu_id=0, detbatch=4, keep_outputs=True):
    """
    Run both AlphaPose and MotionBERT in a single container session.
    Uses /dev/shm (RAM) for temporary storage during processing.
    
    Args:
        video_file: Path to input video file (local file, not in container)
        gpu_id: GPU device ID to use (default: 0)
        detbatch: Detection batch size for AlphaPose (default: 4)
        keep_outputs: If True, copy results to ./outputs/ directory
    """
    
    if not os.path.exists(video_file):
        print(f"[ERROR] Video file not found: {video_file}")
        exit(1)
    
    video_basename = os.path.basename(video_file)
    print(f"[INFO] Processing video: {video_basename}")
    print(f"[INFO] Using GPU: {gpu_id}, Detection batch size: {detbatch}")
    print(f"[INFO] Using RAM-based temporary storage (/dev/shm)")
    
    # Use RAM-based temporary directory
    with tempfile.TemporaryDirectory(dir="/dev/shm") as ram_dir:
        print(f"[INFO] Temporary directory created in RAM: {ram_dir}")
        
        # Create subdirectories in RAM
        video_tmp = os.path.join(ram_dir, "videos")
        output_tmp = os.path.join(ram_dir, "outputs")
        os.makedirs(video_tmp, exist_ok=True)
        os.makedirs(output_tmp, exist_ok=True)
        
        # Copy video to RAM
        print(f"[INFO] Copying video to RAM...")
        video_tmp_path = os.path.join(video_tmp, video_basename)
        shutil.copy2(video_file, video_tmp_path)
        print(f"[INFO] Video copied to: {video_tmp_path}")
        
        # Container paths
        container_video = f"/tmp/videos/{video_basename}"
        container_outputs = "/tmp/outputs"
        
        # Combined command running both in sequence
        combined_cmd = (
            # Step 1: AlphaPose
            f"cd /workspace/AlphaPose && "
            f"python3 demo_inference.py "
            # f"--cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml "
            # f"--checkpoint pretrained_models/fast_res50_256x192.pth "
            f"--video {container_video} "
            f"--outdir {container_outputs}/alphapose_results "
            f"--detbatch {detbatch} "
            f"--gpu {gpu_id} "
            # Step 2: MotionBERT
            # f"cd /workspace/MotionBERT && "
            # f"python infer_wild_mesh.py "
            # f"--vid_path {container_video} "
            # f"--json_path {container_outputs}/alphapose_results/alphapose-results.json "
            # f"--out_path {container_outputs}/motionbert_results"
        )
        
        # Create persistent cache directory for PyTorch models
        cache_dir = os.path.expanduser("~/.cache/torch")
        os.makedirs(cache_dir, exist_ok=True)
        
        cmd = [
            "docker", "run", "--rm", "-it",
            "--gpus", f"device={gpu_id}",
            "--shm-size", "8g",  # Increase shared memory for processing
            "-v", f"{video_tmp}:/tmp/videos:ro",
            "-v", f"{output_tmp}:/tmp/outputs",
            # # Mount AlphaPose models from local machine
            # "-v", f"{os.path.expanduser('~/project/pose_3d/AlphaPose/detector/yolo/data')}:/workspace/AlphaPose/detector/yolo/data:ro",
            # "-v", f"{os.path.expanduser('~/project/pose_3d/AlphaPose/pretrained_models')}:/workspace/AlphaPose/pretrained_models:ro",
            # # Mount MotionBERT checkpoint
            # "-v", f"{os.path.join(PROJECT_ROOT, 'checkpoint')}:/workspace/MotionBERT/checkpoint:ro",
            # Mount PyTorch cache to avoid re-downloading models
            "-v", f"{cache_dir}:/root/.cache/torch",
            "alphapose-motionbert:cuda11.6",
            "bash", "-c", combined_cmd
        ]
        
        print("\n" + "="*60)
        print("[RUNNING] Combined AlphaPose + MotionBERT Pipeline")
        print("="*60)
        print(f"[CMD] {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
            print("\n" + "="*60)
            print("[PIPELINE COMPLETE]")
            print("="*60)
            
            # Copy outputs from RAM to permanent storage if requested
            if keep_outputs:
                outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
                os.makedirs(outputs_dir, exist_ok=True)
                
                print(f"\n[INFO] Copying results from RAM to: {outputs_dir}")
                
                # Copy AlphaPose results
                alphapose_src = os.path.join(output_tmp, "alphapose_results")
                alphapose_dst = os.path.join(outputs_dir, "alphapose_results")
                if os.path.exists(alphapose_src):
                    if os.path.exists(alphapose_dst):
                        shutil.rmtree(alphapose_dst)
                    shutil.copytree(alphapose_src, alphapose_dst)
                    print(f"  ✓ AlphaPose results: {alphapose_dst}")
                
                # Copy MotionBERT results
                motionbert_src = os.path.join(output_tmp, "motionbert_results")
                motionbert_dst = os.path.join(outputs_dir, "motionbert_results")
                if os.path.exists(motionbert_src):
                    if os.path.exists(motionbert_dst):
                        shutil.rmtree(motionbert_dst)
                    shutil.copytree(motionbert_src, motionbert_dst)
                    print(f"  ✓ MotionBERT results: {motionbert_dst}")
                
                print(f"\n[SUCCESS] Results saved to: {outputs_dir}")
            else:
                print(f"\n[INFO] Results in RAM (will be deleted): {output_tmp}")
            
            print("="*60)
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Pipeline failed: {e}")
            raise
        finally:
            # Fix permissions before cleanup (Docker creates files as root)
            print(f"\n[INFO] Cleaning up temporary files in RAM...")
            try:
                # Use sudo to change ownership back to current user
                subprocess.run(
                    ["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", output_tmp],
                    check=False,  # Don't fail if sudo not available
                    stderr=subprocess.DEVNULL
                )
            except Exception:
                pass  # Ignore cleanup errors
        
        # tempfile will automatically clean up when exiting the context


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AlphaPose + MotionBERT pipeline for 3D pose estimation with SMPLX"
    )
    parser.add_argument(
        "--video", 
        required=True, 
        help="Path to input video file (e.g., /path/to/video.mp4 or smallgirl.mp4)"
    )
    parser.add_argument(
        "--gpu", 
        type=int, 
        default=0, 
        help="GPU device ID (default: 0)"
    )
    parser.add_argument(
        "--detbatch", 
        type=int, 
        default=4, 
        help="Detection batch size for AlphaPose (default: 4)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save outputs to disk (keep only in RAM during processing)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("AlphaPose + MotionBERT Pipeline")
    print("="*60)
    
    run_combined_pipeline(
        args.video, 
        args.gpu, 
        args.detbatch, 
        keep_outputs=not args.no_save
    )
