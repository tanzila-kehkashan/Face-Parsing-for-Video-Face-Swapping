#!/usr/bin/env python3
"""Verify project setup"""
import os
import sys
import torch
from pathlib import Path

def verify_setup():
    """Verify project setup is complete"""
    print("Verifying project setup...")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("✗ CUDA not available")
    
    # Check directories
    required_dirs = [
        "datasets/celebamask_hq",
        "models/pretrained",
        "research_finetuning",
        "outputs",
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
    
    # Check model file
    model_path = Path("models/pretrained/79999_iter.pth")
    if model_path.exists():
        print(f"✓ Model file found: {model_path}")
        print(f"  Size: {model_path.stat().st_size / 1024**2:.1f} MB")
    else:
        print(f"✗ Model file missing: {model_path}")
    
    # Check dataset
    dataset_path = Path("datasets/celebamask_hq")
    if dataset_path.exists() and dataset_path.is_dir():
        if dataset_path.is_symlink():
            print(f"✓ Dataset symlink exists: {dataset_path}")
            print(f"  Points to: {os.readlink(dataset_path)}")
        else:
            print(f"✓ Dataset directory exists: {dataset_path}")
    else:
        print(f"✗ Dataset not found: {dataset_path}")
    
    print("\nSetup verification complete!")

if __name__ == "__main__":
    verify_setup()
