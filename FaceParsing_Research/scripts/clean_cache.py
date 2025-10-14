#!/usr/bin/env python3
"""Clean temporary files and cache"""
import shutil
from pathlib import Path

def clean_cache():
    """Remove temporary files"""
    print("Cleaning cache and temporary files...")
    
    # Directories to clean
    clean_dirs = [
        "outputs/logs/training",
        "outputs/tensorboard",
        "__pycache__",
        ".pytest_cache",
    ]
    
    for dir_name in clean_dirs:
        for path in Path(".").rglob(dir_name):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Removed: {path}")
    
    # File patterns to remove
    patterns = ["*.pyc", "*.pyo", "*.log", ".DS_Store"]
    for pattern in patterns:
        for file in Path(".").rglob(pattern):
            file.unlink()
            print(f"Removed: {file}")
    
    print("Cache cleaned!")

if __name__ == "__main__":
    clean_cache()
