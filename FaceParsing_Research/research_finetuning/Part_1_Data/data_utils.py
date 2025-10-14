"""
Data utilities for face parsing dataset
Helper functions for data processing and validation
"""

import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from PIL import Image
from collections import Counter


def verify_dataset_structure(root_dir: str) -> Dict[str, any]:
    """
    Verify CelebAMask-HQ dataset structure and report statistics
    
    Args:
        root_dir: Root directory of the dataset
    
    Returns:
        Dictionary with dataset statistics and issues
    """
    root_path = Path(root_dir)
    report = {
        'valid': True,
        'issues': [],
        'statistics': {},
        'missing_files': []
    }
    
    print("🔍 Verifying dataset structure...")
    
    # Check main directories
    image_dir = root_path / 'CelebA-HQ-img'
    mask_dir = root_path / 'CelebAMask-HQ-mask-anno'
    
    if not image_dir.exists():
        report['valid'] = False
        report['issues'].append(f"Image directory not found: {image_dir}")
    
    if not mask_dir.exists():
        report['valid'] = False
        report['issues'].append(f"Mask directory not found: {mask_dir}")
    
    if not report['valid']:
        return report
    
    # Count images
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    report['statistics']['total_images'] = len(image_files)
    
    # Count mask folders
    mask_folders = sorted([f for f in mask_dir.iterdir() if f.is_dir()])
    report['statistics']['mask_folders'] = len(mask_folders)
    
    # Sample check for mask completeness
    print("Checking mask completeness (sampling 100 images)...")
    sample_size = min(100, len(image_files))
    sample_indices = np.random.choice(len(image_files), sample_size, replace=False)
    
    missing_masks = []
    mask_classes = set()
    
    for idx in tqdm(sample_indices, desc="Checking masks"):
        image_id = int(image_files[idx].stem)
        folder_idx = image_id // 2000
        mask_folder = mask_dir / str(folder_idx)
        
        # Check for each mask class
        expected_classes = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 
                           'nose', 'mouth', 'u_lip', 'l_lip', 'hair']
        
        for class_name in expected_classes:
            mask_file = mask_folder / f"{image_id:05d}_{class_name}.png"
            if not mask_file.exists():
                missing_masks.append(str(mask_file))
            else:
                mask_classes.add(class_name)
    
    report['statistics']['sampled_images'] = sample_size
    report['statistics']['missing_masks_in_sample'] = len(missing_masks)
    report['statistics']['found_mask_classes'] = sorted(list(mask_classes))
    report['missing_files'] = missing_masks[:10]  # Only show first 10
    
    # Check split files
    splits_dir = root_path.parent / 'datasets' / 'splits'
    if splits_dir.exists():
        for split in ['train', 'val', 'test']:
            split_file = splits_dir / f'{split}.txt'
            if split_file.exists():
                with open(split_file, 'r') as f:
                    ids = f.readlines()
                report['statistics'][f'{split}_samples'] = len(ids)
            else:
                report['issues'].append(f"Split file not found: {split_file}")
    
    # Print report
    print("\n📊 Dataset Verification Report:")
    print(f"Valid: {'✓' if report['valid'] else '✗'}")
    print(f"\nStatistics:")
    for key, value in report['statistics'].items():
        print(f"  {key}: {value}")
    
    if report['issues']:
        print(f"\nIssues found:")
        for issue in report['issues']:
            print(f"  - {issue}")
    
    if report['missing_files']:
        print(f"\nSample of missing files:")
        for file in report['missing_files']:
            print(f"  - {file}")
    
    return report


def calculate_dataset_statistics(dataset, num_samples: int = 100) -> Dict:
    """
    Calculate dataset statistics for normalization and class balancing
    
    Args:
        dataset: CelebAMaskHQDataset instance
        num_samples: Number of samples to use for statistics
    
    Returns:
        Dictionary with statistics
    """
    print(f"\n📊 Calculating dataset statistics from {num_samples} samples...")
    
    # Initialize accumulators
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0
    
    class_pixel_counts = np.zeros(len(dataset.MASK_CLASSES))
    total_pixels = 0
    
    # Sample random indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in tqdm(indices, desc="Processing samples"):
        image, mask = dataset[idx]
        
        # Convert to numpy
        image_np = image.numpy()
        mask_np = mask.numpy()
        
        # Accumulate for mean/std (note: image is already normalized)
        # We'll calculate on unnormalized values
        image_unnorm = image_np * np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        image_unnorm += np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        
        pixel_sum += image_unnorm.sum(axis=(1, 2))
        pixel_sq_sum += (image_unnorm ** 2).sum(axis=(1, 2))
        pixel_count += image_unnorm.shape[1] * image_unnorm.shape[2]
        
        # Count mask classes
        for class_idx in range(len(dataset.MASK_CLASSES)):
            class_pixel_counts[class_idx] += (mask_np == class_idx).sum()
        total_pixels += mask_np.size
    
    # Calculate statistics
    pixel_mean = pixel_sum / pixel_count
    pixel_std = np.sqrt(pixel_sq_sum / pixel_count - pixel_mean ** 2)
    
    # Class distribution
    class_distribution = class_pixel_counts / total_pixels
    
    # Create report
    statistics = {
        'pixel_mean': pixel_mean.tolist(),
        'pixel_std': pixel_std.tolist(),
        'class_distribution': {},
        'class_pixel_counts': class_pixel_counts.tolist(),
        'total_pixels': int(total_pixels),
        'samples_used': len(indices)
    }
    
    # Add class names to distribution
    for idx, (class_name, _) in enumerate(dataset.MASK_CLASSES.items()):
        statistics['class_distribution'][class_name] = {
            'percentage': float(class_distribution[idx] * 100),
            'pixel_count': int(class_pixel_counts[idx])
        }
    
    # Print statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"Pixel mean (RGB): {pixel_mean}")
    print(f"Pixel std (RGB): {pixel_std}")
    print(f"\nClass distribution:")
    for class_name, info in statistics['class_distribution'].items():
        print(f"  {class_name:12s}: {info['percentage']:6.2f}%")
    
    return statistics


def create_visualization_grid(dataset, indices: List[int], save_path: str):
    """
    Create a grid visualization of images and masks
    
    Args:
        dataset: CelebAMaskHQDataset instance
        indices: List of indices to visualize
        save_path: Path to save the visualization
    """
    num_samples = len(indices)
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 3, rows * 6))
    if rows * 2 == 2:
        axes = axes.reshape(2, -1)
    
    for idx, sample_idx in enumerate(indices):
        row = idx // cols
        col = idx % cols
        
        # Get sample
        image, mask = dataset[sample_idx]
        
        # Denormalize image
        image_np = image.permute(1, 2, 0).numpy()
        image_np = image_np * np.array([0.229, 0.224, 0.225])
        image_np += np.array([0.485, 0.456, 0.406])
        image_np = np.clip(image_np, 0, 1)
        
        # Convert mask to RGB
        mask_rgb = dataset.mask_to_rgb(mask.numpy())
        
        # Plot image
        axes[row*2, col].imshow(image_np)
        axes[row*2, col].set_title(f'Image {dataset.image_ids[sample_idx]}')
        axes[row*2, col].axis('off')
        
        # Plot mask
        axes[row*2+1, col].imshow(mask_rgb)
        axes[row*2+1, col].set_title('Mask')
        axes[row*2+1, col].axis('off')
    
    # Hide empty subplots
    for idx in range(num_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row*2, col].axis('off')
        axes[row*2+1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")


def validate_data_loader(data_loader, num_batches: int = 5):
    """
    Validate data loader by checking a few batches
    
    Args:
        data_loader: PyTorch DataLoader
        num_batches: Number of batches to check
    """
    print(f"\n🔍 Validating data loader ({num_batches} batches)...")
    
    for batch_idx, (images, masks) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Images dtype: {images.dtype}")
        print(f"  Images range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Masks shape: {masks.shape}")
        print(f"  Masks dtype: {masks.dtype}")
        print(f"  Masks unique values: {masks.unique()}")
        
        # Check for NaN or Inf
        if torch.isnan(images).any():
            print("  ⚠️ WARNING: NaN values found in images!")
        if torch.isinf(images).any():
            print("  ⚠️ WARNING: Inf values found in images!")
        
        # Check mask values
        if masks.min() < 0 or masks.max() >= 19:
            print(f"  ⚠️ WARNING: Invalid mask values! Range: [{masks.min()}, {masks.max()}]")
    
    print("\n✓ Data loader validation complete!")


def create_class_legend(dataset_class, save_path: Optional[str] = None):
    """
    Create a legend showing all mask classes and their colors
    
    Args:
        dataset_class: CelebAMaskHQDataset class (not instance)
        save_path: Optional path to save the legend
    """
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    
    # Hide axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(dataset_class.MASK_CLASSES))
    ax.axis('off')
    
    # Add title
    ax.text(0.5, len(dataset_class.MASK_CLASSES) + 0.5, 'CelebAMask-HQ Classes', 
            fontsize=16, ha='center', weight='bold')
    
    # Add each class
    for idx, (class_name, class_idx) in enumerate(dataset_class.MASK_CLASSES.items()):
        y_pos = len(dataset_class.MASK_CLASSES) - idx - 1
        
        # Color box
        color = np.array(dataset_class.MASK_COLORS[class_idx]) / 255.0
        rect = plt.Rectangle((0.1, y_pos), 0.15, 0.8, 
                           facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        
        # Class name and index
        ax.text(0.3, y_pos + 0.4, f"{class_name} ({class_idx})", 
                fontsize=12, va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Class legend saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_memory_usage(dataset, batch_size: int = 2, num_workers: int = 2):
    """
    Analyze memory usage for different configurations
    
    Args:
        dataset: CelebAMaskHQDataset instance
        batch_size: Batch size to test
        num_workers: Number of workers
    """
    import psutil
    import gc
    
    print(f"\n💾 Analyzing memory usage...")
    
    # Get initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Initial memory: {initial_memory:.1f} MB")
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    # Load a few batches
    max_memory = initial_memory
    for idx, (images, masks) in enumerate(data_loader):
        if idx >= 10:
            break
        
        current_memory = process.memory_info().rss / 1024 / 1024
        max_memory = max(max_memory, current_memory)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            images = images.cuda()
            masks = masks.cuda()
            
            # Check GPU memory
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"  Batch {idx}: CPU: {current_memory:.1f} MB, GPU: {gpu_memory:.1f} MB")
            
            # Clear GPU memory
            del images, masks
            torch.cuda.empty_cache()
    
    # Clean up
    del data_loader
    gc.collect()
    
    print(f"\nMemory usage summary:")
    print(f"  Initial: {initial_memory:.1f} MB")
    print(f"  Peak: {max_memory:.1f} MB")
    print(f"  Increase: {max_memory - initial_memory:.1f} MB")
    
    return {
        'initial_mb': initial_memory,
        'peak_mb': max_memory,
        'increase_mb': max_memory - initial_memory
    }


if __name__ == "__main__":
    # Test utilities
    print("Testing data utilities...")
    
    # Update this path to your dataset
    dataset_path = "datasets/celebamask_hq"
    
    # Verify dataset structure
    report = verify_dataset_structure(dataset_path)
    
    # Save report
    with open("dataset_verification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\nDataset verification complete!")
    print("Report saved to: dataset_verification_report.json")