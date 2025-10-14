"""
CelebAMask-HQ Dataset Class
Handles loading images and assembling segmentation masks from individual files
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class CelebAMaskHQDataset(Dataset):
    """
    Dataset class for CelebAMask-HQ
    Handles the unique structure where masks are stored as separate files
    """
    
    # Mapping of mask classes to indices
    MASK_CLASSES = {
        'background': 0,
        'skin': 1,
        'l_brow': 2,
        'r_brow': 3,
        'l_eye': 4,
        'r_eye': 5,
        'eye_g': 6,      # eyeglasses
        'l_ear': 7,
        'r_ear': 8,
        'ear_r': 9,      # earring
        'nose': 10,
        'mouth': 11,
        'u_lip': 12,
        'l_lip': 13,
        'neck': 14,
        'neck_l': 15,    # necklace
        'cloth': 16,
        'hair': 17,
        'hat': 18,
    }
    
    # Inverse mapping for visualization
    IDX_TO_CLASS = {v: k for k, v in MASK_CLASSES.items()}
    
    # Colors for visualization (RGB format)
    MASK_COLORS = {
        0: (0, 0, 0),        # background - black
        1: (255, 178, 102),  # skin - peach
        2: (255, 0, 0),      # l_brow - red
        3: (255, 85, 0),     # r_brow - orange
        4: (255, 170, 0),    # l_eye - yellow
        5: (255, 255, 0),    # r_eye - bright yellow
        6: (0, 255, 0),      # eye_g - green
        7: (0, 255, 85),     # l_ear - light green
        8: (0, 255, 170),    # r_ear - cyan
        9: (0, 255, 255),    # ear_r - bright cyan
        10: (0, 170, 255),   # nose - light blue
        11: (0, 85, 255),    # mouth - blue
        12: (0, 0, 255),     # u_lip - dark blue
        13: (85, 0, 255),    # l_lip - purple
        14: (170, 0, 255),   # neck - violet
        15: (255, 0, 255),   # neck_l - magenta
        16: (255, 0, 170),   # cloth - pink
        17: (255, 0, 85),    # hair - dark pink
        18: (255, 255, 255), # hat - white
    }
    
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 image_size: int = 384,
                 transform=None,
                 subset_size: Optional[int] = None,
                 cache_masks: bool = False):
        """
        Initialize CelebAMask-HQ dataset
        
        Args:
            root_dir: Root directory containing CelebA-HQ-img and CelebAMask-HQ-mask-anno
            split: 'train', 'val', or 'test'
            image_size: Size to resize images to (default: 384 for GTX 1660 Super)
            transform: Optional transform to apply
            subset_size: If provided, only use this many images (for testing)
            cache_masks: Whether to cache assembled masks in memory
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.cache_masks = cache_masks
        
        # Verify dataset structure
        self.image_dir = self.root_dir / 'CelebA-HQ-img'
        self.mask_dir = self.root_dir / 'CelebAMask-HQ-mask-anno'
        
        if not self.image_dir.exists():
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise ValueError(f"Mask directory not found: {self.mask_dir}")
        
        # Load split file or create default splits
        self.image_ids = self._load_split(split)
        
        # Apply subset if requested
        if subset_size is not None:
            self.image_ids = self.image_ids[:subset_size]
        
        # Cache for assembled masks if enabled
        self.mask_cache = {} if cache_masks else None
        
        print(f"Initialized {split} dataset with {len(self.image_ids)} images")
        print(f"Image size: {image_size}x{image_size}")
        print(f"Mask caching: {'Enabled' if cache_masks else 'Disabled'}")
    
    def _load_split(self, split: str) -> List[int]:
        """Load image IDs for the specified split"""
        # FIX: Use correct path relative to dataset root
        split_file = self.root_dir / 'splits' / f'{split}.txt'
    
        if split_file.exists():
            with open(split_file, 'r') as f:
                lines = f.readlines()
        
            image_ids = []
            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
            
                # Handle both formats: "23303" and "23303.jpg"
                if line.endswith('.jpg') or line.endswith('.png'):
                    # Extract filename without extension
                    image_id = int(Path(line).stem)
                else:
                    # Assume it's already just the ID
                    image_id = int(line)
            
                image_ids.append(image_id)
        
            print(f"Loaded {len(image_ids)} IDs from {split_file}")
        else:
            # Create default splits if files don't exist
            print(f"Split file not found: {split_file}")
            print("Creating default splits...")
        
            # Default split: train=0-999, val=1000-1199, test=1200-1399
            if split == 'train':
                image_ids = list(range(0, 1000))
            elif split == 'val':
                image_ids = list(range(1000, 1200))
            elif split == 'test':
                image_ids = list(range(1200, 1400))
            else:
                raise ValueError(f"Unknown split: {split}")
    
        return image_ids
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get image and mask pair"""
        image_id = self.image_ids[idx]
        
        # Load image
        image = self._load_image(image_id)
        
        # Load or assemble mask
        if self.cache_masks and image_id in self.mask_cache:
            mask = self.mask_cache[image_id]
        else:
            mask = self._load_mask(image_id)
            if self.cache_masks:
                self.mask_cache[image_id] = mask
        
        # Apply transforms if any
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convert to tensors
        image = self._to_tensor(image)
        mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def _load_image(self, image_id: int) -> np.ndarray:
        """Load and resize image"""
        image_path = self.image_dir / f"{image_id}.jpg"
        if not image_path.exists():
            image_path = self.image_dir / f"{image_id}.png"
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize if needed
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        return np.array(image)
    
    def _load_mask(self, image_id: int) -> np.ndarray:
        """Load and assemble mask from individual files"""
        # Initialize empty mask
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        # Determine which folder the masks are in
        folder_idx = image_id // 2000
        mask_folder = self.mask_dir / str(folder_idx)
        
        # Load each mask class
        for class_name, class_idx in self.MASK_CLASSES.items():
            if class_idx == 0:  # Skip background
                continue
            
            # Construct mask filename
            mask_filename = f"{image_id:05d}_{class_name}.png"
            mask_path = mask_folder / mask_filename
            
            if mask_path.exists():
                # Load mask
                class_mask = Image.open(mask_path).convert('L')
                
                # Resize if needed
                if class_mask.size != (self.image_size, self.image_size):
                    class_mask = class_mask.resize(
                        (self.image_size, self.image_size), 
                        Image.NEAREST
                    )
                
                # Apply to main mask
                class_mask_np = np.array(class_mask)
                mask[class_mask_np > 0] = class_idx
        
        return mask
    
    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to tensor with proper ImageNet normalization"""
        # Convert to float and scale to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Use ImageNet normalization for pretrained model compatibility
        mean = np.array([0.485, 0.456, 0.406])  # ImageNet means
        std = np.array([0.229, 0.224, 0.225])   # ImageNet stds
        image = (image - mean) / std

        # Convert to tensor and change to CHW format
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)

        return image
    
    def visualize_sample(self, idx: int, save_path: Optional[str] = None):
        """Visualize a sample with its mask"""
        import matplotlib.pyplot as plt
        
        # Get sample
        image, mask = self[idx]
        
        # Denormalize image
        image = image.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        # Convert mask to RGB
        mask_rgb = self.mask_to_rgb(mask.numpy())
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(image)
        axes[0].set_title(f'Image {self.image_ids[idx]}')
        axes[0].axis('off')
        
        axes[1].imshow(mask_rgb)
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def mask_to_rgb(self, mask: np.ndarray) -> np.ndarray:
        """Convert mask indices to RGB colors"""
        h, w = mask.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_idx, color in self.MASK_COLORS.items():
            rgb_mask[mask == class_idx] = color
        
        return rgb_mask
    
    def get_class_weights(self, num_samples: int = 1000, use_cache: bool = True) -> torch.Tensor:
        """Calculate class weights for weighted loss with caching option"""
        cache_file = self.root_dir / f'class_weights_{num_samples}_{self.split}.pt'
    
        # Try to load from cache first
        if use_cache and cache_file.exists():
            print(f"Loading cached class weights from {cache_file}")
            try:
                cached_weights = torch.load(cache_file, map_location='cpu')
                print(f"Successfully loaded {len(cached_weights)} cached class weights")
                return cached_weights
            except Exception as e:
                print(f"Failed to load cached weights: {e}, recalculating...")
    
        print(f"Calculating class weights from {num_samples} samples...")

        class_counts = np.zeros(len(self.MASK_CLASSES), dtype=np.float64)

        # Sample subset of dataset
        indices = np.random.choice(len(self), min(num_samples, len(self)), replace=False)

        total_pixels_counted = 0
        for idx in tqdm(indices, desc="Counting pixels"):
            _, mask = self[idx]
            mask_np = mask.numpy()
    
            # Count pixels for each class
            unique, counts = np.unique(mask_np, return_counts=True)
            for cls, cnt in zip(unique, counts):
                if cls < len(self.MASK_CLASSES):  # Safety check
                    class_counts[cls] += cnt
    
            total_pixels_counted += mask_np.size

        # Calculate frequencies
        freq = class_counts / total_pixels_counted

        # Use inverse frequency with sqrt dampening (more stable than log)
        # Add smoothing to prevent division by zero
        class_weights = np.ones(len(self.MASK_CLASSES))
        for idx in range(len(self.MASK_CLASSES)):
            if freq[idx] > 0:
                # Use sqrt to dampen extreme weights
                class_weights[idx] = np.sqrt(1.0 / (freq[idx] + 1e-6))

        # Normalize weights to have mean of 1.0
        class_weights = class_weights / class_weights.mean()

        # Apply more conservative capping
        class_weights = np.clip(class_weights, 0.5, 2.5)

        print("\nClass weights (after sqrt dampening and conservative clipping):")
        for idx, weight in enumerate(class_weights):
            class_name = self.IDX_TO_CLASS[idx]
            print(f"  {class_name}: {weight:.3f}")
    
        # Convert to tensor
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
    
        # Cache the weights if requested
        if use_cache:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(weight_tensor, cache_file)
                print(f"Cached class weights to {cache_file}")
            except Exception as e:
                print(f"Failed to cache weights: {e}")

        return weight_tensor


def create_data_loaders(
    root_dir: str,
    batch_size: int = 8,
    image_size: int = 384,
    num_workers: int = 2,
    pin_memory: bool = False,
    subset_size: Optional[int] = None,
    augmentations=None,
    cache_masks: bool = False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation data loaders
    
    Args:
        root_dir: Path to dataset root
        batch_size: Batch size (default: 8, adjust based on GPU memory)
        image_size: Image size (default: 384)
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer (default: False)
        subset_size: If provided, only use this many images
        augmentations: Optional augmentation pipeline
        cache_masks: Whether to cache assembled masks in memory (default: False)
    
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = CelebAMaskHQDataset(
        root_dir=root_dir,
        split='train',
        image_size=image_size,
        transform=augmentations,
        subset_size=subset_size,
        cache_masks=cache_masks
    )
    
    val_dataset = CelebAMaskHQDataset(
        root_dir=root_dir,
        split='val',
        image_size=image_size,
        transform=None,  # No augmentation for validation
        subset_size=subset_size // 5 if subset_size else None,
        cache_masks=cache_masks
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size // 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing CelebAMask-HQ Dataset...")
    
    # You'll need to update this path
    dataset_path = "datasets/celebamask_hq"
    
    # Create dataset
    dataset = CelebAMaskHQDataset(
        root_dir=dataset_path,
        split='train',
        image_size=384,
        subset_size=10,  # Just test with 10 images
        cache_masks=True  # Test with caching enabled
    )
    
    # Test loading
    print(f"\nDataset size: {len(dataset)}")
    
    # Load first sample
    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {mask.unique()}")
    
    # Visualize
    dataset.visualize_sample(0, save_path="sample_visualization.png")
    print("\nSample visualization saved to 'sample_visualization.png'")
    
    # Get class weights
    weights = dataset.get_class_weights(num_samples=10)
    print(f"\nClass weights shape: {weights.shape}")