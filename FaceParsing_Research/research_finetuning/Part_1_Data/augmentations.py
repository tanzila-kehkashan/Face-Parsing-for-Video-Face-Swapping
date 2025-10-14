"""
Augmentation pipeline for face parsing
Light augmentations to show improvement without disrupting model
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch  # Added this import
from typing import Dict, List, Tuple, Optional


class FaceParsingAugmentation:
    """
    Light augmentation pipeline for face parsing
    Designed to improve generalization without disrupting pretrained features
    """
    
    def __init__(self, image_size: int = 384, mode: str = 'train'):
        """
        Initialize augmentation pipeline
        
        Args:
            image_size: Target image size
            mode: 'train', 'val', or 'test'
        """
        self.image_size = image_size
        self.mode = mode
        self.transform = self._build_transform()
    
    def _build_transform(self):
        """Build augmentation pipeline based on mode"""
        if self.mode == 'train':
            # Light augmentations for training
            transform = A.Compose([
                # Spatial augmentations (applied to both image and mask)
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,  # Very small shift
                    scale_limit=0.05,  # Very small scale
                    rotate_limit=5,    # Small rotation
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0
                ),
                
                # Color augmentations (applied to image only)
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1,  # Light brightness change
                        contrast_limit=0.1,    # Light contrast change
                        p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=5,     # Very small hue shift
                        sat_shift_limit=10,    # Small saturation shift
                        val_shift_limit=10,    # Small value shift
                        p=1.0
                    ),
                ], p=0.5),
                
                # Very light blur (helps with generalization)
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ], p=0.1),  # Only 10% chance
                
                # Ensure correct size
                A.Resize(self.image_size, self.image_size, 
                        interpolation=cv2.INTER_LINEAR,
                        always_apply=True),
            ])
        else:
            # No augmentation for validation/test, just resize
            transform = A.Compose([
                A.Resize(self.image_size, self.image_size,
                        interpolation=cv2.INTER_LINEAR,
                        always_apply=True),
            ])
        
        return transform
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """Apply augmentations"""
        augmented = self.transform(image=image, mask=mask)
        return augmented
    
    @staticmethod
    def get_training_augmentation(image_size: int = 384) -> A.Compose:
        """Get training augmentation pipeline"""
        return FaceParsingAugmentation(image_size, mode='train').transform
    
    @staticmethod
    def get_validation_augmentation(image_size: int = 384) -> A.Compose:
        """Get validation augmentation pipeline"""
        return FaceParsingAugmentation(image_size, mode='val').transform


class MixupAugmentation:
    """
    Mixup augmentation for face parsing
    Helps with generalization by mixing samples
    """
    
    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        """
        Initialize mixup augmentation
        
        Args:
            alpha: Beta distribution parameter (higher = more mixing)
            prob: Probability of applying mixup
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, images: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mixup to a batch
        
        Args:
            images: Batch of images (B, C, H, W)
            masks: Batch of masks (B, H, W)
        
        Returns:
            Mixed images and masks
        """
        batch_size = images.size(0)
        
        if np.random.random() > self.prob:
            return images, masks
        
        # Sample lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random shuffle for mixing
        indices = torch.randperm(batch_size).to(images.device)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[indices]
        
        # For masks, we use a different strategy
        # We randomly select pixels from each mask based on lambda
        mask_mix = torch.rand_like(masks.float()) < lam
        mixed_masks = torch.where(mask_mix, masks, masks[indices])
        
        return mixed_images, mixed_masks


class CutMixAugmentation:
    """
    CutMix augmentation adapted for segmentation
    Cuts and pastes regions between samples
    """
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        """
        Initialize CutMix augmentation
        
        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying cutmix
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, images: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply CutMix augmentation"""
        batch_size = images.size(0)
        
        if np.random.random() > self.prob:
            return images, masks
        
        # Sample lambda
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get image dimensions
        _, _, h, w = images.shape
        
        # Sample random box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        # Uniform sampling of center
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Box boundaries
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Random shuffle for mixing
        indices = torch.randperm(batch_size).to(images.device)
        
        # Apply cutmix
        images_mixed = images.clone()
        masks_mixed = masks.clone()
        
        images_mixed[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
        masks_mixed[:, bby1:bby2, bbx1:bbx2] = masks[indices, bby1:bby2, bbx1:bbx2]
        
        return images_mixed, masks_mixed


class TestTimeAugmentation:
    """
    Test-time augmentation for improved predictions
    Averages predictions over multiple augmented versions
    """
    
    def __init__(self, transforms: List[A.Compose]):
        """
        Initialize TTA
        
        Args:
            transforms: List of augmentation pipelines to apply
        """
        self.transforms = transforms
    
    def augment_image(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply all transforms to an image"""
        augmented_images = []
        for transform in self.transforms:
            augmented = transform(image=image)['image']
            augmented_images.append(augmented)
        return augmented_images
    
    @staticmethod
    def get_tta_transforms(image_size: int = 384) -> List[A.Compose]:
        """Get standard TTA transforms"""
        transforms = [
            # Original
            A.Compose([
                A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            ]),
            # Horizontal flip
            A.Compose([
                A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=1.0),
            ]),
            # Vertical flip
            A.Compose([
                A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
                A.VerticalFlip(p=1.0),
            ]),
            # Rotate 90
            A.Compose([
                A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
                A.Rotate(limit=(90, 90), p=1.0),
            ]),
        ]
        return transforms


def visualize_augmentations(image: np.ndarray, mask: np.ndarray, 
                          image_size: int = 384, save_path: Optional[str] = None):
    """Visualize different augmentations"""
    import matplotlib.pyplot as plt
    # Removed problematic relative import - not needed for this function
    
    # Create augmentation pipeline
    aug = FaceParsingAugmentation(image_size=image_size, mode='train')
    
    # Generate multiple augmented versions
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i in range(3):
        for j in range(4):
            idx = i * 4 + j
            
            if idx == 0:
                # Show original
                aug_image = image
                aug_mask = mask
                title = "Original"
            else:
                # Apply augmentation
                augmented = aug(image=image, mask=mask)
                aug_image = augmented['image']
                aug_mask = augmented['mask']
                title = f"Augmented {idx}"
            
            # Show image
            axes[i, j].imshow(aug_image)
            axes[i, j].set_title(title)
            axes[i, j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

class UnifiedAugmentation:
    """
    Unified augmentation that combines albumentations with tensor-based augmentations.
    Provides both numpy-based (spatial) and tensor-based (mixing) augmentations.
    """
    
    def __init__(self, image_size: int = 384, mode: str = 'train'):
        self.mode = mode
        self.image_size = image_size
        
        # Albumentations pipeline (applied during data loading)
        self.spatial_aug = FaceParsingAugmentation(image_size, mode)
        
        # Tensor-based augmentations (applied during training)
        if mode == 'train':
            self.mixup = MixupAugmentation(alpha=0.2, prob=0.3)
            self.cutmix = CutMixAugmentation(alpha=1.0, prob=0.3)
            self.use_tensor_aug = True
        else:
            self.mixup = None
            self.cutmix = None
            self.use_tensor_aug = False
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """Apply spatial augmentations (numpy-based)"""
        return self.spatial_aug(image=image, mask=mask)
    
    def apply_batch_augmentation(self, images: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply tensor-based augmentations to a batch during training.
        
        Args:
            images: Batch of images (B, C, H, W)
            masks: Batch of masks (B, H, W)
            
        Returns:
            Augmented images and masks
        """
        if not self.use_tensor_aug or not self.training:
            return images, masks
        
        # Randomly choose between mixup, cutmix, or no augmentation
        import random
        choice = random.choice(['none', 'mixup', 'cutmix'])
        
        if choice == 'mixup' and self.mixup:
            return self.mixup(images, masks)
        elif choice == 'cutmix' and self.cutmix:
            return self.cutmix(images, masks)
        else:
            return images, masks
    
    @property
    def training(self):
        """Check if in training mode"""
        return self.mode == 'train'
if __name__ == "__main__":
    # Test augmentations
    print("Testing augmentation pipeline...")
    
    # Create dummy data
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    mask = np.random.randint(0, 19, (512, 512), dtype=np.uint8)
    
    # Test training augmentation
    train_aug = FaceParsingAugmentation(image_size=384, mode='train')
    augmented = train_aug(image=image, mask=mask)
    
    print(f"Original image shape: {image.shape}")
    print(f"Original mask shape: {mask.shape}")
    print(f"Augmented image shape: {augmented['image'].shape}")
    print(f"Augmented mask shape: {augmented['mask'].shape}")
    
    # Test validation augmentation
    val_aug = FaceParsingAugmentation(image_size=384, mode='val')
    val_augmented = val_aug(image=image, mask=mask)
    
    print(f"\nValidation augmented image shape: {val_augmented['image'].shape}")
    print(f"Validation augmented mask shape: {val_augmented['mask'].shape}")
    
    print("\nAugmentation pipeline test completed!")