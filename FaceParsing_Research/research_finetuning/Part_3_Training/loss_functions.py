# C:\SwapFace2Pon\FaceParsing_Research\research_finetuning\Part_3_Training\loss_functions.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross-Entropy Loss for segmentation, with optional class weights.
    
    Args:
        weight (torch.Tensor, optional): A manual rescaling weight given to each class.
                                         If given, has to be a tensor of size `C` (num_classes).
        ignore_index (int): Specifies a target value that is ignored and does not contribute
                            to the input gradient. Default: 255.
    """
    def __init__(self, weight: Optional[torch.Tensor] = None, ignore_index: int = 255): # Common ignore_index for segmentation
        super(WeightedCrossEntropyLoss, self).__init__()
        # Register as buffer so it's moved with the model and saved/loaded with state_dict
        self.register_buffer('weight', weight) 
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the weighted cross-entropy loss.

        Args:
            inputs (torch.Tensor): Predicted logits from the model (N, C, H, W).
            targets (torch.Tensor): Ground truth segmentation masks (N, H, W).

        Returns:
            torch.Tensor: The calculated loss.
        """
        # F.cross_entropy expects inputs (N, C, H, W) and targets (N, H, W) for segmentation
        # Ensure targets are long type for cross_entropy
        if targets.dtype != torch.long:
            targets = targets.long()

        # The weight tensor should be on the same device and dtype as inputs for computation
        if self.weight is not None:
            if self.weight.device != inputs.device:
                self.weight = self.weight.to(inputs.device)
            # Cast weight to same dtype as inputs (important for mixed precision)
            if self.weight.dtype != inputs.dtype:
                self.weight = self.weight.to(inputs.dtype)

        loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index
        )
        return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    
    Args:
        num_classes (int): Number of segmentation classes.
        smooth (float): A small constant added to the numerator and denominator to avoid
                        division by zero, and to smooth the loss. Default: 1e-6.
        ignore_index (int): Class index to ignore (e.g., background or void label).
                            Pixels with this label will not contribute to the loss.
                            Default: 255.
    """
    def __init__(self, num_classes: int, smooth: float = 1e-6, ignore_index: int = 255):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the Dice loss with improved numerical stability."""
        # Apply softmax to get probabilities
        probs = F.softmax(inputs, dim=1)

        # Create valid mask
        valid_mask = (targets != self.ignore_index)

        # If no valid pixels, return 0 loss
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device)

        # Clamp targets to valid range and set invalid pixels to 0
        targets_clamped = targets.clone()
        targets_clamped[~valid_mask] = 0  # Set invalid pixels to 0
        targets_clamped = torch.clamp(targets_clamped, 0, self.num_classes - 1)
    
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets_clamped, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Apply valid mask to both predictions and targets
        valid_mask_expanded = valid_mask.unsqueeze(1)
        probs_masked = probs * valid_mask_expanded
        targets_masked = targets_one_hot * valid_mask_expanded

        # Calculate intersection and union for each class
        intersection = (probs_masked * targets_masked).sum(dim=(0, 2, 3))
        pred_sum = probs_masked.sum(dim=(0, 2, 3))
        target_sum = targets_masked.sum(dim=(0, 2, 3))
        union = pred_sum + target_sum

        # Calculate Dice score with larger smooth factor for stability
        smooth = 1.0
        dice_scores = (2.0 * intersection + smooth) / (union + smooth)

        # Only consider classes that actually appear in the batch
        # This prevents division issues and focuses on relevant classes
        valid_classes = target_sum > 0
    
        if valid_classes.any():
            # Return loss for classes present in the batch
            valid_dice_scores = dice_scores[valid_classes]
            dice_loss = 1.0 - valid_dice_scores.mean()
        else:
            # Fallback: if no target classes found, use all classes
            dice_loss = 1.0 - dice_scores.mean()

        return dice_loss


class CombinedLoss(nn.Module):
    """
    Combines Weighted Cross-Entropy Loss and Dice Loss with configurable weights.
    
    Args:
        num_classes (int): Number of segmentation classes.
        ce_weight (float): Weight for the Cross-Entropy Loss component. Must be >= 0.
        dice_weight (float): Weight for the Dice Loss component. Must be >= 0.
        class_weights (torch.Tensor, optional): Class weights for Cross-Entropy Loss.
                                                If provided, must be a tensor of size `num_classes`.
        ignore_index (int): Specifies a target value that is ignored for both losses.
                            Default: 255.
    """
    def __init__(self, 
                 num_classes: int, 
                 ce_weight: float = 0.5, 
                 dice_weight: float = 0.5, 
                 class_weights: Optional[torch.Tensor] = None,
                 ignore_index: int = 255):
        super(CombinedLoss, self).__init__()
        
        if ce_weight < 0 or dice_weight < 0:
            raise ValueError("ce_weight and dice_weight must be non-negative.")
        if ce_weight == 0 and dice_weight == 0:
            raise ValueError("At least one loss weight (ce_weight or dice_weight) must be greater than 0.")
        
        # Normalize weights if they don't sum to 1, for consistent scaling
        total_weight = ce_weight + dice_weight
        self.ce_weight_norm = ce_weight / total_weight
        self.dice_weight_norm = dice_weight / total_weight

        self.ce_loss = WeightedCrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the combined loss.

        Args:
            inputs (torch.Tensor): Predicted logits from the model (N, C, H, W).
            targets (torch.Tensor): Ground truth segmentation masks (N, H, W).

        Returns:
            torch.Tensor: The calculated combined loss.
        """
        loss_ce = self.ce_loss(inputs, targets)
        loss_dice = self.dice_loss(inputs, targets)
        
        # Apply normalized weights
        total_loss = self.ce_weight_norm * loss_ce + self.dice_weight_norm * loss_dice
        return total_loss

if __name__ == "__main__":
    print("Testing loss_functions.py...")

    # --- Test Case 1: WeightedCrossEntropyLoss ---
    print("\n--- Testing WeightedCrossEntropyLoss ---")
    num_classes = 19
    H, W = 384, 384
    batch_size = 2
    
    # Simulate model output (logits)
    inputs_ce = torch.randn(batch_size, num_classes, H, W, requires_grad=True)
    # Simulate ground truth masks
    # Use 255 as ignore_index for some pixels
    targets_ce = torch.randint(0, num_classes, (batch_size, H, W), dtype=torch.long)
    targets_ce[0, :10, :10] = 255 # Set some pixels to ignore_index

    # Example class weights (e.g., inverse frequency)
    # For a real scenario, these would come from data_utils.calculate_dataset_statistics
    class_weights_tensor = torch.rand(num_classes) * 10 
    class_weights_tensor[0] = 0.5 # Example: low weight for background
    class_weights_tensor[1] = 2.0 # Example: high weight for a critical class
    print(f"Example Class Weights: {class_weights_tensor}")

    ce_criterion = WeightedCrossEntropyLoss(weight=class_weights_tensor, ignore_index=255)
    loss_ce = ce_criterion(inputs_ce, targets_ce)
    print(f"Weighted Cross-Entropy Loss: {loss_ce.item():.4f}")
    loss_ce.backward()
    print(f"Gradient for inputs_ce exists: {inputs_ce.grad is not None}")
    inputs_ce.grad.zero_() # Clear gradients for next test

    # --- Test Case 2: DiceLoss ---
    print("\n--- Testing DiceLoss ---")
    inputs_dice = torch.randn(batch_size, num_classes, H, W, requires_grad=True)
    targets_dice = torch.randint(0, num_classes, (batch_size, H, W), dtype=torch.long)
    targets_dice[1, -10:, -10:] = 255 # Set some pixels to ignore_index

    dice_criterion = DiceLoss(num_classes=num_classes, ignore_index=255)
    loss_dice = dice_criterion(inputs_dice, targets_dice)
    print(f"Dice Loss: {loss_dice.item():.4f}")
    loss_dice.backward()
    print(f"Gradient for inputs_dice exists: {inputs_dice.grad is not None}")
    inputs_dice.grad.zero_()

    # --- Test Case 3: CombinedLoss ---
    print("\n--- Testing CombinedLoss ---")
    inputs_combined = torch.randn(batch_size, num_classes, H, W, requires_grad=True)
    targets_combined = torch.randint(0, num_classes, (batch_size, H, W), dtype=torch.long)

    combined_criterion = CombinedLoss(
        num_classes=num_classes,
        ce_weight=0.6,
        dice_weight=0.4,
        class_weights=class_weights_tensor,
        ignore_index=255
    )
    loss_combined = combined_criterion(inputs_combined, targets_combined)
    print(f"Combined Loss: {loss_combined.item():.4f}")
    loss_combined.backward()
    print(f"Gradient for inputs_combined exists: {inputs_combined.grad is not None}")
    inputs_combined.grad.zero_()

    # Test Case 4: CombinedLoss with zero weights (should raise error)
    print("\n--- Testing CombinedLoss with zero weights (expected error) ---")
    try:
        CombinedLoss(num_classes=num_classes, ce_weight=0.0, dice_weight=0.0)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\nAll loss functions tested successfully.")
