# C:\SwapFace2Pon\FaceParsing_Research\research_finetuning\Part_3_Training\train.py

import sys
from pathlib import Path

# --- FIX for ModuleNotFoundError ---
# Add the project root to sys.path to allow absolute imports like 'research_finetuning.Part_1_Data'
# This assumes train.py is located at:
# C:\SwapFace2Pon\FaceParsing_Research\research_finetuning\Part_3_Training\train.py
# So, the project root is 2 levels up from this file.
project_root = Path('/content/drive/MyDrive/FaceParsing_Research')
sys.path.append(str(project_root))
# --- END FIX ---

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import argparse
import yaml
import os
import random
import numpy as np
from typing import Dict, Any, Optional, Union

# Import modules from your project
# These imports will now work correctly due to the sys.path modification
from research_finetuning.Part_1_Data.celebamask_dataset import create_data_loaders, CelebAMaskHQDataset
from research_finetuning.Part_1_Data.augmentations import FaceParsingAugmentation
from research_finetuning.Part_1_Data.data_utils import calculate_dataset_statistics
from research_finetuning.Part_2_Model.model_builder import build_model_from_config
from research_finetuning.Part_3_Training.loss_functions import CombinedLoss
from research_finetuning.Part_3_Training.trainer import EnhancedTrainer
from research_finetuning.Part_3_Training.memory_manager import clear_cuda_cache # For initial setup
# Enhanced metrics import
from research_finetuning.Part_3_Training.metrics import ComprehensiveMetrics


def set_seed(seed: int) -> None:
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Set to False for deterministic behavior, True for speed (less reproducible)
    print(f"Random seed set to {seed}")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main function to set up and run the training process."""
    print("🚀 Starting training script...")

    # Load configuration
    config = load_config(args.config_path)
    print(f"Configuration loaded from: {args.config_path}")

    # ===== BATCH SIZE VALIDATION =====
    # CRITICAL FIX: Validate batch size configuration
    configured_batch_size = config['training']['batch_size']
    print(f"🔍 CONFIGURED BATCH SIZE: {configured_batch_size}")
    
    if configured_batch_size < 1:
        print(f"❌ ERROR: Invalid batch size {configured_batch_size}")
        return
    elif configured_batch_size == 1:
        print(f"⚠️  WARNING: Batch size of 1 detected - this may cause BatchNorm instability")
        print("   Consider using batch_size >= 2 for stable training")
    else:
        print(f"✅ Batch size {configured_batch_size} is valid")
    # ===== END BATCH SIZE VALIDATION =====

    # Set random seed for reproducibility
    set_seed(config['training'].get('seed', 42))

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ===== GPU MEMORY CHECK =====
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🖥️  GPU: {gpu_name}")
        print(f"🖥️  GPU Memory Available: {gpu_memory:.1f} GB")
        
        if gpu_memory < 5.5:
            print("⚠️  WARNING: Low GPU memory detected. Training may be unstable.")
            print("   Consider reducing batch_size or image_size if you encounter OOM errors")
        elif gpu_memory < 8.0:
            print("✅ GPU memory sufficient for current configuration")
        else:
            print("✅ Abundant GPU memory - configuration is well within limits")
    else:
        print("⚠️  WARNING: CUDA not available, falling back to CPU (training will be very slow)")
    # ===== END GPU MEMORY CHECK =====

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training outputs will be saved to: {output_dir}")

    # Save the effective configuration to the output directory for record-keeping
    with open(output_dir / "training_config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Training configuration saved to {output_dir / 'training_config.yaml'}")

    # --- Data Loading ---
    print("\n--- Setting up DataLoaders ---")
    
    # Create augmentation pipeline
    train_aug = FaceParsingAugmentation(
        image_size=config['dataset']['image_size'], 
        mode='train'
    ).transform

    # ===== EXPLICIT BATCH SIZE PASSING =====
    # CRITICAL FIX: Explicitly pass and verify batch size
    print(f"Creating data loaders with batch size: {configured_batch_size}")
    train_loader, val_loader = create_data_loaders(
        root_dir=config['dataset']['root_dir'],
        batch_size=configured_batch_size,  # Use the validated batch size
        image_size=config['dataset']['image_size'],
        num_workers=config['system'].get('num_workers', 2),
        pin_memory=config['system'].get('pin_memory', False),
        subset_size=config['dataset'].get('subset_size', None),
        augmentations=train_aug
    )
    
    # ===== COMPREHENSIVE BATCH SIZE VERIFICATION =====
    # CRITICAL: Verify actual batch size matches configuration
    actual_train_batch_size = train_loader.batch_size
    actual_val_batch_size = val_loader.batch_size
    train_dataset_size = len(train_loader.dataset)
    val_dataset_size = len(val_loader.dataset)
    expected_train_batches = train_dataset_size // actual_train_batch_size
    expected_val_batches = val_dataset_size // actual_val_batch_size
    
    print(f"\n✅ COMPREHENSIVE BATCH SIZE VERIFICATION:")
    print(f"   📊 Dataset Sizes:")
    print(f"      Train Dataset: {train_dataset_size:,} images")
    print(f"      Val Dataset: {val_dataset_size:,} images")
    print(f"   📊 Configured Batch Size: {configured_batch_size}")
    print(f"   📊 Actual Batch Sizes:")
    print(f"      Train Loader: {actual_train_batch_size}")
    print(f"      Val Loader: {actual_val_batch_size}")
    print(f"   📊 Expected vs Actual Batches:")
    print(f"      Train - Expected: {expected_train_batches:,}, Actual: {len(train_loader):,}")
    print(f"      Val - Expected: {expected_val_batches:,}, Actual: {len(val_loader):,}")
    
    # Critical error checking
    batch_size_errors = []
    if actual_train_batch_size != configured_batch_size:
        batch_size_errors.append(f"Train batch size mismatch: expected {configured_batch_size}, got {actual_train_batch_size}")
    if actual_val_batch_size > configured_batch_size:  # Only error if validation is LARGER
        batch_size_errors.append(f"Val batch size too large: expected ≤{configured_batch_size}, got {actual_val_batch_size}")
    
    # Check if number of batches makes sense
    batch_count_tolerance = 1  # Allow 1 batch difference due to drop_last
    if abs(len(train_loader) - expected_train_batches) > batch_count_tolerance:
        batch_size_errors.append(f"Train batch count unexpected: expected ~{expected_train_batches}, got {len(train_loader)}")
    
    if batch_size_errors:
        print(f"\n❌ CRITICAL BATCH SIZE ERRORS:")
        for error in batch_size_errors:
            print(f"   - {error}")
        print(f"\n🛑 Training aborted due to batch size configuration errors.")
        print(f"   Please check your data loader configuration.")
        return
    else:
        print(f"   ✅ All batch size validations passed!")
    # ===== END BATCH SIZE VERIFICATION =====

    # --- Model Initialization ---
    print("\n--- Initializing Model ---")
    # Pass the config to build the BiSeNetLoRA model
    model = build_model_from_config(config)
    model.to(device)
    print(f"Model initialized: {model.__class__.__name__}")
    model.print_model_info() # Call the method to print model details

    # ===== ENHANCED METRICS VALIDATION =====
    print("\n--- Enhanced Metrics Validation ---")
    try:
        # Test comprehensive metrics with dummy data
        test_logits = torch.randn(2, config['model']['num_classes'], 64, 64).to(device)
        test_targets = torch.randint(0, config['model']['num_classes'], (2, 64, 64)).to(device)
        
        class_names = [
            'background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
            'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 'mouth',
            'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat'
        ]
        
        metrics_calc = ComprehensiveMetrics(
            num_classes=config['model']['num_classes'],
            ignore_index=config['dataset'].get('ignore_index', 255),
            class_names=class_names
        )
        
        metrics_calc.update(test_logits, test_targets)
        test_results = metrics_calc.compute_metrics()
        
        print(f"✅ Comprehensive metrics test successful!")
        print(f"   Sample mIoU: {test_results['overall']['mIoU']:.4f}")
        print(f"   Sample Mean F1: {test_results['overall']['Mean_F1']:.4f}")
        print(f"   Classes present: {test_results['overall']['Classes_Present']}")
        print(f"   Metrics tracked: {len(test_results['per_class'])} classes")
        
    except Exception as e:
        print(f"❌ Enhanced metrics validation failed: {e}")
        print("Falling back to basic metrics...")
        if config['training'].get('debug_mode', False):
            import traceback
            traceback.print_exc()
    # ===== END ENHANCED METRICS VALIDATION =====

    # --- Optimizer Setup ---
    print("\n--- Setting up Optimizer ---")
    # Get parameter groups with different learning rates for backbone vs LoRA
    param_groups = []

    # LoRA parameters - use full learning rate
    lora_params = model.get_lora_parameters()
    if lora_params:
        param_groups.append({
            'params': lora_params, 
            'lr': float(config['training']['learning_rate']),
            'name': 'lora'
        })

    # Backbone parameters (non-LoRA) - use lower learning rate
    backbone_params = []
    for name, param in model.model.named_parameters():
        if param.requires_grad and 'lora_' not in name:
            backbone_params.append(param)

    if backbone_params:
        backbone_lr_multiplier = config['advanced'].get('backbone_lr_multiplier', 0.1)
        param_groups.append({
            'params': backbone_params,
            'lr': float(config['training']['learning_rate']) * backbone_lr_multiplier,
            'name': 'backbone'
        })

    # Create optimizer with parameter groups
    optimizer = optim.AdamW(
        param_groups if param_groups else model.parameters(),
        lr=float(config['training']['learning_rate']), 
        weight_decay=float(config['training'].get('weight_decay', 1e-4))
    )

    print(f"Parameter groups: {len(param_groups)}")
    for i, group in enumerate(param_groups):
        group_name = group.get('name', f'group_{i}')
        print(f"  {group_name}: {len(group['params'])} parameters, lr={group['lr']:.6f}")

    # --- Scheduler Setup ---
    print("\n--- Setting up Learning Rate Scheduler ---")
    scheduler_type = config['training'].get('scheduler', 'CosineAnnealingLR')
    if scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['training']['epochs'], 
            eta_min=config['training']['learning_rate'] * 0.01  # 1% of base LR
        )
        print(f"Scheduler: CosineAnnealingLR (T_max={config['training']['epochs']}, eta_min={config['training']['learning_rate'] * 0.01:.2e})")
    else:
        scheduler = None
        print(f"No scheduler configured")

    # --- Loss Function Setup ---
    print("\n--- Setting up Loss Function ---")
    print("Calculating class weights from training dataset...")

    # ===== IMPROVED CLASS WEIGHT CALCULATION =====
    # Create a temporary dataset instance for class weight calculation
    print(f"Creating temporary dataset for class weight calculation...")
    temp_dataset = CelebAMaskHQDataset(
        root_dir=config['dataset']['root_dir'],
        split='train',
        image_size=config['dataset']['image_size'],
        subset_size=2000  # Use more images for statistics (will be limited by num_samples_for_stats)
    )

    # Get the number of samples for statistics
    num_samples_for_stats = config['training'].get('num_samples_for_stats', 5000)
    # Add smoothing configuration
    config['training']['weight_smoothing_power'] = config['training'].get('weight_smoothing_power', 0.5)
    num_samples_for_stats = min(num_samples_for_stats, len(temp_dataset))

    print(f"Calculating class weights from {num_samples_for_stats} samples out of {len(temp_dataset)} available...")

    # Use the dataset's built-in get_class_weights method with caching
    class_weights = temp_dataset.get_class_weights(num_samples=num_samples_for_stats, use_cache=True)

    print(f"Original Class Weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")

    # Apply additional smoothing for stability
    # Use power smoothing to reduce extreme values
    smoothing_power = config['training'].get('weight_smoothing_power', 0.5)
    if smoothing_power != 1.0:
        print(f"Applying power smoothing with exponent {smoothing_power}")
        class_weights = torch.pow(class_weights, smoothing_power)

    # Apply class weight capping with config values
    max_weight = config['training'].get('max_class_weight', 2.5)  # More conservative default
    min_weight = config['training'].get('min_class_weight', 0.5)   # Higher minimum

    print(f"Applying class weight capping: min={min_weight}, max={max_weight}")

    # Cap the weights
    class_weights_original = class_weights.clone()
    class_weights = torch.clamp(class_weights, min=min_weight, max=max_weight)

    # Log which weights were capped
    capped_indices = torch.where((class_weights_original != class_weights))[0]
    if len(capped_indices) > 0:
        print(f"⚠️  Capped {len(capped_indices)} extreme class weights:")
        for idx in capped_indices:
            class_name = temp_dataset.IDX_TO_CLASS.get(int(idx), f"class_{idx}")
            print(f"  {class_name} (Class {idx}): {class_weights_original[idx]:.3f} → {class_weights[idx]:.3f}")
    else:
        print(f"✅ No class weights needed capping - all within reasonable range")

    print(f"Final Class Weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")
    # ===== END CLASS WEIGHT CAPPING =====

    # Optional normalization (usually not needed with conservative capping)
    if config['training'].get('normalize_class_weights', False):
        print("Applying class weight normalization...")
        min_normalized_weight = 0.2
        weight_sum = class_weights.sum()
        class_weights = class_weights / weight_sum * len(class_weights)
        class_weights = torch.clamp(class_weights, min=min_normalized_weight)
        print(f"Normalized Class Weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")

    # Create combined loss function
    criterion = CombinedLoss(
        num_classes=config['model']['num_classes'],
        ce_weight=config['training'].get('ce_loss_weight', 0.2),  # Reduced CE dominance
        dice_weight=config['training'].get('dice_loss_weight', 0.8),  # Increased Dice weight
        class_weights=class_weights.to(device), # Move class weights to device
        ignore_index=config['dataset'].get('ignore_index', 255)
    )
    print(f"Loss Function: CombinedLoss")
    print(f"  Cross-Entropy Weight: {criterion.ce_weight_norm:.2f}")
    print(f"  Dice Loss Weight: {criterion.dice_weight_norm:.2f}")
    print(f"  Using class weights: Yes ({len(class_weights)} classes)")

    # --- Trainer Initialization and Run ---
    print("\n--- Initializing Enhanced Trainer ---")
    trainer = EnhancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=config,
        output_dir=output_dir
    )

    # Optionally load a checkpoint to resume training
    if args.resume_checkpoint:
        print(f"Loading checkpoint from: {args.resume_checkpoint}")
        trainer.load_checkpoint(args.resume_checkpoint)
    
    # ===== PRE-TRAINING VALIDATION =====
    print("\n--- Pre-training Validation ---")
    try:
        # Test loading and processing one batch
        print("Testing data loading...")
        sample_batch = next(iter(train_loader))
        sample_images, sample_masks = sample_batch
        
        print(f"✅ Sample batch loaded successfully:")
        print(f"   Images shape: {sample_images.shape}")
        print(f"   Masks shape: {sample_masks.shape}")
        print(f"   Actual batch size in tensor: {sample_images.shape[0]}")
        print(f"   Image dtype: {sample_images.dtype}, range: [{sample_images.min():.3f}, {sample_images.max():.3f}]")
        print(f"   Mask dtype: {sample_masks.dtype}, unique values: {sample_masks.unique().tolist()}")
        
        # Test forward pass
        print("Testing model forward pass...")
        model.eval()
        with torch.no_grad():
            sample_images = sample_images.to(device)
            sample_masks = sample_masks.to(device)
            outputs = model(sample_images)
            
            print(f"✅ Model forward pass successful:")
            for key, output in outputs.items():
                print(f"   {key}: {output.shape}")
            
            # Test loss calculation
            print("Testing loss calculation...")
            loss = criterion(outputs['out'], sample_masks)
            print(f"✅ Loss calculation successful: {loss.item():.4f}")
            
            # Check for reasonable loss value
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"❌ ERROR: Loss is {loss.item()} - this indicates a problem")
                return
            elif loss.item() > 50.0:
                print(f"⚠️  WARNING: Loss is very high ({loss.item():.4f}) - training may be unstable")
            elif loss.item() < 0.01:
                print(f"⚠️  WARNING: Loss is very low ({loss.item():.4f}) - check if this is expected")
            else:
                print(f"✅ Loss value appears reasonable: {loss.item():.4f}")
                
        model.train()  # Set back to training mode
        
        print("✅ Pre-training validation completed successfully!")
        
    except Exception as e:
        print(f"❌ Pre-training validation failed: {e}")
        print("Training aborted due to validation failure")
        if config['training'].get('debug_mode', False):
            import traceback
            traceback.print_exc()
        return
    # ===== END PRE-TRAINING VALIDATION =====
    
    # Start training
    print(f"\n🏁 Starting training with {len(train_loader):,} training batches and {len(val_loader):,} validation batches...")
    trainer.train()

    print("\n🎉 Training process completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BiSeNetLoRA for Face Parsing.")
    parser.add_argument("--config-path", type=str, 
                        default="configs/default.yaml",
                        help="Path to the training configuration YAML file.")
    parser.add_argument("--output-dir", type=str, 
                        default="outputs/training_runs",
                        help="Directory to save training logs, checkpoints, and TensorBoard data.")
    parser.add_argument("--resume-checkpoint", type=str, 
                        default=None,
                        help="Path to a checkpoint file to resume training from.")
    
    args = parser.parse_args()
    main(args)
