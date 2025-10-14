# trainer.py - Enhanced with debugging and stability features

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Union
import warnings

# Import modules from your project
from research_finetuning.Part_2_Model.model_builder import BiSeNetLoRA
from research_finetuning.Part_3_Training.loss_functions import CombinedLoss
from research_finetuning.Part_3_Training.metrics import calculate_miou
from research_finetuning.Part_3_Training.metrics import ComprehensiveMetrics
from research_finetuning.Part_3_Training.memory_manager import clear_cuda_cache, log_gpu_memory_usage, get_cpu_memory_usage


class EnhancedTrainer:
    """
    Enhanced Trainer with debugging and stability features for BiSeNetLoRA model.
    Includes gradient checking, loss validation, and comprehensive error handling.
    """
    
    def __init__(
        self,
        model: BiSeNetLoRA,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        criterion: CombinedLoss,
        device: torch.device,
        config: Dict[str, Any],
        output_dir: Union[str, Path],
        start_epoch: int = 0
    ):
        """Initialize the Enhanced Trainer with debugging features."""
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        self.output_dir = Path(output_dir)
        self.start_epoch = start_epoch

        # Training parameters from config
        self.epochs = self.config['training']['epochs']
        self.gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        self.mixed_precision = self.config['training'].get('mixed_precision', False)
        self.validation_frequency = self.config['training'].get('validation_frequency', 1)
        self.log_frequency_iter = self.config['training'].get('log_frequency_iter', 10)
        self.checkpoint_frequency_epoch = self.config['training'].get('checkpoint_frequency_epoch', 5)
        self.clear_cache_frequency_iter = self.config['training'].get('clear_cache_frequency_iter', 20)

        # Enhanced debugging parameters
        self.debug_mode = self.config['training'].get('debug_mode', False)
        self.check_gradients = self.config['training'].get('check_gradients', True)
        self.log_grad_norms = self.config['training'].get('log_grad_norms', True)
        self.validate_data = self.config['training'].get('validate_data', True)
        self.max_loss_threshold = self.config['training'].get('max_loss_threshold', 10.0)
        self.min_loss_threshold = self.config['training'].get('min_loss_threshold', 0.001)
        
        # Warmup parameters
        self.use_warmup = self.config['training'].get('use_warmup', False)
        self.warmup_steps = self.config['training'].get('warmup_steps', 100)
        self.warmup_factor = self.config['training'].get('warmup_factor', 0.1)
        self.base_lr = self.config['training'].get('learning_rate', 1e-6)
        
        # Early stopping parameters
        self.early_stopping_patience = self.config['training'].get('early_stopping_patience', 10)
        self.early_stopping_threshold = self.config['training'].get('early_stopping_threshold', 0.001)
        self.best_miou = -1.0
        self.epochs_without_improvement = 0

        # Setup GradScaler for mixed precision
        self.scaler = GradScaler() if self.mixed_precision else None

        # Setup directories and logging
        self._setup_logging()
        
        # Training state tracking
        self.global_step = 0
        self.training_stats = {
            'grad_norms': [],
            'loss_history': [],
            'lr_history': [],
            'nan_batches': 0,
            'inf_batches': 0,
            'skipped_batches': 0
        }

        print(f"Enhanced Trainer initialized with debugging: {self.debug_mode}")
        print(f"Gradient checking: {self.check_gradients}")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Warmup: {self.use_warmup} ({self.warmup_steps} steps)" if self.use_warmup else "No warmup")
    def _validate_batch_size(self):
        """Validate that batch size is working correctly."""
        try:
            # Test batch from train loader
            sample_batch = next(iter(self.train_loader))
            actual_batch_size = sample_batch[0].shape[0]
            configured_batch_size = self.config['training']['batch_size']
            
            print(f"🔍 Batch Size Validation:")
            print(f"   Configured: {configured_batch_size}")
            print(f"   Actual: {actual_batch_size}")
            
            if actual_batch_size != configured_batch_size:
                raise ValueError(f"Batch size mismatch! Expected {configured_batch_size}, got {actual_batch_size}")
                
            if actual_batch_size == 1:
                print("⚠️  WARNING: Batch size is 1, which may cause BatchNorm instability")
                
            return True
            
        except Exception as e:
            print(f"❌ Batch size validation failed: {e}")
            return False
    def _setup_logging(self):
        """Setup TensorBoard and checkpoint directories."""
        log_dir = self.output_dir / "tensorboard" / datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard logs: {log_dir}")

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints: {self.checkpoint_dir}")

        # Debug output directory
        if self.debug_mode:
            self.debug_dir = self.output_dir / "debug"
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"Debug outputs: {self.debug_dir}")

    def _validate_data_batch(self, images: torch.Tensor, masks: torch.Tensor, batch_idx: int) -> bool:
        """Validate input data for NaN/Inf values and correct ranges."""
        # Only validate first few batches and periodically to reduce overhead
        if not self.validate_data:
            return True
    
        # Validate first 3 batches, then every 50th batch
        if batch_idx >= 3 and batch_idx % 50 != 0:
            return True
            
        try:
            # Check images
            if torch.isnan(images).any():
                print(f"WARNING: NaN values in images at batch {batch_idx}")
                return False
            if torch.isinf(images).any():
                print(f"WARNING: Inf values in images at batch {batch_idx}")
                return False
                
            # Check masks
            if torch.isnan(masks).any():
                print(f"WARNING: NaN values in masks at batch {batch_idx}")
                return False
            if (masks < 0).any() or (masks >= self.config['model']['num_classes']).any():
                valid_mask = masks != self.config['dataset'].get('ignore_index', 255)
                invalid_values = masks[valid_mask & ((masks < 0) | (masks >= self.config['model']['num_classes']))]
                if len(invalid_values) > 0:
                    print(f"WARNING: Invalid mask values {invalid_values.unique().tolist()} at batch {batch_idx}")
                    return False
                    
            # Check ranges
            if images.min() < -10 or images.max() > 10:
                print(f"WARNING: Unusual image range [{images.min():.3f}, {images.max():.3f}] at batch {batch_idx}")
                
            return True
            
        except Exception as e:
            print(f"ERROR: Data validation failed at batch {batch_idx}: {e}")
            return False

    def _check_gradients(self) -> Dict[str, float]:
        """Check gradients for NaN/Inf values and compute norms."""
        grad_stats = {
            'total_norm': 0.0,
            'max_grad': 0.0,
            'has_nan': False,
            'has_inf': False,
            'zero_grads': 0
        }
        
        if not self.check_gradients:
            return grad_stats
            
        total_norm = 0.0
        max_grad = 0.0
        zero_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Check for NaN/Inf
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient detected in {name}")
                    grad_stats['has_nan'] = True
                    
                if torch.isinf(param.grad).any():
                    print(f"Inf gradient detected in {name}")
                    grad_stats['has_inf'] = True
                    
                # Compute norms
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_grad = max(max_grad, param.grad.abs().max().item())
                
                # Check for zero gradients
                if param_norm < 1e-8:
                    zero_count += 1
                    
            else:
                zero_count += 1
                
        grad_stats['total_norm'] = total_norm ** 0.5
        grad_stats['max_grad'] = max_grad
        grad_stats['zero_grads'] = zero_count
        
        return grad_stats

    def _apply_warmup(self, step: int):
        """Apply learning rate warmup with proper implementation."""
        if not self.use_warmup or step >= self.warmup_steps:
            return

        # Calculate warmup progress (0 to 1)
        warmup_progress = min(1.0, step / self.warmup_steps)
    
        # Apply to all parameter groups
        for param_group in self.optimizer.param_groups:
            # Store original lr if not already stored
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']
        
            # Linear warmup from warmup_factor * initial_lr to initial_lr
            initial_lr = param_group['initial_lr']
            min_lr = initial_lr * self.warmup_factor
            current_lr = min_lr + (initial_lr - min_lr) * warmup_progress
        
            param_group['lr'] = current_lr

    def _log_training_stats(self, loss: float, grad_stats: Dict[str, float], epoch: int):
        """Log training statistics to TensorBoard."""
        # Log loss and learning rate
        self.writer.add_scalar('Loss/train_step', loss, self.global_step)
        self.writer.add_scalar('Learning_Rate/train_step', 
                              self.optimizer.param_groups[0]['lr'], self.global_step)
        
        # Log gradient statistics
        if self.log_grad_norms and grad_stats['total_norm'] > 0:
            self.writer.add_scalar('Gradients/total_norm', grad_stats['total_norm'], self.global_step)
            self.writer.add_scalar('Gradients/max_grad', grad_stats['max_grad'], self.global_step)
            self.writer.add_scalar('Gradients/zero_grads', grad_stats['zero_grads'], self.global_step)
        
        # Log memory usage
        if self.global_step % (self.log_frequency_iter * 2) == 0:
            gpu_mem = log_gpu_memory_usage("train_step")
            cpu_mem = get_cpu_memory_usage()
            if gpu_mem:
                self.writer.add_scalar('Memory/gpu_allocated_mb', 
                                     gpu_mem.get('allocated_mb', 0), self.global_step)
                self.writer.add_scalar('Memory/gpu_cached_mb', 
                                     gpu_mem.get('cached_mb', 0), self.global_step)
            self.writer.add_scalar('Memory/cpu_used_mb', 
                                 cpu_mem.get('used_mb', 0), self.global_step)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
            """Enhanced training epoch with comprehensive debugging."""
            self.model.train()
            total_loss = 0.0
            valid_batches = 0
        
            # ===== NEW: EPOCH START VALIDATION =====
            if epoch == 0:  # First epoch validation
                if not self._validate_batch_size():
                    return {'avg_loss': float('inf'), 'valid_batches': 0, 'skipped_batches': 0}
            # ===== END NEW SECTION =====
        
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [TRAIN]", leave=False)
        
            for batch_idx, (images, masks) in enumerate(pbar):
                try:
                    # ===== NEW: EARLY BATCH VALIDATION =====
                    # Quick validation for first few batches
                    if batch_idx < 3:
                        # Use pbar.write to maintain progress bar integrity
                        pbar.write(f"Batch {batch_idx}: images {images.shape}, masks {masks.shape}")
                    # ===== END NEW SECTION =====
                
                    # Move data to device
                    images = images.to(self.device, non_blocking=True)
                    masks = masks.to(self.device, non_blocking=True)
                    # Apply batch-level augmentations if available
                    if hasattr(self.train_loader.dataset, 'transform') and hasattr(self.train_loader.dataset.transform, 'apply_batch_augmentation'):
                        images, masks = self.train_loader.dataset.transform.apply_batch_augmentation(images, masks)
                    # Validate input data
                    if not self._validate_data_batch(images, masks, batch_idx):
                        self.training_stats['skipped_batches'] += 1
                        continue

                    # Apply warmup
                    if self.use_warmup:
                        self._apply_warmup(self.global_step)

                    # Forward pass
                    with autocast(enabled=self.mixed_precision):
                        outputs = self.model(images)
                        main_logits = outputs['out']
                    
                        # Calculate main loss
                        loss = self.criterion(main_logits, masks)
                    
                        # ===== MODIFIED: BETTER LOSS VALIDATION =====
                        # Enhanced loss validation
                        if torch.isnan(loss) or torch.isinf(loss):
                            if torch.isnan(loss):
                                self.training_stats['nan_batches'] += 1
                                print(f"❌ NaN loss detected at batch {batch_idx}")
                            if torch.isinf(loss):
                                self.training_stats['inf_batches'] += 1
                                print(f"❌ Inf loss detected at batch {batch_idx}")
                        
                            # Clear gradients and continue
                            self.optimizer.zero_grad()
                            continue
                    
                        # Enhanced loss threshold checking
                        if loss.item() > self.max_loss_threshold:
                            print(f"⚠️  Loss {loss.item():.4f} exceeds threshold {self.max_loss_threshold} at batch {batch_idx}")
                            if loss.item() > self.max_loss_threshold * 2:  # Very high loss
                                print(f"❌ Extremely high loss, skipping batch")
                                self.training_stats['skipped_batches'] += 1
                                self.optimizer.zero_grad()
                                continue
                        # ===== END MODIFIED SECTION =====

                        # Add auxiliary losses if enabled
                        if self.config['training'].get('use_aux_loss', False):
                            aux_loss_weight = self.config['training'].get('aux_loss_weight', 0.4)
                            aux_loss_total = 0
                        
                            if 'aux16' in outputs:
                                aux_loss = self.criterion(outputs['aux16'], masks)
                                if not (torch.isnan(aux_loss) or torch.isinf(aux_loss)):
                                    aux_loss_total += aux_loss
                            if 'aux32' in outputs:
                                aux_loss = self.criterion(outputs['aux32'], masks)
                                if not (torch.isnan(aux_loss) or torch.isinf(aux_loss)):
                                    aux_loss_total += aux_loss
                        
                            if aux_loss_total > 0:
                                loss = loss + aux_loss_total * aux_loss_weight

                    # Backward pass
                    scaled_loss = loss / self.gradient_accumulation_steps
                
                    if self.scaler:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    # Check gradients
                    grad_stats = self._check_gradients()
                
                    # Skip if gradients are invalid
                    if grad_stats['has_nan'] or grad_stats['has_inf']:
                        print(f"❌ Invalid gradients at batch {batch_idx}, skipping...")
                        self.optimizer.zero_grad()
                        self.training_stats['skipped_batches'] += 1
                        continue

                    # Optimizer step
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        # Clip gradients before unscaling for mixed precision
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
    
                        # Clip with a more aggressive threshold initially
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['training'].get('max_grad_norm', 5.0)
                        )
    
                        # Log extreme gradient norms
                        if grad_norm > self.config['training'].get('max_grad_norm', 5.0):
                            print(f"⚠️  Gradient norm {grad_norm:.2f} exceeded threshold, clipped")
    
                        # Step optimizer
                        if self.scaler:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
    
                        self.optimizer.zero_grad()

                    # Update statistics
                    total_loss += loss.item()
                    valid_batches += 1
                    self.global_step += 1
                
                    # Store training statistics
                    self.training_stats['loss_history'].append(loss.item())
                    self.training_stats['lr_history'].append(self.optimizer.param_groups[0]['lr'])
                    if grad_stats['total_norm'] > 0:
                        self.training_stats['grad_norms'].append(grad_stats['total_norm'])

                    # ===== MODIFIED: IMPROVED LOGGING =====
                    # Enhanced logging with batch size verification
                    if self.global_step % self.log_frequency_iter == 0:
                        self._log_training_stats(loss.item(), grad_stats, epoch)
    
                        # Update progress bar without print statements
                        pbar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                            'grad_norm': f"{grad_stats['total_norm']:.2f}",
                            'step': self.global_step
                        })
                    else:
                        # Still update loss in progress bar for every iteration
                        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                    # ===== END MODIFIED SECTION =====

                    # Clear cache periodically
                    if self.global_step % self.clear_cache_frequency_iter == 0:
                        clear_cuda_cache()

                    # Save first batch for debugging
                    if (self.debug_mode and batch_idx == 0 and epoch == 0 and 
                        self.config.get('debug', {}).get('save_first_batch', False)):
                        self._save_debug_batch(images, masks, outputs, batch_idx)

                except Exception as e:
                    print(f"❌ Error in batch {batch_idx}: {e}")
                    self.training_stats['skipped_batches'] += 1
                    if self.debug_mode:
                        import traceback
                        traceback.print_exc()
                    continue

            # Calculate average loss
            if valid_batches > 0:
                avg_loss = total_loss / valid_batches
            else:
                print(f"⚠️  WARNING: No valid batches in epoch {epoch}")
                avg_loss = float('inf')

            # ===== NEW: ENHANCED EPOCH LOGGING =====
            # Enhanced epoch summary
            pbar.write(f"\n📊 Epoch {epoch+1} Summary:")
            pbar.write(f"   Average Loss: {avg_loss:.4f}")
            pbar.write(f"   Valid Batches: {valid_batches}")
            pbar.write(f"   Skipped Batches: {self.training_stats['skipped_batches']}")
            pbar.write(f"   NaN Loss Batches: {self.training_stats['nan_batches']}")
            pbar.write(f"   Inf Loss Batches: {self.training_stats['inf_batches']}")
            # ===== END NEW SECTION =====

            # Log epoch statistics
            self.writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
            self.writer.add_scalar('Training/valid_batches', valid_batches, epoch)
            self.writer.add_scalar('Training/skipped_batches', self.training_stats['skipped_batches'], epoch)
        
            return {
                'avg_loss': avg_loss,
                'valid_batches': valid_batches,
                'skipped_batches': self.training_stats['skipped_batches']
            }

    def _save_debug_batch(self, images: torch.Tensor, masks: torch.Tensor, 
                         outputs: Dict[str, torch.Tensor], batch_idx: int):
        """Save first batch for debugging purposes."""
        try:
            debug_data = {
                'images_stats': {
                    'shape': list(images.shape),
                    'mean': images.mean().item(),
                    'std': images.std().item(),
                    'min': images.min().item(),
                    'max': images.max().item()
                },
                'masks_stats': {
                    'shape': list(masks.shape),
                    'unique_values': masks.unique().tolist(),
                    'min': masks.min().item(),
                    'max': masks.max().item()
                },
                'outputs_stats': {}
            }
        
            for key, output in outputs.items():
                debug_data['outputs_stats'][key] = {
                    'shape': list(output.shape),
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item()
                }
        
            # FIX 1: Define valid_mask before using it
            # Add class distribution in batch
            valid_mask = masks != self.config['dataset'].get('ignore_index', 255)
            unique_classes, counts = torch.unique(masks[valid_mask], return_counts=True)
            debug_data['batch_class_distribution'] = {
                int(cls): int(cnt) for cls, cnt in zip(unique_classes, counts)
            }
        
            # FIX 2: Add loss components if using combined loss
            if hasattr(self.criterion, 'ce_loss') and hasattr(self.criterion, 'dice_loss'):
                with torch.no_grad():
                    # Calculate individual loss components
                    ce_loss = self.criterion.ce_loss(outputs['out'], masks).item()
                    dice_loss = self.criterion.dice_loss(outputs['out'], masks).item()
                    debug_data['loss_components'] = {
                        'ce_loss': ce_loss,
                        'dice_loss': dice_loss,
                        'combined_loss': ce_loss * self.criterion.ce_weight_norm + dice_loss * self.criterion.dice_weight_norm
                    }
        
            # Save to file
            debug_file = self.debug_dir / f"first_batch_debug.json"
            with open(debug_file, 'w') as f:
                json.dump(debug_data, f, indent=2)
        
            pass
        
        except Exception as e:
            print(f"Failed to save debug batch: {e}")

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Enhanced validation with ALL 228 comprehensive metrics"""
        self.model.eval()
        total_val_loss = 0.0
        valid_batches = 0
    
        # Import enhanced metrics
        from research_finetuning.Part_3_Training.metrics import ComprehensiveMetrics
    
        # CelebAMask-HQ class names
        class_names = [
            'background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
            'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 'mouth',
            'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat'
        ]
    
        # Initialize ALL 228 metrics
        metrics_calculator = ComprehensiveMetrics(
            num_classes=self.config['model']['num_classes'],
            ignore_index=self.config['dataset'].get('ignore_index', 255),
            class_names=class_names
        )

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [VAL]", leave=False)

        for batch_idx, (images, masks) in enumerate(pbar):
            try:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                if not self._validate_data_batch(images, masks, batch_idx):
                    continue

                with autocast(enabled=self.mixed_precision):
                    outputs = self.model(images)
                    main_logits = outputs['out']
                    val_loss = self.criterion(main_logits, masks)

                if torch.isnan(val_loss) or torch.isinf(val_loss):
                    continue

                total_val_loss += val_loss.item()
                valid_batches += 1

                # Update ALL 228 metrics
                metrics_calculator.update(main_logits, masks)
                pbar.set_postfix(val_loss=val_loss.item())

                if batch_idx % 10 == 0:
                    clear_cuda_cache()

            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue

        if valid_batches == 0:
            return {'avg_val_loss': float('inf'), 'mIoU': 0.0}

        avg_val_loss = total_val_loss / valid_batches

        # Compute ALL 228 comprehensive metrics
        try:
            results = metrics_calculator.compute_metrics()
            overall = results['overall']
        
            # Log ALL key metrics to TensorBoard
            self.writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
            self.writer.add_scalar('Metrics/mIoU', overall['mIoU'], epoch)
            self.writer.add_scalar('Metrics/Mean_F1', overall['Mean_F1'], epoch)
            self.writer.add_scalar('Metrics/Pixel_Accuracy', overall['Pixel_Accuracy'], epoch)
            self.writer.add_scalar('Metrics/Mean_Precision', overall['Mean_Precision'], epoch)
            self.writer.add_scalar('Metrics/Mean_Recall', overall['Mean_Recall'], epoch)
            self.writer.add_scalar('Metrics/Frequency_Weighted_IoU', overall['Frequency_Weighted_IoU'], epoch)
            self.writer.add_scalar('Metrics/Overall_Accuracy', overall['Overall_Accuracy'], epoch)
            self.writer.add_scalar('Metrics/Classes_Present', overall['Classes_Present'], epoch)
        
            # Print comprehensive results
            print(f"Epoch {epoch+1} - ALL 228 METRICS COMPUTED:")
            print(f"  📊 mIoU: {overall['mIoU']:.4f} | Mean F1: {overall['Mean_F1']:.4f}")
            print(f"  📊 Pixel Accuracy: {overall['Pixel_Accuracy']:.4f} | Overall Accuracy: {overall['Overall_Accuracy']:.4f}")
            print(f"  📈 Precision: {overall['Mean_Precision']:.4f} | Recall: {overall['Mean_Recall']:.4f}")
        
            # Save detailed report every 10 epochs
            if epoch % 10 == 0 and hasattr(self, 'output_dir'):
                report_path = self.output_dir / f"complete_metrics_epoch_{epoch+1}.txt"
                metrics_calculator.save_detailed_report(report_path, results)
            
                # Save confusion matrix visualization
                cm_path = self.output_dir / f"confusion_matrix_epoch_{epoch+1}.png"
                metrics_calculator.plot_confusion_matrix(cm_path)
            
                print(f"  💾 Saved complete report: {report_path}")
                print(f"  📊 Saved confusion matrix: {cm_path}")
        
            return {
                'avg_val_loss': avg_val_loss, 
                'mIoU': overall['mIoU'],
                'Mean_F1': overall['Mean_F1'],
                'Pixel_Accuracy': overall['Pixel_Accuracy']
            }
        
        except Exception as e:
            print(f"Error calculating comprehensive metrics: {e}")
            return {'avg_val_loss': avg_val_loss, 'mIoU': 0.0}

    def train(self) -> None:
        """Enhanced training loop with early stopping and comprehensive monitoring."""
        print(f"\n🚀 Starting enhanced training for {self.epochs} epochs on {self.device}...")
        print(f"Debug mode: {self.debug_mode}")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        
        # ===== NEW: PRE-TRAINING VALIDATION =====
        # Validate setup before starting
        print("\n🔍 Pre-training validation...")
        if not self._validate_batch_size():
            print("❌ Training aborted due to batch size validation failure")
            return
        # ===== END NEW SECTION =====
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.epochs):
            epoch_start_time = time.time()
            
            # Clear cache and log memory
            clear_cuda_cache()
            if self.debug_mode:
                log_gpu_memory_usage(f"start_epoch_{epoch+1}")

            # Training phase
            train_metrics = self._train_epoch(epoch)
            
            print(f"Epoch {epoch+1} Training: Loss: {train_metrics['avg_loss']:.4f}, "
                  f"Valid batches: {train_metrics['valid_batches']}, "
                  f"Skipped: {train_metrics['skipped_batches']}")

            # ===== MODIFIED: ENHANCED FAILURE DETECTION =====
            # Enhanced training failure detection
            if train_metrics['valid_batches'] == 0:
                print("❌ CRITICAL ERROR: No valid training batches. Stopping training.")
                break
                
            if train_metrics['avg_loss'] == float('inf'):
                print("❌ CRITICAL ERROR: Training loss is infinite. Stopping training.")
                break
                
            # Check if too many batches are being skipped
            skip_ratio = train_metrics['skipped_batches'] / (train_metrics['valid_batches'] + train_metrics['skipped_batches'])
            if skip_ratio > 0.5:  # More than 50% skipped
                print(f"⚠️  WARNING: {skip_ratio*100:.1f}% of batches skipped. Training may be unstable.")
            # ===== END MODIFIED SECTION =====

            # Validation phase
            if (epoch + 1) % self.validation_frequency == 0:
                val_metrics = self._validate_epoch(epoch)
                current_miou = val_metrics['mIoU']

                # ===== NEW: VALIDATION FAILURE DETECTION =====
                # Check for validation issues
                val_train_ratio = val_metrics['avg_val_loss'] / train_metrics['avg_loss']
                if val_train_ratio > 5.0:  # Validation loss much higher than training
                    print(f"⚠️  WARNING: Validation loss ({val_metrics['avg_val_loss']:.4f}) >> Training loss ({train_metrics['avg_loss']:.4f})")
                    print(f"   Ratio: {val_train_ratio:.2f}x - This suggests overfitting or data mismatch")
                # ===== END NEW SECTION =====

                # Early stopping check
                if current_miou > self.best_miou + self.early_stopping_threshold:
                    self.best_miou = current_miou
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"🎉 New best mIoU: {self.best_miou:.4f}. Saved best model.")
                else:
                    self.epochs_without_improvement += 1
                    
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"⏹️  Early stopping after {self.epochs_without_improvement} epochs without improvement")
                    break

            # Learning rate scheduling
            if self.scheduler and not self.use_warmup:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['mIoU'])
                else:
                    self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.checkpoint_frequency_epoch == 0:
                self.save_checkpoint(epoch, is_best=False)

            epoch_time = time.time() - epoch_start_time
            print(f"⏱️  Epoch {epoch+1} completed in {epoch_time:.2f}s")
            
            # Log epoch time
            self.writer.add_scalar('Time/epoch_seconds', epoch_time, epoch)

        # Training summary
        total_time = time.time() - start_time
        print(f"\n🏁 Training completed in {total_time:.2f} seconds")
        print(f"🏆 Best mIoU: {self.best_miou:.4f}")
        print(f"📊 Total skipped batches: {self.training_stats['skipped_batches']}")
        print(f"📊 NaN loss batches: {self.training_stats['nan_batches']}")
        print(f"📊 Inf loss batches: {self.training_stats['inf_batches']}")
        
        # Save final training statistics
        if self.debug_mode:
            stats_file = self.debug_dir / "training_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(self.training_stats, f, indent=2, default=str)
            print(f"📁 Training statistics saved to {stats_file}")
        
        self.writer.close()

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint with enhanced metadata."""
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_miou': self.best_miou,
            'config': self.config,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'training_stats': self.training_stats,
            'epochs_without_improvement': self.epochs_without_improvement
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        torch.save(state, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(state, best_path)
            
        # Save epoch-specific checkpoint
        if (epoch + 1) % (self.checkpoint_frequency_epoch * 5) == 0:
            epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(state, epoch_path)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load checkpoint with enhanced error handling."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return

        try:
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.scaler and checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint.get('global_step', 0)
            self.best_miou = checkpoint.get('best_miou', -1.0)
            self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            
            if 'training_stats' in checkpoint:
                self.training_stats.update(checkpoint['training_stats'])
            
            print(f"Resumed from epoch {self.start_epoch}, "
                  f"global step {self.global_step}, "
                  f"best mIoU {self.best_miou:.4f}")
                  
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")


# Create alias for backward compatibility
Trainer = EnhancedTrainer