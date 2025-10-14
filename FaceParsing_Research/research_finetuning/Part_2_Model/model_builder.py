"""
Model builder for BiSeNet with LoRA adaptation
Handles model creation, loading, and configuration
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Union, List, Tuple
import json
import yaml
from datetime import datetime

from .bisenet import BiSeNet, create_bisenet
from .lora_layers import apply_lora_to_model, LoRAConv2d


class BiSeNetLoRA(nn.Module):
    """
    BiSeNet model with LoRA adaptation
    
    This wrapper manages the BiSeNet model with LoRA layers for efficient fine-tuning.
    It handles model creation, weight loading, and LoRA configuration.
    
    Args:
        num_classes: Number of segmentation classes
        backbone: Backbone architecture ('resnet18' or 'resnet34')
        pretrained_path: Path to pretrained weights
        lora_config: LoRA configuration dictionary
    """
    
    def __init__(
        self,
        num_classes: int = 19,
        backbone: str = 'resnet18',
        pretrained_path: Optional[str] = None,
        lora_config: Optional[Dict] = None
    ):
        super().__init__()
        
        # Default LoRA configuration
        self.lora_config = {
            'rank': 64,           # Increased for more capacity
            'alpha': 64,          # Keep 1:1 ratio with rank
            'dropout': 0.1,       # Standard dropout
            'target_modules': None,
            'exclude_modules': ['conv1', 'conv_pred']  # Exclude first and prediction layers
        }
        
        if lora_config:
            self.lora_config.update(lora_config)
        
        # Create base model
        self.model = create_bisenet(
            num_classes=num_classes,
            backbone=backbone,
            pretrained_path=pretrained_path
        )
        
        # Apply LoRA (only if rank > 0)
        if self.lora_config['rank'] > 0:
            self.model, self.lora_stats = apply_lora_to_model(
                self.model,
                rank=self.lora_config['rank'],
                alpha=self.lora_config['alpha'],
                target_modules=self.lora_config['target_modules'],
                exclude_modules=self.lora_config['exclude_modules']
            )
            print(f"✅ LoRA applied: rank={self.lora_config['rank']}")
        else:
            # No LoRA - set up stats for full fine-tuning
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.lora_stats = {
                'total_modules': len(list(self.model.modules())),
                'lora_modules': 0,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'lora_params': 0,
                'compression_ratio': 1.0
            }
            print(f"✅ Full fine-tuning: {trainable_params:,} trainable parameters")
        
        # Store original state dict for restoration
        self.original_state_dict = None
        self._save_original_weights()
    
    def _save_original_weights(self):
        """Save original model weights for later restoration"""
        self.original_state_dict = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:  # Only save frozen weights
                self.original_state_dict[name] = param.data.clone()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model"""
        return self.model(x)
    
    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Get only LoRA parameters for optimization"""
        lora_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'lora_' in name:
                lora_params.append(param)
        return lora_params
    
    def get_parameter_groups(self, base_lr: float = 1e-4) -> List[Dict]:
        """
        Get parameter groups with different learning rates
        
        Args:
            base_lr: Base learning rate
            
        Returns:
            List of parameter groups for optimizer
        """
        # LoRA parameters
        lora_params = self.get_lora_parameters()
        
        # If we have any non-LoRA trainable parameters (shouldn't happen with proper setup)
        other_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'lora_' not in name:
                other_params.append(param)
        
        param_groups = [
            {'params': lora_params, 'lr': base_lr, 'name': 'lora'}
        ]
        
        if other_params:
            param_groups.append({
                'params': other_params, 
                'lr': base_lr * 0.1, 
                'name': 'other'
            })
        
        return param_groups
    
    def merge_lora_weights(self):
        """Merge LoRA weights into base model for inference"""
        for module in self.model.modules():
            if isinstance(module, LoRAConv2d):
                module.merge_weights()
    
    def restore_original_weights(self):
        """Restore original model weights"""
        if self.original_state_dict is None:
            raise ValueError("Original weights not saved!")
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.original_state_dict:
                    param.data.copy_(self.original_state_dict[name])
    
    def save_lora_weights(self, save_path: Union[str, Path]):
        """
        Save only LoRA weights
        
        Args:
            save_path: Path to save LoRA weights
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract LoRA weights
        lora_state_dict = {}
        for name, param in self.model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                lora_state_dict[name] = param.data
        
        # Save with metadata
        checkpoint = {
            'lora_weights': lora_state_dict,
            'lora_config': self.lora_config,
            'lora_stats': self.lora_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, save_path)
        print(f"LoRA weights saved to: {save_path}")
    
    def load_lora_weights(self, load_path: Union[str, Path]):
        """
        Load LoRA weights
        
        Args:
            load_path: Path to LoRA weights file
        """
        load_path = Path(load_path)
        checkpoint = torch.load(load_path, map_location='cpu')
        
        # Load LoRA weights
        lora_weights = checkpoint['lora_weights']
        model_state_dict = self.model.state_dict()
        
        for name, param in lora_weights.items():
            if name in model_state_dict:
                model_state_dict[name].copy_(param)
        
        print(f"LoRA weights loaded from: {load_path}")
    
    def print_model_info(self):
        """Print model information and statistics"""
        print("\n" + "="*60)
        print("BiSeNet-LoRA Model Information")
        print("="*60)
        
        print("\nLoRA Configuration:")
        for key, value in self.lora_config.items():
            print(f"  {key}: {value}")
        
        print("\nModel Statistics:")
        for key, value in self.lora_stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Count LoRA modules by type
        lora_conv_count = sum(1 for m in self.model.modules() if isinstance(m, LoRAConv2d))
        print(f"\nLoRA Modules:")
        print(f"  LoRAConv2d: {lora_conv_count}")
        
        print("="*60 + "\n")


def build_model_from_config(config: Dict) -> BiSeNetLoRA:
    """
    Build model from configuration dictionary
    
    Args:
        config: Configuration dictionary
        
    Returns:
        BiSeNetLoRA model
    """
    model_config = config.get('model', {})
    
    # Extract model parameters
    num_classes = model_config.get('num_classes', 19)
    backbone = model_config.get('backbone', 'resnet18')
    pretrained_path = model_config.get('pretrained_path', None)
    
    # Extract LoRA configuration
    lora_config = {
        'rank': model_config.get('lora_rank', 4),
        'alpha': model_config.get('lora_alpha', 8),
        'dropout': model_config.get('lora_dropout', 0.0),
        'target_modules': model_config.get('lora_target_modules', None),
        'exclude_modules': model_config.get('lora_exclude_modules', ['conv1', 'conv_pred'])
    }
    
    # Create model
    model = BiSeNetLoRA(
        num_classes=num_classes,
        backbone=backbone,
        pretrained_path=pretrained_path,
        lora_config=lora_config
    )
    
    return model


def load_config(config_path: Union[str, Path]) -> Dict:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


if __name__ == "__main__":
    """Test model builder"""
    print("Testing BiSeNet-LoRA model builder...")
    
    # Test configuration
    config = {
        'model': {
            'num_classes': 19,
            'backbone': 'resnet18',
            'pretrained_path': None,
            'lora_rank': 4,
            'lora_alpha': 8,
            'lora_exclude_modules': ['conv1', 'conv_pred']
        }
    }
    
    # Build model
    model = build_model_from_config(config)
    model.print_model_info()
    
    # Test forward pass
    x = torch.randn(2, 3, 384, 384)
    outputs = model(x)
    
    print("\nForward pass test:")
    for key, tensor in outputs.items():
        print(f"  {key}: {tensor.shape}")
    
    # Test parameter groups
    param_groups = model.get_parameter_groups(base_lr=5e-6)
    print("\nParameter groups:")
    for group in param_groups:
        print(f"  {group['name']}: {len(group['params'])} parameters, lr={group['lr']}")
    
    # Test save/load
    print("\nTesting save/load...")
    save_path = Path("test_lora_weights.pth")
    model.save_lora_weights(save_path)
    
    # Clean up
    if save_path.exists():
        save_path.unlink()
    
    print("\nModel builder test completed!")