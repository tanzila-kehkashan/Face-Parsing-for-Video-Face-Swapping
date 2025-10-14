"""
LoRA (Low-Rank Adaptation) layers for efficient fine-tuning
Implements custom LoRA layers for Conv2d operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math


class LoRAConv2d(nn.Module):
    """
    LoRA adapted Conv2d layer
    
    Implements low-rank decomposition for convolutional layers:
    W' = W + BA where B ∈ R^(out_channels × rank) and A ∈ R^(rank × in_channels × kernel_size × kernel_size)
    
    Args:
        conv: Original Conv2d layer to adapt
        rank: Rank of the low-rank decomposition
        alpha: Scaling factor for LoRA updates
        dropout: Dropout probability for LoRA weights
    """
    
    def __init__(
        self, 
        conv: nn.Conv2d,
        rank: int = 4,
        alpha: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.conv = conv
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze original convolution weights
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False
        
        # Get conv parameters
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        
        # Create LoRA parameters
        # A: rank × (in_channels // groups) × kernel_size × kernel_size
        # B: out_channels × rank × 1 × 1
        self.lora_A = nn.Parameter(
            torch.zeros(rank, self.in_channels // self.groups, *self.kernel_size)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_channels, rank, 1, 1)
        )
        
        # Optional dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Initialize LoRA weights
        self.reset_parameters()
        
        # Enable LoRA by default
        self.enabled = True
    
    def reset_parameters(self):
        """Initialize LoRA parameters"""
        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initialize B with zeros
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor with LoRA adaptation applied
        """
        # Original convolution
        out = self.conv(x)
        
        if self.enabled and self.training:
            # Apply LoRA adaptation
            # First apply A projection
            lora_out = F.conv2d(
                x, 
                self.lora_A,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
            
            # Apply dropout
            lora_out = self.dropout(lora_out)
            
            # Then apply B projection
            lora_out = F.conv2d(lora_out, self.lora_B)
            
            # Scale and add to original output
            out = out + lora_out * self.scaling
        
        return out
    
    def merge_weights(self):
        """Merge LoRA weights into the original convolution (for inference)"""
        if self.enabled:
            # Compute merged weight: W' = W + scaling * B @ A
            # Reshape for matrix multiplication
            lora_weight = torch.mm(
                self.lora_B.squeeze(-1).squeeze(-1),  # (out_channels, rank)
                self.lora_A.flatten(1)  # (rank, in_channels * k * k)
            ).reshape_as(self.conv.weight)
            
            self.conv.weight.data += lora_weight * self.scaling
            self.enabled = False
    
    def enable_lora(self, enabled: bool = True):
        """Enable or disable LoRA adaptation"""
        self.enabled = enabled
    
    @property
    def lora_params(self) -> int:
        """Number of trainable LoRA parameters"""
        return self.lora_A.numel() + self.lora_B.numel()


class LoRALinear(nn.Module):
    """
    LoRA adapted Linear layer (for potential future use)
    
    Args:
        linear: Original Linear layer to adapt
        rank: Rank of the low-rank decomposition
        alpha: Scaling factor for LoRA updates
        dropout: Dropout probability for LoRA weights
    """
    
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 4,
        alpha: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        
        # Create LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))
        
        # Optional dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        self.reset_parameters()
        self.enabled = True
    
    def reset_parameters(self):
        """Initialize LoRA parameters"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation"""
        out = self.linear(x)
        
        if self.enabled and self.training:
            lora_out = x @ self.lora_A.T
            lora_out = self.dropout(lora_out)
            lora_out = lora_out @ self.lora_B.T
            out = out + lora_out * self.scaling
        
        return out
    
    def merge_weights(self):
        """Merge LoRA weights into the original layer"""
        if self.enabled:
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.enabled = False
    
    @property
    def lora_params(self) -> int:
        """Number of trainable LoRA parameters"""
        return self.lora_A.numel() + self.lora_B.numel()


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 4,
    alpha: int = 8,
    target_modules: Optional[list] = None,
    exclude_modules: Optional[list] = None
) -> Tuple[nn.Module, dict]:
    """
    Apply LoRA to specified modules in a model
    
    Args:
        model: Model to apply LoRA to
        rank: LoRA rank
        alpha: LoRA scaling factor
        target_modules: List of module name patterns to target (None = all Conv2d)
        exclude_modules: List of module name patterns to exclude
        
    Returns:
        Modified model and statistics dictionary
    """
    if target_modules is None:
        target_modules = ['']  # Target all modules by default
    
    if exclude_modules is None:
        exclude_modules = []
    
    lora_modules = {}
    total_params = 0
    lora_params = 0
    
    def should_apply_lora(name: str) -> bool:
        """Check if LoRA should be applied to this module"""
        # Check if module should be targeted
        is_target = any(target in name for target in target_modules)
        # Check if module should be excluded
        is_excluded = any(exclude in name for exclude in exclude_modules)
        return is_target and not is_excluded
    
    # Replace Conv2d layers with LoRA versions
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and should_apply_lora(name):
            # Get parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            parent = model
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            
            # Create LoRA layer
            lora_conv = LoRAConv2d(module, rank=rank, alpha=alpha)
            setattr(parent, attr_name, lora_conv)
            lora_modules[name] = lora_conv
            
            # Count parameters
            total_params += sum(p.numel() for p in module.parameters())
            lora_params += lora_conv.lora_params
    
    # Freeze all non-LoRA parameters
    # Freeze all non-LoRA parameters (only if LoRA is enabled)
    if rank > 0:  # Only freeze if LoRA is enabled
        for name, param in model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
        print(f"✅ LoRA enabled: {rank} rank, {len(lora_modules)} modules")
    else:
        # LoRA disabled - enable full fine-tuning
        for name, param in model.named_parameters():
            param.requires_grad = True
        print("✅ LoRA disabled - Full fine-tuning enabled")
    
    stats = {
        'total_modules': len(list(model.modules())),
        'lora_modules': len(lora_modules),
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'lora_params': lora_params,
        'compression_ratio': lora_params / total_params if total_params > 0 else 0
    }
    
    return model, stats


if __name__ == "__main__":
    """Test LoRA layers"""
    print("Testing LoRA layers...")
    
    # Test LoRAConv2d
    conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    lora_conv = LoRAConv2d(conv, rank=4, alpha=8)
    
    # Test forward pass
    x = torch.randn(2, 64, 32, 32)
    out = lora_conv(x)
    print(f"LoRAConv2d output shape: {out.shape}")
    print(f"LoRA parameters: {lora_conv.lora_params:,}")
    
    # Test apply_lora_to_model
    from torchvision.models import resnet18
    model = resnet18()
    model, stats = apply_lora_to_model(model, rank=4, alpha=8)
    
    print("\nLoRA application stats:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    
    print("\nLoRA layer test completed!")