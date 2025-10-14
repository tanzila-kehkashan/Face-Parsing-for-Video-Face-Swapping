"""
Utility functions for model handling
Includes weight initialization, model analysis, and debugging tools
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns


def count_parameters(model: nn.Module, trainable_only: bool = False) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        trainable_only: Count only trainable parameters
        
    Returns:
        Dictionary with parameter counts by module type
    """
    param_counts = {}
    total_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            module_type = module.__class__.__name__
            
            if trainable_only:
                n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            else:
                n_params = sum(p.numel() for p in module.parameters())
            
            if n_params > 0:
                if module_type not in param_counts:
                    param_counts[module_type] = 0
                param_counts[module_type] += n_params
                total_params += n_params
    
    param_counts['Total'] = total_params
    return param_counts


def analyze_model_memory(model: nn.Module, input_shape: Tuple[int, ...], 
                        device: str = 'cuda') -> Dict[str, float]:
    """
    Analyze model memory usage
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch_size, channels, height, width)
        device: Device to run analysis on
        
    Returns:
        Dictionary with memory statistics in MB
    """
    model = model.to(device)
    model.eval()
    
    # Clear cache
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Measure memory for forward pass
    x = torch.randn(*input_shape, device=device)
    
    if device == 'cuda':
        start_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        with torch.no_grad():
            outputs = model(x)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        end_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    else:
        # CPU memory measurement is approximate
        import psutil
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024**2
        
        with torch.no_grad():
            outputs = model(x)
        
        end_memory = process.memory_info().rss / 1024**2
        peak_memory = end_memory
    
    # Calculate sizes
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    
    # Estimate activation size (very rough estimate)
    if isinstance(outputs, dict):
        output_size = sum(
            o.numel() * o.element_size() for o in outputs.values()
        ) / 1024**2
    else:
        output_size = outputs.numel() * outputs.element_size() / 1024**2
    
    memory_stats = {
        'param_memory_mb': param_size,
        'peak_memory_mb': peak_memory,
        'forward_memory_mb': peak_memory - start_memory,
        'output_memory_mb': output_size,
        'estimated_gradient_memory_mb': param_size * 2,  # Rough estimate
    }
    
    return memory_stats


def profile_model_speed(model: nn.Module, input_shape: Tuple[int, ...], 
                       num_runs: int = 100, device: str = 'cuda') -> Dict[str, float]:
    """
    Profile model inference speed
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        num_runs: Number of runs for averaging
        device: Device to run on
        
    Returns:
        Dictionary with timing statistics
    """
    model = model.to(device)
    model.eval()
    
    # Warmup
    x = torch.randn(*input_shape, device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    
    # Synchronize
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Time forward passes
    import time
    times = []
    
    for _ in range(num_runs):
        x = torch.randn(*input_shape, device=device)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'fps': 1000 / np.mean(times)
    }


def visualize_lora_weights(model: nn.Module, save_path: Optional[Path] = None):
    """
    Visualize LoRA weight distributions
    
    Args:
        model: Model with LoRA layers
        save_path: Path to save visualization
    """
    from .lora_layers import LoRAConv2d
    
    # Collect LoRA weights
    lora_weights = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LoRAConv2d):
            lora_weights[f"{name}_A"] = module.lora_A.data.cpu().numpy().flatten()
            lora_weights[f"{name}_B"] = module.lora_B.data.cpu().numpy().flatten()
    
    if not lora_weights:
        print("No LoRA weights found in model")
        return
    
    # Create visualization
    n_weights = len(lora_weights)
    fig, axes = plt.subplots(n_weights, 1, figsize=(10, 3 * n_weights))
    
    if n_weights == 1:
        axes = [axes]
    
    for idx, (name, weights) in enumerate(lora_weights.items()):
        ax = axes[idx]
        ax.hist(weights, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.set_title(f"Weight Distribution: {name}")
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Frequency")
        
        # Add statistics
        mean = np.mean(weights)
        std = np.std(weights)
        ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.4f}')
        ax.axvline(mean + std, color='orange', linestyle='--', alpha=0.5)
        ax.axvline(mean - std, color='orange', linestyle='--', alpha=0.5)
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def check_gradient_flow(model: nn.Module, save_path: Optional[Path] = None):
    """
    Check gradient flow through the model
    
    Args:
        model: PyTorch model (after backward pass)
        save_path: Path to save visualization
    """
    # Collect gradients
    ave_grads = []
    max_grads = []
    layers = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().cpu().item())
            max_grads.append(param.grad.abs().max().cpu().item())
    
    if not layers:
        print("No gradients found. Run backward() first.")
        return
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.7)
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient Flow - Average")
    plt.xticks(np.arange(len(ave_grads)), layers, rotation=90)
    
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.7, color='red')
    plt.xlabel("Layers")
    plt.ylabel("Maximum Gradient")
    plt.title("Gradient Flow - Maximum")
    plt.xticks(np.arange(len(max_grads)), layers, rotation=90)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gradient flow visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_model_outputs(model1: nn.Module, model2: nn.Module, 
                         input_tensor: torch.Tensor) -> Dict[str, float]:
    """
    Compare outputs of two models
    
    Args:
        model1: First model
        model2: Second model
        input_tensor: Input tensor
        
    Returns:
        Dictionary with comparison metrics
    """
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        out1 = model1(input_tensor)
        out2 = model2(input_tensor)
    
    # Handle dictionary outputs
    if isinstance(out1, dict):
        out1 = out1['out']
    if isinstance(out2, dict):
        out2 = out2['out']
    
    # Calculate differences
    abs_diff = torch.abs(out1 - out2)
    rel_diff = abs_diff / (torch.abs(out1) + 1e-8)
    
    metrics = {
        'mean_abs_diff': abs_diff.mean().item(),
        'max_abs_diff': abs_diff.max().item(),
        'mean_rel_diff': rel_diff.mean().item(),
        'max_rel_diff': rel_diff.max().item(),
        'cosine_similarity': torch.cosine_similarity(
            out1.flatten(), out2.flatten(), dim=0
        ).item()
    }
    
    return metrics


def save_model_summary(model: nn.Module, save_path: Union[str, Path], 
                      input_shape: Tuple[int, ...] = (1, 3, 384, 384)):
    """
    Save detailed model summary to file
    
    Args:
        model: PyTorch model
        save_path: Path to save summary
        input_shape: Input shape for analysis
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'architecture': str(model),
        'parameter_counts': count_parameters(model),
        'trainable_parameters': count_parameters(model, trainable_only=True),
        'memory_analysis': analyze_model_memory(model, input_shape, device='cpu'),
    }
    
    # Add layer information
    layer_info = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            info = {
                'name': name,
                'type': module.__class__.__name__,
                'parameters': sum(p.numel() for p in module.parameters()),
                'trainable': any(p.requires_grad for p in module.parameters())
            }
            
            # Add specific info for certain layers
            if hasattr(module, 'in_channels'):
                info['in_channels'] = module.in_channels
            if hasattr(module, 'out_channels'):
                info['out_channels'] = module.out_channels
            if hasattr(module, 'kernel_size'):
                info['kernel_size'] = module.kernel_size
            
            layer_info.append(info)
    
    summary['layers'] = layer_info
    
    # Save as JSON
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Model summary saved to: {save_path}")


if __name__ == "__main__":
    """Test model utilities"""
    print("Testing model utilities...")
    
    # Create a simple model for testing
    from .bisenet import create_bisenet
    from .lora_layers import apply_lora_to_model
    
    # Create model
    model = create_bisenet(num_classes=19)
    model, _ = apply_lora_to_model(model)
    
    # Test parameter counting
    param_counts = count_parameters(model)
    print("\nParameter counts by module type:")
    for module_type, count in param_counts.items():
        print(f"  {module_type}: {count:,}")
    
    # Test memory analysis
    memory_stats = analyze_model_memory(model, (2, 3, 384, 384), device='cpu')
    print("\nMemory analysis:")
    for key, value in memory_stats.items():
        print(f"  {key}: {value:.2f} MB")
    
    # Test model summary
    save_model_summary(model, "test_model_summary.json")
    
    # Clean up
    if Path("test_model_summary.json").exists():
        Path("test_model_summary.json").unlink()
    
    print("\nModel utilities test completed!")