"""
Part 2: Model Architecture
BiSeNet with LoRA adaptation for face parsing
"""

# Import main classes and functions for easier access
from .bisenet import BiSeNet, create_bisenet
from .lora_layers import LoRAConv2d, LoRALinear, apply_lora_to_model
from .model_builder import BiSeNetLoRA, build_model_from_config, load_config
from .model_utils import (
    count_parameters,
    analyze_model_memory,
    profile_model_speed,
    visualize_lora_weights,
    check_gradient_flow,
    save_model_summary
)

# Define what's available when using "from Part_2_Model import *"
__all__ = [
    # Core model
    'BiSeNet',
    'create_bisenet',
    
    # LoRA components
    'LoRAConv2d',
    'LoRALinear',
    'apply_lora_to_model',
    
    # Model builder
    'BiSeNetLoRA',
    'build_model_from_config',
    'load_config',
    
    # Utilities
    'count_parameters',
    'analyze_model_memory',
    'profile_model_speed',
    'visualize_lora_weights',
    'check_gradient_flow',
    'save_model_summary'
]

# Package version
__version__ = '0.1.0'

# Package info
__author__ = 'Your Name'
__email__ = 'your.email@example.com'
__description__ = 'BiSeNet with LoRA adaptation for face parsing research'