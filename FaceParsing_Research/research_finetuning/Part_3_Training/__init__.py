"""
Part 3: Training System
This module contains components for the training loop, loss functions, metrics, and memory management.
"""

from .loss_functions import WeightedCrossEntropyLoss, DiceLoss, CombinedLoss
from .metrics import calculate_miou, ComprehensiveMetrics  # Added ComprehensiveMetrics
from .memory_manager import clear_cuda_cache, log_gpu_memory_usage, get_cpu_memory_usage
from .trainer import Trainer
from .train import main as train_main

__all__ = [
    # Loss Functions
    "WeightedCrossEntropyLoss",
    "DiceLoss", 
    "CombinedLoss",

    # Metrics
    "calculate_miou",
    "ComprehensiveMetrics",  # Added new comprehensive metrics

    # Memory Management
    "clear_cuda_cache",
    "log_gpu_memory_usage",
    "get_cpu_memory_usage",

    # Trainer
    "Trainer",
    "train_main"
]

__version__ = "1.0.0"