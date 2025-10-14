# C:\SwapFace2Pon\FaceParsing_Research\research_finetuning\__init__.py

"""
Main package for the Face Parsing Fine-tuning Research Project.
Organizes different parts of the research pipeline.
"""

# Import from Part_1_Data
from .Part_1_Data.celebamask_dataset import CelebAMaskHQDataset, create_data_loaders
from .Part_1_Data.augmentations import (
    FaceParsingAugmentation,
    # MixupAugmentation, # Uncomment if you add these later
    # CutMixAugmentation, # Uncomment if you add these later
    # TestTimeAugmentation # Uncomment if you add these later
)
from .Part_1_Data.data_utils import (
    verify_dataset_structure,
    calculate_dataset_statistics,
    create_visualization_grid,
    validate_data_loader,
    create_class_legend,
    analyze_memory_usage
)

# Import from Part_2_Model
from .Part_2_Model.bisenet import BiSeNet, create_bisenet
from .Part_2_Model.lora_layers import LoRAConv2d, LoRALinear, apply_lora_to_model
from .Part_2_Model.model_builder import BiSeNetLoRA, build_model_from_config, load_config as load_model_config # Renamed to avoid conflict
from .Part_2_Model.model_utils import (
    count_parameters,
    analyze_model_memory,
    profile_model_speed,
    visualize_lora_weights,
    check_gradient_flow,
    save_model_summary
)

# Import from Part_3_Training (NEW)
from .Part_3_Training.loss_functions import WeightedCrossEntropyLoss, DiceLoss, CombinedLoss
from .Part_3_Training.metrics import calculate_miou
from .Part_3_Training.memory_manager import clear_cuda_cache, log_gpu_memory_usage, get_cpu_memory_usage
from .Part_3_Training.trainer import Trainer
from .Part_3_Training.train import main as run_training # Expose the main training function

__all__ = [
    # Part 1: Data Pipeline
    "CelebAMaskHQDataset",
    "create_celebamask_hq_loaders",
    "FaceParsingAugmentation",
    # "MixupAugmentation",
    # "CutMixAugmentation",
    # "TestTimeAugmentation",
    "verify_dataset_structure",
    "calculate_dataset_statistics",
    "create_visualization_grid",
    "validate_data_loader",
    "create_class_legend",
    "analyze_memory_usage",

    # Part 2: Model Architecture
    "BiSeNet",
    "create_bisenet",
    "LoRAConv2d",
    "LoRALinear",
    "apply_lora_to_model",
    "BiSeNetLoRA",
    "build_model_from_config",
    "load_model_config", # Renamed
    "count_parameters",
    "analyze_model_memory",
    "profile_model_speed",
    "visualize_lora_weights",
    "check_gradient_flow",
    "save_model_summary",

    # Part 3: Training System (NEW)
    "WeightedCrossEntropyLoss",
    "DiceLoss",
    "CombinedLoss",
    "calculate_miou",
    "clear_cuda_cache",
    "log_gpu_memory_usage",
    "get_cpu_memory_usage",
    "Trainer",
    "run_training" # Expose the main training function
]

__version__ = "1.0.0"
