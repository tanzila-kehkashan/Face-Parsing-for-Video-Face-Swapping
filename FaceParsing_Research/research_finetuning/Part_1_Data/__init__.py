"""
Research Fine-tuning Project for Face Parsing
Multi-part project implementing face parsing with fine-tuning experiments
"""

# Part 1: Data Pipeline
from .celebamask_dataset import CelebAMaskHQDataset, create_data_loaders
from .augmentations import (
    FaceParsingAugmentation, 
    MixupAugmentation, 
    CutMixAugmentation, 
    TestTimeAugmentation
)
from .data_utils import (
    verify_dataset_structure,
    calculate_dataset_statistics,
    create_visualization_grid,
    validate_data_loader,
    create_class_legend,
    analyze_memory_usage
)

__version__ = "1.0.0"
__author__ = "Research Team"

# Export main components
__all__ = [
    # Part 1: Data Pipeline
    "CelebAMaskHQDataset",
    "create_data_loaders",
    "FaceParsingAugmentation",
    "MixupAugmentation", 
    "CutMixAugmentation",
    "TestTimeAugmentation",
    "verify_dataset_structure",
    "calculate_dataset_statistics",
    "create_visualization_grid",
    "validate_data_loader",
    "create_class_legend",
    "analyze_memory_usage"
]