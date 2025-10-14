"""
Part 0: Setup and Dependencies
This module handles dependency checking, installation, and GPU verification
"""

from .check_dependencies import DependencyChecker
from .install_requirements import RequirementsInstaller
from .verify_gpu import GPUVerifier
from .run_setup import SetupRunner

__all__ = [
    "DependencyChecker",
    "RequirementsInstaller", 
    "GPUVerifier",
    "SetupRunner"
]

__version__ = "1.0.0"