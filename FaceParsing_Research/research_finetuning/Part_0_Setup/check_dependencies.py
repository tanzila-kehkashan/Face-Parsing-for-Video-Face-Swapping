"""
Dependency Checker for Face Parsing Fine-tuning
Checks all required dependencies and system compatibility
"""

import sys
import platform
import subprocess
import importlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class DependencyChecker:
    """Check and verify all project dependencies"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system": {},
            "python": {},
            "packages": {},
            "cuda": {},
            "recommendations": []
        }
    
    def check_system(self) -> Dict:
        """Check system information"""
        print("\n🔍 Checking System Information...")
        
        system_info = {
            "platform": platform.platform(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_version_info": list(sys.version_info),
        }
        
        # Check OS specifics
        if platform.system() == "Windows":
            system_info["os"] = "Windows"
            system_info["os_version"] = platform.win32_ver()[0]
        elif platform.system() == "Linux":
            system_info["os"] = "Linux"
            try:
                with open("/etc/os-release") as f:
                    os_info = dict(line.strip().split("=", 1) for line in f if "=" in line)
                    system_info["os_version"] = os_info.get("PRETTY_NAME", "Unknown").strip('"')
            except:
                system_info["os_version"] = "Unknown"
        else:
            system_info["os"] = platform.system()
            system_info["os_version"] = platform.version()
        
        self.results["system"] = system_info
        
        print(f"✓ OS: {system_info['os']} {system_info.get('os_version', '')}")
        print(f"✓ Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        
        return system_info
    
    def check_python_version(self) -> bool:
        """Verify Python version compatibility"""
        print("\n🐍 Checking Python Version...")
        
        min_version = (3, 8)
        max_version = (3, 11)
        current_version = sys.version_info[:2]
        
        is_compatible = min_version <= current_version <= max_version
        
        self.results["python"] = {
            "version": f"{current_version[0]}.{current_version[1]}",
            "compatible": is_compatible,
            "min_required": f"{min_version[0]}.{min_version[1]}",
            "max_tested": f"{max_version[0]}.{max_version[1]}"
        }
        
        if is_compatible:
            print(f"✓ Python {current_version[0]}.{current_version[1]} is compatible")
        else:
            print(f"✗ Python {current_version[0]}.{current_version[1]} is not compatible")
            print(f"  Required: Python {min_version[0]}.{min_version[1]} - {max_version[0]}.{max_version[1]}")
            self.results["recommendations"].append(
                f"Install Python {min_version[0]}.{min_version[1]} or higher (up to {max_version[0]}.{max_version[1]})"
            )
        
        return is_compatible
    
    def check_cuda(self) -> Dict:
        """Check CUDA and GPU availability"""
        print("\n🎮 Checking CUDA and GPU...")
        
        cuda_info = {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_devices": [],
            "cuda_version": None,
            "cudnn_version": None,
            "driver_version": None
        }
        
        try:
            import torch
            
            cuda_info["cuda_available"] = torch.cuda.is_available()
            cuda_info["gpu_count"] = torch.cuda.device_count()
            
            if cuda_info["cuda_available"]:
                cuda_info["cuda_version"] = torch.version.cuda
                
                # Get GPU details
                for i in range(cuda_info["gpu_count"]):
                    gpu_props = torch.cuda.get_device_properties(i)
                    gpu_info = {
                        "index": i,
                        "name": gpu_props.name,
                        "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                        "memory_total_gb": gpu_props.total_memory / (1024**3),
                        "memory_allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                        "memory_reserved_gb": torch.cuda.memory_reserved(i) / (1024**3),
                    }
                    cuda_info["gpu_devices"].append(gpu_info)
                    
                    print(f"✓ GPU {i}: {gpu_info['name']}")
                    print(f"  Memory: {gpu_info['memory_total_gb']:.1f} GB")
                    print(f"  Compute Capability: {gpu_info['compute_capability']}")
                
                # Get driver version
                try:
                    nvidia_smi = subprocess.run(
                        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                        capture_output=True, text=True
                    )
                    if nvidia_smi.returncode == 0:
                        cuda_info["driver_version"] = nvidia_smi.stdout.strip()
                        print(f"✓ NVIDIA Driver: {cuda_info['driver_version']}")
                except:
                    pass
                
                # Check cuDNN
                cudnn_available = torch.backends.cudnn.is_available()
                if cudnn_available:
                    cuda_info["cudnn_version"] = torch.backends.cudnn.version()
                    print(f"✓ cuDNN: {cuda_info['cudnn_version']}")
            else:
                print("✗ CUDA is not available")
                self.results["recommendations"].append(
                    "Install CUDA 11.8 and appropriate NVIDIA drivers for your GPU"
                )
                
        except ImportError:
            print("✗ PyTorch not installed")
            self.results["recommendations"].append("Install PyTorch with CUDA support")
        
        self.results["cuda"] = cuda_info
        return cuda_info
    
    def check_packages(self) -> Dict:
        """Check required Python packages"""
        print("\n📦 Checking Required Packages...")
        
        required_packages = {
            # Core ML
            "torch": {"min_version": "2.0.0", "critical": True},
            "torchvision": {"min_version": "0.15.0", "critical": True},
            "numpy": {"min_version": "1.24.0", "critical": True},
            
            # Image processing
            "cv2": {"module": "cv2", "package": "opencv-python", "critical": True},
            "PIL": {"module": "PIL", "package": "Pillow", "critical": True},
            "albumentations": {"min_version": "1.3.0", "critical": True},
            
            # ML tools
            "segmentation_models_pytorch": {"min_version": "0.3.0", "critical": True},
            "einops": {"min_version": "0.6.0", "critical": True},
            "timm": {"min_version": "0.9.0", "critical": False},
            
            # Utilities
            "tqdm": {"critical": True},
            "yaml": {"module": "yaml", "package": "pyyaml", "critical": True},
            "matplotlib": {"critical": True},
            "pandas": {"critical": False},
            "tensorboard": {"critical": False},
            
            # Development
            "jupyter": {"critical": False},
            "pytest": {"critical": False},
        }
        
        package_status = {}
        missing_critical = []
        missing_optional = []
        
        for package_name, requirements in required_packages.items():
            module_name = requirements.get("module", package_name)
            install_name = requirements.get("package", package_name)
            is_critical = requirements.get("critical", True)
            min_version = requirements.get("min_version", None)
            
            try:
                module = importlib.import_module(module_name)
                version = getattr(module, "__version__", "Unknown")
                
                status = {
                    "installed": True,
                    "version": version,
                    "critical": is_critical,
                    "meets_requirement": True
                }
                
                # Check version if specified
                if min_version and version != "Unknown":
                    try:
                        from packaging import version as pkg_version
                        status["meets_requirement"] = pkg_version.parse(version) >= pkg_version.parse(min_version)
                    except:
                        # Fallback to simple string comparison
                        status["meets_requirement"] = version >= min_version
                
                package_status[package_name] = status
                
                if status["meets_requirement"]:
                    print(f"✓ {package_name}: {version}")
                else:
                    print(f"⚠ {package_name}: {version} (requires >= {min_version})")
                    
            except ImportError:
                package_status[package_name] = {
                    "installed": False,
                    "critical": is_critical
                }
                
                if is_critical:
                    missing_critical.append(install_name)
                    print(f"✗ {package_name}: Not installed (CRITICAL)")
                else:
                    missing_optional.append(install_name)
                    print(f"⚠ {package_name}: Not installed (optional)")
        
        # Add recommendations
        if missing_critical:
            self.results["recommendations"].append(
                f"Install critical packages: pip install {' '.join(missing_critical)}"
            )
        
        if missing_optional:
            self.results["recommendations"].append(
                f"Consider installing optional packages: pip install {' '.join(missing_optional)}"
            )
        
        self.results["packages"] = package_status
        return package_status
    
    def check_memory_optimization(self) -> Dict:
        """Check and recommend memory optimizations for GTX 1660 Super"""
        print("\n💾 Checking Memory Optimizations...")
        
        optimizations = {
            "mixed_precision_available": False,
            "gradient_checkpointing_available": False,
            "cuda_amp_available": False,
            "recommended_batch_size": 2,
            "recommended_image_size": 384,
            "memory_efficient_attention": False
        }
        
        try:
            import torch
            
            # Check mixed precision support
            optimizations["cuda_amp_available"] = hasattr(torch.cuda.amp, 'autocast')
            optimizations["mixed_precision_available"] = optimizations["cuda_amp_available"]
            
            # Check if GPU supports mixed precision well
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                compute_capability = torch.cuda.get_device_capability(0)
                # GTX 1660 Super has compute capability 7.5
                if compute_capability[0] >= 7:
                    print("✓ GPU supports efficient mixed precision training")
                else:
                    print("⚠ GPU has limited mixed precision support")
                    optimizations["mixed_precision_available"] = False
            
            # Check gradient checkpointing
            optimizations["gradient_checkpointing_available"] = hasattr(torch.utils.checkpoint, 'checkpoint')
            
            if optimizations["mixed_precision_available"]:
                print("✓ Mixed precision training available")
            if optimizations["gradient_checkpointing_available"]:
                print("✓ Gradient checkpointing available")
                
            print(f"\n📊 Recommended settings for GTX 1660 Super (6GB):")
            print(f"  - Batch size: {optimizations['recommended_batch_size']}")
            print(f"  - Image size: {optimizations['recommended_image_size']}x{optimizations['recommended_image_size']}")
            print(f"  - Mixed precision: Enabled")
            print(f"  - Gradient checkpointing: Enabled")
            print(f"  - Gradient accumulation steps: 8")
            
        except ImportError:
            print("✗ Cannot check optimizations - PyTorch not installed")
        
        self.results["optimizations"] = optimizations
        return optimizations
    
    def save_report(self, filename: str = "dependency_report.json"):
        """Save dependency check report"""
        report_path = self.project_root / filename
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Report saved to: {report_path}")
        
        # Also save a human-readable summary
        summary_path = self.project_root / "dependency_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("DEPENDENCY CHECK SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n\n")
            
            f.write("SYSTEM:\n")
            f.write(f"  OS: {self.results['system']['os']} {self.results['system'].get('os_version', '')}\n")
            f.write(f"  Python: {self.results['python']['version']}\n\n")
            
            f.write("GPU:\n")
            if self.results['cuda']['cuda_available']:
                for gpu in self.results['cuda']['gpu_devices']:
                    f.write(f"  {gpu['name']} ({gpu['memory_total_gb']:.1f} GB)\n")
            else:
                f.write("  No CUDA GPU available\n")
            
            f.write("\nRECOMMENDATIONS:\n")
            for rec in self.results['recommendations']:
                f.write(f"  - {rec}\n")
        
        print(f"📄 Summary saved to: {summary_path}")
    
    def run_full_check(self) -> bool:
        """Run all dependency checks"""
        print("🚀 Starting Dependency Check for Face Parsing Fine-tuning")
        print("=" * 60)
        
        # Run all checks
        self.check_system()
        python_ok = self.check_python_version()
        self.check_cuda()
        packages = self.check_packages()
        self.check_memory_optimization()
        
        # Determine overall status
        critical_packages_ok = all(
            pkg["installed"] and pkg.get("meets_requirement", True)
            for pkg in packages.values()
            if pkg.get("critical", True)
        )
        
        cuda_ok = self.results["cuda"]["cuda_available"]
        
        all_ok = python_ok and critical_packages_ok and cuda_ok
        
        print("\n" + "=" * 60)
        if all_ok:
            print("✅ All critical dependencies satisfied!")
            print("🎉 Ready to proceed with face parsing fine-tuning!")
        else:
            print("❌ Some critical dependencies are missing!")
            print("Please address the recommendations above before proceeding.")
        
        # Save report
        self.save_report()
        
        return all_ok


def main():
    """Run dependency check"""
    checker = DependencyChecker()
    success = checker.run_full_check()
    
    if not success:
        print("\n⚠️  Please install missing dependencies before continuing!")
        sys.exit(1)
    else:
        print("\n✅ You can now proceed to the next step!")


if __name__ == "__main__":
    main()