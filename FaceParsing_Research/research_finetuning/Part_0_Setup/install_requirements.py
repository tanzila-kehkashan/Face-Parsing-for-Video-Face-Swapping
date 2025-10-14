"""
Automated requirements installer with GPU optimization
Handles PyTorch installation for GTX 1660 Super
"""

import subprocess
import sys
import platform
from pathlib import Path
from typing import List, Tuple, Optional
import os


class RequirementsInstaller:
    """Install and manage project requirements"""
    
    def __init__(self):
        self.platform = platform.system()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.cuda_version = "11.8"  # Optimal for GTX 1660 Super
        self.pip_args = ["--no-cache-dir", "--upgrade"]
        
    def run_command(self, cmd: List[str]) -> Tuple[bool, str]:
        """Run a command and return success status and output"""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                check=True
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, f"{e.stderr}\n{e.stdout}"
    
    def check_pip(self) -> bool:
        """Ensure pip is up to date"""
        print("📦 Updating pip...")
        success, output = self.run_command([
            sys.executable, "-m", "pip", "install", 
            "--upgrade", "pip", "setuptools", "wheel"
        ])
        
        if success:
            print("✓ pip updated successfully")
        else:
            print("✗ Failed to update pip")
            print(output)
        
        return success
    
    def install_pytorch(self) -> bool:
        """Install PyTorch with CUDA support for GTX 1660 Super"""
        print(f"\n🔥 Installing PyTorch with CUDA {self.cuda_version}...")
        
        # Construct PyTorch installation command
        if self.platform == "Windows":
            torch_cmd = [
                sys.executable, "-m", "pip", "install",
                "torch==2.0.1+cu118",
                "torchvision==0.15.2+cu118",
                "torchaudio==2.0.2+cu118",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ]
        else:
            # Linux/Mac
            torch_cmd = [
                sys.executable, "-m", "pip", "install",
                "torch==2.0.1",
                "torchvision==0.15.2",
                "torchaudio==2.0.2",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ]
        
        print(f"Running: {' '.join(torch_cmd)}")
        success, output = self.run_command(torch_cmd)
        
        if success:
            print("✓ PyTorch installed successfully")
            
            # Verify installation
            try:
                import torch
                print(f"  PyTorch version: {torch.__version__}")
                print(f"  CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"  GPU: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                print(f"⚠ Warning: Could not verify PyTorch installation: {e}")
        else:
            print("✗ Failed to install PyTorch")
            print("Error output:", output)
        
        return success
    
    def install_core_packages(self) -> bool:
        """Install core required packages"""
        print("\n📚 Installing core packages...")
        
        core_packages = [
            "numpy==1.24.3",
            "opencv-python-headless==4.8.0.74",
            "matplotlib==3.7.1",
            "tqdm>=4.65.0",
            "pyyaml==6.0",
            "Pillow>=9.5.0",  # Will try Pillow-SIMD separately
        ]
        
        success, output = self.run_command([
            sys.executable, "-m", "pip", "install", *self.pip_args, *core_packages
        ])
        
        if success:
            print("✓ Core packages installed")
        else:
            print("✗ Failed to install core packages")
            print(output)
        
        return success
    
    def install_ml_packages(self) -> bool:
        """Install ML-specific packages"""
        print("\n🤖 Installing ML packages...")
        
        ml_packages = [
            "albumentations==1.3.1",
            "segmentation-models-pytorch==0.3.3",
            "einops==0.6.1",
            "timm==0.9.2",
            "tensorboardX==2.6",  # Lighter than tensorboard
        ]
        
        # Install packages one by one for better error handling
        failed_packages = []
        
        for package in ml_packages:
            print(f"  Installing {package}...")
            success, output = self.run_command([
                sys.executable, "-m", "pip", "install", *self.pip_args, package
            ])
            
            if success:
                print(f"  ✓ {package}")
            else:
                print(f"  ✗ {package} failed")
                failed_packages.append(package)
        
        if failed_packages:
            print(f"\n⚠ Failed packages: {', '.join(failed_packages)}")
            print("These packages are not critical but recommended.")
            return False
        
        return True
    
    def install_optimized_packages(self) -> bool:
        """Install performance-optimized packages"""
        print("\n⚡ Installing optimized packages...")
        
        # Try Pillow-SIMD for faster image processing
        print("  Attempting Pillow-SIMD installation...")
        simd_success, _ = self.run_command([
            sys.executable, "-m", "pip", "uninstall", "-y", "pillow"
        ])
        
        if self.platform == "Windows":
            # Pillow-SIMD might not have Windows wheels
            print("  ℹ Pillow-SIMD may not be available for Windows, using standard Pillow")
        else:
            simd_success, _ = self.run_command([
                sys.executable, "-m", "pip", "install", "pillow-simd"
            ])
            
            if simd_success:
                print("  ✓ Pillow-SIMD installed (faster image processing)")
            else:
                print("  ⚠ Pillow-SIMD failed, using standard Pillow")
        
        return True
    
    def install_development_packages(self) -> bool:
        """Install development and debugging packages"""
        print("\n🛠 Installing development packages (optional)...")
        
        dev_packages = [
            "jupyter",
            "ipywidgets",
            "black",
            "pytest",
            "pytest-cov",
        ]
        
        # These are optional, so we don't fail if they don't install
        for package in dev_packages:
            success, _ = self.run_command([
                sys.executable, "-m", "pip", "install", package
            ])
            if success:
                print(f"  ✓ {package}")
            else:
                print(f"  ⚠ {package} (optional)")
        
        return True
    
    def create_pip_config(self):
        """Create pip configuration for faster downloads"""
        print("\n⚙️ Optimizing pip configuration...")
        
        pip_config = """[global]
timeout = 60
index-url = https://pypi.org/simple
trusted-host = pypi.org pypi.python.org files.pythonhosted.org
"""
        
        if self.platform == "Windows":
            pip_dir = Path.home() / "pip"
            pip_config_path = pip_dir / "pip.ini"
        else:
            pip_dir = Path.home() / ".config" / "pip"
            pip_config_path = pip_dir / "pip.conf"
        
        try:
            pip_dir.mkdir(parents=True, exist_ok=True)
            pip_config_path.write_text(pip_config)
            print("✓ pip configuration optimized")
        except Exception as e:
            print(f"⚠ Could not create pip config: {e}")
    
    def verify_installation(self) -> bool:
        """Verify all critical packages are installed correctly"""
        print("\n🔍 Verifying installation...")
        
        critical_imports = [
            "torch",
            "torchvision",
            "numpy",
            "cv2",
            "albumentations",
            "segmentation_models_pytorch",
        ]
        
        all_ok = True
        for package in critical_imports:
            try:
                __import__(package)
                print(f"  ✓ {package}")
            except ImportError:
                print(f"  ✗ {package}")
                all_ok = False
        
        # Special check for CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"\n✓ CUDA is available")
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
                print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                print("\n⚠ CUDA is not available - training will be slow on CPU")
        except:
            pass
        
        return all_ok
    
    def run_full_installation(self) -> bool:
        """Run complete installation process"""
        print("🚀 Starting Automated Package Installation")
        print("=" * 60)
        print(f"Platform: {self.platform}")
        print(f"Python: {self.python_version}")
        print(f"Target CUDA: {self.cuda_version}")
        print("=" * 60)
        
        # Create pip config
        self.create_pip_config()
        
        # Update pip
        if not self.check_pip():
            return False
        
        # Install PyTorch first (most critical)
        if not self.install_pytorch():
            print("\n❌ PyTorch installation failed!")
            print("Please install PyTorch manually from https://pytorch.org/")
            return False
        
        # Install other packages
        steps = [
            ("Core packages", self.install_core_packages),
            ("ML packages", self.install_ml_packages),
            ("Optimized packages", self.install_optimized_packages),
            ("Development packages", self.install_development_packages),
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*40}")
            if not step_func():
                print(f"⚠ {step_name} had some issues, but continuing...")
        
        # Verify installation
        print("\n" + "="*60)
        if self.verify_installation():
            print("\n✅ Installation completed successfully!")
            print("🎉 All critical packages are installed and verified!")
            return True
        else:
            print("\n⚠ Some packages failed to install properly")
            print("Please check the errors above and install manually if needed")
            return False


def main():
    """Run installation"""
    installer = RequirementsInstaller()
    
    # Check if we're in the right environment
    if "CONDA_DEFAULT_ENV" in os.environ:
        env_name = os.environ["CONDA_DEFAULT_ENV"]
        if env_name != "faceparse_research":
            print(f"⚠ Warning: Currently in '{env_name}' environment")
            print("Recommended: conda activate faceparse_research")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                return
    
    success = installer.run_full_installation()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()