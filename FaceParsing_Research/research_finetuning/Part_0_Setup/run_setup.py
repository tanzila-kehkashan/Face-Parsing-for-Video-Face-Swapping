"""
Main setup runner for Part 0
Orchestrates dependency checking, installation, and GPU verification
"""

import sys
import os
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from research_finetuning.Part_0_Setup.check_dependencies import DependencyChecker
from research_finetuning.Part_0_Setup.install_requirements import RequirementsInstaller
from research_finetuning.Part_0_Setup.verify_gpu import GPUVerifier


class SetupRunner:
    """Orchestrate the complete setup process"""
    
    def __init__(self, skip_install=False, force_install=False):
        self.skip_install = skip_install
        self.force_install = force_install
        self.project_root = Path(__file__).parent.parent.parent
        self.setup_log = []
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "steps": {}
        }
    
    def log(self, message, level="INFO"):
        """Log setup progress"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {level}: {message}"
        print(entry)
        self.setup_log.append(entry)
    
    def run_dependency_check(self) -> bool:
        """Step 1: Check dependencies"""
        self.log("\n" + "="*60)
        self.log("STEP 1: Checking Dependencies")
        self.log("="*60)
        
        checker = DependencyChecker(self.project_root)
        success = checker.run_full_check()
        
        self.results["steps"]["dependency_check"] = {
            "success": success,
            "report_file": "dependency_report.json"
        }
        
        return success
    
    def run_installation(self) -> bool:
        """Step 2: Install missing packages"""
        if self.skip_install:
            self.log("\nSkipping installation (--skip-install flag)")
            return True
        
        self.log("\n" + "="*60)
        self.log("STEP 2: Installing Requirements")
        self.log("="*60)
        
        # Check if we need to install
        if not self.force_install:
            # Read dependency report
            report_path = self.project_root / "dependency_report.json"
            if report_path.exists():
                with open(report_path, 'r') as f:
                    report = json.load(f)
                
                # Check if all critical packages are installed
                packages = report.get("packages", {})
                missing_critical = [
                    name for name, info in packages.items()
                    if info.get("critical", True) and not info.get("installed", False)
                ]
                
                if not missing_critical and report.get("cuda", {}).get("cuda_available", False):
                    self.log("All critical dependencies already satisfied!")
                    response = input("\nRun installation anyway? (y/n): ")
                    if response.lower() != 'y':
                        return True
        
        installer = RequirementsInstaller()
        success = installer.run_full_installation()
        
        self.results["steps"]["installation"] = {
            "success": success,
            "skipped": False
        }
        
        return success
    
    def run_gpu_verification(self) -> bool:
        """Step 3: Verify and optimize GPU"""
        self.log("\n" + "="*60)
        self.log("STEP 3: GPU Verification and Optimization")
        self.log("="*60)
        
        verifier = GPUVerifier()
        success = verifier.run_full_verification()
        
        self.results["steps"]["gpu_verification"] = {
            "success": success,
            "config_dir": "configs/generated"
        }
        
        return success
    
    def create_environment_check_script(self):
        """Create a quick environment check script"""
        script_content = '''#!/usr/bin/env python3
"""Quick environment check script"""
import torch
import sys

print("🔍 Quick Environment Check")
print("="*40)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("="*40)

# Quick imports test
try:
    import cv2
    import albumentations
    import segmentation_models_pytorch
    print("✓ All key imports successful!")
except ImportError as e:
    print(f"✗ Import error: {e}")
'''
        
        script_path = self.project_root / "quick_check.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        self.log(f"Created quick check script: {script_path}")
    
    def save_setup_results(self):
        """Save setup results"""
        results_path = self.project_root / "setup_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save setup log
        log_path = self.project_root / "part0_setup_log.txt"
        with open(log_path, 'w') as f:
            f.write("\n".join(self.setup_log))
        
        self.log(f"\nResults saved to: {results_path}")
        self.log(f"Log saved to: {log_path}")
    
    def print_next_steps(self):
        """Print next steps for the user"""
        print("\n" + "="*60)
        print("🎉 SETUP COMPLETE!")
        print("="*60)
        
        print("\n📋 Summary:")
        for step_name, step_info in self.results["steps"].items():
            status = "✓" if step_info.get("success", False) else "✗"
            print(f"  {status} {step_name.replace('_', ' ').title()}")
        
        print("\n📝 Next Steps:")
        print("1. Review the generated configuration:")
        print("   - configs/generated/optimized_config.json")
        print("\n2. Test your environment:")
        print("   python quick_check.py")
        print("\n3. Start with Part 1 (Data Pipeline):")
        print("   - Implement data loading for CelebAMask-HQ")
        print("   - Create augmentation pipeline")
        print("\n4. Key settings for your GTX 1660 Super:")
        print("   - Batch size: 2")
        print("   - Image size: 384x384")
        print("   - Mixed precision: Enabled")
        print("   - Gradient accumulation: 8 steps")
        
        print("\n💡 Tips:")
        print("- Close other applications to free GPU memory")
        print("- Monitor GPU temperature during training")
        print("- Use the generated configs for optimal performance")
    
    def run(self) -> bool:
        """Run complete setup process"""
        print("\n" + "🚀 "*20)
        print("FACE PARSING RESEARCH - PART 0: SETUP & DEPENDENCIES")
        print("🚀 "*20)
        
        try:
            # Step 1: Check dependencies
            dep_success = self.run_dependency_check()
            
            # Step 2: Install if needed
            if dep_success or self.force_install:
                install_success = self.run_installation()
            else:
                install_success = False
                self.log("Skipping installation due to dependency check failure", "WARNING")
            
            # Step 3: Verify GPU
            gpu_success = self.run_gpu_verification()
            
            # Create helper scripts
            self.create_environment_check_script()
            
            # Save results
            self.save_setup_results()
            
            # Overall success
            overall_success = dep_success and install_success and gpu_success
            
            if overall_success:
                self.print_next_steps()
            else:
                print("\n⚠️  Setup completed with some issues.")
                print("Please review the logs and fix any problems before proceeding.")
            
            return overall_success
            
        except Exception as e:
            self.log(f"Setup failed with error: {str(e)}", "ERROR")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run Part 0 Setup")
    parser.add_argument("--skip-install", action="store_true",
                       help="Skip package installation")
    parser.add_argument("--force-install", action="store_true",
                       help="Force package installation even if already installed")
    parser.add_argument("--check-only", action="store_true",
                       help="Only run dependency check")
    
    args = parser.parse_args()
    
    if args.check_only:
        # Just run dependency check
        checker = DependencyChecker()
        checker.run_full_check()
    else:
        # Run full setup
        runner = SetupRunner(
            skip_install=args.skip_install,
            force_install=args.force_install
        )
        success = runner.run()
        
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()