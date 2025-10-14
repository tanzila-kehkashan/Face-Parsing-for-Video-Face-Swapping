#!/usr/bin/env python3
"""
FAKESYNCSTUDIO API Startup Script
Professional face swapping API with optimal settings
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        ('fastapi', 'FastAPI web framework'),
        ('uvicorn', 'ASGI server'),
        ('torch', 'PyTorch'),
        ('cv2', 'OpenCV (opencv-python)'),
        ('insightface', 'Face analysis'),
        ('onnx', 'ONNX runtime support'),
        ('numpy', 'Numerical computing'),
    ]
    
    missing = []
    for package, description in required_packages:
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"✅ {description}")
        except ImportError:
            missing.append((package, description))
            print(f"❌ {description} (missing: {package})")
    
    if missing:
        print(f"\n📦 Install missing packages:")
        print("pip install fastapi uvicorn torch torchvision opencv-python insightface onnx onnxruntime")
        print("Or use: pip install -r requirements_api.txt")
        return False
    
    return True

def check_swapface_project():
    """Check if FAKESYNCSTUDIO project is accessible"""
    from settings import SWAPFACE_DIR, verify_models
    
    if not SWAPFACE_DIR.exists():
        print(f"❌ FAKESYNCSTUDIO project not found at: {SWAPFACE_DIR}")
        print("Update SWAPFACE_DIR in settings.py to the correct path")
        return False
    
    print(f"✅ FAKESYNCSTUDIO project found at: {SWAPFACE_DIR}")
    
    # Check for required files
    required_files = [
        'app.py',
        'face_enhancer.py', 
        'face_swapper.py',
        'face_analyser.py',
        'utils.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not (SWAPFACE_DIR / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files in FAKESYNCSTUDIO: {missing_files}")
        return False
    
    print("✅ All required FAKESYNCSTUDIO files found")
    
    # Check models
    if verify_models():
        print("✅ All AI models found")
        return True
    else:
        print("❌ Some AI models are missing")
        print("Download missing models to continue")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = [
        Path("uploads"),
        Path("uploads/sources"),
        Path("uploads/targets"), 
        Path("outputs"),
        Path("logs")
    ]
    
    for dir_path in directories:
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Directory ready: {dir_path}")

def start_api_server(host="0.0.0.0", port=9876, reload=False):
    """Start the API server"""
    cmd = [
        sys.executable, "-m", "uvicorn",
        "main:app",
        "--host", host,
        "--port", str(port),
        "--log-level", "info"
    ]
    
    if reload:
        cmd.append("--reload")
    
    print(f"🚀 Starting FAKESYNCSTUDIO API server...")
    print(f"🌐 Server URL: http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔧 Health Check: http://{host}:{port}/health")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def main():
    """Main startup function"""
    print("🎭 FAKESYNCSTUDIO API v2.0.0")
    print("=" * 50)
    print("Professional AI Face Swapping API")
    print("✨ Features: CodeFormer + Face Parsing + Optimal Settings")
    print("=" * 50)
    
    # Run all checks
    if not check_python_version():
        return False
    
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        return False
    
    print("\n🤖 Checking FAKESYNCSTUDIO project...")
    if not check_swapface_project():
        return False
    
    print("\n📁 Setting up directories...")
    setup_directories()
    
    print("\n🎯 All checks passed! Ready to start API server.")
    
    # Get server configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "9876"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    print(f"\n⚙️ Configuration:")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Reload: {reload}")
    print(f"   CUDA: {'Available' if check_cuda() else 'Not available'}")
    
    # Start server
    start_api_server(host, port, reload)
    return True

def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"💥 Startup failed: {e}")
        sys.exit(1)