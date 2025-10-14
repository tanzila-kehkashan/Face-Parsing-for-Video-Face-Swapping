import os
from pathlib import Path

# Base directory for the API
BASE_DIR = Path(__file__).resolve().parent

# Directory to FAKESYNCSTUDIO project - UPDATE THIS PATH TO YOUR ACTUAL INSTALLATION
SWAPFACE_DIR = Path("C:\FAKESYNCSTUDIO")  # Update this to match your installation path

# Verify FAKESYNCSTUDIO directory exists
if not SWAPFACE_DIR.exists():
    print(f"WARNING: FAKESYNCSTUDIO directory not found at {SWAPFACE_DIR}")
    print("Please update SWAPFACE_DIR in settings.py to the correct path")
elif not (SWAPFACE_DIR / "app.py").exists():
    print(f"WARNING: app.py not found in {SWAPFACE_DIR}")
    print("Please ensure SWAPFACE_DIR points to the correct FAKESYNCSTUDIO installation")
else:
    print(f"✅ FAKESYNCSTUDIO found at {SWAPFACE_DIR}")

# Directories for file storage
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
(UPLOAD_DIR / "sources").mkdir(exist_ok=True)
(UPLOAD_DIR / "targets").mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# API settings
API_HOST = "0.0.0.0"
API_PORT = 9876

# NSFW Detection Settings
NSFW_DETECTION_ENABLED = True
NSFW_MODEL_PATH = SWAPFACE_DIR / "assets" / "pretrained_models" / "open-nsfw.onnx"
NSFW_THRESHOLD = 0.7  # Adjust this value (0.0-1.0, higher = more strict)
NSFW_VIDEO_SAMPLE_FRAMES = 5  # Number of frames to check in videos

# Optimal SwapFace2Pon settings (aligned with simplified version)
# These are automatically applied - no user configuration needed
# Default settings optimized for Normal mode (fast processing)
OPTIMAL_SWAP_SETTINGS = {
    # Face enhancement (disabled by default - Normal mode)
    'face_enhancer_default': 'NONE',
    
    # Face parsing (disabled by default - Normal mode)
    'enable_face_parser': False,
    'mask_includes': [
        "Skin", "R-Eyebrow", "L-Eyebrow", "L-Eye", "R-Eye", 
        "Nose", "Mouth", "L-Lip", "U-Lip"
    ],
    'mask_soft_kernel': 17,
    'mask_soft_iterations': 10,
    'blur_amount': 0.1,
    'erode_amount': 0.15,
    
    # Quality settings (basic defaults)
    'face_scale': 1.0,
    'enable_laplacian_blend': False,  # Disabled for Normal mode
    'crop_top': 0,
    'crop_bott': 511,
    'crop_left': 0,
    'crop_right': 511,
    
    # Default swap behavior
    'swap_option': 'All Face',
    'age': 25,
    'distance_slider': 0.6,
    
    # Detection settings (optimal)
    'detect_condition': 'best detection',
    'detection_size': 640,
    'detection_threshold': 0.6,
    
    # Output settings
    'keep_output_sequence': False,
    
    # NSFW detection settings
    'nsfw_detection_enabled': NSFW_DETECTION_ENABLED,
    'nsfw_threshold': NSFW_THRESHOLD,
}

def get_api_settings(custom_settings=None):
    """
    Get API settings with Normal mode defaults and NSFW protection
    
    Args:
        custom_settings: Optional dict of custom settings for Best mode
    
    Returns:
        dict: Complete settings for processing
    """
    # Start with Normal mode defaults including NSFW settings
    settings = OPTIMAL_SWAP_SETTINGS.copy()
    
    if custom_settings:
        print(f"DEBUG: Applying custom settings: {custom_settings}")
        
        # Simply override defaults with custom settings
        for key, value in custom_settings.items():
            if key in ['face_enhancer_name']:
                # Map face_enhancer_name to face_enhancer_default
                settings['face_enhancer_default'] = value
                print(f"DEBUG: Set face_enhancer_default to {value}")
            else:
                # Direct override for other settings
                settings[key] = value
                print(f"DEBUG: Set {key} to {value}")
        
        print(f"DEBUG: Final settings: {settings}")
    else:
        print("DEBUG: Using default Normal mode settings with NSFW protection")
        print(f"DEBUG: Default settings: {settings}")
    
    return settings

# Model paths (for verification)
MODEL_PATHS = {
    'codeformer': SWAPFACE_DIR / "assets" / "pretrained_models" / "codeformer.onnx",
    'face_parsing': SWAPFACE_DIR / "assets" / "pretrained_models" / "79999_iter.pth", 
    'face_swapping': SWAPFACE_DIR / "assets" / "pretrained_models" / "inswapper_128.onnx",
    'buffalo_l': SWAPFACE_DIR / "assets" / "pretrained_models" / "models" / "buffalo_l",
    'nsfw_detector': NSFW_MODEL_PATH,  # Add NSFW model to verification
}

def verify_models():
    """Verify all required models are present"""
    missing_models = []
    
    for model_name, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            missing_models.append(f"{model_name}: {model_path}")
    
    if missing_models:
        print("❌ Missing required models:")
        for model in missing_models:
            print(f"   - {model}")
        return False
    else:
        print("✅ All required models found (including NSFW detector)")
        return True

def verify_nsfw_model():
    """Specifically verify NSFW model"""
    if NSFW_DETECTION_ENABLED:
        if NSFW_MODEL_PATH.exists():
            print(f"✅ NSFW detector model found at {NSFW_MODEL_PATH}")
            return True
        else:
            print(f"❌ NSFW detector model not found at {NSFW_MODEL_PATH}")
            print("⚠️ NSFW detection will be disabled")
            return False
    else:
        print("ℹ️ NSFW detection is disabled in settings")
        return True

# Verify models on import
if SWAPFACE_DIR.exists():
    verify_models()
    verify_nsfw_model()