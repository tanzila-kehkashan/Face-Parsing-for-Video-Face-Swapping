import os
import cv2
import torch
from upscaler.codeformer import CodeFormerEnhancer

def codeformer_runner(img, model):
    """Run CodeFormer enhancement on the input image"""
    img = model.enhance(img)
    return img

# Only CodeFormer is supported now
DEFAULT_ENHANCER = {
    "CodeFormer": ("./assets/pretrained_models/codeformer.onnx", codeformer_runner)
}

def get_available_enhancer_names():
    """Returns only CodeFormer as available enhancer"""
    available = []
    name, data = list(DEFAULT_ENHANCER.items())[0]
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), data[0])
    if os.path.exists(path):
        available.append(name)
    return available

def load_face_enhancer_model(name='CodeFormer', device="cpu"):
    """Load CodeFormer as the default and only face enhancer"""
    if name != 'CodeFormer':
        name = 'CodeFormer'  # Force CodeFormer usage
        
    model_path, model_runner = DEFAULT_ENHANCER["CodeFormer"]
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_path)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CodeFormer model not found at {model_path}")
    
    model = CodeFormerEnhancer(model_path=model_path, device=device)
    return (model, model_runner)

def enhance_face(img, device="cpu"):
    """Direct function to enhance face using CodeFormer"""
    model, model_runner = load_face_enhancer_model(device=device)
    enhanced_img = model_runner(img, model)
    return enhanced_img