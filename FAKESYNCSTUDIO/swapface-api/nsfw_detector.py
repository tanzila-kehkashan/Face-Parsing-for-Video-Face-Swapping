import cv2
import numpy as np
import onnxruntime
from pathlib import Path
import logging

logger = logging.getLogger("nsfw_detector")

class NSFWDetector:
    def __init__(self, model_path="./assets/pretrained_models/open-nsfw.onnx", threshold=0.7, providers=None):
        """
        Initialize NSFW detector
        
        Args:
            model_path: Path to open-nsfw.onnx model
            threshold: NSFW detection threshold (0.0-1.0, higher = more strict)
            providers: ONNX providers (CUDA/CPU)
        """
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.providers = providers or ["CPUExecutionProvider"]
        self.session = None
        
        # Check if model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"NSFW model not found at: {model_path}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the NSFW detection model"""
        try:
            self.session = onnxruntime.InferenceSession(
                str(self.model_path), 
                providers=self.providers
            )
            logger.info(f"NSFW detector loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load NSFW model: {e}")
            raise
    
    def _preprocess_image(self, image):
        """
        Preprocess image for NSFW detection
        Open NSFW model expects 224x224 RGB image
        """
        if isinstance(image, str):
            # If string path, load image
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image from path: {image}")
        
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 (model input size)
        image = cv2.resize(image, (224, 224))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to NCHW format
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = np.expand_dims(image, axis=0)   # Add batch dimension
        
        return image
    
    def detect_nsfw(self, image):
        """
        Detect if image contains NSFW content
        
        Args:
            image: Image path (str) or numpy array
            
        Returns:
            tuple: (is_nsfw: bool, confidence: float)
        """
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: processed_image})
            
            # Get NSFW probability (usually output[0][0][1] for NSFW class)
            nsfw_confidence = float(output[0][0][1]) if len(output[0][0]) > 1 else float(output[0][0][0])
            
            # Check if exceeds threshold
            is_nsfw = nsfw_confidence > self.threshold
            
            logger.info(f"NSFW detection: confidence={nsfw_confidence:.3f}, threshold={self.threshold}, is_nsfw={is_nsfw}")
            
            return is_nsfw, nsfw_confidence
            
        except Exception as e:
            logger.error(f"NSFW detection failed: {e}")
            # In case of error, be conservative and allow content
            return False, 0.0
    
    def check_video_frames(self, video_path, sample_frames=5):
        """
        Check video for NSFW content by sampling frames
        
        Args:
            video_path: Path to video file
            sample_frames: Number of frames to sample for checking
            
        Returns:
            tuple: (is_nsfw: bool, max_confidence: float, flagged_frames: list)
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError(f"Video has no frames: {video_path}")
            
            # Calculate frame indices to sample
            frame_indices = np.linspace(0, total_frames - 1, min(sample_frames, total_frames), dtype=int)
            
            max_confidence = 0.0
            flagged_frames = []
            is_nsfw = False
            
            for frame_idx in frame_indices:
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Check this frame
                frame_nsfw, frame_confidence = self.detect_nsfw(frame)
                
                if frame_nsfw:
                    is_nsfw = True
                    flagged_frames.append(frame_idx)
                
                max_confidence = max(max_confidence, frame_confidence)
            
            cap.release()
            
            logger.info(f"Video NSFW check: max_confidence={max_confidence:.3f}, flagged_frames={len(flagged_frames)}/{len(frame_indices)}")
            
            return is_nsfw, max_confidence, flagged_frames
            
        except Exception as e:
            logger.error(f"Video NSFW check failed: {e}")
            return False, 0.0, []
    
    def is_content_safe(self, file_path, file_type="auto"):
        """
        Check if content is safe (not NSFW)
        
        Args:
            file_path: Path to image or video file
            file_type: "image", "video", or "auto" to detect
            
        Returns:
            dict: {
                "is_safe": bool,
                "is_nsfw": bool, 
                "confidence": float,
                "details": str
            }
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {
                "is_safe": False,
                "is_nsfw": False,
                "confidence": 0.0,
                "details": f"File not found: {file_path}"
            }
        
        # Auto-detect file type
        if file_type == "auto":
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            
            ext = file_path.suffix.lower()
            if ext in video_extensions:
                file_type = "video"
            elif ext in image_extensions:
                file_type = "image"
            else:
                return {
                    "is_safe": False,
                    "is_nsfw": False,
                    "confidence": 0.0,
                    "details": f"Unsupported file type: {ext}"
                }
        
        try:
            if file_type == "image":
                is_nsfw, confidence = self.detect_nsfw(str(file_path))
                details = f"Image NSFW confidence: {confidence:.3f}"
                
            elif file_type == "video":
                is_nsfw, confidence, flagged_frames = self.check_video_frames(str(file_path))
                details = f"Video NSFW confidence: {confidence:.3f}, flagged frames: {len(flagged_frames)}"
                
            else:
                return {
                    "is_safe": False,
                    "is_nsfw": False,
                    "confidence": 0.0,
                    "details": f"Invalid file type: {file_type}"
                }
            
            return {
                "is_safe": not is_nsfw,
                "is_nsfw": is_nsfw,
                "confidence": confidence,
                "details": details
            }
            
        except Exception as e:
            logger.error(f"Content safety check failed: {e}")
            return {
                "is_safe": False,
                "is_nsfw": False,
                "confidence": 0.0,
                "details": f"Safety check error: {str(e)}"
            }

# Global instance for easy access
_global_nsfw_detector = None

def get_nsfw_detector(model_path="./assets/pretrained_models/open-nsfw.onnx", threshold=0.7, providers=None):
    """Get global NSFW detector instance"""
    global _global_nsfw_detector
    
    if _global_nsfw_detector is None:
        _global_nsfw_detector = NSFWDetector(
            model_path=model_path,
            threshold=threshold,
            providers=providers
        )
    
    return _global_nsfw_detector

def check_content_safety(file_path, file_type="auto", threshold=0.7):
    """
    Convenience function to check if content is safe
    
    Args:
        file_path: Path to file
        file_type: "image", "video", or "auto"
        threshold: NSFW detection threshold
        
    Returns:
        dict: Safety check results
    """
    detector = get_nsfw_detector(threshold=threshold)
    return detector.is_content_safe(file_path, file_type)