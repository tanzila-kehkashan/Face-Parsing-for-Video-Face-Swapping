import os
import sys
import time
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable

from settings import SWAPFACE_DIR, get_api_settings

logger = logging.getLogger("face_swap_processor")

def process_face_swap(
    source_path: Path,
    target_path: Path,
    output_dir: Path,
    settings: Optional[Dict] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None
) -> Tuple[bool, str]:
    """Process face swap using the simplified FAKESYNCSTUDIO pipeline with optimal defaults"""
    
    def log(message: str):
        logger.info(message)
        if log_callback:
            log_callback(message)
    
    def check_cancellation():
        """Enhanced cancellation checker with immediate return"""
        if is_cancelled and is_cancelled():
            log("🛑 Cancellation detected, stopping process immediately...")
            return True
        return False
    
    # Get optimal settings
    api_settings = get_api_settings(settings)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create specific output filename that API expects
    output_name = "api_result"
    output_video_path = output_dir / f"{output_name}.mp4"
    
    # Store original state
    original_cwd = os.getcwd()
    original_path = sys.path.copy()
    
    try:
        log("Initializing FAKESYNCSTUDIO with optimal settings...")
        
        # Early cancellation check
        if check_cancellation():
            return False, "Process cancelled during initialization"
        
        # Change to FAKESYNCSTUDIO directory and add to path
        os.chdir(str(SWAPFACE_DIR))
        sys.path.insert(0, str(SWAPFACE_DIR))
        
        # Import simplified modules
        log("Loading optimized FAKESYNCSTUDIO modules...")
        
        if check_cancellation():
            return False, "Process cancelled during module loading"
        
        import cv2
        import torch
        import insightface
        import onnxruntime
        import numpy as np
        from tqdm import tqdm
        import concurrent.futures
        
        from face_swapper import Inswapper, paste_to_whole
        from face_analyser import get_analysed_data
        from face_enhancer import load_face_enhancer_model
        from face_parsing import init_parsing_model, get_parsed_mask, mask_regions_to_list
        from utils import merge_img_sequence_from_ref, split_list_by_lengths, create_image_grid
        
        if check_cancellation():
            return False, "Process cancelled after module imports"
        
        # Setup device and providers
        USE_CUDA = torch.cuda.is_available()
        PROVIDER = ["CPUExecutionProvider"]
        
        if USE_CUDA:
            available_providers = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                log("✅ CUDA available - enabling GPU acceleration")
                PROVIDER = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                device = "cuda"
            else:
                log("⚠️ CUDA not available - using CPU")
                device = "cpu"
        else:
            log("ℹ️ Using CPU processing")
            device = "cpu"
        
        EMPTY_CACHE = lambda: torch.cuda.empty_cache() if device == "cuda" else None
        
        if check_cancellation():
            return False, "Process cancelled during device setup"
        
        # Initialize models with optimal settings
        log("Loading face analysis model (InsightFace Buffalo_L)...")
        face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", providers=PROVIDER)
        face_analyser.prepare(
            ctx_id=0, 
            det_size=(api_settings['detection_size'], api_settings['detection_size']), 
            det_thresh=api_settings['detection_threshold']
        )
        
        if check_cancellation():
            return False, "Process cancelled after face analyser loading"
        
        log("Loading face swapper model (Inswapper)...")
        batch_size = 32 if device == "cuda" else 1
        face_swapper = Inswapper(
            model_file="./assets/pretrained_models/inswapper_128.onnx",
            batch_size=batch_size,
            providers=PROVIDER
        )

        if check_cancellation():
            return False, "Process cancelled after face swapper loading"

        # Conditional model loading based on settings
        face_enhancer = None
        face_parser = None

        # Debug settings
        log(f"DEBUG: face_enhancer_default = {api_settings.get('face_enhancer_default', 'CodeFormer')}")
        log(f"DEBUG: enable_face_parser = {api_settings.get('enable_face_parser', True)}")

        if api_settings.get('face_enhancer_default', 'CodeFormer') != 'NONE':
            log("Loading CodeFormer enhancement model...")
            face_enhancer = load_face_enhancer_model(name='CodeFormer', device=device)
            if check_cancellation():
                return False, "Process cancelled after face enhancer loading"
        else:
            log("⏩ Skipping CodeFormer loading (Normal mode)")

        if api_settings.get('enable_face_parser', True):
            log("Loading BiSeNet face parsing model...")
            face_parser = init_parsing_model("./assets/pretrained_models/79999_iter.pth", device=device)
            if check_cancellation():
                return False, "Process cancelled after face parser loading"
        else:
            log("⏩ Skipping BiSeNet loading (Normal mode)")
        
        # Verify input files exist
        if not source_path.exists():
            return False, f"Source image not found: {source_path}"
        if not target_path.exists():
            return False, f"Target video not found: {target_path}"
        
        log(f"Source: {source_path.name} ({source_path.stat().st_size} bytes)")
        log(f"Target: {target_path.name} ({target_path.stat().st_size} bytes)")
        
        if check_cancellation():
            return False, "Process cancelled before video extraction"
        
        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp(prefix=f"faceswap_{int(time.time())}_")
        temp_path = Path(temp_dir)
        
        try:
            # Extract video frames
            log("Extracting video frames...")
            image_sequence = []
            cap = cv2.VideoCapture(str(target_path))
            
            if not cap.isOpened():
                return False, f"Could not open video file: {target_path}"
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            log(f"Video info: {total_frames} frames at {fps:.2f} FPS")
            
            curr_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check cancellation more frequently - every 10 frames
                if curr_idx % 10 == 0 and check_cancellation():
                    cap.release()
                    return False, "Process cancelled during frame extraction"
                
                frame_path = temp_path / f"frame_{curr_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                image_sequence.append(str(frame_path))
                curr_idx += 1
                
                if curr_idx % 100 == 0:
                    log(f"Extracted {curr_idx}/{total_frames} frames")
            
            cap.release()
            log(f"✅ Extracted {len(image_sequence)} frames")
            
            if len(image_sequence) == 0:
                return False, "No frames extracted from video"
            
            if check_cancellation():
                return False, "Process cancelled after frame extraction"
            
            # Analyze faces
            log("🧿 Analyzing faces...")
            source_data = str(source_path), api_settings.get('age', 25)
            
            try:
                analysed_targets, analysed_sources, whole_frame_list, num_faces_per_frame = get_analysed_data(
                    face_analyser,
                    image_sequence,
                    source_data,
                    swap_condition=api_settings.get('swap_option', 'All Face'),
                    detect_condition=api_settings.get('detect_condition', 'best detection'),
                    scale=api_settings.get('face_scale', 1.0)
                )
            except Exception as e:
                return False, f"Face analysis failed: {str(e)}"
            
            if check_cancellation():
                return False, "Process cancelled after face analysis"
            
            if len(analysed_targets) == 0:
                return False, "No faces found to swap in the target video"
            
            log(f"✅ Found {len(analysed_targets)} faces to process")
            
            if check_cancellation():
                return False, "Process cancelled before face generation"
            
            # Generate swapped faces
            log("🧶 Generating swapped faces...")
            preds = []
            matrs = []
            
            try:
                batch_count = 0
                for batch_pred, batch_matr in face_swapper.batch_forward(whole_frame_list, analysed_targets, analysed_sources):
                    preds.extend(batch_pred)
                    matrs.extend(batch_matr)
                    batch_count += 1
                    
                    # Check cancellation after every batch
                    if check_cancellation():
                        return False, "Process cancelled during face generation"
                    
                    log(f"Generated face batch {batch_count}")
                    EMPTY_CACHE()
            except Exception as e:
                return False, f"Face swapping failed: {str(e)}"
            
            log(f"✅ Generated {len(preds)} swapped faces")
            
            if check_cancellation():
                return False, "Process cancelled after face generation"
            
            # Face enhancement with CodeFormer (conditional based on settings)
            if face_enhancer is not None:
                log("🎲 Enhancing faces with CodeFormer...")
                try:
                    enhancer_model, enhancer_model_runner = face_enhancer
                    for idx, pred in tqdm(enumerate(preds), total=len(preds), desc="Enhancing faces"):
                        # Check cancellation every 10 faces
                        if idx % 10 == 0 and check_cancellation():
                            return False, "Process cancelled during face enhancement"
                        enhanced_pred = enhancer_model_runner(pred, enhancer_model)
                        preds[idx] = cv2.resize(enhanced_pred, (512, 512))
                    EMPTY_CACHE()
                    log("✅ Face enhancement completed")
                except Exception as e:
                    return False, f"Face enhancement failed: {str(e)}"
            else:
                log("⏩ Skipping face enhancement (Normal mode)")
            
            if check_cancellation():
                return False, "Process cancelled after face enhancement"
            
            # Face parsing for optimal masks (conditional based on settings)
            if face_parser is not None:
                log("🎨 Creating optimal face masks...")
                try:
                    includes = mask_regions_to_list(api_settings['mask_includes'])
                    masks = []
                    
                    batch_count = 0
                    for batch_mask in get_parsed_mask(
                        face_parser,
                        preds,
                        classes=includes,
                        device=device,
                        batch_size=batch_size,
                        softness=int(api_settings['mask_soft_iterations'])
                    ):
                        # Check cancellation after every batch
                        if check_cancellation():
                            return False, "Process cancelled during face parsing"
                        masks.append(batch_mask)
                        EMPTY_CACHE()
                        batch_count += 1
                        if batch_count % 3 == 0:  # Check every 3 batches
                            log(f"Parsed {batch_count} mask batches...")
                    
                    masks = np.concatenate(masks, axis=0) if len(masks) >= 1 else masks
                    log("✅ Face parsing completed")
                except Exception as e:
                    return False, f"Face parsing failed: {str(e)}"
            else:
                log("⏩ Skipping face parsing (Normal mode)")
                # Create dummy masks for normal processing
                masks = [None] * len(preds)
            
            if check_cancellation():
                return False, "Process cancelled after face parsing"
            
            # Split data for processing
            split_preds = split_list_by_lengths(preds, num_faces_per_frame)
            split_matrs = split_list_by_lengths(matrs, num_faces_per_frame)
            split_masks = split_list_by_lengths(masks, num_faces_per_frame)
            
            # Clear memory
            del preds, matrs, masks
            EMPTY_CACHE()
            
            if check_cancellation():
                return False, "Process cancelled before final composition"
            
            # Paste faces back with optimal blending
            log("🧿 Applying final composition with Laplacian blending...")
            crop_mask = (
                api_settings['crop_top'], 
                511 - api_settings['crop_bott'],
                api_settings['crop_left'], 
                511 - api_settings['crop_right']
            )
            
            def post_process_frame(frame_idx, frame_img, split_preds, split_matrs, split_masks):
                """Process a single frame with cancellation check"""
                try:
                    # Check cancellation at frame level
                    if is_cancelled and is_cancelled():
                        return False, "Cancelled during frame processing"
                    
                    whole_img = cv2.imread(frame_img)
                    if whole_img is None:
                        return False, f"Could not read frame {frame_img}"
                                        
                    blend_method = 'laplacian' if api_settings['enable_laplacian_blend'] else 'linear'

                    # Handle both parsed and unparsed mask scenarios
                    if face_parser is not None:
                        # Best mode with parsed masks
                        for p, m, mask in zip(split_preds[frame_idx], split_matrs[frame_idx], split_masks[frame_idx]):
                            # Check cancellation even during individual face processing
                            if is_cancelled and is_cancelled():
                                return False, "Cancelled during face composition"
                            
                            p = cv2.resize(p, (512, 512))
                            mask = cv2.resize(mask, (512, 512)) if mask is not None else None
                            m /= 0.25
                            whole_img = paste_to_whole(
                                p, whole_img, m,
                                mask=mask,
                                crop_mask=crop_mask,
                                blend_method=blend_method,
                                blur_amount=api_settings['blur_amount'],
                                erode_amount=api_settings['erode_amount']
                            )
                    else:
                        # Normal mode without parsed masks
                        for p, m in zip(split_preds[frame_idx], split_matrs[frame_idx]):
                            # Check cancellation even during individual face processing
                            if is_cancelled and is_cancelled():
                                return False, "Cancelled during face composition"
                            
                            p = cv2.resize(p, (512, 512))
                            m /= 0.25
                            whole_img = paste_to_whole(
                                p, whole_img, m,
                                mask=None,
                                crop_mask=crop_mask,
                                blend_method=blend_method,
                                blur_amount=api_settings['blur_amount'],
                                erode_amount=api_settings['erode_amount']
                            )
                    
                    cv2.imwrite(frame_img, whole_img)
                    return True, "Success"
                except Exception as e:
                    return False, str(e)
            
            # Process frames with enhanced cancellation checking
            try:
                # Use single-threaded processing to make cancellation more responsive
                completed = 0
                for idx, frame_img in enumerate(image_sequence):
                    # Check cancellation before each frame
                    if check_cancellation():
                        log("🛑 Cancellation detected during frame processing")
                        return False, "Process cancelled during frame processing"
                    
                    success, result = post_process_frame(idx, frame_img, split_preds, split_matrs, split_masks)
                    if not success:
                        if "Cancelled" in result:
                            return False, result
                        log(f"Warning: Frame processing error: {result}")
                    
                    completed += 1
                    
                    # Check cancellation and log progress every 5 frames
                    if completed % 5 == 0:
                        if check_cancellation():
                            log("🛑 Cancellation detected during frame completion check")
                            return False, "Process cancelled during frame processing"
                        log(f"Processed {completed}/{len(image_sequence)} frames")
                            
            except Exception as e:
                return False, f"Frame processing failed: {str(e)}"
            
            log("✅ Frame processing completed")
            
            if check_cancellation():
                return False, "Process cancelled before video creation"
            
            # Create output video
            log("⌛ Creating final video...")
            try:
                merge_img_sequence_from_ref(str(target_path), image_sequence, str(output_video_path))
            except Exception as e:
                return False, f"Video creation failed: {str(e)}"
            
            if check_cancellation():
                return False, "Process cancelled after video creation"
            
            # Verify output file
            if not output_video_path.exists():
                return False, "Output video file was not created"
            
            file_size = output_video_path.stat().st_size
            if file_size == 0:
                return False, "Output video file is empty"
            
            log(f"✅ Output video created: {output_video_path.name} ({file_size:,} bytes)")
            return True, str(output_video_path)
            
        finally:
            # Clean up temporary directory
            try:
                if temp_path.exists():
                    shutil.rmtree(temp_path)
                    log("🧹 Cleaned up temporary files")
            except Exception as e:
                log(f"Warning: Could not remove temp directory: {e}")
        
    except Exception as e:
        # Check if this was a cancellation-related exception
        if check_cancellation():
            return False, "Process cancelled due to user request"
        
        error_msg = f"Processing failed: {str(e)}"
        log(error_msg)
        logger.exception("Detailed error:")
        return False, error_msg
    
    finally:
        # Restore original state
        try:
            os.chdir(original_cwd)
            sys.path[:] = original_path
        except Exception as e:
            logger.warning(f"Could not restore original state: {e}")