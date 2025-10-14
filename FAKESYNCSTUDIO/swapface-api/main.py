import os
import asyncio
import json
import logging
from nsfw_detector import get_nsfw_detector, check_content_safety
from settings import NSFW_DETECTION_ENABLED, NSFW_THRESHOLD, NSFW_MODEL_PATH
from typing import Dict, List, Optional
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Form, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
import uvicorn
from pathlib import Path
import threading
import queue
import time
import aiofiles
import tempfile

from settings import API_HOST, API_PORT, UPLOAD_DIR, OUTPUT_DIR, get_api_settings, verify_models
from api_utils import generate_unique_id, create_job_directory, get_job_info, clean_up_job, list_jobs
from processor import process_face_swap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("swapface_api")

app = FastAPI(
    title="FAKESYNCSTUDIO API", 
    description="Professional AI Face Swapping API with optimal quality settings",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active jobs and their log queues
active_jobs: Dict[str, str] = {}
job_log_queues: Dict[str, queue.Queue] = {}
cancelled_jobs: Dict[str, bool] = {}
# Add a global dictionary to track processing threads
processing_threads: Dict[str, threading.Thread] = {}
# NSFW Detector global variable
NSFW_DETECTOR = None
def initialize_nsfw_detector():
    """Initialize NSFW detector if enabled"""
    global NSFW_DETECTOR
    
    if NSFW_DETECTION_ENABLED and NSFW_MODEL_PATH.exists():
        try:
            NSFW_DETECTOR = get_nsfw_detector(
                model_path=str(NSFW_MODEL_PATH),
                threshold=NSFW_THRESHOLD,
                providers=PROVIDER  # Use same providers as face swap models
            )
            logger.info("✅ NSFW detector initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize NSFW detector: {e}")
            return False
    else:
        logger.info("ℹ️ NSFW detection disabled or model not found")
        return False

def check_file_safety(file_path, file_type="auto"):
    """
    Check if uploaded file is safe (not NSFW)
    
    Args:
        file_path: Path to the file
        file_type: "image", "video", or "auto"
        
    Returns:
        tuple: (is_safe: bool, message: str)
    """
    if not NSFW_DETECTION_ENABLED or NSFW_DETECTOR is None:
        return True, "NSFW detection disabled"
    
    try:
        safety_result = NSFW_DETECTOR.is_content_safe(file_path, file_type)
        
        if not safety_result["is_safe"]:
            if safety_result["is_nsfw"]:
                return False, f"Content blocked: Inappropriate content detected (confidence: {safety_result['confidence']:.3f})"
            else:
                return False, f"Content blocked: {safety_result['details']}"
        
        return True, f"Content approved (NSFW confidence: {safety_result['confidence']:.3f})"
        
    except Exception as e:
        logger.error(f"NSFW check failed: {e}")
        # In case of error, allow content but log the issue
        return True, f"NSFW check failed: {e}"
# Enhanced cancellation function with more aggressive checking
def check_job_cancellation(job_id: str) -> bool:
    """Check if job should be cancelled and clean up if needed"""
    if cancelled_jobs.get(job_id, False):
        # Immediately mark as cancelled in active jobs
        if job_id in active_jobs:
            active_jobs[job_id] = "cancelled"
        
        # Add cancellation message to logs
        if job_id in job_log_queues:
            try:
                job_log_queues[job_id].put("🛑 Processing cancelled by user request")
                job_log_queues[job_id].put("🧹 Cleaning up resources...")
            except:
                pass
        
        logger.info(f"Job {job_id} cancellation detected")
        return True
    return False

# Optimized file saving function
async def save_uploaded_file_async(upload_file: UploadFile, file_path: Path, chunk_size: int = 8192) -> int:
    """Save uploaded file asynchronously with streaming for better performance"""
    file_path.parent.mkdir(exist_ok=True, parents=True)
    
    total_size = 0
    
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            # Read and write in chunks to avoid memory issues
            while chunk := await upload_file.read(chunk_size):
                await f.write(chunk)
                total_size += len(chunk)
        
        # Verify file was written correctly
        if not file_path.exists() or file_path.stat().st_size == 0:
            raise Exception(f"Failed to write file {file_path}")
        
        actual_size = file_path.stat().st_size
        if actual_size != total_size:
            logger.warning(f"Size mismatch: expected {total_size}, got {actual_size}")
            
        logger.info(f"Successfully saved file: {file_path.name} ({actual_size:,} bytes)")
        return actual_size
        
    except Exception as e:
        # Clean up partial file on error
        if file_path.exists():
            file_path.unlink()
        raise Exception(f"Error saving file {file_path}: {e}")

def background_process_wrapper(job_id: str, source_path: Path, target_path: Path, output_dir: Path, settings: Optional[Dict]):
    """Wrapper function to run face swap in background thread"""
    try:
        # Store the current thread for cancellation purposes
        processing_threads[job_id] = threading.current_thread()
        
        # Run the actual processing
        asyncio.run(generate_face_swap_output(job_id, source_path, target_path, output_dir, settings))
    finally:
        # Clean up thread reference
        if job_id in processing_threads:
            del processing_threads[job_id]

async def generate_face_swap_output(job_id: str, source_path: Path, target_path: Path, output_dir: Path, settings: Optional[Dict]):
    """Background task to run the optimized face swap process and update job status."""
    logger.info(f"Job {job_id}: Starting face swap with optimal settings")
    active_jobs[job_id] = "processing"
    
    # Create log queue for this job
    log_queue = queue.Queue()
    job_log_queues[job_id] = log_queue
    
    def log_callback(message: str):
        """Callback to add log messages to the queue"""
        try:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            log_message = f"{timestamp} - {message}"
            log_queue.put(log_message)
            logger.info(f"Job {job_id}: {message}")
        except Exception as e:
            logger.error(f"Error in log callback: {e}")
    
    try:
        log_callback("🎭 Starting AI face swap...")
        if settings and settings.get('face_enhancer_name') == 'NONE':
            log_callback("⚡ Normal Mode: Fast processing without enhancement")
        else:
            log_callback("✨ Best Mode: Using CodeFormer enhancement + Face parsing + Laplacian blending")
        
        # Create enhanced cancellation checker function
        def is_cancelled():
            """Enhanced cancellation checker with more frequent checks"""
            cancelled = check_job_cancellation(job_id)
            if cancelled:
                log_callback("🛑 Cancellation detected - stopping all processing")
            return cancelled
        
        # Check cancellation before starting processing
        if is_cancelled():
            log_callback("🛑 Job cancelled before processing started")
            return
        
        success, message = process_face_swap(source_path, target_path, output_dir, settings, log_callback, is_cancelled)
        
        # Final check if job was cancelled during processing
        if cancelled_jobs.get(job_id, False):
            active_jobs[job_id] = "cancelled"
            log_callback("🛑 Job cancelled during processing")
            logger.info(f"Job {job_id}: Cancelled by user")
            return
            
        if success:
            active_jobs[job_id] = "completed"
            log_callback(f"🎉 Face swap completed successfully!")
            log_callback(f"📁 Output file: {Path(message).name}")
            logger.info(f"Job {job_id}: Completed successfully. Output: {message}")
            
            # Verify the output file exists and is valid
            if os.path.exists(message):
                file_size = os.path.getsize(message)
                log_callback(f"✅ Output verified: {file_size:,} bytes")
                
                # Additional quality check
                if file_size < 100000:  # Less than 100KB is suspicious for a video
                    log_callback(f"⚠️ Warning: Output file is very small ({file_size} bytes)")
            else:
                log_callback(f"❌ Error: Output file not found at {message}")
                active_jobs[job_id] = "failed"
        else:
            active_jobs[job_id] = "failed"
            log_callback(f"❌ Face swap failed: {message}")
            logger.error(f"Job {job_id}: Face swap failed. Error: {message}")
    
    except Exception as e:
        # Check if this was due to cancellation
        if cancelled_jobs.get(job_id, False):
            active_jobs[job_id] = "cancelled"
            log_callback("🛑 Processing cancelled by user")
        else:
            active_jobs[job_id] = "failed"
            error_msg = f"Unexpected error during processing: {str(e)}"
            log_callback(f"💥 {error_msg}")
            logger.exception(f"Job {job_id}: Unexpected error: {e}")
    
    # Signal end of logs
    log_callback("--- Processing Complete ---")
    
    # Clean up source and target files after processing (keep outputs)
    try:
        if source_path.exists():
            source_path.unlink()
            log_callback("🧹 Cleaned up source file")
        if target_path.exists():
            target_path.unlink() 
            log_callback("🧹 Cleaned up target file")
        # Remove empty directories
        try:
            source_path.parent.rmdir()
            target_path.parent.rmdir()
        except OSError:
            pass  # Directory not empty or doesn't exist
    except Exception as e:
        logger.warning(f"Could not clean up upload files for job {job_id}: {e}")
    
    # Clean up cancellation flag and thread reference
    if job_id in cancelled_jobs:
        del cancelled_jobs[job_id]
    if job_id in processing_threads:
        del processing_threads[job_id]

def is_valid_image_file(filename: str, content_type: Optional[str]) -> bool:
    """Check if file is a valid image based on extension and content type"""
    if not filename:
        return False
    
    # Check file extension
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    ext = Path(filename).suffix.lower()
    
    if ext not in valid_extensions:
        return False
    
    # Check content type if provided (be lenient)
    if content_type:
        valid_types = [
            'image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/webp',
            'application/octet-stream'  # Some clients send this
        ]
        if not any(valid_type in content_type.lower() for valid_type in valid_types):
            return False
    
    return True

def is_valid_video_file(filename: str, content_type: Optional[str]) -> bool:
    """Check if file is a valid video based on extension and content type"""
    if not filename:
        return False
    
    # Check file extension
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    ext = Path(filename).suffix.lower()
    
    if ext not in valid_extensions:
        return False
    
    # Check content type if provided (be lenient)
    if content_type:
        valid_types = [
            'video/mp4', 'video/avi', 'video/mov', 'video/quicktime', 
            'video/x-msvideo', 'video/mkv', 'video/webm',
            'application/octet-stream'  # Some clients send this
        ]
        if not any(valid_type in content_type.lower() for valid_type in valid_types):
            return False
    
    return True

@app.post("/upload/")
async def upload_files(
    request: Request,
    background_tasks: BackgroundTasks,
    source_image: UploadFile = File(..., description="Source face image (JPG, PNG)"),
    target_video: UploadFile = File(..., description="Target video (MP4, AVI, MOV)"),
    settings: Optional[str] = Form(None, description="Optional settings JSON")
):
    """Upload files for face swapping with optimized performance"""
    job_id = generate_unique_id()
    upload_start_time = time.time()
    
    try:
        # Log request details for debugging
        logger.info(f"Job {job_id}: Upload request from {request.client.host if request.client else 'unknown'}")
        logger.info(f"Job {job_id}: Source file: {source_image.filename}, Content-Type: {source_image.content_type}")
        logger.info(f"Job {job_id}: Target file: {target_video.filename}, Content-Type: {target_video.content_type}")
        
        # Validate files exist
        if not source_image.filename:
            raise HTTPException(status_code=400, detail="Source image filename is required")
        if not target_video.filename:
            raise HTTPException(status_code=400, detail="Target video filename is required")
        
        # Validate file types (more lenient validation)
        if not is_valid_image_file(source_image.filename, source_image.content_type):
            logger.warning(f"Job {job_id}: Invalid source image - filename: {source_image.filename}, content-type: {source_image.content_type}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid source image file. Supported formats: JPG, PNG, BMP. Got: {source_image.filename} ({source_image.content_type})"
            )
        
        if not is_valid_video_file(target_video.filename, target_video.content_type):
            logger.warning(f"Job {job_id}: Invalid target video - filename: {target_video.filename}, content-type: {target_video.content_type}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid target video file. Supported formats: MP4, AVI, MOV, MKV. Got: {target_video.filename} ({target_video.content_type})"
            )
        
        # Create job directories
        job_dirs = create_job_directory(job_id)

        # Determine file extensions
        source_ext = Path(source_image.filename).suffix.lower() if source_image.filename else '.jpg'
        target_ext = Path(target_video.filename).suffix.lower() if target_video.filename else '.mp4'
        
        source_path = job_dirs["source"] / f"source_image{source_ext}"
        target_path = job_dirs["target"] / f"target_video{target_ext}"
        output_dir = job_dirs["output"]

        logger.info(f"Job {job_id}: Saving files to {source_path} and {target_path}")

        # Save files asynchronously with streaming
        try:
            logger.info(f"Job {job_id}: Saving source image...")
            source_size = await save_uploaded_file_async(source_image, source_path, chunk_size=8192)
            
            logger.info(f"Job {job_id}: Saving target video...")
            target_size = await save_uploaded_file_async(target_video, target_path, chunk_size=65536)  # Larger chunks for video
        except Exception as e:
            logger.error(f"Job {job_id}: File save error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded files: {str(e)}")
        
        upload_time = time.time() - upload_start_time
        total_size = source_size + target_size
        
        logger.info(f"Job {job_id}: Upload completed in {upload_time:.2f}s - Total size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
        
        # Validate file sizes
        if source_size < 1000:  # Less than 1KB
            raise HTTPException(status_code=400, detail="Source image file is too small or corrupted")
        if target_size < 100000:  # Less than 100KB
            raise HTTPException(status_code=400, detail="Target video file is too small or corrupted")

        # **NEW: NSFW CONTENT SAFETY CHECKS**
        if NSFW_DETECTION_ENABLED and NSFW_DETECTOR is not None:
            logger.info(f"Job {job_id}: Performing content safety checks...")
            
            # Check source image
            source_safe, source_message = check_file_safety(source_path, "image")
            if not source_safe:
                logger.warning(f"Job {job_id}: Source image failed safety check: {source_message}")
                clean_up_job(job_id)
                raise HTTPException(
                    status_code=400, 
                    detail=f"Source image rejected: {source_message}"
                )
            logger.info(f"Job {job_id}: Source image safety check: {source_message}")
            
            # Check target video
            target_safe, target_message = check_file_safety(target_path, "video")
            if not target_safe:
                logger.warning(f"Job {job_id}: Target video failed safety check: {target_message}")
                clean_up_job(job_id)
                raise HTTPException(
                    status_code=400, 
                    detail=f"Target video rejected: {target_message}"
                )
            logger.info(f"Job {job_id}: Target video safety check: {target_message}")
            
            logger.info(f"Job {job_id}: ✅ All content safety checks passed")


        # Parse settings if provided
        parsed_settings = None
        if settings:
            try:
                parsed_settings = json.loads(settings)
                logger.info(f"Job {job_id}: Raw custom settings received: {parsed_settings}")
                
                # Validate settings using our get_api_settings function
                validated_settings = get_api_settings(parsed_settings)
                logger.info(f"Job {job_id}: Validated settings: {validated_settings}")
                parsed_settings = validated_settings
                
            except json.JSONDecodeError as e:
                logger.error(f"Job {job_id}: Settings JSON decode error: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid JSON format for settings: {str(e)}")
        else:
            logger.info(f"Job {job_id}: No custom settings provided, using defaults")

        active_jobs[job_id] = "processing"
        logger.info(f"Job {job_id}: Files uploaded successfully. Starting background processing.")

        # Start background processing in a separate thread (not using FastAPI background tasks)
        thread = threading.Thread(
            target=background_process_wrapper,
            args=(job_id, source_path, target_path, output_dir, parsed_settings)
        )
        thread.daemon = True
        thread.start()

        response_data = {
            "job_id": job_id, 
            "status": "processing",
            "message": "Files uploaded successfully. Processing started with optimal AI settings.",
            "upload_info": {
                "source_size": source_size,
                "target_size": target_size,
                "total_size": total_size,
                "upload_time": round(upload_time, 2),
                "upload_speed_mbps": round((total_size / 1024 / 1024) / upload_time, 2) if upload_time > 0 else 0
            },
            "features": [
                "CodeFormer face enhancement",
                "BiSeNet face parsing", 
                "Laplacian blending",
                "Professional quality output"
            ]
        }
        
        # Add NSFW detection info if enabled
        if NSFW_DETECTION_ENABLED and NSFW_DETECTOR is not None:
            response_data["safety"] = {
                "nsfw_detection": "enabled",
                "content_approved": True,
                "threshold": NSFW_THRESHOLD
            }
        
        return JSONResponse(response_data, status_code=202)

    except HTTPException:
        # Clean up on HTTP errors
        logger.error(f"Job {job_id}: HTTP error during upload")
        clean_up_job(job_id)
        raise
    except Exception as e:
        logger.error(f"Job {job_id}: Unexpected error during upload: {e}")
        clean_up_job(job_id)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Add request size limit middleware
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """Middleware to handle large uploads with timeout and size limits"""
    
    # Set content length limit (500MB)
    max_size = 500 * 1024 * 1024  # 500MB
    
    if request.method == "POST" and request.url.path in ["/upload/"]:
        content_length = request.headers.get("content-length")
        
        if content_length:
            content_length = int(content_length)
            if content_length > max_size:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": f"File too large. Maximum size: {max_size // 1024 // 1024}MB",
                        "max_size_mb": max_size // 1024 // 1024,
                        "received_size_mb": content_length // 1024 // 1024
                    }
                )
    
    # Increase timeout for upload endpoints
    timeout = 3600.0 if request.url.path in ["/upload/"] else 1800.0
    
    try:
        return await asyncio.wait_for(call_next(request), timeout=timeout)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=408,
            content={"detail": f"Request timeout after {timeout/60:.1f} minutes"}
        )

@app.get("/jobs/{job_id}/status/")
async def get_job_status(job_id: str):
    """Get the status of a face swap job with detailed information."""
    try:
        job_info = get_job_info(job_id)
        
        # Override status with active_jobs if it's more current
        if job_id in active_jobs:
            job_info["status"] = active_jobs[job_id]
            
        # Add additional info for active jobs
        if job_id in active_jobs:
            job_info["active"] = True
            status = active_jobs[job_id]
            if status == "processing":
                job_info["message"] = "Processing with optimal AI settings (CodeFormer + Face Parsing + Laplacian Blending)"
            elif status == "completed":
                job_info["message"] = "Face swap completed successfully with professional quality"
            elif status == "failed":
                job_info["message"] = "Processing failed - check logs for details"
            elif status == "cancelled":
                job_info["message"] = "Processing cancelled by user request"
            else:
                job_info["message"] = f"Job is currently {status}"
        
        if job_info["status"] == "not_found":
            raise HTTPException(status_code=404, detail="Job not found.")
        
        logger.info(f"Status check for job {job_id}: {job_info['status']}")
        return JSONResponse(job_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking status for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking job status: {e}")

@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Download the result video for a completed job."""
    try:
        job_info = get_job_info(job_id)
        
        # Check if job is still processing
        if job_id in active_jobs and active_jobs[job_id] not in ["completed", "failed"]:
            raise HTTPException(status_code=400, detail=f"Job {job_id} is still {active_jobs[job_id]}. Please wait for completion.")

        if job_info["status"] != "completed":
            if job_info["status"] == "failed":
                raise HTTPException(status_code=400, detail="Job failed during processing. Check logs for details.")
            else:
                raise HTTPException(status_code=400, detail="Job not yet completed.")
        
        if not job_info.get("output_path"):
            raise HTTPException(status_code=404, detail="Output file path not found.")
        
        output_path = job_info["output_path"]
        if not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail="Output file not found on server.")
        
        # Verify file size
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            raise HTTPException(status_code=500, detail="Output file is empty.")
        
        logger.info(f"Serving result for job {job_id}: {Path(output_path).name} ({file_size:,} bytes)")
        
        return FileResponse(
            path=output_path, 
            media_type="video/mp4", 
            filename=f"swapped_video_{job_id}.mp4",
            headers={
                "Content-Length": str(file_size),
                "X-Processing-Info": "Enhanced with CodeFormer + Face Parsing"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving result for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving result: {e}")

@app.get("/jobs/{job_id}/stream_logs")
async def stream_job_logs(job_id: str):
    """Stream processing logs for a job in real-time."""
    if job_id not in job_log_queues and job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    async def generate_logs():
        # Wait for log queue to be created
        max_wait = 30  # seconds
        wait_count = 0
        while job_id not in job_log_queues and wait_count < max_wait:
            await asyncio.sleep(1)
            wait_count += 1
        
        if job_id not in job_log_queues:
            yield f"data: Job {job_id} log queue not available\n\n"
            return
        
        log_queue = job_log_queues[job_id]
        
        # Send initial status
        yield f"data: 🎭 FAKESYNCSTUDIO API - Professional Quality Processing\n\n"
        yield f"data: Job ID: {job_id}\n\n"
        
        while True:
            try:
                # Non-blocking get with timeout
                message = log_queue.get(timeout=1)
                yield f"data: {message}\n\n"
                
                # Check if job is completed
                if job_id in active_jobs and active_jobs[job_id] in ["completed", "failed", "cancelled"]:
                    # Continue draining the queue for a bit
                    try:
                        while True:
                            message = log_queue.get_nowait()
                            yield f"data: {message}\n\n"
                    except queue.Empty:
                        break
                    yield f"event: end\ndata: Job {active_jobs[job_id]}\n\n"
                    break
                    
            except queue.Empty:
                # Send keepalive
                yield f"data: \n\n"
                continue
            except Exception as e:
                yield f"data: ❌ Error in log stream: {e}\n\n"
                break
        
        # Clean up
        if job_id in job_log_queues:
            del job_log_queues[job_id]
    
    return StreamingResponse(generate_logs(), media_type="text/event-stream")

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and all its files."""
    try:
        # Remove from cancelled jobs if present
        if job_id in cancelled_jobs:
            del cancelled_jobs[job_id]
            
        # Remove from active jobs if present
        if job_id in active_jobs:
            del active_jobs[job_id]
        
        # Remove log queue if present
        if job_id in job_log_queues:
            del job_log_queues[job_id]
        
        # Remove thread reference if present
        if job_id in processing_threads:
            del processing_threads[job_id]
        
        # Clean up job files
        clean_up_job(job_id)
        
        logger.info(f"Job {job_id} deleted successfully")
        return {"message": f"Job {job_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting job: {e}")

@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running face swap job."""
    try:
        if job_id in active_jobs and active_jobs[job_id] == "processing":
            # Mark job as cancelled immediately
            cancelled_jobs[job_id] = True
            active_jobs[job_id] = "cancelling"
            
            # Add cancellation message to logs
            if job_id in job_log_queues:
                try:
                    job_log_queues[job_id].put("🛑 Cancellation requested by user...")
                    job_log_queues[job_id].put("🧹 Cleaning up resources...")
                except queue.Full:
                    pass  # Queue might be full, but that's okay
            
            logger.info(f"Job {job_id} cancellation requested by user")
            
            # Give the process some time to detect cancellation
            await asyncio.sleep(1)
            
            # Force status to cancelled
            active_jobs[job_id] = "cancelled"
            
            return {"message": f"Job {job_id} cancellation requested", "status": "cancelled"}
        elif job_id in active_jobs:
            return {"message": f"Job {job_id} cannot be cancelled (status: {active_jobs[job_id]})", "status": active_jobs[job_id]}
        else:
            raise HTTPException(status_code=404, detail="Job not found or already completed")
            
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error cancelling job: {e}")

@app.get("/jobs/")
async def get_jobs():
    """List all jobs with their current status."""
    try:
        jobs = list_jobs()
        
        # Add active jobs that might not have files yet
        for job_id, status in active_jobs.items():
            if not any(job["id"] == job_id for job in jobs):
                jobs.append({
                    "id": job_id, 
                    "status": status, 
                    "output_path": None,
                    "active": True,
                    "message": f"Job is {status}"
                })
        
        return {
            "jobs": jobs,
            "total": len(jobs),
            "active": len([j for j in jobs if j.get("active", False)]),
            "api_version": "2.0.0"
        }
        
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing jobs: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint with system information."""
    try:
        models_ok = verify_models()
        
        return {
            "status": "healthy" if models_ok else "degraded",
            "message": "FAKESYNCSTUDIO API is running with optimized uploads",
            "version": "2.0.0",
            "upload_optimizations": {
                "streaming_upload": "Enabled",
                "max_file_size": "500MB",
                "async_processing": "Enabled",
                "lenient_validation": "Enabled"
            },
            "features": {
                "face_enhancement": "CodeFormer (always enabled)",
                "face_parsing": "BiSeNet (always enabled)", 
                "blending": "Laplacian pyramid (optimal)",
                "quality": "Professional grade"
            },
            "models": {
                "face_detection": "InsightFace Buffalo_L",
                "face_swapping": "Inswapper", 
                "face_enhancement": "CodeFormer",
                "face_parsing": "BiSeNet"
            },
            "server_config": {
                "max_upload_size": "500MB",
                "upload_timeout": "60 minutes",
                "processing_timeout": "30 minutes",
                "cors": "Enabled for all origins",
                "cuda_support": True,
                "nsfw_detection": NSFW_DETECTION_ENABLED,
                "nsfw_threshold": NSFW_THRESHOLD if NSFW_DETECTION_ENABLED else None
            },
            "models_status": "ok" if models_ok else "missing_models"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Health check failed: {e}"
        }

@app.get("/")
async def root():
    """API root endpoint with usage information."""
    return {
        "name": "FAKESYNCSTUDIO API",
        "version": "2.0.0", 
        "description": "Professional AI Face Swapping with optimized upload performance",
        "endpoints": {
            "upload": "POST /upload/ - Fast streaming upload",
            "status": "GET /jobs/{job_id}/status/ - Check job status",
            "result": "GET /jobs/{job_id}/result - Download result video",
            "logs": "GET /jobs/{job_id}/stream_logs - Stream processing logs",
            "jobs": "GET /jobs/ - List all jobs",
            "health": "GET /health - System health check"
        },
        "optimizations": [
            "⚡ Streaming uploads (no memory limit)",
            "🔄 Async file processing", 
            "📊 Upload progress tracking",
            "⏱️ Extended timeouts for large files",
            "🛡️ Lenient file validation"
        ],
        "features": [
            "✨ CodeFormer face enhancement (always enabled)",
            "🎨 BiSeNet face parsing (always enabled)",
            "🌊 Laplacian blending (optimal quality)",
            "⚡ GPU acceleration (CUDA support)",
            "📊 Real-time progress streaming",
            "🔄 Background processing"
        ]
    }

if __name__ == "__main__":
    print("🎭 Starting FAKESYNCSTUDIO API Server v2.0.0")
    print("⚙️ Initializing content safety systems...")
    
    # Initialize NSFW detector
    nsfw_initialized = initialize_nsfw_detector()
    if nsfw_initialized:
        print("✅ Content safety system active")
    else:
        print("⚠️ Content safety system disabled")
    
    print("⚡ Optimized for fast uploads and large files")
    print(f"🌐 Server will start on http://{API_HOST}:{API_PORT}")
    
    uvicorn.run(
        "main:app", 
        host=API_HOST, 
        port=API_PORT, 
        reload=False,
        log_level="info",
        access_log=True,
        limit_concurrency=10,
        limit_max_requests=1000
    )