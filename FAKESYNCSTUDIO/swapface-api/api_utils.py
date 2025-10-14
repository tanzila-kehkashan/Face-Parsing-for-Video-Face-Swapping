import os
import uuid
import shutil
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import aiofiles

from settings import UPLOAD_DIR, OUTPUT_DIR

def generate_unique_id() -> str:
    """Generate a unique ID for each request"""
    return str(uuid.uuid4())

def create_job_directory(job_id: str) -> Dict[str, Path]:
    """Create directories for a processing job"""
    job_dirs = {
        "source": UPLOAD_DIR / "sources" / job_id,
        "target": UPLOAD_DIR / "targets" / job_id,
        "output": OUTPUT_DIR / job_id,
        "chunks": UPLOAD_DIR / "chunks" / job_id  # For chunked uploads
    }
    
    for dir_path in job_dirs.values():
        dir_path.mkdir(exist_ok=True, parents=True)
    
    return job_dirs

async def save_uploaded_file_async(file_bytes: bytes, file_path: Path, chunk_size: int = 8192) -> int:
    """
    Save uploaded file bytes asynchronously for better performance
    
    Args:
        file_bytes: File data as bytes
        file_path: Destination file path
        chunk_size: Size of chunks to write
    
    Returns:
        int: Number of bytes written
    """
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Write file asynchronously in chunks
        async with aiofiles.open(file_path, "wb") as f:
            total_written = 0
            for i in range(0, len(file_bytes), chunk_size):
                chunk = file_bytes[i:i + chunk_size]
                await f.write(chunk)
                total_written += len(chunk)
        
        # Verify file was written correctly
        if not file_path.exists() or file_path.stat().st_size == 0:
            raise Exception(f"Failed to write file {file_path}")
        
        actual_size = file_path.stat().st_size
        print(f"Successfully saved file: {file_path} ({actual_size:,} bytes)")
        
        return actual_size
            
    except Exception as e:
        # Clean up partial file on error
        if file_path.exists():
            try:
                file_path.unlink()
            except:
                pass
        raise Exception(f"Error saving file {file_path}: {e}")

def save_uploaded_file(file_bytes: bytes, file_path: Path) -> None:
    """
    Synchronous version for backwards compatibility
    """
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        
        # Verify file was written correctly
        if not file_path.exists() or file_path.stat().st_size == 0:
            raise Exception(f"Failed to write file {file_path}")
            
        print(f"Successfully saved file: {file_path} ({file_path.stat().st_size:,} bytes)")
            
    except Exception as e:
        # Clean up partial file on error
        if file_path.exists():
            try:
                file_path.unlink()
            except:
                pass
        raise Exception(f"Error saving file {file_path}: {e}")

def get_job_info(job_id: str) -> Dict:
    """Get information about a job's status and files"""
    output_dir = OUTPUT_DIR / job_id
    
    if not output_dir.exists():
        return {"status": "not_found"}
    
    # Look for any video files in the output directory
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    output_files = []
    
    for ext in video_extensions:
        output_files.extend(list(output_dir.glob(f"*{ext}")))
    
    # Also check for specific filename patterns
    specific_patterns = ["swapped_video.mp4", "result.mp4", "output_*.mp4", "api_result_*.mp4", "api_result.mp4"]
    for pattern in specific_patterns:
        output_files.extend(list(output_dir.glob(pattern)))
    
    # Remove duplicates and sort by size (largest first)
    output_files = list(set(output_files))
    
    if not output_files:
        # Check if there are any files at all in the directory
        all_files = list(output_dir.glob("*"))
        if all_files:
            print(f"Job {job_id}: Found files but no videos: {[f.name for f in all_files[:5]]}")  # Show first 5
        return {"status": "processing"}
    
    # Get the largest/most recent file (likely the correct output)
    if len(output_files) > 1:
        # Prefer files with larger size (likely the actual video)
        output_files.sort(key=lambda f: f.stat().st_size, reverse=True)
    
    output_file = output_files[0]
    
    # Verify the file is not empty and is a reasonable size
    file_size = output_file.stat().st_size
    if file_size < 1000:  # Less than 1KB is probably not a real video
        print(f"Job {job_id}: Output file {output_file.name} is too small ({file_size} bytes)")
        return {"status": "processing"}
    
    print(f"Job {job_id}: Found output file: {output_file.name} ({file_size:,} bytes)")
    
    return {
        "status": "completed",
        "output_path": str(output_file),
        "file_size": file_size,
        "created_at": datetime.fromtimestamp(output_file.stat().st_mtime).isoformat()
    }

def clean_up_job(job_id: str, cleanup_outputs: bool = True) -> None:
    """Remove all files related to a job
    
    Args:
        job_id: The job ID to clean up
        cleanup_outputs: Whether to also clean up output files (default: True)
    """
    dirs_to_remove = [
        UPLOAD_DIR / "sources" / job_id,
        UPLOAD_DIR / "targets" / job_id,
        UPLOAD_DIR / "chunks" / job_id,  # Clean up chunk directories too
    ]
    
    if cleanup_outputs:
        dirs_to_remove.append(OUTPUT_DIR / job_id)
    
    for dir_path in dirs_to_remove:
        if dir_path.exists() and dir_path.is_dir():
            try:
                shutil.rmtree(dir_path)
                print(f"Cleaned up directory: {dir_path}")
            except OSError as e:
                print(f"Error cleaning up directory {dir_path}: {e}")

def clean_up_old_chunks(max_age_hours: int = 24) -> None:
    """
    Clean up old chunk directories that might be left from failed uploads
    
    Args:
        max_age_hours: Remove chunk directories older than this many hours
    """
    chunks_dir = UPLOAD_DIR / "chunks"
    if not chunks_dir.exists():
        return
    
    import time
    current_time = time.time()
    cutoff_time = current_time - (max_age_hours * 3600)
    
    cleaned_count = 0
    for chunk_dir in chunks_dir.iterdir():
        if chunk_dir.is_dir():
            try:
                dir_mtime = chunk_dir.stat().st_mtime
                if dir_mtime < cutoff_time:
                    shutil.rmtree(chunk_dir)
                    cleaned_count += 1
                    print(f"Cleaned up old chunk directory: {chunk_dir}")
            except OSError as e:
                print(f"Error cleaning up old chunk directory {chunk_dir}: {e}")
    
    if cleaned_count > 0:
        print(f"Cleaned up {cleaned_count} old chunk directories")

async def merge_chunks_async(job_id: str, file_type: str, total_chunks: int, output_path: Path) -> int:
    """
    Merge uploaded chunks into a single file asynchronously
    
    Args:
        job_id: Job identifier
        file_type: 'source' or 'target'
        total_chunks: Total number of chunks expected
        output_path: Final output file path
    
    Returns:
        int: Total size of merged file
    """
    chunks_dir = UPLOAD_DIR / "chunks" / job_id
    
    # Verify all chunks exist
    chunk_paths = []
    for i in range(total_chunks):
        chunk_path = chunks_dir / f"{file_type}_chunk_{i:04d}"
        if not chunk_path.exists():
            raise Exception(f"Missing chunk {i} for {file_type} file")
        chunk_paths.append(chunk_path)
    
    # Merge chunks asynchronously
    total_size = 0
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    async with aiofiles.open(output_path, 'wb') as output_file:
        for chunk_path in chunk_paths:
            async with aiofiles.open(chunk_path, 'rb') as chunk_file:
                chunk_data = await chunk_file.read()
                await output_file.write(chunk_data)
                total_size += len(chunk_data)
    
    # Clean up chunks
    for chunk_path in chunk_paths:
        try:
            chunk_path.unlink()
        except OSError:
            pass
    
    # Remove chunk directory if empty
    try:
        chunks_dir.rmdir()
    except OSError:
        pass
    
    print(f"Merged {total_chunks} chunks into {output_path} ({total_size:,} bytes)")
    return total_size

def list_jobs() -> List[Dict]:
    """List all job directories and infer their status"""
    jobs_list: List[Dict] = []
    
    # Check output directory for jobs
    if OUTPUT_DIR.exists():
        for job_dir in OUTPUT_DIR.iterdir():
            if job_dir.is_dir():
                job_id = job_dir.name
                job_info = get_job_info(job_id)
                if job_info["status"] != "not_found":
                    jobs_list.append({
                        "id": job_id, 
                        "status": job_info["status"], 
                        "output_path": job_info.get("output_path"),
                        "file_size": job_info.get("file_size"),
                        "created_at": job_info.get("created_at")
                    })
    
    # Sort by creation time (newest first)
    jobs_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    return jobs_list

def get_disk_usage() -> Dict[str, int]:
    """Get disk usage information for upload and output directories"""
    def get_directory_size(path: Path) -> int:
        """Get total size of directory in bytes"""
        total = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        pass
        except (OSError, FileNotFoundError):
            pass
        return total
    
    usage = {
        "upload_dir_bytes": get_directory_size(UPLOAD_DIR) if UPLOAD_DIR.exists() else 0,
        "output_dir_bytes": get_directory_size(OUTPUT_DIR) if OUTPUT_DIR.exists() else 0,
    }
    
    # Add human-readable versions
    for key in list(usage.keys()):
        bytes_value = usage[key]
        mb_key = key.replace("_bytes", "_mb")
        usage[mb_key] = round(bytes_value / 1024 / 1024, 2)
    
    return usage

def cleanup_failed_jobs(max_age_hours: int = 2) -> int:
    """
    Clean up directories for jobs that appear to have failed
    (have upload directories but no output files and are old)
    
    Args:
        max_age_hours: Consider jobs older than this failed
    
    Returns:
        int: Number of jobs cleaned up
    """
    import time
    current_time = time.time()
    cutoff_time = current_time - (max_age_hours * 3600)
    
    cleaned_count = 0
    
    # Check for failed uploads (have source/target dirs but no outputs)
    for source_dir in (UPLOAD_DIR / "sources").glob("*"):
        if source_dir.is_dir():
            job_id = source_dir.name
            
            # Check if job is old enough
            try:
                dir_mtime = source_dir.stat().st_mtime
                if dir_mtime > cutoff_time:
                    continue  # Too recent, skip
            except OSError:
                continue
            
            # Check if there's no output
            output_dir = OUTPUT_DIR / job_id
            if not output_dir.exists() or not any(output_dir.glob("*.mp4")):
                # No output found, clean up
                try:
                    clean_up_job(job_id, cleanup_outputs=True)
                    cleaned_count += 1
                    print(f"Cleaned up failed job: {job_id}")
                except Exception as e:
                    print(f"Error cleaning up failed job {job_id}: {e}")
    
    # Also clean up old chunks
    clean_up_old_chunks(max_age_hours)
    
    return cleaned_count