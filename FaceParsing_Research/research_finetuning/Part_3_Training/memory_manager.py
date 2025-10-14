# C:\SwapFace2Pon\FaceParsing_Research\research_finetuning\Part_3_Training\memory_manager.py

import torch
import gc
import os
import psutil # For CPU memory usage
from typing import Dict

def clear_cuda_cache() -> None:
    """
    Clears CUDA memory cache to free up GPU memory.
    Also calls Python's garbage collector to release unreferenced objects.
    This is crucial for managing VRAM in PyTorch, especially with limited memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        # print("CUDA cache cleared and garbage collected.") # Uncomment for verbose output
    # else:
        # print("CUDA not available, no cache to clear.") # Uncomment for verbose output


def log_gpu_memory_usage(stage: str = "current") -> Dict[str, float]:
    """
    Logs and returns current GPU memory usage statistics.

    Args:
        stage (str): A descriptive string indicating the stage of logging (e.g., "before_batch", "after_epoch").

    Returns:
        Dict[str, float]: A dictionary containing memory statistics in MB.
                          Keys include 'allocated_mb', 'cached_mb', 'peak_allocated_mb', 'peak_cached_mb'.
                          Returns empty dict if CUDA is not available.
    """
    mem_info = {}
    if torch.cuda.is_available():
        allocated_bytes = torch.cuda.memory_allocated()
        cached_bytes = torch.cuda.memory_reserved() # Total memory PyTorch has reserved
        
        # Convert bytes to MB
        allocated_mb = allocated_bytes / (1024 * 1024)
        cached_mb = cached_bytes / (1024 * 1024)
        
        # Peak memory usage since the start of the program or last reset
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_cached_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)

        mem_info = {
            'allocated_mb': allocated_mb,
            'cached_mb': cached_mb,
            'peak_allocated_mb': peak_allocated_mb,
            'peak_cached_mb': peak_cached_mb
        }
        
        # print(f"GPU Memory ({stage}): Allocated: {allocated_mb:.2f} MB, Cached: {cached_mb:.2f} MB, "
        #       f"Peak Allocated: {peak_allocated_mb:.2f} MB, Peak Cached: {peak_cached_mb:.2f} MB") # Uncomment for verbose output
    # else:
        # print(f"GPU Memory ({stage}): CUDA not available.") # Uncomment for verbose output
        
    return mem_info


def get_cpu_memory_usage() -> Dict[str, float]:
    """
    Returns current CPU memory usage statistics for the current process.

    Returns:
        Dict[str, float]: A dictionary containing memory statistics in MB.
                          Keys include 'used_mb' (Resident Set Size), 'total_mb' (system total), 'percent_used'.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    used_mb = mem_info.rss / (1024 * 1024) # Resident Set Size (physical memory used by the process)
    total_mb = psutil.virtual_memory().total / (1024 * 1024) # Total system memory
    percent_used = (used_mb / total_mb) * 100 if total_mb > 0 else 0
    
    # print(f"CPU Memory: Used: {used_mb:.2f} MB, Total: {total_mb:.2f} MB, Percent Used: {percent_used:.2f}%") # Uncomment for verbose output
    
    return {
        'used_mb': used_mb,
        'total_mb': total_mb,
        'percent_used': percent_used
    }


if __name__ == "__main__":
    print("Testing memory_manager.py...")

    # Test 1: Initial memory usage
    print("\n--- Initial Memory Usage ---")
    gpu_mem_init = log_gpu_memory_usage("initial")
    cpu_mem_init = get_cpu_memory_usage()
    print(f"Initial GPU: {gpu_mem_init}")
    print(f"Initial CPU: {cpu_mem_init}")

    # Test 2: Allocate some tensors to simulate memory usage
    if torch.cuda.is_available():
        print("\n--- Allocating CUDA tensors ---")
        # Allocate a large tensor that should take some VRAM
        dummy_tensor_gpu = torch.randn(2048, 2048, 4, device='cuda') # Approx 32MB
        print("Dummy GPU tensor allocated.")
        gpu_mem_after_alloc = log_gpu_memory_usage("after_alloc_gpu")
        print(f"After GPU allocation: {gpu_mem_after_alloc}")

        # Test 3: Clear CUDA cache
        print("\n--- Clearing CUDA cache ---")
        clear_cuda_cache()
        gpu_mem_after_clear = log_gpu_memory_usage("after_clear_gpu")
        print(f"After GPU cache clear: {gpu_mem_after_clear}")
        
        # Verify that allocated memory has decreased
        # Note: allocated_mb might not drop to 0 immediately due to PyTorch's caching allocator,
        # but it should be significantly less than after allocation.
        if gpu_mem_after_clear['allocated_mb'] < gpu_mem_after_alloc['allocated_mb']:
            print("CUDA memory allocation decreased after clearing cache. Success.")
        else:
            print("CUDA memory allocation did NOT decrease as expected after clearing cache. Check output.")
            
        del dummy_tensor_gpu # Ensure tensor is truly gone
        clear_cuda_cache() # Clear again to ensure all is released

    else:
        print("\nCUDA not available, skipping GPU memory tests.")

    # Test 4: Allocate some tensors on CPU
    print("\n--- Allocating CPU tensors ---")
    dummy_tensor_cpu = torch.randn(5000, 5000, 3) # Approx 75MB
    print("Dummy CPU tensor allocated.")
    cpu_mem_after_alloc = get_cpu_memory_usage()
    print(f"After CPU allocation: {cpu_mem_after_alloc}")

    # Test 5: Deallocate CPU tensors and check memory
    print("\n--- Deallocating CPU tensors ---")
    del dummy_tensor_cpu
    gc.collect() # Force garbage collection
    cpu_mem_after_dealloc = get_cpu_memory_usage()
    print(f"After CPU deallocation: {cpu_mem_after_dealloc}")

    if cpu_mem_after_dealloc['used_mb'] < cpu_mem_after_alloc['used_mb']:
        print("CPU memory usage decreased after deallocation. Success.")
    else:
        print("CPU memory usage did NOT decrease as expected after deallocation. Check output.")

    print("\nMemory manager tests completed.")
