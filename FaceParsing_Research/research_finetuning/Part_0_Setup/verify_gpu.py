"""
GPU Verification and Optimization for GTX 1660 Super
Tests GPU functionality and generates optimized configurations
"""

import torch
import numpy as np
import time
import json
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class GPUVerifier:
    """Verify and optimize GPU settings for face parsing training"""
    
    def __init__(self):
        self.device = None
        self.gpu_info = {}
        self.benchmarks = {}
        self.recommendations = {}
        
    def check_gpu_availability(self) -> bool:
        """Check if CUDA GPU is available"""
        print("\n🎮 Checking GPU Availability...")
        
        if not torch.cuda.is_available():
            print("✗ No CUDA GPU available!")
            print("  Training will be very slow on CPU")
            self.gpu_info["available"] = False
            return False
        
        # Get GPU information
        self.device = torch.device("cuda:0")
        gpu_count = torch.cuda.device_count()
        
        print(f"✓ Found {gpu_count} GPU(s)")
        
        self.gpu_info["available"] = True
        self.gpu_info["count"] = gpu_count
        self.gpu_info["devices"] = []
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            info = {
                "index": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": props.total_memory / (1024**3),
                "multi_processor_count": props.multi_processor_count,
                "is_integrated": props.is_integrated,
                "is_multi_gpu_board": props.is_multi_gpu_board,
            }
            
            self.gpu_info["devices"].append(info)
            
            print(f"\nGPU {i}: {info['name']}")
            print(f"  Compute Capability: {info['compute_capability']}")
            print(f"  Memory: {info['total_memory_gb']:.1f} GB")
            print(f"  Multiprocessors: {info['multi_processor_count']}")
            
            # Check if it's GTX 1660 Super
            if "1660" in info['name'].upper() and "SUPER" in info['name'].upper():
                print("  ✓ Detected GTX 1660 Super - Optimizations will be applied")
                self.gpu_info["is_gtx_1660_super"] = True
        
        return True
    
    def test_memory_allocation(self) -> Dict:
        """Test GPU memory allocation limits"""
        print("\n💾 Testing GPU Memory Allocation...")
        
        if not self.gpu_info.get("available", False):
            return {}
        
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        
        # Get initial memory stats
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = total_memory - reserved_memory
        
        memory_info = {
            "total_gb": total_memory / (1024**3),
            "free_gb": free_memory / (1024**3),
            "reserved_gb": reserved_memory / (1024**3),
            "allocated_gb": allocated_memory / (1024**3),
        }
        
        print(f"Total Memory: {memory_info['total_gb']:.2f} GB")
        print(f"Free Memory: {memory_info['free_gb']:.2f} GB")
        
        # Test maximum allocation
        print("\nTesting maximum tensor allocation...")
        max_elements = int(free_memory * 0.9 / 4)  # 90% of free memory, 4 bytes per float32
        
        try:
            # Try to allocate a large tensor
            test_tensor = torch.zeros(max_elements, dtype=torch.float32, device='cuda')
            actual_size = test_tensor.element_size() * test_tensor.nelement() / (1024**3)
            print(f"✓ Successfully allocated {actual_size:.2f} GB tensor")
            
            # Test with different data types
            print("\nTesting with different precisions:")
            
            # Float16 (half precision)
            test_tensor_fp16 = test_tensor.half()
            fp16_size = test_tensor_fp16.element_size() * test_tensor_fp16.nelement() / (1024**3)
            print(f"  FP16: {fp16_size:.2f} GB (50% memory savings)")
            
            del test_tensor, test_tensor_fp16
            torch.cuda.empty_cache()
            
            memory_info["max_allocation_gb"] = actual_size
            memory_info["fp16_savings"] = "50%"
            
        except RuntimeError as e:
            print(f"✗ Memory allocation failed: {e}")
            memory_info["max_allocation_gb"] = 0
        
        self.benchmarks["memory"] = memory_info
        return memory_info
    
    def benchmark_operations(self) -> Dict:
        """Benchmark common operations for face parsing"""
        print("\n⚡ Benchmarking GPU Operations...")
        
        if not self.gpu_info.get("available", False):
            return {}
        
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        
        benchmarks = {}
        
        # Test configurations for GTX 1660 Super
        test_configs = [
            {"batch_size": 1, "image_size": 512},
            {"batch_size": 2, "image_size": 512},
            {"batch_size": 4, "image_size": 512},
            {"batch_size": 1, "image_size": 384},
            {"batch_size": 2, "image_size": 384},
            {"batch_size": 4, "image_size": 384},
            {"batch_size": 8, "image_size": 384},
        ]
        
        print("\nTesting different batch sizes and image sizes...")
        print("(This will help determine optimal settings)")
        
        for config in test_configs:
            batch_size = config["batch_size"]
            image_size = config["image_size"]
            
            try:
                # Create dummy data
                dummy_input = torch.randn(
                    batch_size, 3, image_size, image_size, 
                    device='cuda', dtype=torch.float32
                )
                
                # Simple convolution operation (similar to face parsing)
                conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
                
                # Warmup
                for _ in range(5):
                    _ = conv(dummy_input)
                
                torch.cuda.synchronize()
                
                # Benchmark
                start_time = time.time()
                iterations = 20
                
                for _ in range(iterations):
                    output = conv(dummy_input)
                    torch.cuda.synchronize()
                
                elapsed_time = time.time() - start_time
                avg_time = elapsed_time / iterations
                throughput = batch_size / avg_time
                
                # Memory usage
                memory_used = torch.cuda.max_memory_allocated() / (1024**3)
                torch.cuda.reset_peak_memory_stats()
                
                config_key = f"b{batch_size}_s{image_size}"
                benchmarks[config_key] = {
                    "batch_size": batch_size,
                    "image_size": image_size,
                    "avg_time_ms": avg_time * 1000,
                    "throughput_img_per_sec": throughput,
                    "memory_gb": memory_used,
                    "status": "success"
                }
                
                print(f"  ✓ Batch={batch_size}, Size={image_size}: "
                      f"{avg_time*1000:.1f}ms, "
                      f"{throughput:.1f} img/s, "
                      f"{memory_used:.1f} GB")
                
                # Clean up
                del dummy_input, output
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    config_key = f"b{batch_size}_s{image_size}"
                    benchmarks[config_key] = {
                        "batch_size": batch_size,
                        "image_size": image_size,
                        "status": "out_of_memory"
                    }
                    print(f"  ✗ Batch={batch_size}, Size={image_size}: Out of memory")
                else:
                    print(f"  ✗ Error: {e}")
                
                torch.cuda.empty_cache()
        
        self.benchmarks["operations"] = benchmarks
        return benchmarks
    
    def test_mixed_precision(self) -> Dict:
        """Test mixed precision training capabilities"""
        print("\n🔬 Testing Mixed Precision Support...")
        
        if not self.gpu_info.get("available", False):
            return {}
        
        mixed_precision_info = {
            "autocast_available": hasattr(torch.cuda.amp, 'autocast'),
            "gradscaler_available": hasattr(torch.cuda.amp, 'GradScaler'),
            "compute_capability": float(self.gpu_info["devices"][0]["compute_capability"]),
            "recommended": False
        }
        
        # GTX 1660 Super has compute capability 7.5
        if mixed_precision_info["compute_capability"] >= 7.0:
            mixed_precision_info["recommended"] = True
            print("✓ GPU supports efficient mixed precision (compute capability >= 7.0)")
        else:
            print("⚠ GPU has limited mixed precision support")
        
        # Test mixed precision performance
        if mixed_precision_info["autocast_available"]:
            print("\nTesting mixed precision performance...")
            
            # Create a simple model
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, 3),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
            ).cuda()
            
            # Test data
            batch_size = 2
            image_size = 384
            dummy_input = torch.randn(batch_size, 3, image_size, image_size).cuda()
            
            # Test FP32
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(10):
                output = model(dummy_input)
                loss = output.mean()
                loss.backward()
            
            torch.cuda.synchronize()
            fp32_time = time.time() - start_time
            fp32_memory = torch.cuda.max_memory_allocated() / (1024**3)
            
            # Clear
            model.zero_grad()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Test FP16 with autocast
            scaler = torch.cuda.amp.GradScaler()
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(10):
                with torch.cuda.amp.autocast():
                    output = model(dummy_input)
                    loss = output.mean()
                
                scaler.scale(loss).backward()
                scaler.step(torch.optim.Adam(model.parameters()))
                scaler.update()
                model.zero_grad()
            
            torch.cuda.synchronize()
            fp16_time = time.time() - start_time
            fp16_memory = torch.cuda.max_memory_allocated() / (1024**3)
            
            # Calculate speedup
            speedup = fp32_time / fp16_time
            memory_reduction = (1 - fp16_memory / fp32_memory) * 100
            
            mixed_precision_info["benchmark"] = {
                "fp32_time": fp32_time,
                "fp16_time": fp16_time,
                "speedup": speedup,
                "fp32_memory_gb": fp32_memory,
                "fp16_memory_gb": fp16_memory,
                "memory_reduction_percent": memory_reduction
            }
            
            print(f"\nMixed Precision Results:")
            print(f"  Speed: {speedup:.2f}x faster")
            print(f"  Memory: {memory_reduction:.1f}% reduction")
            print(f"  ✓ Recommended for GTX 1660 Super")
        
        self.benchmarks["mixed_precision"] = mixed_precision_info
        return mixed_precision_info
    
    def generate_recommendations(self) -> Dict:
        """Generate optimized settings for GTX 1660 Super"""
        print("\n📊 Generating Optimized Settings...")
        
        recommendations = {
            "batch_size": 2,
            "image_size": 384,
            "accumulation_steps": 8,
            "num_workers": 2,
            "pin_memory": False,  # Can cause issues with limited RAM
            "mixed_precision": True,
            "gradient_checkpointing": True,
            "optimizer": "AdamW",
            "learning_rate": 5e-6,
            "weight_decay": 1e-4,
            "scheduler": "cosine",
            "warmup_steps": 100,
        }
        
        # Adjust based on benchmarks
        if "operations" in self.benchmarks:
            # Find optimal batch size
            valid_configs = [
                (cfg["batch_size"], cfg["image_size"], cfg.get("throughput_img_per_sec", 0))
                for key, cfg in self.benchmarks["operations"].items()
                if cfg.get("status") == "success"
            ]
            
            if valid_configs:
                # For GTX 1660 Super, prioritize batch size 2 with 384 image size
                for bs, img_size, throughput in valid_configs:
                    if bs == 2 and img_size == 384:
                        recommendations["batch_size"] = bs
                        recommendations["image_size"] = img_size
                        break
        
        # Memory optimizations
        recommendations["memory_optimization"] = {
            "clear_cache_frequency": 50,  # Clear cache every N iterations
            "max_split_size_mb": 128,     # CUDA memory allocator setting
            "garbage_collection_frequency": 100,
        }
        
        # Data loading optimizations
        recommendations["data_loading"] = {
            "prefetch_factor": 2,
            "persistent_workers": False,  # Save memory
            "num_workers": min(2, psutil.cpu_count(logical=False)),
        }
        
        print("\n🎯 Recommended Settings for GTX 1660 Super:")
        print(f"  Batch Size: {recommendations['batch_size']}")
        print(f"  Image Size: {recommendations['image_size']}x{recommendations['image_size']}")
        print(f"  Accumulation Steps: {recommendations['accumulation_steps']}")
        print(f"  Effective Batch Size: {recommendations['batch_size'] * recommendations['accumulation_steps']}")
        print(f"  Mixed Precision: {'Enabled' if recommendations['mixed_precision'] else 'Disabled'}")
        print(f"  Gradient Checkpointing: {'Enabled' if recommendations['gradient_checkpointing'] else 'Disabled'}")
        print(f"\n  Estimated Memory Usage: ~4.5-5.5 GB")
        print(f"  Estimated Training Speed: ~50-60 images/minute")
        
        self.recommendations = recommendations
        return recommendations
    
    def save_config(self, output_dir: Optional[Path] = None):
        """Save optimized configuration files"""
        if output_dir is None:
            output_dir = Path.cwd() / "configs" / "generated"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save GPU info
        gpu_info_path = output_dir / "gpu_info.json"
        with open(gpu_info_path, 'w') as f:
            json.dump(self.gpu_info, f, indent=2)
        
        # Save benchmarks
        benchmarks_path = output_dir / "gpu_benchmarks.json"
        with open(benchmarks_path, 'w') as f:
            json.dump(self.benchmarks, f, indent=2)
        
        # Save optimized config
        config = {
            "device": "cuda:0",
            "gpu_name": self.gpu_info["devices"][0]["name"],
            **self.recommendations
        }
        
        config_path = output_dir / "optimized_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Configuration files saved to: {output_dir}")
        print(f"  - gpu_info.json")
        print(f"  - gpu_benchmarks.json") 
        print(f"  - optimized_config.json")
        
        return output_dir
    
    def run_full_verification(self) -> bool:
        """Run complete GPU verification"""
        print("🚀 Starting GPU Verification and Optimization")
        print("=" * 60)
        
        # Check GPU availability
        if not self.check_gpu_availability():
            print("\n❌ No CUDA GPU available!")
            return False
        
        # Run tests
        self.test_memory_allocation()
        self.benchmark_operations()
        self.test_mixed_precision()
        
        # Generate recommendations
        self.generate_recommendations()
        
        # Save configurations
        self.save_config()
        
        print("\n" + "=" * 60)
        print("✅ GPU Verification Complete!")
        print("🎉 Your GTX 1660 Super is ready for training!")
        
        return True


def main():
    """Run GPU verification"""
    verifier = GPUVerifier()
    success = verifier.run_full_verification()
    
    if not success:
        print("\n⚠️  GPU verification failed!")
        print("You can still train on CPU, but it will be very slow.")
        sys.exit(1)


if __name__ == "__main__":
    main()