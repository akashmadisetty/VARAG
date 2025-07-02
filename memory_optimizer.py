#!/usr/bin/env python3
"""
Memory Optimization Script for ColPali Model Comparison

This script provides memory optimization utilities for running ColPali models on T4 GPUs.
It includes functions for memory monitoring, model management, and optimization settings.
"""

import torch
import gc
import os
import psutil
from typing import Dict, Optional, Tuple

def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage information"""
    if not torch.cuda.is_available():
        return {"available": 0, "total": 0, "used": 0, "free": 0}
    
    # Get memory info in GB
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    cached = torch.cuda.memory_reserved(0) / (1024**3)
    free = total - cached
    
    return {
        "total": total,
        "allocated": allocated,
        "cached": cached,
        "free": free,
        "utilization": (cached / total) * 100
    }

def get_system_memory_info() -> Dict[str, float]:
    """Get current system memory usage information"""
    memory = psutil.virtual_memory()
    return {
        "total": memory.total / (1024**3),
        "available": memory.available / (1024**3),
        "used": memory.used / (1024**3),
        "percent": memory.percent
    }

def print_memory_status():
    """Print current memory status"""
    gpu_info = get_gpu_memory_info()
    sys_info = get_system_memory_info()
    
    print("🔍 Memory Status")
    print("=" * 50)
    print(f"🖥️  GPU Memory:")
    print(f"   Total: {gpu_info['total']:.2f} GB")
    print(f"   Allocated: {gpu_info['allocated']:.2f} GB")
    print(f"   Cached: {gpu_info['cached']:.2f} GB")
    print(f"   Free: {gpu_info['free']:.2f} GB")
    print(f"   Utilization: {gpu_info['utilization']:.1f}%")
    print(f"💻 System Memory:")
    print(f"   Total: {sys_info['total']:.2f} GB")
    print(f"   Used: {sys_info['used']:.2f} GB")
    print(f"   Available: {sys_info['available']:.2f} GB")
    print(f"   Usage: {sys_info['percent']:.1f}%")
    print("=" * 50)

def optimize_for_t4_gpu():
    """Apply optimization settings for T4 GPU"""
    print("🔧 Applying T4 GPU optimizations...")
    
    if torch.cuda.is_available():
        # Reserve some memory for other processes
        torch.cuda.set_per_process_memory_fraction(0.85)
        
        # Check BFloat16 support
        if torch.cuda.is_bf16_supported():
            print("✅ BFloat16 supported - models will use BFloat16 precision")
        else:
            print("⚠️ BFloat16 not supported - models will use Float16 precision")
        
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            print("✅ Flash attention enabled")
        except:
            print("⚠️ Flash attention not available")
        
        # Set deterministic algorithms for consistent memory usage
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        print("✅ T4 GPU optimizations applied")
    else:
        print("⚠️ CUDA not available")

def clear_memory():
    """Aggressively clear GPU and system memory"""
    print("🧹 Clearing memory...")
    
    # Clear Python garbage
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    
    print("✅ Memory cleared")

def memory_efficient_inference_settings():
    """Apply settings for memory-efficient inference"""
    print("🎯 Applying memory-efficient inference settings...")
    
    # Disable gradient computation globally
    torch.set_grad_enabled(False)
    
    # Set inference mode
    if hasattr(torch, '_C') and hasattr(torch._C, '_set_inference_mode'):
        torch._C._set_inference_mode(True)
    
    # Use mixed precision
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print("✅ Memory-efficient inference settings applied")

def estimate_model_memory(model_name: str) -> float:
    """Estimate memory requirement for a ColPali model"""
    # Rough estimates based on model sizes (in GB)
    model_memory_estimates = {
        "vidore/colpali-v1.3": 3.5,  # Base model
        "akashmadisetty/colpali-merged-model-hi-10k": 3.5,  # Fine-tuned model (similar size)
    }
    
    base_estimate = model_memory_estimates.get(model_name, 4.0)
    
    # Add overhead for processing (embeddings, attention maps, etc.)
    total_estimate = base_estimate * 1.8  # 80% overhead
    
    return total_estimate

def check_memory_for_comparison() -> Tuple[bool, str]:
    """Check if there's enough memory for model comparison"""
    gpu_info = get_gpu_memory_info()
    
    if gpu_info["total"] == 0:
        return False, "No GPU available"
    
    # Estimate memory needed for comparison (sequential loading)
    single_model_memory = estimate_model_memory("vidore/colpali-v1.3")
    
    if gpu_info["free"] < single_model_memory:
        return False, f"Insufficient GPU memory. Need: {single_model_memory:.1f}GB, Available: {gpu_info['free']:.1f}GB"
    
    return True, f"Memory check passed. Available: {gpu_info['free']:.1f}GB, Needed: {single_model_memory:.1f}GB"

def optimize_gradio_settings():
    """Apply optimizations for Gradio interface"""
    print("🎨 Applying Gradio optimizations...")
    
    # Set environment variables for optimization
    os.environ["GRADIO_TEMP_DIR"] = "/tmp/gradio"
    os.environ["GRADIO_SERVER_PORT"] = "7860"
    
    # Limit concurrent requests
    os.environ["GRADIO_CONCURRENCY_COUNT"] = "1"
    
    print("✅ Gradio optimizations applied")

def get_optimization_recommendations() -> list:
    """Get memory optimization recommendations based on current system"""
    recommendations = []
    
    gpu_info = get_gpu_memory_info()
    sys_info = get_system_memory_info()
    
    if gpu_info["total"] > 0 and gpu_info["total"] < 20:  # T4 or similar
        recommendations.append("🔧 Use sequential model loading instead of simultaneous")
        recommendations.append("🖼️ Limit similarity map visualizations to 6 per image")
        recommendations.append("📏 Use image thumbnailing to reduce memory usage")
        recommendations.append("🧹 Clear memory between model switches")
    
    if gpu_info["utilization"] > 80:
        recommendations.append("⚠️ High GPU utilization - consider reducing batch size")
        recommendations.append("🔄 Use model switching instead of keeping both models loaded")
    
    if sys_info["percent"] > 80:
        recommendations.append("💻 High system memory usage - close unnecessary applications")
    
    if gpu_info["free"] < 2.0:
        recommendations.append("🚨 Low GPU memory - enable aggressive memory clearing")
        recommendations.append("📉 Consider using lower precision (half/bfloat16)")
    
    return recommendations

def run_memory_optimization():
    """Run complete memory optimization routine"""
    print("🚀 Running Memory Optimization for ColPali Models")
    print("=" * 60)
    
    print_memory_status()
    
    print("\n🔧 Applying Optimizations...")
    optimize_for_t4_gpu()
    memory_efficient_inference_settings()
    optimize_gradio_settings()
    clear_memory()
    
    print("\n🎯 Optimization Recommendations:")
    recommendations = get_optimization_recommendations()
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print("\n📊 Memory Check for Model Comparison:")
    can_compare, message = check_memory_for_comparison()
    print(f"{'✅' if can_compare else '❌'} {message}")
    
    print("\n" + "=" * 60)
    print("✅ Memory optimization complete!")
    
    return can_compare

if __name__ == "__main__":
    run_memory_optimization()
