#!/usr/bin/env python3
"""
Optimized Startup Script for ColPali Model Comparison

This script starts the Gradio app with memory optimizations for T4 GPU.
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Start ColPali Model Comparison with Memory Optimizations")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share feature")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio server")
    parser.add_argument("--memory-check", action="store_true", default=True, help="Run memory check before starting")
    parser.add_argument("--optimize", action="store_true", default=True, help="Apply memory optimizations")
    
    args = parser.parse_args()
    
    print("🚀 Starting ColPali Model Comparison with T4 GPU Optimizations")
    print("=" * 60)
    
    if args.memory_check:
        print("🔍 Running memory check...")
        try:
            from memory_optimizer import run_memory_optimization
            can_run = run_memory_optimization()
            if not can_run:
                print("❌ Memory check failed. Please free up GPU memory and try again.")
                return 1
        except ImportError:
            print("⚠️ Memory optimizer not available - proceeding anyway")
    
    if args.optimize:
        print("🔧 Applying memory optimizations...")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        os.environ["GRADIO_CONCURRENCY_COUNT"] = "1"
        os.environ["GRADIO_SERVER_PORT"] = str(args.port)
        
        # Add precision safety
        print("🔧 Applying precision safety measures...")
        os.environ["FORCE_SAFE_PRECISION"] = "1"
    
    print("🎯 Starting Gradio interface...")
    try:
        from comp_demo import gradio_interface
        app = gradio_interface()
        
        print(f"✅ App ready! Open in browser: http://localhost:{args.port}")
        if args.share:
            print("🌐 Share link will be generated...")
            
        app.launch(
            share=args.share,
            server_port=args.port,
            inbrowser=True,
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
