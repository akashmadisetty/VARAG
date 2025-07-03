"""
Modal deployment configuration for ColPali comparison apps.

This module provides four different entrypoints for deploying ColPali comparison UIs:

1. comp_demo() - Original comparison app with full feature set
2. compv2() - Clean split-screen comparison UI (recommended for T4/L4)
3. start_optimized() - Optimized startup script


Usage:
    # Deploy the clean split-screen comparison UI (recommended for T4/L4)
    python -m modal run modal_test.py::compv2
    
    # Deploy the original comparison demo
    python -m modal run modal_test.py::comp_demo
    
    # Deploy the optimized startup version
    python -m modal run modal_test.py::start_optimized
    
    # Deploy the UNLIMITED COMPUTE version (H100 + 128GB RAM)
    python -m modal run modal_test.py::comp_unlimited

Each function will:
- Set up the required environment with CUDA support
- Mount the persistent volume for model caching
- Launch a Gradio app accessible via a public URL
- Handle memory optimization for T4/L4 GPUs
"""

import modal
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import os

app = modal.App("colpali-finetuning")

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"


# Persistent volume for model caching and data storage
col_vol=modal.Volume.from_name("colpali-engine-compare",create_if_missing=True)
VOLUME_PATH="/root/colpali-engine-compare"
HF_CACHE_PATH = f"{VOLUME_PATH}/hf_cache"
MODEL_PATH = f"{VOLUME_PATH}/models"

# Docker image with all dependencies
inference_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
   .apt_install("git")
    .run_commands([
        "git clone https://github.com/akashmadisetty/VARAG",
        "cd VARAG && pip install -e ."
    ])
    .pip_install("colpali-engine[interpretability]")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0", 
        "torchaudio==2.6.0",
        "xformers",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "huggingface_hub[hf-transfer]",
        "gradio",
    )
    .env({
        "HF_HUB_CACHE": HF_CACHE_PATH, 
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "GEMINI_API_KEY":"AIzaSyDT2XwygpeaLwT1qtNfjoCQrOzoeK_4Q2E",
    })
)

# Function for comp_demo.py - Original comparison app
@app.function(
    image=inference_image,
    gpu="L4",
    timeout=7200,  # 2 hour timeout
    volumes={
        VOLUME_PATH: col_vol,
    },
    secrets=[modal.Secret.from_name("hf-wandb-vyoman-secrets")]  # For HF token
)
def comp_demo():
    import sys
    import os
    
    # Setup environment and paths
    os.environ["HF_HUB_CACHE"] = HF_CACHE_PATH
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Change to VARAG directory
    varag_path = "/root/VARAG"
    if os.path.exists(varag_path):
        sys.path.insert(0, varag_path)
        os.chdir(varag_path)
    
    # Import after path setup
    import comp_demo
    import gradio as gr

    print(f"🚀 Starting comp_demo with models:")
    print(f"   Base: vidore/colpali-v1.3")
    print(f"   Fine-tuned: akashmadisetty/colpali-merged-model-hi-10k")
    print(f"   Cache: {HF_CACHE_PATH}")    # Initialize the Gradio interface
    app = comp_demo.gradio_interface()

    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
    
    return {"status": "comp_demo app launched successfully"}


# Function for comp_v2.py - Clean split-screen comparison UI
@app.function(
    image=inference_image,
    gpu="L4",
    timeout=7200,  # 2 hour timeout
    volumes={
        VOLUME_PATH: col_vol,
    },
    secrets=[modal.Secret.from_name("hf-wandb-vyoman-secrets")]  # For HF token
)
def compv2():
    import sys
    import os
    
    # Setup environment and paths
    os.environ["HF_HUB_CACHE"] = HF_CACHE_PATH
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Change to VARAG directory
    varag_path = "/root/VARAG"
    if os.path.exists(varag_path):
        sys.path.insert(0, varag_path)
        os.chdir(varag_path)
    
    # Import after path setup
    import comp_v2
    import gradio as gr

    print(f"🚀 Starting comp_v2 (clean split-screen UI)")
    print(f"   Cache: {HF_CACHE_PATH}")
    print(f"   Models: Will be loaded on-demand for comparison")    # Initialize the clean split-screen comparison UI
    app = comp_v2.create_interface()

    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
    
    return {"status": "comp_v2 app launched successfully"}


# Function for start_optimized.py - Optimized startup script
@app.function(
    image=inference_image,
    gpu="L4",
    timeout=7200,  # 2 hour timeout
    volumes={
        VOLUME_PATH: col_vol,
    },
    secrets=[modal.Secret.from_name("hf-wandb-vyoman-secrets")]  # For HF token
)
def start_optimized():
    import sys
    import os
    
    # Setup environment and paths
    os.environ["HF_HUB_CACHE"] = HF_CACHE_PATH
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Change to VARAG directory
    varag_path = "/root/VARAG"
    if os.path.exists(varag_path):
        sys.path.insert(0, varag_path)
        os.chdir(varag_path)
      # Import after path setup
    import start_optimized

    print(f"🚀 Starting optimized ColPali app")
    print(f"   Cache: {HF_CACHE_PATH}")

    # Set up optimizations like start_optimized does
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["GRADIO_CONCURRENCY_COUNT"] = "1"
    os.environ["FORCE_SAFE_PRECISION"] = "1"
    
    # Import and create the app directly like start_optimized does
    import comp_demo
    app = comp_demo.gradio_interface()

    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
    
    return {"status": "start_optimized app launched successfully"}


# Legacy function for backward compatibility
@app.function(
    image=inference_image,
    gpu="L4",
    timeout=7200,  # 2 hour timeout
    volumes={
        VOLUME_PATH: col_vol,
    },
    secrets=[modal.Secret.from_name("hf-wandb-vyoman-secrets")]  # For HF token
)
def gradio_interface():
    """Legacy function - use comp_demo() instead"""
    return comp_demo.remote()

