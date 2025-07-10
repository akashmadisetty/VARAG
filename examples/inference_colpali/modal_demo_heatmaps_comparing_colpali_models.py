"""
Modal deployment configuration for ColPali comparison apps.

This module provides four different entrypoints for deploying ColPali comparison UIs:

1. comparision_demo() - Original comparison app with full feature set (optimized for T4/L4)
2. comparision_demo_original() - UNOPTIMIZED original app with both models loaded (no constraints)


Usage:
    # Deploy the UNOPTIMIZED original demo (both models loaded, no constraints)
    python -m modal run modal_demo_heatmaps_comparing_colpali_models.py::comparision_demo_original
    
    # Deploy the optimized comparison app (for T4/L4)
    python -m modal run modal_demo_heatmaps_comparing_colpali_models.py::comparision_demo


Each function will:
- Set up the required environment with CUDA support
- Mount the persistent volume for model caching
- Launch a Gradio app accessible via a public URL
- Handle memory optimization for T4/L4 GPUs (except comparision_demo_original)
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

# Function for comparision_demo.py - Original comparison app
@app.function(
    image=inference_image,
    gpu="L4",
    timeout=7200,  # 2 hour timeout
    volumes={
        VOLUME_PATH: col_vol,
    },
    secrets=[modal.Secret.from_name("hf-wandb-vyoman-secrets")]  # For HF token
)
def comparision_demo():
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
    try:
        # Try importing from the current directory or relative path (in container)
        import sys
        import importlib.util
        
        # Try several possible locations
        possible_paths = [
            os.path.join(os.getcwd(), "examples/inference_colpali/comparision_demo.py"),
            os.path.join(os.getcwd(), "comparision_demo.py"),
            os.path.join(os.path.dirname(__file__), "comparision_demo.py")
        ]
        
        module_loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Loading comparision_demo from: {path}")
                spec = importlib.util.spec_from_file_location("comparision_demo", path)
                comparision_demo = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(comparision_demo)
                module_loaded = True
                break
                
        if not module_loaded:
            print("Fallback to direct import")
            import comparision_demo
    except Exception as e:
        print(f"Error importing comparision_demo: {e}")
        print("Attempting to locate the file:")
        os.system("find /root -name 'comparision_demo.py'")
        raise
    
    import gradio as gr

    print(f"🚀 Starting comparision_demo with models:")
    print(f"   Base: vidore/colpali-v1.3")
    print(f"   Fine-tuned: akashmadisetty/colpali-merged-model-hi-10k")
    print(f"   Cache: {HF_CACHE_PATH}")    # Initialize the Gradio interface
    app = comparision_demo.gradio_interface()

    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
    
    return {"status": "comparision_demo app launched successfully"}


# Function for comparision_demo_original - UNOPTIMIZED original app with both models
@app.function(
    image=inference_image,
    gpu="A100",  # Use powerful GPU for unoptimized version
    timeout=7200,  # 2 hour timeout
    volumes={
        VOLUME_PATH: col_vol,
    },
    secrets=[modal.Secret.from_name("hf-wandb-vyoman-secrets")]  # For HF token
)
def comparision_demo_original():
    import sys
    import os
    import time
    import torch
    import gc
    import gradio as gr
    import pandas as pd
    from PIL import Image
    import base64
    import io
    import concurrent.futures
    from collections import namedtuple
    from dotenv import load_dotenv
    from typing import List, Dict, Any
    
    # Setup environment - NO optimizations
    os.environ["HF_HUB_CACHE"] = HF_CACHE_PATH
    # Remove all memory constraints - no PYTORCH_CUDA_ALLOC_CONF
    
    # Change to VARAG directory
    varag_path = "/root/VARAG"
    if os.path.exists(varag_path):
        sys.path.insert(0, varag_path)
        os.chdir(varag_path)
    
    # Import after path setup
    from sentence_transformers import SentenceTransformer
    from varag.rag import SimpleRAG, VisionRAG, ColpaliRAG, HybridColpaliRAG
    from varag.vlms import OpenAI
    from varag.llms import OpenAI as OpenAILLM
    from varag.vlms import LiteLLMVLM 
    from varag.llms import LiteLLM 
    from varag.chunking import FixedTokenChunker
    from varag.utils import get_model_colpali,create_similarity_mapper, analyze_multiple_images
    import lancedb
    
    load_dotenv()
    
    print(f"🚀 Starting UNOPTIMIZED ColPali Comparison App")
    print(f"💾 Cache: {HF_CACHE_PATH}")
    print(f"🔥 Loading both models simultaneously (no constraints)")
    print("=" * 60)
    
    # Use local temporary database to avoid permission conflicts with persistent volume
    import tempfile
    temp_db_dir = tempfile.mkdtemp(prefix="rag_demo_original_")
    print(f"📁 Using temporary database at: {temp_db_dir}")
    
    try:
        original_db = lancedb.connect(temp_db_dir)
        print("✅ Database connection established")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        # Fallback to memory-only database
        print("🔄 Falling back to in-memory database")
        original_db = lancedb.connect(":memory:")
    
    # Initialize embedding models
    print("📦 Loading text embedding model...")
    text_embedding_model = SentenceTransformer("BAAI/bge-base-en", trust_remote_code=True)
    print("✅ Text embedding model loaded")
    
    print("📦 Loading image embedding model...")
    image_embedding_model = SentenceTransformer("jinaai/jina-clip-v1", trust_remote_code=True)
    print("✅ Image embedding model loaded")
    
    # Load BOTH ColPali models simultaneously (original approach)
    print("📦 Loading base ColPali model (vidore/colpali-v1.3)...")
    base_colpali_model, base_colpali_processor = get_model_colpali("vidore/colpali-v1.3")
    base_similarity_mapper = create_similarity_mapper(base_colpali_model, base_colpali_processor)
    print("✅ Base ColPali model loaded successfully")
    
    print("📦 Loading fine-tuned ColPali model (akashmadisetty/colpali-merged-model-hi-10k)...")
    finetuned_colpali_model, finetuned_colpali_processor = get_model_colpali("akashmadisetty/colpali-merged-model-hi-10k")
    finetuned_similarity_mapper = create_similarity_mapper(finetuned_colpali_model, finetuned_colpali_processor)
    print("✅ Fine-tuned ColPali model loaded successfully")
    
    # Initialize ALL 4 RAG instances with unique table names to avoid conflicts
    simple_rag = SimpleRAG(
        text_embedding_model=text_embedding_model, 
        db=original_db, 
        table_name="originalSimpleDemo"
    )
    
    vision_rag = VisionRAG(
        image_embedding_model=image_embedding_model, 
        db=original_db, 
        table_name="originalVisionDemo"
    )
    
    # Use base model for main ColPali RAG (can be switched in UI)
    colpali_rag = ColpaliRAG(
        colpali_model=base_colpali_model,
        colpali_processor=base_colpali_processor,
        db=original_db,
        table_name="originalColpaliDemo",
    )
    
    hybrid_rag = HybridColpaliRAG(
        colpali_model=base_colpali_model,
        colpali_processor=base_colpali_processor,
        image_embedding_model=image_embedding_model,
        db=original_db,
        table_name="originalHybridDemo",
    )
    
    # Initialize VLM and LLM
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if gemini_api_key:
        gemini_model = "gemini/gemini-2.5-flash-preview-04-17"
        gem_llm = LiteLLM(model=gemini_model, api_key=gemini_api_key, verbose=False)
        gem_vlm = LiteLLMVLM(model=gemini_model, api_key=gemini_api_key, verbose=False)
        llm = gem_llm
        vlm = gem_vlm
        print(f"✅ Using Gemini with model: {gemini_model}")
    else:
        vlm = OpenAI()
        llm = OpenAILLM()
        print("✅ Using OpenAI provider")
    
    print("✅ All models and RAG systems initialized successfully!")
    
    # Define result structure
    IngestResult = namedtuple("IngestResult", ["status_text", "progress_table"])
    
    def ingest_data(pdf_files, use_ocr, chunk_size, progress=gr.Progress()):
        """Ingest PDFs into all 4 RAG systems"""
        if not pdf_files:
            return IngestResult("❌ No PDF files uploaded", pd.DataFrame())
        
        file_paths = [pdf_file.name for pdf_file in pdf_files]
        total_start_time = time.time()
        progress_data = []
        
        # SimpleRAG
        yield IngestResult(
            status_text="Starting SimpleRAG ingestion...\n",
            progress_table=pd.DataFrame(progress_data),
        )
        start_time = time.time()
        simple_rag.index(
            file_paths,
            recursive=False,
            chunking_strategy=FixedTokenChunker(chunk_size=chunk_size),
            metadata={"source": "gradio_upload"},
            overwrite=True,
            verbose=True,
            ocr=use_ocr,
        )
        simple_time = time.time() - start_time
        progress_data.append({"Technique": "SimpleRAG", "Time Taken (s)": f"{simple_time:.2f}"})
        yield IngestResult(
            status_text=f"SimpleRAG complete: {simple_time:.2f}s\n\n",
            progress_table=pd.DataFrame(progress_data),
        )
        
        # VisionRAG
        yield IngestResult(
            status_text="Starting VisionRAG ingestion...\n",
            progress_table=pd.DataFrame(progress_data),
        )
        start_time = time.time()
        vision_rag.index(file_paths, overwrite=False, recursive=False, verbose=True)
        vision_time = time.time() - start_time
        progress_data.append({"Technique": "VisionRAG", "Time Taken (s)": f"{vision_time:.2f}"})
        yield IngestResult(
            status_text=f"VisionRAG complete: {vision_time:.2f}s\n\n",
            progress_table=pd.DataFrame(progress_data),
        )
        
        # ColpaliRAG
        yield IngestResult(
            status_text="Starting ColpaliRAG ingestion...\n",
            progress_table=pd.DataFrame(progress_data),
        )
        start_time = time.time()
        colpali_rag.index(file_paths, overwrite=False, recursive=False, verbose=True)
        colpali_time = time.time() - start_time
        progress_data.append({"Technique": "ColpaliRAG", "Time Taken (s)": f"{colpali_time:.2f}"})
        yield IngestResult(
            status_text=f"ColpaliRAG complete: {colpali_time:.2f}s\n\n",
            progress_table=pd.DataFrame(progress_data),
        )
        
        # HybridColpaliRAG
        yield IngestResult(
            status_text="Starting HybridColpaliRAG ingestion...\n",
            progress_table=pd.DataFrame(progress_data),
        )
        start_time = time.time()
        hybrid_rag.index(file_paths, overwrite=False, recursive=False, verbose=True)
        hybrid_time = time.time() - start_time
        progress_data.append({"Technique": "HybridColpaliRAG", "Time Taken (s)": f"{hybrid_time:.2f}"})
        yield IngestResult(
            status_text=f"HybridColpaliRAG complete: {hybrid_time:.2f}s\n\n",
            progress_table=pd.DataFrame(progress_data),
        )
        
        total_time = time.time() - total_start_time
        progress_data.append({"Technique": "Total", "Time Taken (s)": f"{total_time:.2f}"})
        yield IngestResult(
            status_text=f"✅ All ingestion complete! Total time: {total_time:.2f}s",
            progress_table=pd.DataFrame(progress_data),
        )
    
    def retrieve_data(query, top_k, sequential=False):
        """Retrieve from all 4 RAG systems"""
        results = {}
        timings = {}
        
        def retrieve_simple():
            start_time = time.time()
            simple_results = simple_rag.search(query, k=top_k)
            simple_context = []
            for i, r in enumerate(simple_results, 1):
                context_piece = f"Result {i}:\n"
                context_piece += f"Source: {r.get('document_name', 'Unknown')}\n"
                context_piece += f"Chunk Index: {r.get('chunk_index', 'Unknown')}\n"
                context_piece += f"Content:\n{r['text']}\n"
                context_piece += "-" * 40 + "\n"
                simple_context.append(context_piece)
            simple_context = "\n".join(simple_context)
            end_time = time.time()
            return "SimpleRAG", simple_context, end_time - start_time
        
        def retrieve_vision():
            start_time = time.time()
            vision_results = vision_rag.search(query, k=top_k)
            vision_images = [r["image"] for r in vision_results]
            end_time = time.time()
            return "VisionRAG", vision_images, end_time - start_time
        
        def retrieve_colpali():
            start_time = time.time()
            colpali_results = colpali_rag.search(query, k=top_k)
            colpali_images = [r["image"] for r in colpali_results]
            end_time = time.time()
            return "ColpaliRAG", colpali_images, end_time - start_time
        
        def retrieve_hybrid():
            start_time = time.time()
            hybrid_results = hybrid_rag.search(query, k=top_k, use_image_search=True)
            hybrid_images = [r["image"] for r in hybrid_results]
            end_time = time.time()
            return "HybridColpaliRAG", hybrid_images, end_time - start_time
        
        retrieval_functions = [retrieve_simple, retrieve_vision, retrieve_colpali, retrieve_hybrid]
        
        if sequential:
            for func in retrieval_functions:
                rag_type, content, timing = func()
                results[rag_type] = content
                timings[rag_type] = timing
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_results = [executor.submit(func) for func in retrieval_functions]
                for future in concurrent.futures.as_completed(future_results):
                    rag_type, content, timing = future.result()
                    results[rag_type] = content
                    timings[rag_type] = timing
        
        return results, timings
    
    def query_data(query, retrieved_results):
        """Query all RAG systems with VLM"""
        results = {}
        
        # SimpleRAG
        simple_context = retrieved_results["SimpleRAG"]
        simple_response = llm.query(
            context=simple_context,
            system_prompt="Given the below information answer the questions",
            query=query,
        )
        results["SimpleRAG"] = {"response": simple_response, "context": simple_context}
        
        # VisionRAG
        vision_images = retrieved_results["VisionRAG"]
        vision_context = f"Query: {query}\n\nRelevant image information:\n" + "\n".join(
            [f"Image {i+1}" for i in range(len(vision_images))]
        )
        vision_response = vlm.query(vision_context, vision_images, max_tokens=500)
        results["VisionRAG"] = {
            "response": vision_response,
            "context": vision_context,
            "images": vision_images,
        }
        
        # ColpaliRAG
        colpali_images = retrieved_results["ColpaliRAG"]
        colpali_context = f"Query: {query}\n\nRelevant image information:\n" + "\n".join(
            [f"Image {i+1}" for i in range(len(colpali_images))]
        )
        colpali_response = vlm.query(colpali_context, colpali_images, max_tokens=500)
        results["ColpaliRAG"] = {
            "response": colpali_response,
            "context": colpali_context,
            "images": colpali_images,
        }
          # HybridColpaliRAG
        hybrid_images = retrieved_results["HybridColpaliRAG"]
        hybrid_context = f"Query: {query}\n\nRelevant image information:\n" + "\n".join(
            [f"Image {i+1}" for i in range(len(hybrid_images))]
        )
        hybrid_response = vlm.query(hybrid_context, hybrid_images, max_tokens=500)
        results["HybridColpaliRAG"] = {
            "response": hybrid_response,
            "context": hybrid_context,
            "images": hybrid_images,
        }
        
        return results
    
    def base64_to_pil(base64_str: str) -> Image.Image:
        """Convert base64 string to PIL Image"""
        return Image.open(io.BytesIO(base64.b64decode(base64_str)))
    
    def compare_colpali_models(query, colpali_images):
        """Compare both ColPali models side by side with detailed token analysis"""
        if not colpali_images or not query:
            return ("❌ No images or query available", 
                    gr.DataFrame(visible=False), gr.DataFrame(visible=False),
                    [[] for _ in range(10)], [[] for _ in range(10)], 
                    [gr.Row(visible=False) for _ in range(10)], 
                    [gr.Markdown(visible=False) for _ in range(10)], 
                    [gr.Markdown(visible=False) for _ in range(10)])
        
        print(f"🔍 Comparing base vs fine-tuned models for query: '{query}' on {len(colpali_images)} images")
        
        # Analyze with base model (unoptimized - process all images)
        print(f"🔄 Analyzing with base model...")
        base_analysis = analyze_multiple_images(base_similarity_mapper, colpali_images, query)
        print(f"✅ Base model analysis complete")
          # Analyze with fine-tuned model (unoptimized - process all images)
        print(f"🔄 Analyzing with fine-tuned model...")
        finetuned_analysis = analyze_multiple_images(finetuned_similarity_mapper, colpali_images, query)
        print(f"✅ Fine-tuned model analysis complete")
        
        # Prepare token analysis data
        base_token_analysis_data = []
        finetuned_token_analysis_data = []
        
        if base_analysis and len(base_analysis) > 0 and base_analysis[0]["success"]:
            base_result = base_analysis[0]
            for rank, token_data in enumerate(base_result.get("token_scores", []), 1):
                token = token_data["token"]
                sim = token_data["max_similarity"]
                base_token_analysis_data.append([token, f"{sim:.3f}", rank])
        
        if finetuned_analysis and len(finetuned_analysis) > 0 and finetuned_analysis[0]["success"]:
            finetuned_result = finetuned_analysis[0]
            for rank, token_data in enumerate(finetuned_result.get("token_scores", []), 1):
                token = token_data["token"]
                sim = token_data["max_similarity"]
                finetuned_token_analysis_data.append([token, f"{sim:.3f}", rank])
        
        # Create detailed comparison status
        status_text = f"""✅ **Model Comparison Complete for Query:** "{query}"

**📊 Results Summary:**
- **Base Model (vidore/colpali-v1.3):** {len(base_analysis)} images analyzed
- **Fine-tuned Model (akashmadisetty/colpali-merged-model-hi-10k):** {len(finetuned_analysis)} images analyzed

**💡 How to compare:** 
- Look at token similarity scores in the tables below
- Compare attention patterns in the side-by-side galleries
- Higher similarity scores indicate stronger model focus on that token

**🚀 UNOPTIMIZED VERSION:** Both models loaded simultaneously, all images processed, no memory constraints"""
        
        # Create token analysis DataFrames
        base_token_df = gr.DataFrame(
            value=base_token_analysis_data,
            headers=["Token", "Max Similarity", "Rank"],
            visible=True
        )
        
        finetuned_token_df = gr.DataFrame(
            value=finetuned_token_analysis_data,
            headers=["Token", "Max Similarity", "Rank"],
            visible=True
        )
        
        # Prepare outputs for galleries (unoptimized - process all images and visualizations)
        base_gallery_updates = []
        finetuned_gallery_updates = []
        row_updates = []
        base_page_info_updates = []
        finetuned_page_info_updates = []
        
        # Process up to 10 images for multi-page display (unoptimized version)
        max_images = min(len(base_analysis), len(finetuned_analysis), 10)
        
        for i in range(10):  # Support up to 10 galleries in UI
            if i < max_images:
                base_result = base_analysis[i]
                finetuned_result = finetuned_analysis[i]
                
                if base_result["success"] and finetuned_result["success"]:
                    # Convert base64 visualizations to images for base model (no size limits)
                    base_vis_images = []
                    for j, vis_b64 in enumerate(base_result["visualizations"]):  # All visualizations
                        try:
                            img_data = base64.b64decode(vis_b64)
                            img = Image.open(io.BytesIO(img_data))
                            # NO thumbnail resizing in unoptimized version
                            base_vis_images.append(img)
                        except Exception as e:
                            print(f"Error processing base model visualization {j}: {e}")
                            continue
                    
                    # Convert base64 visualizations to images for fine-tuned model (no size limits)
                    finetuned_vis_images = []
                    for j, vis_b64 in enumerate(finetuned_result["visualizations"]):  # All visualizations
                        try:
                            img_data = base64.b64decode(vis_b64)
                            img = Image.open(io.BytesIO(img_data))
                            # NO thumbnail resizing in unoptimized version
                            finetuned_vis_images.append(img)
                        except Exception as e:
                            print(f"Error processing fine-tuned model visualization {j}: {e}")
                            continue
                    
                    base_gallery_updates.append(base_vis_images)
                    finetuned_gallery_updates.append(finetuned_vis_images)
                    row_updates.append(gr.Row(visible=True))
                    
                    # Create detailed page info for base model (show all tokens, not just top 5)
                    base_token_breakdown = []
                    for rank, token_data in enumerate(base_result.get("token_scores", []), 1):
                        token = token_data["token"]
                        sim = token_data["max_similarity"]
                        base_token_breakdown.append(f"{rank}. **'{token}'** ({sim:.3f})")
                    
                    base_page_info_text = f"""**Page {i+1} - Base Model Tokens:**
{chr(10).join(base_token_breakdown[:10])}  
**Visualizations:** {len(base_vis_images)} (full resolution)"""
                    
                    # Create detailed page info for fine-tuned model (show all tokens, not just top 5)
                    finetuned_token_breakdown = []
                    for rank, token_data in enumerate(finetuned_result.get("token_scores", []), 1):
                        token = token_data["token"]
                        sim = token_data["max_similarity"]
                        finetuned_token_breakdown.append(f"{rank}. **'{token}'** ({sim:.3f})")
                    
                    finetuned_page_info_text = f"""**Page {i+1} - Fine-tuned Model Tokens:**
{chr(10).join(finetuned_token_breakdown[:10])}  
**Visualizations:** {len(finetuned_vis_images)} (full resolution)"""
                    
                    base_page_info_updates.append(gr.Markdown(value=base_page_info_text, visible=True))
                    finetuned_page_info_updates.append(gr.Markdown(value=finetuned_page_info_text, visible=True))
                else:
                    base_gallery_updates.append([])
                    finetuned_gallery_updates.append([])
                    row_updates.append(gr.Row(visible=False))
                    base_page_info_updates.append(gr.Markdown(visible=False))
                    finetuned_page_info_updates.append(gr.Markdown(visible=False))
            else:
                base_gallery_updates.append([])
                finetuned_gallery_updates.append([])
                row_updates.append(gr.Row(visible=False))
                base_page_info_updates.append(gr.Markdown(visible=False))
                finetuned_page_info_updates.append(gr.Markdown(visible=False))
        
        # Return all outputs in the expected order
        return (status_text, base_token_df, finetuned_token_df) + tuple(base_gallery_updates) + tuple(finetuned_gallery_updates) + tuple(row_updates) + tuple(base_page_info_updates) + tuple(finetuned_page_info_updates)
    
    # Create the Gradio interface
    with gr.Blocks(theme=gr.themes.Monochrome(radius_size=gr.themes.sizes.radius_none)) as demo:
        gr.Markdown("""
        # 👁️👁️ Vision RAG Playground - ORIGINAL UNOPTIMIZED VERSION
        
        ### Explore and Compare Vision-Augmented Retrieval Techniques
        Built on [VARAG](https://github.com/adithya-s-k/VARAG) - Vision-Augmented Retrieval and Generation
        
        **🔥 UNOPTIMIZED VERSION**: Both models loaded simultaneously, no memory constraints!
        
        1. **Simple RAG**: Text-based retrieval with OCR support for scanned documents.
        2. **Vision RAG**: Combines text and image retrieval using cross-modal embeddings.
        3. **ColPali RAG**: Embeds entire document pages as images for layout-aware retrieval.
        4. **Hybrid ColPali RAG**: Two-stage retrieval combining image embeddings and ColPali's token-level matching.
        """)
        
        with gr.Tab("Ingest Data"):
            pdf_input = gr.File(label="Upload PDF(s)", file_count="multiple", file_types=[".pdf"])
            use_ocr = gr.Checkbox(label="Use OCR (for SimpleRAG)")
            chunk_size = gr.Slider(50, 5000, value=300, step=10, label="Chunk Size (for SimpleRAG)")
            ingest_button = gr.Button("Ingest PDFs")
            ingest_output = gr.Markdown(label="Ingestion Status")
            progress_table = gr.DataFrame(label="Ingestion Progress", headers=["Technique", "Time Taken (s)"])
        
        with gr.Tab("Retrieve and Query Data"):
            query_input = gr.Textbox(label="Enter your query")
            top_k_slider = gr.Slider(1, 10, value=3, step=1, label="Top K Results")
            sequential_checkbox = gr.Checkbox(label="Sequential Retrieval", value=False)
            retrieve_button = gr.Button("Retrieve")
            query_button = gr.Button("Query")
            
            retrieval_timing = gr.DataFrame(label="Retrieval Timings", headers=["RAG Type", "Time (s)"])
            
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("SimpleRAG", open=True):
                        simple_content = gr.Textbox(label="SimpleRAG Content", lines=10, max_lines=10)
                        simple_response = gr.Markdown(label="SimpleRAG Response")
                with gr.Column():
                    with gr.Accordion("VisionRAG", open=True):
                        vision_gallery = gr.Gallery(label="VisionRAG Images")
                        vision_response = gr.Markdown(label="VisionRAG Response")
            
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("ColpaliRAG", open=True):
                        colpali_gallery = gr.Gallery(label="ColpaliRAG Images")
                        colpali_response = gr.Markdown(label="ColpaliRAG Response")
                with gr.Column():
                    with gr.Accordion("HybridColpaliRAG", open=True):
                        hybrid_gallery = gr.Gallery(label="HybridColpaliRAG Images")
                        hybrid_response = gr.Markdown(label="HybridColpaliRAG Response")
        
        with gr.Tab("ColPali Model Comparison"):
            gr.Markdown("""
            ## � ColPali Model Comparison - Base vs Fine-tuned
            
            **Left Half**: Base Model (`vidore/colpali-v1.3`)  
            **Right Half**: Fine-tuned Model (`akashmadisetty/colpali-merged-model-hi-10k`)
              First perform a retrieval in the "Retrieve and Query Data" tab, then come here to compare models.
            """)
            
            with gr.Row():
                current_query_display = gr.Textbox(
                    label="Current Query", 
                    value="No query yet - perform a retrieval first", 
                    interactive=False,
                    lines=2
                )
                compare_button = gr.Button("🎯 Compare ColPali Models", variant="primary", size="lg")
            
            comparison_status = gr.Markdown("Ready for model comparison...")
            
            with gr.Row():
                retrieved_images_gallery = gr.Gallery(
                    label="Retrieved Images (ColPali)",
                    columns=3,
                    rows=2,
                    height="400px"
                )
            
            # Token Analysis Results for both models
            with gr.Row():
                with gr.Column():
                    base_token_analysis = gr.DataFrame(
                        label="Base Model Token Analysis (vidore/colpali-v1.3)",
                        headers=["Token", "Max Similarity", "Rank"],
                        visible=False
                    )
                with gr.Column():
                    finetuned_token_analysis = gr.DataFrame(
                        label="Fine-tuned Model Token Analysis (akashmadisetty/colpali-merged-model-hi-10k)",
                        headers=["Token", "Max Similarity", "Rank"],
                        visible=False
                    )
            
            # Dynamic galleries for each retrieved image - side by side comparison
            comparison_galleries = []
            for i in range(10):  # Support up to 10 retrieved images
                with gr.Row(visible=False) as comparison_row:
                    with gr.Column():
                        base_page_info = gr.Markdown(f"### 📄 Page {i+1} - Base Model (vidore/colpali-v1.3)", visible=True)
                        base_similarity_gallery = gr.Gallery(
                            label=f"Base Model Token Similarity Maps",
                            show_label=True,
                            columns=3,
                            rows=2,
                            height="500px"
                        )
                    with gr.Column():
                        finetuned_page_info = gr.Markdown(f"### 📄 Page {i+1} - Fine-tuned Model (akashmadisetty/colpali-merged-model-hi-10k)", visible=True)
                        finetuned_similarity_gallery = gr.Gallery(
                            label=f"Fine-tuned Model Token Similarity Maps",
                            show_label=True,
                            columns=3,
                            rows=2,
                            height="500px"
                        )
                comparison_galleries.append((comparison_row, base_similarity_gallery, finetuned_similarity_gallery, base_page_info, finetuned_page_info))
            
            with gr.Row():
                interpretation_info = gr.Markdown("""
                **How to use:**
                1. Go to "Retrieve and Query Data" tab
                2. Enter a query and click "Retrieve"
                3. Come back here and click "🎯 Compare ColPali Models"
                4. Analyze token scores and similarity maps for both models
                  **Understanding the Results:**
                - **Token Analysis Tables**: Show how much each word/token contributes to the query match
                - **Similarity Maps**: Visual heatmaps showing where the model focuses for each token
                - **Multiple Pages**: If ColPali retrieved multiple pages, each will have its own comparison section
                """)
        
        with gr.Tab("Settings"):
            api_key_input = gr.Textbox(label="OpenAI API Key", type="password")
            update_api_button = gr.Button("Update API Key")
            api_update_status = gr.Textbox(label="API Update Status")
            
            simple_table_input = gr.Textbox(label="SimpleRAG Table Name", value="originalSimpleDemo")
            vision_table_input = gr.Textbox(label="VisionRAG Table Name", value="originalVisionDemo")
            colpali_table_input = gr.Textbox(label="ColpaliRAG Table Name", value="originalColpaliDemo")
            hybrid_table_input = gr.Textbox(label="HybridColpaliRAG Table Name", value="originalHybridDemo")
            update_table_button = gr.Button("Update Table Names")
            table_update_status = gr.Textbox(label="Table Update Status")
        
        # State variables
        retrieved_results = gr.State({})
        current_query = gr.State("")
        
        def update_retrieval_results(query, top_k, sequential):
            results, timings = retrieve_data(query, top_k, sequential)
            timing_df = pd.DataFrame(list(timings.items()), columns=["RAG Type", "Time (s)"])
            return (
                results["SimpleRAG"],
                results["VisionRAG"], 
                results["ColpaliRAG"],
                results["HybridColpaliRAG"],
                timing_df,
                results,
                query
            )
        
        def update_query_results(query, retrieved_results):
            if not retrieved_results:
                return "❌ No retrieval results", "❌ No retrieval results", "❌ No retrieval results", "❌ No retrieval results"
            
            results = query_data(query, retrieved_results)
            return (
                results["SimpleRAG"]["response"],
                results["VisionRAG"]["response"],
                results["ColpaliRAG"]["response"],
                results["HybridColpaliRAG"]["response"],
            )
        
        def update_api_key(api_key):
            os.environ["OPENAI_API_KEY"] = api_key
            return "✅ API key updated successfully"
        
        def change_table(simple_table, vision_table, colpali_table, hybrid_table):
            simple_rag.change_table(simple_table)
            vision_rag.change_table(vision_table)
            colpali_rag.change_table(colpali_table)
            hybrid_rag.change_table(hybrid_table)
            return "✅ Table names updated successfully"
        
        def handle_model_comparison(retrieved_results, current_query):
            if not retrieved_results or "ColpaliRAG" not in retrieved_results:
                # Return error for all 55 expected outputs
                empty_galleries = [[] for _ in range(10)]
                empty_rows = [gr.Row(visible=False) for _ in range(10)]
                empty_markdowns = [gr.Markdown(visible=False) for _ in range(10)]
                return (
                    "❌ No ColPali results available",  # comparison_status
                    [],  # retrieved_images_gallery
                    gr.DataFrame(visible=False),  # base_token_analysis
                    gr.DataFrame(visible=False),  # finetuned_token_analysis  
                    "No query available"  # current_query_display
                ) + tuple(empty_galleries) + tuple(empty_galleries) + tuple(empty_rows) + tuple(empty_markdowns) + tuple(empty_markdowns)
            
            if not current_query:
                # Return error for all 55 expected outputs
                empty_galleries = [[] for _ in range(10)]
                empty_rows = [gr.Row(visible=False) for _ in range(10)]
                empty_markdowns = [gr.Markdown(visible=False) for _ in range(10)]
                return (
                    "❌ No query available",  # comparison_status
                    [],  # retrieved_images_gallery
                    gr.DataFrame(visible=False),  # base_token_analysis
                    gr.DataFrame(visible=False),  # finetuned_token_analysis
                    "No query available"  # current_query_display
                ) + tuple(empty_galleries) + tuple(empty_galleries) + tuple(empty_rows) + tuple(empty_markdowns) + tuple(empty_markdowns)
            
            # Get ColPali images - they are already PIL Image objects from the RAG system
            colpali_images = retrieved_results["ColpaliRAG"]
            
            # Ensure we have valid PIL images
            valid_images = []
            for img in colpali_images:
                if isinstance(img, Image.Image):
                    valid_images.append(img)
                elif isinstance(img, str):
                    # If it's a base64 string, convert it
                    try:
                        valid_images.append(base64_to_pil(img))
                    except Exception as e:
                        print(f"Error converting base64 image: {e}")
                        continue
                else:
                    print(f"Unexpected image type: {type(img)}")
                    continue
            
            # Get comparison results from compare_colpali_models
            comparison_results = compare_colpali_models(current_query, valid_images)
            
            # Extract the results (compare_colpali_models returns 53 items)
            status_text = comparison_results[0]
            base_token_df = comparison_results[1] 
            finetuned_token_df = comparison_results[2]
            # Items 3-12: base galleries, 13-22: finetuned galleries, 23-32: rows, 33-42: base infos, 43-52: finetuned infos
            
            # Now return all 55 expected outputs
            return (
                status_text,  # comparison_status
                valid_images,  # retrieved_images_gallery (this was missing!)
                base_token_df,  # base_token_analysis
                finetuned_token_df,  # finetuned_token_analysis
                current_query  # current_query_display
            ) + comparison_results[3:]  # All the remaining 50 outputs from compare_colpali_models
        
        # Event handlers
        ingest_button.click(
            ingest_data,
            inputs=[pdf_input, use_ocr, chunk_size],
            outputs=[ingest_output, progress_table]
        )
        
        retrieve_button.click(
            update_retrieval_results,
            inputs=[query_input, top_k_slider, sequential_checkbox],
            outputs=[
                simple_content, vision_gallery, colpali_gallery, hybrid_gallery,
                retrieval_timing, retrieved_results, current_query
            ]
        )
        
        query_button.click(
            update_query_results,
            inputs=[query_input, retrieved_results],
            outputs=[simple_response, vision_response, colpali_response, hybrid_response]        )
        
        compare_button.click(
            handle_model_comparison,
            inputs=[retrieved_results, current_query],
            outputs=[
                comparison_status, retrieved_images_gallery, 
                base_token_analysis, finetuned_token_analysis, current_query_display
            ] + [gallery for _, gallery, _, _, _ in comparison_galleries] + 
              [gallery for _, _, gallery, _, _ in comparison_galleries] + 
              [row for row, _, _, _, _ in comparison_galleries] + 
              [info for _, _, _, info, _ in comparison_galleries] + 
              [info for _, _, _, _, info in comparison_galleries]
        )
        
        update_api_button.click(
            update_api_key,
            inputs=[api_key_input],
            outputs=[api_update_status]
        )
        
        update_table_button.click(
            change_table,
            inputs=[simple_table_input, vision_table_input, colpali_table_input, hybrid_table_input],
            outputs=[table_update_status]
        )
    
    print("🎉 Original unoptimized interface ready!")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
    
    return {"status": "comparision_demo_original app launched successfully"}

