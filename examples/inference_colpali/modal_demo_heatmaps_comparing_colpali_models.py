import modal
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import os
import base64
import io

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
        "matplotlib",
        "scipy",
    )
    .run_commands(["ls -al",])
    .env({
        "HF_HUB_CACHE": HF_CACHE_PATH, 
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "GEMINI_API_KEY":"INSERT_API_KEY_HERE",
    })

)
@app.function(
    image=inference_image,
    gpu="L4",  # Use powerful GPU for unoptimized version
    timeout=7200,  # 2 hour timeout
    volumes={
        VOLUME_PATH: col_vol,
    },
    secrets=[modal.Secret.from_name("hf-wandb-vyoman-secrets")]  # For HF token
)
def comparision_demo():
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
    import matplotlib.pyplot as plt
    import numpy as np
    
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
    

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    print(f"🚀 Starting ColPali Comparison App")
    print(f"💾 Cache: {HF_CACHE_PATH}")
    print(f"🔥 Loading both models")
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
    
    
    if gemini_api_key:
        gemini_model = "gemini/gemini-2.5-flash"
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
    
    def query_data_single(query, retrieved_results, model_name):
        """Helper function to query a single RAG model with error handling"""
        try:
            if model_name == "SimpleRAG":
                print(f"🔄 Querying {model_name}...")
                simple_context = retrieved_results["SimpleRAG"]
                simple_response = llm.query(
                    context=simple_context,
                    system_prompt="Given the below information answer the questions",
                    query=query,
                )
                print(f"✅ {model_name} completed")
                return {"response": simple_response, "context": simple_context}
            
            elif model_name == "VisionRAG":
                print(f"🔄 Querying {model_name}...")
                vision_images = retrieved_results["VisionRAG"]
                vision_context = f"Query: {query}\n\nRelevant image information:\n" + "\n".join(
                    [f"Image {i+1}" for i in range(len(vision_images))]
                )
                vision_response = vlm.query(vision_context, vision_images, max_tokens=500)
                print(f"✅ {model_name} completed")
                return {
                    "response": vision_response,
                    "context": vision_context,
                    "images": vision_images,
                }
            
            elif model_name == "ColpaliRAG":
                print(f"🔄 Querying {model_name}...")
                colpali_images = retrieved_results["ColpaliRAG"]
                colpali_context = f"Query: {query}\n\nRelevant image information:\n" + "\n".join(
                    [f"Image {i+1}" for i in range(len(colpali_images))]
                )
                colpali_response = vlm.query(colpali_context, colpali_images, max_tokens=500)
                print(f"✅ {model_name} completed")
                return {
                    "response": colpali_response,
                    "context": colpali_context,
                    "images": colpali_images,
                }
            
            elif model_name == "HybridColpaliRAG":
                print(f"🔄 Querying {model_name}...")
                hybrid_images = retrieved_results["HybridColpaliRAG"]
                hybrid_context = f"Query: {query}\n\nRelevant image information:\n" + "\n".join(
                    [f"Image {i+1}" for i in range(len(hybrid_images))]
                )
                hybrid_response = vlm.query(hybrid_context, hybrid_images, max_tokens=500)
                print(f"✅ {model_name} completed")
                return {
                    "response": hybrid_response,
                    "context": hybrid_context,
                    "images": hybrid_images,
                }
            
            return {"response": f"❌ Model {model_name} not recognized", "context": ""}
            
        except Exception as e:
            error_msg = f"❌ Error in {model_name}: {str(e)}"
            print(error_msg)
            return {"response": error_msg, "context": ""}
    
    def query_data(query, retrieved_results):
        """Query all RAG systems with rate limiting and progressive updates"""
        if not retrieved_results:
            return "❌ No retrieval results", "❌ No retrieval results", "❌ No retrieval results", "❌ No retrieval results"
        
        # Initialize empty responses
        responses = {
            "SimpleRAG": {"response": "🔄 Waiting to process..."},
            "VisionRAG": {"response": "🔄 Waiting to process..."},
            "ColpaliRAG": {"response": "🔄 Waiting to process..."},
            "HybridColpaliRAG": {"response": "🔄 Waiting to process..."}
        }
        
        # Process SimpleRAG first
        responses["SimpleRAG"]["response"] = "🔄 Processing SimpleRAG..."
        yield (
            responses["SimpleRAG"]["response"],
            responses["VisionRAG"]["response"],
            responses["ColpaliRAG"]["response"],
            responses["HybridColpaliRAG"]["response"],
        )
        
        responses["SimpleRAG"] = query_data_single(query, retrieved_results, "SimpleRAG")
        yield (
            responses["SimpleRAG"]["response"],
            responses["VisionRAG"]["response"],
            responses["ColpaliRAG"]["response"],
            responses["HybridColpaliRAG"]["response"],
        )
        
        # Wait to avoid rate limit
        print("⏱️ Waiting 30 seconds before VisionRAG...")
        time.sleep(30)
        
        # Process VisionRAG
        responses["VisionRAG"]["response"] = "🔄 Processing VisionRAG..."
        yield (
            responses["SimpleRAG"]["response"],
            responses["VisionRAG"]["response"],
            responses["ColpaliRAG"]["response"],
            responses["HybridColpaliRAG"]["response"],
        )
        
        responses["VisionRAG"] = query_data_single(query, retrieved_results, "VisionRAG")
        yield (
            responses["SimpleRAG"]["response"],
            responses["VisionRAG"]["response"],
            responses["ColpaliRAG"]["response"],
            responses["HybridColpaliRAG"]["response"],
        )
        
        # Wait to avoid rate limit
        print("⏱️ Waiting 30 seconds before ColpaliRAG...")
        time.sleep(30)
        
        # Process ColpaliRAG
        responses["ColpaliRAG"]["response"] = "🔄 Processing ColpaliRAG..."
        yield (
            responses["SimpleRAG"]["response"],
            responses["VisionRAG"]["response"],
            responses["ColpaliRAG"]["response"],
            responses["HybridColpaliRAG"]["response"],
        )
        
        responses["ColpaliRAG"] = query_data_single(query, retrieved_results, "ColpaliRAG")
        yield (
            responses["SimpleRAG"]["response"],
            responses["VisionRAG"]["response"],
            responses["ColpaliRAG"]["response"],
            responses["HybridColpaliRAG"]["response"],
        )
        
        # Wait longer before final processing
        print("⏱️ Waiting 60 seconds before HybridColpaliRAG...")
        time.sleep(60)
        
        # Process HybridColpaliRAG
        responses["HybridColpaliRAG"]["response"] = "🔄 Processing HybridColpaliRAG..."
        yield (
            responses["SimpleRAG"]["response"],
            responses["VisionRAG"]["response"],
            responses["ColpaliRAG"]["response"],
            responses["HybridColpaliRAG"]["response"],
        )
        
        responses["HybridColpaliRAG"] = query_data_single(query, retrieved_results, "HybridColpaliRAG")
        yield (
            responses["SimpleRAG"]["response"],
            responses["VisionRAG"]["response"],
            responses["ColpaliRAG"]["response"],
            responses["HybridColpaliRAG"]["response"],
        )
        
        print("✅ All RAG queries completed successfully!")

    def base64_to_pil(base64_str: str) -> Image.Image:
        """Convert base64 string to PIL Image"""
        return Image.open(io.BytesIO(base64.b64decode(base64_str)))
    
    class AggregatedHeatmapGenerator:
        """
        Generates aggregated heatmaps by averaging similarity scores across all query tokens.
        """
        
        def __init__(self, model, processor, device=None):
            from colpali_engine.utils.torch_utils import get_torch_device
            self.model = model
            self.processor = processor
            self.device = device or get_torch_device("auto")
            
        def generate_aggregated_similarity_map(self, image: Image.Image, query: str):
            """Generate aggregated similarity map by averaging across all query tokens."""
            try:
                from colpali_engine.interpretability import get_similarity_maps_from_embeddings
                from colpali_engine.utils.torch_utils import get_torch_device
                
                # Preprocess inputs
                batch_images = self.processor.process_images([image]).to(self.device)
                batch_queries = self.processor.process_queries([query]).to(self.device)
                
                # Forward passes
                with torch.no_grad():
                    image_embeddings = self.model.forward(**batch_images)
                    query_embeddings = self.model.forward(**batch_queries)
                
                # Convert embeddings to float32 to avoid BFloat16 issues
                if image_embeddings.dtype == torch.bfloat16:
                    image_embeddings = image_embeddings.float()
                if query_embeddings.dtype == torch.bfloat16:
                    query_embeddings = query_embeddings.float()
                
                # Get image patch information
                n_patches = self.processor.get_n_patches(image_size=image.size, patch_size=self.model.patch_size)
                image_mask = self.processor.get_image_mask(batch_images)
                
                # Generate similarity maps for all tokens
                batched_similarity_maps = get_similarity_maps_from_embeddings(
                    image_embeddings=image_embeddings,
                    query_embeddings=query_embeddings,
                    n_patches=n_patches,
                    image_mask=image_mask,
                )
                
                # Get similarity map for our input image: (query_length, n_patches_x, n_patches_y)
                similarity_maps = batched_similarity_maps[0]
                
                # Ensure similarity maps are in float32
                if similarity_maps.dtype == torch.bfloat16:
                    similarity_maps = similarity_maps.float()
                
                # Get query tokens for reference
                query_tokens = self.processor.tokenizer.tokenize(query)
                
                # Aggregate across all tokens (average)
                aggregated_map = torch.mean(similarity_maps, dim=0)
                
                # Convert to numpy with explicit float32
                aggregated_map_np = aggregated_map.cpu().numpy().astype('float32')
                
                metadata = {
                    "query": query,
                    "query_tokens": query_tokens,
                    "num_tokens": len(query_tokens),
                    "min_score": float(aggregated_map_np.min()),
                    "max_score": float(aggregated_map_np.max()),
                    "mean_score": float(aggregated_map_np.mean()),
                }
                
                return aggregated_map_np, metadata
                
            except Exception as e:
                print(f"❌ Error generating aggregated similarity map: {e}")
                return None, {"error": str(e)}
    
        def create_aggregated_heatmap_visualization(self, image: Image.Image, aggregated_map, metadata):
            """Create visualization of aggregated heatmap."""
            try:
                import matplotlib.pyplot as plt
                from scipy.ndimage import zoom
                
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                ax1.imshow(image)
                ax1.set_title("Original Image", fontweight='bold')
                ax1.axis('off')
                
                # Pure heatmap
                im2 = ax2.imshow(aggregated_map, cmap="Blues", interpolation='bilinear')
                ax2.set_title(f"Aggregated Heatmap\n({metadata['num_tokens']} tokens)", fontweight='bold')
                ax2.axis('off')
                plt.colorbar(im2, ax=ax2, shrink=0.8)
                
                # Overlay
                ax3.imshow(image)
                scale_x = image.size[0] / aggregated_map.shape[1]
                scale_y = image.size[1] / aggregated_map.shape[0]
                heatmap_resized = zoom(aggregated_map, (scale_y, scale_x), order=1)
                ax3.imshow(heatmap_resized, cmap="Blues", alpha=0.6, 
                          extent=[0, image.size[0], image.size[1], 0])
                ax3.set_title("Overlay", fontweight='bold')
                ax3.axis('off')
                
                plt.tight_layout()
                
                # Convert to PIL Image instead of base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                result_image = Image.open(buffer).copy()  # Make a copy to avoid buffer issues
                plt.close()
                
                return result_image
                
            except Exception as e:
                print(f"❌ Error creating visualization: {e}")
                return None

    def create_aggregated_comparison_heatmaps(base_generator, finetuned_generator, image, query):
        """Create side-by-side comparison of aggregated heatmaps."""
        try:
            import matplotlib.pyplot as plt
            from scipy.ndimage import zoom
            
            # Generate heatmaps for both models
            base_map, base_meta = base_generator.generate_aggregated_similarity_map(image, query)
            finetuned_map, finetuned_meta = finetuned_generator.generate_aggregated_similarity_map(image, query)
            
            if base_map is None or finetuned_map is None:
                print("❌ Failed to generate aggregated heatmaps")
                return None
            
            # Create comparison visualization
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Row 1: Base Model
            axes[0,0].imshow(image)
            axes[0,0].set_title("Original Image", fontweight='bold')
            axes[0,0].axis('off')
            
            im1 = axes[0,1].imshow(base_map, cmap="Blues", interpolation='bilinear')
            axes[0,1].set_title(f"Base Model Heatmap\nMean: {base_meta['mean_score']:.4f}", fontweight='bold')
            axes[0,1].axis('off')
            plt.colorbar(im1, ax=axes[0,1], shrink=0.8)
            
            axes[0,2].imshow(image)
            scale_x = image.size[0] / base_map.shape[1]
            scale_y = image.size[1] / base_map.shape[0]
            base_resized = zoom(base_map, (scale_y, scale_x), order=1)
            axes[0,2].imshow(base_resized, cmap="Blues", alpha=0.6, 
                            extent=[0, image.size[0], image.size[1], 0])
            axes[0,2].set_title("Base Model Overlay", fontweight='bold')
            axes[0,2].axis('off')
            
            # Row 2: Fine-tuned Model
            axes[1,0].imshow(image)
            axes[1,0].set_title("Original Image", fontweight='bold')
            axes[1,0].axis('off')
            
            im2 = axes[1,1].imshow(finetuned_map, cmap="Blues", interpolation='bilinear')
            axes[1,1].set_title(f"Fine-tuned Model Heatmap\nMean: {finetuned_meta['mean_score']:.4f}", fontweight='bold')
            axes[1,1].axis('off')
            plt.colorbar(im2, ax=axes[1,1], shrink=0.8)
            
            axes[1,2].imshow(image)
            finetuned_resized = zoom(finetuned_map, (scale_y, scale_x), order=1)
            axes[1,2].imshow(finetuned_resized, cmap="Blues", alpha=0.6,
                            extent=[0, image.size[0], image.size[1], 0])
            axes[1,2].set_title("Fine-tuned Model Overlay", fontweight='bold')
            axes[1,2].axis('off')
            
            plt.suptitle(f'ColPali Model Comparison: "{query}"', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Convert to PIL Image
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            result_image = Image.open(buffer).copy()
            plt.close()
            
            return result_image, base_meta, finetuned_meta
            
        except Exception as e:
            print(f"❌ Error creating aggregated comparison: {e}")
            return None, None, None

    def compare_colpali_models(query, colpali_images):
        """Compare both ColPali models with BOTH aggregated heatmaps AND individual token analysis"""
        if not colpali_images or not query:
            empty_galleries = [[] for _ in range(20)]  # 10 base + 10 finetuned
            empty_rows = [gr.Row(visible=False) for _ in range(10)]
            empty_markdowns = [gr.Markdown(visible=False) for _ in range(20)]  # 10 base + 10 finetuned
            return ("❌ No images or query available", 
                    gr.DataFrame(visible=False), gr.DataFrame(visible=False),
                    []) + tuple(empty_galleries) + tuple(empty_rows) + tuple(empty_markdowns)
        
        print(f"🔍 Comparing base vs fine-tuned models for query: '{query}' on {len(colpali_images)} images")
        
        try:
            # Initialize aggregated heatmap generators
            base_generator = AggregatedHeatmapGenerator(base_colpali_model, base_colpali_processor)
            finetuned_generator = AggregatedHeatmapGenerator(finetuned_colpali_model, finetuned_colpali_processor)
            
            # Generate aggregated comparison heatmaps for each image
            aggregated_comparison_images = []
            for i, image in enumerate(colpali_images[:3]):  # Limit to first 3 images for aggregated view
                print(f"🔄 Generating aggregated heatmaps for image {i+1}...")
                comparison_result, base_meta, finetuned_meta = create_aggregated_comparison_heatmaps(
                    base_generator, finetuned_generator, image, query
                )
                if comparison_result:
                    aggregated_comparison_images.append(comparison_result)
                    print(f"✅ Aggregated heatmap {i+1} complete")
            
            # Analyze with base model for individual tokens
            print(f"🔄 Analyzing with base model for individual tokens...")
            base_analysis = analyze_multiple_images(base_similarity_mapper, colpali_images, query)
            print(f"✅ Base model analysis complete: {len(base_analysis) if base_analysis else 0} results")
            
            # Analyze with fine-tuned model for individual tokens
            print(f"🔄 Analyzing with fine-tuned model for individual tokens...")
            finetuned_analysis = analyze_multiple_images(finetuned_similarity_mapper, colpali_images, query)
            print(f"✅ Fine-tuned model analysis complete: {len(finetuned_analysis) if finetuned_analysis else 0} results")
            
        except Exception as e:
            error_msg = f"❌ Error during model analysis: {str(e)}"
            print(error_msg)
            empty_galleries = [[] for _ in range(20)]  # 10 base + 10 finetuned
            empty_rows = [gr.Row(visible=False) for _ in range(10)]
            empty_markdowns = [gr.Markdown(visible=False) for _ in range(20)]  # 10 base + 10 finetuned
            return (error_msg,
                    gr.DataFrame(visible=False), gr.DataFrame(visible=False),
                    []) + tuple(empty_galleries) + tuple(empty_rows) + tuple(empty_markdowns)
        
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
        status_text = f"""✅ **Complete Model Comparison for Query:** "{query}"

**📊 Results Summary:**
- **Base Model (vidore/colpali-v1.3):** {len(base_analysis)} images analyzed
- **Fine-tuned Model (akashmadisetty/colpali-merged-model-hi-10k):** {len(finetuned_analysis)} images analyzed
- **Aggregated Heatmaps Generated:** {len(aggregated_comparison_images)} comparison images

**🎯 Two Analysis Types Available:**

1. **Aggregated Similarity Maps**: 
   - Overall attention patterns averaged across all query tokens
   - Shows general focus areas for each model
   - Useful for understanding global attention differences

2. **Individual Token Analysis**:
   - Per-token similarity heatmaps for detailed inspection
   - Token-wise similarity scores and rankings
   - Allows precise comparison of how each model processes specific words

**💡 How to compare:** 
- **Aggregated view**: Look at overall attention patterns in the comparison images above
- **Token tables**: Compare similarity scores between models for each word
- **Individual galleries**: Examine per-token heatmaps for detailed analysis
- Higher similarity scores indicate stronger model focus on that token"""
        
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
        
        # Prepare outputs for galleries
        base_gallery_updates = []
        finetuned_gallery_updates = []
        row_updates = []
        base_page_info_updates = []
        finetuned_page_info_updates = []
        
        # Process up to 10 images for multi-page display
        max_images = min(len(base_analysis), len(finetuned_analysis), 10)
        
        for i in range(10):  # Support up to 10 galleries in UI
            if i < max_images:
                base_result = base_analysis[i]
                finetuned_result = finetuned_analysis[i]
                
                if base_result["success"] and finetuned_result["success"]:
                    # Convert base64 visualizations to images for base model
                    base_vis_images = []
                    for j, vis_b64 in enumerate(base_result["visualizations"]):
                        try:
                            img_data = base64.b64decode(vis_b64)
                            img = Image.open(io.BytesIO(img_data))
                            base_vis_images.append(img)
                        except Exception as e:
                            print(f"Error processing base model visualization {j}: {e}")
                            continue
                    
                    # Convert base64 visualizations to images for fine-tuned model
                    finetuned_vis_images = []
                    for j, vis_b64 in enumerate(finetuned_result["visualizations"]):
                        try:
                            img_data = base64.b64decode(vis_b64)
                            img = Image.open(io.BytesIO(img_data))
                            finetuned_vis_images.append(img)
                        except Exception as e:
                            print(f"Error processing fine-tuned model visualization {j}: {e}")
                            continue
                    
                    base_gallery_updates.append(base_vis_images)
                    finetuned_gallery_updates.append(finetuned_vis_images)
                    row_updates.append(gr.Row(visible=True))
                    
                    # Create detailed page info for base model
                    base_token_breakdown = []
                    for rank, token_data in enumerate(base_result.get("token_scores", []), 1):
                        token = token_data["token"]
                        sim = token_data["max_similarity"]
                        base_token_breakdown.append(f"{rank}. **'{token}'** ({sim:.3f})")
                    
                    base_page_info_text = f"""**Page {i+1} - Base Model Tokens:**
{chr(10).join(base_token_breakdown[:10])}  
**Visualizations:** {len(base_vis_images)}"""
                    
                    # Create detailed page info for fine-tuned model
                    finetuned_token_breakdown = []
                    for rank, token_data in enumerate(finetuned_result.get("token_scores", []), 1):
                        token = token_data["token"]
                        sim = token_data["max_similarity"]
                        finetuned_token_breakdown.append(f"{rank}. **'{token}'** ({sim:.3f})")
                    
                    finetuned_page_info_text = f"""**Page {i+1} - Fine-tuned Model Tokens:**
{chr(10).join(finetuned_token_breakdown[:10])}  
**Visualizations:** {len(finetuned_vis_images)}"""
                    
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
        # Format: (status, base_token_df, finetuned_token_df, aggregated_gallery, base_galleries, finetuned_galleries, rows, base_infos, finetuned_infos)
        return (status_text, base_token_df, finetuned_token_df, aggregated_comparison_images) + tuple(base_gallery_updates) + tuple(finetuned_gallery_updates) + tuple(row_updates) + tuple(base_page_info_updates) + tuple(finetuned_page_info_updates)
    
    # Clean up - we're using the backup approach so these functions are not needed
    # The backup approach uses analyze_multiple_images directly instead of the aggregated approach

    # ...existing code for other functions...

    # Create the Gradio interface
    with gr.Blocks(theme=gr.themes.Monochrome(radius_size=gr.themes.sizes.radius_none)) as demo:
        gr.Markdown("""
        # 👁️👁️ Vision RAG Playground
        
        ### Explore and Compare Vision-Augmented Retrieval Techniques
        Built on [VARAG](https://github.com/adithya-s-k/VARAG) - Vision-Augmented Retrieval and Generation
        
        
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
            
            # Add progress indicator for query processing
            gr.Markdown("### ⚡ Query Processing Status")
            query_button = gr.Button("🚀 Query All RAG Systems", variant="primary")
            gr.Markdown("**Note**: This will take ~2 minutes due to rate limiting (30s + 30s + 60s delays)")
            
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
            ## 🔍 ColPali Model Comparison - Base vs Fine-tuned
            
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
            
            # Aggregated Similarity Maps Comparison
            with gr.Row():
                aggregated_comparison_gallery = gr.Gallery(
                    label="🔥 Aggregated Similarity Maps Comparison (Base vs Fine-tuned)",
                    show_label=True,
                    columns=2,
                    rows=2,
                    height="600px"
                )
            
            gr.Markdown("""
            ---
            ## 📊 Individual Token Analysis
            
            Below you'll find detailed token-by-token analysis showing how each model processes individual words in your query.
            """)
            
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
                **How to use this comprehensive comparison:**
                
                1. **Step 1**: Go to "Retrieve and Query Data" tab
                2. **Step 2**: Enter a query and click "Retrieve"  
                3. **Step 3**: Come back here and click "🎯 Compare ColPali Models"
                4. **Step 4**: Analyze both types of results:

                **📊 Understanding the Two Analysis Types:**
                
                **🔥 Aggregated Similarity Maps** (Top section):
                - Shows overall attention patterns averaged across all query tokens
                - Side-by-side comparison images of base vs fine-tuned models
                - Great for understanding general differences in model behavior
                - Each row shows: Original → Pure Heatmap → Overlay for each model
                
                **📝 Individual Token Analysis** (Bottom section):
                - **Token Tables**: Detailed similarity scores for each word in your query
                - **Per-Page Galleries**: Token-by-token heatmaps for precise analysis  
                - **Multiple Pages**: If ColPali retrieved multiple pages, each gets its own section
                - Compare how each model processes individual words vs overall patterns
                
                **💡 Analysis Tips:**
                - Look for differences in attention strength (color intensity)
                - Compare token rankings between models in the tables
                - Check if fine-tuned model focuses on more relevant areas
                - Use aggregated view for overview, individual tokens for details
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
                # Return empty/error state with correct number of outputs (6 + 50 = 56 total)
                empty_galleries = [[] for _ in range(20)]  # 10 base + 10 finetuned
                empty_rows = [gr.Row(visible=False) for _ in range(10)]
                empty_markdowns = [gr.Markdown(visible=False) for _ in range(20)]  # 10 base + 10 finetuned
                return ("❌ No ColPali results available", [], gr.DataFrame(visible=False), gr.DataFrame(visible=False), 
                        [], "No query available") + tuple(empty_galleries) + tuple(empty_rows) + tuple(empty_markdowns)
            
            if not current_query:
                # Return empty/error state with correct number of outputs (6 + 50 = 56 total)
                empty_galleries = [[] for _ in range(20)]  # 10 base + 10 finetuned
                empty_rows = [gr.Row(visible=False) for _ in range(10)]
                empty_markdowns = [gr.Markdown(visible=False) for _ in range(20)]  # 10 base + 10 finetuned
                return ("❌ No query available", [], gr.DataFrame(visible=False), gr.DataFrame(visible=False), 
                        [], "No query available") + tuple(empty_galleries) + tuple(empty_rows) + tuple(empty_markdowns)
            
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
            
            # Get comparison results from compare_colpali_models (returns 54 items: status, base_df, finetuned_df, aggregated_gallery + 50 comparison items)
            comparison_results = compare_colpali_models(current_query, valid_images)
            
            # Convert images to gallery format for display
            retrieved_images_for_display = []
            for img in valid_images[:5]:  # Show top 5 images
                retrieved_images_for_display.append(img)  # Gradio can handle PIL Images directly
            
            # Return in the correct format: 56 total outputs
            # Expected order: status, images_gallery, base_token_df, finetuned_token_df, aggregated_gallery, current_query, then 50 comparison items
            return (comparison_results[0],  # status
                    retrieved_images_for_display,  # images_gallery 
                    comparison_results[1],  # base_token_df
                    comparison_results[2],  # finetuned_token_df
                    comparison_results[3],  # aggregated_gallery
                    current_query) + comparison_results[4:]  # current_query + 50 comparison items

        
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
        
        # Use generator pattern for progressive updates with rate limiting
        query_button.click(
            query_data,
            inputs=[query_input, retrieved_results],
            outputs=[simple_response, vision_response, colpali_response, hybrid_response]
        )
        
        compare_button.click(
            handle_model_comparison,
            inputs=[retrieved_results, current_query],
            outputs=[
                comparison_status, retrieved_images_gallery, 
                base_token_analysis, finetuned_token_analysis, 
                aggregated_comparison_gallery, current_query_display
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
    
    print("🎉 Interface ready with rate limiting!")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
    return demo

