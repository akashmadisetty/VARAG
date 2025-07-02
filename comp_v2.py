import os
import gc
import time
import torch
import gradio as gr
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any
import lancedb

# Import VARAG components
from varag.rag import ColpaliRAG
from varag.utils import get_model_colpali
from colpali_similarity_v2 import create_similarity_mapper, analyze_multiple_images

# Import memory optimization utilities
try:
    from memory_optimizer import (
        print_memory_status, 
        clear_memory, 
        optimize_for_t4_gpu,
        memory_efficient_inference_settings
    )
    print("✅ Memory optimizer loaded")
except ImportError:
    print("⚠️ Memory optimizer not available - using basic memory management")
    def print_memory_status(): pass
    def clear_memory(): 
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    def optimize_for_t4_gpu(): pass
    def memory_efficient_inference_settings(): pass

# Import safe model loader
try:
    from safe_model_loader import safe_get_model_colpali
    print("✅ Safe model loader available")
    use_safe_loader = True
except ImportError:
    print("⚠️ Safe model loader not available - using default")
    use_safe_loader = False

load_dotenv()

# Apply memory optimizations early
print("🔧 Applying memory optimizations for T4 GPU...")
optimize_for_t4_gpu()
memory_efficient_inference_settings()

# Model configurations
BASE_MODEL_NAME = "vidore/colpali-v1.3"
FINETUNED_MODEL_NAME = "akashmadisetty/colpali-merged-model-hi-10k"

class ColPaliModelComparison:
    """Class to handle ColPali model comparison with memory-efficient loading"""
    
    def __init__(self):
        self.base_model = None
        self.base_processor = None
        self.finetuned_model = None
        self.finetuned_processor = None
        self.base_rag = None
        self.finetuned_rag = None
        self.base_similarity_mapper = None
        self.finetuned_similarity_mapper = None
        self.db = lancedb.connect("~/colpali_comparison_db")
        
    def load_base_model(self):
        """Load the base ColPali model"""
        try:
            print(f"🔄 Loading base model: {BASE_MODEL_NAME}")
            clear_memory()
            
            if use_safe_loader:
                self.base_model, self.base_processor = safe_get_model_colpali(BASE_MODEL_NAME)
            else:
                self.base_model, self.base_processor = get_model_colpali(BASE_MODEL_NAME)
            
            # Create RAG instance
            self.base_rag = ColpaliRAG(
                colpali_model=self.base_model,
                colpali_processor=self.base_processor,
                db=self.db,
                table_name="base_model_comparison"
            )
            
            # Create similarity mapper
            self.base_similarity_mapper = create_similarity_mapper(self.base_model, self.base_processor)
            
            print(f"✅ Base model loaded successfully")
            print_memory_status()
            return True
            
        except Exception as e:
            print(f"❌ Error loading base model: {e}")
            return False
    
    def load_finetuned_model(self):
        """Load the fine-tuned ColPali model"""
        try:
            print(f"🔄 Loading fine-tuned model: {FINETUNED_MODEL_NAME}")
            
            # Clear base model from memory first
            if self.base_model is not None:
                del self.base_model
                del self.base_processor
                del self.base_similarity_mapper
                self.base_model = None
                self.base_processor = None
                self.base_similarity_mapper = None
            
            clear_memory()
            
            if use_safe_loader:
                self.finetuned_model, self.finetuned_processor = safe_get_model_colpali(FINETUNED_MODEL_NAME)
            else:
                self.finetuned_model, self.finetuned_processor = get_model_colpali(FINETUNED_MODEL_NAME)
            
            # Create RAG instance
            self.finetuned_rag = ColpaliRAG(
                colpali_model=self.finetuned_model,
                colpali_processor=self.finetuned_processor,
                db=self.db,
                table_name="finetuned_model_comparison"
            )
            
            # Create similarity mapper
            self.finetuned_similarity_mapper = create_similarity_mapper(self.finetuned_model, self.finetuned_processor)
            
            print(f"✅ Fine-tuned model loaded successfully")
            print_memory_status()
            return True
            
        except Exception as e:
            print(f"❌ Error loading fine-tuned model: {e}")
            return False
    
    def reload_base_model(self):
        """Reload base model after fine-tuned model was loaded"""
        # Clear fine-tuned model
        if self.finetuned_model is not None:
            del self.finetuned_model
            del self.finetuned_processor
            del self.finetuned_similarity_mapper
            self.finetuned_model = None
            self.finetuned_processor = None
            self.finetuned_similarity_mapper = None
        
        return self.load_base_model()

# Initialize the comparison class
comparison = ColPaliModelComparison()

def ingest_pdf(pdf_files, progress=gr.Progress()):
    """Ingest PDF files for both models"""
    if not pdf_files:
        return "❌ No PDF files provided", "", ""
    
    file_paths = [pdf_file.name for pdf_file in pdf_files]
    results = []
    
    try:
        # Load and index with base model
        yield "🔄 Loading base model...", "", ""
        if not comparison.load_base_model():
            return "❌ Failed to load base model", "", ""
        
        yield "🔄 Indexing with base model...", "", ""
        start_time = time.time()
        comparison.base_rag.index(file_paths, overwrite=True, recursive=False, verbose=True)
        base_time = time.time() - start_time
        results.append(f"✅ Base model indexing: {base_time:.2f}s")
        
        # Load and index with fine-tuned model
        yield f"✅ Base model indexed in {base_time:.2f}s\n🔄 Loading fine-tuned model...", "", ""
        if not comparison.load_finetuned_model():
            return "❌ Failed to load fine-tuned model", "", ""
        
        yield f"✅ Base model indexed in {base_time:.2f}s\n🔄 Indexing with fine-tuned model...", "", ""
        start_time = time.time()
        comparison.finetuned_rag.index(file_paths, overwrite=True, recursive=False, verbose=True)
        finetuned_time = time.time() - start_time
        results.append(f"✅ Fine-tuned model indexing: {finetuned_time:.2f}s")
        
        final_status = "\n".join(results)
        final_status += f"\n\n📊 Total time: {base_time + finetuned_time:.2f}s"
        
        return final_status, "✅ Ready for retrieval", "✅ Ready for retrieval"
        
    except Exception as e:
        return f"❌ Error during ingestion: {e}", "❌ Error", "❌ Error"

def retrieve_and_compare(query, top_k):
    """Retrieve top-k results from both models and generate similarity maps"""
    if not query.strip():
        return "❌ Please enter a query", [], [], "", "", [], []
    
    try:
        # Ensure base model is loaded for retrieval
        if comparison.base_model is None:
            status_msg = "🔄 Reloading base model for retrieval..."
            if not comparison.reload_base_model():
                return "❌ Failed to reload base model", [], [], "", "", [], []
        
        # Retrieve from base model
        print(f"🔍 Searching with base model for: '{query}'")
        base_start = time.time()
        base_results = comparison.base_rag.search(query, k=top_k)
        base_time = time.time() - base_start
        base_images = [r["image"] for r in base_results]
        
        # Generate similarity maps for base model
        base_sim_maps = []
        if comparison.base_similarity_mapper and base_images:
            try:
                base_analysis = analyze_multiple_images(
                    comparison.base_similarity_mapper, 
                    base_images, 
                    query
                )
                base_sim_maps = [result['similarity_map'] for result in base_analysis['results']]
            except Exception as e:
                print(f"⚠️ Error generating base similarity maps: {e}")
        
        base_status = f"✅ Base Model Results ({base_time:.2f}s)\nFound {len(base_images)} images"
        
        # Switch to fine-tuned model
        if not comparison.load_finetuned_model():
            return base_status, base_images, base_sim_maps, "❌ Failed to load fine-tuned model", [], []
        
        # Retrieve from fine-tuned model
        print(f"🔍 Searching with fine-tuned model for: '{query}'")
        finetuned_start = time.time()
        finetuned_results = comparison.finetuned_rag.search(query, k=top_k)
        finetuned_time = time.time() - finetuned_start
        finetuned_images = [r["image"] for r in finetuned_results]
        
        # Generate similarity maps for fine-tuned model
        finetuned_sim_maps = []
        if comparison.finetuned_similarity_mapper and finetuned_images:
            try:
                finetuned_analysis = analyze_multiple_images(
                    comparison.finetuned_similarity_mapper, 
                    finetuned_images, 
                    query
                )
                finetuned_sim_maps = [result['similarity_map'] for result in finetuned_analysis['results']]
            except Exception as e:
                print(f"⚠️ Error generating fine-tuned similarity maps: {e}")
        
        finetuned_status = f"✅ Fine-tuned Model Results ({finetuned_time:.2f}s)\nFound {len(finetuned_images)} images"
        
        return (
            base_status, base_images, base_sim_maps,
            finetuned_status, finetuned_images, finetuned_sim_maps
        )
        
    except Exception as e:
        return f"❌ Error during retrieval: {e}", [], [], "❌ Error", [], []

def create_interface():
    """Create the Gradio interface with split-screen comparison"""
    
    with gr.Blocks(title="ColPali Model Comparison", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🔍 ColPali Model Comparison Dashboard
        
        Compare the base ColPali model (`vidore/colpali-v1.3`) with the fine-tuned model (`akashmadisetty/colpali-merged-model-hi-10k`).
        
        **Steps:**
        1. Upload PDF files to ingest
        2. Enter your query and select top-k results
        3. Compare retrieval results and similarity maps side-by-side
        """)
        
        # PDF Upload Section
        with gr.Row():
            with gr.Column():
                pdf_files = gr.File(
                    label="📁 Upload PDF Files", 
                    file_count="multiple", 
                    file_types=[".pdf"]
                )
                ingest_btn = gr.Button("🚀 Ingest PDFs", variant="primary", size="lg")
        
        with gr.Row():
            ingest_status = gr.Textbox(
                label="📊 Ingestion Status", 
                lines=5, 
                interactive=False
            )
        
        # Query Section
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="🔍 Enter Your Query", 
                    placeholder="What information are you looking for?",
                    lines=2
                )
            with gr.Column(scale=1):
                top_k_slider = gr.Slider(
                    minimum=1, 
                    maximum=10, 
                    value=3, 
                    step=1, 
                    label="📊 Top K Results"
                )
                search_btn = gr.Button("🔍 Search & Compare", variant="primary", size="lg")
        
        # Results Section - Split Screen
        with gr.Row():
            # Left Half - Base Model
            with gr.Column(scale=1):
                gr.Markdown("## 🏛️ Base Model (vidore/colpali-v1.3)")
                base_status = gr.Textbox(
                    label="Status", 
                    lines=3, 
                    interactive=False,
                    value="Waiting for PDFs to be ingested..."
                )
                
                gr.Markdown("### 📸 Retrieved Images")
                base_images_gallery = gr.Gallery(
                    label="Base Model Results",
                    show_label=False,
                    elem_id="base_gallery",
                    columns=2,
                    rows=2,
                    height="400px"
                )
                
                gr.Markdown("### 🎯 Similarity Maps")
                base_similarity_gallery = gr.Gallery(
                    label="Base Model Similarity Maps",
                    show_label=False,
                    elem_id="base_sim_gallery",
                    columns=2,
                    rows=2,
                    height="400px"
                )
            
            # Right Half - Fine-tuned Model
            with gr.Column(scale=1):
                gr.Markdown("## 🎯 Fine-tuned Model (akashmadisetty/colpali-merged-model-hi-10k)")
                finetuned_status = gr.Textbox(
                    label="Status", 
                    lines=3, 
                    interactive=False,
                    value="Waiting for PDFs to be ingested..."
                )
                
                gr.Markdown("### 📸 Retrieved Images")
                finetuned_images_gallery = gr.Gallery(
                    label="Fine-tuned Model Results",
                    show_label=False,
                    elem_id="finetuned_gallery",
                    columns=2,
                    rows=2,
                    height="400px"
                )
                
                gr.Markdown("### 🎯 Similarity Maps")
                finetuned_similarity_gallery = gr.Gallery(
                    label="Fine-tuned Model Similarity Maps",
                    show_label=False,
                    elem_id="finetuned_sim_gallery",
                    columns=2,
                    rows=2,
                    height="400px"
                )
        
        # Memory Status
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### 💡 Tips for Best Performance:
                - **Memory Optimization**: Models are loaded sequentially to avoid CUDA OOM errors
                - **T4 GPU Support**: Optimized for T4 GPUs with automatic precision handling
                - **Comparison**: Results are displayed side-by-side for easy visual comparison
                - **Similarity Maps**: Show which parts of the images the models focus on for your query
                """)
        
        # Event Handlers
        ingest_btn.click(
            fn=ingest_pdf,
            inputs=[pdf_files],
            outputs=[ingest_status, base_status, finetuned_status]
        )
        
        search_btn.click(
            fn=retrieve_and_compare,
            inputs=[query_input, top_k_slider],
            outputs=[
                base_status, base_images_gallery, base_similarity_gallery,
                finetuned_status, finetuned_images_gallery, finetuned_similarity_gallery
            ]
        )
    
    return app

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ColPali Model Comparison App")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share feature")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    args = parser.parse_args()
    
    print("🚀 Starting ColPali Model Comparison App...")
    print_memory_status()
    
    app = create_interface()
    app.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0" if args.share else "127.0.0.1"
    )