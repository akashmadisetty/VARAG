import gradio as gr
import os
import lancedb
import torch
import gc
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import List
from PIL import Image
import base64
import io
import time
from collections import namedtuple
import pandas as pd
import concurrent.futures
from varag.rag import SimpleRAG, VisionRAG, ColpaliRAG, HybridColpaliRAG
from varag.vlms import OpenAI
from varag.llms import OpenAI as OpenAILLM
from varag.vlms import LiteLLMVLM 
from varag.llms import LiteLLM 
from varag.chunking import FixedTokenChunker
from varag.utils import get_model_colpali, create_similarity_mapper,analyze_multiple_images
import argparse


load_dotenv()

# Initialize shared database
shared_db = lancedb.connect("~/rag_demo_db")

# Initialize embedding models
# text_embedding_model = SentenceTransformer("all-MiniLM-L6-v2", trust_remote_code=True)
text_embedding_model = SentenceTransformer("BAAI/bge-base-en", trust_remote_code=True)
# text_embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5", trust_remote_code=True)
# text_embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5", trust_remote_code=True)
image_embedding_model = SentenceTransformer(
    "jinaai/jina-clip-v1", trust_remote_code=True
)

# Model configurations
BASE_MODEL_NAME = "vidore/colpali-v1.3"
FINETUNED_MODEL_NAME = "akashmadisetty/colpali-merged-model-hi-10k"

# Currently active model - start with base model for backward compatibility
current_model_name = BASE_MODEL_NAME

# Load model without optimizations
colpali_model, colpali_processor = get_model_colpali(current_model_name)
print(f"✅ ColPali model loaded: {current_model_name}")

# Initialize similarity mapper for current model
try:
    similarity_mapper = create_similarity_mapper(colpali_model, colpali_processor)
    print("✅ ColPali similarity mapper initialized successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not initialize similarity mapper: {e}")
    similarity_mapper = None

# Model management functions for dynamic loading
def load_model(model_name):
    """Load a specific ColPali model and clear previous model from memory"""
    global colpali_model, colpali_processor, current_model_name, similarity_mapper
    
    if current_model_name == model_name:
        print(f"✅ Model {model_name} already loaded")
        return True
    
    try:
        print(f"🔄 Switching from {current_model_name} to {model_name}...")
        
        # Clear current model from memory
        if 'colpali_model' in globals():
            del colpali_model
        if 'colpali_processor' in globals():
            del colpali_processor
        if 'similarity_mapper' in globals():
            del similarity_mapper
        
        # Basic garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load new model without optimizations
        colpali_model, colpali_processor = get_model_colpali(model_name)
        print(f"✅ New model loaded: {model_name}")
        
        # Create new similarity mapper
        similarity_mapper = create_similarity_mapper(colpali_model, colpali_processor)
        
        current_model_name = model_name
        print(f"✅ Successfully switched to {model_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error switching to {model_name}: {e}")
        return False

def get_current_model_info():
    """Get information about currently loaded model"""
    return {
        "name": current_model_name,
        "is_base": current_model_name == BASE_MODEL_NAME,
        "is_finetuned": current_model_name == FINETUNED_MODEL_NAME
    }

# Initialize RAG instances
simple_rag = SimpleRAG(
    text_embedding_model=text_embedding_model, db=shared_db, table_name="simpleDemo"
)
vision_rag = VisionRAG(
    image_embedding_model=image_embedding_model, db=shared_db, table_name="visionDemo"
)
colpali_rag = ColpaliRAG(
    colpali_model=colpali_model,
    colpali_processor=colpali_processor,
    db=shared_db,
    table_name="colpaliDemo",
)
hybrid_rag = HybridColpaliRAG(
    colpali_model=colpali_model,
    colpali_processor=colpali_processor,
    image_embedding_model=image_embedding_model,
    db=shared_db,
    table_name="hybridDemo",
)

# Initialize VLM
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize LLM and VLM with Groq by default
if gemini_api_key:
    gemini_model = "gemini/gemini-2.5-flash-preview-04-17"
    gem_llm = LiteLLM(model=gemini_model, api_key=gemini_api_key, verbose=False)
    gem_vlm = LiteLLMVLM(model=gemini_model, api_key=gemini_api_key, verbose=False)

    llm = gem_llm
    vlm = gem_vlm
    print(f"Using Groq with model: {gemini_model}")
else:
    # For backward compatibility, use the existing initialization
    vlm = OpenAI()
    llm = OpenAILLM()
    print("Switching to OpenAI provider as no LiteLLM API key is provided.")

IngestResult = namedtuple("IngestResult", ["status_text", "progress_table"])


def ingest_data(pdf_files, use_ocr, chunk_size, progress=gr.Progress()):
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
    progress_data.append(
        {"Technique": "SimpleRAG", "Time Taken (s)": f"{simple_time:.2f}"}
    )
    yield IngestResult(
        status_text=f"SimpleRAG ingestion complete. Time taken: {simple_time:.2f} seconds\n\n",
        progress_table=pd.DataFrame(progress_data),
    )
    # progress(0.25, desc="SimpleRAG complete")

    # VisionRAG
    yield IngestResult(
        status_text="Starting VisionRAG ingestion...\n",
        progress_table=pd.DataFrame(progress_data),
    )
    start_time = time.time()
    vision_rag.index(file_paths, overwrite=False, recursive=False, verbose=True)
    vision_time = time.time() - start_time
    progress_data.append(
        {"Technique": "VisionRAG", "Time Taken (s)": f"{vision_time:.2f}"}
    )
    yield IngestResult(
        status_text=f"VisionRAG ingestion complete. Time taken: {vision_time:.2f} seconds\n\n",
        progress_table=pd.DataFrame(progress_data),
    )
    # progress(0.5, desc="VisionRAG complete")

    # ColpaliRAG
    yield IngestResult(
        status_text="Starting ColpaliRAG ingestion...\n",
        progress_table=pd.DataFrame(progress_data),
    )
    start_time = time.time()
    colpali_rag.index(file_paths, overwrite=False, recursive=False, verbose=True)
    colpali_time = time.time() - start_time
    progress_data.append(
        {"Technique": "ColpaliRAG", "Time Taken (s)": f"{colpali_time:.2f}"}
    )
    yield IngestResult(
        status_text=f"ColpaliRAG ingestion complete. Time taken: {colpali_time:.2f} seconds\n\n",
        progress_table=pd.DataFrame(progress_data),
    )
    # progress(0.75, desc="ColpaliRAG complete")

    # HybridColpaliRAG
    yield IngestResult(
        status_text="Starting HybridColpaliRAG ingestion...\n",
        progress_table=pd.DataFrame(progress_data),
    )
    start_time = time.time()
    hybrid_rag.index(file_paths, overwrite=False, recursive=False, verbose=True)
    hybrid_time = time.time() - start_time
    progress_data.append(
        {"Technique": "HybridColpaliRAG", "Time Taken (s)": f"{hybrid_time:.2f}"}
    )
    yield IngestResult(
        status_text=f"HybridColpaliRAG ingestion complete. Time taken: {hybrid_time:.2f} seconds\n\n",
        progress_table=pd.DataFrame(progress_data),
    )
    # progress(1.0, desc="HybridColpaliRAG complete")

    total_time = time.time() - total_start_time
    progress_data.append({"Technique": "Total", "Time Taken (s)": f"{total_time:.2f}"})
    yield IngestResult(
        status_text=f"Total ingestion time: {total_time:.2f} seconds",
        progress_table=pd.DataFrame(progress_data),
    )


def retrieve_data(query, top_k, sequential=False):
    results = {}
    timings = {}

    def retrieve_simple():
        start_time = time.time()
        simple_results = simple_rag.search(query, k=top_k)

        print(simple_results)

        simple_context = []
        for i, r in enumerate(simple_results, 1):
            context_piece = f"Result {i}:\n"
            context_piece += f"Source: {r.get('document_name', 'Unknown')}\n"
            context_piece += f"Chunk Index: {r.get('chunk_index', 'Unknown')}\n"

            context_piece += f"Content:\n{r['text']}\n"
            context_piece += "-" * 40 + "\n"  # Separator
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

    retrieval_functions = [
        retrieve_simple,
        retrieve_vision,
        retrieve_colpali,
        retrieve_hybrid,
    ]

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


def update_api_key(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    return "API key updated successfully."


def change_table(simple_table, vision_table, colpali_table, hybrid_table):
    simple_rag.change_table(simple_table)
    vision_rag.change_table(vision_table)
    colpali_rag.change_table(colpali_table)
    hybrid_rag.change_table(hybrid_table)
    return "Table names updated successfully."


def get_query_token_breakdown(query):
    """
    Analyze query tokens using the ColPali processor to show how the query is tokenized.
    Returns a formatted markdown string showing the token breakdown.
    """
    if not query or not query.strip():
        return "**Token Analysis:** No query provided"
    
    try:
        # Use the current ColPali processor to tokenize the query
        if colpali_processor:
            # Tokenize the query (similar to how ColPali does it internally)
            inputs = colpali_processor.process_queries([query])
            
            # Get tokens from the processor
            if hasattr(colpali_processor, 'tokenizer'):
                tokens = colpali_processor.tokenizer.tokenize(query)
                token_ids = colpali_processor.tokenizer.encode(query, add_special_tokens=False)
                
                # Create a nice breakdown
                breakdown_text = f"""**🔍 Query Token Breakdown**

**Original Query:** `{query}`

**Number of Tokens:** {len(tokens)} tokens

**Token Details:**
"""
                
                for i, (token, token_id) in enumerate(zip(tokens, token_ids), 1):
                    # Clean up token representation for display
                    display_token = token.replace('▁', ' ').strip()
                    if display_token.startswith(' '):
                        display_token = f"[SPACE]{display_token}"
                    breakdown_text += f"- **{i}.** `{display_token}` (ID: {token_id})\n"
                
                breakdown_text += f"""
**💡 Understanding Tokens:**
- ColPali analyzes each token separately to understand query intent
- More tokens = more detailed similarity maps generated  
- Special tokens like `▁` represent word boundaries
- Each token gets its own attention visualization
"""
                
                print(f"🔤 Query tokenized into {len(tokens)} tokens: {tokens}")
                return breakdown_text
            else:
                return f"**Token Analysis:** Query tokenization not available (no tokenizer found)"
                
    except Exception as e:
        print(f"❌ Error in token breakdown: {e}")
        return f"**Token Analysis:** Error analyzing query tokens: {str(e)}"
    
    return f"**Token Analysis:** Could not analyze tokens for query: `{query}`"


def gradio_interface():
    with gr.Blocks(
        theme=gr.themes.Monochrome(radius_size=gr.themes.sizes.radius_none)
    ) as demo:
        gr.Markdown(
            """
# 👁️👁️ Vision RAG Playground

### Explore and Compare Vision-Augmented Retrieval Techniques
Built on [VARAG](https://github.com/adithya-s-k/VARAG) - Vision-Augmented Retrieval and Generation

**[⭐ Star the Repository](https://github.com/adithya-s-k/VARAG)** to support the project!

1. **Simple RAG**: Text-based retrieval with OCR support for scanned documents.
2. **Vision RAG**: Combines text and image retrieval using cross-modal embeddings.
3. **ColPali RAG**: Embeds entire document pages as images for layout-aware retrieval.
4. **Hybrid ColPali RAG**: Two-stage retrieval combining image embeddings and ColPali's token-level matching.

            """
        )

        with gr.Tab("Ingest Data"):
            pdf_input = gr.File(
                label="Upload PDF(s)", file_count="multiple", file_types=[".pdf"]
            )
            use_ocr = gr.Checkbox(label="Use OCR (for SimpleRAG)")
            chunk_size = gr.Slider(
                50, 5000, value=300, step=10, label="Chunk Size (for SimpleRAG)"
            )
            ingest_button = gr.Button("Ingest PDFs")
            ingest_output = gr.Markdown(
                label="Ingestion Status :",
            )
            progress_table = gr.DataFrame(
                label="Ingestion Progress", headers=["Technique", "Time Taken (s)"]
            )

        with gr.Tab("Retrieve and Query Data"):
            query_input = gr.Textbox(label="Enter your query")
            top_k_slider = gr.Slider(1, 10, value=3, step=1, label="Top K Results")
            sequential_checkbox = gr.Checkbox(label="Sequential Retrieval", value=False)
            retrieve_button = gr.Button("Retrieve")
            query_button = gr.Button("Query")

            retrieval_timing = gr.DataFrame(
                label="Retrieval Timings", headers=["RAG Type", "Time (s)"]
            )

            with gr.Row():
                with gr.Column():
                    with gr.Accordion("SimpleRAG", open=True):
                        simple_content = gr.Textbox(
                            label="SimpleRAG Content", lines=10, max_lines=10
                        )
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

        with gr.Tab("Interpret ColPali"):
            gr.Markdown("""
            ## 🔍 ColPali Model Comparison Dashboard
            
            This section helps you compare the base ColPali model (`vidore/colpali-v1.3`) with your fine-tuned model (`akashmadisetty/colpali-merged-model-hi-10k`).            First, perform a retrieval in the "Retrieve and Query Data" tab, then come here to analyze and compare the results.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    current_query_display = gr.Textbox(
                        label="Current Query", 
                        value="No query yet - perform a retrieval first", 
                        interactive=False,
                        lines=3
                    )
                    
                    # Token breakdown display
                    token_breakdown_display = gr.Markdown(
                        label="Query Token Breakdown",
                        value="**Token Analysis:** No query tokenized yet",
                        visible=True
                    )
                    
                    # Model selection and management
                    with gr.Row():
                        current_model_display = gr.Textbox(
                            label="Currently Loaded Model",
                            value=current_model_name,
                            interactive=False
                        )                    
                    with gr.Row():
                        model_selector = gr.Dropdown(
                            choices=[BASE_MODEL_NAME, FINETUNED_MODEL_NAME],
                            value=current_model_name,
                            label="Switch Model",
                            interactive=True
                        )
                        switch_model_button = gr.Button("🔄 Switch Model", variant="secondary", size="sm")
                    
                    refresh_interpretation_button = gr.Button("🔄 Refresh ColPali Results", variant="primary")
                    generate_comparison_button = gr.Button("🎯 Generate Model Comparison", variant="secondary")
                    
                with gr.Column(scale=2):
                    colpali_interpretation_gallery = gr.Gallery(
                        label="ColPali Retrieved Images",
                        show_label=True,
                        elem_id="colpali_interpretation",
                        columns=2,
                        rows=2,
                        height="400px"
                    )
            
            # Model Comparison Section
            with gr.Row():
                comparison_status = gr.Markdown("**Model Comparison:** Click 'Generate Model Comparison' to analyze token-level attention differences")
            
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
                3. Come back here and click "🔄 Refresh ColPali Results"
                4. Analyze what images ColPali found relevant to your query
                
                **Advanced Analysis:**
                - For detailed similarity mapping and token analysis, check out the 
                  [ColPali Interpretation Notebook](docs/colpali_interpretation.ipynb)
                - This notebook provides visual attention maps, token importance analysis, 
                  and query optimization suggestions
                """)
                
            with gr.Row():
                gr.Markdown("""
                ### 🚀 Advanced Features Coming Soon:
                - **Query Optimization**: Suggestions for better retrieval
                - **Comparative Analysis**: Compare multiple retrievals
                """)

        with gr.Tab("Settings"):
            api_key_input = gr.Textbox(label="OpenAI API Key", type="password")
            update_api_button = gr.Button("Update API Key")
            api_update_status = gr.Textbox(label="API Update Status")

            simple_table_input = gr.Textbox(
                label="SimpleRAG Table Name", value="simpleDemo"
            )
            vision_table_input = gr.Textbox(
                label="VisionRAG Table Name", value="visionDemo"
            )
            colpali_table_input = gr.Textbox(
                label="ColpaliRAG Table Name", value="colpaliDemo"
            )
            hybrid_table_input = gr.Textbox(
                label="HybridColpaliRAG Table Name", value="hybridDemo"
            )
            update_table_button = gr.Button("Update Table Names")
            table_update_status = gr.Textbox(label="Table Update Status")

        retrieved_results = gr.State({})
        current_query = gr.State("")

        def switch_model_handler(selected_model):
            """Handle model switching"""
            try:
                if load_model(selected_model):
                    return f"✅ Successfully switched to: {selected_model}", selected_model
                else:
                    return f"❌ Failed to switch to: {selected_model}", get_current_model_info()["name"]
            except Exception as e:
                return f"❌ Error switching model: {str(e)}", get_current_model_info()["name"]

        def refresh_colpali_interpretation(retrieved_results, current_query):
            """Refresh the ColPali interpretation with current query and results"""
            if not retrieved_results or "ColpaliRAG" not in retrieved_results:
                return "No ColPali results available - perform a retrieval first", [], "**Token Analysis:** No query yet"
            
            if not current_query:
                return "No query available", [], "**Token Analysis:** No query available"
            
            colpali_images = retrieved_results.get("ColpaliRAG", [])
            
            # Get token breakdown for the current query
            token_breakdown = get_query_token_breakdown(current_query)
            
            return current_query, colpali_images, token_breakdown

        def generate_model_comparison_for_images(retrieved_results, current_query):
            """Generate and compare similarity maps for both base and fine-tuned ColPali models"""
            if not similarity_mapper:
                return ["⚠️ Similarity mapper not available"] + [gr.DataFrame(visible=False)] * 2 + [[]] * 20 + [gr.Row(visible=False)] * 10 + [gr.Markdown(visible=False)] * 20

            if not retrieved_results or "ColpaliRAG" not in retrieved_results:
                return ["❌ No ColPali results available - perform a retrieval first"] + [gr.DataFrame(visible=False)] * 2 + [[]] * 20 + [gr.Row(visible=False)] * 10 + [gr.Markdown(visible=False)] * 20
            
            if not current_query:
                return ["❌ No query available"] + [gr.DataFrame(visible=False)] * 2 + [[]] * 20 + [gr.Row(visible=False)] * 10 + [gr.Markdown(visible=False)] * 20
            
            colpali_images = retrieved_results.get("ColpaliRAG", [])
            
            if not colpali_images:
                return ["❌ No images to analyze"] + [gr.DataFrame(visible=False)] * 2 + [[]] * 20 + [gr.Row(visible=False)] * 10 + [gr.Markdown(visible=False)] * 20
            
            # Update status
            status_text = f"🔄 Comparing models on {len(colpali_images)} images for query: '{current_query}'..."
              # Generate similarity maps for both models
            try:
                # Store current model info
                original_model = get_current_model_info()
                
                # Generate results for base model
                print(f"🔄 Loading base model for comparison...")
                if not load_model(BASE_MODEL_NAME):
                    return ["❌ Failed to load base model"] + [gr.DataFrame(visible=False)] * 2 + [[]] * 20 + [gr.Row(visible=False)] * 10 + [gr.Markdown(visible=False)] * 20
                
                base_results = analyze_multiple_images(similarity_mapper, colpali_images, current_query)
                print(f"✅ Base model analysis complete")
                
                # Generate results for fine-tuned model
                print(f"🔄 Loading fine-tuned model for comparison...")
                if not load_model(FINETUNED_MODEL_NAME):
                    return ["❌ Failed to load fine-tuned model"] + [gr.DataFrame(visible=False)] * 2 + [[]] * 20 + [gr.Row(visible=False)] * 10 + [gr.Markdown(visible=False)] * 20
                
                finetuned_results = analyze_multiple_images(similarity_mapper, colpali_images, current_query)
                print(f"✅ Fine-tuned model analysis complete")
                
                # Restore original model if different
                if original_model["name"] != current_model_name:
                    print(f"🔄 Restoring original model: {original_model['name']}")
                    load_model(original_model["name"])
                
                # Prepare comparison status and token analysis
                base_token_analysis_data = []
                finetuned_token_analysis_data = []
                
                if base_results and len(base_results) > 0 and base_results[0]["success"]:
                    base_result = base_results[0]
                    for rank, token_data in enumerate(base_result.get("token_scores", []), 1):
                        token = token_data["token"]
                        sim = token_data["max_similarity"]
                        base_token_analysis_data.append([token, f"{sim:.3f}", rank])
                
                if finetuned_results and len(finetuned_results) > 0 and finetuned_results[0]["success"]:
                    finetuned_result = finetuned_results[0]
                    for rank, token_data in enumerate(finetuned_result.get("token_scores", []), 1):
                        token = token_data["token"]
                        sim = token_data["max_similarity"]
                        finetuned_token_analysis_data.append([token, f"{sim:.3f}", rank])
                
                # Create detailed comparison status
                status_text = f"""✅ **Model Comparison Complete for Query:** "{current_query}"

**📊 Results Summary:**
- **Base Model (vidore/colpali-v1.3):** {len(base_results)} images analyzed
- **Fine-tuned Model (akashmadisetty/colpali-merged-model-hi-10k):** {len(finetuned_results)} images analyzed

**💡 How to compare:** 
- Look at token similarity scores in the tables below
- Compare attention patterns in the side-by-side galleries
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
                    visible=True                )
                
                # Prepare outputs for galleries
                base_gallery_updates = []
                finetuned_gallery_updates = []
                row_updates = []
                base_page_info_updates = []
                finetuned_page_info_updates = []
                
                # Process images
                max_images = min(len(base_results), len(finetuned_results), 10)
                
                for i in range(10):  # Still support up to 10 galleries in UI
                    if i < max_images:
                        base_result = base_results[i]
                        finetuned_result = finetuned_results[i]
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
                            for rank, token_data in enumerate(base_result.get("token_scores", [])[:10], 1):  # Top 10 tokens
                                token = token_data["token"]
                                sim = token_data["max_similarity"]
                                base_token_breakdown.append(f"{rank}. **'{token}'** ({sim:.3f})")
                            
                            base_page_info_text = f"""**Top tokens by importance:**
{chr(10).join(base_token_breakdown)}  
**Visualizations:** {len(base_vis_images)}"""
                            
                            # Create detailed page info for fine-tuned model
                            finetuned_token_breakdown = []
                            for rank, token_data in enumerate(finetuned_result.get("token_scores", [])[:10], 1):  # Top 10 tokens
                                token = token_data["token"]
                                sim = token_data["max_similarity"]
                                finetuned_token_breakdown.append(f"{rank}. **'{token}'** ({sim:.3f})")
                            
                            finetuned_page_info_text = f"""**Top tokens by importance:**
{chr(10).join(finetuned_token_breakdown)}  
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
                  # Return: status, base_token_df, finetuned_token_df, base_galleries, finetuned_galleries, rows, base_page_infos, finetuned_page_infos
                return [status_text] + [base_token_df] + [finetuned_token_df] + base_gallery_updates + finetuned_gallery_updates + row_updates + base_page_info_updates + finetuned_page_info_updates
                
            except Exception as e:
                error_msg = f"❌ Error generating model comparison: {str(e)}"
                print(f"Comparison error: {e}")
                # Try to restore original model on error
                try:
                    if 'original_model' in locals():
                        load_model(original_model["name"])
                except:
                    pass
                return [error_msg] + [gr.DataFrame(visible=False)] * 2 + [[]] * 20 + [gr.Row(visible=False)] * 10 + [gr.Markdown(visible=False)] * 20

        def update_retrieval_results(query, top_k, sequential):
            results, timings = retrieve_data(query, top_k, sequential)
            timing_df = pd.DataFrame(
                list(timings.items()), columns=["RAG Type", "Time (s)"]
            )
            return (
                results["SimpleRAG"],
                results["VisionRAG"],
                results["ColpaliRAG"],
                results["HybridColpaliRAG"],
                timing_df,
                results,
                query,  # Store the current query
            )

        retrieve_button.click(
            update_retrieval_results,
            inputs=[query_input, top_k_slider, sequential_checkbox],
            outputs=[
                simple_content,
                vision_gallery,
                colpali_gallery,
                hybrid_gallery,
                retrieval_timing,
                retrieved_results,
                current_query,
            ],
        )

        def update_query_results(query, retrieved_results):
            # Initialize empty responses
            responses = {
                "SimpleRAG": {"response": "Processing..."},
                "VisionRAG": {"response": "Waiting..."},
                "ColpaliRAG": {"response": "Waiting..."},
                "HybridColpaliRAG": {"response": "Waiting..."}
            }
            
            # Process SimpleRAG first
            responses["SimpleRAG"] = query_data_single(query, retrieved_results, "SimpleRAG")
            yield (
                responses["SimpleRAG"]["response"],
                responses["VisionRAG"]["response"],
                responses["ColpaliRAG"]["response"],
                responses["HybridColpaliRAG"]["response"],
            )
            
            # Wait to avoid rate limit
            time.sleep(30)
            
            # Process VisionRAG
            responses["VisionRAG"]["response"] = "Processing..."
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
            
            # Process ColpaliRAG
            time.sleep(30)
            responses["ColpaliRAG"]["response"] = "Processing..."
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
            
            # Process HybridColpaliRAG
            time.sleep(60)
            responses["HybridColpaliRAG"]["response"] = "Processing..."
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

        # Helper function to query a single RAG model
        def query_data_single(query, retrieved_results, model_name):
            if model_name == "SimpleRAG":
                simple_context = retrieved_results["SimpleRAG"]
                simple_response = llm.query(
                    context=simple_context,
                    system_prompt="Given the below information answer the questions",
                    query=query,
                )
                return {"response": simple_response, "context": simple_context}
            
            elif model_name == "VisionRAG":
                vision_images = retrieved_results["VisionRAG"]
                vision_context = f"Query: {query}\n\nRelevant image information:\n" + "\n".join(
                    [f"Image {i+1}" for i in range(len(vision_images))]
                )
                vision_response = vlm.query(vision_context, vision_images, max_tokens=500)
                return {
                    "response": vision_response,
                    "context": vision_context,
                    "images": vision_images,
                }
            
            elif model_name == "ColpaliRAG":
                colpali_images = retrieved_results["ColpaliRAG"]
                colpali_context = f"Query: {query}\n\nRelevant image information:\n" + "\n".join(
                    [f"Image {i+1}" for i in range(len(colpali_images))]
                )
                colpali_response = vlm.query(colpali_context, colpali_images, max_tokens=500)
                return {
                    "response": colpali_response,
                    "context": colpali_context,
                    "images": colpali_images,
                }
            
            elif model_name == "HybridColpaliRAG":
                hybrid_images = retrieved_results["HybridColpaliRAG"]
                hybrid_context = f"Query: {query}\n\nRelevant image information:\n" + "\n".join(
                    [f"Image {i+1}" for i in range(len(hybrid_images))]
                )
                hybrid_response = vlm.query(hybrid_context, hybrid_images, max_tokens=500)
                return {
                    "response": hybrid_response,
                    "context": hybrid_context,
                    "images": hybrid_images,
                }
            
            return {"response": "Model not recognized", "context": ""}

        # Update button click to use the generator pattern
        query_button.click(
            update_query_results,
            inputs=[query_input, retrieved_results],
            outputs=[
                simple_response,
                vision_response,
                colpali_response,
                hybrid_response
            ]
        )        # Model switching handler
        switch_model_button.click(
            switch_model_handler,
            inputs=[model_selector],
            outputs=[comparison_status, current_model_display]        )
        
        # ColPali interpretation refresh handler
        refresh_interpretation_button.click(
            refresh_colpali_interpretation,
            inputs=[retrieved_results, current_query],
            outputs=[current_query_display, colpali_interpretation_gallery, token_breakdown_display]
        )
        
        # Model comparison generation handler
        generate_comparison_button.click(
            generate_model_comparison_for_images,
            inputs=[retrieved_results, current_query],
            outputs=[comparison_status] + [base_token_analysis] + [finetuned_token_analysis] + 
                    [base_gallery for _, base_gallery, _, _, _ in comparison_galleries] + 
                    [finetuned_gallery for _, _, finetuned_gallery, _, _ in comparison_galleries] + 
                    [row for row, _, _, _, _ in comparison_galleries] + 
                    [base_info for _, _, _, base_info, _ in comparison_galleries] + 
                    [finetuned_info for _, _, _, _, finetuned_info in comparison_galleries]
        )

        ingest_button.click(
            ingest_data,
            inputs=[pdf_input, use_ocr, chunk_size],
            outputs=[ingest_output, progress_table],
        )

        update_api_button.click(
            update_api_key, inputs=[api_key_input], outputs=api_update_status
        )

        update_table_button.click(
            change_table,
            inputs=[
                simple_table_input,
                vision_table_input,
                colpali_table_input,
                hybrid_table_input,
            ],
            outputs=table_update_status,        )

    return demo


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="VisionRAG Gradio App")
    parser.add_argument(
        "--share", action="store_true", help="Enable Gradio share feature"
    )
    return parser.parse_args()


# Launch the app
if __name__ == "__main__":
    args = parse_args()
    app = gradio_interface()
    app.launch(share=args.share)
