import gradio as gr
import os
import lancedb
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
from varag.utils import get_model_colpali
import argparse
from colpali_similarity_v2 import create_similarity_mapper,analyze_multiple_images

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
colpali_model, colpali_processor = get_model_colpali("vidore/colpali-v1.3")

# Initialize ColPali similarity mapper
try:
    similarity_mapper = create_similarity_mapper(colpali_model, colpali_processor)
    print("✅ ColPali similarity mapper initialized successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not initialize similarity mapper: {e}")
    similarity_mapper = None

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
            ## 🔍 ColPali Interpretation Dashboard
            
            This section helps you understand what ColPali RAG is retrieving and how. 
            First, perform a retrieval in the "Retrieve and Query Data" tab, then come here to analyze the results.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    current_query_display = gr.Textbox(
                        label="Current Query", 
                        value="No query yet - perform a retrieval first", 
                        interactive=False,
                        lines=3
                    )
                    refresh_interpretation_button = gr.Button("🔄 Refresh ColPali Results", variant="primary")
                    generate_similarity_button = gr.Button("🎯 Generate Similarity Maps", variant="secondary")
                    
                with gr.Column(scale=2):
                    colpali_interpretation_gallery = gr.Gallery(
                        label="ColPali Retrieved Images",
                        show_label=True,
                        elem_id="colpali_interpretation",
                        columns=2,
                        rows=2,
                        height="400px"
                    )
              # Similarity Maps Section
            with gr.Row():
                similarity_status = gr.Markdown("**Similarity Maps:** Click 'Generate Similarity Maps' to analyze token-level attention")
            
            # Token Analysis Results
            with gr.Row():
                token_analysis = gr.DataFrame(
                    label="Token Analysis Results",
                    headers=["Token", "Max Similarity", "Rank"],
                    visible=False
                )
            
            # Dynamic galleries for each retrieved image
            similarity_galleries = []
            for i in range(10):  # Support up to 10 retrieved images
                with gr.Row(visible=False) as similarity_row:
                    with gr.Column():
                        page_info = gr.Markdown(f"### 📄 Page {i+1} Token Analysis", visible=True)
                        similarity_gallery = gr.Gallery(
                            label=f"Token Similarity Maps (ordered by importance)",
                            show_label=True,
                            columns=3,
                            rows=2,
                            height="500px"
                        )
                    similarity_galleries.append((similarity_row, similarity_gallery, page_info))
            
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

        def refresh_colpali_interpretation(retrieved_results, current_query):
            """Refresh the ColPali interpretation with current query and results"""
            if not retrieved_results or "ColpaliRAG" not in retrieved_results:
                return "No ColPali results available - perform a retrieval first", []
            
            if not current_query:
                return "No query available", []
            
            colpali_images = retrieved_results.get("ColpaliRAG", [])
            
            return current_query, colpali_images

        def generate_similarity_maps_for_images(retrieved_results, current_query):
            """Generate similarity maps for all retrieved ColPali images"""
            if not similarity_mapper:
                return ["⚠️ Similarity mapper not available"] + [gr.DataFrame(visible=False)] + [[]] * 10 + [gr.Row(visible=False)] * 10 + [gr.Markdown(visible=False)] * 10

            if not retrieved_results or "ColpaliRAG" not in retrieved_results:
                return ["❌ No ColPali results available - perform a retrieval first"] + [gr.DataFrame(visible=False)] + [[]] * 10 + [gr.Row(visible=False)] * 10 + [gr.Markdown(visible=False)] * 10
            
            if not current_query:
                return ["❌ No query available"] + [gr.DataFrame(visible=False)] + [[]] * 10 + [gr.Row(visible=False)] * 10 + [gr.Markdown(visible=False)] * 10
            
            colpali_images = retrieved_results.get("ColpaliRAG", [])
            
            if not colpali_images:
                return ["❌ No images to analyze"] + [gr.DataFrame(visible=False)] + [[]] * 10 + [gr.Row(visible=False)] * 10 + [gr.Markdown(visible=False)] * 10
            
            # Update status
            status_text = f"🔄 Analyzing {len(colpali_images)} images for query: '{current_query}'..."
            
            # Generate similarity maps for each image
            try:
                results = analyze_multiple_images(similarity_mapper, colpali_images, current_query)
                
                # Prepare detailed status and token analysis
                token_analysis_data = []
                if results and len(results) > 0 and results[0]["success"]:
                    first_result = results[0]
                    token_info = []
                    
                    # Create token analysis table
                    for rank, token_data in enumerate(first_result.get("token_scores", []), 1):
                        token = token_data["token"]
                        sim = token_data["max_similarity"]
                        token_info.append(f"'{token}' (sim: {sim:.3f})")
                        token_analysis_data.append([token, f"{sim:.3f}", rank])
                    
                    token_summary = " | ".join(token_info)
                    status_text = f"""✅ **Analysis Complete for Query:** "{current_query}"

**🎯 Token Analysis Results:**
{token_summary}

**📊 Generated {first_result['num_visualizations']} similarity maps for {len(results)} images**

**💡 How to read:** Each visualization shows where the model focuses for each token. Higher similarity scores (closer to 1.0) indicate stronger attention. Images are ordered by token importance (highest similarity first)."""
                
                # Create token analysis DataFrame
                token_df = gr.DataFrame(
                    value=token_analysis_data,
                    headers=["Token", "Max Similarity", "Rank"],
                    visible=True
                )
                
                # Prepare outputs
                gallery_updates = []
                row_updates = []
                page_info_updates = []
                
                for i in range(10):  # Maximum 10 galleries
                    if i < len(results):
                        result = results[i]
                        if result["success"]:
                            # Convert base64 visualizations to images that Gradio can display
                            vis_images = []
                            for vis_b64 in result["visualizations"]:
                                # Create a temporary image from base64
                                img_data = base64.b64decode(vis_b64)
                                img = Image.open(io.BytesIO(img_data))
                                vis_images.append(img)
                            
                            gallery_updates.append(vis_images)
                            row_updates.append(gr.Row(visible=True))
                            
                            # Create detailed page info with token breakdown
                            token_breakdown = []
                            for rank, token_data in enumerate(result.get("token_scores", []), 1):
                                token = token_data["token"]
                                sim = token_data["max_similarity"]
                                token_breakdown.append(f"{rank}. **'{token}'** (similarity: {sim:.3f})")
                            
                            page_info_text = f"""### 📄 Page {i+1} - Token Similarity Analysis

**Query tokens (ranked by importance):**
{chr(10).join(token_breakdown)}

**Total visualizations:** {result['num_visualizations']}"""
                            
                            page_info_updates.append(gr.Markdown(value=page_info_text, visible=True))
                        else:
                            gallery_updates.append([])
                            row_updates.append(gr.Row(visible=False))
                            page_info_updates.append(gr.Markdown(visible=False))
                    else:
                        gallery_updates.append([])
                        row_updates.append(gr.Row(visible=False))
                        page_info_updates.append(gr.Markdown(visible=False))
                
                return [status_text] + [token_df] + gallery_updates + row_updates + page_info_updates
                
            except Exception as e:
                error_msg = f"❌ Error generating similarity maps: {str(e)}"
                return [error_msg] + [gr.DataFrame(visible=False)] + [[]] * 10 + [gr.Row(visible=False)] * 10 + [gr.Markdown(visible=False)] * 10

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
        )        # ColPali interpretation refresh handler
        refresh_interpretation_button.click(
            refresh_colpali_interpretation,
            inputs=[retrieved_results, current_query],
            outputs=[current_query_display, colpali_interpretation_gallery]
        )        # Similarity maps generation handler
        generate_similarity_button.click(
            generate_similarity_maps_for_images,
            inputs=[retrieved_results, current_query],
            outputs=[similarity_status] + [token_analysis] + [gallery for _, gallery, _ in similarity_galleries] + [row for row, _, _ in similarity_galleries] + [info for _, _, info in similarity_galleries]
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
            outputs=table_update_status,
        )

        refresh_interpretation_button.click(
            refresh_colpali_interpretation,
            inputs=[retrieved_results, current_query],
            outputs=[current_query_display, colpali_interpretation_gallery],
        )

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
