"""
ColPali Similarity Mapping Module

This module provides functionality to generate similarity maps for ColPali retrieval results.
It analyzes token-level attention and creates visualizations showing where the model
focuses on each image for different query tokens.
"""

import torch
import base64
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import warnings

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

try:
    from colpali_engine.interpretability import (
        get_similarity_maps_from_embeddings,
        plot_similarity_map,
        plot_all_similarity_maps,
    )
    from colpali_engine.utils.torch_utils import get_torch_device
    COLPALI_AVAILABLE = True
except ImportError:
    print("Warning: ColPali interpretability tools not available")
    COLPALI_AVAILABLE = False


class ColPaliSimilarityMapper:
    """
    A class to generate and visualize ColPali similarity maps for retrieved images.
    """
    
    def __init__(self, model, processor, device=None):
        """
        Initialize the similarity mapper.
        
        Args:
            model: ColPali model instance
            processor: ColPali processor instance
            device: Device to run computations on (auto-detected if None)
        """
        self.model = model
        self.processor = processor
        self.device = device or get_torch_device("auto")
        
    def base64_to_pil(self, base64_str: str) -> Image.Image:
        """Convert base64 string to PIL Image"""
        return Image.open(io.BytesIO(base64.b64decode(base64_str)))
    
    def pil_to_base64(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def generate_similarity_maps(self, image: Image.Image, query: str) -> Tuple[torch.Tensor, List[str], Dict]:
        """
        Generate similarity maps for ALL tokens in the query.
        
        Args:
            image: PIL Image to analyze
            query: Query string
            
        Returns:
            Tuple of (similarity_maps, query_tokens, metadata)
        """
        if not COLPALI_AVAILABLE:
            raise ImportError("ColPali interpretability tools not available")
        
        print(f"🔄 Processing query: '{query}'")
        
        # Preprocess inputs
        batch_images = self.processor.process_images([image]).to(self.device)
        batch_queries = self.processor.process_queries([query]).to(self.device)
        
        # Forward passes
        print("⚡ Computing embeddings...")
        with torch.no_grad():
            image_embeddings = self.model.forward(**batch_images)
            query_embeddings = self.model.forward(**batch_queries)
        
        # Get the number of image patches
        n_patches = self.processor.get_n_patches(
            image_size=image.size, 
            patch_size=self.model.patch_size
        )
        
        # Get the tensor mask to filter out embeddings not related to the image
        image_mask = self.processor.get_image_mask(batch_images)
        
        # Generate the similarity maps
        print("🎯 Generating similarity maps...")
        batched_similarity_maps = get_similarity_maps_from_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
            n_patches=n_patches,
            image_mask=image_mask,
        )
        
        # Get the similarity map for our input image
        similarity_maps = batched_similarity_maps[0]  # (query_length, n_patches_x, n_patches_y)
        
        # Tokenize the query
        query_tokens = self.processor.tokenizer.tokenize(query)
        
        print(f"✅ Generated similarity maps for {len(query_tokens)} tokens")
        
        # Create metadata
        metadata = {
            "image_size": image.size,
            "n_patches": n_patches,
            "similarity_shape": similarity_maps.shape,
            "max_similarity": similarity_maps.max().item(),
            "query": query,
            "num_tokens": len(query_tokens),
        }
        
        return similarity_maps, query_tokens, metadata
    
    def create_all_similarity_visualizations(
        self, 
        image: Image.Image, 
        similarity_maps: torch.Tensor, 
        query_tokens: List[str]
    ) -> List[str]:
        """
        Create similarity visualizations for all tokens using plot_all_similarity_maps.
        
        Args:
            image: Original PIL Image
            similarity_maps: Tensor of similarity maps for all tokens
            query_tokens: List of all query tokens
            
        Returns:
            List of base64-encoded PNG images
        """
        print(f"🎨 Creating visualizations for all {len(query_tokens)} tokens...")
        
        # Use plot_all_similarity_maps to generate all plots at once
        plots = plot_all_similarity_maps(
            image=image,
            query_tokens=query_tokens,
            similarity_maps=similarity_maps,
        )
        
        visualizations = []
        
        for idx, (fig, ax) in enumerate(plots):
            try:
                # Convert matplotlib figure to base64
                buffer = io.BytesIO()
                fig.savefig(buffer, format='PNG', bbox_inches='tight', dpi=100, facecolor='white')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                visualizations.append(img_base64)
                
                # Clean up to free memory
                plt.close(fig)
                buffer.close()
                
                token_text = query_tokens[idx] if idx < len(query_tokens) else f"Token_{idx}"
                max_sim = similarity_maps[idx].max().item() if idx < similarity_maps.shape[0] else 0.0
                print(f"  ✅ {idx+1}/{len(query_tokens)}: '{token_text}' (sim: {max_sim:.3f})")
                
            except Exception as e:
                print(f"  ❌ Error generating visualization for token {idx}: {str(e)}")
                continue
        
        print(f"🎉 Generated {len(visualizations)} visualizations successfully!")
        return visualizations
    
    def analyze_image_with_query(
        self, 
        image_input: Union[str, Image.Image], 
        query: str
    ) -> Dict:
        """
        Complete analysis pipeline for a single image - processes ALL tokens.
        
        Args:
            image_input: Base64-encoded image string OR PIL Image object
            query: Query string
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            print(f"🚀 Starting similarity analysis")
            
            # Handle both base64 strings and PIL Images
            if isinstance(image_input, str):
                # It's a base64 string
                image = self.base64_to_pil(image_input)
                print("📸 Converted base64 to PIL Image")
            elif isinstance(image_input, Image.Image):
                # It's already a PIL Image
                image = image_input
                print("📸 Using PIL Image directly")
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}. Expected str (base64) or PIL Image.")
            
            # Generate similarity maps for all tokens
            similarity_maps, query_tokens, metadata = self.generate_similarity_maps(image, query)
            
            # Create visualizations using plot_all_similarity_maps
            visualizations = self.create_all_similarity_visualizations(
                image, similarity_maps, query_tokens
            )
            
            # Calculate token importance scores for all tokens
            token_scores = []
            for i, token in enumerate(query_tokens):
                if i < similarity_maps.shape[0]:
                    max_sim = similarity_maps[i].max().item()
                    token_scores.append({
                        "token": token,
                        "index": i,
                        "max_similarity": max_sim
                    })
            
            # Sort by importance
            token_scores.sort(key=lambda x: x["max_similarity"], reverse=True)
            
            result = {
                "success": True,
                "visualizations": visualizations,
                "token_scores": token_scores,
                "metadata": metadata,
                "num_tokens": len(query_tokens),
                "num_visualizations": len(visualizations),
            }
            
            print(f"✅ Analysis complete: {len(visualizations)} visualizations for {len(query_tokens)} tokens")
            return result
            
        except Exception as e:
            error_msg = f"❌ Error in analysis: {str(e)}"
            print(error_msg)
            return {
                "success": False,
                "error": str(e),
                "visualizations": [],
                "token_scores": [],
                "metadata": {},
                "num_tokens": 0,
                "num_visualizations": 0,
            }


def create_similarity_mapper(model, processor):
    """
    Factory function to create a ColPaliSimilarityMapper instance.
    
    Args:
        model: ColPali model instance
        processor: ColPali processor instance
        
    Returns:
        ColPaliSimilarityMapper instance or None if creation fails
    """
    if not COLPALI_AVAILABLE:
        print("❌ Cannot create similarity mapper: ColPali interpretability tools not available")
        return None
    
    try:
        mapper = ColPaliSimilarityMapper(model, processor)
        print("✅ ColPali similarity mapper created successfully")
        return mapper
    except Exception as e:
        print(f"❌ Error creating similarity mapper: {e}")
        return None


def analyze_multiple_images(
    similarity_mapper: ColPaliSimilarityMapper,
    images: List[Union[str, Image.Image]],  # Can be base64 strings or PIL Images
    query: str
) -> List[Dict]:
    """
    Analysis of multiple images with the same query - processes ALL tokens.
    
    Args:
        similarity_mapper: ColPaliSimilarityMapper instance
        images: List of base64-encoded images OR PIL Image objects
        query: Query string
        
    Returns:
        List of analysis results for each image
    """
    if not similarity_mapper:
        print("❌ No similarity mapper provided")
        return [{"success": False, "error": "No similarity mapper available"}] * len(images)
    
    results = []
    
    print(f"\n🚀 Multi-Image Analysis Started")
    print(f"📊 Processing {len(images)} images")
    print(f"🔍 Query: '{query}'")
    print(f"🎯 Processing ALL tokens in query")
    print("-" * 60)
    
    for i, image_input in enumerate(images):
        print(f"\n📄 Processing Page {i+1}/{len(images)}...")
        
        # Determine image type for better error reporting
        if isinstance(image_input, str):
            print(f"📸 Input type: Base64 string (length: {len(image_input)})")
        elif isinstance(image_input, Image.Image):
            print(f"📸 Input type: PIL Image ({image_input.size}, {image_input.mode})")
        else:
            print(f"📸 Input type: Unknown ({type(image_input)})")
        
        result = similarity_mapper.analyze_image_with_query(image_input, query)
        result["image_index"] = i
        result["page_info"] = f"Page {i+1}"
        
        if result["success"]:
            print(f"✅ Page {i+1}: Generated {result['num_visualizations']} visualizations")
        else:
            print(f"❌ Page {i+1}: Failed - {result.get('error', 'Unknown error')}")
        
        results.append(result)
    
    # Summary
    successful = sum(1 for r in results if r.get("success", False))
    total_visualizations = sum(r.get("num_visualizations", 0) for r in results)
    
    print(f"\n🎉 Analysis Complete!")
    print(f"✅ Successfully processed: {successful}/{len(images)} pages")
    print(f"🎨 Total visualizations generated: {total_visualizations}")
    print("-" * 60)
    
    return results


# Example usage function that mimics your provided example
def example_usage():
    """
    Example showing how to use the refactored similarity mapper.
    This mimics the example you provided.
    """
    import torch
    from colpali_engine.models import ColPali, ColPaliProcessor
    from colpali_engine.utils.torch_utils import get_torch_device
    
    model_name = "vidore/colpali-v1.3"
    device = get_torch_device("auto")
    
    # Load the model
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    
    # Load the processor
    processor = ColPaliProcessor.from_pretrained(model_name)
    
    # Create similarity mapper
    similarity_mapper = create_similarity_mapper(model, processor)
    
    # Load the image and query
    image = Image.open("research_paper_gpu.png")
    query = "How many GPUs are used in the training of this model?"
    
    # Analyze the image with the query
    result = similarity_mapper.analyze_image_with_query(image, query)
    
    if result["success"]:
        print(f"Generated {result['num_visualizations']} similarity map visualizations")
        print(f"Token scores: {result['token_scores']}")
        
        # The visualizations are now in result["visualizations"] as base64 strings
        # You can save them or display them as needed
        for i, viz_base64 in enumerate(result["visualizations"]):
            # Convert back to PIL and save if needed
            viz_img = Image.open(io.BytesIO(base64.b64decode(viz_base64)))
            viz_img.save(f"similarity_map_{i}.png")
    else:
        print(f"Analysis failed: {result['error']}")