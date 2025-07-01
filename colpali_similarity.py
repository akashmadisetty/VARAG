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
from typing import List, Dict, Tuple, Optional
import warnings

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

try:
    from colpali_engine.interpretability import (
        get_similarity_maps_from_embeddings,
        plot_similarity_map,
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
        Generate similarity maps for a given image and query.
        
        Args:
            image: PIL Image to analyze
            query: Query string
            
        Returns:
            Tuple of (similarity_maps, query_tokens, metadata)
        """
        if not COLPALI_AVAILABLE:
            raise ImportError("ColPali interpretability tools not available")
            
        # Preprocess inputs
        batch_images = self.processor.process_images([image]).to(self.device)
        batch_queries = self.processor.process_queries([query]).to(self.device)
        
        # Forward passes
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
        
        # Create metadata
        metadata = {
            "image_size": image.size,
            "n_patches": n_patches,
            "similarity_shape": similarity_maps.shape,
            "max_similarity": similarity_maps.max().item(),
            "query": query
        }
        
        return similarity_maps, query_tokens, metadata
    
    def create_token_similarity_visualizations(
        self, 
        image: Image.Image, 
        similarity_maps: torch.Tensor, 
        query_tokens: List[str],
        max_tokens: Optional[int] = None
    ) -> List[str]:
        """
        Create similarity visualizations for each token and return as base64 strings.
        
        Args:
            image: Original PIL Image
            similarity_maps: Tensor of similarity maps
            query_tokens: List of query tokens
            max_tokens: Maximum number of tokens to visualize (None for all)
            
        Returns:
            List of base64-encoded PNG images
        """
        visualizations = []
        num_tokens = len(query_tokens)
        
        if max_tokens:
            num_tokens = min(num_tokens, max_tokens)
        
        for token_idx in range(num_tokens):
            if token_idx >= len(query_tokens):
                break
                
            try:
                # Get similarity map for this token
                current_similarity_map = similarity_maps[token_idx]
                max_sim = current_similarity_map.max().item()
                
                # Create the visualization
                fig, ax = plot_similarity_map(
                    image=image,
                    similarity_map=current_similarity_map,
                    figsize=(8, 8),
                    show_colorbar=True,
                )
                
                # Set title with token information
                token_text = query_tokens[token_idx]
                ax.set_title(
                    f"Token #{token_idx}: '{token_text}'\nMax Similarity: {max_sim:.3f}",
                    fontsize=14,
                    fontweight='bold'
                )
                
                # Save to base64
                buffer = io.BytesIO()
                fig.savefig(buffer, format='PNG', bbox_inches='tight', dpi=100)
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                visualizations.append(img_base64)
                
                # Clean up
                plt.close(fig)
                buffer.close()
                
            except Exception as e:
                print(f"Error generating visualization for token {token_idx}: {str(e)}")
                continue
        
        return visualizations
    
    def analyze_image_with_query(
        self, 
        base64_image: str, 
        query: str, 
        max_tokens: Optional[int] = None
    ) -> Dict:
        """
        Complete analysis pipeline for a single image.
        
        Args:
            base64_image: Base64-encoded image string
            query: Query string
            max_tokens: Maximum number of tokens to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Convert base64 to PIL
            image = self.base64_to_pil(base64_image)
            
            # Generate similarity maps
            similarity_maps, query_tokens, metadata = self.generate_similarity_maps(image, query)
            
            # Create visualizations
            visualizations = self.create_token_similarity_visualizations(
                image, similarity_maps, query_tokens, max_tokens
            )
            
            # Calculate token importance scores
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
            
            return {
                "success": True,
                "visualizations": visualizations,
                "token_scores": token_scores,
                "metadata": metadata,
                "num_tokens": len(query_tokens),
                "num_visualizations": len(visualizations)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "visualizations": [],
                "token_scores": [],
                "metadata": {},
                "num_tokens": 0,
                "num_visualizations": 0
            }


def create_similarity_mapper(model, processor):
    """
    Factory function to create a ColPaliSimilarityMapper instance.
    
    Args:
        model: ColPali model instance
        processor: ColPali processor instance
        
    Returns:
        ColPaliSimilarityMapper instance
    """
    return ColPaliSimilarityMapper(model, processor)


def analyze_multiple_images(
    similarity_mapper: ColPaliSimilarityMapper,
    base64_images: List[str],
    query: str,
    max_tokens_per_image: Optional[int] = None
) -> List[Dict]:
    """
    Analyze multiple images with the same query.
    
    Args:
        similarity_mapper: ColPaliSimilarityMapper instance
        base64_images: List of base64-encoded images
        query: Query string
        max_tokens_per_image: Maximum tokens to analyze per image
        
    Returns:
        List of analysis results for each image
    """
    results = []
    
    for i, base64_image in enumerate(base64_images):
        print(f"Analyzing image {i+1}/{len(base64_images)}...")
        result = similarity_mapper.analyze_image_with_query(
            base64_image, query, max_tokens_per_image
        )
        result["image_index"] = i
        results.append(result)
    
    return results
