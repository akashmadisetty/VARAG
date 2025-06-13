import os
from typing import Optional, List, Union, Dict, Any
import warnings
import base64
from PIL import Image
import litellm
from io import BytesIO
from varag.vlms import BaseVLM

class LiteLLMVLM(BaseVLM):
    """
    A flexible wrapper for Vision Language Models using LiteLLM as the backend.
    Supports multiple providers including OpenAI, Anthropic, Groq, and others.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_images: int = 5,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the VLM wrapper.
        
        Args:
            model: The model identifier (e.g., "gpt-4o", "meta-llama/llama-4-scout-17b-16e-instruct")
            api_key: Optional API key (will use environment variables if not provided)
            api_base: Optional API base URL
            max_images: Maximum number of images to include in a single request
            verbose: Whether to print verbose logs
            **kwargs: Additional arguments to pass to litellm
            
        Raises:
            ValueError: If the model doesn't support vision
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.max_images = max_images
        self.verbose = verbose
        self.kwargs = kwargs
        
        # Set verbosity of litellm
        litellm.set_verbose = self.verbose
        
        # Check if model supports vision
        self._is_vision_model = litellm.supports_vision(model=self.model)
        if not self._is_vision_model:
            raise ValueError(
                f"Model '{model}' does not appear to support vision inputs. "
                "Please choose a vision-capable model."
            )
        
        # Validate environment and model access
        self._validate_setup()
        
    def _validate_setup(self) -> None:
        """Validate the environment setup and model access."""
        # Validate environment variables
        env_config = litellm.validate_environment(model=self.model)
        if not env_config["keys_in_environment"] and not self.api_key:
            raise ValueError(f"Missing API key for model {self.model}")
            
        # Validate model access
        if not litellm.check_valid_key(model=self.model, api_key=self.api_key):
            raise ValueError(f"Invalid API key for model {self.model}")
            
    def _encode_image(self, image: Image.Image) -> str:
        """Encode a PIL Image to base64."""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
            
    def _prepare_messages(self, query: str, images: List[Union[str, Image.Image]]) -> List[Dict[str, Any]]:
        """
        Prepare messages for the VLM API call.
        
        Args:
            query: The user query
            images: List of images (paths or PIL Images)
            
        Returns:
            List of message dictionaries for the API call
        """
        content = []
        
        # Add text content
        content.append({"type": "text", "text": query})
        
        # Add images (limited by max_images)
        image_count = 0
        for img in images:
            if image_count >= self.max_images:
                warnings.warn(f"Only using the first {self.max_images} images.")
                break
                
            if isinstance(img, str):
                # Assume it's a base64 string or file path
                if img.startswith(('http://', 'https://', 'data:')):
                    image_url = img
                else:
                    # It's a file path
                    with open(img, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode('utf-8')
                        image_url = f"data:image/jpeg;base64,{img_data}"
            else:
                # It's a PIL Image
                img_data = self._encode_image(img)
                image_url = f"data:image/jpeg;base64,{img_data}"
                
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
            image_count += 1
            
        return [{"role": "user", "content": content}]
        
    def query(
        self,
        query: str,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Query the vision model.
        
        Args:
            query: The query text
            images: Single image or list of images (paths or PIL Images)
            max_tokens: Maximum tokens in the response
            **kwargs: Additional arguments to pass to litellm
            
        Returns:
            The model's response text
        """
        # Handle single image case
        if not isinstance(images, list):
            images = [images]
            
        messages = self._prepare_messages(query=query, images=images)
        
        # Merge instance kwargs with method kwargs
        call_kwargs = {**self.kwargs, **kwargs}
        if max_tokens:
            call_kwargs["max_tokens"] = max_tokens
            
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                api_base=self.api_base,
                **call_kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error calling VLM API: {str(e)}")
            
    
            
    def __call__(
        self,
        query: str, 
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        **kwargs
    ) -> str:
        """
        Shorthand for query().
        
        Args:
            images: Single image or list of images (paths or PIL Images)
            query: The query text
            **kwargs: Additional arguments to pass to query()
            
        Returns:
            The model's response text
        """
        return self.query(query=query, images=images, **kwargs)