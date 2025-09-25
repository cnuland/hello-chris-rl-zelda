"""Visual encoder for converting screen pixels to LLM-friendly format."""

import numpy as np
import base64
import io
from PIL import Image
from typing import Dict, Any, Optional


class VisualEncoder:
    """Encodes PyBoy screen data for LLM consumption."""
    
    def __init__(self, target_size: tuple = (160, 144)):
        """Initialize visual encoder.
        
        Args:
            target_size: Target image size for LLM (width, height)
        """
        self.target_size = target_size
        
    def encode_screen_for_llm(self, screen_array: np.ndarray) -> Dict[str, Any]:
        """Convert screen pixels to LLM-friendly format.
        
        Args:
            screen_array: RGB screen array from PyBoy (144, 160, 3)
            
        Returns:
            Dictionary with visual data for LLM
        """
        # Convert to PIL Image for processing
        image = Image.fromarray(screen_array.astype(np.uint8))
        
        # Resize if needed (Game Boy screen is small enough as-is)
        if image.size != self.target_size:
            image = image.resize(self.target_size, Image.NEAREST)
        
        # Convert to base64 for LLM API
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            'image_data': image_b64,
            'format': 'png',
            'size': self.target_size,
            'description': 'Current game screen showing Link, enemies, items, and environment'
        }
    
    def detect_visual_elements(self, screen_array: np.ndarray) -> Dict[str, Any]:
        """Basic computer vision to identify game elements.
        
        Args:
            screen_array: RGB screen array from PyBoy
            
        Returns:
            Detected visual elements
        """
        # This would need actual computer vision implementation
        # For now, return placeholder structure
        return {
            'detected_enemies': [],  # Could use template matching
            'visible_items': [],     # Detect item sprites
            'link_sprite': None,     # Link's current sprite/animation
            'ui_elements': {         # Health hearts, rupee counter, etc.
                'hearts_visible': 0,
                'rupee_display': None,
                'text_boxes': []
            }
        }
    
    def describe_screen_content(self, screen_array: np.ndarray) -> str:
        """Generate text description of screen for LLM context.
        
        Args:
            screen_array: RGB screen array from PyBoy
            
        Returns:
            Text description of what's visible
        """
        # Basic analysis - this could be much more sophisticated
        height, width, _ = screen_array.shape
        
        # Analyze color distribution to infer environment type
        avg_colors = np.mean(screen_array, axis=(0, 1))
        
        description_parts = []
        
        # Basic environment detection based on dominant colors
        if avg_colors[1] > avg_colors[0] and avg_colors[1] > avg_colors[2]:  # Green dominant
            description_parts.append("grassy outdoor area")
        elif avg_colors[2] > avg_colors[0] and avg_colors[2] > avg_colors[1]:  # Blue dominant
            description_parts.append("water or indoor area")
        else:
            description_parts.append("mixed terrain")
            
        # This is very basic - would need proper sprite recognition
        return f"Screen shows {', '.join(description_parts)} with various game elements visible"
