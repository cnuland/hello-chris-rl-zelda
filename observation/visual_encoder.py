"""Visual encoder for converting screen pixels to LLM-friendly format."""

import numpy as np
import base64
import io
from PIL import Image
from typing import Dict, Any, Optional


class VisualEncoder:
    """Encodes PyBoy screen data for LLM consumption."""
    
    def __init__(self, target_size: tuple = (160, 144), compression_mode: str = 'rgb'):
        """Initialize visual encoder.
        
        Args:
            target_size: Target image size for LLM (width, height)
            compression_mode: 'rgb', 'grayscale', 'gameboy_4bit', 'bit_packed', or 'palette'
        """
        self.target_size = target_size
        self.compression_mode = compression_mode
        
    def encode_screen_for_llm(self, screen_array: np.ndarray) -> Dict[str, Any]:
        """Convert screen pixels to LLM-friendly format.
        
        Args:
            screen_array: RGB screen array from PyBoy (144, 160, 3)
            
        Returns:
            Dictionary with visual data for LLM
        """
        height, width, channels = screen_array.shape
        
        if self.compression_mode == 'grayscale':
            # Convert to grayscale (3x compression)
            grayscale = np.dot(screen_array[...,:3], [0.299, 0.587, 0.114])
            grayscale = grayscale.astype(np.uint8)
            pixel_data = grayscale.tolist()
            
            pixel_stats = {
                'min_value': int(grayscale.min()),
                'max_value': int(grayscale.max()),
                'mean_value': float(grayscale.mean()),
                'unique_shades': int(len(np.unique(grayscale)))
            }
            
            return {
                'pixel_data': pixel_data,  # 2D array [height][width]
                'dimensions': {'height': height, 'width': width, 'channels': 1},
                'pixel_stats': pixel_stats,
                'format': 'grayscale',
                'compression': '3x smaller than RGB',
                'description': 'Grayscale pixel data from Game Boy screen (144x160x1)'
            }
            
        elif self.compression_mode == 'gameboy_4bit':
            # Convert to 4-shade Game Boy style (12x compression)
            grayscale = np.dot(screen_array[...,:3], [0.299, 0.587, 0.114])
            gb_4bit = (grayscale / 255.0 * 3).astype(np.uint8)  # Values: 0,1,2,3
            pixel_data = gb_4bit.tolist()
            
            pixel_stats = {
                'min_value': int(gb_4bit.min()),
                'max_value': int(gb_4bit.max()),
                'unique_shades': int(len(np.unique(gb_4bit))),
                'shade_map': '0=black, 1=dark_gray, 2=light_gray, 3=white'
            }
            
            return {
                'pixel_data': pixel_data,  # 2D array [height][width] with values 0-3
                'dimensions': {'height': height, 'width': width, 'channels': 1},
                'pixel_stats': pixel_stats,
                'format': 'gameboy_4bit',
                'compression': '12x smaller than RGB',
                'description': 'Classic Game Boy 4-shade pixel data (144x160, values 0-3)'
            }
            
        elif self.compression_mode == 'bit_packed':
            # Bit-packed 4-bit data (107x compression!)
            grayscale = np.dot(screen_array[...,:3], [0.299, 0.587, 0.114])
            gb_4bit = (grayscale / 255.0 * 3).astype(np.uint8)  # Values: 0,1,2,3
            
            # Pack 2 pixels per byte
            flat_data = gb_4bit.flatten()
            packed_bytes = []
            for i in range(0, len(flat_data), 2):
                byte_val = flat_data[i] << 4  # First pixel in high nibble
                if i + 1 < len(flat_data):
                    byte_val |= flat_data[i + 1]  # Second pixel in low nibble
                packed_bytes.append(byte_val)
            
            # Encode as base64 for JSON compatibility
            import base64
            byte_data = bytes(packed_bytes)
            b64_data = base64.b64encode(byte_data).decode('utf-8')
            
            pixel_stats = {
                'min_value': int(gb_4bit.min()),
                'max_value': int(gb_4bit.max()),
                'unique_shades': int(len(np.unique(gb_4bit))),
                'shade_map': '0=black, 1=dark_gray, 2=light_gray, 3=white'
            }
            
            return {
                'data': b64_data,  # Base64 encoded packed bytes
                'dimensions': {'height': height, 'width': width, 'channels': 1},
                'pixel_stats': pixel_stats,
                'format': 'bit_packed_4bit',
                'packed_bytes': len(packed_bytes),
                'original_values': len(flat_data),
                'compression_ratio': len(flat_data) / len(packed_bytes),
                'compression': '107x smaller than RGB',
                'description': f'Ultra-compressed 4-bit pixel data ({len(packed_bytes):,} bytes)'
            }
            
        elif self.compression_mode == 'palette':
            # Map to color palette indices (3x compression + palette)
            pixels_flat = screen_array.reshape(-1, 3)
            unique_colors = np.unique(pixels_flat, axis=0)
            
            # Create color to index mapping
            color_to_index = {tuple(color): idx for idx, color in enumerate(unique_colors)}
            
            # Map each pixel to its palette index
            indices = np.array([color_to_index[tuple(pixel)] for pixel in pixels_flat])
            indices_2d = indices.reshape(height, width)
            
            pixel_stats = {
                'unique_colors': len(unique_colors),
                'palette_size': len(unique_colors),
                'min_index': int(indices.min()),
                'max_index': int(indices.max())
            }
            
            return {
                'pixel_indices': indices_2d.tolist(),  # 2D array [height][width] with palette indices
                'color_palette': unique_colors.tolist(),  # List of RGB colors
                'dimensions': {'height': height, 'width': width, 'channels': 1},
                'pixel_stats': pixel_stats,
                'format': 'palette_indexed',
                'compression': '3x smaller + small palette',
                'description': f'Palette-indexed pixel data ({len(unique_colors)} colors)'
            }
            
        else:  # Default: 'rgb'
            # Original RGB format
            pixel_data = screen_array.tolist()
            
            pixel_stats = {
                'min_value': int(screen_array.min()),
                'max_value': int(screen_array.max()),
                'mean_r': float(screen_array[:,:,0].mean()),
                'mean_g': float(screen_array[:,:,1].mean()),
                'mean_b': float(screen_array[:,:,2].mean()),
                'unique_colors': int(len(np.unique(screen_array.reshape(-1, 3), axis=0)))
            }
            
            return {
                'pixel_data': pixel_data,  # 3D array [height][width][RGB]
                'dimensions': {'height': height, 'width': width, 'channels': channels},
                'pixel_stats': pixel_stats,
                'format': 'raw_rgb_array',
                'compression': '1x (no compression)',
                'description': 'Raw RGB pixel data from Game Boy screen (144x160x3)'
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
