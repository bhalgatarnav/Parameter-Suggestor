import torch
import numpy as np
from PIL import Image
import logging
from typing import Dict, Any, Optional, Tuple
import cv2

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        """Initialize the image processor"""
        self.logger = logger

    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """Process an image and extract material-relevant features"""
        try:
            # Convert PIL Image to numpy array
            image_array = np.array(image)
            
            # Extract basic image features
            basic_features = self._extract_basic_features(image_array)
            
            # Extract texture features
            texture_features = self._extract_texture_features(image_array)
            
            # Extract color features
            color_features = self._extract_color_features(image_array)
            
            return {
                "basic_features": basic_features,
                "texture_features": texture_features,
                "color_features": color_features
            }
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return self._generate_fallback_features()

    def _extract_basic_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract basic image features like brightness and contrast"""
        try:
            # Ensure image is in correct format
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = image.astype(np.uint8)
            
            # Calculate basic statistics
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            contrast = std_brightness / mean_brightness if mean_brightness > 0 else 0
            
            return {
                "brightness": float(mean_brightness / 255.0),
                "contrast": float(contrast),
                "uniformity": float(1.0 - std_brightness / 255.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting basic features: {str(e)}")
            return {
                "brightness": 0.5,
                "contrast": 0.5,
                "uniformity": 0.5
            }

    def _extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract texture-related features"""
        try:
            # Ensure image is in correct format
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = image.astype(np.uint8)
            
            # Calculate gradients
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Calculate texture metrics
            roughness = np.mean(gradient_magnitude) / 255.0
            variance = np.var(gradient_magnitude) / (255.0 ** 2)
            
            # Calculate local binary pattern or similar texture metric
            # This is a simplified version
            texture_pattern = cv2.blur(gray, (5,5)) - gray
            pattern_strength = np.mean(np.abs(texture_pattern)) / 255.0
            
            return {
                "roughness": float(roughness),
                "texture_variance": float(variance),
                "pattern_strength": float(pattern_strength)
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting texture features: {str(e)}")
            return {
                "roughness": 0.5,
                "texture_variance": 0.5,
                "pattern_strength": 0.5
            }
    
    def _extract_color_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract color-related features"""
        try:
            # Ensure image is in RGB
            if len(image.shape) != 3:
                return self._generate_fallback_color_features()
            
            # Calculate average color
            mean_color = np.mean(image, axis=(0,1))
            
            # Calculate color variance
            color_variance = np.var(image, axis=(0,1))
            
            # Calculate color histogram
            hist_r = np.histogram(image[:,:,0], bins=8, range=(0,256))[0]
            hist_g = np.histogram(image[:,:,1], bins=8, range=(0,256))[0]
            hist_b = np.histogram(image[:,:,2], bins=8, range=(0,256))[0]
            
            # Normalize histograms
            hist_r = hist_r / np.sum(hist_r)
            hist_g = hist_g / np.sum(hist_g)
            hist_b = hist_b / np.sum(hist_b)
            
            return {
                "mean_color": {
                    "r": float(mean_color[0] / 255.0),
                    "g": float(mean_color[1] / 255.0),
                    "b": float(mean_color[2] / 255.0)
                },
                "color_variance": {
                    "r": float(color_variance[0] / (255.0 ** 2)),
                    "g": float(color_variance[1] / (255.0 ** 2)),
                    "b": float(color_variance[2] / (255.0 ** 2))
                },
                "color_distribution": {
                    "r": hist_r.tolist(),
                    "g": hist_g.tolist(),
                    "b": hist_b.tolist()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting color features: {str(e)}")
            return self._generate_fallback_color_features()
    
    def _generate_fallback_features(self) -> Dict[str, Any]:
        """Generate fallback features when processing fails"""
        return {
            "basic_features": {
                "brightness": 0.5,
                "contrast": 0.5,
                "uniformity": 0.5
            },
            "texture_features": {
                "roughness": 0.5,
                "texture_variance": 0.5,
                "pattern_strength": 0.5
            },
            "color_features": self._generate_fallback_color_features()
        }
    
    def _generate_fallback_color_features(self) -> Dict[str, Any]:
        """Generate fallback color features"""
        return {
            "mean_color": {
                "r": 0.5,
                "g": 0.5,
                "b": 0.5
            },
            "color_variance": {
                "r": 0.0,
                "g": 0.0,
                "b": 0.0
            },
            "color_distribution": {
                "r": [0.125] * 8,
                "g": [0.125] * 8,
                "b": [0.125] * 8
            }
        } 