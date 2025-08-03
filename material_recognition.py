import torch
from PIL import Image
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class MaterialRecognizer:
    def __init__(self):
        """Initialize the material recognition system"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
            
        # Load models when needed to save memory
        self.material_model = None
        self.depth_model = None
        self.normal_model = None
    
    def recognize_material(self, image: Image.Image) -> Dict[str, Any]:
        """Main method for material recognition"""
        return self.analyze_image(image)
        
    def analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze an image and return material properties"""
        try:
            # Basic image preprocessing
            image = image.convert('RGB')
            image = image.resize((224, 224))
            image_array = np.array(image, dtype=np.uint8)  # Ensure uint8 type
            image_array_float = image_array.astype(np.float32) / 255.0  # Convert to float for analysis
            
            # Extract basic material properties
            material_analysis = self._analyze_material_type(image_array_float)
            surface_properties = self._analyze_surface_properties(image_array_float)
            depth_features = self._analyze_depth_features(image_array_float)
            
            # Generate rendering advice
            rendering_advice = self._generate_rendering_advice(
                material_analysis,
                surface_properties,
                depth_features
            )
            
            # Find similar materials
            suggested_materials = self._find_similar_materials(
                material_analysis["primary_material"],
                surface_properties
            )
            
            return {
                "material_analysis": material_analysis,
                "surface_properties": surface_properties,
                "depth_features": depth_features,
                "rendering_advice": rendering_advice,
                "suggested_materials": suggested_materials
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return self._generate_fallback_analysis()

    def _analyze_material_type(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Analyze the primary material type and confidence"""
        # For now, return mock analysis
        # In production, this would use a trained material classification model
        return {
            "primary_material": "metal",
            "confidence": 0.85,
            "secondary_materials": [
                {"material": "plastic", "confidence": 0.15},
                {"material": "ceramic", "confidence": 0.05}
            ]
        }
    
    def _analyze_surface_properties(self, image_array: np.ndarray) -> Dict[str, float]:
        """Analyze surface properties like roughness and metalness"""
        # Mock surface property analysis
        return {
            "roughness": 0.65,
            "metalness": 0.8,
            "normal_strength": 0.5,
            "displacement_scale": 0.3
        }
    
    def _analyze_depth_features(self, image_array: np.ndarray) -> Dict[str, float]:
        """Analyze depth and surface detail features"""
        # Mock depth analysis
        return {
            "surface_variation": 0.45,
            "depth_complexity": 0.3,
            "feature_density": 0.6,
            "pattern_regularity": 0.8
        }
    
    def _generate_rendering_advice(
        self,
        material_analysis: Dict[str, Any],
        surface_properties: Dict[str, float],
        depth_features: Dict[str, float]
    ) -> Dict[str, str]:
        """Generate rendering advice based on analysis"""
        return {
            "lighting_setup": "Use three-point lighting with strong rim light to emphasize surface detail",
            "camera_settings": "Consider macro shots to highlight surface texture",
            "post_processing": "Apply subtle ambient occlusion to enhance depth perception",
            "optimization_tips": "Bake textures for real-time applications"
        }
    
    def _find_similar_materials(
        self,
        primary_material: str,
        properties: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Find similar materials in the database"""
        # Mock similar materials
        return [
            {
                "name": "Brushed Aluminum",
                "similarity_score": 0.92,
                "key_properties": {
                    "roughness": 0.6,
                    "metalness": 0.9
                }
            },
            {
                "name": "Stainless Steel",
                "similarity_score": 0.85,
                "key_properties": {
                    "roughness": 0.4,
                    "metalness": 0.95
                }
            },
            {
                "name": "Chrome",
                "similarity_score": 0.78,
                "key_properties": {
                    "roughness": 0.2,
                    "metalness": 1.0
                }
            }
        ]
    
    def _generate_fallback_analysis(self) -> Dict[str, Any]:
        """Generate fallback analysis when processing fails"""
        return {
            "material_analysis": {
                "primary_material": "unknown",
                "confidence": 0.0,
                "secondary_materials": []
            },
            "surface_properties": {
                "roughness": 0.5,
                "metalness": 0.0,
                "normal_strength": 0.5,
                "displacement_scale": 0.0
            },
            "depth_features": {
                "surface_variation": 0.0,
                "depth_complexity": 0.0,
                "feature_density": 0.0,
                "pattern_regularity": 0.0
            },
            "rendering_advice": {
                "lighting_setup": "Use standard three-point lighting",
                "camera_settings": "Standard settings recommended",
                "post_processing": "No specific recommendations",
                "optimization_tips": "Follow general optimization guidelines"
            },
            "suggested_materials": []
        } 