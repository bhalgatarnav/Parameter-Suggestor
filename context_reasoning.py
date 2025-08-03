import openai
import json
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ContextReasoner:
    def __init__(self, openai_api_key: str):
        self.api_key = openai_api_key
        openai.api_key = openai_api_key
        
    def analyze_context(
        self,
        material_info: Dict[str, Any],
        user_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate context-aware material constraints and recommendations
        """
        try:
            # Prepare material description
            material_desc = f"Primary material: {material_info['primary_material']} ({material_info['confidence']:.2%} confidence)\n"
            material_desc += "Semantic tags: " + ", ".join(material_info['semantic_tags']) + "\n"
            
            if user_context:
                material_desc += f"User context: {user_context}\n"
            
            # Enhanced prompt for better structured analysis
            prompt = f"""
            As a material science and 3D rendering expert, analyze this material and provide detailed recommendations:

            {material_desc}

            Provide a comprehensive analysis covering:

            1. Material Properties:
               - Physical characteristics (hardness, density, texture)
               - Optical properties (reflectivity, translucency, color response)
               - Environmental behavior (temperature response, moisture sensitivity)
               - Aging characteristics (weathering, patina development)

            2. Use Cases:
               - Primary applications (list 3-5 with reasoning)
               - Industry-specific uses
               - Design considerations
               - Limitations and constraints

            3. Manufacturing & Processing:
               - Recommended manufacturing methods
               - Surface treatment options
               - Quality control considerations
               - Cost implications
               - Sustainability aspects

            4. Rendering Guidelines:
               - Material setup (PBR parameters, texturing approach)
               - Lighting considerations (key, fill, rim light setups)
               - Camera settings (focal length, depth of field)
               - Post-processing requirements
               - Common pitfalls to avoid

            5. Alternative Materials:
               - Direct substitutes
               - Alternative options with different properties
               - Cost-effective alternatives
               - Sustainable alternatives

            Format the response as a detailed JSON object with these main categories and structured subcategories.
            Include numerical values where applicable (e.g., roughness ranges, recommended light intensities).
            """
            
            # Get GPT-4 response with higher temperature for more creative suggestions
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a combined material science expert and 3D rendering specialist providing detailed technical recommendations."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Parse and validate response
            try:
                recommendations = json.loads(response.choices[0].message.content)
                
                # Validate required sections
                required_sections = [
                    "material_properties",
                    "use_cases",
                    "manufacturing",
                    "rendering_guidelines",
                    "alternatives"
                ]
                
                for section in required_sections:
                    if section not in recommendations:
                        recommendations[section] = self._get_default_section(section)
                
                return recommendations
                
            except json.JSONDecodeError:
                logger.error("Failed to parse GPT-4 response as JSON")
                return self._get_default_recommendations()
            
        except Exception as e:
            logger.error(f"Error in context reasoning: {str(e)}")
            return self._get_default_recommendations()
    
    def _get_default_section(self, section: str) -> Dict[str, Any]:
        """Provide default values for missing sections"""
        defaults = {
            "material_properties": {
                "physical": {
                    "hardness": "medium",
                    "density": "medium",
                    "texture": "standard"
                },
                "optical": {
                    "reflectivity": 0.5,
                    "translucency": 0.0,
                    "color_response": "neutral"
                },
                "environmental": {
                    "temperature_sensitive": False,
                    "moisture_sensitive": False
                },
                "aging": {
                    "weathering_resistant": True,
                    "patina": "none"
                }
            },
            "use_cases": {
                "primary_applications": ["general purpose"],
                "industry_specific": [],
                "design_considerations": ["standard usage"],
                "limitations": ["none specified"]
            },
            "manufacturing": {
                "methods": ["standard processing"],
                "surface_treatments": ["none required"],
                "quality_control": ["visual inspection"],
                "cost_category": "medium",
                "sustainability": "moderate"
            },
            "rendering_guidelines": {
                "material_setup": {
                    "roughness": 0.5,
                    "metalness": 0.0,
                    "normal_strength": 1.0
                },
                "lighting": {
                    "setup": "three-point lighting",
                    "key_light": {"intensity": 1.0, "position": [1, 1, 1]},
                    "fill_light": {"intensity": 0.5, "position": [-1, 0.5, 1]},
                    "rim_light": {"intensity": 0.7, "position": [0, 1, -1]}
                },
                "camera": {
                    "focal_length": 50,
                    "f_stop": 5.6,
                    "distance": 5
                },
                "post_processing": ["standard color correction"]
            },
            "alternatives": {
                "direct_substitutes": ["similar materials"],
                "different_properties": [],
                "cost_effective": [],
                "sustainable": []
            }
        }
        
        return defaults.get(section, {})
    
    def _get_default_recommendations(self) -> Dict[str, Any]:
        """Return complete default recommendations"""
        return {
            section: self._get_default_section(section)
            for section in [
                "material_properties",
                "use_cases",
                "manufacturing",
                "rendering_guidelines",
                "alternatives"
            ]
        }
    
    def generate_rendering_constraints(
        self,
        material_info: Dict[str, Any],
        context_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate specific rendering constraints based on material and context
        """
        try:
            # Enhanced prompt for more detailed rendering constraints
            prompt = f"""
            As a 3D rendering expert, provide detailed rendering constraints for a {material_info['primary_material']} material
            with these characteristics:
            - Tags: {', '.join(material_info['semantic_tags'])}
            - Use case: {context_info['use_cases']['primary_applications'][0]}
            
            Provide specific technical constraints for:

            1. Lighting Setup:
               - Key light (position, intensity, color temperature, softness)
               - Fill light (position, intensity, color temperature, softness)
               - Rim light (position, intensity, color temperature, softness)
               - Environment lighting (HDRI selection, rotation, intensity)
               - Shadow settings (softness, density)

            2. Camera Settings:
               - Focal length
               - Aperture (f-stop)
               - Shutter speed
               - ISO
               - Distance from subject
               - Composition guidelines
               - Depth of field characteristics

            3. Material Settings:
               - Base color (RGB values)
               - Roughness range (min-max)
               - Metalness range (min-max)
               - Normal strength range
               - Displacement settings
               - Subsurface scattering (if applicable)
               - Clearcoat settings (if applicable)
               - Anisotropy settings (if applicable)

            4. Post-processing Requirements:
               - Color grading
               - Contrast adjustments
               - Sharpness settings
               - Bloom/glow settings
               - Ambient occlusion settings
               - Screen space reflections
               - Depth of field post-processing

            Format the response as a detailed JSON object with specific numerical values and ranges.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a technical 3D rendering expert providing specific numerical constraints and settings."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent technical specifications
                max_tokens=1000
            )
            
            try:
                constraints = json.loads(response.choices[0].message.content)
                return constraints
            except json.JSONDecodeError:
                logger.error("Failed to parse rendering constraints response")
                return self._get_default_rendering_constraints()
            
        except Exception as e:
            logger.error(f"Error generating rendering constraints: {str(e)}")
            return self._get_default_rendering_constraints()
    
    def _get_default_rendering_constraints(self) -> Dict[str, Any]:
        """Return default rendering constraints"""
        return {
            "lighting": {
                "key_light": {
                    "position": [1, 1, 1],
                    "intensity": 1.0,
                    "color_temperature": 6500,
                    "softness": 0.5
                },
                "fill_light": {
                    "position": [-1, 0.5, 1],
                    "intensity": 0.5,
                    "color_temperature": 6500,
                    "softness": 0.7
                },
                "rim_light": {
                    "position": [0, 1, -1],
                    "intensity": 0.7,
                    "color_temperature": 6500,
                    "softness": 0.3
                },
                "environment": {
                    "hdri": "studio_clear_sky",
                    "rotation": 0,
                    "intensity": 1.0
                },
                "shadows": {
                    "softness": 0.5,
                    "density": 1.0
                }
            },
            "camera": {
                "focal_length": 50,
                "aperture": 5.6,
                "shutter_speed": "1/125",
                "iso": 100,
                "distance": 5,
                "composition": "centered",
                "depth_of_field": "medium"
            },
            "material": {
                "base_color": [0.8, 0.8, 0.8],
                "roughness_range": [0.2, 0.8],
                "metalness_range": [0.0, 1.0],
                "normal_strength_range": [0.5, 1.5],
                "displacement": {
                    "scale": 0.1,
                    "midlevel": 0.5
                },
                "subsurface": {
                    "enabled": False,
                    "radius": [1.0, 1.0, 1.0]
                },
                "clearcoat": {
                    "enabled": False,
                    "roughness": 0.0
                },
                "anisotropic": {
                    "enabled": False,
                    "rotation": 0.0
                }
            },
            "post_processing": {
                "color_grading": {
                    "contrast": 1.0,
                    "saturation": 1.0,
                    "gamma": 1.0
                },
                "sharpness": 0.5,
                "bloom": {
                    "enabled": True,
                    "intensity": 0.5,
                    "threshold": 1.0
                },
                "ambient_occlusion": {
                    "enabled": True,
                    "intensity": 0.7
                },
                "screen_space_reflections": {
                    "enabled": True,
                    "quality": "medium"
                },
                "depth_of_field": {
                    "enabled": True,
                    "focal_distance": 5.0,
                    "aperture": 5.6
                }
            }
        }

# Initialize reasoner
context_reasoner = None  # Will be initialized with API key in main.py 