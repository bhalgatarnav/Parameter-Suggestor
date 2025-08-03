import openai
from typing import Dict, Any, List, Optional
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class CreativeExplorer:
    def __init__(self):
        """Initialize creative material explorer"""
        self.logger = logger
        self.design_moods = {
            "minimal": ["clean", "essential", "refined", "precise"],
            "brutalist": ["raw", "bold", "industrial", "exposed"],
            "surreal": ["unexpected", "dreamlike", "organic", "fluid"],
            "playful": ["vibrant", "dynamic", "energetic", "fun"],
            "meditative": ["calm", "natural", "balanced", "serene"],
            "futuristic": ["innovative", "sleek", "technological", "avant-garde"]
        }
        
        self.context_mappings = {
            "indoor": ["interior", "controlled-environment", "climate-controlled"],
            "outdoor": ["weatherproof", "exposed", "natural-aging"],
            "tactile": ["textured", "touchable", "physical-interaction"],
            "digital": ["screen-optimized", "virtual-reality", "rendered"],
            "functional": ["practical", "durable", "utilitarian"],
            "decorative": ["aesthetic", "ornamental", "visual-focus"]
        }
    
    async def generate_creative_insights(
        self,
        material_properties: Dict[str, Any],
        design_intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate creative insights and stories about the material"""
        try:
            # Prepare context for GPT
            prompt = self._create_creative_prompt(material_properties, design_intent)
            
            # Generate creative insights
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a creative material design collaborator with deep knowledge of 
                        architecture, product design, and material science. You think laterally and make 
                        unexpected connections. Your goal is to inspire and provoke new material directions."""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Higher temperature for more creative responses
                max_tokens=300
            )
            
            insights = response.choices[0].message.content
            
            # Generate material story
            story_response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a material storyteller who connects materials to their cultural, 
                        historical, and emotional contexts. Share engaging stories about how materials shape 
                        human experiences."""
                    },
                    {
                        "role": "user",
                        "content": f"Tell a brief, inspiring story about this material and its use in {design_intent.get('context', 'design')}: {material_properties['name']}"
                    }
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            story = story_response.choices[0].message.content
            
            # Generate hybrid suggestions
            hybrid_response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a material innovation expert who creates unexpected material 
                        combinations. Suggest creative hybrid materials that blend different properties 
                        and characteristics."""
                    },
                    {
                        "role": "user",
                        "content": f"Suggest 2-3 unexpected material combinations using {material_properties['name']} as a base, considering the mood: {design_intent.get('mood', 'neutral')}"
                    }
                ],
                temperature=0.8,
                max_tokens=150
            )
            
            hybrid_suggestions = hybrid_response.choices[0].message.content
            
            return {
                "creative_insights": insights,
                "material_story": story,
                "hybrid_suggestions": hybrid_suggestions,
                "design_recommendations": self._generate_design_recommendations(
                    material_properties,
                    design_intent
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error generating creative insights: {str(e)}")
            return self._generate_fallback_insights(material_properties)
    
    def _create_creative_prompt(
        self,
        material_properties: Dict[str, Any],
        design_intent: Dict[str, Any]
    ) -> str:
        """Create prompt for creative material exploration"""
        mood_keywords = self.design_moods.get(
            design_intent.get('mood', 'neutral'),
            ["balanced", "versatile"]
        )
        
        context_keywords = []
        for context in design_intent.get('context', []):
            context_keywords.extend(self.context_mappings.get(context, []))
        
        return f"""Explore creative possibilities for this material:

Material: {material_properties['name']}
Properties: {json.dumps(material_properties.get('properties', {}), indent=2)}
Design Intent: {design_intent.get('description', 'General design application')}
Mood: {', '.join(mood_keywords)}
Context: {', '.join(context_keywords)}

Provide creative insights covering:
1. Unexpected applications or combinations
2. Emotional and sensory qualities
3. Design opportunities and challenges
4. Creative material manipulations
5. Sustainability and future potential

Focus on inspiring new directions and unexpected connections.
Be specific about material properties while encouraging experimentation."""
    
    def _generate_design_recommendations(
        self,
        material_properties: Dict[str, Any],
        design_intent: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate specific design recommendations"""
        try:
            mood = design_intent.get('mood', 'neutral')
            mood_keywords = self.design_moods.get(mood, ["balanced", "versatile"])
            
            recommendations = []
            
            # Surface treatment recommendation
            if material_properties.get('properties', {}).get('roughness', 0.5) > 0.6:
                recommendations.append({
                    "type": "surface_treatment",
                    "title": "Surface Enhancement",
                    "description": f"Consider polishing or smoothing techniques to create contrast with the natural roughness, emphasizing the {mood_keywords[0]} quality"
                })
            
            # Light interaction recommendation
            if material_properties.get('properties', {}).get('metalness', 0) > 0.4:
                recommendations.append({
                    "type": "lighting",
                    "title": "Light Play",
                    "description": f"Experiment with directional lighting to enhance reflectivity and create {mood_keywords[1]} effects"
                })
            
            # Pattern recommendation
            if 'pattern' in str(material_properties.get('description', '')).lower():
                recommendations.append({
                    "type": "pattern",
                    "title": "Pattern Evolution",
                    "description": f"Consider scaling or rotating patterns to achieve a more {mood_keywords[2]} appearance"
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating design recommendations: {str(e)}")
            return []
    
    def _generate_fallback_insights(self, material_properties: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback insights when GPT fails"""
        return {
            "creative_insights": f"Explore the unique properties of {material_properties['name']} through experimental applications and surface treatments.",
            "material_story": f"{material_properties['name']} has been used in various design contexts, each bringing out different aspects of its character.",
            "hybrid_suggestions": f"Consider combining {material_properties['name']} with contrasting textures or complementary materials.",
            "design_recommendations": [
                {
                    "type": "general",
                    "title": "Experimental Approach",
                    "description": "Start with small-scale material experiments to understand behavior and possibilities."
                }
            ]
        } 