from fastapi import FastAPI, Depends, HTTPException, status, Request, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from database import SessionLocal, engine, test_connection
from models import Material, FeedbackLog, Base
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import base64
from io import BytesIO
from PIL import Image
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
import uvicorn
import openai
import requests

# Import our components
from material_recognition import MaterialRecognizer
from image_processor import ImageProcessor
from material_visualization import MaterialVisualizer
from material_analysis_router import router as material_analysis_router
from utils.image_utils import safe_image_processing, standardize_api_response

# Test database connection
test_connection()

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Get configuration from environment variables
HOST = os.getenv('BACKEND_HOST', 'localhost')
PORT = int(os.getenv('BACKEND_PORT', '8000'))
FRONTEND_PORT = int(os.getenv('FRONTEND_PORT', '8501'))

# Initialize FastAPI
app = FastAPI(
    title="Material Recommendation API",
    description="AI-powered material recommendation system for 3D designers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

# Configure CORS
allowed_origins = [
    f"http://localhost:{FRONTEND_PORT}",
    f"http://127.0.0.1:{FRONTEND_PORT}",
    f"http://{HOST}:{FRONTEND_PORT}",
    "http://localhost:8501",  # Default Streamlit port
    "http://127.0.0.1:8501",
    "*"  # Allow all origins during development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
material_recognizer = MaterialRecognizer()
material_processor = ImageProcessor()

# Initialize material visualizer
material_visualizer = MaterialVisualizer()

# Create database tables
Base.metadata.create_all(bind=engine)

# Load embedding model (only once at startup)
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Successfully loaded embedding model")
except Exception as e:
    logger.error(f"Failed to load embedding model: {str(e)}")
    raise

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models for request/response validation
class FeedbackRequest(BaseModel):
    session_id: str
    material_id: int
    recommendation_rank: int
    interaction_type: str  # e.g., "click", "like", "dislike", "modify"
    user_rating: Optional[int] = None  # 1-5 scale
    modification_details: Optional[Dict] = None  # Any modifications made
    context_snapshot: Optional[Dict] = None  # Snapshot of recommendation context

class ImageAnalysisRequest(BaseModel):
    image: str  # base64 encoded
    analyze_texture: Optional[bool] = False
    extract_colors: Optional[bool] = False

class MaterialUpdateRequest(BaseModel):
    material_id: int
    updates: Dict  # Key-value pairs of properties to update

class RecommendationRequest(BaseModel):
    query: str
    filters: Optional[Dict] = None
    top_n: Optional[int] = 3

class RecommendationResponse(BaseModel):
    materials: List[Dict]
    session_id: str
    generated_at: datetime

# Helper functions
def calculate_similarity(query_embedding: np.ndarray, material_embedding: np.ndarray) -> float:
    """Calculate cosine similarity between query and material embeddings"""
    return cosine_similarity([query_embedding], [material_embedding])[0][0]

def apply_filters(query: Session, filters: Dict) -> Session:
    """Apply filters to SQLAlchemy query"""
    if not filters:
        return query
    
    if 'sustainability' in filters:
        query = query.filter(Material.sustainability_level >= filters['sustainability'])
    
    if 'material_class' in filters:
        query = query.filter(Material.material_class.ilike(f"%{filters['material_class']}%"))
    
    if 'physical_equivalent' in filters:
        query = query.filter(Material.has_physical_equivalent == filters['physical_equivalent'])
    
    if 'min_metalness' in filters:
        query = query.filter(Material.metalness >= filters['min_metalness'])
    
    if 'max_roughness' in filters:
        query = query.filter(Material.roughness <= filters['max_roughness'])
    
    return query

def material_to_dict(material: Material) -> Dict:
    """Convert Material object to dictionary with all relationships"""
    return {
        "id": material.id,
        "name": material.name,
        "metalness": material.metalness,
        "roughness": material.roughness,
        "key_shot_path": material.keyshot_category,
        "sustainability": material.sustainability_level,
        "manufacturability": material.manufacturability_notes,
        "physical_equivalent": material.has_physical_equivalent,
        "description": material.description,
        "appearance_tags": [tag.tag for tag in material.appearance_tags],
        "intent_tags": [tag.tag for tag in material.intent_tags],
        "use_cases": [uc.use_case for uc in material.suggested_use_cases],
        "environment_suitability": [env.env_condition for env in material.environment_suitability],
        "user_keywords": [kw.keyword for kw in material.user_keywords],
        "preview_image_path": material.preview_image_path,
        "texture_maps": [tmap.map_type for tmap in material.texture_map_types]
    }

def analyze_query_with_gpt(query: str) -> Dict[str, Any]:
    """Use GPT to analyze and structure the user's query with comprehensive material understanding"""
    try:
        prompt = f"""You are a materials science expert. Analyze this material query: "{query}"
        
        First, identify the primary material class. The query MUST contain one of these material classes:
        - plastic
        - metal
        - wood
        - glass
        - ceramic
        - composite
        - fabric
        - stone
        - concrete
        
        If the query mentions a specific material (e.g., 'oak' for wood, 'PET' for plastic), classify it under the appropriate primary class.
        If no material class is explicitly mentioned, infer it from the context and requirements.
        
        Provide a comprehensive analysis in this JSON format:
        {{
            "material_types": {{
                "primary": ["string"],  # MUST contain exactly one material from the above list
                "secondary": ["string"],  # Other compatible materials
                "alternatives": ["string"]  # Fallback options if primary/secondary not available
            }},
            "physical_properties": {{
                "roughness": {{"requirement": "high/medium/low", "range": [min, max]}},
                "metalness": {{"requirement": "high/medium/low", "range": [min, max]}},
                "transparency": {{"requirement": "high/medium/low", "value": float}},
                "surface_finish": ["string"]
            }},
            "visual_traits": {{
                "appearance": ["string"],
                "light_interaction": ["string"]
            }},
            "sustainability": {{
                "requirement_level": "high/medium/low",
                "eco_friendly": boolean,
                "recyclable": boolean
            }},
            "application": {{
                "use_case": ["string"],
                "environment": ["string"]
            }},
            "search_keywords": ["string"]
        }}
        
        For example, if the query is "biodegradable plastic for kitchentop":
        - Primary material MUST be "plastic"
        - Secondary materials could be ["composite"]
        - Alternatives could be ["ceramic", "stone"]
        - Sustainability should indicate eco_friendly: true
        
        Analyze the query thoroughly and ensure the primary material class is correctly identified."""
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a materials science expert with deep knowledge of material properties and classifications."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # Lower temperature for more consistent analysis
            max_tokens=1000
        )
        
        # Parse and validate the response
        analysis = json.loads(response.choices[0].message.content)
        
        # Validate primary material class
        if not analysis["material_types"]["primary"]:
            logger.warning("GPT analysis missing primary material class, using fallback")
            analysis["material_types"]["primary"] = ["plastic"]  # Safe fallback for most queries
        
        # Enhance analysis with derived properties
        analysis["derived_requirements"] = derive_additional_requirements(analysis)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in GPT query analysis: {str(e)}")
        # Return robust fallback analysis
        return create_fallback_analysis(query)

def derive_additional_requirements(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Derive additional material requirements based on primary analysis"""
    derived = {
        "property_ranges": {},
        "compatibility_requirements": [],
        "processing_requirements": [],
        "quality_indicators": []
    }
    
    try:
        # Derive property ranges
        if "roughness" in analysis.get("physical_properties", {}):
            roughness_req = analysis["physical_properties"]["roughness"]
            if roughness_req["requirement"] == "high":
                derived["property_ranges"]["roughness"] = [0.7, 1.0]
            elif roughness_req["requirement"] == "low":
                derived["property_ranges"]["roughness"] = [0.0, 0.3]
            else:
                derived["property_ranges"]["roughness"] = [0.3, 0.7]
        
        # Derive compatibility requirements
        if analysis.get("application", {}).get("environment"):
            for env in analysis["application"]["environment"]:
                if "outdoor" in env.lower():
                    derived["compatibility_requirements"].extend([
                        "UV resistant",
                        "Weather resistant",
                        "Temperature stable"
                    ])
                elif "industrial" in env.lower():
                    derived["compatibility_requirements"].extend([
                        "Chemical resistant",
                        "Wear resistant",
                        "Heat resistant"
                    ])
        
        # Derive processing requirements
        if analysis.get("sustainability", {}).get("eco_friendly"):
            derived["processing_requirements"].extend([
                "Low energy processing",
                "Non-toxic processing",
                "Minimal waste generation"
            ])
        
        # Derive quality indicators
        derived["quality_indicators"] = [
            f"Surface finish quality: {', '.join(analysis.get('physical_properties', {}).get('surface_finish', []))}",
            f"Visual consistency: {', '.join(analysis.get('visual_traits', {}).get('appearance', []))}",
            f"Performance metrics: {', '.join(analysis.get('application', {}).get('use_case', []))}"
        ]
        
        return derived
        
    except Exception as e:
        logger.error(f"Error deriving additional requirements: {str(e)}")
        return derived

def create_fallback_analysis(query: str) -> Dict[str, Any]:
    """Create robust fallback analysis when GPT analysis fails"""
    words = query.lower().split()
    
    # Basic material type detection
    material_types = {
        "metal": ["metal", "steel", "aluminum", "copper", "iron", "metallic"],
        "wood": ["wood", "timber", "wooden", "bamboo"],
        "plastic": ["plastic", "polymer", "pvc", "acrylic"],
        "glass": ["glass", "transparent", "translucent"],
        "ceramic": ["ceramic", "porcelain", "clay"],
        "fabric": ["fabric", "textile", "cloth", "fiber"]
    }
    
    # Property indicators
    property_indicators = {
        "rough": {"roughness": {"requirement": "high", "range": [0.7, 1.0]}},
        "smooth": {"roughness": {"requirement": "low", "range": [0.0, 0.3]}},
        "shiny": {"metalness": {"requirement": "high", "range": [0.7, 1.0]}},
        "matte": {"roughness": {"requirement": "high", "range": [0.7, 1.0]}},
        "transparent": {"transparency": {"requirement": "high", "value": 0.9}},
        "translucent": {"transparency": {"requirement": "medium", "value": 0.5}}
    }
    
    # Sustainability indicators
    sustainability_indicators = ["eco", "sustainable", "green", "recyclable", "biodegradable"]
    
    # Create basic analysis
    primary_material = None
    for mat_type, keywords in material_types.items():
        if any(word in words for word in keywords):
            primary_material = mat_type
            break
    
    # Fix the dictionary comprehension
    physical_props = {}
    for word in words:
        if word in property_indicators:
            physical_props.update(property_indicators[word])
    
    return {
        "material_types": {
            "primary": primary_material or "general",
            "secondary": [],
            "alternatives": []
        },
        "physical_properties": physical_props,
        "visual_traits": {
            "appearance": [word for word in words if word in ["glossy", "matte", "textured", "patterned"]],
            "effects": []
        },
        "sustainability": {
            "requirement_level": "high" if any(word in words for word in sustainability_indicators) else "medium",
            "eco_friendly": any(word in words for word in sustainability_indicators),
            "recyclable": "recyclable" in words
        },
        "search_keywords": words,
        "priority_factors": ["material_type", "physical_properties"]
    }

def generate_gpt_reasoning(
    query: str,
    query_analysis: Dict[str, Any],
    material: Material,
    match_reasons: Dict[str, float]
) -> str:
    """Generate comprehensive reasoning using GPT about material suitability"""
    try:
        # Prepare detailed material details
        material_details = {
            "name": material.name,
            "class": material.material_class,
            "properties": {
                "metalness": material.metalness,
                "roughness": material.roughness,
                "visual_behavior": material.visual_behavior,
                "sustainability": material.sustainability_level,
                "color_family": material.color_family,
                "physical_equivalent": material.has_physical_equivalent,
                "real_world_reference": material.real_world_reference
            },
            "appearance": {
                "visual_behavior": material.visual_behavior,
                "color": {
                    "family": material.color_family,
                    "rgb": [
                        material.color_r or 0.5,
                        material.color_g or 0.5,
                        material.color_b or 0.5
                    ]
                }
            },
            "manufacturing": {
                "notes": material.manufacturability_notes,
                "sustainability": material.sustainability_level
            },
            "applications": {
                "use_cases": [uc.use_case for uc in material.suggested_use_cases],
                "environment": [env.env_condition for env in material.environment_suitability],
                "keywords": [kw.keyword for kw in material.user_keywords]
            }
        }
        
        prompt = f"""As a materials expert, analyze why this material matches the user's requirements.

User Query: "{query}"

Material Properties:
{json.dumps(material_details, indent=2)}

Query Analysis:
{json.dumps(query_analysis, indent=2)}

Match Strengths:
{json.dumps(match_reasons, indent=2)}

Provide a detailed analysis covering:
1. Primary Match Analysis:
   - How well does this material match the core requirements?
   - What are the key matching properties?
   - Rate the match quality (excellent/good/fair) with specific reasons

2. Technical Properties:
   - Analyze metalness, roughness, and visual behavior
   - Compare with user's requirements
   - Highlight any notable technical advantages

3. Visual and Aesthetic Match:
   - Evaluate color and appearance alignment
   - Discuss visual behavior and light interaction
   - Note any unique visual characteristics

4. Practical Considerations:
   - Manufacturing feasibility
   - Environmental suitability
   - Real-world applications

5. Limitations and Trade-offs:
   - Identify any potential drawbacks
   - Suggest mitigations or alternatives
   - Note important considerations for implementation

Format the response as a clear, technical analysis. Be specific about numbers and measurements.
If certain aspects don't match well, explain why and provide context.
Keep the response focused and actionable."""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are a materials science expert specializing in analyzing material properties 
                    and matching them to specific requirements. Your analysis should be technical, precise, and actionable.
                    Always provide specific numbers and comparisons. If something doesn't match well, explain why and suggest alternatives."""
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        reasoning = response.choices[0].message.content.strip()
        
        # Calculate match confidence
        confidence_score = sum(match_reasons.values()) / len(match_reasons) if match_reasons else 0.5
        confidence_summary = f"\n\nMatch Confidence: {confidence_score:.0%}"
        
        return reasoning + confidence_summary

    except Exception as e:
        logger.error(f"Error generating GPT reasoning: {str(e)}")
        
        # Generate a more detailed fallback analysis based on material properties
        try:
            # Property match analysis
            roughness_match = "Excellent" if abs(material.roughness - 0.5) < 0.2 else "Good" if abs(material.roughness - 0.5) < 0.4 else "Limited"
            metalness_match = "Excellent" if abs(material.metalness - 0.5) < 0.2 else "Good" if abs(material.metalness - 0.5) < 0.4 else "Limited"
            
            # Visual characteristics
            visual_desc = material.visual_behavior if material.visual_behavior else "Standard visual properties"
            
            # Sustainability analysis
            sustainability = material.sustainability_level.capitalize() if material.sustainability_level else "Medium"
            
            # Use cases
            use_cases = [uc.use_case for uc in material.suggested_use_cases]
            use_case_text = ", ".join(use_cases[:3]) if use_cases else "General purpose"
            
            # Environment suitability
            environments = [env.env_condition for env in material.environment_suitability]
            env_text = ", ".join(environments[:3]) if environments else "Standard conditions"
            
            # Generate structured fallback response
            fallback = f"""Material Analysis:

1. Primary Match Evaluation:
   - Material Class: {material.material_class}
   - Overall Suitability: {roughness_match.lower() if roughness_match == "Limited" else "Good"} match for requirements
   - Key Strength: {material.name} properties align with {use_case_text}

2. Technical Properties:
   - Roughness: {roughness_match} match (Value: {material.roughness:.2f})
   - Metalness: {metalness_match} match (Value: {material.metalness:.2f})
   - Visual Behavior: {visual_desc}

3. Environmental & Manufacturing:
   - Sustainability Rating: {sustainability}
   - Suitable Environments: {env_text}
   - Manufacturing Notes: {material.manufacturability_notes if material.manufacturability_notes else "Standard manufacturing process"}

4. Implementation Considerations:
   - Physical Reference Available: {"Yes" if material.has_physical_equivalent else "No"}
   - Real-world Reference: {material.real_world_reference if material.real_world_reference else "Not specified"}
   - Color Family: {material.color_family if material.color_family else "Standard"}

5. Recommendations:
   - Best Used For: {use_case_text}
   - Environmental Conditions: {env_text}
   - Consider adjusting {', '.join([p for p, m in [('roughness', roughness_match), ('metalness', metalness_match)] if m == "Limited"])} if critical to application

Match Confidence: {int(sum([
    1.0 if m == "Excellent" else 0.7 if m == "Good" else 0.3
    for m in [roughness_match, metalness_match]
]) / 2 * 100)}%"""
            
            return fallback
            
        except Exception as fallback_error:
            logger.error(f"Error generating fallback analysis: {str(fallback_error)}")
            return """Material Analysis:
1. Primary Match: Basic material properties align with requirements
2. Technical Properties: Standard material specifications
3. Visual Characteristics: Default material appearance
4. Practical Aspects: General-purpose application
5. Important Notes: See technical specifications for details

Match Confidence: Calculating..."""

def calculate_match_confidence(match_reasons: Dict[str, float]) -> float:
    """Calculate overall match confidence based on match reasons"""
    try:
        # Define weights for different match types
        weights = {
            "semantic_similarity": 0.3,
            "material_class_match": 0.2,
            "roughness_match": 0.1,
            "metalness_match": 0.1,
            "visual_match": 0.1,
            "sustainability_match": 0.1,
            "tag_matches": 0.1
        }
        
        # Calculate weighted score
        total_weight = 0
        weighted_score = 0
        
        for reason, score in match_reasons.items():
            if reason in weights:
                weighted_score += score * weights[reason]
                total_weight += weights[reason]
        
        if total_weight > 0:
            return weighted_score / total_weight
        return 0.5
        
    except Exception as e:
        logger.error(f"Error calculating match confidence: {str(e)}")
        return 0.5

def generate_fallback_reasoning(material: Material, query_analysis: Dict[str, Any]) -> str:
    """Generate basic reasoning when GPT reasoning fails"""
    try:
        parts = []
        
        # Material type match
        if material.material_class and query_analysis.get("material_types", {}).get("primary"):
            if query_analysis["material_types"]["primary"] in material.material_class.lower():
                parts.append(f"This {material.material_class} material matches your primary material requirement.")
        
        # Physical properties
        props = []
        if material.roughness is not None:
            props.append(f"roughness of {material.roughness:.2f}")
        if material.metalness is not None:
            props.append(f"metalness of {material.metalness:.2f}")
        if props:
            parts.append(f"It has physical properties including {', '.join(props)}.")
        
        # Sustainability
        if material.sustainability_level:
            parts.append(f"The material has {material.sustainability_level} sustainability rating.")
        
        # Use cases
        use_cases = [uc.use_case for uc in material.suggested_use_cases]
        if use_cases:
            parts.append(f"Suitable for: {', '.join(use_cases)}.")
        
        # Manufacturing
        if material.manufacturability_notes:
            parts.append(f"Manufacturing note: {material.manufacturability_notes}")
        
        return " ".join(parts)
        
    except Exception as e:
        logger.error(f"Error generating fallback reasoning: {str(e)}")
        return f"This {material.material_class} material matches some of your requirements based on its properties."

def validate_property_constraints(material_class: str, properties: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
    """Validate and adjust property constraints based on material class"""
    warnings = []
    adjusted_props = properties.copy()
    
    # Define material-specific property ranges
    MATERIAL_RANGES = {
        "wood": {
            "roughness": (0.35, 0.85),
            "metalness": (0.0, 0.15),
            "normal_strength": (0.4, 0.9)
        },
        "metal": {
            "roughness": (0.1, 0.7),
            "metalness": (0.7, 1.0),
            "normal_strength": (0.2, 0.8)
        },
        "plastic": {
            "roughness": (0.2, 0.8),
            "metalness": (0.0, 0.3),
            "normal_strength": (0.3, 0.7)
        },
        "glass": {
            "roughness": (0.05, 0.3),
            "metalness": (0.0, 0.2),
            "normal_strength": (0.1, 0.5)
        },
        "ceramic": {
            "roughness": (0.2, 0.6),
            "metalness": (0.0, 0.2),
            "normal_strength": (0.3, 0.8)
        }
    }
    
    if material_class.lower() in MATERIAL_RANGES:
        ranges = MATERIAL_RANGES[material_class.lower()]
        
        for prop, (min_val, max_val) in ranges.items():
            if prop in adjusted_props:
                orig_val = adjusted_props[prop]
                
                # If value is outside range, clamp it and add warning
                if orig_val < min_val:
                    adjusted_props[prop] = min_val
                    warnings.append(
                        f"{material_class.title()} materials typically have {prop} values "
                        f"no lower than {min_val:.2f}. Adjusted from {orig_val:.2f}."
                    )
                elif orig_val > max_val:
                    adjusted_props[prop] = max_val
                    warnings.append(
                        f"{material_class.title()} materials typically have {prop} values "
                        f"no higher than {max_val:.2f}. Adjusted from {orig_val:.2f}."
                    )
    
    return adjusted_props, warnings

def get_property_match_score(material: Material, properties: Dict[str, float], material_class: str) -> Tuple[float, List[str]]:
    """Calculate how well a material matches the desired properties"""
    score = 1.0
    reasons = []
    
    # Validate and adjust properties for material class
    adjusted_props, warnings = validate_property_constraints(material_class, properties)
    
    # Weight factors for different properties
    WEIGHTS = {
        "roughness": 0.4,
        "metalness": 0.3,
        "normal_strength": 0.3
    }
    
    # Calculate property match scores
    for prop, weight in WEIGHTS.items():
        if prop in adjusted_props and hasattr(material, prop):
            desired_val = adjusted_props[prop]
            actual_val = getattr(material, prop)
            
            # Calculate normalized difference
            diff = abs(desired_val - actual_val)
            prop_score = max(0, 1 - (diff / 0.5))  # 0.5 is the maximum meaningful difference
            
            # Apply weight to score
            score *= (1 - weight + (weight * prop_score))
            
            # Add reasoning
            if prop_score > 0.8:
                reasons.append(f"Excellent {prop} match")
            elif prop_score > 0.6:
                reasons.append(f"Good {prop} match")
            elif prop_score > 0.4:
                reasons.append(f"Moderate {prop} match")
            else:
                reasons.append(f"Limited {prop} match")
    
    return score, reasons, warnings

# Core recommendation function
def recommend_materials(
    query: str, 
    db: Session, 
    filters: Dict[str, Any],
    embeddings: Optional[np.ndarray] = None,
    top_n: int = 3
) -> List[Dict]:
    try:
        # Start with base query
        materials_query = db.query(Material)
        
        # Apply filters
        if filters.get("roughness"):
            materials_query = materials_query.filter(
                Material.roughness.between(
                    float(filters["roughness"]) - 0.2,
                    float(filters["roughness"]) + 0.2
                )
            )
        
        if filters.get("metalness"):
            materials_query = materials_query.filter(
                Material.metalness.between(
                    float(filters["metalness"]) - 0.2,
                    float(filters["metalness"]) + 0.2
                )
            )
        
        if filters.get("semantic_tags"):
            for tag in filters["semantic_tags"]:
                materials_query = materials_query.filter(
                    Material.appearance_tags.any(tag.lower())
                )
        
        # Get materials
        materials = materials_query.all()
        
        # Calculate similarities
        similarities = []
        for material in materials:
            try:
                # Combine text and embedding similarity if available
                text_similarity = 0.5  # Default similarity if no embedding
                if material.embedding:
                    try:
                        text_similarity = calculate_similarity(
                            model.encode(query),
                            np.array(json.loads(material.embedding))
                        )
                    except Exception as e:
                        logger.warning(f"Error calculating text similarity for material {material.id}: {str(e)}")
                
                embedding_similarity = 0.0
                if embeddings is not None and material.visual_embedding:
                    try:
                        embedding_similarity = calculate_similarity(
                            embeddings,
                            np.array(json.loads(material.visual_embedding))
                        )
                    except Exception as e:
                        logger.warning(f"Error calculating visual similarity for material {material.id}: {str(e)}")
                
                # Weighted combination
                final_similarity = 0.7 * text_similarity + 0.3 * embedding_similarity
                similarities.append((material, final_similarity))
            except Exception as e:
                logger.warning(f"Error processing material {material.id}: {str(e)}")
                continue
        
        # Sort and prepare results
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        
        for material, score in similarities[:top_n]:
            material_data = material_to_dict(material)
            material_data["similarity_score"] = float(score)
            results.append(material_data)
        
        return results
    
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        raise

def get_diverse_recommendations(
    materials: List[Material],
    query: str,
    properties: Optional[Dict[str, float]] = None,
    top_n: int = 3
) -> List[Dict]:
    """Get diverse recommendations using GPT for query understanding and property matching"""
    try:
        # Analyze query using GPT
        query_analysis = analyze_query_with_gpt(query)
        
        # Extract material class from query
        primary_material_class = None
        if query_analysis["material_types"]["primary"]:
            primary_material_class = query_analysis["material_types"]["primary"][0].lower()
        
        # Filter materials by class first
        filtered_materials = []
        for material in materials:
            if not material.material_class:
                continue
                
            material_class_lower = material.material_class.lower()
            
            # Check if material class matches query
            if primary_material_class and primary_material_class in material_class_lower:
                filtered_materials.append(material)
            # If no materials found with primary class, check secondary classes
            elif not filtered_materials and query_analysis["material_types"]["secondary"]:
                for secondary_class in query_analysis["material_types"]["secondary"]:
                    if secondary_class.lower() in material_class_lower:
                        filtered_materials.append(material)
        
        # If no materials found in primary or secondary classes, use alternatives
        if not filtered_materials and query_analysis["material_types"]["alternatives"]:
            for material in materials:
                if not material.material_class:
                    continue
                material_class_lower = material.material_class.lower()
                for alt_class in query_analysis["material_types"]["alternatives"]:
                    if alt_class.lower() in material_class_lower:
                        filtered_materials.append(material)
        
        # If still no materials found, use all materials (but this shouldn't happen with proper GPT analysis)
        if not filtered_materials:
            logger.warning(f"No materials found for class {primary_material_class}, using all materials")
            filtered_materials = materials
        
        # Calculate similarities and gather metadata
        material_scores = []
        seen_classes = set()
        property_warnings = []
        
        for material in filtered_materials:
            try:
                # Initialize match reasons
                match_reasons = {}
                
                # Calculate base similarity using embeddings
                base_similarity = 0.5  # Default score if no embedding
                if material.embedding:
                    try:
                        query_embedding = model.encode(query)
                        mat_embedding = np.array(json.loads(material.embedding))
                        base_similarity = cosine_similarity([query_embedding], [mat_embedding])[0][0]
                        match_reasons["semantic_similarity"] = float(base_similarity)
                    except Exception as embed_error:
                        logger.warning(f"Error calculating embedding similarity: {str(embed_error)}")
                
                # Calculate property-based scores if properties provided
                property_score = 1.0
                property_reasons = []
                if properties and material.material_class:
                    property_score, reasons, warnings = get_property_match_score(
                        material,
                        properties,
                        material.material_class
                    )
                    property_reasons.extend(reasons)
                    property_warnings.extend(warnings)
                    match_reasons["property_match"] = float(property_score)
                
                # Visual traits match
                if material.visual_behavior:
                    for trait in query_analysis["visual_traits"]["appearance"]:
                        if trait.lower() in material.visual_behavior.lower():
                            property_score *= 1.15
                            match_reasons["visual_match"] = 1.15
                
                # Sustainability match (important for biodegradable queries)
                sustainability_score = 1.0
                if "biodegradable" in query.lower() or query_analysis["sustainability"].get("eco_friendly"):
                    if material.sustainability_level == "high":
                        sustainability_score = 1.5
                        match_reasons["sustainability_match"] = 1.5
                    elif material.sustainability_level == "medium":
                        sustainability_score = 1.25
                        match_reasons["sustainability_match"] = 1.25
                
                # Tag matches
                matching_tags = [
                    tag.tag for tag in material.appearance_tags 
                    if any(kw.lower() in tag.tag.lower() for kw in query_analysis["search_keywords"])
                ]
                if matching_tags:
                    property_score *= (1.0 + len(matching_tags) * 0.1)
                    match_reasons["tag_matches"] = len(matching_tags)
                
                # Apply diversity penalty
                diversity_penalty = 0.8 if material.material_class in seen_classes else 1.0
                
                # Calculate final score - weighted combination of all factors
                final_score = (
                    base_similarity * 0.3 +      # Semantic relevance
                    property_score * 0.3 +       # Property match
                    sustainability_score * 0.4    # Sustainability/biodegradability
                ) * diversity_penalty
                
                # Generate reasoning using GPT
                try:
                    reasoning = generate_gpt_reasoning(
                        query=query,
                        query_analysis=query_analysis,
                        material=material,
                        match_reasons=match_reasons
                    )
                    if property_reasons:
                        reasoning += "\n\nProperty Analysis:\n- " + "\n- ".join(property_reasons)
                except Exception as e:
                    logger.warning(f"Error generating reasoning for material {material.id}: {str(e)}")
                    reasoning = generate_fallback_reasoning(material, query_analysis)
                
                # Gather all material metadata
                material_data = {
                "id": material.id,
                "name": material.name,
                "key_shot_path": material.keyshot_category,
                    "material_class": material.material_class,
                "properties": {
                        "metalness": float(material.metalness),
                        "roughness": float(material.roughness),
                        "color": {
                            "r": float(material.color_r) if material.color_r else 0.5,
                            "g": float(material.color_g) if material.color_g else 0.5,
                            "b": float(material.color_b) if material.color_b else 0.5
                        },
                        "specular": {
                            "r": float(material.specular_r) if material.specular_r else 0.5,
                            "g": float(material.specular_g) if material.specular_g else 0.5,
                            "b": float(material.specular_b) if material.specular_b else 0.5
                        },
                        "sustainability": material.sustainability_level,
                        "physical_equivalent": bool(material.has_physical_equivalent),
                        "visual_behavior": material.visual_behavior,
                        "color_family": material.color_family,
                        "manufacturability": material.manufacturability_notes,
                        "appearance_tags": [tag.tag for tag in material.appearance_tags],
                        "intent_tags": [tag.tag for tag in material.intent_tags],
                        "texture_maps": [map_type.map_type for map_type in material.texture_map_types],
                        "use_cases": [uc.use_case for uc in material.suggested_use_cases],
                        "environment": [env.env_condition for env in material.environment_suitability],
                        "keywords": [kw.keyword for kw in material.user_keywords]
                    },
                    "preview_image": material.preview_image_path,
                    "similarity_score": float(final_score),
                    "reasoning": reasoning,
                    "match_reasons": match_reasons,
                    "property_warnings": property_warnings if property_warnings else None
                }
                
                material_scores.append(material_data)
                seen_classes.add(material.material_class)
                
            except Exception as material_error:
                logger.warning(f"Error processing material {material.id}: {str(material_error)}")
                continue
        
        # Sort by score and get top N diverse results
        material_scores.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return material_scores[:top_n]
        
    except Exception as e:
        logger.error(f"Error in get_diverse_recommendations: {str(e)}")
        raise

@app.post("/recommend")
async def get_recommendations(request: Request):
    """Endpoint for getting material recommendations"""
    db = SessionLocal()
    try:
        # Parse JSON data from request
        data = await request.json()
        query = data.get("query", "").lower()
        properties = data.get("properties", {})
        
        # Get all materials from database
        materials = db.query(Material).all()
        
        # Get recommendations with GPT-powered analysis and property matching
        results = get_diverse_recommendations(materials, query, properties)
        
        # Extract any property warnings
        warnings = []
        for result in results:
            if result.get("property_warnings"):
                warnings.extend(result["property_warnings"])
        
        return JSONResponse(content={
            "materials": results,
            "warnings": list(set(warnings))  # Remove duplicates
        })
        
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
    
@app.post("/feedback/", status_code=status.HTTP_201_CREATED)
async def log_feedback(
    feedback: FeedbackRequest, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Endpoint for logging user feedback on recommendations"""
    try:
        # Create new feedback log entry
        db_feedback = FeedbackLog(
            session_id=feedback.session_id,
            material_id=feedback.material_id,
            recommendation_rank=feedback.recommendation_rank,
            interaction_type=feedback.interaction_type,
            user_rating=feedback.user_rating,
            modification_details=json.dumps(feedback.modification_details) if feedback.modification_details else None,
            context_snapshot=json.dumps(feedback.context_snapshot) if feedback.context_snapshot else None,
            created_at=datetime.now()
        )
        
        # Add and commit to database
        db.add(db_feedback)
        db.commit()
        db.refresh(db_feedback)
        
        # Log feedback for analysis
        logger.info(
            f"Feedback logged - Session: {feedback.session_id}, "
            f"Material: {feedback.material_id}, "
            f"Type: {feedback.interaction_type}, "
            f"Rating: {feedback.user_rating}"
        )
        
        # Update material ratings in background
        background_tasks.add_task(
            update_material_ratings,
            db=db,
            material_id=feedback.material_id,
            rating=feedback.user_rating
        )
        
        return {
            "status": "success",
            "feedback_id": db_feedback.id,
            "message": "Feedback logged successfully"
        }
    
    except Exception as e:
        db.rollback()
        logger.error(f"Feedback logging error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log feedback: {str(e)}"
        )

async def update_material_ratings(db: Session, material_id: int, rating: Optional[int]):
    """Update material ratings based on user feedback"""
    try:
        if rating is None:
            return
            
        material = db.query(Material).filter(Material.id == material_id).first()
        if not material:
            return
            
        # Update rating metrics
        material.total_ratings = (material.total_ratings or 0) + 1
        material.average_rating = (
            ((material.average_rating or 0) * (material.total_ratings - 1) + rating)
            / material.total_ratings
        )
        
        db.commit()
        logger.info(f"Updated ratings for material {material_id}")
        
    except Exception as e:
        logger.error(f"Error updating material ratings: {str(e)}")
        db.rollback()

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze uploaded image with robust error handling"""
    try:
        # Read file content
        content = await file.read()
        
        # Process image safely
        pil_image, cv2_image = safe_image_processing(content)
        
        # Initialize recognizer
        recognizer = MaterialRecognizer()
        
        # Analyze image
        analysis_result = recognizer.analyze_image(pil_image)
        
        return standardize_api_response(
            success=True,
            data=analysis_result
        )
        
    except ValueError as e:
        logger.error(f"Image validation error: {str(e)}")
        return standardize_api_response(
            success=False,
            error={"code": "VALIDATION_ERROR", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        return standardize_api_response(
            success=False,
            error={"code": "PROCESSING_ERROR", "message": "Failed to analyze image"}
        )

@app.get("/materials/{material_id}")
async def get_material(
    material_id: int, 
    db: Session = Depends(get_db)
):
    """Endpoint to get detailed information about a specific material"""
    try:
        material = db.query(Material).filter(Material.id == material_id).first()
        if not material:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Material not found"
            )
        
        return material_to_dict(material)
        
    except Exception as e:
        logger.error(f"Material fetch error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch material"
        )

@app.post("/materials/update/")
async def update_material(
    request: MaterialUpdateRequest,
    db: Session = Depends(get_db)
):
    """Endpoint to update material properties"""
    try:
        material = db.query(Material).filter(Material.id == request.material_id).first()
        if not material:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Material not found"
            )
        
        # Apply updates
        for key, value in request.updates.items():
            if hasattr(material, key):
                setattr(material, key, value)
        
        db.commit()
        logger.info(f"Updated material {request.material_id}")
        
        return {"status": "success", "updated_fields": list(request.updates.keys())}
    
    except Exception as e:
        db.rollback()
        logger.error(f"Material update error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update material"
        )

@app.post("/visualize-material")
async def visualize_material(request: Request):
    """Generate material maps and preview using Stable Diffusion XL"""
    try:
        data = await request.json()
        material_properties = data.get("material_properties", {})
        geometry = data.get("geometry", "flat_surface")
        lighting = data.get("lighting", "studio")
        size = data.get("size", [1024, 1024])
        
        # Process visualization request
        map_urls = material_visualizer.process_material_request(
            material_properties=material_properties,
            geometry=geometry,
            lighting=lighting,
            size=tuple(size)
        )
        
        return JSONResponse(content={
            "status": "success",
            "maps": map_urls
        })
        
    except Exception as e:
        logger.error(f"Material visualization error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate material visualization: {str(e)}"
        )

# Error handler for CORS issues
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 403:
        # Check if the request origin is in allowed origins
        request_origin = request.headers.get("origin")
        if request_origin not in allowed_origins:
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "CORS error: Origin not allowed",
                    "allowed_origins": allowed_origins,
                    "request_origin": request_origin
                }
            )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail)}
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    # Print startup information
    print(f"\nStarting Material AI Advisor Backend")
    print(f"{'='*50}")
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Allowed Origins:")
    for origin in allowed_origins:
        print(f"  - {origin}")
    print(f"{'='*50}\n")
    
    try:
        uvicorn.run(
            "main:app",
            host=HOST,
            port=PORT,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        raise