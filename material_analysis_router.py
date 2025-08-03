from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
import shutil
import os
from tempfile import NamedTemporaryFile
from image_analyzer import ImageAnalyzer
from intent_parser import IntentParser
from material_matcher import MaterialMatcher
from utils.vision_analyzer import generate_design_insights, generate_mood_board
from PIL import Image

router = APIRouter()
image_analyzer = ImageAnalyzer()
intent_parser = IntentParser()
material_matcher = MaterialMatcher()

@router.post("/analyze-material")
async def analyze_material(
    file: UploadFile = File(...),
    image_content: str = Form(...),
    inspiration_type: str = Form(...),
    additional_notes: Optional[str] = Form(None)
):
    """
    Analyze uploaded image and find matching materials with design insights
    """
    try:
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Open image for both technical and design analysis
        image = Image.open(temp_path)
        
        # Generate design insights first
        user_context = {
            "image_content": image_content,
            "inspiration_type": inspiration_type,
            "additional_notes": additional_notes
        }
        design_insights = generate_design_insights(image, user_context)
        
        # Extract technical features
        features = image_analyzer.analyze_image(temp_path)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        # Find matching materials
        matches = material_matcher.find_matches(features, {
            "inspiration_type": inspiration_type,
            "style_keywords": design_insights.get("style_keywords", [])
        })
        
        # Generate mood board
        mood_board = generate_mood_board(design_insights, matches)
        
        return {
            "success": True,
            "data": {
                "technical_analysis": {
                    "material_analysis": features.get("material_analysis", {}),
                    "surface_properties": features.get("surface_properties", {})
                },
                "design_insights": design_insights,
                "mood_board": mood_board,
                "matches": matches
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": {
                "code": "PROCESSING_ERROR",
                "message": f"Failed to analyze image: {str(e)}"
            }
        }

@router.get("/inspiration-types")
async def get_inspiration_types():
    """Get available inspiration types"""
    return {
        "types": intent_parser.inspiration_types,
        "attributes": intent_parser.attribute_list,
        "use_cases": intent_parser.use_cases
    } 