import numpy as np
import json
import openai
import os
import re
from dotenv import load_dotenv
from database import SessionLocal
from models import Material
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from fastapi import HTTPException
from pydantic import BaseModel

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

class MaterialRecommender:
    def __init__(self):
        self.db_session = SessionLocal()
        self.cache = {}  # Simple cache for embeddings

    def __del__(self):
        self.db_session.close()

    def get_material_embedding(self, material_id: int) -> np.ndarray:
        """Get or compute material embedding"""
        if material_id in self.cache:
            return self.cache[material_id]
        
        material = self.db_session.query(Material).get(material_id)
        if not material:
            raise ValueError(f"Material {material_id} not found")
        
        # Combine material properties into a text representation
        text = f"{material.name} {material.material_class} {material.description} "
        text += " ".join([tag.tag for tag in material.appearance_tags])
        text += " ".join([tag.tag for tag in material.intent_tags])
        text += " ".join([uc.use_case for uc in material.suggested_use_cases])
        
        # Generate embedding
        embedding = model.encode([text])[0]
        self.cache[material_id] = embedding
        return embedding

    def recommend(self, query: str, filters: Optional[Dict] = None, top_n: int = 3) -> List[Dict]:
        """Get material recommendations based on query and filters"""
        try:
            # Generate query embedding
            query_embedding = model.encode([query])[0]
            
            # Get all materials
            materials = self.db_session.query(Material)
        
            # Apply filters if provided
            if filters:
                if 'sustainability' in filters:
                    materials = materials.filter(Material.sustainability_level >= filters['sustainability'])
                if 'material_class' in filters:
                    materials = materials.filter(Material.material_class.ilike(f"%{filters['material_class']}%"))
                if 'min_metalness' in filters:
                    materials = materials.filter(Material.metalness >= filters['min_metalness'])
                if 'max_roughness' in filters:
                    materials = materials.filter(Material.roughness <= filters['max_roughness'])
            
            # Calculate similarities and rank materials
            results = []
            for material in materials:
                try:
                    material_embedding = self.get_material_embedding(material.id)
                    similarity = cosine_similarity([query_embedding], [material_embedding])[0][0]
                    
                    # Calculate detailed match reasons
                    match_reasons = {
                        "semantic_similarity": float(similarity),
                        "property_match": self._calculate_property_match(material, filters or {}),
                        "application_match": self._calculate_application_match(material, query),
                        "technical_match": self._calculate_technical_match(material, filters or {})
                    }
                    
                    # Calculate overall match score
                    match_score = (
                        match_reasons["semantic_similarity"] * 0.4 +
                        match_reasons["property_match"] * 0.3 +
                        match_reasons["application_match"] * 0.2 +
                        match_reasons["technical_match"] * 0.1
                    )
                    
                    # Convert material to dict with all relationships
                    material_dict = {
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
                        "texture_maps": [tmap.map_type for tmap in material.texture_map_types],
                        "similarity_score": float(match_score),
                        "match_reasons": match_reasons
                    }
                    
                    results.append(material_dict)
                except Exception as e:
                    print(f"Error processing material {material.id}: {str(e)}")
                    continue
            
            # Sort by similarity and return top N
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return results[:top_n]
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

    def _calculate_property_match(self, material: Material, filters: Dict) -> float:
        """Calculate how well material properties match the filters"""
        match_score = 1.0
        
        if 'min_metalness' in filters:
            if material.metalness >= filters['min_metalness']:
                match_score *= 1.0
            else:
                match_score *= max(0.1, material.metalness / filters['min_metalness'])
        
        if 'max_roughness' in filters:
            if material.roughness <= filters['max_roughness']:
                match_score *= 1.0
            else:
                match_score *= max(0.1, filters['max_roughness'] / material.roughness)
        
        if 'sustainability' in filters:
            sustainability_levels = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
            material_level = sustainability_levels.get(material.sustainability_level, 0.5)
            required_level = sustainability_levels.get(filters['sustainability'], 0.5)
            match_score *= max(0.1, material_level / required_level)
        
        return match_score

    def _calculate_application_match(self, material: Material, query: str) -> float:
        """Calculate how well material matches the application context"""
        query_lower = query.lower()
        match_score = 0.5  # Base score
        
        # Check use cases
        for use_case in material.suggested_use_cases:
            if any(word in query_lower for word in use_case.use_case.lower().split()):
                match_score = min(1.0, match_score + 0.2)
        
        # Check environment suitability
        for env in material.environment_suitability:
            if any(word in query_lower for word in env.env_condition.lower().split()):
                match_score = min(1.0, match_score + 0.2)
        
        # Check keywords
        for kw in material.user_keywords:
            if kw.keyword.lower() in query_lower:
                match_score = min(1.0, match_score + 0.1)
        
        return match_score

    def _calculate_technical_match(self, material: Material, filters: Dict) -> float:
        """Calculate technical property match score"""
        match_score = 1.0
        
        # Physical properties match
        if material.has_physical_equivalent:
            match_score *= 1.2
        
        # Visual behavior match
        if material.visual_behavior and any(
            behavior in material.visual_behavior.lower()
            for behavior in ['realistic', 'accurate', 'precise', 'high-quality']
        ):
            match_score *= 1.1
        
        # Color match (if specified in filters)
        if all(key in filters for key in ['color_r', 'color_g', 'color_b']):
            color_diff = sum([
                abs(material.color_r - filters['color_r']),
                abs(material.color_g - filters['color_g']),
                abs(material.color_b - filters['color_b'])
            ]) / 3.0
            match_score *= max(0.1, 1.0 - color_diff)
        
        return min(1.0, match_score)

# Create recommender instance
recommender = MaterialRecommender()

class FeedbackRequest(BaseModel):
    material_id: int
    feedback_type: str
    reason: Optional[str] = None

# Feedback endpoint
async def submit_feedback(feedback: FeedbackRequest):
    try:
        # Log feedback to database
        db = SessionLocal()
        feedback_log = FeedbackLog(
            material_id=feedback.material_id,
            interaction_type=feedback.feedback_type,
            context_snapshot={"reason": feedback.reason} if feedback.reason else None
        )
        db.add(feedback_log)
        db.commit()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()