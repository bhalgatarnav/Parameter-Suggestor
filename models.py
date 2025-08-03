from sqlalchemy import Column, Integer, String, Float, Text, Boolean, TIMESTAMP, JSON, Enum, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('API_KEY')

class Material(Base):
    __tablename__ = "materials"
    id = Column(Integer, primary_key=True, autoincrement=True)
    material_id = Column(String(20), nullable=False)
    name = Column(String(100), nullable=False)
    metalness = Column(Float)
    roughness = Column(Float)
    color_r = Column(Float)
    color_g = Column(Float)
    color_b = Column(Float)
    specular_r = Column(Float)
    specular_g = Column(Float)
    specular_b = Column(Float)
    material_class = Column(String(50))
    description = Column(Text)
    visual_behavior = Column(Text)
    color_family = Column(String(50))
    has_physical_equivalent = Column(Boolean)
    real_world_reference = Column(Text)
    preview_image_path = Column(String(255))
    final_prompt = Column(Text)
    keyshot_category = Column(Text)
    sustainability_level = Column(Enum('low', 'medium', 'high'))
    manufacturability_notes = Column(Text)
    last_updated = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    embedding = Column(JSON, nullable=True)
    
    # Relationships to child tables
    appearance_tags = relationship("AppearanceTag", back_populates="material")
    intent_tags = relationship("IntentTag", back_populates="material")
    texture_map_types = relationship("TextureMapType", back_populates="material")
    suggested_use_cases = relationship("SuggestedUseCase", back_populates="material")
    environment_suitability = relationship("EnvironmentSuitability", back_populates="material")
    user_keywords = relationship("UserKeyword", back_populates="material")

class AppearanceTag(Base):
    __tablename__ = "appearance_tags"
    id = Column(Integer, primary_key=True, autoincrement=True)
    material_id = Column(Integer, ForeignKey('materials.id'), nullable=False)
    tag = Column(String(50), nullable=False)
    
    material = relationship("Material", back_populates="appearance_tags")

class IntentTag(Base):
    __tablename__ = "intent_tags"
    id = Column(Integer, primary_key=True, autoincrement=True)
    material_id = Column(Integer, ForeignKey('materials.id'), nullable=False)
    tag = Column(String(50), nullable=False)
    
    material = relationship("Material", back_populates="intent_tags")

class TextureMapType(Base):
    __tablename__ = "texture_map_types"
    id = Column(Integer, primary_key=True, autoincrement=True)
    material_id = Column(Integer, ForeignKey('materials.id'), nullable=False)
    map_type = Column(String(50), nullable=False)
    
    material = relationship("Material", back_populates="texture_map_types")

class SuggestedUseCase(Base):
    __tablename__ = "suggested_use_cases"
    id = Column(Integer, primary_key=True, autoincrement=True)
    material_id = Column(Integer, ForeignKey('materials.id'), nullable=False)
    use_case = Column(String(50), nullable=False)
    
    material = relationship("Material", back_populates="suggested_use_cases")

class EnvironmentSuitability(Base):
    __tablename__ = "environment_suitability"
    id = Column(Integer, primary_key=True, autoincrement=True)
    material_id = Column(Integer, ForeignKey('materials.id'), nullable=False)
    env_condition = Column(String(50), nullable=False)
    
    material = relationship("Material", back_populates="environment_suitability")

class UserKeyword(Base):
    __tablename__ = "user_keywords"
    id = Column(Integer, primary_key=True, autoincrement=True)
    material_id = Column(Integer, ForeignKey('materials.id'), nullable=False)
    keyword = Column(String(50), nullable=False)
    
    material = relationship("Material", back_populates="user_keywords")

# Feedback log table
class FeedbackLog(Base):
    __tablename__ = "feedback_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), nullable=False)
    material_id = Column(Integer, ForeignKey('materials.id'), nullable=False)
    recommendation_rank = Column(Integer)
    interaction_type = Column(String(20))  # 'selected', 'viewed', 'dismissed'
    user_rating = Column(Integer)
    modification_details = Column(JSON)
    context_snapshot = Column(JSON)
    timestamp = Column(TIMESTAMP, server_default=func.now())
    
    material = relationship("Material")