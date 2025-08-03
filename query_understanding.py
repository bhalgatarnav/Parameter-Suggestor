import logging
from typing import Dict, List, Any, Optional, Tuple
import spacy
import numpy as np
from transformers import pipeline
from dataclasses import dataclass
import re
from thefuzz import fuzz
import json

logger = logging.getLogger(__name__)

@dataclass
class QueryIntent:
    primary_intent: str
    confidence: float
    material_type: Optional[str] = None
    properties: Dict[str, float] = None
    context: Dict[str, str] = None
    modifiers: List[str] = None

class QueryUnderstanding:
    def __init__(self):
        # Load NLP models
        self.nlp = spacy.load("en_core_web_lg")
        self.zero_shot = pipeline("zero-shot-classification")
        
        # Load material knowledge base
        self.material_knowledge = self._load_material_knowledge()
        
        # Intent categories
        self.intent_categories = [
            "material_search",
            "property_adjustment",
            "rendering_setup",
            "comparison",
            "technical_question"
        ]
        
        # Property patterns
        self.property_patterns = {
            "roughness": r"rough(ness)?|smooth(ness)?",
            "metalness": r"metal(lic|ness)?|shiny",
            "transparency": r"transpar(ent|ency)|clear|opaque",
            "glossiness": r"gloss(y|iness)|sheen",
            "reflectivity": r"reflect(ive|ion)|mirror"
        }
        
        # Context patterns
        self.context_patterns = {
            "lighting": r"light(ing)?|illuminat(ed|ion)|bright|dark",
            "environment": r"indoor|outdoor|studio|natural|artificial",
            "distance": r"close(-|\s)?up|far|distance|macro"
        }
    
    def _load_material_knowledge(self) -> Dict[str, Any]:
        """Load material knowledge base"""
        try:
            with open("data/material_knowledge.json") as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback to basic knowledge
            return {
                "categories": [
                    "metal", "wood", "plastic", "glass", "ceramic",
                    "fabric", "stone", "liquid", "organic"
                ],
                "properties": {
                    "metal": {"metalness": 0.9, "roughness": 0.4},
                    "wood": {"metalness": 0.0, "roughness": 0.7},
                    "glass": {"metalness": 0.0, "roughness": 0.1},
                    # Add more defaults
                }
            }
    
    def parse_query(self, query: str) -> QueryIntent:
        """Parse and understand user query"""
        # Clean query
        query = query.lower().strip()
        
        # Get query intent
        intent, confidence = self._classify_intent(query)
        
        # Extract material type
        material_type = self._extract_material_type(query)
        
        # Extract properties
        properties = self._extract_properties(query)
        
        # Extract context
        context = self._extract_context(query)
        
        # Extract modifiers
        modifiers = self._extract_modifiers(query)
        
        return QueryIntent(
            primary_intent=intent,
            confidence=confidence,
            material_type=material_type,
            properties=properties,
            context=context,
            modifiers=modifiers
        )
    
    def _classify_intent(self, query: str) -> Tuple[str, float]:
        """Classify query intent using zero-shot classification"""
        result = self.zero_shot(
            query,
            candidate_labels=self.intent_categories,
            hypothesis_template="This is a {} query."
        )
        return result["labels"][0], result["scores"][0]
    
    def _extract_material_type(self, query: str) -> Optional[str]:
        """Extract material type from query"""
        doc = self.nlp(query)
        
        # Check for direct material mentions
        for token in doc:
            if token.text in self.material_knowledge["categories"]:
                return token.text
        
        # Check for similar terms
        best_match = None
        best_score = 0
        for category in self.material_knowledge["categories"]:
            for token in doc:
                score = fuzz.ratio(token.text, category)
                if score > 80 and score > best_score:
                    best_match = category
                    best_score = score
        
        return best_match
    
    def _extract_properties(self, query: str) -> Dict[str, float]:
        """Extract material properties and their values"""
        properties = {}
        
        for prop, pattern in self.property_patterns.items():
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                # Look for numeric values
                context = query[max(0, match.start()-20):min(len(query), match.end()+20)]
                numbers = re.findall(r"(\d+(\.\d+)?)", context)
                
                if numbers:
                    # Use explicit value
                    properties[prop] = float(numbers[0][0])
                else:
                    # Infer value from context
                    properties[prop] = self._infer_property_value(prop, context)
        
        return properties
    
    def _infer_property_value(self, property_name: str, context: str) -> float:
        """Infer property value from context"""
        # Define property-specific inference rules
        intensity_words = {
            "very": 0.9,
            "highly": 0.8,
            "moderately": 0.5,
            "slightly": 0.3,
            "barely": 0.1
        }
        
        # Check for intensity modifiers
        value = 0.5  # Default middle value
        for word, modifier in intensity_words.items():
            if word in context:
                value = modifier
                break
        
        # Adjust for negations
        if "not" in context or "isn't" in context:
            value = 1.0 - value
        
        return value
    
    def _extract_context(self, query: str) -> Dict[str, str]:
        """Extract rendering and usage context"""
        context = {}
        
        for ctx_type, pattern in self.context_patterns.items():
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                context[ctx_type] = match.group()
        
        return context
    
    def _extract_modifiers(self, query: str) -> List[str]:
        """Extract query modifiers"""
        doc = self.nlp(query)
        modifiers = []
        
        # Extract comparative modifiers
        for token in doc:
            if token.dep_ == "amod" and token.tag_ in ["JJR", "JJS"]:
                modifiers.append(token.text)
        
        return modifiers
    
    def clarify_ambiguity(self, query: str, intent: QueryIntent) -> List[Dict[str, Any]]:
        """Generate clarifying questions for ambiguous queries"""
        clarifications = []
        
        # Check confidence threshold
        if intent.confidence < 0.7:
            clarifications.append({
                "type": "intent",
                "question": f"Are you looking to: 1) Find a material, 2) Adjust properties, or 3) Get rendering advice?"
            })
        
        # Check for missing material type
        if not intent.material_type and intent.primary_intent == "material_search":
            clarifications.append({
                "type": "material_type",
                "question": "What type of material are you looking for? (e.g., metal, wood, plastic)"
            })
        
        # Check for vague properties
        if intent.properties:
            for prop, value in intent.properties.items():
                if value == 0.5:  # Default middle value indicates uncertainty
                    clarifications.append({
                        "type": "property",
                        "property": prop,
                        "question": f"How {prop} should the material be? (very, moderately, slightly)"
                    })
        
        return clarifications
    
    def refine_query(self, original_query: str, clarification_responses: Dict[str, str]) -> str:
        """Refine query based on clarification responses"""
        refined_query = original_query
        
        for response_type, response in clarification_responses.items():
            if response_type == "intent":
                refined_query = f"{response} {refined_query}"
            elif response_type == "material_type":
                refined_query = f"{response} {refined_query}"
            elif response_type == "property":
                refined_query = refined_query.replace(
                    response_type,
                    f"{response} {response_type}"
                )
        
        return refined_query

# Initialize query understanding
query_processor = QueryUnderstanding() 