from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class SearchMode(Enum):
    VISUAL = "visual"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    PROPERTY = "property"

class MaterialCategory(Enum):
    METAL = "metal"
    WOOD = "wood"
    PLASTIC = "plastic"
    GLASS = "glass"
    CERAMIC = "ceramic"
    FABRIC = "fabric"
    STONE = "stone"
    LIQUID = "liquid"
    ORGANIC = "organic"

@dataclass
class VisualReference:
    image: np.ndarray
    regions_of_interest: Optional[List[Dict[str, int]]] = None  # For partial image search
    weight: float = 1.0

@dataclass
class SearchCriteria:
    # Visual criteria
    reference_images: Optional[List[VisualReference]] = None
    
    # Semantic criteria
    description: Optional[str] = None
    material_type: Optional[MaterialCategory] = None
    
    # Property ranges (None means no constraint)
    roughness_range: Optional[tuple[float, float]] = None
    metalness_range: Optional[tuple[float, float]] = None
    transparency_range: Optional[tuple[float, float]] = None
    
    # Usage context
    intended_use: Optional[str] = None
    environment: Optional[str] = None
    
    # Rendering context
    lighting_condition: Optional[str] = None
    camera_distance: Optional[str] = None
    
    # Search behavior
    mode: SearchMode = SearchMode.HYBRID
    max_results: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert criteria to dictionary for API"""
        return {
            "visual_refs": [
                {
                    "image": ref.image.tolist(),
                    "roi": ref.regions_of_interest,
                    "weight": ref.weight
                }
                for ref in (self.reference_images or [])
            ],
            "description": self.description,
            "material_type": self.material_type.value if self.material_type else None,
            "properties": {
                "roughness": self.roughness_range,
                "metalness": self.metalness_range,
                "transparency": self.transparency_range
            },
            "context": {
                "use": self.intended_use,
                "environment": self.environment,
                "lighting": self.lighting_condition,
                "camera_distance": self.camera_distance
            },
            "search_config": {
                "mode": self.mode.value,
                "max_results": self.max_results
            }
        }

class SearchPreset:
    """Predefined search configurations for common scenarios"""
    
    @staticmethod
    def from_reference_photo(image: np.ndarray) -> SearchCriteria:
        """Create criteria for matching a reference photo"""
        return SearchCriteria(
            reference_images=[VisualReference(image=image)],
            mode=SearchMode.VISUAL,
            max_results=5
        )
    
    @staticmethod
    def for_close_up_render(description: str) -> SearchCriteria:
        """Create criteria for detailed close-up rendering"""
        return SearchCriteria(
            description=description,
            camera_distance="close",
            max_results=3,
            mode=SearchMode.SEMANTIC
        )
    
    @staticmethod
    def for_environment_shot(description: str, environment: str) -> SearchCriteria:
        """Create criteria for environmental rendering"""
        return SearchCriteria(
            description=description,
            environment=environment,
            camera_distance="far",
            max_results=5,
            mode=SearchMode.HYBRID
        )
    
    @staticmethod
    def for_product_viz(
        description: str,
        reference_image: Optional[np.ndarray] = None
    ) -> SearchCriteria:
        """Create criteria for product visualization"""
        return SearchCriteria(
            description=description,
            reference_images=[VisualReference(image=reference_image)] if reference_image else None,
            lighting_condition="studio",
            camera_distance="medium",
            mode=SearchMode.HYBRID,
            max_results=3
        )

class SearchMetrics:
    """Track and validate search performance"""
    
    def __init__(self):
        self.total_searches = 0
        self.successful_searches = 0
        self.avg_response_time = 0
        self.feedback_scores = []
        self.cache_hits = 0
        self.mode_distribution = {mode: 0 for mode in SearchMode}
    
    def log_search(
        self,
        criteria: SearchCriteria,
        response_time: float,
        success: bool,
        cache_hit: bool
    ):
        """Log metrics for a search operation"""
        self.total_searches += 1
        if success:
            self.successful_searches += 1
        
        # Update average response time
        self.avg_response_time = (
            (self.avg_response_time * (self.total_searches - 1) + response_time)
            / self.total_searches
        )
        
        # Track cache performance
        if cache_hit:
            self.cache_hits += 1
        
        # Track search mode usage
        self.mode_distribution[criteria.mode] += 1
    
    def log_feedback(self, score: float):
        """Log user feedback score (0-1)"""
        self.feedback_scores.append(score)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            "total_searches": self.total_searches,
            "success_rate": self.successful_searches / max(1, self.total_searches),
            "avg_response_time": self.avg_response_time,
            "cache_hit_rate": self.cache_hits / max(1, self.total_searches),
            "avg_feedback_score": np.mean(self.feedback_scores) if self.feedback_scores else 0,
            "mode_distribution": {
                mode.value: count / max(1, self.total_searches)
                for mode, count in self.mode_distribution.items()
            }
        }

class NaturalSearchInterface:
    """Designer-friendly search interface"""
    
    def __init__(self):
        self.metrics = SearchMetrics()
        self.recent_searches = []  # Cache recent searches
    
    def search_by_example(
        self,
        reference_image: np.ndarray,
        context: Optional[str] = None
    ) -> SearchCriteria:
        """Search using a reference image"""
        criteria = SearchPreset.from_reference_photo(reference_image)
        if context:
            criteria.description = context
            criteria.mode = SearchMode.HYBRID
        return criteria
    
    def search_by_description(
        self,
        description: str,
        material_type: Optional[MaterialCategory] = None,
        rendering_context: Optional[Dict[str, str]] = None
    ) -> SearchCriteria:
        """Search using natural language description"""
        criteria = SearchCriteria(
            description=description,
            material_type=material_type,
            mode=SearchMode.SEMANTIC
        )
        
        if rendering_context:
            criteria.lighting_condition = rendering_context.get("lighting")
            criteria.camera_distance = rendering_context.get("distance")
            criteria.environment = rendering_context.get("environment")
        
        return criteria
    
    def search_by_properties(
        self,
        roughness: Optional[float] = None,
        metalness: Optional[float] = None,
        transparency: Optional[float] = None,
        tolerance: float = 0.2
    ) -> SearchCriteria:
        """Search by specific material properties"""
        criteria = SearchCriteria(mode=SearchMode.PROPERTY)
        
        if roughness is not None:
            criteria.roughness_range = (max(0, roughness - tolerance), min(1, roughness + tolerance))
        if metalness is not None:
            criteria.metalness_range = (max(0, metalness - tolerance), min(1, metalness + tolerance))
        if transparency is not None:
            criteria.transparency_range = (max(0, transparency - tolerance), min(1, transparency + tolerance))
        
        return criteria
    
    def refine_search(
        self,
        previous_criteria: SearchCriteria,
        adjustments: Dict[str, Any]
    ) -> SearchCriteria:
        """Refine previous search results"""
        new_criteria = previous_criteria
        
        for key, value in adjustments.items():
            if key == "rougher":
                new_criteria.roughness_range = (
                    new_criteria.roughness_range[0] + 0.1,
                    new_criteria.roughness_range[1] + 0.1
                )
            elif key == "smoother":
                new_criteria.roughness_range = (
                    new_criteria.roughness_range[0] - 0.1,
                    new_criteria.roughness_range[1] - 0.1
                )
            elif key == "more_metallic":
                new_criteria.metalness_range = (
                    new_criteria.metalness_range[0] + 0.1,
                    new_criteria.metalness_range[1] + 0.1
                )
            # Add more refinement options
        
        return new_criteria

# Initialize interface
search_interface = NaturalSearchInterface() 