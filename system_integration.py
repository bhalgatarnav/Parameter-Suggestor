import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import importlib
import sys
from concurrent.futures import ThreadPoolExecutor
import traceback

# Import all system components
from search_features import search_interface
from validation import performance_validator
from material_recognition import material_recognizer
from context_reasoning import context_reasoner
from preprocessing import ImagePreprocessor
from database import search_engine

logger = logging.getLogger(__name__)

class SystemIntegrationManager:
    def __init__(self):
        self.components = {
            "search_interface": search_interface,
            "performance_validator": performance_validator,
            "material_recognizer": material_recognizer,
            "context_reasoner": context_reasoner,
            "search_engine": search_engine
        }
        
        self.dependency_graph = {
            "search_interface": ["material_recognizer", "context_reasoner", "search_engine"],
            "material_recognizer": ["search_engine"],
            "context_reasoner": ["search_engine"],
            "performance_validator": ["search_interface"],
            "search_engine": []
        }
        
        # Integration metrics
        self.integration_metrics = {
            "component_health": {},
            "api_latency": {},
            "error_rates": {},
            "component_usage": {},
            "cross_component_timing": {}
        }
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check health of all system components"""
        health_status = {}
        
        for component_name, component in self.components.items():
            try:
                # Check if component is loaded
                if component is None:
                    health_status[component_name] = {
                        "status": "error",
                        "message": "Component not loaded"
                    }
                    continue
                
                # Check if required methods exist
                required_methods = self._get_required_methods(component_name)
                missing_methods = [
                    method for method in required_methods
                    if not hasattr(component, method)
                ]
                
                if missing_methods:
                    health_status[component_name] = {
                        "status": "warning",
                        "message": f"Missing methods: {missing_methods}"
                    }
                    continue
                
                # Test basic functionality
                self._test_component(component_name, component)
                
                health_status[component_name] = {
                    "status": "healthy",
                    "message": "All checks passed"
                }
                
            except Exception as e:
                health_status[component_name] = {
                    "status": "error",
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
        
        self.integration_metrics["component_health"] = health_status
        return health_status
    
    def _get_required_methods(self, component_name: str) -> List[str]:
        """Get required methods for each component"""
        method_requirements = {
            "search_interface": [
                "search_by_example",
                "search_by_description",
                "search_by_properties"
            ],
            "material_recognizer": [
                "recognize_material",
                "get_material_embeddings"
            ],
            "context_reasoner": [
                "analyze_context",
                "generate_rendering_constraints"
            ],
            "search_engine": [
                "search",
                "rebuild_index"
            ],
            "performance_validator": [
                "validate_search_results",
                "generate_performance_report"
            ]
        }
        return method_requirements.get(component_name, [])
    
    def _test_component(self, component_name: str, component: Any):
        """Test basic component functionality"""
        if component_name == "search_interface":
            # Test search interface with dummy data
            try:
                component.search_by_description("test material")
            except Exception as e:
                raise Exception(f"Search interface test failed: {str(e)}")
        
        elif component_name == "material_recognizer":
            # Test recognizer is initialized
            if not hasattr(component, "clip_model"):
                raise Exception("CLIP model not initialized")
        
        # Add more component-specific tests
    
    def monitor_cross_component_performance(self) -> Dict[str, Any]:
        """Monitor performance between component interactions"""
        metrics = {}
        
        # Test search pipeline
        start_time = time.time()
        try:
            # Simulate search request
            description = "brushed metal surface"
            
            # Track component timing
            timings = {}
            
            # 1. Context reasoning
            t0 = time.time()
            context = self.components["context_reasoner"].analyze_context(
                {"description": description}
            )
            timings["context_reasoning"] = time.time() - t0
            
            # 2. Material recognition
            t0 = time.time()
            material_info = self.components["material_recognizer"].recognize_material(
                description
            )
            timings["material_recognition"] = time.time() - t0
            
            # 3. Search
            t0 = time.time()
            search_results = self.components["search_engine"].search(
                material_info["embeddings"]
            )
            timings["search"] = time.time() - t0
            
            metrics["pipeline_timings"] = timings
            metrics["total_time"] = time.time() - start_time
            metrics["success"] = True
            
        except Exception as e:
            metrics["success"] = False
            metrics["error"] = str(e)
        
        self.integration_metrics["cross_component_timing"] = metrics
        return metrics
    
    def track_component_usage(self, component_name: str, method_name: str, execution_time: float):
        """Track component usage and performance"""
        if component_name not in self.integration_metrics["component_usage"]:
            self.integration_metrics["component_usage"][component_name] = {
                "call_count": 0,
                "avg_execution_time": 0,
                "methods": {}
            }
        
        component_metrics = self.integration_metrics["component_usage"][component_name]
        component_metrics["call_count"] += 1
        
        # Update average execution time
        prev_avg = component_metrics["avg_execution_time"]
        component_metrics["avg_execution_time"] = (
            (prev_avg * (component_metrics["call_count"] - 1) + execution_time)
            / component_metrics["call_count"]
        )
        
        # Track method-specific metrics
        if method_name not in component_metrics["methods"]:
            component_metrics["methods"][method_name] = {
                "call_count": 0,
                "avg_execution_time": 0
            }
        
        method_metrics = component_metrics["methods"][method_name]
        method_metrics["call_count"] += 1
        prev_avg = method_metrics["avg_execution_time"]
        method_metrics["avg_execution_time"] = (
            (prev_avg * (method_metrics["call_count"] - 1) + execution_time)
            / method_metrics["call_count"]
        )
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive integration metrics"""
        return {
            "component_health": self.integration_metrics["component_health"],
            "cross_component_performance": self.integration_metrics["cross_component_timing"],
            "component_usage": self.integration_metrics["component_usage"],
            "system_status": self._get_system_status()
        }
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        component_health = self.integration_metrics["component_health"]
        
        return {
            "healthy_components": sum(
                1 for status in component_health.values()
                if status["status"] == "healthy"
            ),
            "total_components": len(component_health),
            "system_ready": all(
                status["status"] == "healthy"
                for status in component_health.values()
            )
        }

# Initialize system integration manager
system_manager = SystemIntegrationManager() 