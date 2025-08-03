import numpy as np
from typing import Dict, List, Any, Optional
import logging
import time
from dataclasses import dataclass
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    accuracy: float
    response_time: float
    relevance_score: float
    user_satisfaction: float
    timestamp: datetime = datetime.now()

class PerformanceValidator:
    def __init__(self, results_dir: str = "validation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Performance thresholds
        self.thresholds = {
            "response_time": 2.0,  # seconds
            "accuracy": 0.8,
            "relevance": 0.7,
            "satisfaction": 0.75
        }
        
        # Historical results
        self.results_history: List[ValidationResult] = []
        self.load_history()
    
    def load_history(self):
        """Load historical validation results"""
        history_file = self.results_dir / "validation_history.json"
        if history_file.exists():
            with open(history_file) as f:
                data = json.load(f)
                self.results_history = [
                    ValidationResult(
                        accuracy=r["accuracy"],
                        response_time=r["response_time"],
                        relevance_score=r["relevance_score"],
                        user_satisfaction=r["user_satisfaction"],
                        timestamp=datetime.fromisoformat(r["timestamp"])
                    )
                    for r in data
                ]
    
    def save_history(self):
        """Save validation results"""
        history_file = self.results_dir / "validation_history.json"
        with open(history_file, "w") as f:
            json.dump(
                [
                    {
                        "accuracy": r.accuracy,
                        "response_time": r.response_time,
                        "relevance_score": r.relevance_score,
                        "user_satisfaction": r.user_satisfaction,
                        "timestamp": r.timestamp.isoformat()
                    }
                    for r in self.results_history
                ],
                f,
                indent=2
            )
    
    def validate_search_results(
        self,
        results: List[Dict[str, Any]],
        expected_materials: Optional[List[str]] = None,
        response_time: float = 0.0,
        user_feedback: Optional[Dict[str, float]] = None
    ) -> ValidationResult:
        """
        Validate search results against expected materials and user feedback
        """
        # Calculate accuracy if expected materials provided
        accuracy = 0.0
        if expected_materials:
            found_materials = set(r["name"] for r in results)
            expected_set = set(expected_materials)
            accuracy = len(found_materials.intersection(expected_set)) / len(expected_set)
        
        # Calculate relevance score based on similarity scores
        relevance_score = np.mean([r.get("similarity_score", 0) for r in results])
        
        # Get user satisfaction from feedback
        satisfaction = 0.0
        if user_feedback:
            satisfaction = np.mean(list(user_feedback.values()))
        
        result = ValidationResult(
            accuracy=accuracy,
            response_time=response_time,
            relevance_score=relevance_score,
            user_satisfaction=satisfaction
        )
        
        self.results_history.append(result)
        self.save_history()
        
        return result
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.results_history:
            return {"error": "No validation data available"}
        
        # Calculate metrics
        recent_results = self.results_history[-100:]  # Last 100 results
        
        metrics = {
            "overall": {
                "accuracy": np.mean([r.accuracy for r in recent_results]),
                "avg_response_time": np.mean([r.response_time for r in recent_results]),
                "avg_relevance": np.mean([r.relevance_score for r in recent_results]),
                "user_satisfaction": np.mean([r.user_satisfaction for r in recent_results])
            },
            "thresholds_met": {
                "response_time": np.mean([r.response_time <= self.thresholds["response_time"] for r in recent_results]),
                "accuracy": np.mean([r.accuracy >= self.thresholds["accuracy"] for r in recent_results]),
                "relevance": np.mean([r.relevance_score >= self.thresholds["relevance"] for r in recent_results]),
                "satisfaction": np.mean([r.user_satisfaction >= self.thresholds["satisfaction"] for r in recent_results])
            },
            "trends": {
                "response_time": self._calculate_trend([r.response_time for r in recent_results]),
                "accuracy": self._calculate_trend([r.accuracy for r in recent_results]),
                "relevance": self._calculate_trend([r.relevance_score for r in recent_results]),
                "satisfaction": self._calculate_trend([r.user_satisfaction for r in recent_results])
            }
        }
        
        # Generate visualizations
        self._generate_performance_plots(recent_results)
        
        return metrics
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction and magnitude"""
        if len(values) < 2:
            return "insufficient_data"
        
        slope = np.polyfit(range(len(values)), values, 1)[0]
        
        if abs(slope) < 0.01:
            return "stable"
        elif slope > 0:
            return "improving" if slope > 0.05 else "slightly_improving"
        else:
            return "declining" if slope < -0.05 else "slightly_declining"
    
    def _generate_performance_plots(self, results: List[ValidationResult]):
        """Generate performance visualization plots"""
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time series data
        dates = [r.timestamp for r in results]
        
        # Response Time
        response_times = [r.response_time for r in results]
        ax1.plot(dates, response_times, 'b-')
        ax1.axhline(y=self.thresholds["response_time"], color='r', linestyle='--')
        ax1.set_title('Response Time')
        ax1.set_ylabel('Seconds')
        
        # Accuracy
        accuracies = [r.accuracy for r in results]
        ax2.plot(dates, accuracies, 'g-')
        ax2.axhline(y=self.thresholds["accuracy"], color='r', linestyle='--')
        ax2.set_title('Accuracy')
        
        # Relevance
        relevance = [r.relevance_score for r in results]
        ax3.plot(dates, relevance, 'm-')
        ax3.axhline(y=self.thresholds["relevance"], color='r', linestyle='--')
        ax3.set_title('Relevance Score')
        
        # User Satisfaction
        satisfaction = [r.user_satisfaction for r in results]
        ax4.plot(dates, satisfaction, 'c-')
        ax4.axhline(y=self.thresholds["satisfaction"], color='r', linestyle='--')
        ax4.set_title('User Satisfaction')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "performance_trends.png")
        plt.close()

# Initialize validator
performance_validator = PerformanceValidator() 