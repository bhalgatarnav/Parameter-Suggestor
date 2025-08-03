from database import SessionLocal
from models import FeedbackLog
import numpy as np
from sklearn.linear_model import Ridge
import joblib

def analyze_feedback():
    db = SessionLocal()
    
    # Get all feedback
    feedbacks = db.query(FeedbackLog).all()
    
    if len(feedbacks) < 10:
        print("Not enough feedback data to analyze")
        return
    
    # Prepare data for training
    X = []
    y = []
    for fb in feedbacks:
        features = [
            fb.recommendation_rank,
            1 if fb.interaction_type == "selected" else 0,
            fb.user_rating or 3  # Default to neutral if not rated
        ]
        X.append(features)
        y.append(fb.user_rating or 3)
    
    # Train model
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    
    # Save model for later use
    joblib.dump(model, "feedback_model.pkl")
    print("Feedback model trained and saved")
    
    # Calculate key metrics
    avg_rating = np.mean(y)
    selection_rate = sum(1 for fb in feedbacks if fb.interaction_type == "selected") / len(feedbacks)
    
    print(f"Average rating: {avg_rating:.2f}")
    print(f"Selection rate: {selection_rate:.2%}")
    print(f"Total feedback entries: {len(feedbacks)}")

if __name__ == "__main__":
    analyze_feedback()