# src/utils.py
import logging
import sys
import os
import joblib
import pandas as pd
import numpy as np
from typing import Any, Dict

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

def setup_plotting():
    """Setup matplotlib and seaborn for better plots"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    sns.set_style("whitegrid")
    return plt, sns

def load_model(model_path: str) -> Any:
    """
    Load a trained model from file with error handling
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model object
    """
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise

def save_model(model: Any, model_path: str) -> None:
    """
    Save a trained model to file
    
    Args:
        model: Model object to save
        model_path: Path where to save the model
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        joblib.dump(model, model_path)
        logger.info(f"Model saved successfully to {model_path}")
        
    except Exception as e:
        logger.error(f"Failed to save model to {model_path}: {e}")
        raise

def validate_features(features: Dict[str, Any], required_features: list) -> bool:
    """
    Validate that all required features are present
    
    Args:
        features: Dictionary of features
        required_features: List of required feature names
        
    Returns:
        Boolean indicating if features are valid
    """
    missing_features = [feat for feat in required_features if feat not in features]
    
    if missing_features:
        logger.warning(f"Missing required features: {missing_features}")
        return False
    
    return True

def calculate_derived_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate derived features from basic features
    
    Args:
        features: Dictionary of basic features
        
    Returns:
        Dictionary with added derived features
    """
    derived = features.copy()
    
    # Calculate price to dish ratio
    if 'avg_price' in features and 'dish_count' in features:
        derived['price_to_dish_ratio'] = (
            features['avg_price'] / max(features['dish_count'], 1)
        )
    
    # Calculate rating count per dish
    if 'total_rating_count' in features and 'dish_count' in features:
        derived['rating_count_per_dish'] = (
            features['total_rating_count'] / max(features['dish_count'], 1)
        )
    
    # Calculate price volatility
    if 'price_std' in features and 'avg_price' in features:
        derived['price_volatility'] = (
            features['price_std'] / max(features['avg_price'], 1)
        )
    
    # Determine if has high rating variance
    if 'rating_std' in features:
        derived['has_high_variance'] = 1 if features['rating_std'] > 0.5 else 0
    
    return derived

def format_probability(probability: float) -> str:
    """
    Format probability as percentage string
    
    Args:
        probability: Probability value between 0 and 1
        
    Returns:
        Formatted percentage string
    """
    return f"{probability:.1%}"

def get_confidence_level(probability: float) -> str:
    """
    Get confidence level based on probability
    
    Args:
        probability: Probability value between 0 and 1
        
    Returns:
        Confidence level string
    """
    if probability >= 0.8:
        return "high"
    elif probability >= 0.6:
        return "medium"
    else:
        return "low"

# Feature mapping for the optimized models
def get_feature_mapping() -> Dict[str, Any]:
    """
    Get the optimized feature mapping for 30 features
    
    Returns:
        Dictionary with feature mapping information
    """
    return {
        'basic_features': [
            'avg_price', 'dish_count', 'total_rating_count', 'rating_std',
            'price_std', 'category_diversity', 'price_to_dish_ratio',
            'rating_count_per_dish', 'has_high_variance'
        ],
        'city_mapping': {
            'mumbai': 17, 'delhi': 18, 'bangalore': 10, 'bengaluru': 10,
            'chennai': 12, 'kolkata': 15, 'hyderabad': 13, 'ahmedabad': 9,
            'chandigarh': 11, 'lucknow': 16, 'jaipur': 14, 'other': 19
        },
        'total_features': 30
    }

class PerformanceTimer:
    """Context manager for timing code execution"""
    
    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = pd.Timestamp.now()
        logger.info(f"Starting {self.operation_name}...")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = pd.Timestamp.now()
        duration = (end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            logger.info(f"Completed {self.operation_name} in {duration:.3f}s")
        else:
            logger.error(f"{self.operation_name} failed after {duration:.3f}s")