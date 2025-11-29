# src/config.py
import os
from typing import Dict, List, Any

# Application Configuration
class AppConfig:
    """Application configuration settings"""
    
    # Application info
    APP_NAME = "Swiggy Restaurant Analytics"
    APP_VERSION = "2.1.0"
    APP_DESCRIPTION = "Machine Learning Powered Restaurant Insights"
    
    # Model configuration
    MODEL_FILES = {
        'high_rated': 'models/deployment_high_rated_model.pkl',
        'popular': 'models/deployment_popular_model.pkl',
        'premium': 'models/deployment_premium_model.pkl'
    }
    
    # Feature configuration
    REQUIRED_FEATURES = [
        'avg_price', 'dish_count', 'total_rating_count', 'avg_rating',
        'median_rating', 'rating_std', 'price_std', 'category_diversity',
        'city'
    ]
    
    # Prediction thresholds
    PREDICTION_THRESHOLDS = {
        'high_rated': 0.5,
        'popular': 0.5,
        'premium': 0.5
    }
    
    # Confidence levels
    CONFIDENCE_LEVELS = {
        'high': 0.8,
        'medium': 0.6,
        'low': 0.0
    }

# Feature configuration
class FeatureConfig:
    """Feature engineering configuration"""
    
    # Feature ranges for validation
    FEATURE_RANGES = {
        'avg_price': (0, 5000),
        'dish_count': (1, 1000),
        'total_rating_count': (0, 100000),
        'avg_rating': (1.0, 5.0),
        'median_rating': (1.0, 5.0),
        'rating_std': (0.0, 2.0),
        'price_std': (0.0, 1000.0),
        'category_diversity': (1, 50)
    }
    
    # Default values for features
    DEFAULT_VALUES = {
        'avg_price': 450,
        'dish_count': 85,
        'total_rating_count': 1200,
        'avg_rating': 4.1,
        'median_rating': 4.2,
        'rating_std': 0.3,
        'price_std': 150.0,
        'category_diversity': 8,
        'city': 'mumbai'
    }
    
    # Optimal values for YES predictions
    OPTIMAL_VALUES = {
        'high_rated': {
            'avg_rating': 4.5,
            'rating_std': 0.2,
            'total_rating_count': 2000,
            'city': 'mumbai'
        },
        'popular': {
            'total_rating_count': 2500,
            'rating_count_per_dish': 50,
            'city': 'delhi'
        },
        'premium': {
            'avg_price': 600,
            'price_to_dish_ratio': 12,
            'category_diversity': 6
        }
    }

# Model configuration
class ModelConfig:
    """Model-specific configuration"""
    
    # Model performance expectations
    EXPECTED_PERFORMANCE = {
        'feature_preparation': 0.01,  # seconds
        'prediction': 0.1,            # seconds
        'total_response': 0.2         # seconds
    }
    
    # Feature importance (for reference)
    FEATURE_IMPORTANCE = {
        'high_rated': ['avg_rating', 'rating_std', 'total_rating_count'],
        'popular': ['total_rating_count', 'rating_count_per_dish', 'city'],
        'premium': ['avg_price', 'price_to_dish_ratio', 'category_diversity']
    }

# API Configuration
class APIConfig:
    """API configuration settings"""
    
    # For local development (if needed)
    LOCAL_API_URL = "http://localhost:8000"
    
    # Endpoints
    ENDPOINTS = {
        'health': '/health',
        'predict_high_rated': '/predict/high-rated',
        'predict_popular': '/predict/popular', 
        'predict_premium': '/predict/premium',
        'suggested_features': '/suggested-features'
    }
    
    # Timeout settings
    TIMEOUT = 30

# Streamlit Configuration
class StreamlitConfig:
    """Streamlit app configuration"""
    
    # Page configuration
    PAGE_CONFIG = {
        'page_title': "Swiggy Restaurant Analytics",
        'page_icon': "🍽️",
        'layout': "wide",
        'initial_sidebar_state': "expanded"
    }
    
    # UI settings
    THEME = {
        'primary_color': '#FF6B6B',
        'backgroundColor': '#f0f2f6',
        'secondaryBackgroundColor': '#ffffff',
        'textColor': '#262730'
    }

# Business rules for fallback predictions
class BusinessRules:
    """Business rule thresholds for fallback predictions"""
    
    HIGH_RATED_THRESHOLD = 4.2
    POPULAR_THRESHOLD = 100
    PREMIUM_THRESHOLD = 400
    
    @classmethod
    def is_high_rated(cls, features: Dict[str, Any]) -> bool:
        return features.get('avg_rating', 0) >= cls.HIGH_RATED_THRESHOLD
    
    @classmethod
    def is_popular(cls, features: Dict[str, Any]) -> bool:
        return features.get('total_rating_count', 0) >= cls.POPULAR_THRESHOLD
    
    @classmethod
    def is_premium(cls, features: Dict[str, Any]) -> bool:
        return features.get('avg_price', 0) >= cls.PREMIUM_THRESHOLD

# Export configurations for easy import
app_config = AppConfig()
feature_config = FeatureConfig()
model_config = ModelConfig()
api_config = APIConfig()
streamlit_config = StreamlitConfig()
business_rules = BusinessRules()