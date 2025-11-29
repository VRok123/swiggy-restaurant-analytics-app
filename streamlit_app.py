# streamlit_app.py - COMPLETE STANDALONE VERSION FOR DEPLOYMENT
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import InconsistentVersionWarning
import warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add path to import your models
sys.path.append('src')

# Try to import local modules, but continue if they fail
try:
    from utils import logger, setup_plotting, load_model
    from config import *
    LOCAL_IMPORTS = True
except ImportError as e:
    LOCAL_IMPORTS = False
    # Create simple fallbacks
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

class StandaloneStreamlitDashboard:
    def __init__(self):
        self.setup_page()
        self.ml_models = self.load_ml_models()
    
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Swiggy Restaurant Analytics",
            page_icon="🍽️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #FF6B6B;
            text-align: center;
            margin-bottom: 2rem;
        }
        .prediction-card {
            padding: 1.5rem;
            border-radius: 10px;
            background-color: #f0f2f6;
            margin: 1rem 0;
        }
        .metric-card {
            padding: 1rem;
            border-radius: 10px;
            background-color: #ffffff;
            border-left: 4px solid #FF6B6B;
        }
        .ml-success {
            border-left: 4px solid #28a745;
        }
        .ml-warning {
            border-left: 4px solid #ffc107;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @st.cache_resource
    def load_ml_models(_self):
        """Load ML models directly in Streamlit"""
        models = {}
        model_files = {
            'high_rated': 'models/deployment_high_rated_model.pkl',
            'popular': 'models/deployment_popular_model.pkl', 
            'premium': 'models/deployment_premium_model.pkl'
        }
        
        for model_name, model_path in model_files.items():
            try:
                if LOCAL_IMPORTS:
                    models[model_name] = load_model(model_path)
                else:
                    # Fallback loading method
                    if os.path.exists(model_path):
                        models[model_name] = joblib.load(model_path)
                    else:
                        st.warning(f"Model file not found: {model_path}")
            except Exception as e:
                st.warning(f"Could not load {model_name} model: {e}")
        
        return models if models else None
    
    def prepare_features_optimized(self, features):
        """Same feature preparation as your FastAPI app - CORRECT 30-feature mapping"""
        # Initialize feature vector with zeros - EXACTLY 30 features
        feature_vector = [0.0] * 30
        
        # Set basic features using direct index assignment
        feature_vector[0] = float(features.get('avg_price', 0))
        feature_vector[1] = float(features.get('dish_count', 0))
        feature_vector[2] = float(features.get('total_rating_count', 0))
        feature_vector[3] = float(features.get('rating_std', 0))
        feature_vector[4] = float(features.get('price_std', 0))
        feature_vector[5] = float(features.get('category_diversity', 0))
        feature_vector[6] = float(features.get('price_to_dish_ratio', 0))
        feature_vector[7] = float(features.get('rating_count_per_dish', 0))
        feature_vector[8] = float(features.get('has_high_variance', 0))
        
        # CORRECT city encoding - using exact training data city columns
        city = features.get('city', 'other').lower()
        
        # Map to exact city columns from training data
        if city == 'mumbai':
            feature_vector[17] = 1.0   # city_mumbai
        elif city == 'delhi':
            feature_vector[18] = 1.0   # city_new delhi
        elif city in ['bangalore', 'bengaluru']:
            feature_vector[10] = 1.0   # city_bengaluru
        elif city == 'chennai':
            feature_vector[12] = 1.0   # city_chennai
        elif city == 'kolkata':
            feature_vector[15] = 1.0   # city_kolkata
        elif city == 'hyderabad':
            feature_vector[13] = 1.0   # city_hyderabad
        elif city == 'ahmedabad':
            feature_vector[9] = 1.0    # city_ahmedabad
        elif city == 'chandigarh':
            feature_vector[11] = 1.0   # city_chandigarh
        elif city == 'lucknow':
            feature_vector[16] = 1.0   # city_lucknow
        elif city == 'jaipur':
            feature_vector[14] = 1.0   # city_jaipur
        else:
            feature_vector[19] = 1.0   # city_other
        
        # Set derived features efficiently (using remaining indices 20-29)
        rating_std = features.get('rating_std', 1.0)
        feature_vector[20] = 1.0 if rating_std < 0.3 else 0.0
        
        avg_price = features.get('avg_price', 0)
        if avg_price > 500:
            feature_vector[21] = 2.0
        elif avg_price > 300:
            feature_vector[21] = 1.0
        else:
            feature_vector[21] = 0.0
        
        rating_count = features.get('total_rating_count', 0)
        feature_vector[22] = min(rating_count / 1000.0, 1.0)
        
        return np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    
    def make_ml_prediction_standalone(self, features, prediction_type):
        """Make prediction using loaded models"""
        if self.ml_models and prediction_type in self.ml_models:
            try:
                feature_array = self.prepare_features_optimized(features)
                model = self.ml_models[prediction_type]
                probability = model.predict_proba(feature_array)[0, 1]
                prediction = int(probability >= 0.5)
                    
                # Determine confidence level
                if probability >= 0.8:
                    confidence = "high"
                elif probability >= 0.6:
                    confidence = "medium"
                else:
                    confidence = "low"
                    
                return {
                    'prediction': prediction,
                    'probability': round(probability, 4),
                    'confidence': confidence,
                    'features_used': feature_array.shape[1]
                }
            except Exception as e:
                st.error(f"Prediction error for {prediction_type}: {e}")
            
        return None

    def render_header(self):
        """Render the dashboard header"""
        st.markdown('<h1 class="main-header">🍽️ Swiggy Restaurant Analytics</h1>', unsafe_allow_html=True)
        st.markdown("### Machine Learning Powered Restaurant Insights - DEPLOYED VERSION")
            
        # Deployment status
        if self.ml_models:
            st.success("✅ ML Models: Loaded Successfully - Standalone Mode")
            st.sidebar.markdown("### 📊 Deployment Status")
            st.sidebar.success("**Mode**: Standalone (No API Required)")
            st.sidebar.write(f"**Models Loaded**: {', '.join(self.ml_models.keys())}")
            st.sidebar.write("**Feature Mapping**: Correct 30-feature")
            st.sidebar.write("**Performance**: Optimized")
        else:
            st.warning("⚠️ ML Models: Using Rule-Based Analysis")
            st.sidebar.markdown("### 📊 Deployment Status")
            st.sidebar.warning("**Mode**: Rule-Based Fallback")
            st.sidebar.write("**Models**: Not loaded")
            st.sidebar.write("**Using**: Business rule thresholds")
            
        return True
    
    def render_sidebar(self):
        """Render the sidebar with input controls"""
        st.sidebar.markdown("## 🔧 Restaurant Features")
        
        # Basic restaurant information
        st.sidebar.markdown("### 📍 Basic Information")
        city = st.sidebar.selectbox(
            "City",
            ["mumbai", "delhi", "bangalore", "chennai", "kolkata", "hyderabad", "pune", "ahmedabad", "other"]
        )
        
        st.sidebar.markdown("### 💰 Pricing & Menu")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            avg_price = st.sidebar.number_input("Average Price (INR)", min_value=0, max_value=2000, value=650, step=10)
            dish_count = st.sidebar.number_input("Number of Dishes", min_value=1, max_value=500, value=45, step=1)
        
        with col2:
            price_std = st.sidebar.number_input("Price Standard Deviation", min_value=0.0, max_value=500.0, value=180.0, step=10.0)
            category_diversity = st.sidebar.number_input("Category Diversity", min_value=1, max_value=20, value=6, step=1)
        
        st.sidebar.markdown("### ⭐ Ratings & Popularity")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            avg_rating = st.sidebar.slider("Average Rating", min_value=1.0, max_value=5.0, value=4.6, step=0.1)
            median_rating = st.sidebar.slider("Median Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
        
        with col2:
            total_rating_count = st.sidebar.number_input("Total Rating Count", min_value=0, max_value=50000, value=3500, step=50)
            rating_std = st.sidebar.slider("Rating Standard Deviation", min_value=0.0, max_value=2.0, value=0.2, step=0.1)
        
        # Derived features (calculated automatically)
        price_to_dish_ratio = avg_price / max(dish_count, 1)
        rating_count_per_dish = total_rating_count / max(dish_count, 1)
        has_high_variance = 1 if rating_std > 0.5 else 0
        price_volatility = price_std / max(avg_price, 1)
        
        # Store all features
        features = {
            'avg_price': avg_price,
            'dish_count': dish_count,
            'total_rating_count': total_rating_count,
            'avg_rating': avg_rating,
            'median_rating': median_rating,
            'rating_std': rating_std,
            'price_std': price_std,
            'category_diversity': category_diversity,
            'price_to_dish_ratio': round(price_to_dish_ratio, 2),
            'rating_count_per_dish': round(rating_count_per_dish, 2),
            'has_high_variance': has_high_variance,
            'price_volatility': round(price_volatility, 2),
            'city': city
        }
        
        # Show derived features
        with st.sidebar.expander("📈 Derived Features"):
            st.write(f"**Price per Dish:** ₹{price_to_dish_ratio:.2f}")
            st.write(f"**Ratings per Dish:** {rating_count_per_dish:.1f}")
            st.write(f"**Price Volatility:** {price_volatility:.2f}")
            st.write(f"**High Rating Variance:** {'Yes' if has_high_variance else 'No'}")
        
        # Quick tips
        with st.sidebar.expander("💡 Quick Tips"):
            st.write("**For YES predictions:**")
            st.write("- Rating: 4.5+")
            st.write("- Total Ratings: 2500+") 
            st.write("- Price: ₹600+")
            st.write("- City: Mumbai/Delhi")
        
        return features
    
    def render_prediction_cards(self, features):
        """Render prediction results as cards"""
        st.markdown("## 🎯 Restaurant Predictions")
        
        # Try to get ML predictions
        ml_predictions = {}
        ml_success = False
        
        if self.ml_models:
            for prediction_type in ['high_rated', 'popular', 'premium']:
                prediction = self.make_ml_prediction_standalone(features, prediction_type)
                if prediction:
                    ml_predictions[prediction_type] = prediction
                    ml_success = True
        
        # Show status
        if ml_success:
            st.success("✅ ML Predictions: Active - Using trained machine learning models")
        else:
            st.warning("⚠️ ML Predictions: Fallback Mode - Using rule-based analysis")
        
        # Create columns for predictions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'high_rated' in ml_predictions:
                self.render_ml_prediction_card(ml_predictions['high_rated'], "⭐ High-Rated Restaurant")
            else:
                self.render_rule_based_card(
                    "⭐ High-Rated Restaurant",
                    "Restaurants with rating ≥ 4.2",
                    features['avg_rating'] >= 4.2,
                    features['avg_rating'] / 5.0,
                    "High-rated probability based on current rating"
                )
        
        with col2:
            if 'popular' in ml_predictions:
                self.render_ml_prediction_card(ml_predictions['popular'], "🔥 Popular Restaurant")
            else:
                self.render_rule_based_card(
                    "🔥 Popular Restaurant", 
                    "Restaurants with 100+ ratings",
                    features['total_rating_count'] >= 100,
                    min(features['total_rating_count'] / 1000.0, 1.0),
                    "Popularity probability based on rating count"
                )
        
        with col3:
            if 'premium' in ml_predictions:
                self.render_ml_prediction_card(ml_predictions['premium'], "💎 Premium Restaurant")
            else:
                self.render_rule_based_card(
                    "💎 Premium Restaurant",
                    "Restaurants in top 20% price range",
                    features['avg_price'] > 400,
                    min(features['avg_price'] / 1000.0, 1.0),
                    "Premium probability based on pricing"
                )
        
        # Show technical details in expander
        with st.expander("🔧 Technical Details"):
            if ml_success:
                st.success("**ML Integration**: ✅ Working - Models receiving proper 30-feature format")
                st.info(f"**Features Used**: {ml_predictions.get('high_rated', {}).get('features_used', 'N/A')} features per prediction")
                st.info("**Deployment**: Standalone - No API required")
            else:
                st.error("**Current Issue**: Models not loaded or prediction failed")
                st.info("**Solution**: Check model files in models/ directory")
                st.write("**Temporary Solution**: Using rule-based analysis based on business thresholds")

    def render_ml_prediction_card(self, prediction_data, title):
        """Render ML prediction card"""
        prediction = prediction_data.get('prediction', 0)
        probability = prediction_data.get('probability', 0.5)
        confidence = prediction_data.get('confidence', 'medium')
        
        if prediction == 1:
            emoji = "✅"
            message = "YES"
            bg_color = "#d4edda"
            text_color = "#155724"
            border_color = "green"
        else:
            emoji = "❌"
            message = "NO"
            bg_color = "#f8d7da"
            text_color = "#721c24"
            border_color = "red"
        
        st.markdown(f"""
        <div style="padding: 1.5rem; border-radius: 10px; background-color: {bg_color}; border-left: 4px solid {border_color}; margin: 1rem 0;" class="ml-success">
            <h3 style="margin: 0; color: {text_color};">{emoji} {title}</h3>
            <p style="margin: 0.5rem 0; color: {text_color}; opacity: 0.8;">ML Model Prediction</p>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                <span style="font-size: 1.5rem; font-weight: bold; color: {text_color};">{message}</span>
                <span style="font-size: 1.2rem; color: {text_color};" title="ML Model Confidence">{probability:.1%}</span>
            </div>
            <div style="margin-top: 0.5rem;">
                <small style="color: {text_color}; opacity: 0.8;">Confidence: {confidence.upper()}</small>
            </div>
            <div style="margin-top: 0.5rem;">
                <small style="color: {text_color}; opacity: 0.8;">🎯 ML Model</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_rule_based_card(self, title, description, prediction, probability, tooltip):
        """Render rule-based prediction card"""
        if prediction:
            emoji = "✅"
            message = "YES"
            bg_color = "#e2e3e5"
            text_color = "#383d41"
            border_color = "#6c757d"
            confidence = "high" if probability > 0.8 else "medium" if probability > 0.6 else "low"
        else:
            emoji = "❌"
            message = "NO"
            bg_color = "#e2e3e5"
            text_color = "#383d41"
            border_color = "#6c757d"
            confidence = "low"
        
        st.markdown(f"""
        <div style="padding: 1.5rem; border-radius: 10px; background-color: {bg_color}; border-left: 4px solid {border_color}; margin: 1rem 0;" class="ml-warning">
            <h3 style="margin: 0; color: {text_color};">{emoji} {title}</h3>
            <p style="margin: 0.5rem 0; color: {text_color}; opacity: 0.8;">{description}</p>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                <span style="font-size: 1.5rem; font-weight: bold; color: {text_color};">{message}</span>
                <span style="font-size: 1.2rem; color: {text_color};" title="{tooltip}">{probability:.1%}</span>
            </div>
            <div style="margin-top: 0.5rem;">
                <small style="color: {text_color}; opacity: 0.8;">Confidence: {confidence.upper()}</small>
            </div>
            <div style="margin-top: 0.5rem;">
                <small style="color: {text_color}; opacity: 0.8;">📊 Rule-Based</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_feature_analysis(self, features):
        """Render feature analysis and insights"""
        st.markdown("## 📊 Feature Analysis")
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["📈 Feature Overview", "🎯 Business Insights", "📋 Feature Data"])
        
        with tab1:
            self.render_feature_overview(features)
        
        with tab2:
            self.render_business_insights(features)
        
        with tab3:
            self.render_feature_data(features)

    def render_feature_overview(self, features):
        """Render feature overview with visualizations"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Price and rating metrics
            st.subheader("💰 Pricing Metrics")
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Average Price", f"₹{features['avg_price']}")
                st.metric("Price per Dish", f"₹{features['price_to_dish_ratio']}")
            with metric_col2:
                st.metric("Price Volatility", f"{features['price_volatility']:.2f}")
                st.metric("Dish Count", features['dish_count'])
        
        with col2:
            # Rating metrics
            st.subheader("⭐ Rating Metrics")
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Average Rating", f"{features['avg_rating']}/5")
                st.metric("Total Ratings", f"{features['total_rating_count']:,}")
            with metric_col2:
                st.metric("Rating Consistency", "High" if features['has_high_variance'] == 0 else "Low")
                st.metric("Ratings per Dish", f"{features['rating_count_per_dish']:.1f}")
        
        # Create a simple bar chart for key metrics
        st.subheader("📊 Key Metrics Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics_to_plot = ['avg_price', 'dish_count', 'total_rating_count', 'category_diversity']
        metric_names = ['Avg Price', 'Dish Count', 'Rating Count', 'Categories']
        values = [features[metric] for metric in metrics_to_plot]
        
        # Normalize values for better visualization
        max_val = max(values) if max(values) > 0 else 1
        normalized_values = [v / max_val * 100 for v in values]
        
        bars = ax.bar(metric_names, normalized_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_ylabel('Normalized Score (%)')
        ax.set_title('Restaurant Metrics (Normalized)')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:,}', ha='center', va='bottom')
        
        st.pyplot(fig)

    def render_business_insights(self, features):
        """Render business insights based on features"""
        st.subheader("💡 Business Insights")
        
        insights = []
        
        # Generate insights based on feature values
        if features['avg_rating'] >= 4.2:
            insights.append("✅ **High Potential**: Current ratings suggest this could be a high-rated restaurant")
        else:
            insights.append("📈 **Improvement Opportunity**: Ratings are below the high-rated threshold (4.2+)")
        
        if features['total_rating_count'] >= 100:
            insights.append("🔥 **Popular Spot**: Has sufficient ratings to be considered popular")
        else:
            insights.append("👥 **Growth Potential**: Needs more ratings to reach popular status")
        
        if features['avg_price'] > 400:
            insights.append("💎 **Premium Positioning**: Pricing suggests a premium restaurant")
        else:
            insights.append("💰 **Value Focus**: Competitive pricing indicates value positioning")
        
        if features['rating_std'] < 0.5:
            insights.append("🎯 **Consistent Quality**: Low rating variance indicates consistent customer experience")
        else:
            insights.append("⚠️ **Inconsistent Experience**: High rating variance suggests mixed customer feedback")
        
        if features['dish_count'] > 50:
            insights.append("📋 **Extensive Menu**: Wide variety of dishes available")
        else:
            insights.append("🍽️ **Focused Menu**: Curated selection of dishes")
        
        # Display insights
        for insight in insights:
            st.write(insight)
        
        # Recommendations
        st.subheader("🎯 Recommendations")
        if features['avg_rating'] < 4.2 and features['total_rating_count'] < 100:
            st.info("**Focus on Quality & Marketing**: Improve food quality and actively encourage customer reviews to boost ratings and popularity")
        elif features['avg_rating'] >= 4.2 and features['total_rating_count'] < 100:
            st.info("**Leverage High Ratings**: Use excellent ratings in marketing to attract more customers and increase review count")
        elif features['avg_price'] > 500:
            st.info("**Premium Experience**: Ensure service and ambiance match the premium pricing to justify the cost")
        else:
            st.info("**Solid Foundation**: Maintain current quality standards while exploring opportunities for menu expansion or premium offerings")

    def render_feature_data(self, features):
        """Render feature data without using dataframe to avoid serialization issues"""
        st.subheader("📋 Feature Data")
        
        # Display features in a clean format without using dataframe
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**💰 Pricing & Menu**")
            st.write(f"- Average Price: ₹{features['avg_price']}")
            st.write(f"- Dish Count: {features['dish_count']}")
            st.write(f"- Price Std Dev: {features['price_std']}")
            st.write(f"- Category Diversity: {features['category_diversity']}")
            st.write(f"- Price per Dish: ₹{features['price_to_dish_ratio']}")
            st.write(f"- Price Volatility: {features['price_volatility']:.2f}")
        
        with col2:
            st.write("**⭐ Ratings & Popularity**")
            st.write(f"- Average Rating: {features['avg_rating']}/5")
            st.write(f"- Median Rating: {features['median_rating']}/5")
            st.write(f"- Total Ratings: {features['total_rating_count']:,}")
            st.write(f"- Rating Std Dev: {features['rating_std']}")
            st.write(f"- Ratings per Dish: {features['rating_count_per_dish']:.1f}")
            st.write(f"- Rating Consistency: {'High' if features['has_high_variance'] == 0 else 'Low'}")
        
        st.write("**📍 Location**")
        st.write(f"- City: {features['city'].title()}")

    def render_batch_analysis(self):
        """Render batch analysis section"""
        st.markdown("## 📊 Batch Analysis")
        
        st.info("💡 **Batch Analysis**: Upload multiple restaurant data for bulk predictions")
        
        uploaded_file = st.file_uploader("Upload CSV file with restaurant data", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                batch_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {len(batch_data)} restaurants")
                
                # Show sample data
                with st.expander("📋 View Uploaded Data"):
                    st.dataframe(batch_data.head())
                
                # Check required columns
                required_columns = ['avg_price', 'dish_count', 'total_rating_count', 'avg_rating', 'city']
                missing_columns = [col for col in required_columns if col not in batch_data.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                else:
                    if st.button("🚀 Run Batch Predictions"):
                        with st.spinner("Running batch predictions..."):
                            results = []
                            for idx, row in batch_data.iterrows():
                                features = {
                                    'avg_price': row['avg_price'],
                                    'dish_count': row['dish_count'],
                                    'total_rating_count': row['total_rating_count'],
                                    'avg_rating': row['avg_rating'],
                                    'median_rating': row.get('median_rating', row['avg_rating']),
                                    'rating_std': row.get('rating_std', 0.3),
                                    'price_std': row.get('price_std', 150.0),
                                    'category_diversity': row.get('category_diversity', 6),
                                    'price_to_dish_ratio': row['avg_price'] / max(row['dish_count'], 1),
                                    'rating_count_per_dish': row['total_rating_count'] / max(row['dish_count'], 1),
                                    'has_high_variance': 1 if row.get('rating_std', 0.3) > 0.5 else 0,
                                    'price_volatility': row.get('price_std', 150.0) / max(row['avg_price'], 1),
                                    'city': row['city']
                                }
                                
                                # Get predictions
                                prediction_results = {'restaurant_id': idx}
                                for pred_type in ['high_rated', 'popular', 'premium']:
                                    pred = self.make_ml_prediction_standalone(features, pred_type)
                                    if pred:
                                        prediction_results[f'{pred_type}_pred'] = pred['prediction']
                                        prediction_results[f'{pred_type}_prob'] = pred['probability']
                                    else:
                                        prediction_results[f'{pred_type}_pred'] = 0
                                        prediction_results[f'{pred_type}_prob'] = 0.0
                                
                                results.append(prediction_results)
                            
                            # Create results dataframe
                            results_df = pd.DataFrame(results)
                            st.success(f"✅ Completed predictions for {len(results_df)} restaurants")
                            
                            # Show results
                            st.subheader("📈 Batch Prediction Results")
                            st.dataframe(results_df)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv"
                            )
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            # Placeholder for future batch functionality
            with st.expander("🔮 Batch Analysis Features"):
                st.write("""
                **Batch Analysis Features:**
                
                - 📁 **CSV Upload**: Upload restaurant data in bulk
                - 🚀 **Batch Predictions**: Get predictions for multiple restaurants at once
                - 📊 **Comparative Analysis**: Compare multiple restaurants side by side
                - 📥 **Export Results**: Download predictions as CSV
                - 📈 **Trend Analysis**: Identify patterns across multiple restaurants
                
                **Required CSV Columns:**
                - avg_price, dish_count, total_rating_count, avg_rating, city
                - Optional: median_rating, rating_std, price_std, category_diversity
                """)

    def run(self):
        """Run the Streamlit dashboard"""
        # Render header
        self.render_header()
        
        # Create main layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            features = self.render_sidebar()
        
        with col2:
            self.render_prediction_cards(features)
            self.render_feature_analysis(features)
        
        # Batch analysis at the bottom
        self.render_batch_analysis()

def main():
    """Main function for standalone deployment"""
    print("🚀 Starting Standalone Streamlit Dashboard - DEPLOYMENT READY")
    print("=" * 70)
    
    try:
        # Initialize and run the dashboard
        dashboard = StandaloneStreamlitDashboard()
        dashboard.run()
        
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        logger.error(f"❌ Dashboard failed: {e}")

if __name__ == "__main__":
    main()