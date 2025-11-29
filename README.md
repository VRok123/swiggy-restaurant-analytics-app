# 🍽️ Swiggy Restaurant Analytics

Machine Learning Powered Restaurant Insights and Predictions

## 🚀 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://swiggy-restaurant-analytics-app-92d47mxr4mfjz3ryld8c7q.streamlit.app/)

## 📊 Features
- **ML Predictions**: High-Rated, Popular, and Premium restaurant classifications
- **Interactive Dashboard**: Real-time feature analysis
- **Batch Processing**: Upload CSV for bulk predictions
- **Business Insights**: Actionable recommendations

## 🛠️ Tech Stack
- Streamlit
- Scikit-learn
- LightGBM & XGBoost
- Pandas & NumPy

## 🎯 Prediction Models
- **High-Rated**: Identifies restaurants with excellent consistent ratings
- **Popular**: Detects highly engaged and popular restaurants  
- **Premium**: Classifies premium-priced establishments

## 📁 Project Structure
swiggy-restaurant-analytics-app/
├── streamlit_app.py
├── requirements.txt
├── models/
│ ├── deployment_high_rated_model.pkl
│ ├── deployment_popular_model.pkl
│ └── deployment_premium_model.pkl
└── src/
├── utils.py
└── config.py
