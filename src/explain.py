import os
import joblib
import shap
import pandas as pd
import numpy as np

def generate_shap_explainer():
    print("Initializing SHAP Explainability Pipeline...")
    
    # Paths
    model_path = "models/random_forest_72h.pkl" # Note: contains your Ridge model
    scaler_path = "models/scaler.pkl"
    data_path = "data/processed/lahore_features.csv"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler artifacts missing. Train the model first.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        print("Historical feature dataset not found. Generating sample background data...")
        return None
        
    feature_cols = [
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "surface_pressure", "wind_speed_10m", "wind_direction_10m",
        "hour", "day", "day_of_week", "month", "year", "is_weekend"
    ]
    
    X = df[feature_cols].copy()
    X_scaled = scaler.transform(X)
    
    # Initialize SHAP LinearExplainer for Ridge Regression (or TreeExplainer if using Tree models)
    # Using background sample of 100 rows for speed
    background_data = X_scaled[:100]
    explainer = shap.LinearExplainer(model, background_data)
    
    # Compute SHAP values for recent data
    shap_values = explainer(X_scaled[:50])
    
    print("✅ SHAP explainer successfully initialized and evaluated!")
    return explainer, shap_values

if __name__ == "__main__":
    generate_shap_explainer()