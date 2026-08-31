import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
import requests

# ============================================================
# DATA FETCHING
# ============================================================

def get_open_meteo_weather():
    """Fetches real-time base weather for Lahore using Open-Meteo API."""
    print("Fetching live weather data from Open-Meteo...")
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=31.5497&longitude=74.3436&current=temperature_2m,relative_humidity_2m,wind_speed_10m&timezone=auto"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        current = data.get("current", {})
        return pd.DataFrame([{
            "temperature_2m": current.get("temperature_2m", 30.0),
            "relative_humidity_2m": current.get("relative_humidity_2m", 50.0),
            "wind_speed_10m": current.get("wind_speed_10m", 5.0)
        }])
    except Exception as e:
        print(f"Warning: Open-Meteo fetch failed ({e}). Using default fallback weather.")
        return pd.DataFrame([{
            "temperature_2m": 30.0, 
            "relative_humidity_2m": 50.0, 
            "wind_speed_10m": 5.0
        }])

# ============================================================
# FORECAST GENERATION
# ============================================================

def generate_72h_forecast_features(base_weather_df):
    """Generates 72 rows of future weather/time features with natural daily drift."""
    print("Generating 72-hour future feature matrix...")
    future_rows = []
    start_time = datetime.now()
    
    base_temp = float(base_weather_df["temperature_2m"].values[0])
    base_humidity = float(base_weather_df["relative_humidity_2m"].values[0])
    base_wind = float(base_weather_df["wind_speed_10m"].values[0])

    for i in range(72):
        target_time = start_time + timedelta(hours=i)
        hour = target_time.hour
        
        # ADD DAILY DRIFT: This fixes the repeating sine wave!
        day_index = i // 24 
        daily_temp_drift = day_index * 1.5  
        daily_hum_drift = day_index * -3.0  
        daily_wind_drift = day_index * 0.5  

        temp_variation = 5 * np.sin(2 * np.pi * (hour - 6) / 24) 
        humidity_variation = -10 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        row = {
            "temperature_2m": base_temp + temp_variation + daily_temp_drift,
            "relative_humidity_2m": np.clip(base_humidity + humidity_variation + daily_hum_drift, 10, 100),
            "precipitation": 0.0,
            "surface_pressure": 1010.0,
            "wind_speed_10m": max(1.0, base_wind + daily_wind_drift + np.random.normal(0, 0.5)),
            "wind_direction_10m": 180.0,
            "hour": hour,
            "day": target_time.day,
            "day_of_week": target_time.weekday(),
            "month": target_time.month,
            "year": target_time.year,
            "is_weekend": 1 if target_time.weekday() >= 5 else 0
        }
        future_rows.append(row)
        
    return pd.DataFrame(future_rows)

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STARTING 72-HOUR AQI INFERENCE PIPELINE")
    print("=" * 60)
    
    # 1. Load Model and Scaler
    model_path = "models/random_forest_72h.pkl"
    scaler_path = "models/scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or Scaler not found. Run train_models.py first.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # 2. Define the exact 12 columns the model was trained on
    expected_features = [
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "surface_pressure", "wind_speed_10m", "wind_direction_10m",
        "hour", "day", "day_of_week", "month", "year", "is_weekend"
    ]

    # 3. Generate Future Weather Data
    base_weather_df = get_open_meteo_weather()
    future_features_df = generate_72h_forecast_features(base_weather_df)

    # Ensure columns match training exactly
    future_features_df = future_features_df[expected_features]

    # 4. Scale the Data (Required for Ridge Regression)
    print("Scaling features and generating predictions...")
    X_scaled = scaler.transform(future_features_df)

    # 5. Predict PM2.5 (and prevent negative values)
    predictions = model.predict(X_scaled)
    predictions = [max(0.0, p) for p in predictions]
    
    # 6. Save Results
    future_features_df["predicted_pm25"] = predictions
    os.makedirs("outputs", exist_ok=True)
    
    output_csv = "outputs/72h_predictions.csv"
    future_features_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Inference complete! Predictions saved to {output_csv}")
    print("\nPreview of first 5 hours:")
    print(future_features_df[["hour", "temperature_2m", "predicted_pm25"]].head())