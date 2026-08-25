import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_72h_forecast_features(base_weather_df):
    """Generates 72 rows of future features with natural hourly variations."""
    future_rows = []
    start_time = datetime.now()
    
    # Extract base values from the current weather API fetch
    base_temp = float(base_weather_df["temperature_2m"].values[0]) if "temperature_2m" in base_weather_df else 25.0
    base_humidity = float(base_weather_df["relative_humidity_2m"].values[0]) if "relative_humidity_2m" in base_weather_df else 50.0
    base_wind = float(base_weather_df["wind_speed_10m"].values[0]) if "wind_speed_10m" in base_weather_df else 5.0

    for i in range(72):
        target_time = start_time + timedelta(hours=i)
        hour = target_time.hour
        
        # Create a natural diurnal cycle (colder at night, warmer in afternoon)
        temp_variation = 5 * np.sin(2 * np.pi * (hour - 6) / 24) 
        humidity_variation = -10 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        row = {
            "temperature_2m": base_temp + temp_variation,
            "relative_humidity_2m": np.clip(base_humidity + humidity_variation, 10, 100),
            "precipitation": 0.0,
            "surface_pressure": 1010.0,
            "wind_speed_10m": max(1.0, base_wind + np.random.normal(0, 0.5)),
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