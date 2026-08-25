"""
AQI Prediction using Feast Online Feature Store
"""

import os
import sys
import joblib  # <--- UPDATED: Changed from pickle to joblib
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_REPO = os.path.join(
    PROJECT_ROOT,
    "feature_store",
    "adequate_stud",
    "feature_repo",
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "random_forest_72h.pkl",
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "scaler.pkl",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "predictions",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FEAST FEATURES
# ============================================================

FEATURE_NAMES = [
    "lahore_air_quality_features:temperature_2m",
    "lahore_air_quality_features:relative_humidity_2m",
    "lahore_air_quality_features:precipitation",
    "lahore_air_quality_features:surface_pressure",
    "lahore_air_quality_features:wind_speed_10m",
    "lahore_air_quality_features:wind_direction_10m",
    "lahore_air_quality_features:hour",
    "lahore_air_quality_features:day",
    "lahore_air_quality_features:day_of_week",
    "lahore_air_quality_features:month",
    "lahore_air_quality_features:year",
    "lahore_air_quality_features:is_weekend",
]

# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """Load trained ML model."""
    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    # UPDATED: Using joblib instead of pickle
    model = joblib.load(MODEL_PATH)

    print(f"Model loaded successfully:")
    print(MODEL_PATH)
    return model

# ============================================================
# LOAD SCALER
# ============================================================

def load_scaler():
    """Load scaler if one exists."""
    if not os.path.exists(SCALER_PATH):
        print("\nNo scaler.pkl found.")
        print("Continuing without scaler.")
        return None

    # UPDATED: Using joblib instead of pickle
    scaler = joblib.load(SCALER_PATH)

    print("Scaler loaded successfully.")
    return scaler

# ============================================================
# GET FEATURES FROM FEAST
# ============================================================

def get_feast_features():
    """Retrieve the latest Lahore features from Feast."""
    print("\n" + "=" * 60)
    print("CONNECTING TO FEAST")
    print("=" * 60)

    try:
        from feast import FeatureStore
    except ImportError:
        raise ImportError(
            "Feast is not installed in the active virtual environment."
        )

    if not os.path.exists(os.path.join(FEATURE_REPO, "feature_store.yaml")):
        raise FileNotFoundError(
            "Feast feature_store.yaml not found:\n"
            + os.path.join(FEATURE_REPO, "feature_store.yaml")
        )

    print(f"Feature repository:")
    print(FEATURE_REPO)

    store = FeatureStore(repo_path=FEATURE_REPO)
    print("\nRequesting latest features from Feast...")

    result = store.get_online_features(
        features=FEATURE_NAMES,
        entity_rows=[{"location_id": 1}],
    ).to_dict()

    df = pd.DataFrame(result)
    print("\nFeast features retrieved successfully.")

    if "location_id" in df.columns:
        df = df.drop(columns=["location_id"])

    print("\nRetrieved Features:")
    print("-" * 60)
    for column in df.columns:
        print(f"{column}: {df[column].iloc[0]}")

    return df

# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_features(df):
    """Prepare Feast data for model prediction."""
    print("\n" + "=" * 60)
    print("PREPARING MODEL FEATURES")
    print("=" * 60)

    feature_columns = [
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "surface_pressure", "wind_speed_10m", "wind_direction_10m",
        "hour", "day", "day_of_week", "month", "year", "is_weekend",
    ]

    missing = [col for col in feature_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = df[feature_columns].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isnull().any().any():
        print("\nWarning: Missing values detected.")
        X = X.fillna(X.median(numeric_only=True))

    print("Feature preparation successful.")
    return X

# ============================================================
# PREDICT
# ============================================================

def generate_forecast(model, X, start_time):
    """Generate 72-hour forecast."""
    print("\n" + "=" * 60)
    print("GENERATING 72-HOUR FORECAST")
    print("=" * 60)

    predictions = []
    current_features = X.copy()

    try:
        prediction = model.predict(current_features)
        prediction = np.asarray(prediction).flatten()
        
        print(f"Model returned {len(prediction)} prediction(s).")

        if len(prediction) >= 72:
            predictions = prediction[:72]
        else:
            if len(prediction) == 1:
                base_prediction = float(prediction[0])
                predictions = [base_prediction for _ in range(72)]
            else:
                predictions = list(prediction)
                while len(predictions) < 72:
                    predictions.append(predictions[-1])
                predictions = predictions[:72]

    except Exception as e:
        raise RuntimeError(f"Model prediction failed:\n{e}")

    timestamps = [start_time + timedelta(hours=i + 1) for i in range(72)]

    forecast = pd.DataFrame(
        {
            "forecast_hour": range(1, 73),
            "timestamp": timestamps,
            "predicted_pm2_5": predictions,
        }
    )

    forecast["predicted_pm2_5"] = forecast["predicted_pm2_5"].clip(lower=0)
    return forecast

# ============================================================
# AQI CATEGORY
# ============================================================

def pm25_category(value):
    """Simple PM2.5 category."""
    if value <= 12: return "Good"
    elif value <= 35.4: return "Moderate"
    elif value <= 55.4: return "Unhealthy for Sensitive Groups"
    elif value <= 150.4: return "Unhealthy"
    elif value <= 250.4: return "Very Unhealthy"
    else: return "Hazardous"

def add_categories(forecast):
    forecast["category"] = forecast["predicted_pm2_5"].apply(pm25_category)
    return forecast

# ============================================================
# SAVE CSV & PLOT
# ============================================================

def save_forecast(forecast):
    output_file = os.path.join(OUTPUT_DIR, "latest_72_hour_prediction.csv")
    forecast.to_csv(output_file, index=False)
    
    print("\nForecast CSV saved:")
    print(output_file)
    return output_file

def create_plot(forecast):
    output_file = os.path.join(OUTPUT_DIR, "latest_72_hour_prediction.png")
    
    plt.figure(figsize=(14, 6))
    plt.plot(forecast["timestamp"], forecast["predicted_pm2_5"], marker="o", linewidth=2, markersize=3)
    plt.title("Lahore PM2.5 72-Hour Forecast")
    plt.xlabel("Time")
    plt.ylabel("Predicted PM2.5")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    print("\nForecast plot saved:")
    print(output_file)
    return output_file

# ============================================================
# DISPLAY SUMMARY
# ============================================================

def display_summary(forecast):
    print("\n" + "=" * 60)
    print("FORECAST SUMMARY")
    print("=" * 60)
    print(f"Minimum PM2.5: {forecast['predicted_pm2_5'].min():.2f}")
    print(f"Maximum PM2.5: {forecast['predicted_pm2_5'].max():.2f}")
    print(f"Average PM2.5: {forecast['predicted_pm2_5'].mean():.2f}")
    print("\nFirst 5 predictions:")
    print(forecast.head(5).to_string(index=False))

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n")
    print("=" * 60)
    print("LAHORE AIR QUALITY PREDICTION")
    print("FEAST + MACHINE LEARNING")
    print("=" * 60)

    try:
        feast_df = get_feast_features()
        X = prepare_features(feast_df)
        model = load_model()
        scaler = load_scaler()

        if scaler is not None:
            try:
                X_model = scaler.transform(X)
                print("\nScaler applied successfully.")
            except Exception as e:
                print("\nScaler could not be applied.")
                print(f"Reason: {e}")
                print("Using raw features instead.")
                X_model = X
        else:
            X_model = X

        start_time = datetime.now()
        try:
            start_time = datetime(
                int(X["year"].iloc[0]), int(X["month"].iloc[0]),
                int(X["day"].iloc[0]), int(X["hour"].iloc[0]),
            )
        except Exception:
            pass

        print(f"\nForecast starts from: {start_time}")

        forecast = generate_forecast(model, X_model, start_time)
        forecast = add_categories(forecast)
        
        display_summary(forecast)
        save_forecast(forecast)
        create_plot(forecast)

        print("\n" + "=" * 60)
        print("PREDICTION COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("PREDICTION FAILED")
        print("=" * 60)
        print(f"\nError:\n{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()