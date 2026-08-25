"""
AQI Model Training & Evaluation Pipeline
Trains multiple model families and registers the best model based on R2 / RMSE metrics.
"""

import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Candidate Model Imports
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

# ============================================================
# PATHS CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "lahore_aqi_historical.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_72h.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# DATA PREPARATION
# ============================================================

def load_and_prepare_data():
    """Load historical features and target variable."""
    print("=" * 60)
    print("LOADING TRAINING DATA")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at: {DATA_PATH}")
        print("Generating synthetic historical dataset for demonstration...")
        df = generate_synthetic_data()
    else:
        df = pd.read_csv(DATA_PATH)

    feature_cols = [
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "surface_pressure", "wind_speed_10m", "wind_direction_10m",
        "hour", "day", "day_of_week", "month", "year", "is_weekend"
    ]
    target_col = "pm2_5"

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    return X, y, feature_cols


def generate_synthetic_data(samples=2000):
    """Fallback generator for historical AQI data if dataset file is absent."""
    np.random.seed(42)
    data = {
        "temperature_2m": np.random.uniform(5, 45, samples),
        "relative_humidity_2m": np.random.uniform(20, 95, samples),
        "precipitation": np.random.exponential(0.5, samples),
        "surface_pressure": np.random.uniform(970, 1015, samples),
        "wind_speed_10m": np.random.uniform(0.5, 15, samples),
        "wind_direction_10m": np.random.uniform(0, 360, samples),
        "hour": np.random.randint(0, 24, samples),
        "day": np.random.randint(1, 29, samples),
        "day_of_week": np.random.randint(0, 7, samples),
        "month": np.random.randint(1, 13, samples),
        "year": np.random.choice([2023, 2024, 2025], samples),
        "is_weekend": np.random.choice([0, 1], samples),
    }
    df = pd.DataFrame(data)
    # Target simulation: higher humidity & low wind = higher PM2.5
    df["pm2_5"] = (
        30.0 
        + 0.8 * df["relative_humidity_2m"] 
        - 2.5 * df["wind_speed_10m"] 
        + 1.2 * df["temperature_2m"] 
        + np.random.normal(0, 10, samples)
    ).clip(lower=0)
    return df


# ============================================================
# MODEL EXPERIMENTATION & EVALUATION
# ============================================================

def train_and_evaluate(X, y):
    """Train multiple candidate models and compare metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)

    candidate_models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
    }

    results = []
    trained_models = {}

    print("\n" + "=" * 60)
    print("TRAINING & EVALUATING CANDIDATE MODELS")
    print("=" * 60)

    for name, model in candidate_models.items():
        # Neural Network and Ridge use scaled features; Trees can use raw features
        if name in ["Ridge Regression", "Neural Network (MLP)"]:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append({
            "Model": name,
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2 Score": round(r2, 4),
        })
        trained_models[name] = model

    # Display Leaderboard
    leaderboard = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
    print("\nMODEL PERFORMANCE LEADERBOARD:")
    print("-" * 60)
    print(leaderboard.to_string(index=False))
    print("-" * 60)

    # Select Best Model based on highest R2 Score
    best_model_name = leaderboard.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    print(f"\n🏆 WINNING MODEL: {best_model_name}")
    print(f"Saving winning model to: {BEST_MODEL_PATH}")
    joblib.dump(best_model, BEST_MODEL_PATH)

    return leaderboard


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    X, y, _ = load_and_prepare_data()
    train_and_evaluate(X, y)
    print("\nTraining & evaluation workflow completed successfully!")


if __name__ == "__main__":
    main()