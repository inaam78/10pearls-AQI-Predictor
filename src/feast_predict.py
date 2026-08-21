import os
import pandas as pd
import joblib
from feast import FeatureStore

# 1. Define paths based on your project structure
REPO_PATH = "feature_store/adequate_stud/feature_repo"
MODEL_PATH = "models/random_forest_72h.pkl"

# 2. Define the exact feature order your model expects
EXPECTED_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "hour",
    "day",
    "day_of_week",
    "month",
    "year",
    "is_weekend"
]

def main():
    print("Loading features from Feast...")
    try:
        store = FeatureStore(repo_path=REPO_PATH)
        print("Feast connection successful!")
    except Exception as e:
        print(f"Error connecting to Feast: {e}")
        return

    # 3. Format the feature names for Feast (ViewName:FeatureName)
    feature_refs = [f"lahore_air_quality_features:{f}" for f in EXPECTED_FEATURES]

    print("\nRetrieving latest Lahore features...")
    try:
        feature_vector = store.get_online_features(
            features=feature_refs,
            entity_rows=[{"location_id": 1}]
        ).to_dict()
        print("Features retrieved successfully!")
    except Exception as e:
        print(f"Error retrieving features: {e}")
        return

    # 4. Convert to Pandas DataFrame and lock in the exact column order
    df_features = pd.DataFrame(feature_vector)
    X_latest = df_features[EXPECTED_FEATURES]

    # 5. Load the 72-hour Random Forest model
    print(f"\nLoading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f" Error: Model not found at {MODEL_PATH}. Check your folder structure.")
        return
        
    try:
        model = joblib.load(MODEL_PATH) 
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 6. Generate the 72-hour forecast
    print("Generating 72-hour PM2.5 forecast...")
    prediction = model.predict(X_latest)
    
    print("\n" + "="*50)
    
    # Extract the first prediction row safely
    raw_output = prediction[0]
    
    try:
        # Try to extract a single float value if it's nested (like [[value]])
        if hasattr(raw_output, "__len__") and len(raw_output) == 1:
            pred_value = float(raw_output[0])
        else:
            pred_value = float(raw_output)
            
        print(f" Predicted 72-Hour PM2.5 (Future Day 3): {pred_value:.2f}")
    except (TypeError, ValueError):
        # If the model predicts multiple outputs and cannot be converted to a single float
        print(f" Raw Predicted Array: {raw_output}")
        
    print("="*50 + "\n")

if __name__ == "__main__":
    main()