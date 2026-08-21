import os
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# LAHORE AQI - 72 HOUR PM2.5 MODEL TRAINING

print("=" * 60)
print("LAHORE AQI - 72 HOUR PM2.5 MODEL TRAINING")
print("=" * 60)


# 1. LOAD DATA

print("\nLoading datasets...")

train_path = "data/splits/train.csv"
validation_path = "data/splits/validation.csv"
test_path = "data/splits/test.csv"

train_df = pd.read_csv(train_path)
validation_df = pd.read_csv(validation_path)
test_df = pd.read_csv(test_path)

print("Training shape:", train_df.shape)
print("Validation shape:", validation_df.shape)
print("Testing shape:", test_df.shape)


# 2. IDENTIFY TARGET COLUMNS

target_columns = [
    f"pm2_5_t+{i}"
    for i in range(1, 73)
]

print("\nNumber of target columns:", len(target_columns))


# 3. SELECT FEATURES

feature_columns = [
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

print("\nFeatures:")
for feature in feature_columns:
    print("-", feature)


# 4. PREPARE X AND Y

X_train = train_df[feature_columns]
y_train = train_df[target_columns]

X_validation = validation_df[feature_columns]
y_validation = validation_df[target_columns]

X_test = test_df[feature_columns]
y_test = test_df[target_columns]

print("\nTraining features:", X_train.shape)
print("Training targets:", y_train.shape)

print("Validation features:", X_validation.shape)
print("Validation targets:", y_validation.shape)

print("Testing features:", X_test.shape)
print("Testing targets:", y_test.shape)


# 5. TRAIN MODEL

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST MODEL")
print("=" * 60)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nTraining started...")

model.fit(X_train, y_train)

print("Training completed successfully!")


# 6. VALIDATION PREDICTIONS

print("\nGenerating validation predictions...")

validation_predictions = model.predict(X_validation)

print("Validation predictions generated.")


# 7. VALIDATION METRICS

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

overall_mae = mean_absolute_error(
    y_validation,
    validation_predictions
)

overall_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        validation_predictions
    )
)

print(f"\nOverall MAE:  {overall_mae:.2f}")
print(f"Overall RMSE: {overall_rmse:.2f}")


# ------------------------------------------------------------
# 8. METRICS FOR IMPORTANT FORECAST HOURS
# ------------------------------------------------------------

print("\nForecast-specific performance:")

forecast_hours = [1, 6, 12, 24, 48, 72]

for hour in forecast_hours:

    index = hour - 1

    mae = mean_absolute_error(
        y_validation.iloc[:, index],
        validation_predictions[:, index]
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_validation.iloc[:, index],
            validation_predictions[:, index]
        )
    )

    print(
        f"{hour:2d}-hour -> "
        f"MAE: {mae:.2f} | "
        f"RMSE: {rmse:.2f}"
    )


# 9. FEATURE IMPORTANCE

print("\nCalculating feature importance...")

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": importance
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)


# 10. SAVE MODEL

os.makedirs("models", exist_ok=True)
os.makedirs("results/models", exist_ok=True)

model_path = "models/random_forest_72h.pkl"

joblib.dump(model, model_path)

print("\nModel saved to:")
print(model_path)


# 11. SAVE FEATURE IMPORTANCE

importance_path = "results/models/feature_importance.csv"

feature_importance.to_csv(
    importance_path,
    index=False
)

print("\nFeature importance saved to:")
print(importance_path)


# 12. SAVE VALIDATION PREDICTIONS

prediction_df = pd.DataFrame(
    validation_predictions,
    columns=target_columns
)

prediction_df.insert(
    0,
    "timestamp",
    validation_df["timestamp"].values
)

prediction_path = "results/models/validation_predictions.csv"

prediction_df.to_csv(
    prediction_path,
    index=False
)

print("\nValidation predictions saved to:")
print(prediction_path)
print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)