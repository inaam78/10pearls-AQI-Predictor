import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# CONFIGURATION

MODEL_PATH = "models/random_forest_72h.pkl"
TEST_PATH = "data/splits/test.csv"

RESULTS_DIR = "results/models"

PERFORMANCE_PATH = os.path.join(
    RESULTS_DIR,
    "test_performance.csv"
)

PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR,
    "test_predictions.csv"
)


# CREATE RESULTS DIRECTORY

os.makedirs(RESULTS_DIR, exist_ok=True)


# HEADER

print("=" * 70)
print("LAHORE AQI - FINAL MODEL EVALUATION")
print("=" * 70)


# LOAD MODEL

print("\nLoading trained model...")

try:

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully!")

except Exception as e:

    print("\nERROR: Could not load model.")
    print(str(e))
    raise


# LOAD TEST DATA

print("\nLoading test dataset...")

test_df = pd.read_csv(TEST_PATH)

print("Test dataset loaded successfully!")
print(f"Test shape: {test_df.shape}")


# FEATURE COLUMNS

FEATURE_COLUMNS = [
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


# TARGET COLUMNS

TARGET_COLUMNS = [
    f"pm2_5_t+{i}"
    for i in range(1, 73)
]


# CHECK COLUMNS

missing_features = [
    col for col in FEATURE_COLUMNS
    if col not in test_df.columns
]

missing_targets = [
    col for col in TARGET_COLUMNS
    if col not in test_df.columns
]


if missing_features:

    raise ValueError(
        f"Missing feature columns: {missing_features}"
    )


if missing_targets:

    raise ValueError(
        f"Missing target columns: {missing_targets}"
    )


# PREPARE DATA

X_test = test_df[FEATURE_COLUMNS]

y_test = test_df[TARGET_COLUMNS]


print(f"\nTest features: {X_test.shape}")
print(f"Test targets: {y_test.shape}")



# CHECK MISSING VALUES

missing_features_count = X_test.isna().sum().sum()
missing_targets_count = y_test.isna().sum().sum()


print("\nMissing values:")
print(f"Features: {missing_features_count}")
print(f"Targets: {missing_targets_count}")


if missing_features_count > 0:

    raise ValueError(
        "Test features contain missing values."
    )


if missing_targets_count > 0:

    raise ValueError(
        "Test targets contain missing values."
    )


# GENERATE PREDICTIONS

print("\nGenerating test predictions...")

y_pred = model.predict(X_test)

print("Test predictions generated successfully!")

print(f"Prediction shape: {y_pred.shape}")


# CONVERT TO NUMPY


y_test_np = y_test.to_numpy()

y_pred_np = np.asarray(y_pred)


# OVERALL METRICS

overall_mae = mean_absolute_error(
    y_test_np.flatten(),
    y_pred_np.flatten()
)


overall_rmse = np.sqrt(
    mean_squared_error(
        y_test_np.flatten(),
        y_pred_np.flatten()
    )
)


overall_r2 = r2_score(
    y_test_np.flatten(),
    y_pred_np.flatten()
)


print("\n" + "=" * 70)
print("OVERALL MODEL PERFORMANCE")
print("=" * 70)

print(f"\nOverall MAE : {overall_mae:.2f}")
print(f"Overall RMSE: {overall_rmse:.2f}")
print(f"Overall R²  : {overall_r2:.4f}")


# FORECAST-SPECIFIC PERFORMANCE

print("\n" + "=" * 70)
print("FORECAST-SPECIFIC PERFORMANCE")
print("=" * 70)


forecast_hours = [
    1,
    6,
    12,
    24,
    48,
    72
]


performance_results = []


for hour in forecast_hours:

    index = hour - 1

    actual = y_test_np[:, index]

    predicted = y_pred_np[:, index]


    mae = mean_absolute_error(
        actual,
        predicted
    )


    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )


    r2 = r2_score(
        actual,
        predicted
    )


    print(
        f"{hour:2d}-hour -> "
        f"MAE: {mae:.2f} | "
        f"RMSE: {rmse:.2f} | "
        f"R²: {r2:.4f}"
    )


    performance_results.append({

        "forecast_hour": hour,

        "mae": mae,

        "rmse": rmse,

        "r2": r2
    })


# ALL 72 HOURS PERFORMANCE

print("\nCalculating performance for all 72 forecast hours...")


all_hour_results = []


for hour in range(1, 73):

    index = hour - 1

    actual = y_test_np[:, index]

    predicted = y_pred_np[:, index]


    mae = mean_absolute_error(
        actual,
        predicted
    )


    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )


    r2 = r2_score(
        actual,
        predicted
    )


    all_hour_results.append({

        "forecast_hour": hour,

        "mae": mae,

        "rmse": rmse,

        "r2": r2
    })


# SAVE PERFORMANCE

performance_df = pd.DataFrame(
    all_hour_results
)


performance_df.to_csv(
    PERFORMANCE_PATH,
    index=False
)


print(
    f"\nPerformance saved to:\n"
    f"{PERFORMANCE_PATH}"
)


# SAVE PREDICTIONS

prediction_df = pd.DataFrame()


prediction_df["timestamp"] = test_df["timestamp"]


prediction_df["actual_pm2_5"] = test_df["pm2_5"]


for i in range(72):

    prediction_df[
        f"pm2_5_t+{i + 1}"
    ] = y_pred_np[:, i]


prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)


print(
    f"\nTest predictions saved to:\n"
    f"{PREDICTIONS_PATH}"
)


# FINAL SUMMARY

print("\n" + "=" * 70)
print("FINAL MODEL SUMMARY")
print("=" * 70)

print(f"\nMAE : {overall_mae:.2f}")
print(f"RMSE: {overall_rmse:.2f}")
print(f"R²  : {overall_r2:.4f}")

print("\nForecast Horizon: 72 Hours")

print("\nEvaluation completed successfully!")

print("=" * 70)