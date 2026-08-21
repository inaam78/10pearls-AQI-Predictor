"""
LAHORE AQI - 72 HOUR PREDICTION ANALYSIS

Analyzes the final Random Forest model predictions.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# CONFIGURATION

TEST_FILE = "data/splits/test.csv"

PREDICTIONS_FILE = "results/models/test_predictions.csv"

OUTPUT_DIR = "results/models/analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("LAHORE AQI - 72 HOUR PREDICTION ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_data = pd.read_csv(TEST_FILE)

print("Test dataset loaded successfully!")

print("Shape:", test_data.shape)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("\nLoading model predictions...")

predictions = pd.read_csv(PREDICTIONS_FILE)

print("Prediction dataset loaded successfully!")

print("Shape:", predictions.shape)


# DISPLAY COLUMNS

print("\nPrediction columns:")

print(predictions.columns.tolist())


# CHECK ROW COUNTS

if len(test_data) != len(predictions):

    raise ValueError(
        f"Row count mismatch!\n"
        f"Test dataset: {len(test_data)}\n"
        f"Predictions: {len(predictions)}"
    )


print("\nRow count verified:")
print(len(test_data), "rows")


# FIND TARGET COLUMNS

target_columns = [
    f"pm2_5_t+{i}"
    for i in range(1, 73)
]


print("\nChecking actual target columns...")

missing_targets = [
    column
    for column in target_columns
    if column not in test_data.columns
]


if missing_targets:

    print("\nMissing target columns:")

    for column in missing_targets:
        print(column)

    raise ValueError(
        "Some target columns are missing from test.csv."
    )


print("72 actual target columns found.")


# FIND PREDICTION COLUMNS

prediction_columns = [
    f"pm2_5_t+{i}"
    for i in range(1, 73)
]


missing_predictions = [
    column
    for column in prediction_columns
    if column not in predictions.columns
]


if missing_predictions:

    print("\nMissing prediction columns:")

    for column in missing_predictions:
        print(column)

    raise ValueError(
        "Some prediction columns are missing from "
        "test_predictions.csv."
    )


print("72 prediction columns found.")


# EXTRACT ACTUAL VALUES

actual_values = test_data[
    target_columns
].apply(
    pd.to_numeric,
    errors="coerce"
)


# EXTRACT PREDICTED VALUES

predicted_values = predictions[
    prediction_columns
].apply(
    pd.to_numeric,
    errors="coerce"
)


# CHECK MISSING VALUES

print("\nChecking actual values...")

actual_missing = actual_values.isna().sum().sum()

print(
    "Missing actual target values:",
    actual_missing
)


print("\nChecking predictions...")

prediction_missing = predicted_values.isna().sum().sum()

print(
    "Missing prediction values:",
    prediction_missing
)


if actual_missing > 0:

    raise ValueError(
        "Actual target values contain missing data."
    )


if prediction_missing > 0:

    raise ValueError(
        "Model predictions contain missing data."
    )


# CALCULATE PERFORMANCE FOR EACH HOUR

print("\nCalculating performance for all 72 hours...")

performance_results = []


for hour in range(1, 73):

    actual = actual_values[
        f"pm2_5_t+{hour}"
    ].values

    predicted = predicted_values[
        f"pm2_5_t+{hour}"
    ].values


    # MAE

    mae = np.mean(
        np.abs(
            actual - predicted
        )
    )


    # RMSE

    rmse = np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )


    # Mean actual

    mean_actual = np.mean(actual)


    # Mean prediction

    mean_predicted = np.mean(predicted)


    performance_results.append({

        "forecast_hour": hour,

        "mae": mae,

        "rmse": rmse,

        "mean_actual": mean_actual,

        "mean_predicted": mean_predicted

    })


performance = pd.DataFrame(
    performance_results
)


# SAVE PERFORMANCE

performance_file = os.path.join(
    OUTPUT_DIR,
    "72_hour_performance.csv"
)


performance.to_csv(
    performance_file,
    index=False
)


print("\n72-hour performance saved to:")

print(performance_file)


# DISPLAY KEY PERFORMANCE

print("\n" + "=" * 70)
print("FORECAST-SPECIFIC PERFORMANCE")
print("=" * 70)


for hour in [1, 6, 12, 24, 48, 72]:

    row = performance[
        performance["forecast_hour"] == hour
    ].iloc[0]


    print(
        f"{hour:2d}-hour -> "
        f"MAE: {row['mae']:.2f} | "
        f"RMSE: {row['rmse']:.2f}"
    )


# OVERALL PERFORMANCE

all_actual = actual_values.values.flatten()

all_predicted = predicted_values.values.flatten()


overall_mae = np.mean(
    np.abs(
        all_actual - all_predicted
    )
)


overall_rmse = np.sqrt(
    np.mean(
        (all_actual - all_predicted) ** 2
    )
)


print("\n" + "=" * 70)
print("OVERALL TEST PERFORMANCE")
print("=" * 70)

print(
    f"Overall MAE : {overall_mae:.2f}"
)

print(
    f"Overall RMSE: {overall_rmse:.2f}"
)


# 1. MAE BY FORECAST HORIZON

print("\nCreating MAE plot...")


plt.figure(figsize=(12, 6))


plt.plot(
    performance["forecast_hour"],
    performance["mae"],
    marker="o",
    markersize=3
)


plt.xlabel(
    "Forecast Horizon (Hours)"
)

plt.ylabel(
    "MAE"
)

plt.title(
    "PM2.5 Prediction MAE Across 72-Hour Forecast"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "mae_by_forecast_hour.png"
    ),
    dpi=300
)


plt.close()


# 2. RMSE BY FORECAST HORIZON

print("Creating RMSE plot...")


plt.figure(figsize=(12, 6))


plt.plot(
    performance["forecast_hour"],
    performance["rmse"],
    marker="o",
    markersize=3
)


plt.xlabel(
    "Forecast Horizon (Hours)"
)

plt.ylabel(
    "RMSE"
)

plt.title(
    "PM2.5 Prediction RMSE Across 72-Hour Forecast"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "rmse_by_forecast_hour.png"
    ),
    dpi=300
)


plt.close()


# 3. ACTUAL VS PREDICTED - 1 HOUR

print("Creating 1-hour actual vs predicted plot...")


actual_1h = actual_values[
    "pm2_5_t+1"
].values


predicted_1h = predicted_values[
    "pm2_5_t+1"
].values


sample_size = min(
    300,
    len(actual_1h)
)


plt.figure(figsize=(14, 6))


plt.plot(
    actual_1h[:sample_size],
    label="Actual PM2.5"
)


plt.plot(
    predicted_1h[:sample_size],
    label="Predicted PM2.5"
)


plt.xlabel(
    "Test Sample"
)

plt.ylabel(
    "PM2.5"
)


plt.title(
    "Actual vs Predicted PM2.5 - 1 Hour Ahead"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "actual_vs_predicted_1h.png"
    ),
    dpi=300
)


plt.close()


# 4. ACTUAL VS PREDICTED - 24 HOURS

print("Creating 24-hour actual vs predicted plot...")


actual_24h = actual_values[
    "pm2_5_t+24"
].values


predicted_24h = predicted_values[
    "pm2_5_t+24"
].values


sample_size = min(
    300,
    len(actual_24h)
)


plt.figure(figsize=(14, 6))


plt.plot(
    actual_24h[:sample_size],
    label="Actual PM2.5"
)


plt.plot(
    predicted_24h[:sample_size],
    label="Predicted PM2.5"
)


plt.xlabel(
    "Test Sample"
)

plt.ylabel(
    "PM2.5"
)


plt.title(
    "Actual vs Predicted PM2.5 - 24 Hours Ahead"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "actual_vs_predicted_24h.png"
    ),
    dpi=300
)


plt.close()


# 5. ACTUAL VS PREDICTED - 72 HOURS

print("Creating 72-hour actual vs predicted plot...")


actual_72h = actual_values[
    "pm2_5_t+72"
].values


predicted_72h = predicted_values[
    "pm2_5_t+72"
].values


sample_size = min(
    300,
    len(actual_72h)
)


plt.figure(figsize=(14, 6))


plt.plot(
    actual_72h[:sample_size],
    label="Actual PM2.5"
)


plt.plot(
    predicted_72h[:sample_size],
    label="Predicted PM2.5"
)


plt.xlabel(
    "Test Sample"
)

plt.ylabel(
    "PM2.5"
)


plt.title(
    "Actual vs Predicted PM2.5 - 72 Hours Ahead"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "actual_vs_predicted_72h.png"
    ),
    dpi=300
)


plt.close()


# 6. PREDICTION ERROR DISTRIBUTION

print(
    "Creating prediction error distribution..."
)


errors = (
    all_predicted - all_actual
)


plt.figure(figsize=(10, 6))


plt.hist(
    errors,
    bins=50
)


plt.axvline(
    0,
    linestyle="--"
)


plt.xlabel(
    "Prediction Error"
)

plt.ylabel(
    "Frequency"
)


plt.title(
    "Distribution of PM2.5 Prediction Errors"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "prediction_error_distribution.png"
    ),
    dpi=300
)


plt.close()


# 7. ACTUAL VS PREDICTED SCATTER

print(
    "Creating actual vs predicted scatter plot..."
)


sample_size = min(
    10000,
    len(all_actual)
)


rng = np.random.default_rng(42)


indices = rng.choice(
    len(all_actual),
    size=sample_size,
    replace=False
)


actual_sample = all_actual[
    indices
]


predicted_sample = all_predicted[
    indices
]


plt.figure(figsize=(8, 8))


plt.scatter(
    actual_sample,
    predicted_sample,
    alpha=0.3
)


minimum = min(
    actual_sample.min(),
    predicted_sample.min()
)


maximum = max(
    actual_sample.max(),
    predicted_sample.max()
)


plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)


plt.xlabel(
    "Actual PM2.5"
)

plt.ylabel(
    "Predicted PM2.5"
)


plt.title(
    "Actual vs Predicted PM2.5"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "actual_vs_predicted_scatter.png"
    ),
    dpi=300
)


plt.close()


# 8. SAMPLE 72-HOUR FORECAST

print(
    "Creating sample 72-hour forecast..."
)


sample_index = 0


sample_actual = actual_values.iloc[
    sample_index
].values


sample_predicted = predicted_values.iloc[
    sample_index
].values


hours = np.arange(
    1,
    73
)


plt.figure(figsize=(14, 6))


plt.plot(
    hours,
    sample_actual,
    marker="o",
    markersize=3,
    label="Actual PM2.5"
)


plt.plot(
    hours,
    sample_predicted,
    marker="o",
    markersize=3,
    label="Predicted PM2.5"
)


plt.xlabel(
    "Forecast Horizon (Hours)"
)

plt.ylabel(
    "PM2.5"
)


plt.title(
    "Example 72-Hour PM2.5 Forecast"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.xticks(
    np.arange(
        0,
        73,
        6
    )
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "sample_72_hour_forecast.png"
    ),
    dpi=300
)


plt.close()


# 9. SUMMARY

summary = pd.DataFrame({

    "metric": [

        "Overall MAE",

        "Overall RMSE",

        "1-hour MAE",

        "6-hour MAE",

        "12-hour MAE",

        "24-hour MAE",

        "48-hour MAE",

        "72-hour MAE",

        "1-hour RMSE",

        "6-hour RMSE",

        "12-hour RMSE",

        "24-hour RMSE",

        "48-hour RMSE",

        "72-hour RMSE"

    ],

    "value": [

        overall_mae,

        overall_rmse,

        performance.loc[
            performance["forecast_hour"] == 1,
            "mae"
        ].iloc[0],

        performance.loc[
            performance["forecast_hour"] == 6,
            "mae"
        ].iloc[0],

        performance.loc[
            performance["forecast_hour"] == 12,
            "mae"
        ].iloc[0],

        performance.loc[
            performance["forecast_hour"] == 24,
            "mae"
        ].iloc[0],

        performance.loc[
            performance["forecast_hour"] == 48,
            "mae"
        ].iloc[0],

        performance.loc[
            performance["forecast_hour"] == 72,
            "mae"
        ].iloc[0],

        performance.loc[
            performance["forecast_hour"] == 1,
            "rmse"
        ].iloc[0],

        performance.loc[
            performance["forecast_hour"] == 6,
            "rmse"
        ].iloc[0],

        performance.loc[
            performance["forecast_hour"] == 12,
            "rmse"
        ].iloc[0],

        performance.loc[
            performance["forecast_hour"] == 24,
            "rmse"
        ].iloc[0],

        performance.loc[
            performance["forecast_hour"] == 48,
            "rmse"
        ].iloc[0],

        performance.loc[
            performance["forecast_hour"] == 72,
            "rmse"
        ].iloc[0]

    ]

})


summary_file = os.path.join(
    OUTPUT_DIR,
    "prediction_analysis_summary.csv"
)


summary.to_csv(
    summary_file,
    index=False
)


# FINAL OUTPUT

print("\n" + "=" * 70)
print("PREDICTION ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)


print(
    f"\nOverall MAE : {overall_mae:.2f}"
)


print(
    f"Overall RMSE: {overall_rmse:.2f}"
)


print("\nAnalysis results saved in:")

print(
    OUTPUT_DIR
)


print("\nGenerated files:")


for file in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(
        "-",
        file
    )


print("\n" + "=" * 70)