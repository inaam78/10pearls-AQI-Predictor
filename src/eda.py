import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# LAHORE AQI PREDICTION - EXPLORATORY DATA ANALYSIS

print("\n" + "=" * 70)
print("LAHORE AQI PREDICTION - EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# PATHS

INPUT_FILE = "data/processed/lahore_features.csv"
EDA_DIR = "results/eda"

os.makedirs(EDA_DIR, exist_ok=True)


# LOAD DATA

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# BASIC DATA INFORMATION


print("\n" + "=" * 70)
print("BASIC DATA INFORMATION")
print("=" * 70)

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())


# TIMESTAMP CONVERSION

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


print("\nDate Range:")
print(df["timestamp"].min())
print("to")
print(df["timestamp"].max())


# DUPLICATE CHECK

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

duplicate_rows = df.duplicated().sum()

duplicate_timestamps = df["timestamp"].duplicated().sum()

print("\nDuplicate rows:", duplicate_rows)
print("Duplicate timestamps:", duplicate_timestamps)


# MISSING VALUE ANALYSIS

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missing_values = df.isnull().sum()

missing_percentage = (
    df.isnull().mean() * 100
).round(2)

missing_report = pd.DataFrame({
    "missing_count": missing_values,
    "missing_percentage": missing_percentage
})

print("\nMissing Values:")
print(missing_report)

missing_report.to_csv(
    f"{EDA_DIR}/missing_values.csv"
)


# NUMERICAL SUMMARY

print("\n" + "=" * 70)
print("NUMERICAL SUMMARY")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include=np.number
).columns

summary = df[numeric_columns].describe().T

print(summary)

summary.to_csv(
    f"{EDA_DIR}/numerical_summary.csv"
)


# AQI RELATED VARIABLES

aqi_variables = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust"
]

available_aqi_variables = [
    column
    for column in aqi_variables
    if column in df.columns
]

print("\n" + "=" * 70)
print("AIR POLLUTANT SUMMARY")
print("=" * 70)

print(
    df[available_aqi_variables].describe().T
)


# WEATHER VARIABLES

weather_variables = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m"
]

available_weather_variables = [
    column
    for column in weather_variables
    if column in df.columns
]

print("\n" + "=" * 70)
print("WEATHER VARIABLE SUMMARY")
print("=" * 70)

print(
    df[available_weather_variables].describe().T
)


# HOURLY POLLUTION PATTERN

print("\n" + "=" * 70)
print("HOURLY POLLUTION PATTERN")
print("=" * 70)

hourly_pm25 = (
    df.groupby("hour")["pm2_5"]
    .mean()
)

print("\nAverage PM2.5 by Hour:")
print(hourly_pm25)


plt.figure(figsize=(12, 6))

plt.plot(
    hourly_pm25.index,
    hourly_pm25.values,
    marker="o"
)

plt.title(
    "Average PM2.5 Concentration by Hour"
)

plt.xlabel("Hour of Day")

plt.ylabel("Average PM2.5")

plt.xticks(range(24))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{EDA_DIR}/hourly_pm25.png",
    dpi=300
)

plt.close()


# DAILY POLLUTION PATTERN

daily_pm25 = (
    df.groupby("day_of_week")["pm2_5"]
    .mean()
)

print("\nAverage PM2.5 by Day of Week:")
print(daily_pm25)


plt.figure(figsize=(10, 6))

plt.plot(
    daily_pm25.index,
    daily_pm25.values,
    marker="o"
)

plt.title(
    "Average PM2.5 by Day of Week"
)

plt.xlabel("Day of Week")

plt.ylabel("Average PM2.5")

plt.xticks(
    range(7),
    [
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri",
        "Sat",
        "Sun"
    ]
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{EDA_DIR}/daily_pm25.png",
    dpi=300
)

plt.close()


# MONTHLY POLLUTION PATTERN

monthly_pm25 = (
    df.groupby("month")["pm2_5"]
    .mean()
)

print("\nAverage PM2.5 by Month:")
print(monthly_pm25)


plt.figure(figsize=(12, 6))

plt.plot(
    monthly_pm25.index,
    monthly_pm25.values,
    marker="o"
)

plt.title(
    "Average PM2.5 by Month"
)

plt.xlabel("Month")

plt.ylabel("Average PM2.5")

plt.xticks(range(1, 13))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{EDA_DIR}/monthly_pm25.png",
    dpi=300
)

plt.close()


# YEARLY POLLUTION PATTERN

yearly_pm25 = (
    df.groupby("year")["pm2_5"]
    .mean()
)

print("\nAverage PM2.5 by Year:")
print(yearly_pm25)


plt.figure(figsize=(10, 6))

plt.plot(
    yearly_pm25.index,
    yearly_pm25.values,
    marker="o"
)

plt.title(
    "Average PM2.5 by Year"
)

plt.xlabel("Year")

plt.ylabel("Average PM2.5")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{EDA_DIR}/yearly_pm25.png",
    dpi=300
)

plt.close()


# PM2.5 TIME SERIES

print("\nCreating PM2.5 time-series plot...")

plt.figure(figsize=(15, 6))

plt.plot(
    df["timestamp"],
    df["pm2_5"]
)

plt.title(
    "PM2.5 Concentration Over Time"
)

plt.xlabel("Timestamp")

plt.ylabel("PM2.5")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{EDA_DIR}/pm25_time_series.png",
    dpi=300
)

plt.close()


# PM10 TIME SERIES

print("Creating PM10 time-series plot...")

plt.figure(figsize=(15, 6))

plt.plot(
    df["timestamp"],
    df["pm10"]
)

plt.title(
    "PM10 Concentration Over Time"
)

plt.xlabel("Timestamp")

plt.ylabel("PM10")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{EDA_DIR}/pm10_time_series.png",
    dpi=300
)

plt.close()


# POLLUTANT DISTRIBUTIONS

print("\nCreating pollutant distribution plots...")

for column in available_aqi_variables:

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df[column],
        kde=True
    )

    plt.title(
        f"Distribution of {column}"
    )

    plt.xlabel(column)

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        f"{EDA_DIR}/distribution_{column}.png",
        dpi=300
    )

    plt.close()


# WEATHER VS PM2.5

print("\nCreating weather-pollution relationship plots...")

relationship_variables = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "surface_pressure",
    "precipitation"
]

for column in relationship_variables:

    if column not in df.columns:
        continue

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df[column],
        df["pm2_5"],
        alpha=0.3
    )

    plt.title(
        f"{column} vs PM2.5"
    )

    plt.xlabel(column)

    plt.ylabel("PM2.5")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"{EDA_DIR}/{column}_vs_pm25.png",
        dpi=300
    )

    plt.close()


# CORRELATION ANALYSIS

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

correlation_columns = [
    column
    for column in (
        available_weather_variables
        + available_aqi_variables
        + [
            "hour",
            "day",
            "day_of_week",
            "month",
            "year",
            "is_weekend"
        ]
    )
    if column in df.columns
]

correlation_matrix = df[
    correlation_columns
].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)


correlation_matrix.to_csv(
    f"{EDA_DIR}/correlation_matrix.csv"
)


plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title(
    "Correlation Matrix - Lahore AQI Dataset"
)

plt.tight_layout()

plt.savefig(
    f"{EDA_DIR}/correlation_heatmap.png",
    dpi=300
)

plt.close()


# WEEKEND VS WEEKDAY

print("\n" + "=" * 70)
print("WEEKEND VS WEEKDAY ANALYSIS")
print("=" * 70)

weekend_analysis = (
    df.groupby("is_weekend")["pm2_5"]
    .mean()
)

print(
    "\nAverage PM2.5:"
)

print(
    "Weekday:",
    weekend_analysis.get(0)
)

print(
    "Weekend:",
    weekend_analysis.get(1)
)


# SEASON ANALYSIS

if "season" in df.columns:

    print("\n" + "=" * 70)
    print("SEASONAL ANALYSIS")
    print("=" * 70)

    seasonal_pm25 = (
        df.groupby("season")["pm2_5"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print(
        "\nAverage PM2.5 by Season:"
    )

    print(
        seasonal_pm25
    )

    plt.figure(figsize=(10, 6))

    seasonal_pm25.plot(
        kind="bar"
    )

    plt.title(
        "Average PM2.5 by Season"
    )

    plt.xlabel("Season")

    plt.ylabel("Average PM2.5")

    plt.tight_layout()

    plt.savefig(
        f"{EDA_DIR}/seasonal_pm25.png",
        dpi=300
    )

    plt.close()


# EXTREME POLLUTION VALUES

print("\n" + "=" * 70)
print("EXTREME POLLUTION VALUES")
print("=" * 70)

print("\nHighest PM2.5 values:")

highest_pm25 = (
    df[
        [
            "timestamp",
            "pm2_5"
        ]
    ]
    .sort_values(
        "pm2_5",
        ascending=False
    )
    .head(10)
)

print(
    highest_pm25
)


print("\nHighest PM10 values:")

highest_pm10 = (
    df[
        [
            "timestamp",
            "pm10"
        ]
    ]
    .sort_values(
        "pm10",
        ascending=False
    )
    .head(10)
)

print(
    highest_pm10
)


# OUTLIER CHECK

print("\n" + "=" * 70)
print("OUTLIER CHECK")
print("=" * 70)

for column in available_aqi_variables:

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR

    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound)
        |
        (df[column] > upper_bound)
    ]

    print(
        f"{column}: "
        f"{len(outliers)} potential outliers"
    )


# AQI FEATURE RELATIONSHIPS

if "pm2_5" in df.columns:

    print("\n" + "=" * 70)
    print("PM2.5 CORRELATION WITH OTHER VARIABLES")
    print("=" * 70)

    pm25_correlation = (
        correlation_matrix["pm2_5"]
        .sort_values(
            ascending=False
        )
    )

    print(
        pm25_correlation
    )

    pm25_correlation.to_csv(
        f"{EDA_DIR}/pm25_correlations.csv"
    )


# DATA QUALITY REPORT

print("\n" + "=" * 70)
print("DATA QUALITY REPORT")
print("=" * 70)

print(
    "\nTotal rows:",
    len(df)
)

print(
    "Total columns:",
    len(df.columns)
)

print(
    "Duplicate rows:",
    duplicate_rows
)

print(
    "Duplicate timestamps:",
    duplicate_timestamps
)

print(
    "Total missing values:",
    df.isnull().sum().sum()
)

print(
    "Start date:",
    df["timestamp"].min()
)

print(
    "End date:",
    df["timestamp"].max()
)


# SAVE CLEAN EDA DATASET

eda_summary = {
    "total_rows": len(df),
    "total_columns": len(df.columns),
    "duplicate_rows": duplicate_rows,
    "duplicate_timestamps": duplicate_timestamps,
    "missing_values": int(
        df.isnull().sum().sum()
    ),
    "start_date": str(
        df["timestamp"].min()
    ),
    "end_date": str(
        df["timestamp"].max()
    )
}

eda_summary_df = pd.DataFrame(
    [eda_summary]
)

eda_summary_df.to_csv(
    f"{EDA_DIR}/eda_summary.csv",
    index=False
)


# COMPLETED

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nEDA results saved in: {EDA_DIR}"
)

print("\nGenerated files include:")

print("- missing_values.csv")
print("- numerical_summary.csv")
print("- correlation_matrix.csv")
print("- correlation_heatmap.png")
print("- hourly_pm25.png")
print("- daily_pm25.png")
print("- monthly_pm25.png")
print("- yearly_pm25.png")
print("- pm25_time_series.png")
print("- pm10_time_series.png")
print("- pollutant distribution plots")
print("- weather vs PM2.5 plots")
print("- seasonal_pm25.png")
print("- pm25_correlations.csv")
print("- eda_summary.csv")

print("\n" + "=" * 70)