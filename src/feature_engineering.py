import pandas as pd
import numpy as np

# ============================================================
# LAHORE AQI FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("LAHORE AQI FEATURE ENGINEERING")
print("=" * 60)

# LOAD DATA
input_file = "data/processed/lahore_combined_raw.csv"
df = pd.read_csv(input_file)

print("\nDataset loaded successfully!")
print("Original shape:", df.shape)

# TIMESTAMP
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# TIME FEATURES
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month
df["year"] = df["timestamp"].dt.year
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# SEASON
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

df["season"] = df["month"].apply(get_season)

# ============================================================
# NEW: ATMOSPHERIC PERSISTENCE (LAGS & ROLLING FEATURES)
# ============================================================
print("\nEngineering temporal lag and rolling features...")

# Automatically detect PM2.5 column name
pm_col = 'pm2_5' if 'pm2_5' in df.columns else 'pm25' if 'pm25' in df.columns else None

if pm_col:
    # Historical Lags
    df['pm25_lag_1h'] = df[pm_col].shift(1)
    df['pm25_lag_3h'] = df[pm_col].shift(3)
    df['pm25_lag_24h'] = df[pm_col].shift(24)
    
    # 6-Hour Rolling Average (captures pollution buildup)
    df['pm25_rolling_mean_6h'] = df[pm_col].shift(1).rolling(window=6, min_periods=1).mean()

# Weather Interactions (if meteorological data is present)
if 'temperature_2m' in df.columns and 'relative_humidity_2m' in df.columns:
    df['temp_humidity_index'] = df['temperature_2m'] * (df['relative_humidity_2m'] / 100.0)

# Drop NaN values created by the 24-hour shift
df = df.dropna().reset_index(drop=True)

# ============================================================
# CHECK DATA
# ============================================================
print("\nFeature engineering completed!")
print("\nNew time & lag features:", ["hour", "day", "day_of_week", "month", "year", "is_weekend", "season", "pm25_lag_1h", "pm25_lag_24h"])

print("\nFinal shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDate range:\n", df["timestamp"].min(), "to", df["timestamp"].max())

# SAVE
output_file = "data/processed/lahore_features.csv"
df.to_csv(output_file, index=False)

print("\nFeature dataset saved to:", output_file)
print("\nFirst 5 rows:\n", df.head())

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)