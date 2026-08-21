import os
import pandas as pd


print("\n" + "=" * 60)
print("MERGING LAHORE WEATHER + AIR QUALITY DATA")
print("=" * 60)


# FILE PATHS

WEATHER_FILE = "data/raw/lahore_hourly_raw.csv"

AIR_QUALITY_FILE = "data/raw/lahore_air_quality_complete.csv"

OUTPUT_FILE = "data/processed/lahore_combined_raw.csv"


# LOAD DATA

print("\nLoading weather dataset...")

weather = pd.read_csv(
    WEATHER_FILE
)

print(
    "Weather shape:",
    weather.shape
)


print("\nLoading air-quality dataset...")

air_quality = pd.read_csv(
    AIR_QUALITY_FILE
)

print(
    "Air-quality shape:",
    air_quality.shape
)


# CONVERT TIMESTAMP

weather["timestamp"] = pd.to_datetime(
    weather["timestamp"]
)

air_quality["timestamp"] = pd.to_datetime(
    air_quality["timestamp"]
)


# CHECK DUPLICATES

print("\nWeather duplicate timestamps:")

print(
    weather["timestamp"].duplicated().sum()
)


print("\nAir-quality duplicate timestamps:")

print(
    air_quality["timestamp"].duplicated().sum()
)


# MERGE

print("\nMerging datasets...")

df = pd.merge(
    weather,
    air_quality,
    on="timestamp",
    how="inner",
    suffixes=("_weather", "_air")
)


# REMOVE DUPLICATE LOCATION COLUMNS

if "city_weather" in df.columns:

    df["city"] = df["city_weather"]

    df.drop(
        columns=[
            "city_weather",
            "city_air"
        ],
        inplace=True
    )


if "latitude_weather" in df.columns:

    df["latitude"] = df["latitude_weather"]

    df.drop(
        columns=[
            "latitude_weather",
            "latitude_air"
        ],
        inplace=True
    )


if "longitude_weather" in df.columns:

    df["longitude"] = df["longitude_weather"]

    df.drop(
        columns=[
            "longitude_weather",
            "longitude_air"
        ],
        inplace=True
    )


# SORT

df = df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


# CHECK DATA

print("\n" + "=" * 60)
print("MERGED DATASET INFORMATION")
print("=" * 60)

print(
    "\nShape:",
    df.shape
)

print(
    "\nDate range:"
)

print(
    df["timestamp"].min(),
    "to",
    df["timestamp"].max()
)


print(
    "\nMissing values:"
)

print(
    df.isnull().sum()
)


print(
    "\nDuplicate timestamps:",
    df["timestamp"].duplicated().sum()
)


print(
    "\nColumns:"
)

print(
    df.columns.tolist()
)


# SAVE

os.makedirs(
    "data/processed",
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nCombined dataset saved to:"
)

print(
    OUTPUT_FILE
)


print("\nFirst 5 rows:")

print(
    df.head()
)


print("\n" + "=" * 60)
print("MERGE COMPLETED SUCCESSFULLY")
print("=" * 60)