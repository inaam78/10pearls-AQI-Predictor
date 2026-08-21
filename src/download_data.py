import os
import requests
import pandas as pd


# CONFIGURATION

LATITUDE = 31.5497
LONGITUDE = 74.3436

START_DATE = "2021-01-01"
END_DATE = "2025-12-31"

OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "lahore_hourly_raw.csv"
)


# CREATE OUTPUT DIRECTORY

os.makedirs(OUTPUT_DIR, exist_ok=True)


# OPEN-METEO API

URL = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": ",".join([
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m"
    ]),
    "timezone": "Asia/Karachi"
}


# DOWNLOAD DATA

print("=" * 60)
print("LAHORE HISTORICAL WEATHER DATA DOWNLOAD")
print("=" * 60)

print("\nDownloading data from Open-Meteo...")

response = requests.get(
    URL,
    params=params,
    timeout=120
)

response.raise_for_status()

data = response.json()

print("Data downloaded successfully!")


# CONVERT TO DATAFRAME

hourly_data = data["hourly"]

df = pd.DataFrame(hourly_data)

df["timestamp"] = pd.to_datetime(
    df["time"]
)

df = df.drop(
    columns=["time"]
)


# ADD LOCATION INFORMATION

df["city"] = "Lahore"

df["latitude"] = LATITUDE

df["longitude"] = LONGITUDE


# REORDER COLUMNS

df = df[
    [
        "timestamp",
        "city",
        "latitude",
        "longitude",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m"
    ]
]


# SAVE RAW DATA

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# INFORMATION

print("\nDataset Information")
print("-" * 60)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nDate Range:")

print(
    df["timestamp"].min(),
    "to",
    df["timestamp"].max()
)

print("\nColumns:")

print(
    df.columns.tolist()
)

print("\nFirst 5 rows:")

print(
    df.head()
)

print("\nMissing Values:")

print(
    df.isnull().sum()
)

print("\n" + "=" * 60)

print("RAW DATA SAVED SUCCESSFULLY")

print("=" * 60)

print(
    f"\nFile: {OUTPUT_FILE}"
)