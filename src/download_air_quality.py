import os
import requests
import pandas as pd


# LAHORE HISTORICAL AIR QUALITY DATA DOWNLOAD

print("\n" + "=" * 60)
print("LAHORE HISTORICAL AIR QUALITY DATA DOWNLOAD")
print("=" * 60)


# LOCATION

LATITUDE = 31.5497
LONGITUDE = 74.3436

START_DATE = "2021-01-01"
END_DATE = "2025-12-31"

CITY = "Lahore"


# API

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,

    "hourly": ",".join([
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "dust"
    ]),

    "timezone": "Asia/Karachi"
}


# DOWNLOAD DATA

print("\nDownloading air-quality data from Open-Meteo...")

response = requests.get(
    URL,
    params=params,
    timeout=120
)

response.raise_for_status()

data = response.json()

print("Data downloaded successfully!")


# CREATE DATAFRAME

hourly = data["hourly"]

df = pd.DataFrame(hourly)


# RENAME TIMESTAMP

df.rename(
    columns={"time": "timestamp"},
    inplace=True
)


# ADD LOCATION INFORMATION

df.insert(
    1,
    "city",
    CITY
)

df.insert(
    2,
    "latitude",
    LATITUDE
)

df.insert(
    3,
    "longitude",
    LONGITUDE
)


# CONVERT TIMESTAMP

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)


# SORT DATA

df = df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


# CHECK DUPLICATES

duplicates = df["timestamp"].duplicated().sum()

print("\nDuplicate timestamps:", duplicates)


# CHECK MISSING VALUES

print("\nMissing Values:")

print(
    df.isnull().sum()
)


# SAVE DATA

output_dir = "data/raw"

os.makedirs(
    output_dir,
    exist_ok=True
)

output_file = os.path.join(
    output_dir,
    "lahore_air_quality_raw.csv"
)

df.to_csv(
    output_file,
    index=False
)


# DATASET INFORMATION

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nRows:", len(df))

print(
    "Columns:",
    len(df.columns)
)

print(
    "\nDate Range:"
)

print(
    df["timestamp"].min(),
    "to",
    df["timestamp"].max()
)

print(
    "\nColumns:"
)

print(
    df.columns.tolist()
)


print(
    "\nFirst 5 rows:"
)

print(
    df.head()
)


print(
    "\nFile:",
    output_file
)

print(
    "\nAir-quality download completed successfully!"
)

print("=" * 60)