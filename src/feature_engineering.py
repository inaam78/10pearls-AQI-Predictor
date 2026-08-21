import pandas as pd


# LAHORE AQI FEATURE ENGINEERING

print("\n" + "=" * 60)
print("LAHORE AQI FEATURE ENGINEERING")
print("=" * 60)


# LOAD DATA

input_file = "data/processed/lahore_combined_raw.csv"

df = pd.read_csv(input_file)

print("\nDataset loaded successfully!")

print(
    "Original shape:",
    df.shape
)


# TIMESTAMP

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# TIME FEATURES

df["hour"] = df["timestamp"].dt.hour

df["day"] = df["timestamp"].dt.day

df["day_of_week"] = df["timestamp"].dt.dayofweek

df["month"] = df["timestamp"].dt.month

df["year"] = df["timestamp"].dt.year

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)


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


df["season"] = df["month"].apply(
    get_season
)


# CHECK DATA

print("\nFeature engineering completed!")

print(
    "\nNew time features:"
)

print([
    "hour",
    "day",
    "day_of_week",
    "month",
    "year",
    "is_weekend",
    "season"
])


# DATA INFORMATION

print(
    "\nFinal shape:",
    df.shape
)

print(
    "\nMissing values:"
)

print(
    df.isnull().sum()
)


print(
    "\nDate range:"
)

print(
    df["timestamp"].min(),
    "to",
    df["timestamp"].max()
)


# SAVE

output_file = (
    "data/processed/"
    "lahore_features.csv"
)

df.to_csv(
    output_file,
    index=False
)


print(
    "\nFeature dataset saved to:"
)

print(
    output_file
)


print("\nFirst 5 rows:")

print(
    df.head()
)


print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)