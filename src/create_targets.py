import pandas as pd
import os

# LAHORE AQI PROJECT - CREATE 72-HOUR FUTURE TARGETS

INPUT_FILE = "data/processed/lahore_features.csv"
OUTPUT_FILE = "data/processed/lahore_model_data.csv"

print("=" * 60)
print("CREATING 72-HOUR FUTURE PM2.5 TARGETS")
print("=" * 60)

# 1. Load dataset

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully!")
print(f"Original shape: {df.shape}")

# 2. Convert timestamp

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort chronologically
df = df.sort_values("timestamp").reset_index(drop=True)

# 3. Check timestamp frequency

time_difference = df["timestamp"].diff().dropna()

print("\nTimestamp difference:")
print(time_difference.value_counts().head())

# 4. Create future PM2.5 targets

print("\nCreating future PM2.5 targets...")

for hour in range(1, 73):
    df[f"pm2_5_t+{hour}"] = df["pm2_5"].shift(-hour)

# 5. Count target columns

target_columns = [
    f"pm2_5_t+{hour}"
    for hour in range(1, 73)
]

print(f"\nNumber of target columns: {len(target_columns)}")

print("\nTarget columns:")
print(target_columns)

# 6. Remove rows without complete 72-hour future targets

before = len(df)

df = df.dropna(subset=target_columns).reset_index(drop=True)

after = len(df)

print("\nRows removed because 72-hour future data was unavailable:")
print(before - after)

print(f"Rows remaining: {after}")

# 7. Check missing values

print("\nMissing values in model dataset:")

missing = df.isnull().sum()

print(missing[missing > 0])

# 8. Check duplicate timestamps
# ------------------------------------------------------------

duplicates = df["timestamp"].duplicated().sum()

print(f"\nDuplicate timestamps: {duplicates}")


# 9. Save dataset

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("TARGET CREATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nFinal shape: {df.shape}")

print("\nDate range:")
print(df["timestamp"].min())
print("to")
print(df["timestamp"].max())

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("\nFirst 5 target values:")

print(
    df[
        [
            "timestamp",
            "pm2_5",
            "pm2_5_t+1",
            "pm2_5_t+2",
            "pm2_5_t+3",
            "pm2_5_t+24",
            "pm2_5_t+48",
            "pm2_5_t+72",
        ]
    ].head()
)

print("\nLast 5 target values:")

print(
    df[
        [
            "timestamp",
            "pm2_5",
            "pm2_5_t+1",
            "pm2_5_t+24",
            "pm2_5_t+48",
            "pm2_5_t+72",
        ]
    ].tail()
)

print("\n" + "=" * 60)