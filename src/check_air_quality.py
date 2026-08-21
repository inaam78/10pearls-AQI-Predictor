import pandas as pd


FILE = "data/raw/lahore_air_quality_raw.csv"

print("\n" + "=" * 60)
print("AIR QUALITY DATA QUALITY CHECK")
print("=" * 60)


# Load dataset
df = pd.read_csv(FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])


# BASIC INFORMATION

print("\nTotal rows:", len(df))

print(
    "Date range:",
    df["timestamp"].min(),
    "to",
    df["timestamp"].max()
)


# MISSING DATA BY YEAR

df["year"] = df["timestamp"].dt.year

pollutants = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust"
]


print("\nMissing values by year:")

for year in sorted(df["year"].unique()):

    yearly = df[df["year"] == year]

    print("\nYear:", year)

    for column in pollutants:

        missing = yearly[column].isna().sum()

        total = len(yearly)

        percentage = (missing / total) * 100

        print(
            f"{column:20s} "
            f"{missing:5d} missing "
            f"({percentage:.2f}%)"
        )


# ROWS WITH COMPLETE AIR QUALITY DATA

complete = df[pollutants].notna().all(axis=1)

print("\n" + "=" * 60)

print(
    "Rows with ALL air-quality values available:",
    complete.sum()
)

print(
    "Rows with at least one missing air-quality value:",
    (~complete).sum()
)



# DATE RANGE WITH COMPLETE DATA

complete_df = df[complete]

if len(complete_df) > 0:

    print(
        "\nComplete air-quality date range:"
    )

    print(
        complete_df["timestamp"].min(),
        "to",
        complete_df["timestamp"].max()
    )


# SAVE COMPLETE DATA

complete_df = complete_df.drop(
    columns=["year"]
)

output_file = "data/raw/lahore_air_quality_complete.csv"

complete_df.to_csv(
    output_file,
    index=False
)

print(
    "\nComplete dataset saved to:",
    output_file
)

print("\n" + "=" * 60)
print("CHECK COMPLETED")
print("=" * 60)