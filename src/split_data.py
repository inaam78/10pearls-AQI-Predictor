import pandas as pd
import os

# TIME-BASED TRAIN / VALIDATION / TEST SPLIT

INPUT_FILE = "data/processed/lahore_model_data.csv"

OUTPUT_DIR = "data/splits"

TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.csv")
VALIDATION_FILE = os.path.join(OUTPUT_DIR, "validation.csv")
TEST_FILE = os.path.join(OUTPUT_DIR, "test.csv")

print("=" * 70)
print("LAHORE AQI - TIME-BASED DATA SPLITTING")
print("=" * 70)

# 1. Load dataset

print("\nLoading model dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")

# 2. Convert timestamp

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort chronologically
df = df.sort_values("timestamp").reset_index(drop=True)

# 3. Display date range

print("\nComplete dataset date range:")

print("Start:", df["timestamp"].min())
print("End:  ", df["timestamp"].max())

# 4. Determine split points

start_date = df["timestamp"].min()
end_date = df["timestamp"].max()

total_duration = end_date - start_date

# 70% training
train_end = start_date + total_duration * 0.70

# Next 15% validation
validation_end = start_date + total_duration * 0.85

print("\nSplit strategy:")
print("Training:   70%")
print("Validation: 15%")
print("Testing:    15%")

print("\nSplit dates:")
print("Training ends:   ", train_end)
print("Validation ends: ", validation_end)

# 5. Create chronological splits

train = df[df["timestamp"] <= train_end].copy()

validation = df[
    (df["timestamp"] > train_end) &
    (df["timestamp"] <= validation_end)
].copy()

test = df[df["timestamp"] > validation_end].copy()

# 6. Reset indexes

train = train.reset_index(drop=True)
validation = validation.reset_index(drop=True)
test = test.reset_index(drop=True)

# 7. Create output directory

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 8. Save datasets

train.to_csv(TRAIN_FILE, index=False)

validation.to_csv(VALIDATION_FILE, index=False)

test.to_csv(TEST_FILE, index=False)

# 9. Display results

print("\n" + "=" * 70)
print("SPLIT RESULTS")
print("=" * 70)

print("\nTRAINING DATA")
print("-" * 40)
print("Rows:", len(train))
print("Start:", train["timestamp"].min())
print("End:  ", train["timestamp"].max())

print("\nVALIDATION DATA")
print("-" * 40)
print("Rows:", len(validation))
print("Start:", validation["timestamp"].min())
print("End:  ", validation["timestamp"].max())

print("\nTEST DATA")
print("-" * 40)
print("Rows:", len(test))
print("Start:", test["timestamp"].min())
print("End:  ", test["timestamp"].max())

# 10. Verify no overlap

print("\n" + "=" * 70)
print("CHECKING FOR DATA LEAKAGE")
print("=" * 70)

print(
    "\nTraining maximum timestamp:",
    train["timestamp"].max()
)

print(
    "Validation minimum timestamp:",
    validation["timestamp"].min()
)

print(
    "\nValidation maximum timestamp:",
    validation["timestamp"].max()
)

print(
    "Test minimum timestamp:",
    test["timestamp"].min()
)

# 11. Verify chronological order

chronological = (
    train["timestamp"].max()
    < validation["timestamp"].min()
    and
    validation["timestamp"].max()
    < test["timestamp"].min()
)

if chronological:
    print("\n✓ Chronological order verified")
    print("✓ No temporal overlap between datasets")
    print("✓ No random shuffling used")
    print("✓ Data leakage check passed")

else:
    print("\n✗ WARNING: Temporal overlap detected!")

# 12. Verify target columns

target_columns = [
    f"pm2_5_t+{hour}"
    for hour in range(1, 73)
]

print("\nTarget columns:", len(target_columns))

missing_targets = []

for dataset_name, dataset in [
    ("Training", train),
    ("Validation", validation),
    ("Test", test)
]:

    missing = dataset[target_columns].isnull().sum().sum()

    print(
        f"{dataset_name} missing target values:",
        missing
    )

    if missing > 0:
        missing_targets.append(dataset_name)

# 13. Final summary

print("\n" + "=" * 70)
print("SPLIT COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles created:")

print(TRAIN_FILE)
print(VALIDATION_FILE)
print(TEST_FILE)

print("\nFinal dataset sizes:")

print("Training:   ", len(train))
print("Validation: ", len(validation))
print("Testing:    ", len(test))

print("\nTotal:", len(train) + len(validation) + len(test))

print("\n" + "=" * 70)