from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Project root = folder containing "scripts"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "dataset"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. FIND JM1 CSV FILE
# ============================================================

csv_files = list(RAW_DIR.glob("*.csv"))

if not csv_files:
    print("ERROR: No CSV file found in:")
    print(RAW_DIR)
    sys.exit(1)

# Prefer a file containing "jm1" in its name
jm1_files = [
    file for file in csv_files
    if "jm1" in file.stem.lower()
]

if len(jm1_files) == 1:
    DATASET_PATH = jm1_files[0]

elif len(jm1_files) > 1:
    print("Multiple JM1 CSV files were found:\n")

    for i, file in enumerate(jm1_files, start=1):
        print(f"{i}. {file.name}")

    print("\nPlease keep only the correct JM1 dataset in data/raw/")
    sys.exit(1)

elif len(csv_files) == 1:
    DATASET_PATH = csv_files[0]

else:
    print("Multiple CSV files were found:\n")

    for i, file in enumerate(csv_files, start=1):
        print(f"{i}. {file.name}")

    print(
        "\nPlease keep only the JM1 dataset in "
        "data/raw/ or rename it to include 'JM1'."
    )

    sys.exit(1)


print("=" * 70)
print("JM1 DATASET INSPECTION")
print("=" * 70)

print(f"\nDataset file:")
print(DATASET_PATH)


# ============================================================
# 3. LOAD DATASET
# ============================================================

try:
    df = pd.read_csv(DATASET_PATH)

except Exception as e:
    print("\nERROR while reading the CSV:")
    print(e)
    sys.exit(1)


# ============================================================
# 4. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC DATASET INFORMATION")
print("=" * 70)

print(f"\nNumber of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")

print("\nColumn names:")

for i, column in enumerate(df.columns, start=1):
    print(f"{i:02d}. {column}")


# ============================================================
# 5. DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("2. DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ============================================================
# 6. FIRST FIVE ROWS
# ============================================================

print("\n" + "=" * 70)
print("3. FIRST FIVE ROWS")
print("=" * 70)

print(df.head().to_string())


# ============================================================
# 7. LAST FIVE ROWS
# ============================================================

print("\n" + "=" * 70)
print("4. LAST FIVE ROWS")
print("=" * 70)

print(df.tail().to_string())


# ============================================================
# 8. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("5. MISSING VALUES")
print("=" * 70)

missing_df = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isna().sum().values,
    "missing_percentage": (
        df.isna().mean().values * 100
    )
})

missing_df = missing_df.sort_values(
    by="missing_count",
    ascending=False
)

print(missing_df.to_string(index=False))


# ============================================================
# 9. DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 70)
print("6. DUPLICATE ROWS")
print("=" * 70)

duplicate_count = int(df.duplicated().sum())

duplicate_percentage = (
    duplicate_count / len(df) * 100
    if len(df) > 0
    else 0
)

print(f"Duplicate rows       : {duplicate_count}")
print(f"Duplicate percentage : {duplicate_percentage:.4f}%")


# ============================================================
# 10. INFINITE VALUES
# ============================================================

print("\n" + "=" * 70)
print("7. INFINITE VALUES")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns

if len(numeric_columns) > 0:

    infinite_counts = np.isinf(
        df[numeric_columns]
    ).sum()

    infinite_counts = infinite_counts[
        infinite_counts > 0
    ]

    if len(infinite_counts) == 0:
        print("No infinite values found.")

    else:
        print("Infinite values found:")
        print(infinite_counts.to_string())

else:
    print("No numeric columns found.")


# ============================================================
# 11. UNIQUE VALUES
# ============================================================

print("\n" + "=" * 70)
print("8. UNIQUE VALUES")
print("=" * 70)

unique_df = pd.DataFrame({
    "column": df.columns,
    "unique_values": [
        df[column].nunique(dropna=False)
        for column in df.columns
    ]
})

print(unique_df.to_string(index=False))


# ============================================================
# 12. CONSTANT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("9. CONSTANT FEATURES")
print("=" * 70)

constant_columns = [
    column
    for column in df.columns
    if df[column].nunique(dropna=False) <= 1
]

if constant_columns:
    print("Constant columns found:")

    for column in constant_columns:
        print(f"- {column}")

else:
    print("No constant columns found.")


# ============================================================
# 13. TARGET COLUMN DETECTION
# ============================================================

print("\n" + "=" * 70)
print("10. TARGET VARIABLE")
print("=" * 70)

# Look for common defect target names
target_candidates = [
    "defects",
    "defect",
    "bug",
    "bugs",
    "label",
    "target"
]

column_lookup = {
    column.lower().strip(): column
    for column in df.columns
}

target_column = None

for candidate in target_candidates:

    if candidate in column_lookup:
        target_column = column_lookup[candidate]
        break


if target_column is None:

    print("WARNING: Target column could not be detected automatically.")

    print("\nAvailable columns:")

    for column in df.columns:
        print(f"- {column}")

    print(
        "\nPlease check the target column manually before "
        "continuing to Step 2."
    )

else:

    print(f"Detected target column: {target_column}")

    print("\nTarget value counts:")

    print(
        df[target_column]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nTarget distribution (%):")

    target_distribution = (
        df[target_column]
        .value_counts(
            normalize=True,
            dropna=False
        )
        .mul(100)
    )

    print(target_distribution.to_string())


# ============================================================
# 14. NUMERICAL DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("11. NUMERICAL DESCRIPTIVE STATISTICS")
print("=" * 70)

if len(numeric_columns) > 0:

    descriptive_stats = df[numeric_columns].describe().T

    print(
        descriptive_stats.to_string()
    )

else:

    descriptive_stats = pd.DataFrame()

    print("No numeric columns found.")


# ============================================================
# 15. NON-NUMERIC COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("12. NON-NUMERIC COLUMNS")
print("=" * 70)

non_numeric_columns = df.select_dtypes(
    exclude=[np.number]
).columns.tolist()

if non_numeric_columns:

    print("Non-numeric columns:")

    for column in non_numeric_columns:
        print(
            f"- {column} "
            f"(dtype={df[column].dtype})"
        )

else:

    print("All columns are numeric.")


# ============================================================
# 16. DATASET MEMORY USAGE
# ============================================================

print("\n" + "=" * 70)
print("13. MEMORY USAGE")
print("=" * 70)

memory_mb = df.memory_usage(
    deep=True
).sum() / (1024 ** 2)

print(
    f"Dataset memory usage: "
    f"{memory_mb:.4f} MB"
)


# ============================================================
# 17. SAVE DATASET PROFILE
# ============================================================

profile_df = pd.DataFrame({
    "column": df.columns,
    "dtype": [
        str(df[column].dtype)
        for column in df.columns
    ],
    "missing_count": [
        int(df[column].isna().sum())
        for column in df.columns
    ],
    "missing_percentage": [
        float(df[column].isna().mean() * 100)
        for column in df.columns
    ],
    "unique_values": [
        int(df[column].nunique(dropna=False))
        for column in df.columns
    ]
})

profile_path = RESULTS_DIR / "dataset_profile.csv"

profile_df.to_csv(
    profile_path,
    index=False
)


# ============================================================
# 18. SAVE MISSING-VALUE REPORT
# ============================================================

missing_path = RESULTS_DIR / "missing_values.csv"

missing_df.to_csv(
    missing_path,
    index=False
)


# ============================================================
# 19. SAVE UNIQUE-VALUE REPORT
# ============================================================

unique_path = RESULTS_DIR / "unique_values.csv"

unique_df.to_csv(
    unique_path,
    index=False
)


# ============================================================
# 20. SAVE DESCRIPTIVE STATISTICS
# ============================================================

if not descriptive_stats.empty:

    descriptive_path = (
        RESULTS_DIR /
        "descriptive_statistics.csv"
    )

    descriptive_stats.to_csv(
        descriptive_path
    )


# ============================================================
# 21. SAVE TARGET DISTRIBUTION
# ============================================================

if target_column is not None:

    target_counts = (
        df[target_column]
        .value_counts(dropna=False)
        .reset_index()
    )

    target_counts.columns = [
        "class",
        "count"
    ]

    target_counts["percentage"] = (
        target_counts["count"]
        / len(df)
        * 100
    )

    target_path = (
        RESULTS_DIR /
        "class_distribution.csv"
    )

    target_counts.to_csv(
        target_path,
        index=False
    )


# ============================================================
# 22. SAVE SUMMARY INFORMATION
# ============================================================

summary = {
    "dataset_file": DATASET_PATH.name,
    "number_of_rows": int(df.shape[0]),
    "number_of_columns": int(df.shape[1]),
    "duplicate_rows": duplicate_count,
    "duplicate_percentage": duplicate_percentage,
    "target_column": target_column,
    "numeric_columns": len(numeric_columns),
    "non_numeric_columns": len(non_numeric_columns),
    "constant_columns": constant_columns,
    "memory_usage_mb": memory_mb
}

summary_path = RESULTS_DIR / "dataset_summary.json"

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ============================================================
# 23. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("DATASET INSPECTION COMPLETED")
print("=" * 70)

print("\nResults saved to:")

print(RESULTS_DIR)

print("\nGenerated files:")

for file in sorted(RESULTS_DIR.iterdir()):

    if file.is_file():
        print(f"- {file.name}")

print("\nIMPORTANT:")
print("The original JM1 dataset has NOT been modified.")
