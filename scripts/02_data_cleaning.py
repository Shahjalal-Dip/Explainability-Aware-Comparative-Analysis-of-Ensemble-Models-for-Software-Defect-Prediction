from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "cleaning"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOCATE JM1 DATASET
# ============================================================

csv_files = list(RAW_DIR.glob("*.csv"))

if not csv_files:
    print("ERROR: No CSV file found in:")
    print(RAW_DIR)
    sys.exit(1)


jm1_files = [
    file for file in csv_files
    if "jm1" in file.stem.lower()
]


if len(jm1_files) == 1:

    DATASET_PATH = jm1_files[0]

elif len(jm1_files) > 1:

    print("Multiple JM1 files found:")
    for file in jm1_files:
        print(f"- {file.name}")

    sys.exit(1)

elif len(csv_files) == 1:

    DATASET_PATH = csv_files[0]

else:

    print("Multiple CSV files found:")
    for file in csv_files:
        print(f"- {file.name}")

    print("\nPlease keep only the JM1 dataset in data/raw/.")

    sys.exit(1)


# ============================================================
# 3. LOAD RAW DATASET
# ============================================================

print("=" * 70)
print("JM1 DATA CLEANING AND PREPROCESSING")
print("=" * 70)

print(f"\nInput dataset:")
print(DATASET_PATH)

try:

    df = pd.read_csv(DATASET_PATH)

except Exception as e:

    print("\nERROR while loading dataset:")
    print(e)
    sys.exit(1)


original_rows = len(df)
original_columns = len(df.columns)


print(f"\nOriginal rows    : {original_rows}")
print(f"Original columns : {original_columns}")


# ============================================================
# 4. IDENTIFY TARGET
# ============================================================

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

    print("\nERROR: Target column could not be identified.")

    print("\nAvailable columns:")
    for column in df.columns:
        print(f"- {column}")

    sys.exit(1)


print(f"\nTarget column: {target_column}")


# ============================================================
# 5. INITIAL DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("1. INITIAL DATA QUALITY CHECK")
print("=" * 70)


missing_before = int(df.isna().sum().sum())


numeric_columns_before = df.select_dtypes(
    include=[np.number]
).columns.tolist()


if numeric_columns_before:

    infinite_before = int(
        np.isinf(
            df[numeric_columns_before]
        ).sum().sum()
    )

else:

    infinite_before = 0


duplicate_rows_before = int(
    df.duplicated().sum()
)


print(f"\nMissing values       : {missing_before}")
print(f"Infinite values      : {infinite_before}")
print(f"Exact duplicate rows : {duplicate_rows_before}")


# ============================================================
# 6. DUPLICATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("2. DUPLICATE ANALYSIS")
print("=" * 70)


# Features = everything except target
feature_columns = [
    column
    for column in df.columns
    if column != target_column
]


# ------------------------------------------------------------
# 6.1 Exact duplicate rows
# ------------------------------------------------------------

exact_duplicate_mask = df.duplicated(
    keep=False
)

exact_duplicate_count = int(
    df.duplicated().sum()
)


# ------------------------------------------------------------
# 6.2 Duplicate feature vectors
# ------------------------------------------------------------

feature_duplicate_mask = df.duplicated(
    subset=feature_columns,
    keep=False
)

feature_duplicate_count = int(
    df.duplicated(
        subset=feature_columns
    ).sum()
)


print(
    f"\nExact duplicate rows "
    f"(excluding first occurrence): "
    f"{exact_duplicate_count}"
)

print(
    f"Rows belonging to repeated "
    f"feature vectors: "
    f"{int(feature_duplicate_mask.sum())}"
)


# ============================================================
# 7. CHECK FOR CONFLICTING DUPLICATE LABELS
# ============================================================

print("\n" + "=" * 70)
print("3. DUPLICATE LABEL CONSISTENCY")
print("=" * 70)


# For each identical feature vector,
# count the number of unique target labels.

duplicate_feature_groups = (
    df.groupby(feature_columns, dropna=False)[target_column]
    .nunique()
)


conflicting_feature_groups = (
    duplicate_feature_groups[
        duplicate_feature_groups > 1
    ]
)


conflicting_group_count = int(
    len(conflicting_feature_groups)
)


print(
    f"\nFeature groups with conflicting labels: "
    f"{conflicting_group_count}"
)


if conflicting_group_count > 0:

    print(
        "\nWARNING:"
        "\nSome identical feature vectors have "
        "different defect labels."
    )

    print(
        "\nThese observations require special "
        "investigation before final cleaning."
    )

else:

    print(
        "\nNo conflicting labels were found "
        "among identical feature vectors."
    )


# ============================================================
# 8. SAVE DUPLICATE ANALYSIS
# ============================================================

duplicate_summary = pd.DataFrame({
    "metric": [
        "original_rows",
        "exact_duplicate_rows",
        "exact_duplicate_percentage",
        "rows_in_repeated_feature_groups",
        "conflicting_feature_groups"
    ],

    "value": [
        original_rows,
        exact_duplicate_count,
        (
            exact_duplicate_count
            / original_rows
            * 100
        ),
        int(feature_duplicate_mask.sum()),
        conflicting_group_count
    ]
})


duplicate_summary_path = (
    RESULTS_DIR /
    "duplicate_analysis_summary.csv"
)

duplicate_summary.to_csv(
    duplicate_summary_path,
    index=False
)


# ============================================================
# 9. REMOVE EXACT DUPLICATE OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("4. DUPLICATE REMOVAL")
print("=" * 70)


if conflicting_group_count > 0:

    print(
        "\nERROR: Conflicting duplicate labels "
        "were detected."
    )

    print(
        "The script will NOT create the final "
        "cleaned dataset automatically."
    )

    print(
        "\nPlease investigate the conflicting "
        "feature groups before proceeding."
    )

    sys.exit(1)


# Since no conflicting labels exist,
# retain one copy of each exact observation.

df_clean = df.drop_duplicates(
    keep="first"
).copy()


rows_after_duplicate_removal = len(
    df_clean
)

removed_duplicates = (
    original_rows
    - rows_after_duplicate_removal
)


print(
    f"\nRows before duplicate removal : "
    f"{original_rows}"
)

print(
    f"Rows after duplicate removal  : "
    f"{rows_after_duplicate_removal}"
)

print(
    f"Rows removed                   : "
    f"{removed_duplicates}"
)


# ============================================================
# 10. CONVERT TARGET TO BINARY INTEGER
# ============================================================

print("\n" + "=" * 70)
print("5. TARGET VARIABLE CONVERSION")
print("=" * 70)


print(
    f"\nOriginal target dtype: "
    f"{df_clean[target_column].dtype}"
)


# Handle Boolean target explicitly.
if pd.api.types.is_bool_dtype(
    df_clean[target_column]
):

    df_clean[target_column] = (
        df_clean[target_column]
        .astype(int)
    )

else:

    # Try numeric conversion
    try:

        df_clean[target_column] = pd.to_numeric(
            df_clean[target_column]
        )

    except Exception as e:

        print(
            "\nERROR: Target cannot be "
            "converted to numeric values."
        )

        print(e)

        sys.exit(1)


print(
    f"New target dtype: "
    f"{df_clean[target_column].dtype}"
)


print("\nTarget distribution:")

target_counts = (
    df_clean[target_column]
    .value_counts()
    .sort_index()
)

print(target_counts.to_string())


# ============================================================
# 11. VERIFY TARGET VALUES
# ============================================================

unique_target_values = sorted(
    df_clean[target_column]
    .dropna()
    .unique()
    .tolist()
)


print(
    f"\nUnique target values: "
    f"{unique_target_values}"
)


if not set(unique_target_values).issubset(
    {0, 1}
):

    print(
        "\nWARNING: Target contains values "
        "other than 0 and 1."
    )

    print(
        "Please investigate before modeling."
    )

    sys.exit(1)


# ============================================================
# 12. VERIFY FEATURE TYPES
# ============================================================

print("\n" + "=" * 70)
print("6. FEATURE TYPE CHECK")
print("=" * 70)


feature_columns = [
    column
    for column in df_clean.columns
    if column != target_column
]


non_numeric_features = (
    df_clean[feature_columns]
    .select_dtypes(
        exclude=[np.number]
    )
    .columns
    .tolist()
)


if non_numeric_features:

    print(
        "\nNon-numeric predictor columns found:"
    )

    for column in non_numeric_features:
        print(f"- {column}")

    print(
        "\nThese must be handled before "
        "model training."
    )

else:

    print(
        "\nAll 21 predictor variables are numeric."
    )


# ============================================================
# 13. CHECK MISSING VALUES AFTER CLEANING
# ============================================================

print("\n" + "=" * 70)
print("7. POST-CLEANING QUALITY CHECK")
print("=" * 70)


missing_after = int(
    df_clean.isna().sum().sum()
)


numeric_columns_after = df_clean.select_dtypes(
    include=[np.number]
).columns


if len(numeric_columns_after) > 0:

    infinite_after = int(
        np.isinf(
            df_clean[numeric_columns_after]
        ).sum().sum()
    )

else:

    infinite_after = 0


constant_columns = [
    column
    for column in df_clean.columns
    if df_clean[column].nunique(
        dropna=False
    ) <= 1
]


print(
    f"\nMissing values       : "
    f"{missing_after}"
)

print(
    f"Infinite values      : "
    f"{infinite_after}"
)

print(
    f"Constant columns     : "
    f"{len(constant_columns)}"
)


if constant_columns:

    print("\nConstant columns:")

    for column in constant_columns:
        print(f"- {column}")


# ============================================================
# 14. SAVE CLEANED DATASET
# ============================================================

cleaned_dataset_path = (
    PROCESSED_DIR /
    "JM1_clean.csv"
)


df_clean.to_csv(
    cleaned_dataset_path,
    index=False
)


# ============================================================
# 15. SAVE TARGET DISTRIBUTION
# ============================================================

clean_target_distribution = (
    df_clean[target_column]
    .value_counts()
    .sort_index()
    .reset_index()
)

clean_target_distribution.columns = [
    "defects",
    "count"
]

clean_target_distribution["percentage"] = (
    clean_target_distribution["count"]
    / len(df_clean)
    * 100
)


clean_target_distribution.to_csv(
    RESULTS_DIR /
    "cleaned_class_distribution.csv",
    index=False
)


# ============================================================
# 16. SAVE CLEANING SUMMARY
# ============================================================

cleaning_summary = {

    "input_file": DATASET_PATH.name,

    "original_rows": int(original_rows),

    "original_columns": int(original_columns),

    "duplicate_rows_removed": int(
        removed_duplicates
    ),

    "rows_after_cleaning": int(
        len(df_clean)
    ),

    "columns_after_cleaning": int(
        len(df_clean.columns)
    ),

    "target_column": target_column,

    "target_values": unique_target_values,

    "missing_values_after_cleaning": int(
        missing_after
    ),

    "infinite_values_after_cleaning": int(
        infinite_after
    ),

    "constant_columns": constant_columns,

    "non_numeric_features": non_numeric_features
}


summary_path = (
    RESULTS_DIR /
    "cleaning_summary.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        cleaning_summary,
        file,
        indent=4
    )


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("DATA CLEANING COMPLETED")
print("=" * 70)


print(
    f"\nCleaned dataset saved to:"
    f"\n{cleaned_dataset_path}"
)


print(
    "\nCleaning results saved to:"
    f"\n{RESULTS_DIR}"
)


print("\nFinal dataset:")
print(
    f"Rows    : {df_clean.shape[0]}"
)

print(
    f"Columns : {df_clean.shape[1]}"
)


print("\nGenerated files:")

for file in sorted(
    RESULTS_DIR.iterdir()
):

    if file.is_file():
        print(f"- {file.name}")


print("\nIMPORTANT:")
print(
    "The original dataset in data/raw/ "
    "has NOT been modified."
)
