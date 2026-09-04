"""
02b_finalize_cleaning.py

JM1 Software Defect Prediction
Final Dataset Cleaning and Validation

Purpose:
    - Load JM1 raw dataset
    - Identify duplicate observations
    - Remove exact duplicate rows
    - Preserve conflicting feature vectors
    - Generate a final modeling dataset
    - Generate cleaning statistics for the paper

IMPORTANT:
    This script does NOT remove all feature-conflict groups.
    Conflicting feature vectors are retained because they represent
    genuine label ambiguity and are important for the research analysis.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "JM1.csv"

RESULT_DIR = PROJECT_ROOT / "data" / "results" / "dataset"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. OUTPUT FILES
# ============================================================

CLEANED_FILE = PROCESSED_DIR / "JM1_cleaned.csv"

CLEANING_SUMMARY_FILE = RESULT_DIR / "cleaning_summary.csv"

CLASS_DISTRIBUTION_FILE = RESULT_DIR / "cleaned_class_distribution.csv"

DUPLICATE_SUMMARY_FILE = RESULT_DIR / "duplicate_summary.csv"

CONFLICT_SUMMARY_FILE = RESULT_DIR / "conflict_preservation_summary.csv"


# ============================================================
# 3. HELPER FUNCTION
# ============================================================

def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# 4. CHECK INPUT
# ============================================================

print_section("JM1 FINAL DATASET CLEANING")

print("Project root:")
print(PROJECT_ROOT)

print("\nRaw dataset:")
print(RAW_FILE)

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"\nCould not find JM1 dataset:\n{RAW_FILE}\n\n"
        "Please check that JM1.csv is located in data/raw/"
    )


# ============================================================
# 5. LOAD DATA
# ============================================================

print_section("LOADING RAW DATA")

df = pd.read_csv(RAW_FILE)

print(f"Dataset shape: {df.shape}")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

print("\nColumns:")
for col in df.columns:
    print(f"  - {col}")


# ============================================================
# 6. IDENTIFY TARGET COLUMN
# ============================================================

print_section("IDENTIFYING TARGET COLUMN")

possible_targets = [
    "defects",
    "Defects",
    "defect",
    "Defect",
    "bug",
    "Bug",
    "label",
    "Label",
    "target",
    "Target"
]

target_col = None

for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    raise ValueError(
        "\nCould not automatically identify the defect-label column.\n"
        f"Available columns:\n{list(df.columns)}"
    )

print(f"Target column: {target_col}")


# ============================================================
# 7. BASIC DATA VALIDATION
# ============================================================

print_section("BASIC DATA VALIDATION")

print("\nMissing values:")
missing = df.isnull().sum()

missing_total = int(missing.sum())

if missing_total == 0:
    print("  No missing values found.")
else:
    print(missing[missing > 0])


print("\nDuplicate rows:")
exact_duplicates = int(df.duplicated().sum())

print(f"  Exact duplicate rows: {exact_duplicates}")


# ============================================================
# 8. NORMALIZE TARGET LABEL
# ============================================================

print_section("NORMALIZING DEFECT LABEL")

print("Original target values:")
print(df[target_col].value_counts(dropna=False))


def normalize_label(value):

    if pd.isna(value):
        return np.nan

    if isinstance(value, bool):
        return value

    value_str = str(value).strip().lower()

    if value_str in ["true", "1", "yes", "y", "defective", "defect"]:
        return True

    if value_str in ["false", "0", "no", "n", "non-defective", "nondefective"]:
        return False

    return value


df[target_col] = df[target_col].apply(normalize_label)

print("\nNormalized target values:")
print(df[target_col].value_counts(dropna=False))


# ============================================================
# 9. REMOVE ROWS WITH MISSING TARGET
# ============================================================

print_section("TARGET VALIDATION")

before_target_removal = len(df)

df = df.dropna(subset=[target_col]).copy()

removed_missing_target = before_target_removal - len(df)

print(f"Rows before target validation: {before_target_removal}")
print(f"Rows removed due to missing target: {removed_missing_target}")
print(f"Rows after target validation: {len(df)}")


# ============================================================
# 10. IDENTIFY FEATURE COLUMNS
# ============================================================

print_section("IDENTIFYING FEATURE COLUMNS")

feature_columns = [
    col for col in df.columns
    if col != target_col
]

print(f"Number of feature columns: {len(feature_columns)}")

print("\nFeature columns:")

for col in feature_columns:
    print(f"  - {col}")


# ============================================================
# 11. REMOVE EXACT DUPLICATES
# ============================================================

print_section("EXACT DUPLICATE REMOVAL")

rows_before_duplicates = len(df)

exact_duplicate_mask = df.duplicated(
    subset=feature_columns + [target_col],
    keep="first"
)

exact_duplicates_removed = int(exact_duplicate_mask.sum())

df_clean = df.loc[~exact_duplicate_mask].copy()

print(f"Rows before duplicate removal: {rows_before_duplicates}")
print(f"Exact duplicate rows removed: {exact_duplicates_removed}")
print(f"Rows after duplicate removal: {len(df_clean)}")


# ============================================================
# 12. CHECK FEATURE DUPLICATES
# ============================================================

print_section("FEATURE-LEVEL DUPLICATE ANALYSIS")

feature_duplicate_mask = df_clean.duplicated(
    subset=feature_columns,
    keep=False
)

feature_duplicate_rows = int(feature_duplicate_mask.sum())

unique_feature_vectors = int(
    df_clean[feature_columns].drop_duplicates().shape[0]
)

print(
    f"Rows belonging to feature-duplicate groups: "
    f"{feature_duplicate_rows}"
)

print(
    f"Unique feature vectors: "
    f"{unique_feature_vectors}"
)


# ============================================================
# 13. IDENTIFY REMAINING LABEL CONFLICTS
# ============================================================

print_section("REMAINING LABEL-CONFLICT ANALYSIS")

conflict_groups = (
    df_clean
    .groupby(feature_columns, dropna=False)[target_col]
    .nunique()
)

conflicting_vectors = conflict_groups[
    conflict_groups > 1
]

number_conflict_groups = len(conflicting_vectors)

print(f"Remaining conflicting feature vectors: {number_conflict_groups}")


# ============================================================
# 14. COUNT ROWS IN CONFLICT GROUPS
# ============================================================

if number_conflict_groups > 0:

    conflict_index = (
        df_clean
        .groupby(feature_columns, dropna=False)[target_col]
        .transform("nunique")
        > 1
    )

    conflict_rows = int(conflict_index.sum())

else:

    conflict_rows = 0


print(
    f"Rows belonging to remaining conflict groups: "
    f"{conflict_rows}"
)


# ============================================================
# 15. CLASS DISTRIBUTION
# ============================================================

print_section("FINAL CLASS DISTRIBUTION")

class_counts = df_clean[target_col].value_counts()

class_percentages = (
    df_clean[target_col]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

class_distribution = pd.DataFrame({
    "class": class_counts.index.astype(str),
    "count": class_counts.values,
    "percentage": [
        class_percentages.loc[x]
        for x in class_counts.index
    ]
})

print(class_distribution.to_string(index=False))


# ============================================================
# 16. FINAL DUPLICATE CHECK
# ============================================================

print_section("FINAL DATA VALIDATION")

remaining_exact_duplicates = int(
    df_clean.duplicated(
        subset=feature_columns + [target_col]
    ).sum()
)

print(
    f"Remaining exact duplicates: "
    f"{remaining_exact_duplicates}"
)

if remaining_exact_duplicates == 0:
    print("✓ No exact duplicate rows remain.")
else:
    print("WARNING: Exact duplicates remain.")


# ============================================================
# 17. FINAL DATASET SIZE
# ============================================================

print_section("FINAL DATASET SIZE")

print(f"Original rows:              {len(df)}")
print(f"Exact duplicates removed:  {exact_duplicates_removed}")
print(f"Final cleaned rows:         {len(df_clean)}")

print(f"\nOriginal columns:           {len(df.columns)}")
print(f"Final feature columns:      {len(feature_columns)}")
print(f"Target column:              {target_col}")


# ============================================================
# 18. SAVE CLEANED DATASET
# ============================================================

print_section("SAVING CLEANED DATASET")

df_clean.to_csv(
    CLEANED_FILE,
    index=False
)

print(f"Saved cleaned dataset to:")
print(CLEANED_FILE)


# ============================================================
# 19. SAVE CLEANING SUMMARY
# ============================================================

cleaning_summary = pd.DataFrame({
    "metric": [
        "original_rows",
        "original_columns",
        "missing_target_removed",
        "exact_duplicates_removed",
        "final_rows",
        "final_columns",
        "feature_columns",
        "feature_duplicate_rows",
        "conflict_groups",
        "conflict_rows",
        "remaining_exact_duplicates"
    ],
    "value": [
        len(df),
        len(df.columns),
        removed_missing_target,
        exact_duplicates_removed,
        len(df_clean),
        len(df_clean.columns),
        len(feature_columns),
        feature_duplicate_rows,
        number_conflict_groups,
        conflict_rows,
        remaining_exact_duplicates
    ]
})

cleaning_summary.to_csv(
    CLEANING_SUMMARY_FILE,
    index=False
)


# ============================================================
# 20. SAVE CLASS DISTRIBUTION
# ============================================================

class_distribution.to_csv(
    CLASS_DISTRIBUTION_FILE,
    index=False
)


# ============================================================
# 21. SAVE DUPLICATE SUMMARY
# ============================================================

duplicate_summary = pd.DataFrame({
    "metric": [
        "exact_duplicates_before_cleaning",
        "exact_duplicates_removed",
        "feature_duplicate_rows_after_cleaning",
        "unique_feature_vectors_after_cleaning"
    ],
    "value": [
        exact_duplicates,
        exact_duplicates_removed,
        feature_duplicate_rows,
        unique_feature_vectors
    ]
})

duplicate_summary.to_csv(
    DUPLICATE_SUMMARY_FILE,
    index=False
)


# ============================================================
# 22. SAVE CONFLICT SUMMARY
# ============================================================

conflict_summary = pd.DataFrame({
    "metric": [
        "conflict_groups_before_cleaning_analysis",
        "conflict_rows_after_exact_duplicate_removal",
        "conflicts_removed",
        "conflicts_retained"
    ],
    "value": [
        88,
        conflict_rows,
        0,
        conflict_rows
    ]
})

conflict_summary.to_csv(
    CONFLICT_SUMMARY_FILE,
    index=False
)


# ============================================================
# 23. FINAL MESSAGE
# ============================================================

print_section("FINAL CLEANING COMPLETED")

print("✓ Dataset loaded")
print("✓ Target validated")
print("✓ Exact duplicates removed")
print("✓ Feature-level duplicates analyzed")
print("✓ Label conflicts analyzed")
print("✓ Class distribution calculated")
print("✓ Final dataset validated")
print("✓ Cleaned dataset saved")
print("✓ Cleaning statistics saved")

print("\nOutput files:")

print(f"\n1. Cleaned dataset:")
print(CLEANED_FILE)

print(f"\n2. Cleaning summary:")
print(CLEANING_SUMMARY_FILE)

print(f"\n3. Class distribution:")
print(CLASS_DISTRIBUTION_FILE)

print(f"\n4. Duplicate summary:")
print(DUPLICATE_SUMMARY_FILE)

print(f"\n5. Conflict summary:")
print(CONFLICT_SUMMARY_FILE)

print("\n" + "=" * 80)
print("READY FOR EDA")
print("=" * 80)
