"""
04a_train_test_split.py

JM1 Software Defect Prediction
Independent Development/Test Split

Purpose:
    Create a fixed, stratified 80/20 development-test split.

Design:
    80% Development Set
        └── Used for all cross-validation,
            SMOTE, hyperparameter tuning,
            SHAP analysis and model development.

    20% Test Set
        └── Completely untouched until final evaluation.

Important:
    - Stratified split
    - Random state = 42
    - No SMOTE
    - No model training
    - No feature selection
    - No hyperparameter tuning

Input:
    data/processed/JM1_cleaned.csv

Outputs:
    data/processed/JM1/JM1_development.csv
    data/processed/JM1/JM1_test.csv
    data/results/dataset/JM1_train_test_split_summary.csv
"""


from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "JM1_cleaned.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "JM1"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "dataset"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

TARGET = "defects"

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("JM1 INDEPENDENT TRAIN/TEST SPLIT")
print("=" * 80)

print(f"\nInput dataset:")
print(INPUT_FILE)

print("\nExperiment design:")
print("Development set : 80%")
print("Test set        : 20%")
print("Split type      : Stratified")
print(f"Random state    : {RANDOM_STATE}")

print("\nImportant:")
print("The test set will remain completely untouched")
print("until final model evaluation.")


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_FILE}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("LOADING CLEANED JM1 DATASET")
print("=" * 80)

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Dataset shape: {df.shape}"
)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)


# ============================================================
# TARGET CHECK
# ============================================================

if TARGET not in df.columns:

    raise ValueError(
        f"Target column '{TARGET}' was not found."
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("ORIGINAL CLASS DISTRIBUTION")
print("=" * 80)

class_counts = (
    df[TARGET]
    .value_counts()
    .sort_index()
)

for class_value, count in class_counts.items():

    percentage = (
        count / len(df)
    ) * 100

    class_name = (
        "Non-defective"
        if int(class_value) == 0
        else "Defective"
    )

    print(
        f"{class_name:15s} "
        f"({class_value}): "
        f"{count:,} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# STRATIFIED TRAIN/TEST SPLIT
# ============================================================

print("\n" + "=" * 80)
print("CREATING STRATIFIED 80/20 SPLIT")
print("=" * 80)


development_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df[TARGET],
)


# ============================================================
# RESET INDEX
# ============================================================

development_df = (
    development_df
    .reset_index(drop=True)
)

test_df = (
    test_df
    .reset_index(drop=True)
)


# ============================================================
# VERIFY NO ROW OVERLAP
# ============================================================

print("\nChecking for row overlap...")

# Create deterministic row signatures
development_signatures = set(
    development_df.astype(str).agg(
        "|".join,
        axis=1
    )
)

test_signatures = set(
    test_df.astype(str).agg(
        "|".join,
        axis=1
    )
)

overlap = (
    development_signatures
    & test_signatures
)

if len(overlap) > 0:

    raise RuntimeError(
        f"ERROR: {len(overlap)} overlapping rows detected."
    )

print("✓ No row overlap detected.")


# ============================================================
# SAVE DEVELOPMENT SET
# ============================================================

development_file = (
    OUTPUT_DIR
    / "JM1_development.csv"
)

development_df.to_csv(
    development_file,
    index=False
)


# ============================================================
# SAVE TEST SET
# ============================================================

test_file = (
    OUTPUT_DIR
    / "JM1_test.csv"
)

test_df.to_csv(
    test_file,
    index=False
)


# ============================================================
# DISPLAY SPLIT INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("SPLIT RESULTS")
print("=" * 80)

print(
    f"\nOriginal dataset : "
    f"{len(df):,} rows"
)

print(
    f"Development set  : "
    f"{len(development_df):,} rows "
    f"({len(development_df) / len(df) * 100:.2f}%)"
)

print(
    f"Test set         : "
    f"{len(test_df):,} rows "
    f"({len(test_df) / len(df) * 100:.2f}%)"
)


# ============================================================
# CLASS DISTRIBUTION FUNCTION
# ============================================================

def print_distribution(
    data,
    name
):

    print(
        f"\n{name} class distribution:"
    )

    counts = (
        data[TARGET]
        .value_counts()
        .sort_index()
    )

    for class_value, count in counts.items():

        percentage = (
            count / len(data)
        ) * 100

        class_name = (
            "Non-defective"
            if int(class_value) == 0
            else "Defective"
        )

        print(
            f"  {class_name:15s}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )


print_distribution(
    development_df,
    "Development"
)

print_distribution(
    test_df,
    "Test"
)


# ============================================================
# CREATE SUMMARY TABLE
# ============================================================

summary_rows = []

for split_name, data in [
    ("Full Dataset", df),
    ("Development", development_df),
    ("Test", test_df),
]:

    counts = (
        data[TARGET]
        .value_counts()
    )

    total = len(data)

    non_defective = int(
        counts.get(0, 0)
    )

    defective = int(
        counts.get(1, 0)
    )

    summary_rows.append(
        {
            "split": split_name,
            "total_samples": total,
            "non_defective": non_defective,
            "defective": defective,
            "non_defective_percentage":
                non_defective / total * 100,
            "defective_percentage":
                defective / total * 100,
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = (
    RESULTS_DIR
    / "JM1_train_test_split_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("TRAIN/TEST SPLIT COMPLETED SUCCESSFULLY")
print("=" * 80)

print("\nFiles created:")

print(
    f"✓ {development_file}"
)

print(
    f"✓ {test_file}"
)

print(
    f"✓ {summary_file}"
)

print("\n" + "=" * 80)
print("IMPORTANT EXPERIMENTAL RULE")
print("=" * 80)

print(
    """
Development set:
    Used for cross-validation,
    SMOTE experiments,
    hyperparameter tuning,
    feature analysis,
    SHAP analysis,
    RF-RID development,
    and model selection.

Test set:
    NEVER use for:
        - SMOTE
        - feature selection
        - hyperparameter tuning
        - model selection
        - SHAP-based model selection

The test set is reserved for FINAL evaluation only.
"""
)

print("=" * 80)
print("READY FOR DEVELOPMENT-SET MODELING")
print("=" * 80)
