"""
03_eda.py

JM1 Exploratory Data Analysis (EDA)

Purpose:
    Perform exploratory analysis on the finalized cleaned JM1 dataset
    before cross-validation and predictive modeling.

Input:
    data/processed/JM1_cleaned.csv

Outputs:
    data/results/dataset/
        eda_dataset_overview.csv
        eda_feature_statistics.csv
        eda_missing_values.csv
        eda_class_distribution.csv
        eda_outlier_summary.csv
        eda_correlation_matrix.csv
        eda_high_correlations.csv
        eda_summary.txt

    figures/eda/
        class_distribution.png
        feature_distributions.png
        feature_boxplots.png
        correlation_heatmap.png

Important:
    This script ONLY performs EDA.
    It does not perform SMOTE.
    It does not perform train/test splitting.
    It does not perform model training.
    It does not modify the cleaned dataset.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "JM1_cleaned.csv"

RESULT_DIR = PROJECT_ROOT / "data" / "results" / "dataset"
FIGURE_DIR = PROJECT_ROOT / "figures" / "eda"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def save_text(filename, text):
    output_path = RESULT_DIR / filename
    output_path.write_text(text, encoding="utf-8")
    return output_path


# ============================================================
# START
# ============================================================

print_header("JM1 EXPLORATORY DATA ANALYSIS")

print("Project root:")
print(PROJECT_ROOT)

print("\nInput dataset:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    print("\nERROR:")
    print(f"Could not find cleaned dataset:")
    print(INPUT_FILE)
    print("\nRun the cleaning step first:")
    print("python scripts\\02b_finalize_cleaning.py")
    sys.exit(1)


# ============================================================
# LOAD DATA
# ============================================================

print_header("LOADING CLEANED DATASET")

df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")
print(f"Number of rows: {len(df):,}")
print(f"Number of columns: {len(df.columns)}")

print("\nColumns:")

for column in df.columns:
    print(f"  - {column}")


# ============================================================
# TARGET COLUMN
# ============================================================

print_header("IDENTIFYING TARGET COLUMN")

TARGET = "defects"

if TARGET not in df.columns:
    print(f"ERROR: Target column '{TARGET}' was not found.")
    sys.exit(1)

print(f"Target column: {TARGET}")


# ============================================================
# FEATURE COLUMNS
# ============================================================

feature_columns = [column for column in df.columns if column != TARGET]

print("\nNumber of feature columns:", len(feature_columns))

print("\nFeature columns:")

for column in feature_columns:
    print(f"  - {column}")


# ============================================================
# BASIC DATASET OVERVIEW
# ============================================================

print_header("DATASET OVERVIEW")

numeric_features = df[feature_columns].select_dtypes(
    include=[np.number]
).columns.tolist()

non_numeric_features = [
    column for column in feature_columns
    if column not in numeric_features
]

overview = pd.DataFrame({
    "metric": [
        "total_rows",
        "total_columns",
        "feature_count",
        "target_column",
        "numeric_feature_count",
        "non_numeric_feature_count",
        "missing_value_count",
        "exact_duplicate_count"
    ],
    "value": [
        len(df),
        len(df.columns),
        len(feature_columns),
        TARGET,
        len(numeric_features),
        len(non_numeric_features),
        int(df.isna().sum().sum()),
        int(df.duplicated().sum())
    ]
})

overview_path = RESULT_DIR / "eda_dataset_overview.csv"
overview.to_csv(overview_path, index=False)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Features: {len(feature_columns)}")
print(f"Numeric features: {len(numeric_features)}")
print(f"Missing values: {int(df.isna().sum().sum()):,}")
print(f"Exact duplicate rows: {int(df.duplicated().sum()):,}")


# ============================================================
# DATA TYPES
# ============================================================

print_header("DATA TYPES")

for column in df.columns:
    print(f"{column:25s} {str(df[column].dtype)}")


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

print_header("MISSING VALUE ANALYSIS")

missing = df.isna().sum()

missing_percentage = (
    missing / len(df) * 100
)

missing_summary = pd.DataFrame({
    "feature": missing.index,
    "missing_count": missing.values,
    "missing_percentage": missing_percentage.values
})

missing_summary = missing_summary.sort_values(
    by="missing_count",
    ascending=False
)

missing_path = RESULT_DIR / "eda_missing_values.csv"
missing_summary.to_csv(missing_path, index=False)

if missing.sum() == 0:
    print("No missing values found.")
else:
    print(missing_summary.to_string(index=False))


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print_header("TARGET CLASS DISTRIBUTION")

class_counts = df[TARGET].value_counts(dropna=False)

class_percentages = (
    class_counts / len(df) * 100
)

class_distribution = pd.DataFrame({
    "class": class_counts.index.astype(str),
    "count": class_counts.values,
    "percentage": class_percentages.values
})

class_distribution_path = RESULT_DIR / "eda_class_distribution.csv"
class_distribution.to_csv(
    class_distribution_path,
    index=False
)

print(class_distribution.to_string(index=False))


# ============================================================
# CLASS IMBALANCE RATIO
# ============================================================

print_header("CLASS IMBALANCE ANALYSIS")

if len(class_counts) == 2:

    majority_count = class_counts.max()
    minority_count = class_counts.min()

    imbalance_ratio = majority_count / minority_count

    print(f"Majority class count : {majority_count:,}")
    print(f"Minority class count : {minority_count:,}")
    print(f"Imbalance ratio      : {imbalance_ratio:.4f}")

else:
    imbalance_ratio = np.nan

    print(
        "Warning: target does not contain exactly two classes."
    )


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

print_header("DESCRIPTIVE STATISTICS")

feature_statistics = df[numeric_features].describe().T

feature_statistics["missing_count"] = (
    df[numeric_features].isna().sum()
)

feature_statistics["missing_percentage"] = (
    df[numeric_features].isna().mean() * 100
)

feature_statistics["skewness"] = (
    df[numeric_features].skew()
)

feature_statistics["zero_count"] = (
    (df[numeric_features] == 0).sum()
)

feature_statistics["zero_percentage"] = (
    (df[numeric_features] == 0).mean() * 100
)

feature_statistics = feature_statistics.reset_index()

feature_statistics = feature_statistics.rename(
    columns={"index": "feature"}
)

statistics_path = RESULT_DIR / "eda_feature_statistics.csv"

feature_statistics.to_csv(
    statistics_path,
    index=False
)

print(
    feature_statistics[
        [
            "feature",
            "mean",
            "std",
            "min",
            "50%",
            "max",
            "skewness"
        ]
    ].to_string(index=False)
)


# ============================================================
# OUTLIER ANALYSIS
# ============================================================

print_header("OUTLIER ANALYSIS USING IQR")

outlier_records = []

for feature in numeric_features:

    values = df[feature].dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (df[feature] < lower_bound)
        | (df[feature] > upper_bound)
    )

    outlier_count = int(outlier_mask.sum())

    outlier_percentage = (
        outlier_count / len(df) * 100
    )

    outlier_records.append({
        "feature": feature,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outlier_count,
        "outlier_percentage": outlier_percentage
    })


outlier_summary = pd.DataFrame(outlier_records)

outlier_summary = outlier_summary.sort_values(
    by="outlier_count",
    ascending=False
)

outlier_path = RESULT_DIR / "eda_outlier_summary.csv"

outlier_summary.to_csv(
    outlier_path,
    index=False
)

print(
    outlier_summary[
        [
            "feature",
            "outlier_count",
            "outlier_percentage"
        ]
    ].to_string(index=False)
)


# ============================================================
# PEARSON CORRELATION
# ============================================================

print_header("PEARSON CORRELATION ANALYSIS")

correlation_matrix = df[numeric_features].corr(
    method="pearson"
)

correlation_path = RESULT_DIR / "eda_correlation_matrix.csv"

correlation_matrix.to_csv(correlation_path)

print("Correlation matrix calculated.")


# ============================================================
# HIGH CORRELATION PAIRS
# ============================================================

print_header("HIGHLY CORRELATED FEATURE PAIRS")

correlation_pairs = []

for i in range(len(numeric_features)):

    for j in range(i + 1, len(numeric_features)):

        feature_1 = numeric_features[i]
        feature_2 = numeric_features[j]

        correlation = correlation_matrix.loc[
            feature_1,
            feature_2
        ]

        if abs(correlation) >= 0.80:

            correlation_pairs.append({
                "feature_1": feature_1,
                "feature_2": feature_2,
                "pearson_correlation": correlation,
                "absolute_correlation": abs(correlation)
            })


high_correlation = pd.DataFrame(
    correlation_pairs
)

if not high_correlation.empty:

    high_correlation = high_correlation.sort_values(
        by="absolute_correlation",
        ascending=False
    )

    print(
        high_correlation.to_string(index=False)
    )

else:

    print(
        "No feature pairs with |Pearson r| >= 0.80."
    )


high_correlation_path = (
    RESULT_DIR / "eda_high_correlations.csv"
)

high_correlation.to_csv(
    high_correlation_path,
    index=False
)


# ============================================================
# TARGET CORRELATION
# ============================================================

print_header("FEATURE-TARGET CORRELATION")

target_numeric = df[TARGET]

if target_numeric.dtype == bool:

    target_numeric = target_numeric.astype(int)

elif target_numeric.dtype == object:

    unique_target = target_numeric.dropna().unique()

    if len(unique_target) == 2:

        mapping = {
            unique_target[0]: 0,
            unique_target[1]: 1
        }

        target_numeric = target_numeric.map(mapping)


if pd.api.types.is_numeric_dtype(target_numeric):

    target_correlations = []

    for feature in numeric_features:

        correlation = df[feature].corr(
            target_numeric,
            method="pearson"
        )

        target_correlations.append({
            "feature": feature,
            "target_pearson_correlation": correlation,
            "absolute_correlation": abs(correlation)
        })

    target_correlation_df = pd.DataFrame(
        target_correlations
    ).sort_values(
        by="absolute_correlation",
        ascending=False
    )

    target_correlation_df.to_csv(
        RESULT_DIR / "eda_feature_target_correlation.csv",
        index=False
    )

    print(
        target_correlation_df.to_string(index=False)
    )


# ============================================================
# FIGURE 1: CLASS DISTRIBUTION
# ============================================================

print_header("GENERATING CLASS DISTRIBUTION FIGURE")

plt.figure(figsize=(7, 5))

labels = class_distribution["class"].astype(str)
counts = class_distribution["count"]

plt.bar(labels, counts)

plt.xlabel("Defect Class")
plt.ylabel("Number of Observations")
plt.title("JM1 Defect Class Distribution")

plt.tight_layout()

class_figure = FIGURE_DIR / "class_distribution.png"

plt.savefig(
    class_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {class_figure}")


# ============================================================
# FIGURE 2: FEATURE DISTRIBUTIONS
# ============================================================

print_header("GENERATING FEATURE DISTRIBUTION FIGURE")

# Use separate figures rather than subplots to keep figures
# publication-friendly.

for feature in numeric_features:

    plt.figure(figsize=(7, 5))

    plt.hist(
        df[feature].dropna(),
        bins=30
    )

    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {feature}")

    plt.tight_layout()

    safe_name = (
        feature
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    output_file = (
        FIGURE_DIR /
        f"distribution_{safe_name}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


print(
    f"Saved {len(numeric_features)} feature distribution figures."
)


# ============================================================
# FIGURE 3: FEATURE BOXPLOTS
# ============================================================

print_header("GENERATING FEATURE BOXPLOT FIGURES")

for feature in numeric_features:

    plt.figure(figsize=(7, 5))

    plt.boxplot(
        df[feature].dropna(),
        vert=True
    )

    plt.ylabel(feature)
    plt.title(f"Boxplot of {feature}")

    plt.tight_layout()

    safe_name = (
        feature
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    output_file = (
        FIGURE_DIR /
        f"boxplot_{safe_name}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


print(
    f"Saved {len(numeric_features)} feature boxplot figures."
)


# ============================================================
# FIGURE 4: CORRELATION HEATMAP
# ============================================================

print_header("GENERATING CORRELATION HEATMAP")

plt.figure(
    figsize=(
        max(12, len(numeric_features) * 0.6),
        max(10, len(numeric_features) * 0.5)
    )
)

plt.imshow(
    correlation_matrix,
    aspect="auto",
    interpolation="nearest"
)

plt.colorbar(
    label="Pearson Correlation"
)

plt.xticks(
    range(len(numeric_features)),
    numeric_features,
    rotation=90
)

plt.yticks(
    range(len(numeric_features)),
    numeric_features
)

plt.title(
    "Pearson Correlation Matrix of JM1 Software Metrics"
)

plt.tight_layout()

heatmap_path = (
    FIGURE_DIR / "correlation_heatmap.png"
)

plt.savefig(
    heatmap_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {heatmap_path}")


# ============================================================
# GENERATE TEXT SUMMARY
# ============================================================

print_header("GENERATING EDA SUMMARY REPORT")

summary_lines = []

summary_lines.append(
    "JM1 EXPLORATORY DATA ANALYSIS SUMMARY"
)

summary_lines.append(
    "=" * 70
)

summary_lines.append(
    f"Dataset rows: {len(df):,}"
)

summary_lines.append(
    f"Dataset columns: {len(df.columns)}"
)

summary_lines.append(
    f"Number of features: {len(feature_columns)}"
)

summary_lines.append(
    f"Numeric features: {len(numeric_features)}"
)

summary_lines.append(
    f"Missing values: {int(df.isna().sum().sum()):,}"
)

summary_lines.append(
    f"Exact duplicate rows: {int(df.duplicated().sum()):,}"
)

summary_lines.append("")

summary_lines.append(
    "TARGET DISTRIBUTION"
)

summary_lines.append(
    "-" * 70
)

for _, row in class_distribution.iterrows():

    summary_lines.append(
        f"Class {row['class']}: "
        f"{int(row['count']):,} "
        f"({row['percentage']:.2f}%)"
    )


if len(class_counts) == 2:

    summary_lines.append("")

    summary_lines.append(
        f"Class imbalance ratio: "
        f"{imbalance_ratio:.4f}"
    )


summary_lines.append("")

summary_lines.append(
    "HIGH CORRELATION PAIRS"
)

summary_lines.append(
    "-" * 70
)

if not high_correlation.empty:

    for _, row in high_correlation.iterrows():

        summary_lines.append(
            f"{row['feature_1']} <-> "
            f"{row['feature_2']}: "
            f"r = {row['pearson_correlation']:.4f}"
        )

else:

    summary_lines.append(
        "No pairs with |Pearson r| >= 0.80."
    )


summary_lines.append("")

summary_lines.append(
    "EDA INTERPRETATION NOTES"
)

summary_lines.append(
    "-" * 70
)

summary_lines.append(
    "1. The cleaned JM1 dataset contains "
    f"{len(df):,} observations and "
    f"{len(feature_columns)} software metrics."
)

summary_lines.append(
    "2. The target distribution should be considered "
    "when designing stratified cross-validation."
)

summary_lines.append(
    "3. SMOTE is intentionally NOT applied during EDA."
)

summary_lines.append(
    "4. Outlier identification is descriptive only; "
    "no observations are removed by this script."
)

summary_lines.append(
    "5. Pearson correlation is exploratory here. "
    "Formal multicollinearity analysis using Pearson "
    "correlation and VIF will be performed in the "
    "dedicated multicollinearity stage."
)

summary_lines.append(
    "6. No model training or SHAP analysis is performed "
    "in this stage."
)

summary_text = "\n".join(summary_lines)

summary_path = save_text(
    "eda_summary.txt",
    summary_text
)

print(summary_text)


# ============================================================
# FINAL OUTPUT
# ============================================================

print_header("EDA COMPLETED")

print("Input:")
print(INPUT_FILE)

print("\nDataset:")
print(f"  Rows:     {len(df):,}")
print(f"  Features: {len(feature_columns)}")
print(f"  Target:   {TARGET}")

print("\nResults saved to:")
print(RESULT_DIR)

print("\nFigures saved to:")
print(FIGURE_DIR)

print("\nOutput files:")

output_files = [
    "eda_dataset_overview.csv",
    "eda_feature_statistics.csv",
    "eda_missing_values.csv",
    "eda_class_distribution.csv",
    "eda_outlier_summary.csv",
    "eda_correlation_matrix.csv",
    "eda_high_correlations.csv",
    "eda_feature_target_correlation.csv",
    "eda_summary.txt"
]

for filename in output_files:

    path = RESULT_DIR / filename

    if path.exists():
        print(f"  ✓ {filename}")

print("\nEDA figures:")

print("  ✓ class_distribution.png")
print("  ✓ correlation_heatmap.png")
print(
    f"  ✓ {len(numeric_features)} feature distribution figures"
)
print(
    f"  ✓ {len(numeric_features)} feature boxplot figures"
)

print("\n" + "=" * 80)
print("READY FOR CROSS-VALIDATION / BASELINE MODELING")
print("=" * 80)
