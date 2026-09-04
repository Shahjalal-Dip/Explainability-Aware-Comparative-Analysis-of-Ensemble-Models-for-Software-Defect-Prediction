"""
06_SMOTE_VS_NO_SMOTE_COMPARISON.PY

Purpose
-------
Compare baseline model performance with and without SMOTE
using the JM1 development set only.

Experimental design
-------------------
- Dataset: JM1 Development Set
- Development set: 80% of original dataset
- Test set: 20% and completely untouched
- Validation: Stratified 5-Fold Cross-Validation
- No-SMOTE results: loaded from Step 04b
- SMOTE results: loaded from Step 05
- Statistical unit: identical CV folds
- Random state: 42

Models
------
1. Random Forest
2. XGBoost
3. LightGBM
4. MLP

Metrics
-------
1. Accuracy
2. Precision
3. Recall
4. Specificity
5. F1
6. MCC
7. ROC-AUC
8. PR-AUC

Outputs
-------
data/results/performance/smote_vs_no_smote/
    smote_vs_no_smote_comparison.csv
    smote_vs_no_smote_fold_comparison.csv
    smote_vs_no_smote_summary.csv
    smote_vs_no_smote_report.txt

figures/performance/smote_vs_no_smote/
    smote_vs_no_smote_model_performance.png
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# =============================================================================
# INPUT DIRECTORIES
# =============================================================================

NO_SMOTE_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "performance"
    / "development_baseline_no_smote"
)

SMOTE_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "performance"
    / "development_smote"
)


# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "performance"
    / "smote_vs_no_smote"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "performance"
    / "smote_vs_no_smote"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FILE PATHS
# =============================================================================

NO_SMOTE_FOLD_FILE = (
    NO_SMOTE_DIR
    / "development_baseline_no_smote_fold_results.csv"
)

NO_SMOTE_SUMMARY_FILE = (
    NO_SMOTE_DIR
    / "development_baseline_no_smote_summary.csv"
)

SMOTE_FOLD_FILE = (
    SMOTE_DIR
    / "development_smote_fold_results.csv"
)

SMOTE_SUMMARY_FILE = (
    SMOTE_DIR
    / "development_smote_summary.csv"
)


# =============================================================================
# OUTPUT FILES
# =============================================================================

COMPARISON_FILE = (
    RESULTS_DIR
    / "smote_vs_no_smote_comparison.csv"
)

FOLD_COMPARISON_FILE = (
    RESULTS_DIR
    / "smote_vs_no_smote_fold_comparison.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "smote_vs_no_smote_summary.csv"
)

REPORT_FILE = (
    RESULTS_DIR
    / "smote_vs_no_smote_report.txt"
)

FIGURE_FILE = (
    FIGURES_DIR
    / "smote_vs_no_smote_model_performance.png"
)


# =============================================================================
# METRICS
# =============================================================================

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "mcc",
    "roc_auc",
    "pr_auc",
]


METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "f1": "F1",
    "mcc": "MCC",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}


MODELS = [
    "RF",
    "XGB",
    "LGBM",
    "MLP",
]


MODEL_LABELS = {
    "RF": "Random Forest",
    "XGB": "XGBoost",
    "LGBM": "LightGBM",
    "MLP": "MLP",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_column(df, possible_names):
    """
    Find the first matching column from a list of possible names.
    Matching is case-insensitive.
    """

    lower_map = {
        str(col).lower(): col
        for col in df.columns
    }

    for name in possible_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    return None


def standardize_fold_results(df, method_name):
    """
    Standardize fold-result column names so that the no-SMOTE and
    SMOTE result files can be compared reliably.
    """

    df = df.copy()

    # -------------------------------------------------------------------------
    # Find model column
    # -------------------------------------------------------------------------

    model_col = find_column(
        df,
        [
            "model",
            "Model",
        ],
    )

    if model_col is None:
        raise ValueError(
            f"Could not find model column in {method_name} results.\n"
            f"Available columns: {list(df.columns)}"
        )

    # -------------------------------------------------------------------------
    # Find fold column
    # -------------------------------------------------------------------------

    fold_col = find_column(
        df,
        [
            "fold",
            "Fold",
        ],
    )

    if fold_col is None:
        raise ValueError(
            f"Could not find fold column in {method_name} results.\n"
            f"Available columns: {list(df.columns)}"
        )

    # -------------------------------------------------------------------------
    # Rename core columns
    # -------------------------------------------------------------------------

    rename_dict = {
        model_col: "model",
        fold_col: "fold",
    }

    for metric in METRICS:

        possible = [
            metric,
            f"{metric}_score",
            f"{metric}_mean",
        ]

        column = find_column(df, possible)

        if column is not None:
            rename_dict[column] = metric

    df = df.rename(columns=rename_dict)

    # -------------------------------------------------------------------------
    # Validate metric columns
    # -------------------------------------------------------------------------

    missing_metrics = [
        metric
        for metric in METRICS
        if metric not in df.columns
    ]

    if missing_metrics:
        raise ValueError(
            f"Missing metrics in {method_name} fold results: "
            f"{missing_metrics}\n"
            f"Available columns: {list(df.columns)}"
        )

    # -------------------------------------------------------------------------
    # Keep only required columns
    # -------------------------------------------------------------------------

    df = df[
        [
            "model",
            "fold",
            *METRICS,
        ]
    ].copy()

    # -------------------------------------------------------------------------
    # Standardize model names
    # -------------------------------------------------------------------------

    df["model"] = (
        df["model"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # -------------------------------------------------------------------------
    # Convert fold to integer
    # -------------------------------------------------------------------------

    df["fold"] = pd.to_numeric(
        df["fold"],
        errors="coerce"
    )

    if df["fold"].isna().any():
        raise ValueError(
            f"Invalid fold values found in {method_name} results."
        )

    df["fold"] = df["fold"].astype(int)

    # -------------------------------------------------------------------------
    # Convert metrics to numeric
    # -------------------------------------------------------------------------

    for metric in METRICS:

        df[metric] = pd.to_numeric(
            df[metric],
            errors="coerce"
        )

        if df[metric].isna().any():
            raise ValueError(
                f"Missing/non-numeric values found in "
                f"{method_name} metric: {metric}"
            )

    return df


def calculate_percentage_change(smote_value, no_smote_value):
    """
    Calculate percentage change from no-SMOTE to SMOTE.
    """

    if no_smote_value == 0:
        return np.nan

    return (
        (smote_value - no_smote_value)
        / abs(no_smote_value)
    ) * 100.0


# =============================================================================
# HEADER
# =============================================================================

print("=" * 90)
print("JM1 DEVELOPMENT-SET SMOTE VS NO-SMOTE COMPARISON")
print("=" * 90)

print()
print("Experimental design:")
print("Dataset           : JM1 Development Set")
print("Development size  : 80% of original dataset")
print("Test set          : 20% - COMPLETELY UNTOUCHED")
print("Validation        : Stratified 5-Fold CV")
print("Comparison        : SMOTE vs No-SMOTE")
print("SMOTE             : Training folds ONLY")
print("Hyperparameter    : Baseline configuration")
print("Random state      :", RANDOM_STATE)

print()
print("IMPORTANT:")
print("This comparison uses development-set cross-validation results only.")
print("The locked 20% test set is NOT accessed.")


# =============================================================================
# CHECK INPUT FILES
# =============================================================================

print()
print("=" * 90)
print("CHECKING INPUT RESULTS")
print("=" * 90)

print()
print("No-SMOTE fold results:")
print(NO_SMOTE_FOLD_FILE)

print()
print("SMOTE fold results:")
print(SMOTE_FOLD_FILE)


if not NO_SMOTE_FOLD_FILE.exists():
    raise FileNotFoundError(
        "\nNo-SMOTE fold-results file was not found:\n"
        f"{NO_SMOTE_FOLD_FILE}\n\n"
        "Run Step 04b before running Step 06."
    )


if not SMOTE_FOLD_FILE.exists():
    raise FileNotFoundError(
        "\nSMOTE fold-results file was not found:\n"
        f"{SMOTE_FOLD_FILE}\n\n"
        "Run Step 05 before running Step 06."
    )


# =============================================================================
# LOAD FOLD RESULTS
# =============================================================================

print()
print("=" * 90)
print("LOADING NO-SMOTE RESULTS")
print("=" * 90)

no_smote_raw = pd.read_csv(NO_SMOTE_FOLD_FILE)

print()
print("No-SMOTE raw shape:", no_smote_raw.shape)
print("No-SMOTE columns:")
print(list(no_smote_raw.columns))


print()
print("=" * 90)
print("LOADING SMOTE RESULTS")
print("=" * 90)

smote_raw = pd.read_csv(SMOTE_FOLD_FILE)

print()
print("SMOTE raw shape:", smote_raw.shape)
print("SMOTE columns:")
print(list(smote_raw.columns))


# =============================================================================
# STANDARDIZE RESULTS
# =============================================================================

no_smote = standardize_fold_results(
    no_smote_raw,
    "No-SMOTE"
)

smote = standardize_fold_results(
    smote_raw,
    "SMOTE"
)


# =============================================================================
# VALIDATE MODEL/FOLD STRUCTURE
# =============================================================================

print()
print("=" * 90)
print("VALIDATING CV STRUCTURE")
print("=" * 90)

print()
print("No-SMOTE models:")
print(sorted(no_smote["model"].unique()))

print()
print("SMOTE models:")
print(sorted(smote["model"].unique()))

print()
print("No-SMOTE folds:")
print(sorted(no_smote["fold"].unique()))

print()
print("SMOTE folds:")
print(sorted(smote["fold"].unique()))


expected_models = set(MODELS)

no_smote_models = set(no_smote["model"].unique())
smote_models = set(smote["model"].unique())

if no_smote_models != expected_models:
    raise ValueError(
        "No-SMOTE model set does not match expected models.\n"
        f"Expected: {expected_models}\n"
        f"Found: {no_smote_models}"
    )


if smote_models != expected_models:
    raise ValueError(
        "SMOTE model set does not match expected models.\n"
        f"Expected: {expected_models}\n"
        f"Found: {smote_models}"
    )


no_smote_folds = set(no_smote["fold"].unique())
smote_folds = set(smote["fold"].unique())

if no_smote_folds != smote_folds:
    raise ValueError(
        "No-SMOTE and SMOTE do not contain the same CV folds.\n"
        f"No-SMOTE: {no_smote_folds}\n"
        f"SMOTE: {smote_folds}"
    )


# =============================================================================
# CHECK DUPLICATES
# =============================================================================

no_smote_duplicates = no_smote.duplicated(
    subset=["model", "fold"]
).sum()

smote_duplicates = smote.duplicated(
    subset=["model", "fold"]
).sum()


if no_smote_duplicates > 0:
    raise ValueError(
        "Duplicate model/fold combinations found in No-SMOTE results."
    )


if smote_duplicates > 0:
    raise ValueError(
        "Duplicate model/fold combinations found in SMOTE results."
    )


# =============================================================================
# MERGE FOLD RESULTS
# =============================================================================

print()
print("=" * 90)
print("CREATING PAIRED FOLD COMPARISON")
print("=" * 90)

no_smote_for_merge = no_smote.copy()
smote_for_merge = smote.copy()

no_smote_for_merge = no_smote_for_merge.rename(
    columns={
        metric: f"{metric}_no_smote"
        for metric in METRICS
    }
)

smote_for_merge = smote_for_merge.rename(
    columns={
        metric: f"{metric}_smote"
        for metric in METRICS
    }
)

fold_comparison = pd.merge(
    no_smote_for_merge,
    smote_for_merge,
    on=["model", "fold"],
    how="inner",
    validate="one_to_one",
)


expected_rows = len(MODELS) * len(no_smote_folds)

if len(fold_comparison) != expected_rows:
    raise ValueError(
        "Unexpected number of paired fold results.\n"
        f"Expected: {expected_rows}\n"
        f"Found: {len(fold_comparison)}"
    )


# =============================================================================
# CALCULATE FOLD-LEVEL DIFFERENCES
# =============================================================================

for metric in METRICS:

    fold_comparison[
        f"{metric}_difference"
    ] = (
        fold_comparison[f"{metric}_smote"]
        - fold_comparison[f"{metric}_no_smote"]
    )


# =============================================================================
# SAVE FOLD COMPARISON
# =============================================================================

fold_comparison.to_csv(
    FOLD_COMPARISON_FILE,
    index=False
)

print()
print("Saved paired fold comparison:")
print(FOLD_COMPARISON_FILE)


# =============================================================================
# CREATE MODEL-LEVEL SUMMARY
# =============================================================================

print()
print("=" * 90)
print("CALCULATING SMOTE VS NO-SMOTE SUMMARY")
print("=" * 90)


summary_rows = []


for model in MODELS:

    model_data = fold_comparison[
        fold_comparison["model"] == model
    ].copy()

    for metric in METRICS:

        no_smote_values = model_data[
            f"{metric}_no_smote"
        ].values

        smote_values = model_data[
            f"{metric}_smote"
        ].values

        no_smote_mean = np.mean(no_smote_values)
        no_smote_std = np.std(
            no_smote_values,
            ddof=1
        )

        smote_mean = np.mean(smote_values)
        smote_std = np.std(
            smote_values,
            ddof=1
        )

        difference = (
            smote_mean
            - no_smote_mean
        )

        percentage_change = calculate_percentage_change(
            smote_mean,
            no_smote_mean
        )

        if difference > 0:
            better_method = "SMOTE"
        elif difference < 0:
            better_method = "No-SMOTE"
        else:
            better_method = "Equal"

        summary_rows.append(
            {
                "model": model,
                "metric": metric,
                "no_smote_mean": no_smote_mean,
                "no_smote_std": no_smote_std,
                "smote_mean": smote_mean,
                "smote_std": smote_std,
                "difference_smote_minus_no_smote": difference,
                "percentage_change": percentage_change,
                "better_method": better_method,
            }
        )


summary = pd.DataFrame(summary_rows)


# =============================================================================
# SAVE LONG-FORM COMPARISON
# =============================================================================

summary.to_csv(
    COMPARISON_FILE,
    index=False
)

print()
print("Saved detailed comparison:")
print(COMPARISON_FILE)


# =============================================================================
# CREATE WIDE SUMMARY
# =============================================================================

wide_summary_rows = []


for model in MODELS:

    row = {
        "model": model
    }

    model_summary = summary[
        summary["model"] == model
    ]

    for metric in METRICS:

        metric_row = model_summary[
            model_summary["metric"] == metric
        ].iloc[0]

        row[f"{metric}_no_smote"] = (
            metric_row["no_smote_mean"]
        )

        row[f"{metric}_no_smote_std"] = (
            metric_row["no_smote_std"]
        )

        row[f"{metric}_smote"] = (
            metric_row["smote_mean"]
        )

        row[f"{metric}_smote_std"] = (
            metric_row["smote_std"]
        )

        row[f"{metric}_difference"] = (
            metric_row[
                "difference_smote_minus_no_smote"
            ]
        )

        row[f"{metric}_percentage_change"] = (
            metric_row["percentage_change"]
        )

        row[f"{metric}_better_method"] = (
            metric_row["better_method"]
        )

    wide_summary_rows.append(row)


wide_summary = pd.DataFrame(
    wide_summary_rows
)


# =============================================================================
# SAVE WIDE SUMMARY
# =============================================================================

wide_summary.to_csv(
    SUMMARY_FILE,
    index=False
)

print()
print("Saved model-level summary:")
print(SUMMARY_FILE)


# =============================================================================
# PRINT MAIN COMPARISON TABLE
# =============================================================================

print()
print("=" * 90)
print("SMOTE VS NO-SMOTE PERFORMANCE COMPARISON")
print("=" * 90)


display_table = summary.copy()

display_table["model"] = display_table["model"].map(
    MODEL_LABELS
)

display_table["metric"] = display_table["metric"].map(
    METRIC_LABELS
)

display_table["No-SMOTE"] = display_table.apply(
    lambda row:
        f"{row['no_smote_mean']:.4f} ± {row['no_smote_std']:.4f}",
    axis=1
)

display_table["SMOTE"] = display_table.apply(
    lambda row:
        f"{row['smote_mean']:.4f} ± {row['smote_std']:.4f}",
    axis=1
)

display_table["Δ"] = display_table[
    "difference_smote_minus_no_smote"
].map(
    lambda x: f"{x:+.4f}"
)

display_table["Change (%)"] = display_table[
    "percentage_change"
].map(
    lambda x: f"{x:+.2f}%"
)

display_table["Better"] = display_table[
    "better_method"
]

display_table = display_table[
    [
        "model",
        "metric",
        "No-SMOTE",
        "SMOTE",
        "Δ",
        "Change (%)",
        "Better",
    ]
]

print()
print(display_table.to_string(index=False))


# =============================================================================
# CREATE PERFORMANCE FIGURE
# =============================================================================

print()
print("=" * 90)
print("GENERATING SMOTE VS NO-SMOTE PERFORMANCE FIGURE")
print("=" * 90)


# -------------------------------------------------------------------------
# Figure design
# -------------------------------------------------------------------------
#
# The figure shows the eight performance metrics.
#
# For each metric:
#   - four No-SMOTE bars
#   - four SMOTE bars
#
# Models are grouped within each metric.
# -------------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(16, 8)
)


x = np.arange(len(METRICS))

width = 0.10

offsets = [
    -1.5 * width,
    -0.5 * width,
    0.5 * width,
    1.5 * width,
]


# -------------------------------------------------------------------------
# No-SMOTE bars
# -------------------------------------------------------------------------

for i, model in enumerate(MODELS):

    values = []

    for metric in METRICS:

        row = summary[
            (summary["model"] == model)
            & (summary["metric"] == metric)
        ].iloc[0]

        values.append(
            row["no_smote_mean"]
        )

    ax.bar(
        x + offsets[i],
        values,
        width,
        label=f"{MODEL_LABELS[model]} - No-SMOTE",
        alpha=0.65,
    )


# -------------------------------------------------------------------------
# SMOTE bars
# -------------------------------------------------------------------------

smote_offsets = [
    offset + 4.0 * width
    for offset in offsets
]

for i, model in enumerate(MODELS):

    values = []

    for metric in METRICS:

        row = summary[
            (summary["model"] == model)
            & (summary["metric"] == metric)
        ].iloc[0]

        values.append(
            row["smote_mean"]
        )

    ax.bar(
        x + smote_offsets[i],
        values,
        width,
        label=f"{MODEL_LABELS[model]} - SMOTE",
        alpha=0.65,
    )


# -------------------------------------------------------------------------
# Axis formatting
# -------------------------------------------------------------------------

ax.set_title(
    "JM1 Development-Set Model Performance: SMOTE vs No-SMOTE",
    fontsize=14,
    fontweight="bold",
)

ax.set_xlabel(
    "Performance Metric",
    fontsize=11,
    fontweight="bold",
)

ax.set_ylabel(
    "Mean CV Score",
    fontsize=11,
    fontweight="bold",
)

ax.set_xticks(
    x + 2.0 * width
)

ax.set_xticklabels(
    [
        METRIC_LABELS[metric]
        for metric in METRICS
    ]
)

ax.set_ylim(
    0,
    1.05
)

ax.grid(
    axis="y",
    alpha=0.25
)

ax.legend(
    fontsize=8,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
)

plt.tight_layout()


# -------------------------------------------------------------------------
# Save figure
# -------------------------------------------------------------------------

plt.savefig(
    FIGURE_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print()
print("Saved:")
print(FIGURE_FILE)


# =============================================================================
# CREATE TEXT REPORT
# =============================================================================

print()
print("=" * 90)
print("GENERATING TEXT REPORT")
print("=" * 90)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "JM1 DEVELOPMENT-SET SMOTE VS NO-SMOTE COMPARISON\n"
    )

    report.write(
        "=" * 90 + "\n\n"
    )

    report.write(
        "Experimental design\n"
    )

    report.write(
        "--------------------\n"
    )

    report.write(
        "Dataset: JM1 Development Set\n"
    )

    report.write(
        "Development set: 80% of original dataset\n"
    )

    report.write(
        "Test set: 20% - COMPLETELY UNTOUCHED\n"
    )

    report.write(
        "Validation: Stratified 5-Fold Cross-Validation\n"
    )

    report.write(
        "Comparison: SMOTE vs No-SMOTE\n"
    )

    report.write(
        "SMOTE applied: Training folds only\n"
    )

    report.write(
        "Random state: 42\n\n"
    )

    report.write(
        "Models\n"
    )

    report.write(
        "------\n"
    )

    for model in MODELS:

        report.write(
            f"- {MODEL_LABELS[model]}\n"
        )

    report.write("\n")

    report.write(
        "Performance comparison\n"
    )

    report.write(
        "-----------------------\n\n"
    )

    for model in MODELS:

        report.write(
            f"{MODEL_LABELS[model]}\n"
        )

        report.write(
            "-" * 40 + "\n"
        )

        model_summary = summary[
            summary["model"] == model
        ]

        for metric in METRICS:

            row = model_summary[
                model_summary["metric"] == metric
            ].iloc[0]

            report.write(
                f"{METRIC_LABELS[metric]}:\n"
            )

            report.write(
                f"  No-SMOTE : "
                f"{row['no_smote_mean']:.4f} "
                f"± {row['no_smote_std']:.4f}\n"
            )

            report.write(
                f"  SMOTE    : "
                f"{row['smote_mean']:.4f} "
                f"± {row['smote_std']:.4f}\n"
            )

            report.write(
                f"  Difference: "
                f"{row['difference_smote_minus_no_smote']:+.4f}\n"
            )

            report.write(
                f"  Change   : "
                f"{row['percentage_change']:+.2f}%\n"
            )

            report.write(
                f"  Better   : "
                f"{row['better_method']}\n\n"
            )

        report.write("\n")

    report.write(
        "Interpretation note\n"
    )

    report.write(
        "-------------------\n"
    )

    report.write(
        "Positive differences indicate that SMOTE produced a higher "
        "mean cross-validation score than No-SMOTE.\n"
    )

    report.write(
        "Negative differences indicate that No-SMOTE produced a higher "
        "mean cross-validation score.\n"
    )

    report.write(
        "This comparison is based exclusively on development-set "
        "cross-validation results. The locked test set was not used.\n"
    )


print()
print("Saved:")
print(REPORT_FILE)


# =============================================================================
# OVERALL INTERPRETATION
# =============================================================================

print()
print("=" * 90)
print("KEY SMOTE EFFECTS")
print("=" * 90)


for model in MODELS:

    model_summary = summary[
        summary["model"] == model
    ]

    print()
    print(f"{MODEL_LABELS[model]}:")

    for metric in METRICS:

        row = model_summary[
            model_summary["metric"] == metric
        ].iloc[0]

        print(
            f"  {METRIC_LABELS[metric]:12s} "
            f"No-SMOTE={row['no_smote_mean']:.4f} | "
            f"SMOTE={row['smote_mean']:.4f} | "
            f"Δ={row['difference_smote_minus_no_smote']:+.4f}"
        )


# =============================================================================
# COMPLETION
# =============================================================================

print()
print("=" * 90)
print("SMOTE VS NO-SMOTE COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 90)

print()
print("Models:")
print("  ✓ Random Forest")
print("  ✓ XGBoost")
print("  ✓ LightGBM")
print("  ✓ MLP")

print()
print("Comparison:")
print("  ✓ No-SMOTE development CV")
print("  ✓ SMOTE development CV")
print("  ✓ Same Stratified 5-Fold structure")
print("  ✓ Fold-level comparison")
print("  ✓ Mean and standard deviation")
print("  ✓ SMOTE minus No-SMOTE difference")
print("  ✓ Percentage change")

print()
print("Test set:")
print("  ✓ NOT USED")
print("  ✓ Remains untouched")

print()
print("Metrics:")
print("  ✓ Accuracy")
print("  ✓ Precision")
print("  ✓ Recall")
print("  ✓ Specificity")
print("  ✓ F1")
print("  ✓ MCC")
print("  ✓ ROC-AUC")
print("  ✓ PR-AUC")

print()
print("Generated:")
print("  ✓ Fold-level comparison CSV")
print("  ✓ Detailed comparison CSV")
print("  ✓ Model-level summary CSV")
print("  ✓ Text report")
print("  ✓ SMOTE vs No-SMOTE performance figure")

print()
print("Results directory:")
print(RESULTS_DIR)

print()
print("Figures directory:")
print(FIGURES_DIR)

print()
print("=" * 90)
print("NEXT STEP: 07_FINAL_MODEL_SELECTION.PY")
print("=" * 90)
