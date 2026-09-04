"""
04b_baseline_models_development.py

JM1 Software Defect Prediction
Development-Set Baseline Evaluation WITHOUT SMOTE

Purpose:
    Establish the primary no-SMOTE baseline using only the
    80% development set.

Dataset:
    JM1

Development set:
    80% of cleaned JM1 dataset

Test set:
    20% of cleaned JM1 dataset
    NOT USED in this script.

Models:
    1. Random Forest
    2. XGBoost
    3. LightGBM
    4. MLP

Validation:
    Stratified 5-Fold Cross-Validation

Important:
    - No SMOTE
    - No hyperparameter tuning
    - No test-set access
    - No feature selection
    - No model selection using test data

Metrics:
    Accuracy
    Precision
    Recall
    Specificity
    F1
    MCC
    ROC-AUC
    PR-AUC

Outputs:
    data/results/performance/development_baseline_no_smote/
    figures/performance/development_baseline_no_smote/

Figures:
    - Comprehensive baseline model performance
    - Accuracy comparison
    - F1 comparison
    - MCC comparison
    - ROC-AUC comparison
    - PR-AUC comparison
    - Confusion matrices
"""


from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "JM1"
    / "JM1_development.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "performance"
    / "development_baseline_no_smote"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "performance"
    / "development_baseline_no_smote"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

TARGET = "defects"

RANDOM_STATE = 42

N_SPLITS = 5

MODEL_ORDER = [
    "RF",
    "XGB",
    "LGBM",
    "MLP",
]


# ============================================================
# HEADER
# ============================================================

print("=" * 90)
print("JM1 DEVELOPMENT-SET BASELINE MODEL EVALUATION")
print("WITHOUT SMOTE")
print("=" * 90)

print("\nInput dataset:")
print(INPUT_FILE)

print("\nExperimental design:")
print("Dataset           : JM1 Development Set")
print("Development size  : 80% of original dataset")
print("Test set          : 20% - COMPLETELY UNTOUCHED")
print("Validation        : Stratified 5-Fold CV")
print("SMOTE             : NOT APPLIED")
print("Hyperparameter    : Baseline configuration")
print(f"Random state      : {RANDOM_STATE}")


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nDevelopment dataset was not found:\n{INPUT_FILE}\n\n"
        "Run 04a_train_test_split.py first."
    )


# ============================================================
# LOAD DEVELOPMENT DATASET
# ============================================================

print("\n" + "=" * 90)
print("LOADING DEVELOPMENT DATASET")
print("=" * 90)

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
# FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[TARGET]
)

y = (
    df[TARGET]
    .astype(bool)
    .astype(int)
)


print(
    f"\nTarget column: {TARGET}"
)

print(
    f"Number of features: {X.shape[1]}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 90)
print("DEVELOPMENT-SET CLASS DISTRIBUTION")
print("=" * 90)

class_counts = (
    y.value_counts()
    .sort_index()
)

for class_value in [0, 1]:

    count = int(
        class_counts.get(
            class_value,
            0
        )
    )

    percentage = (
        count / len(y)
    ) * 100

    class_name = (
        "Non-defective"
        if class_value == 0
        else "Defective"
    )

    print(
        f"{class_name:15s}: "
        f"{count:,} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {

    "RF": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=None,
    ),

    "XGB": XGBClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=0,
    ),

    "LGBM": LGBMClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    ),

    "MLP": Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(100,),
                    max_iter=500,
                    random_state=RANDOM_STATE,
                )
            ),
        ]
    ),
}


# ============================================================
# STRATIFIED 5-FOLD CROSS-VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# STORAGE
# ============================================================

all_results = []

all_predictions = []

all_confusion = []


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 90)
print("STARTING STRATIFIED 5-FOLD CROSS-VALIDATION")
print("=" * 90)


for model_name, model in models.items():

    print("\n" + "-" * 90)
    print(f"MODEL: {model_name}")
    print("-" * 90)

    for fold, (train_idx, validation_idx) in enumerate(
        skf.split(X, y),
        start=1
    ):

        # ----------------------------------------------------
        # TRAINING / VALIDATION SPLIT
        # ----------------------------------------------------

        X_train = X.iloc[
            train_idx
        ]

        X_validation = X.iloc[
            validation_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        y_validation = y.iloc[
            validation_idx
        ]

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # PREDICT CLASS
        # ----------------------------------------------------

        y_pred = model.predict(
            X_validation
        )

        # ----------------------------------------------------
        # PREDICT PROBABILITY
        # ----------------------------------------------------

        if hasattr(
            model,
            "predict_proba"
        ):

            y_probability = (
                model
                .predict_proba(
                    X_validation
                )[:, 1]
            )

        else:

            y_probability = (
                model
                .predict(
                    X_validation
                )
            )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_validation,
            y_pred
        )

        precision = precision_score(
            y_validation,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_validation,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_validation,
            y_pred,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_validation,
            y_pred
        )

        roc_auc = roc_auc_score(
            y_validation,
            y_probability
        )

        pr_auc = average_precision_score(
            y_validation,
            y_probability
        )

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_validation,
            y_pred,
            labels=[0, 1]
        )

        tn, fp, fn, tp = (
            cm.ravel()
        )

        # ----------------------------------------------------
        # SPECIFICITY
        # ----------------------------------------------------

        if (
            tn + fp
        ) > 0:

            specificity = (
                tn / (tn + fp)
            )

        else:

            specificity = 0.0

        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        all_results.append(
            {
                "model": model_name,
                "fold": fold,

                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "mcc": mcc,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,

                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            }
        )

        # ----------------------------------------------------
        # STORE PREDICTIONS
        # ----------------------------------------------------

        prediction_df = pd.DataFrame(
            {
                "model": model_name,
                "fold": fold,

                "true_label":
                    y_validation.values,

                "predicted_label":
                    y_pred,

                "predicted_probability":
                    y_probability,
            }
        )

        all_predictions.append(
            prediction_df
        )

        # ----------------------------------------------------
        # STORE CONFUSION MATRIX
        # ----------------------------------------------------

        all_confusion.append(
            {
                "model": model_name,
                "fold": fold,

                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            }
        )

        # ----------------------------------------------------
        # PRINT FOLD RESULTS
        # ----------------------------------------------------

        print(
            f"Fold {fold}: "
            f"Accuracy={accuracy:.4f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f} | "
            f"MCC={mcc:.4f} | "
            f"ROC-AUC={roc_auc:.4f} | "
            f"PR-AUC={pr_auc:.4f}"
        )


# ============================================================
# DATAFRAMES
# ============================================================

results_df = pd.DataFrame(
    all_results
)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

confusion_df = pd.DataFrame(
    all_confusion
)


# ============================================================
# SAVE FOLD RESULTS
# ============================================================

fold_results_file = (
    RESULTS_DIR
    / "development_baseline_no_smote_fold_results.csv"
)

results_df.to_csv(
    fold_results_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary_df = (
    results_df
    .groupby("model")
    .agg(

        accuracy_mean=(
            "accuracy",
            "mean"
        ),

        accuracy_std=(
            "accuracy",
            "std"
        ),

        precision_mean=(
            "precision",
            "mean"
        ),

        precision_std=(
            "precision",
            "std"
        ),

        recall_mean=(
            "recall",
            "mean"
        ),

        recall_std=(
            "recall",
            "std"
        ),

        specificity_mean=(
            "specificity",
            "mean"
        ),

        specificity_std=(
            "specificity",
            "std"
        ),

        f1_mean=(
            "f1",
            "mean"
        ),

        f1_std=(
            "f1",
            "std"
        ),

        mcc_mean=(
            "mcc",
            "mean"
        ),

        mcc_std=(
            "mcc",
            "std"
        ),

        roc_auc_mean=(
            "roc_auc",
            "mean"
        ),

        roc_auc_std=(
            "roc_auc",
            "std"
        ),

        pr_auc_mean=(
            "pr_auc",
            "mean"
        ),

        pr_auc_std=(
            "pr_auc",
            "std"
        ),
    )
    .reset_index()
)


# ============================================================
# ORDER SUMMARY BY MODEL
# ============================================================

summary_df["model"] = pd.Categorical(
    summary_df["model"],
    categories=MODEL_ORDER,
    ordered=True
)

summary_df = (
    summary_df
    .sort_values("model")
    .reset_index(drop=True)
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = (
    RESULTS_DIR
    / "development_baseline_no_smote_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_file = (
    RESULTS_DIR
    / "development_baseline_no_smote_predictions.csv"
)

predictions_df.to_csv(
    predictions_file,
    index=False
)


# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================

confusion_file = (
    RESULTS_DIR
    / "development_baseline_no_smote_confusion_matrices.csv"
)

confusion_df.to_csv(
    confusion_file,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("DEVELOPMENT-SET BASELINE PERFORMANCE")
print("=" * 90)

display_columns = [
    "model",

    "accuracy_mean",
    "accuracy_std",

    "precision_mean",
    "precision_std",

    "recall_mean",
    "recall_std",

    "specificity_mean",
    "specificity_std",

    "f1_mean",
    "f1_std",

    "mcc_mean",
    "mcc_std",

    "roc_auc_mean",
    "roc_auc_std",

    "pr_auc_mean",
    "pr_auc_std",
]

print(
    summary_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# COMPREHENSIVE BASELINE PERFORMANCE FIGURE
# ============================================================

print("\n" + "=" * 90)
print("GENERATING BASELINE MODEL PERFORMANCE FIGURE")
print("=" * 90)


performance_metrics = [
    "accuracy_mean",
    "precision_mean",
    "recall_mean",
    "specificity_mean",
    "f1_mean",
    "mcc_mean",
    "roc_auc_mean",
    "pr_auc_mean",
]

metric_labels = [
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "MCC",
    "ROC-AUC",
    "PR-AUC",
]


# ------------------------------------------------------------
# Extract values
# ------------------------------------------------------------

performance_matrix = []

for model_name in MODEL_ORDER:

    row = summary_df[
        summary_df["model"]
        == model_name
    ]

    if row.empty:
        raise ValueError(
            f"Model '{model_name}' was not found "
            "in the summary results."
        )

    values = [
        row.iloc[0][metric]
        for metric in performance_metrics
    ]

    performance_matrix.append(
        values
    )


performance_matrix = np.array(
    performance_matrix
)


# ------------------------------------------------------------
# Figure configuration
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(16, 9)
)

x = np.arange(
    len(metric_labels)
)

bar_width = 0.20


# ------------------------------------------------------------
# Plot each model
# ------------------------------------------------------------

bars_rf = ax.bar(
    x - 1.5 * bar_width,
    performance_matrix[0],
    bar_width,
    label="RF",
    color="#4C78A8"
)

bars_xgb = ax.bar(
    x - 0.5 * bar_width,
    performance_matrix[1],
    bar_width,
    label="XGB",
    color="#F58518"
)

bars_lgbm = ax.bar(
    x + 0.5 * bar_width,
    performance_matrix[2],
    bar_width,
    label="LGBM",
    color="#54A24B"
)

bars_mlp = ax.bar(
    x + 1.5 * bar_width,
    performance_matrix[3],
    bar_width,
    label="MLP",
    color="#E45756"
)


# ------------------------------------------------------------
# Add numerical values to bars
# ------------------------------------------------------------

all_bars = [
    bars_rf,
    bars_xgb,
    bars_lgbm,
    bars_mlp,
]

for bars in all_bars:

    for bar in bars:

        height = bar.get_height()

        ax.annotate(
            f"{height:.3f}",
            xy=(
                bar.get_x()
                + bar.get_width() / 2,
                height
            ),
            xytext=(
                0,
                4
            ),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            rotation=90,
        )


# ------------------------------------------------------------
# Axis labels
# ------------------------------------------------------------

ax.set_xlabel(
    "Performance Metric",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel(
    "Score",
    fontsize=14,
    fontweight="bold"
)

ax.set_title(
    "JM1 Development-Set Baseline Model Performance Without SMOTE",
    fontsize=18,
    fontweight="bold",
    pad=20
)


# ------------------------------------------------------------
# X-axis
# ------------------------------------------------------------

ax.set_xticks(
    x
)

ax.set_xticklabels(
    metric_labels,
    fontsize=11
)


# ------------------------------------------------------------
# Y-axis
# ------------------------------------------------------------

ax.set_ylim(
    0,
    1.08
)

ax.tick_params(
    axis="y",
    labelsize=11
)


# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.30
)

ax.set_axisbelow(
    True
)


# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

ax.legend(
    title="Model",
    fontsize=11,
    title_fontsize=12,
    loc="upper right",
    frameon=True
)


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

plt.tight_layout()


# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------

baseline_performance_figure = (
    FIGURES_DIR
    / "development_baseline_model_performance_no_smote.png"
)

plt.savefig(
    baseline_performance_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Saved: {baseline_performance_figure}"
)


# ============================================================
# INDIVIDUAL ACCURACY FIGURE
# ============================================================

print("\nGenerating accuracy comparison...")

accuracy_values = []

accuracy_errors = []

for model_name in MODEL_ORDER:

    row = summary_df[
        summary_df["model"]
        == model_name
    ]

    accuracy_values.append(
        row.iloc[0]["accuracy_mean"]
    )

    accuracy_errors.append(
        row.iloc[0]["accuracy_std"]
    )


fig, ax = plt.subplots(
    figsize=(9, 6)
)

bars = ax.bar(
    MODEL_ORDER,
    accuracy_values,
    yerr=accuracy_errors,
    capsize=5,
    color=[
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
    ]
)

ax.set_xlabel(
    "Model",
    fontsize=12
)

ax.set_ylabel(
    "Accuracy",
    fontsize=12
)

ax.set_title(
    "Development-Set Accuracy Without SMOTE",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylim(
    0,
    1.05
)

for bar, value in zip(
    bars,
    accuracy_values
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,

        bar.get_height(),

        f"{value:.4f}",

        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )


ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

accuracy_figure = (
    FIGURES_DIR
    / "development_accuracy_no_smote.png"
)

plt.savefig(
    accuracy_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {accuracy_figure}"
)


# ============================================================
# F1 FIGURE
# ============================================================

print("Generating F1 comparison...")

f1_values = []

f1_errors = []

for model_name in MODEL_ORDER:

    row = summary_df[
        summary_df["model"]
        == model_name
    ]

    f1_values.append(
        row.iloc[0]["f1_mean"]
    )

    f1_errors.append(
        row.iloc[0]["f1_std"]
    )


fig, ax = plt.subplots(
    figsize=(9, 6)
)

bars = ax.bar(
    MODEL_ORDER,
    f1_values,
    yerr=f1_errors,
    capsize=5,
    color=[
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
    ]
)

ax.set_xlabel(
    "Model",
    fontsize=12
)

ax.set_ylabel(
    "F1 Score",
    fontsize=12
)

ax.set_title(
    "Development-Set F1 Score Without SMOTE",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylim(
    0,
    1.0
)

for bar, value in zip(
    bars,
    f1_values
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,

        bar.get_height(),

        f"{value:.4f}",

        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )


ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

f1_figure = (
    FIGURES_DIR
    / "development_f1_no_smote.png"
)

plt.savefig(
    f1_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {f1_figure}"
)


# ============================================================
# MCC FIGURE
# ============================================================

print("Generating MCC comparison...")

mcc_values = []

mcc_errors = []

for model_name in MODEL_ORDER:

    row = summary_df[
        summary_df["model"]
        == model_name
    ]

    mcc_values.append(
        row.iloc[0]["mcc_mean"]
    )

    mcc_errors.append(
        row.iloc[0]["mcc_std"]
    )


fig, ax = plt.subplots(
    figsize=(9, 6)
)

bars = ax.bar(
    MODEL_ORDER,
    mcc_values,
    yerr=mcc_errors,
    capsize=5,
    color=[
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
    ]
)

ax.set_xlabel(
    "Model",
    fontsize=12
)

ax.set_ylabel(
    "Matthews Correlation Coefficient",
    fontsize=12
)

ax.set_title(
    "Development-Set MCC Without SMOTE",
    fontsize=14,
    fontweight="bold"
)

for bar, value in zip(
    bars,
    mcc_values
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,

        bar.get_height(),

        f"{value:.4f}",

        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )


ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

mcc_figure = (
    FIGURES_DIR
    / "development_mcc_no_smote.png"
)

plt.savefig(
    mcc_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {mcc_figure}"
)


# ============================================================
# ROC-AUC FIGURE
# ============================================================

print("Generating ROC-AUC comparison...")

roc_values = []

roc_errors = []

for model_name in MODEL_ORDER:

    row = summary_df[
        summary_df["model"]
        == model_name
    ]

    roc_values.append(
        row.iloc[0]["roc_auc_mean"]
    )

    roc_errors.append(
        row.iloc[0]["roc_auc_std"]
    )


fig, ax = plt.subplots(
    figsize=(9, 6)
)

bars = ax.bar(
    MODEL_ORDER,
    roc_values,
    yerr=roc_errors,
    capsize=5,
    color=[
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
    ]
)

ax.set_xlabel(
    "Model",
    fontsize=12
)

ax.set_ylabel(
    "ROC-AUC",
    fontsize=12
)

ax.set_title(
    "Development-Set ROC-AUC Without SMOTE",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylim(
    0,
    1.05
)

for bar, value in zip(
    bars,
    roc_values
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,

        bar.get_height(),

        f"{value:.4f}",

        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )


ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

roc_figure = (
    FIGURES_DIR
    / "development_roc_auc_no_smote.png"
)

plt.savefig(
    roc_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {roc_figure}"
)


# ============================================================
# PR-AUC FIGURE
# ============================================================

print("Generating PR-AUC comparison...")

pr_values = []

pr_errors = []

for model_name in MODEL_ORDER:

    row = summary_df[
        summary_df["model"]
        == model_name
    ]

    pr_values.append(
        row.iloc[0]["pr_auc_mean"]
    )

    pr_errors.append(
        row.iloc[0]["pr_auc_std"]
    )


fig, ax = plt.subplots(
    figsize=(9, 6)
)

bars = ax.bar(
    MODEL_ORDER,
    pr_values,
    yerr=pr_errors,
    capsize=5,
    color=[
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
    ]
)

ax.set_xlabel(
    "Model",
    fontsize=12
)

ax.set_ylabel(
    "PR-AUC",
    fontsize=12
)

ax.set_title(
    "Development-Set PR-AUC Without SMOTE",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylim(
    0,
    1.05
)

for bar, value in zip(
    bars,
    pr_values
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,

        bar.get_height(),

        f"{value:.4f}",

        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )


ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

pr_figure = (
    FIGURES_DIR
    / "development_pr_auc_no_smote.png"
)

plt.savefig(
    pr_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {pr_figure}"
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

print("\nGenerating confusion matrices...")

for model_name in MODEL_ORDER:

    model_cm = confusion_df[
        confusion_df["model"]
        == model_name
    ]

    cm = np.array(
        [
            [
                model_cm["TN"].sum(),
                model_cm["FP"].sum(),
            ],

            [
                model_cm["FN"].sum(),
                model_cm["TP"].sum(),
            ],
        ]
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="viridis"
    )

    ax.set_title(
        f"{model_name} - Development Set\n"
        "Confusion Matrix Without SMOTE",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Predicted Label",
        fontsize=12
    )

    ax.set_ylabel(
        "True Label",
        fontsize=12
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_xticklabels(
        [
            "Non-defective",
            "Defective"
        ]
    )

    ax.set_yticklabels(
        [
            "Non-defective",
            "Defective"
        ]
    )

    # --------------------------------------------------------
    # Add values
    # --------------------------------------------------------

    max_value = cm.max()

    for i in range(2):

        for j in range(2):

            # Choose text color based on cell intensity
            if cm[i, j] > max_value * 0.5:
                text_color = "black"
            else:
                text_color = "white"

            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=17,
                fontweight="bold",
                color=text_color
            )

    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    colorbar = plt.colorbar(
        image,
        ax=ax
    )

    colorbar.ax.tick_params(
        labelsize=10
    )

    plt.tight_layout()

    cm_file = (
        FIGURES_DIR
        / f"{model_name.lower()}_confusion_matrix_development_no_smote.png"
    )

    plt.savefig(
        cm_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {cm_file}"
    )


# ============================================================
# TEXT REPORT
# ============================================================

report_file = (
    RESULTS_DIR
    / "development_baseline_no_smote_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "JM1 DEVELOPMENT-SET BASELINE "
        "EVALUATION WITHOUT SMOTE\n"
    )

    f.write(
        "=" * 80
        + "\n\n"
    )

    f.write(
        f"Development samples: {len(df):,}\n"
    )

    f.write(
        f"Number of features: {X.shape[1]}\n"
    )

    f.write(
        f"Validation: Stratified {N_SPLITS}-Fold CV\n"
    )

    f.write(
        "SMOTE: Not applied\n"
    )

    f.write(
        "Test set: Not used\n\n"
    )

    f.write(
        "MODEL PERFORMANCE\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    for _, row in summary_df.iterrows():

        f.write(
            f"\n{row['model']}\n"
        )

        f.write(
            f"Accuracy    : "
            f"{row['accuracy_mean']:.4f} "
            f"+/- {row['accuracy_std']:.4f}\n"
        )

        f.write(
            f"Precision   : "
            f"{row['precision_mean']:.4f} "
            f"+/- {row['precision_std']:.4f}\n"
        )

        f.write(
            f"Recall      : "
            f"{row['recall_mean']:.4f} "
            f"+/- {row['recall_std']:.4f}\n"
        )

        f.write(
            f"Specificity : "
            f"{row['specificity_mean']:.4f} "
            f"+/- {row['specificity_std']:.4f}\n"
        )

        f.write(
            f"F1          : "
            f"{row['f1_mean']:.4f} "
            f"+/- {row['f1_std']:.4f}\n"
        )

        f.write(
            f"MCC         : "
            f"{row['mcc_mean']:.4f} "
            f"+/- {row['mcc_std']:.4f}\n"
        )

        f.write(
            f"ROC-AUC     : "
            f"{row['roc_auc_mean']:.4f} "
            f"+/- {row['roc_auc_std']:.4f}\n"
        )

        f.write(
            f"PR-AUC      : "
            f"{row['pr_auc_mean']:.4f} "
            f"+/- {row['pr_auc_std']:.4f}\n"
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 90)
print("DEVELOPMENT-SET BASELINE COMPLETED SUCCESSFULLY")
print("=" * 90)

print("\nModels:")
print("  ✓ Random Forest")
print("  ✓ XGBoost")
print("  ✓ LightGBM")
print("  ✓ MLP")

print("\nValidation:")
print("  ✓ Stratified 5-Fold Cross-Validation")
print("  ✓ Development set only")
print("  ✓ No SMOTE")
print("  ✓ No hyperparameter tuning")

print("\nTest set:")
print("  ✓ NOT USED")
print("  ✓ Remains untouched")

print("\nMetrics:")
print("  ✓ Accuracy")
print("  ✓ Precision")
print("  ✓ Recall")
print("  ✓ Specificity")
print("  ✓ F1")
print("  ✓ MCC")
print("  ✓ ROC-AUC")
print("  ✓ PR-AUC")

print("\nGenerated figures:")
print("  ✓ Comprehensive baseline model performance")
print("  ✓ Accuracy comparison")
print("  ✓ F1 comparison")
print("  ✓ MCC comparison")
print("  ✓ ROC-AUC comparison")
print("  ✓ PR-AUC comparison")
print("  ✓ RF confusion matrix")
print("  ✓ XGB confusion matrix")
print("  ✓ LGBM confusion matrix")
print("  ✓ MLP confusion matrix")

print("\nResults directory:")
print(RESULTS_DIR)

print("\nFigures directory:")
print(FIGURES_DIR)

print("\n" + "=" * 90)
print("NEXT STEP: 05_SMOTE_EXPERIMENT.PY")
print("=" * 90)
