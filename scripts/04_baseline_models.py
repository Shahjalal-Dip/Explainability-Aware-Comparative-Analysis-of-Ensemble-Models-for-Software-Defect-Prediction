"""
04_baseline_models.py

JM1 Software Defect Prediction
Baseline Model Evaluation WITHOUT SMOTE

Models:
    1. Random Forest
    2. XGBoost
    3. LightGBM
    4. MLP

Evaluation:
    Stratified 5-Fold Cross-Validation

Important:
    - NO SMOTE is applied in this script.
    - No hyperparameter tuning is performed here.
    - This script establishes baseline performance.
    - SMOTE comparison is performed separately in 05_smote_experiment.py.

Outputs:
    data/results/performance/
    figures/performance/
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
    / "JM1_cleaned.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "performance"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "performance"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 80)
print("JM1 BASELINE MODEL EVALUATION - WITHOUT SMOTE")
print("=" * 80)

print(f"Project root:\n{PROJECT_ROOT}")
print(f"\nInput dataset:\n{INPUT_FILE}")

print("\n" + "=" * 80)
print("EXPERIMENT DESIGN")
print("=" * 80)

print("Cross-validation : Stratified 5-Fold")
print("SMOTE            : NOT APPLIED")
print("Hyperparameter   : Default baseline configuration")
print(f"Random state     : {RANDOM_STATE}")


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nCould not find dataset:\n{INPUT_FILE}\n\n"
        "Make sure 02b_finalize_cleaning.py has been executed."
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("LOADING CLEANED DATASET")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# TARGET
# ============================================================

TARGET = "defects"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found."
    )

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(bool).astype(int)

print(f"\nTarget column: {TARGET}")
print(f"Number of features: {X.shape[1]}")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("CLASS DISTRIBUTION")
print("=" * 80)

class_counts = y.value_counts().sort_index()

print(
    f"Non-defective (0): {class_counts.get(0, 0):,} "
    f"({(class_counts.get(0, 0) / len(y)) * 100:.2f}%)"
)

print(
    f"Defective (1):     {class_counts.get(1, 0):,} "
    f"({(class_counts.get(1, 0) / len(y)) * 100:.2f}%)"
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
            )
        ]
    ),
}


# ============================================================
# CROSS-VALIDATION
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
fold_predictions = []
confusion_results = []


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("STARTING STRATIFIED 5-FOLD CROSS-VALIDATION")
print("=" * 80)

for model_name, model in models.items():

    print("\n" + "-" * 80)
    print(f"MODEL: {model_name}")
    print("-" * 80)

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X, y),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.fit(X_train, y_train)

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        y_pred = model.predict(X_test)

        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]

        else:
            y_prob = model.predict(X_test)

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

        pr_auc = average_precision_score(
            y_test,
            y_prob
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        tn, fp, fn, tp = cm.ravel()

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0
        )

        # ----------------------------------------------------
        # STORE METRICS
        # ----------------------------------------------------

        result = {
            "model": model_name,
            "fold": fold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        }

        all_results.append(result)

        # ----------------------------------------------------
        # STORE PREDICTIONS
        # ----------------------------------------------------

        fold_prediction = pd.DataFrame(
            {
                "model": model_name,
                "fold": fold,
                "true_label": y_test.values,
                "predicted_label": y_pred,
                "predicted_probability": y_prob,
            }
        )

        fold_predictions.append(
            fold_prediction
        )

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        confusion_results.append(
            {
                "model": model_name,
                "fold": fold,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            }
        )

        print(
            f"Fold {fold}: "
            f"Accuracy={accuracy:.4f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f} | "
            f"ROC-AUC={roc_auc:.4f} | "
            f"PR-AUC={pr_auc:.4f}"
        )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)

predictions_df = pd.concat(
    fold_predictions,
    ignore_index=True
)

confusion_df = pd.DataFrame(
    confusion_results
)


# ============================================================
# SAVE FOLD RESULTS
# ============================================================

fold_results_file = (
    RESULTS_DIR
    / "baseline_no_smote_fold_results.csv"
)

results_df.to_csv(
    fold_results_file,
    index=False
)

print("\n" + "=" * 80)
print("FOLD-LEVEL RESULTS SAVED")
print("=" * 80)

print(fold_results_file)


# ============================================================
# MODEL SUMMARY
# ============================================================

summary_df = (
    results_df
    .groupby("model")
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),

        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),

        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),

        specificity_mean=("specificity", "mean"),
        specificity_std=("specificity", "std"),

        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),

        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),

        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
    )
    .reset_index()
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = (
    RESULTS_DIR
    / "baseline_no_smote_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)

print("\n" + "=" * 80)
print("BASELINE PERFORMANCE SUMMARY")
print("=" * 80)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print(f"\nSaved:\n{summary_file}")


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_file = (
    RESULTS_DIR
    / "baseline_no_smote_predictions.csv"
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
    / "baseline_no_smote_confusion_matrices.csv"
)

confusion_df.to_csv(
    confusion_file,
    index=False
)


# ============================================================
# FIGURE 1
# MODEL PERFORMANCE COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("GENERATING MODEL PERFORMANCE COMPARISON")
print("=" * 80)

metrics = [
    "accuracy_mean",
    "precision_mean",
    "recall_mean",
    "f1_mean",
    "roc_auc_mean",
    "pr_auc_mean",
]

metric_labels = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "PR-AUC",
]

models_order = [
    "RF",
    "XGB",
    "LGBM",
    "MLP",
]

x = np.arange(len(models_order))
width = 0.12

fig, ax = plt.subplots(
    figsize=(12, 7)
)

for i, (metric, label) in enumerate(
    zip(metrics, metric_labels)
):

    values = []

    for model_name in models_order:

        row = summary_df[
            summary_df["model"] == model_name
        ]

        if len(row) == 1:
            values.append(
                row.iloc[0][metric]
            )
        else:
            values.append(0)

    ax.bar(
        x + (i - 2.5) * width,
        values,
        width,
        label=label
    )


ax.set_xlabel(
    "Model",
    fontsize=12
)

ax.set_ylabel(
    "Score",
    fontsize=12
)

ax.set_title(
    "Baseline Model Performance on JM1 Dataset (Without SMOTE)",
    fontsize=14,
    fontweight="bold"
)

ax.set_xticks(x)
ax.set_xticklabels(models_order)

ax.set_ylim(
    0,
    1.05
)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=3
)

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

performance_figure = (
    FIGURES_DIR
    / "baseline_model_performance_no_smote.png"
)

plt.savefig(
    performance_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {performance_figure}")


# ============================================================
# FIGURE 2
# F1 SCORE COMPARISON
# ============================================================

print("\nGenerating F1 comparison figure...")

fig, ax = plt.subplots(
    figsize=(9, 6)
)

f1_values = []

for model_name in models_order:

    row = summary_df[
        summary_df["model"] == model_name
    ]

    f1_values.append(
        row.iloc[0]["f1_mean"]
    )


bars = ax.bar(
    models_order,
    f1_values
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
    "F1 Score Comparison - Without SMOTE",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylim(
    0,
    max(f1_values) * 1.2
)

for bar, value in zip(
    bars,
    f1_values
):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

f1_figure = (
    FIGURES_DIR
    / "baseline_f1_comparison_no_smote.png"
)

plt.savefig(
    f1_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {f1_figure}")


# ============================================================
# FIGURE 3
# ROC-AUC COMPARISON
# ============================================================

print("\nGenerating ROC-AUC comparison figure...")

fig, ax = plt.subplots(
    figsize=(9, 6)
)

roc_values = []

for model_name in models_order:

    row = summary_df[
        summary_df["model"] == model_name
    ]

    roc_values.append(
        row.iloc[0]["roc_auc_mean"]
    )


bars = ax.bar(
    models_order,
    roc_values
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
    "ROC-AUC Comparison - Without SMOTE",
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
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

roc_figure = (
    FIGURES_DIR
    / "baseline_roc_auc_comparison_no_smote.png"
)

plt.savefig(
    roc_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {roc_figure}")


# ============================================================
# FIGURE 4
# PR-AUC COMPARISON
# ============================================================

print("\nGenerating PR-AUC comparison figure...")

fig, ax = plt.subplots(
    figsize=(9, 6)
)

pr_values = []

for model_name in models_order:

    row = summary_df[
        summary_df["model"] == model_name
    ]

    pr_values.append(
        row.iloc[0]["pr_auc_mean"]
    )


bars = ax.bar(
    models_order,
    pr_values
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
    "PR-AUC Comparison - Without SMOTE",
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
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

pr_figure = (
    FIGURES_DIR
    / "baseline_pr_auc_comparison_no_smote.png"
)

plt.savefig(
    pr_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {pr_figure}")


# ============================================================
# FIGURE 5
# CONFUSION MATRIX HEATMAPS
# ============================================================

print("\nGenerating confusion matrix figures...")

for model_name in models_order:

    model_cm = confusion_df[
        confusion_df["model"] == model_name
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
        figsize=(6, 5)
    )

    image = ax.imshow(
        cm,
        interpolation="nearest"
    )

    ax.set_title(
        f"{model_name} - Confusion Matrix\nWithout SMOTE",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "True Label"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Non-defective", "Defective"]
    )

    ax.set_yticklabels(
        ["Non-defective", "Defective"]
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=13
            )

    plt.colorbar(
        image,
        ax=ax
    )

    plt.tight_layout()

    cm_file = (
        FIGURES_DIR
        / f"{model_name.lower()}_confusion_matrix_no_smote.png"
    )

    plt.savefig(
        cm_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {cm_file}")


# ============================================================
# GENERATE TEXT REPORT
# ============================================================

report_file = (
    RESULTS_DIR
    / "baseline_no_smote_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "JM1 BASELINE MODEL EVALUATION - WITHOUT SMOTE\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Dataset rows: {len(df):,}\n"
    )

    f.write(
        f"Number of features: {X.shape[1]}\n"
    )

    f.write(
        f"Cross-validation: Stratified {N_SPLITS}-Fold\n"
    )

    f.write(
        "SMOTE: Not applied\n\n"
    )

    f.write(
        "MODEL PERFORMANCE\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    for _, row in summary_df.iterrows():

        f.write(
            f"\n{row['model']}\n"
        )

        f.write(
            f"Accuracy : {row['accuracy_mean']:.4f} "
            f"+/- {row['accuracy_std']:.4f}\n"
        )

        f.write(
            f"Precision: {row['precision_mean']:.4f} "
            f"+/- {row['precision_std']:.4f}\n"
        )

        f.write(
            f"Recall   : {row['recall_mean']:.4f} "
            f"+/- {row['recall_std']:.4f}\n"
        )

        f.write(
            f"F1       : {row['f1_mean']:.4f} "
            f"+/- {row['f1_std']:.4f}\n"
        )

        f.write(
            f"ROC-AUC  : {row['roc_auc_mean']:.4f} "
            f"+/- {row['roc_auc_std']:.4f}\n"
        )

        f.write(
            f"PR-AUC   : {row['pr_auc_mean']:.4f} "
            f"+/- {row['pr_auc_std']:.4f}\n"
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("BASELINE MODELING COMPLETED")
print("=" * 80)

print("\nModels evaluated:")
print("  ✓ Random Forest")
print("  ✓ XGBoost")
print("  ✓ LightGBM")
print("  ✓ MLP")

print("\nValidation:")
print("  ✓ Stratified 5-Fold Cross-Validation")
print("  ✓ No SMOTE")
print("  ✓ No hyperparameter tuning")

print("\nResult files:")
print(
    f"  ✓ {fold_results_file.name}"
)

print(
    f"  ✓ {summary_file.name}"
)

print(
    f"  ✓ {predictions_file.name}"
)

print(
    f"  ✓ {confusion_file.name}"
)

print(
    f"  ✓ {report_file.name}"
)

print("\nFigures:")
print(
    "  ✓ baseline_model_performance_no_smote.png"
)

print(
    "  ✓ baseline_f1_comparison_no_smote.png"
)

print(
    "  ✓ baseline_roc_auc_comparison_no_smote.png"
)

print(
    "  ✓ baseline_pr_auc_comparison_no_smote.png"
)

print(
    "  ✓ RF/XGB/LGBM/MLP confusion matrices"
)

print("\n" + "=" * 80)
print("READY FOR 05_SMOTE_EXPERIMENT.PY")
print("=" * 80)
