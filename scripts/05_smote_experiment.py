"""
05_smote_experiment.py

JM1 Software Defect Prediction
Development-Set SMOTE Experiment

Purpose:
    Evaluate the effect of SMOTE on software defect prediction
    using only the 80% development set.

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

SMOTE:
    Applied ONLY to the training portion of each fold.

Important:
    - No SMOTE is applied to validation data.
    - No SMOTE is applied to the test set.
    - No test-set access.
    - No hyperparameter tuning.
    - No feature selection.
    - No data leakage.
    - Same baseline model configurations as Step 04b.

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
    data/results/performance/development_smote/

Figures:
    figures/performance/development_smote/
        development_smote_model_performance.png
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

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


# ------------------------------------------------------------
# INPUT DATASET
# ------------------------------------------------------------

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "JM1"
    / "JM1_development.csv"
)


# ------------------------------------------------------------
# RESULTS DIRECTORY
# ------------------------------------------------------------

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "performance"
    / "development_smote"
)


# ------------------------------------------------------------
# FIGURES DIRECTORY
# ------------------------------------------------------------

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "performance"
    / "development_smote"
)


# ------------------------------------------------------------
# CREATE DIRECTORIES
# ------------------------------------------------------------

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


# ============================================================
# HEADER
# ============================================================

print("=" * 90)
print("JM1 DEVELOPMENT-SET SMOTE EXPERIMENT")
print("=" * 90)

print("\nInput dataset:")
print(INPUT_FILE)

print("\nExperimental design:")
print("Dataset           : JM1 Development Set")
print("Development size  : 80% of original dataset")
print("Test set          : 20% - COMPLETELY UNTOUCHED")
print("Validation        : Stratified 5-Fold CV")
print("SMOTE             : Applied to training folds ONLY")
print("Validation data   : NOT oversampled")
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
# LOAD DEVELOPMENT DATA
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
# ORIGINAL CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 90)
print("ORIGINAL DEVELOPMENT-SET CLASS DISTRIBUTION")
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
# SMOTE CONFIGURATION
# ============================================================

smote = SMOTE(
    random_state=RANDOM_STATE
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
print("STARTING STRATIFIED 5-FOLD CROSS-VALIDATION WITH SMOTE")
print("=" * 90)

print(
    "\nIMPORTANT:"
)

print(
    "SMOTE is applied independently inside each training fold."
)

print(
    "Validation folds remain in their original class distribution."
)


for model_name, model in models.items():

    print("\n" + "-" * 90)
    print(f"MODEL: {model_name}")
    print("-" * 90)

    for fold, (train_idx, validation_idx) in enumerate(
        skf.split(X, y),
        start=1
    ):

        # ----------------------------------------------------
        # ORIGINAL TRAINING / VALIDATION SPLIT
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
        # CLASS DISTRIBUTION BEFORE SMOTE
        # ----------------------------------------------------

        train_class_counts_before = (
            y_train.value_counts()
            .sort_index()
        )

        minority_before = int(
            train_class_counts_before.get(
                1,
                0
            )
        )

        majority_before = int(
            train_class_counts_before.get(
                0,
                0
            )
        )

        # ----------------------------------------------------
        # APPLY SMOTE TO TRAINING FOLD ONLY
        # ----------------------------------------------------

        X_train_smote, y_train_smote = (
            smote.fit_resample(
                X_train,
                y_train
            )
        )

        # ----------------------------------------------------
        # CLASS DISTRIBUTION AFTER SMOTE
        # ----------------------------------------------------

        train_class_counts_after = (
            pd.Series(y_train_smote)
            .value_counts()
            .sort_index()
        )

        minority_after = int(
            train_class_counts_after.get(
                1,
                0
            )
        )

        majority_after = int(
            train_class_counts_after.get(
                0,
                0
            )
        )

        # ----------------------------------------------------
        # TRAIN MODEL
        # ----------------------------------------------------

        model.fit(
            X_train_smote,
            y_train_smote
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

                "train_samples_before_smote":
                    len(y_train),

                "train_nondefective_before_smote":
                    majority_before,

                "train_defective_before_smote":
                    minority_before,

                "train_samples_after_smote":
                    len(y_train_smote),

                "train_nondefective_after_smote":
                    majority_after,

                "train_defective_after_smote":
                    minority_after,

                "validation_samples":
                    len(y_validation),

                "validation_nondefective":
                    int(
                        (y_validation == 0).sum()
                    ),

                "validation_defective":
                    int(
                        (y_validation == 1).sum()
                    ),

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

        print(
            f"         SMOTE: "
            f"{len(y_train):,} → "
            f"{len(y_train_smote):,} training samples"
        )

        print(
            f"         Classes after SMOTE: "
            f"Non-defective={majority_after:,}, "
            f"Defective={minority_after:,}"
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
    / "development_smote_fold_results.csv"
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
# SAVE SUMMARY
# ============================================================

summary_file = (
    RESULTS_DIR
    / "development_smote_summary.csv"
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
    / "development_smote_predictions.csv"
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
    / "development_smote_confusion_matrices.csv"
)

confusion_df.to_csv(
    confusion_file,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("DEVELOPMENT-SET SMOTE PERFORMANCE")
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
# COMBINED SMOTE PERFORMANCE FIGURE
# ============================================================

print("\n" + "=" * 90)
print("GENERATING SMOTE PERFORMANCE FIGURE")
print("=" * 90)


# ------------------------------------------------------------
# MODEL ORDER
# ------------------------------------------------------------

model_order = [
    "RF",
    "XGB",
    "LGBM",
    "MLP",
]


# ------------------------------------------------------------
# PERFORMANCE METRICS
# ------------------------------------------------------------

metric_names = [

    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "MCC",
    "ROC-AUC",
    "PR-AUC",
]


metric_columns = [

    "accuracy_mean",
    "precision_mean",
    "recall_mean",
    "specificity_mean",
    "f1_mean",
    "mcc_mean",
    "roc_auc_mean",
    "pr_auc_mean",
]


# ------------------------------------------------------------
# CREATE PERFORMANCE MATRIX
# ------------------------------------------------------------

performance_matrix = []

error_matrix = []


for model_name in model_order:

    row = summary_df[
        summary_df["model"]
        == model_name
    ]

    if row.empty:

        raise ValueError(
            f"Model '{model_name}' was not found "
            "in the SMOTE summary."
        )

    performance_values = [

        row.iloc[0]["accuracy_mean"],

        row.iloc[0]["precision_mean"],

        row.iloc[0]["recall_mean"],

        row.iloc[0]["specificity_mean"],

        row.iloc[0]["f1_mean"],

        row.iloc[0]["mcc_mean"],

        row.iloc[0]["roc_auc_mean"],

        row.iloc[0]["pr_auc_mean"],
    ]

    error_values = [

        row.iloc[0]["accuracy_std"],

        row.iloc[0]["precision_std"],

        row.iloc[0]["recall_std"],

        row.iloc[0]["specificity_std"],

        row.iloc[0]["f1_std"],

        row.iloc[0]["mcc_std"],

        row.iloc[0]["roc_auc_std"],

        row.iloc[0]["pr_auc_std"],
    ]

    performance_matrix.append(
        performance_values
    )

    error_matrix.append(
        error_values
    )


# ------------------------------------------------------------
# CONVERT TO NUMPY
# ------------------------------------------------------------

performance_matrix = np.array(
    performance_matrix
)

error_matrix = np.array(
    error_matrix
)


# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

x = np.arange(
    len(metric_names)
)

width = 0.20


fig, ax = plt.subplots(
    figsize=(15, 7)
)


for i, model_name in enumerate(
    model_order
):

    positions = (
        x
        + (i - 1.5) * width
    )

    bars = ax.bar(
        positions,

        performance_matrix[i],

        width,

        yerr=error_matrix[i],

        capsize=3,

        label=model_name
    )

    # --------------------------------------------------------
    # VALUE LABELS
    # --------------------------------------------------------

    for bar, value in zip(
        bars,
        performance_matrix[i]
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height()
            + 0.01,

            f"{value:.3f}",

            ha="center",

            va="bottom",

            fontsize=8,

            rotation=90
        )


# ------------------------------------------------------------
# AXIS LABELS
# ------------------------------------------------------------

ax.set_xlabel(
    "Performance Metric",
    fontsize=12
)

ax.set_ylabel(
    "Score",
    fontsize=12
)


# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------

ax.set_title(
    "JM1 Development-Set Baseline Model Performance With SMOTE",
    fontsize=14,
    fontweight="bold"
)


# ------------------------------------------------------------
# X TICKS
# ------------------------------------------------------------

ax.set_xticks(
    x
)

ax.set_xticklabels(
    metric_names,
    fontsize=10
)


# ------------------------------------------------------------
# Y AXIS
# ------------------------------------------------------------

ax.set_ylim(
    0,
    1.05
)


# ------------------------------------------------------------
# GRID
# ------------------------------------------------------------

ax.grid(
    axis="y",
    alpha=0.3
)


# ------------------------------------------------------------
# LEGEND
# ------------------------------------------------------------

ax.legend(
    title="Model"
)


# ------------------------------------------------------------
# LAYOUT
# ------------------------------------------------------------

plt.tight_layout()


# ------------------------------------------------------------
# SAVE FIGURE
# ------------------------------------------------------------

performance_figure = (
    FIGURES_DIR
    / "development_smote_model_performance.png"
)


plt.savefig(
    performance_figure,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"Saved: {performance_figure}"
)


# ============================================================
# TEXT REPORT
# ============================================================

report_file = (
    RESULTS_DIR
    / "development_smote_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "JM1 DEVELOPMENT-SET SMOTE EXPERIMENT\n"
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
        "SMOTE: Applied to training folds only\n"
    )

    f.write(
        "Validation folds: Original class distribution preserved\n"
    )

    f.write(
        "Test set: Not used\n\n"
    )

    f.write(
        "ORIGINAL CLASS DISTRIBUTION\n"
    )

    f.write(
        "-" * 80
        + "\n"
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

        f.write(
            f"{class_name}: "
            f"{count:,} "
            f"({percentage:.2f}%)\n"
        )

    f.write(
        "\nMODEL PERFORMANCE\n"
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
print("DEVELOPMENT-SET SMOTE EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 90)

print("\nModels:")
print("  ✓ Random Forest")
print("  ✓ XGBoost")
print("  ✓ LightGBM")
print("  ✓ MLP")

print("\nValidation:")
print("  ✓ Stratified 5-Fold Cross-Validation")
print("  ✓ Development set only")
print("  ✓ SMOTE applied to training folds only")
print("  ✓ Validation folds NOT oversampled")
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

print("\nGenerated figure:")
print("  ✓ Combined SMOTE model performance")

print("\nResults directory:")
print(RESULTS_DIR)

print("\nFigures directory:")
print(FIGURES_DIR)

print("\n" + "=" * 90)
print("NEXT STEP: 06_SMOTE_VS_NO_SMOTE_COMPARISON.PY")
print("=" * 90)
