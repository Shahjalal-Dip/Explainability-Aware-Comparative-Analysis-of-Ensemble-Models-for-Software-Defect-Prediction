
"""
07_hyperparameter_tuning.py

JM1 SOFTWARE DEFECT PREDICTION
STEP 6: HYPERPARAMETER TUNING

Experimental design:
    - Development set only
    - Locked 20% test set is NEVER accessed
    - Stratified 5-Fold Cross-Validation
    - SMOTE applied to training folds ONLY
    - Validation folds remain untouched
    - Hyperparameter tuning performed within CV
    - Primary optimization metric: F1
    - Secondary metrics: MCC, PR-AUC, ROC-AUC

Models:
    - Random Forest
    - XGBoost
    - LightGBM
    - MLP

Selected preprocessing branch:
    - SMOTE

Random state:
    42
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# =============================================================================
# CONFIGURATION
# =============================================================================

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5

TARGET = "defects"


# =============================================================================
# PROJECT PATHS
# =============================================================================

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
    / "hyperparameter_tuning"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# OUTPUT FILES
# =============================================================================

BEST_PARAMS_FILE = (
    RESULTS_DIR
    / "best_hyperparameters_smote.csv"
)

CV_RESULTS_FILE = (
    RESULTS_DIR
    / "hyperparameter_tuning_cv_results.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "hyperparameter_tuning_summary.csv"
)

REPORT_FILE = (
    RESULTS_DIR
    / "hyperparameter_tuning_report.txt"
)


# =============================================================================
# HEADER
# =============================================================================

print("=" * 90)
print("JM1 DEVELOPMENT-SET HYPERPARAMETER TUNING")
print("SMOTE BRANCH")
print("=" * 90)

print()
print("Input dataset:")
print(INPUT_FILE)

print()
print("Experimental design:")
print("Dataset           : JM1 Development Set")
print("Development size  : 80% of original dataset")
print("Test set          : 20% - COMPLETELY UNTOUCHED")
print("Validation        : Stratified 5-Fold CV")
print("SMOTE             : Applied to training folds ONLY")
print("Validation data   : NOT oversampled")
print("Optimization      : F1-score")
print("Secondary metrics : MCC, ROC-AUC, PR-AUC")
print("Random state      :", RANDOM_STATE)

print()
print("IMPORTANT:")
print("The locked 20% test set is NOT accessed.")
print("SMOTE is applied INSIDE each CV training fold only.")
print("Hyperparameter selection is performed using development-set CV only.")


# =============================================================================
# LOAD DATASET
# =============================================================================

print()
print("=" * 90)
print("LOADING DEVELOPMENT DATASET")
print("=" * 90)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input dataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")
print(f"Rows: {len(df):,}")
print(f"Columns: {df.shape[1]}")


# =============================================================================
# VALIDATE TARGET
# =============================================================================

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found."
    )

X = df.drop(columns=[TARGET])
y = df[TARGET]

print()
print(f"Target column: {TARGET}")
print(f"Number of features: {X.shape[1]}")


# =============================================================================
# CLASS DISTRIBUTION
# =============================================================================

print()
print("=" * 90)
print("DEVELOPMENT-SET CLASS DISTRIBUTION")
print("=" * 90)

class_counts = y.value_counts().sort_index()

for class_value, count in class_counts.items():

    percentage = (
        count / len(y)
    ) * 100

    if class_value == 0:
        label = "Non-defective"
    else:
        label = "Defective"

    print(
        f"{label:<16}: {count:,} "
        f"({percentage:.2f}%)"
    )


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

models = {

    "RF": ImbPipeline(
        steps=[
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE
                )
            ),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight=None
                )
            )
        ]
    ),

    "XGB": ImbPipeline(
        steps=[
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE
                )
            ),
            (
                "model",
                XGBClassifier(
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    n_jobs=-1,
                    verbosity=0
                )
            )
        ]
    ),

    "LGBM": ImbPipeline(
        steps=[
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE
                )
            ),
            (
                "model",
                LGBMClassifier(
                    random_state=RANDOM_STATE,
                    verbosity=-1,
                    n_jobs=-1
                )
            )
        ]
    ),

    "MLP": ImbPipeline(
        steps=[
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                MLPClassifier(
                    random_state=RANDOM_STATE,
                    max_iter=500
                )
            )
        ]
    )
}


# =============================================================================
# HYPERPARAMETER SEARCH SPACES
# =============================================================================

param_grids = {

    # -------------------------------------------------------------------------
    # RANDOM FOREST
    # -------------------------------------------------------------------------
    "RF": {
        "model__n_estimators": [
            200,
            300,
            500
        ],

        "model__max_depth": [
            None,
            10,
            20
        ],

        "model__min_samples_split": [
            2,
            5
        ],

        "model__min_samples_leaf": [
            1,
            2
        ],

        "model__max_features": [
            "sqrt",
            "log2"
        ]
    },


    # -------------------------------------------------------------------------
    # XGBOOST
    # -------------------------------------------------------------------------
    "XGB": {
        "model__n_estimators": [
            200,
            300,
            500
        ],

        "model__max_depth": [
            3,
            5,
            7
        ],

        "model__learning_rate": [
            0.01,
            0.05,
            0.10
        ],

        "model__subsample": [
            0.8,
            1.0
        ],

        "model__colsample_bytree": [
            0.8,
            1.0
        ]
    },


    # -------------------------------------------------------------------------
    # LIGHTGBM
    # -------------------------------------------------------------------------
    "LGBM": {
        "model__n_estimators": [
            200,
            300,
            500
        ],

        "model__num_leaves": [
            15,
            31,
            63
        ],

        "model__learning_rate": [
            0.01,
            0.05,
            0.10
        ],

        "model__max_depth": [
            -1,
            10,
            20
        ],

        "model__min_child_samples": [
            10,
            20,
            30
        ]
    },


    # -------------------------------------------------------------------------
    # MLP
    # -------------------------------------------------------------------------
    "MLP": {
        "model__hidden_layer_sizes": [
            (50,),
            (100,),
            (100, 50)
        ],

        "model__activation": [
            "relu",
            "tanh"
        ],

        "model__alpha": [
            0.0001,
            0.001,
            0.01
        ],

        "model__learning_rate_init": [
            0.001,
            0.01
        ]
    }
}


# =============================================================================
# SCORING
# =============================================================================

scoring = {
    "f1": "f1",
    "mcc": "matthews_corrcoef",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision"
}


# =============================================================================
# STORAGE
# =============================================================================

best_parameter_records = []
all_cv_results = []
summary_records = []


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

print()
print("=" * 90)
print("STARTING HYPERPARAMETER TUNING")
print("=" * 90)

print()
print("Primary optimization metric: F1")
print("All preprocessing is contained inside the CV pipeline.")


for model_name, pipeline in models.items():

    print()
    print("-" * 90)
    print(f"MODEL: {model_name}")
    print("-" * 90)

    print("Searching hyperparameters...")
    print("SMOTE is applied independently inside each training fold.")

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[model_name],
        scoring=scoring,
        refit="f1",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
        verbose=1
    )

    search.fit(X, y)

    # -------------------------------------------------------------------------
    # BEST RESULT
    # -------------------------------------------------------------------------

    best_index = search.best_index_

    best_f1 = (
        search.cv_results_["mean_test_f1"][best_index]
    )

    best_f1_std = (
        search.cv_results_["std_test_f1"][best_index]
    )

    best_mcc = (
        search.cv_results_["mean_test_mcc"][best_index]
    )

    best_mcc_std = (
        search.cv_results_["std_test_mcc"][best_index]
    )

    best_roc_auc = (
        search.cv_results_["mean_test_roc_auc"][best_index]
    )

    best_roc_auc_std = (
        search.cv_results_["std_test_roc_auc"][best_index]
    )

    best_pr_auc = (
        search.cv_results_["mean_test_pr_auc"][best_index]
    )

    best_pr_auc_std = (
        search.cv_results_["std_test_pr_auc"][best_index]
    )

    print()
    print(f"Best F1:       {best_f1:.4f} ± {best_f1_std:.4f}")
    print(f"Best MCC:      {best_mcc:.4f} ± {best_mcc_std:.4f}")
    print(f"Best ROC-AUC:  {best_roc_auc:.4f} ± {best_roc_auc_std:.4f}")
    print(f"Best PR-AUC:   {best_pr_auc:.4f} ± {best_pr_auc_std:.4f}")

    print()
    print("Best hyperparameters:")

    for parameter, value in search.best_params_.items():

        print(
            f"  {parameter}: {value}"
        )

    # -------------------------------------------------------------------------
    # SAVE BEST PARAMETERS
    # -------------------------------------------------------------------------

    parameter_record = {
        "model": model_name,
        "best_f1_mean": best_f1,
        "best_f1_std": best_f1_std,
        "best_mcc_mean": best_mcc,
        "best_mcc_std": best_mcc_std,
        "best_roc_auc_mean": best_roc_auc,
        "best_roc_auc_std": best_roc_auc_std,
        "best_pr_auc_mean": best_pr_auc,
        "best_pr_auc_std": best_pr_auc_std,
        "best_parameters": str(
            search.best_params_
        )
    }

    best_parameter_records.append(
        parameter_record
    )

    # -------------------------------------------------------------------------
    # SAVE COMPLETE GRID SEARCH RESULTS
    # -------------------------------------------------------------------------

    model_cv_results = pd.DataFrame(
        search.cv_results_
    )

    model_cv_results.insert(
        0,
        "model",
        model_name
    )

    all_cv_results.append(
        model_cv_results
    )

    # -------------------------------------------------------------------------
    # SUMMARY RECORD
    # -------------------------------------------------------------------------

    summary_records.append(
        {
            "model": model_name,
            "best_f1_mean": best_f1,
            "best_f1_std": best_f1_std,
            "best_mcc_mean": best_mcc,
            "best_mcc_std": best_mcc_std,
            "best_roc_auc_mean": best_roc_auc,
            "best_roc_auc_std": best_roc_auc_std,
            "best_pr_auc_mean": best_pr_auc,
            "best_pr_auc_std": best_pr_auc_std
        }
    )


# =============================================================================
# SAVE BEST HYPERPARAMETERS
# =============================================================================

best_parameters_df = pd.DataFrame(
    best_parameter_records
)

best_parameters_df.to_csv(
    BEST_PARAMS_FILE,
    index=False
)


# =============================================================================
# SAVE COMPLETE CV RESULTS
# =============================================================================

complete_cv_results = pd.concat(
    all_cv_results,
    ignore_index=True
)

complete_cv_results.to_csv(
    CV_RESULTS_FILE,
    index=False
)


# =============================================================================
# SAVE SUMMARY
# =============================================================================

summary_df = pd.DataFrame(
    summary_records
)

summary_df = summary_df.sort_values(
    by=[
        "best_f1_mean",
        "best_mcc_mean",
        "best_pr_auc_mean"
    ],
    ascending=False
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# =============================================================================
# DISPLAY FINAL TUNING SUMMARY
# =============================================================================

print()
print("=" * 90)
print("HYPERPARAMETER TUNING SUMMARY")
print("=" * 90)

display_columns = [
    "model",
    "best_f1_mean",
    "best_f1_std",
    "best_mcc_mean",
    "best_mcc_std",
    "best_roc_auc_mean",
    "best_roc_auc_std",
    "best_pr_auc_mean",
    "best_pr_auc_std"
]

print(
    summary_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# =============================================================================
# GENERATE TEXT REPORT
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
        "JM1 DEVELOPMENT-SET HYPERPARAMETER TUNING REPORT\n"
    )

    report.write(
        "=" * 80 + "\n\n"
    )

    report.write(
        "Experimental Design\n"
    )

    report.write(
        "-------------------\n"
    )

    report.write(
        "Dataset: JM1 Development Set\n"
    )

    report.write(
        "Validation: Stratified 5-Fold Cross-Validation\n"
    )

    report.write(
        "SMOTE: Training folds only\n"
    )

    report.write(
        "Test set: Completely untouched\n"
    )

    report.write(
        "Primary optimization metric: F1-score\n"
    )

    report.write(
        "Random state: 42\n\n"
    )

    report.write(
        "Tuning Summary\n"
    )

    report.write(
        "--------------\n\n"
    )

    for _, row in summary_df.iterrows():

        model_name = row["model"]

        report.write(
            f"{model_name}\n"
        )

        report.write(
            f"  F1:       "
            f"{row['best_f1_mean']:.4f} "
            f"+/- {row['best_f1_std']:.4f}\n"
        )

        report.write(
            f"  MCC:      "
            f"{row['best_mcc_mean']:.4f} "
            f"+/- {row['best_mcc_std']:.4f}\n"
        )

        report.write(
            f"  ROC-AUC:  "
            f"{row['best_roc_auc_mean']:.4f} "
            f"+/- {row['best_roc_auc_std']:.4f}\n"
        )

        report.write(
            f"  PR-AUC:   "
            f"{row['best_pr_auc_mean']:.4f} "
            f"+/- {row['best_pr_auc_std']:.4f}\n"
        )

        matching_parameters = best_parameters_df[
            best_parameters_df["model"] == model_name
        ]

        if not matching_parameters.empty:

            parameters = matching_parameters.iloc[0][
                "best_parameters"
            ]

            report.write(
                f"  Parameters: {parameters}\n"
            )

        report.write("\n")

    report.write(
        "Methodological Notes\n"
    )

    report.write(
        "--------------------\n"
    )

    report.write(
        "1. The locked 20% test set was never accessed.\n"
    )

    report.write(
        "2. SMOTE was applied only inside training folds.\n"
    )

    report.write(
        "3. Validation folds retained their original class distribution.\n"
    )

    report.write(
        "4. Hyperparameters were selected using development-set CV.\n"
    )

    report.write(
        "5. F1-score was used as the primary optimization criterion.\n"
    )

    report.write(
        "6. MCC, ROC-AUC and PR-AUC were retained as secondary measures.\n"
    )


print(
    f"Saved report:\n{REPORT_FILE}"
)


# =============================================================================
# FINAL STATUS
# =============================================================================

print()
print("=" * 90)
print("HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY")
print("=" * 90)

print()
print("Models tuned:")
print("  ✓ Random Forest")
print("  ✓ XGBoost")
print("  ✓ LightGBM")
print("  ✓ MLP")

print()
print("Validation:")
print("  ✓ Stratified 5-Fold Cross-Validation")
print("  ✓ Development set only")
print("  ✓ SMOTE inside training folds only")
print("  ✓ Validation folds NOT oversampled")
print("  ✓ Test set NOT accessed")

print()
print("Optimization:")
print("  ✓ F1-score")
print("  ✓ MCC")
print("  ✓ ROC-AUC")
print("  ✓ PR-AUC")

print()
print("Generated:")
print("  ✓ Best hyperparameters CSV")
print("  ✓ Complete CV search results CSV")
print("  ✓ Tuning summary CSV")
print("  ✓ Text report")

print()
print("Results directory:")
print(RESULTS_DIR)

print()
print("=" * 90)
print("NEXT STEP: 08_FINAL_MODEL_SELECTION.PY")
print("=" * 90)
