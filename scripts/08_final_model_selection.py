"""
================================================================================
STEP 7: FINAL MODEL SELECTION
================================================================================

Project:
Evaluating SHAP Explanation Consistency in Ensemble Models
for Software Defect Prediction

Dataset:
NASA JM1

Purpose:
Select the final tuned configuration for each model family based on
development-set cross-validation performance.

Models:
- Random Forest (RF)
- XGBoost (XGB)
- LightGBM (LGBM)
- Multi-Layer Perceptron (MLP)

Selection hierarchy:
1. F1       -> Primary criterion
2. MCC      -> Secondary criterion
3. PR-AUC   -> Supporting criterion
4. ROC-AUC  -> Supporting criterion

IMPORTANT:
- Development set only
- Test set remains completely locked
- No test-set information is used
- No retraining is performed
- No SHAP analysis is performed

Next step:
Step 8 -> Retrain selected final models on the entire development set.
================================================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "performance"
    / "hyperparameter_tuning"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "performance"
    / "final_model_selection"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. INPUT FILE
# =============================================================================

TUNING_SUMMARY_FILE = (
    RESULTS_DIR
    / "hyperparameter_tuning_summary.csv"
)


# =============================================================================
# 3. CONSOLE HEADER
# =============================================================================

print("=" * 80)
print("STEP 7: FINAL MODEL SELECTION")
print("=" * 80)

print()
print("Research project:")
print("Evaluating SHAP Explanation Consistency in Ensemble Models")
print("for Software Defect Prediction")

print()
print("Dataset: NASA JM1")

print()
print("MODEL SELECTION DESIGN")
print("-" * 80)
print("Data used        : Development set only")
print("Cross-validation : Stratified 5-fold CV")
print("Training strategy: SMOTE inside CV training folds")
print("Test set         : LOCKED / NOT USED")
print("Retraining       : NOT performed in this step")

print()
print("Selection hierarchy:")
print("1. F1       -> Primary criterion")
print("2. MCC      -> Secondary criterion")
print("3. PR-AUC   -> Supporting criterion")
print("4. ROC-AUC  -> Supporting criterion")

print()
print("NOTE:")
print("Accuracy is not available in the Step 6 tuning summary.")
print("It will be reported during independent test evaluation in Step 9.")


# =============================================================================
# 4. CHECK INPUT DIRECTORY
# =============================================================================

if not RESULTS_DIR.exists():

    raise FileNotFoundError(
        "\nERROR: Hyperparameter tuning results directory not found:\n"
        f"{RESULTS_DIR}\n\n"
        "Please run Step 6 hyperparameter tuning first."
    )


# =============================================================================
# 5. CHECK INPUT FILE
# =============================================================================

if not TUNING_SUMMARY_FILE.exists():

    available_files = list(RESULTS_DIR.glob("*.csv"))

    print()
    print("Available CSV files:")

    for file in available_files:
        print(" -", file.name)

    raise FileNotFoundError(
        "\nERROR: Required tuning summary file was not found:\n"
        f"{TUNING_SUMMARY_FILE}"
    )


# =============================================================================
# 6. LOAD TUNING SUMMARY
# =============================================================================

print()
print("Loading tuning summary:")
print(TUNING_SUMMARY_FILE)

summary = pd.read_csv(TUNING_SUMMARY_FILE)

print()
print("Tuning summary shape:", summary.shape)

print()
print("Columns found:")

for column in summary.columns:
    print(" -", column)


# =============================================================================
# 7. EXPECTED COLUMNS
# =============================================================================

required_columns = [
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

missing_columns = [
    column
    for column in required_columns
    if column not in summary.columns
]

if missing_columns:

    raise ValueError(
        "\nERROR: The following expected columns are missing:\n"
        + "\n".join(f" - {column}" for column in missing_columns)
    )


# =============================================================================
# 8. STANDARDIZE MODEL NAMES
# =============================================================================

def normalize_model_name(name):

    text = str(name).strip().lower()

    if text in [
        "rf",
        "random forest",
        "randomforest",
        "random_forest"
    ]:
        return "RF"

    if text in [
        "xgb",
        "xgboost",
        "xg boost"
    ]:
        return "XGB"

    if text in [
        "lgbm",
        "lightgbm",
        "light gbm"
    ]:
        return "LGBM"

    if text in [
        "mlp",
        "neural network",
        "neural_network"
    ]:
        return "MLP"

    return str(name).strip()


summary["Model_Standardized"] = (
    summary["model"].apply(normalize_model_name)
)


# =============================================================================
# 9. CONVERT PERFORMANCE VALUES TO NUMERIC
# =============================================================================

performance_columns = [
    "best_f1_mean",
    "best_f1_std",
    "best_mcc_mean",
    "best_mcc_std",
    "best_roc_auc_mean",
    "best_roc_auc_std",
    "best_pr_auc_mean",
    "best_pr_auc_std"
]

for column in performance_columns:

    summary[column] = pd.to_numeric(
        summary[column],
        errors="coerce"
    )


# =============================================================================
# 10. VALIDATE MODEL FAMILIES
# =============================================================================

expected_models = [
    "RF",
    "XGB",
    "LGBM",
    "MLP"
]

available_models = (
    summary["Model_Standardized"]
    .unique()
    .tolist()
)

print()
print("Available model families:")

for model in available_models:
    print(" -", model)


missing_models = [
    model
    for model in expected_models
    if model not in available_models
]

if missing_models:

    raise ValueError(
        "\nERROR: Expected model families are missing:\n"
        + "\n".join(f" - {model}" for model in missing_models)
    )


# =============================================================================
# 11. SELECT FINAL CONFIGURATION FOR EACH MODEL
# =============================================================================

print()
print("=" * 80)
print("SELECTING FINAL CONFIGURATION FOR EACH MODEL")
print("=" * 80)


selected_rows = []


for model in expected_models:

    model_data = summary[
        summary["Model_Standardized"] == model
    ].copy()

    if model_data.empty:

        raise ValueError(
            f"No tuning result found for model: {model}"
        )

    # -------------------------------------------------------------------------
    # Selection hierarchy
    # -------------------------------------------------------------------------
    #
    # Primary   : F1
    # Secondary : MCC
    # Third     : PR-AUC
    # Fourth    : ROC-AUC
    #
    # Accuracy is unavailable in the tuning summary.
    # -------------------------------------------------------------------------

    model_data = model_data.sort_values(
        by=[
            "best_f1_mean",
            "best_mcc_mean",
            "best_pr_auc_mean",
            "best_roc_auc_mean"
        ],
        ascending=[
            False,
            False,
            False,
            False
        ]
    )

    best_row = model_data.iloc[0].copy()

    selected_rows.append(best_row)

    print()
    print(f"MODEL: {model}")
    print("-" * 50)

    print(
        f"F1       : "
        f"{best_row['best_f1_mean']:.4f} "
        f"+/- {best_row['best_f1_std']:.4f}"
    )

    print(
        f"MCC      : "
        f"{best_row['best_mcc_mean']:.4f} "
        f"+/- {best_row['best_mcc_std']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{best_row['best_roc_auc_mean']:.4f} "
        f"+/- {best_row['best_roc_auc_std']:.4f}"
    )

    print(
        f"PR-AUC   : "
        f"{best_row['best_pr_auc_mean']:.4f} "
        f"+/- {best_row['best_pr_auc_std']:.4f}"
    )


# =============================================================================
# 12. CREATE SELECTION DATAFRAME
# =============================================================================

selected = pd.DataFrame(
    selected_rows
).reset_index(drop=True)


# =============================================================================
# 13. OVERALL RANKING
# =============================================================================

selected = selected.sort_values(
    by=[
        "best_f1_mean",
        "best_mcc_mean",
        "best_pr_auc_mean",
        "best_roc_auc_mean"
    ],
    ascending=[
        False,
        False,
        False,
        False
    ]
).reset_index(drop=True)


selected["Overall_Rank"] = (
    np.arange(len(selected)) + 1
)


# =============================================================================
# 14. ADD METHODOLOGICAL METADATA
# =============================================================================

selected["Selected_Training_Strategy"] = "SMOTE"

selected["Selection_Primary_Criterion"] = "F1"

selected["Selection_Secondary_Criterion"] = "MCC"

selected["Test_Set_Used"] = False

selected["Retraining_Performed"] = False

selected["Final_Model_Selected"] = True


# =============================================================================
# 15. CREATE CLEAN FINAL SELECTION TABLE
# =============================================================================

final_selection = selected[
    [
        "Overall_Rank",
        "Model_Standardized",

        "best_f1_mean",
        "best_f1_std",

        "best_mcc_mean",
        "best_mcc_std",

        "best_roc_auc_mean",
        "best_roc_auc_std",

        "best_pr_auc_mean",
        "best_pr_auc_std",

        "Selected_Training_Strategy",

        "Selection_Primary_Criterion",
        "Selection_Secondary_Criterion",

        "Test_Set_Used",
        "Retraining_Performed",
        "Final_Model_Selected"
    ]
].copy()


# =============================================================================
# 16. RENAME COLUMNS FOR READABILITY
# =============================================================================

final_selection = final_selection.rename(
    columns={
        "Model_Standardized": "Model",

        "best_f1_mean": "F1_Mean",
        "best_f1_std": "F1_STD",

        "best_mcc_mean": "MCC_Mean",
        "best_mcc_std": "MCC_STD",

        "best_roc_auc_mean": "ROC_AUC_Mean",
        "best_roc_auc_std": "ROC_AUC_STD",

        "best_pr_auc_mean": "PR_AUC_Mean",
        "best_pr_auc_std": "PR_AUC_STD"
    }
)


# =============================================================================
# 17. SAVE FINAL MODEL SELECTION
# =============================================================================

selection_csv = (
    OUTPUT_DIR
    / "final_model_selection.csv"
)

final_selection.to_csv(
    selection_csv,
    index=False
)


# =============================================================================
# 18. CREATE FINAL MODEL REGISTRY
# =============================================================================

registry = final_selection[
    [
        "Overall_Rank",
        "Model",
        "F1_Mean",
        "F1_STD",
        "MCC_Mean",
        "MCC_STD",
        "ROC_AUC_Mean",
        "ROC_AUC_STD",
        "PR_AUC_Mean",
        "PR_AUC_STD",
        "Selected_Training_Strategy"
    ]
].copy()


registry_csv = (
    OUTPUT_DIR
    / "final_model_registry.csv"
)

registry.to_csv(
    registry_csv,
    index=False
)


# =============================================================================
# 19. CREATE TEXT REPORT
# =============================================================================

report_file = (
    OUTPUT_DIR
    / "final_model_selection_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as file:

    file.write("=" * 80 + "\n")
    file.write("STEP 7: FINAL MODEL SELECTION REPORT\n")
    file.write("=" * 80 + "\n\n")

    file.write(
        "Project: Evaluating SHAP Explanation Consistency "
        "in Ensemble Models for Software Defect Prediction\n"
    )

    file.write(
        "Dataset: NASA JM1\n\n"
    )

    file.write(
        "EXPERIMENTAL DESIGN\n"
    )

    file.write(
        "-" * 80 + "\n"
    )

    file.write(
        "Data used: Development set only\n"
    )

    file.write(
        "Cross-validation: Stratified 5-fold CV\n"
    )

    file.write(
        "Training strategy: SMOTE applied within training folds\n"
    )

    file.write(
        "Independent test set: NOT USED\n"
    )

    file.write(
        "Retraining: NOT performed in Step 7\n\n"
    )

    file.write(
        "MODEL SELECTION CRITERIA\n"
    )

    file.write(
        "-" * 80 + "\n"
    )

    file.write(
        "Primary criterion: F1\n"
    )

    file.write(
        "Secondary criterion: MCC\n"
    )

    file.write(
        "Supporting criterion: PR-AUC\n"
    )

    file.write(
        "Supporting criterion: ROC-AUC\n"
    )

    file.write(
        "Accuracy: Not available in Step 6 tuning summary\n\n"
    )

    file.write(
        "SELECTED FINAL MODELS\n"
    )

    file.write(
        "-" * 80 + "\n\n"
    )

    for _, row in final_selection.iterrows():

        file.write(
            f"Rank {int(row['Overall_Rank'])}: "
            f"{row['Model']}\n"
        )

        file.write(
            f"  F1       = "
            f"{row['F1_Mean']:.4f} "
            f"+/- {row['F1_STD']:.4f}\n"
        )

        file.write(
            f"  MCC      = "
            f"{row['MCC_Mean']:.4f} "
            f"+/- {row['MCC_STD']:.4f}\n"
        )

        file.write(
            f"  ROC-AUC  = "
            f"{row['ROC_AUC_Mean']:.4f} "
            f"+/- {row['ROC_AUC_STD']:.4f}\n"
        )

        file.write(
            f"  PR-AUC   = "
            f"{row['PR_AUC_Mean']:.4f} "
            f"+/- {row['PR_AUC_STD']:.4f}\n"
        )

        file.write(
            "  Training strategy = SMOTE\n"
        )

        file.write(
            "  Test set used = NO\n\n"
        )

    file.write(
        "=" * 80 + "\n"
    )

    file.write(
        "METHODOLOGICAL NOTE\n"
    )

    file.write(
        "=" * 80 + "\n"
    )

    file.write(
        "Four final model families are retained for the subsequent "
        "comparative SHAP analysis.\n"
    )

    file.write(
        "The model with the highest overall predictive ranking is "
        "not used to eliminate the other model families because "
        "cross-model explanation consistency is a central objective "
        "of the research.\n"
    )

    file.write(
        "The independent test set remained completely locked "
        "throughout model selection.\n"
    )

    file.write(
        "The selected configurations will be retrained on the "
        "entire development set in Step 8.\n"
    )


# =============================================================================
# 20. FINAL CONSOLE SUMMARY
# =============================================================================

print()
print()
print("=" * 80)
print("FINAL MODEL SELECTION COMPLETE")
print("=" * 80)

print()
print("Final model ranking:")
print("-" * 80)

for _, row in final_selection.iterrows():

    print(
        f"Rank {int(row['Overall_Rank'])}: "
        f"{row['Model']:<5} | "
        f"F1={row['F1_Mean']:.4f} | "
        f"MCC={row['MCC_Mean']:.4f} | "
        f"ROC-AUC={row['ROC_AUC_Mean']:.4f} | "
        f"PR-AUC={row['PR_AUC_Mean']:.4f}"
    )


# =============================================================================
# 21. IDENTIFY BEST OVERALL PREDICTIVE MODEL
# =============================================================================

best_overall = final_selection.iloc[0]

print()
print("=" * 80)
print("BEST OVERALL PREDICTIVE MODEL")
print("=" * 80)

print(
    f"Model    : {best_overall['Model']}"
)

print(
    f"F1       : {best_overall['F1_Mean']:.4f}"
)

print(
    f"MCC      : {best_overall['MCC_Mean']:.4f}"
)

print(
    f"ROC-AUC  : {best_overall['ROC_AUC_Mean']:.4f}"
)

print(
    f"PR-AUC   : {best_overall['PR_AUC_Mean']:.4f}"
)


# =============================================================================
# 22. OUTPUT FILES
# =============================================================================

print()
print()
print("Output files:")
print("-" * 80)

print(selection_csv)
print(registry_csv)
print(report_file)


# =============================================================================
# 23. TEST SET STATUS
# =============================================================================

print()
print("=" * 80)
print("TEST SET STATUS: LOCKED")
print("=" * 80)

print()
print("No test-set data were loaded or used.")
print("No retraining was performed.")
print("No SHAP analysis was performed.")

print()
print("Step 7 finished successfully.")

print()
print("NEXT STEP:")
print("Step 8 -> Retrain final selected models")
print("on the ENTIRE development set.")

print("=" * 80)
