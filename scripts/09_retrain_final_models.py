"""
================================================================================
STEP 8: RETRAIN FINAL MODELS
================================================================================

Project:
Evaluating SHAP Explanation Consistency in Ensemble Models
for Software Defect Prediction

Dataset:
NASA JM1

Purpose:
Retrain the final selected model configurations on the ENTIRE development
dataset using SMOTE.

IMPORTANT:
- Development set only.
- Independent test set is NOT loaded.
- SMOTE is applied to the entire development set for final training.
- No test evaluation is performed.
- No SHAP analysis is performed.
- Model selection was completed in Step 7.

Final models:
1. Random Forest (RF)
2. XGBoost (XGB)
3. LightGBM (LGBM)
4. Multi-Layer Perceptron (MLP)

Random state:
42
================================================================================
"""

from pathlib import Path
import json
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEVELOPMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "JM1"
    / "JM1_development.csv"
)

SELECTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "performance"
    / "final_model_selection"
    / "final_model_selection.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "final"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "performance"
    / "final_models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

RANDOM_STATE = 42
TARGET = "defects"

SMOTE_K_NEIGHBORS = 5


# =============================================================================
# 3. VERIFIED HYPERPARAMETERS FROM STEP 7
# =============================================================================
#
# These are the exact best configurations obtained from your completed
# hyperparameter tuning experiment.
#
# RF:
# F1       = 0.4308
# MCC      = 0.2654
# ROC-AUC  = 0.6953
# PR-AUC   = 0.4266
#
# XGB:
# F1       = 0.4292
# MCC      = 0.2428
# ROC-AUC  = 0.6796
# PR-AUC   = 0.4118
#
# LGBM:
# F1       = 0.3938
# MCC      = 0.2681
# ROC-AUC  = 0.6896
# PR-AUC   = 0.4233
#
# MLP:
# F1       = 0.4204
# MCC      = 0.2282
# ROC-AUC  = 0.6574
# PR-AUC   = 0.3990
# =============================================================================

FINAL_HYPERPARAMETERS = {

    "RF": {
        "n_estimators": 300,
        "max_depth": 10,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "min_samples_split": 5
    },

    "XGB": {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },

    "LGBM": {
        "n_estimators": 200,
        "max_depth": 20,
        "learning_rate": 0.01,
        "num_leaves": 63,
        "min_child_samples": 30
    },

    "MLP": {
        "hidden_layer_sizes": (50,),
        "activation": "tanh",
        "alpha": 0.01,
        "learning_rate_init": 0.001
    }
}


# =============================================================================
# 4. HEADER
# =============================================================================

print("=" * 80)
print("STEP 8: RETRAIN FINAL MODELS")
print("=" * 80)

print()
print("Research project:")
print("Evaluating SHAP Explanation Consistency in Ensemble Models")
print("for Software Defect Prediction")

print()
print("Dataset: NASA JM1")

print()
print("TRAINING DESIGN")
print("-" * 80)

print("Development set : ENTIRE development set")
print("Training strategy: SMOTE")
print("Random state     :", RANDOM_STATE)
print("Test set         : NOT LOADED")
print("SHAP analysis    : NOT PERFORMED")
print("Model selection  : ALREADY COMPLETED")


# =============================================================================
# 5. VERIFY DEVELOPMENT DATASET
# =============================================================================

if not DEVELOPMENT_FILE.exists():

    raise FileNotFoundError(
        "\nERROR: Development dataset not found:\n"
        f"{DEVELOPMENT_FILE}"
    )


# =============================================================================
# 6. LOAD DEVELOPMENT DATA
# =============================================================================

print()
print("Loading development dataset:")
print(DEVELOPMENT_FILE)

data = pd.read_csv(
    DEVELOPMENT_FILE
)

print()
print(
    "Development dataset shape:",
    data.shape
)


# =============================================================================
# 7. VALIDATE DATASET
# =============================================================================

if TARGET not in data.columns:

    raise ValueError(
        f"Target column '{TARGET}' was not found."
    )


X = data.drop(
    columns=[TARGET]
)

y = data[TARGET].astype(int)


print()
print(
    "Feature matrix shape:",
    X.shape
)

print(
    "Target shape        :",
    y.shape
)


# =============================================================================
# 8. CHECK FEATURES
# =============================================================================

non_numeric_columns = X.select_dtypes(
    exclude=[np.number]
).columns.tolist()

if non_numeric_columns:

    raise ValueError(
        "Non-numeric feature columns detected:\n"
        + "\n".join(
            f" - {column}"
            for column in non_numeric_columns
        )
    )


missing_values = int(
    X.isna().sum().sum()
)

print()
print(
    "Total missing feature values:",
    missing_values
)

if missing_values > 0:

    raise ValueError(
        "Missing values detected in development data."
    )


# =============================================================================
# 9. ORIGINAL CLASS DISTRIBUTION
# =============================================================================

print()
print("Original development class distribution:")

original_distribution = (
    y.value_counts()
    .sort_index()
)

for label, count in original_distribution.items():

    print(
        f"Class {label}: {count}"
    )


print()
print("Original development class proportions:")

original_proportions = (
    y.value_counts(
        normalize=True
    )
    .sort_index()
)

for label, proportion in original_proportions.items():

    print(
        f"Class {label}: {proportion:.6f}"
    )


# =============================================================================
# 10. VERIFY STEP 7 MODEL SELECTION
# =============================================================================

if not SELECTION_FILE.exists():

    raise FileNotFoundError(
        "\nERROR: Step 7 final model selection file not found:\n"
        f"{SELECTION_FILE}"
    )


print()
print("Loading Step 7 final model selection:")

selection = pd.read_csv(
    SELECTION_FILE
)

print()
print(
    "Selected models:",
    selection["Model"].tolist()
)


expected_models = [
    "RF",
    "XGB",
    "LGBM",
    "MLP"
]


for model_name in expected_models:

    if model_name not in selection["Model"].values:

        raise ValueError(
            f"Selected model '{model_name}' "
            "was not found in Step 7 results."
        )


# =============================================================================
# 11. DISPLAY VERIFIED HYPERPARAMETERS
# =============================================================================

print()
print("=" * 80)
print("FINAL HYPERPARAMETERS")
print("=" * 80)


for model_name in expected_models:

    print()
    print(model_name)
    print("-" * 40)

    for parameter, value in FINAL_HYPERPARAMETERS[
        model_name
    ].items():

        print(
            f"{parameter}: {value}"
        )


# =============================================================================
# 12. APPLY SMOTE
# =============================================================================

print()
print("=" * 80)
print("APPLYING SMOTE")
print("=" * 80)

print()
print("SMOTE configuration:")
print("Random state:", RANDOM_STATE)
print("k_neighbors :", SMOTE_K_NEIGHBORS)


smote = SMOTE(
    random_state=RANDOM_STATE,
    k_neighbors=SMOTE_K_NEIGHBORS
)


X_resampled, y_resampled = smote.fit_resample(
    X,
    y
)


# =============================================================================
# 13. SMOTE RESULTS
# =============================================================================

print()
print("Before SMOTE:")
print(
    f"Samples: {len(X)}"
)

for label, count in original_distribution.items():

    print(
        f"Class {label}: {count}"
    )


resampled_distribution = (
    pd.Series(y_resampled)
    .value_counts()
    .sort_index()
)


print()
print("After SMOTE:")
print(
    f"Samples: {len(X_resampled)}"
)

for label, count in resampled_distribution.items():

    print(
        f"Class {label}: {count}"
    )


# =============================================================================
# 14. BUILD MODELS
# =============================================================================

def build_rf():

    params = FINAL_HYPERPARAMETERS["RF"].copy()

    params.update({
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    })

    return RandomForestClassifier(
        **params
    )


def build_xgb():

    params = FINAL_HYPERPARAMETERS["XGB"].copy()

    params.update({
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
        "n_jobs": -1,
        "verbosity": 0
    })

    return XGBClassifier(
        **params
    )


def build_lgbm():

    params = FINAL_HYPERPARAMETERS["LGBM"].copy()

    params.update({
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "n_jobs": -1
    })

    return LGBMClassifier(
        **params
    )


def build_mlp():

    params = FINAL_HYPERPARAMETERS["MLP"].copy()

    params.update({
        "random_state": RANDOM_STATE,
        "max_iter": 500
    })

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                MLPClassifier(
                    **params
                )
            )
        ]
    )


# =============================================================================
# 15. TRAIN FINAL MODELS
# =============================================================================

print()
print()
print("=" * 80)
print("TRAINING FINAL MODELS")
print("=" * 80)


model_builders = {
    "RF": build_rf,
    "XGB": build_xgb,
    "LGBM": build_lgbm,
    "MLP": build_mlp
}


trained_models = {}

training_records = []


for model_name in expected_models:

    print()
    print("-" * 80)
    print(
        f"TRAINING FINAL {model_name} MODEL"
    )
    print("-" * 80)

    print()
    print("Building model...")

    model = model_builders[
        model_name
    ]()

    print("Fitting on entire SMOTE-resampled development set...")

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        model.fit(
            X_resampled,
            y_resampled
        )

    trained_models[
        model_name
    ] = model

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------

    model_file = (
        MODEL_DIR
        / f"{model_name.lower()}_final.pkl"
    )

    with open(
        model_file,
        "wb"
    ) as file:

        pickle.dump(
            model,
            file
        )

    print()
    print(
        "SUCCESS: Model trained and saved."
    )

    print(
        "File:",
        model_file
    )

    training_records.append({
        "Model": model_name,
        "Training_Strategy": "SMOTE",
        "Original_Development_Samples": len(X),
        "SMOTE_Development_Samples": len(X_resampled),
        "Class_0_Before_SMOTE": int(
            original_distribution.get(0, 0)
        ),
        "Class_1_Before_SMOTE": int(
            original_distribution.get(1, 0)
        ),
        "Class_0_After_SMOTE": int(
            resampled_distribution.get(0, 0)
        ),
        "Class_1_After_SMOTE": int(
            resampled_distribution.get(1, 0)
        ),
        "Random_State": RANDOM_STATE,
        "Test_Set_Used": False,
        "SHAP_Analysis_Performed": False
    })


# =============================================================================
# 16. SAVE TRAINING REGISTRY
# =============================================================================

training_registry = pd.DataFrame(
    training_records
)

registry_file = (
    RESULT_DIR
    / "final_model_training_registry.csv"
)

training_registry.to_csv(
    registry_file,
    index=False
)


# =============================================================================
# 17. SAVE HYPERPARAMETER CONFIGURATION
# =============================================================================

configuration_records = []

for model_name in expected_models:

    record = {
        "Model": model_name,
        "Training_Strategy": "SMOTE",
        "Random_State": RANDOM_STATE,
        "Test_Set_Used": False
    }

    for parameter, value in FINAL_HYPERPARAMETERS[
        model_name
    ].items():

        record[
            parameter
        ] = str(value)

    configuration_records.append(
        record
    )


configuration_df = pd.DataFrame(
    configuration_records
)

configuration_file = (
    RESULT_DIR
    / "final_model_configurations.csv"
)

configuration_df.to_csv(
    configuration_file,
    index=False
)


# =============================================================================
# 18. SAVE SMOTE INFORMATION
# =============================================================================

smote_information = {
    "training_dataset": "JM1_development",
    "original_samples": int(len(X)),
    "resampled_samples": int(len(X_resampled)),
    "original_class_0": int(
        original_distribution.get(0, 0)
    ),
    "original_class_1": int(
        original_distribution.get(1, 0)
    ),
    "resampled_class_0": int(
        resampled_distribution.get(0, 0)
    ),
    "resampled_class_1": int(
        resampled_distribution.get(1, 0)
    ),
    "random_state": RANDOM_STATE,
    "k_neighbors": SMOTE_K_NEIGHBORS
}


smote_file = (
    RESULT_DIR
    / "smote_training_information.json"
)

with open(
    smote_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        smote_information,
        file,
        indent=4
    )


# =============================================================================
# 19. SAVE MODEL MANIFEST
# =============================================================================

manifest = {
    "project": (
        "Evaluating SHAP Explanation Consistency "
        "in Ensemble Models for Software Defect Prediction"
    ),

    "dataset": "NASA JM1",

    "target": TARGET,

    "training_dataset": "JM1_development",

    "development_samples": int(len(X)),

    "smote_samples": int(len(X_resampled)),

    "training_strategy": "SMOTE",

    "random_state": RANDOM_STATE,

    "test_set_used": False,

    "shap_analysis_performed": False,

    "models": {
        model_name: {
            "file": (
                str(
                    MODEL_DIR
                    / f"{model_name.lower()}_final.pkl"
                )
            ),
            "hyperparameters": {
                key: str(value)
                for key, value in FINAL_HYPERPARAMETERS[
                    model_name
                ].items()
            }
        }
        for model_name in expected_models
    }
}


manifest_file = (
    MODEL_DIR
    / "final_model_manifest.json"
)

with open(
    manifest_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        manifest,
        file,
        indent=4
    )


# =============================================================================
# 20. FINAL OUTPUT
# =============================================================================

print()
print()
print("=" * 80)
print("STEP 8 COMPLETED SUCCESSFULLY")
print("=" * 80)

print()
print("Final models:")

for model_name in expected_models:

    print(
        f"  {model_name:<5} -> "
        f"{model_name.lower()}_final.pkl"
    )


print()
print("Training data:")
print(
    f"  Original development set : {len(X)}"
)

print(
    f"  After SMOTE              : {len(X_resampled)}"
)


print()
print("Class distribution after SMOTE:")

for label, count in resampled_distribution.items():

    print(
        f"  Class {label}: {count}"
    )


print()
print("Saved model directory:")
print(
    MODEL_DIR
)


print()
print("Saved result directory:")
print(
    RESULT_DIR
)


print()
print("Registry:")
print(
    registry_file
)


print()
print("Configuration:")
print(
    configuration_file
)


print()
print("SMOTE information:")
print(
    smote_file
)


print()
print("Manifest:")
print(
    manifest_file
)


# =============================================================================
# 21. TEST SET PROTECTION
# =============================================================================

print()
print("=" * 80)
print("TEST SET PROTECTION CHECK")
print("=" * 80)

print()
print("Independent test set: NOT LOADED")
print("Independent test set: NOT USED")
print("Independent test set: STILL LOCKED")

print()
print("=" * 80)
print("NEXT STEP: STEP 9 - INDEPENDENT TEST EVALUATION")
print("=" * 80)
