"""
10_independent_test_evaluation.py

JM1 Independent Test Evaluation
================================

Purpose
-------
Evaluate the four final models trained in Step 8 on the locked,
independent 20% JM1 test set.

Final models:
    - Random Forest (RF)
    - XGBoost (XGB)
    - LightGBM (LGBM)
    - Multi-Layer Perceptron (MLP)

Important methodological rules
------------------------------
1. The independent test set is NEVER used for training.
2. SMOTE is NOT applied to the test set.
3. No hyperparameter tuning is performed.
4. No threshold tuning is performed.
5. Final models are loaded from disk exactly as trained.
6. Classification threshold remains the default 0.50.
7. Test predictions and probabilities are saved for later SHAP analysis.
"""

from pathlib import Path
import pickle
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PROJECT CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(
    r"E:\Programming\Evaluating SHAP Explanation Consistency in Ensemble Models for Software Defect Prediction"
)

TARGET_COLUMN = "defects"

RANDOM_STATE = 42

# -------------------------------------------------------------------------
# IMPORTANT:
# The development dataset used earlier was:
#
# data\processed\JM1\JM1_development.csv
#
# Therefore the primary expected test location is:
#
# data\processed\JM1\JM1_test.csv
# -------------------------------------------------------------------------

TEST_FILE_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "JM1" / "JM1_test.csv",
    PROJECT_ROOT / "data" / "processed" / "JM1_test.csv",
]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "final"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "performance"
    / "independent_test"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "figures"
    / "performance"
    / "independent_test"
)


# Final model files created by Step 8
MODEL_FILES = {
    "RF": "rf_final.pkl",
    "XGB": "xgb_final.pkl",
    "LGBM": "lgbm_final.pkl",
    "MLP": "mlp_final.pkl",
}


# =============================================================================
# 2. EXPECTED LOCKED TEST-SET INTEGRITY
# =============================================================================
#
# These values come from the locked 80/20 stratified split:
#
# Total test samples = 1,782
# Class 0 = 1,381
# Class 1 =   401
#
# If these checks fail, the script stops rather than silently evaluating
# the wrong dataset.
# =============================================================================

EXPECTED_TEST_ROWS = 1782
EXPECTED_CLASS_0 = 1381
EXPECTED_CLASS_1 = 401
EXPECTED_FEATURE_COUNT = 21


# =============================================================================
# 3. CREATE OUTPUT DIRECTORIES
# =============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

def locate_test_file():
    """
    Locate the existing locked JM1 test dataset.

    The preferred location is:
        data/processed/JM1/JM1_test.csv

    A fallback location is checked only if the preferred location
    does not exist.
    """

    print("\nSearching for locked JM1 test dataset...")

    for candidate in TEST_FILE_CANDIDATES:

        print(f"Checking:")
        print(f"    {candidate}")

        if candidate.exists():

            print("\nTest dataset found:")
            print(candidate)

            return candidate

    # ---------------------------------------------------------------------
    # If the known candidate locations fail, search ONLY within
    # data/processed for an exact filename match.
    #
    # This prevents accidentally selecting an unrelated CSV.
    # ---------------------------------------------------------------------

    processed_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
    )

    exact_matches = list(
        processed_dir.rglob("JM1_test.csv")
    )

    if len(exact_matches) == 1:

        print("\nTest dataset found by exact filename search:")
        print(exact_matches[0])

        return exact_matches[0]

    if len(exact_matches) > 1:

        print("\nMultiple JM1_test.csv files were found:")

        for path in exact_matches:
            print(f"    {path}")

        raise RuntimeError(
            "\nMultiple files named JM1_test.csv were found. "
            "Please keep only the intended locked test set."
        )

    raise FileNotFoundError(
        "\nCould not find JM1_test.csv.\n\n"
        "Expected locations include:\n"
        f"    {TEST_FILE_CANDIDATES[0]}\n"
        f"    {TEST_FILE_CANDIDATES[1]}\n\n"
        "Please verify that the locked test dataset exists."
    )


def load_pickle_model(model_path):
    """
    Load a previously trained model.
    """

    if not model_path.exists():

        raise FileNotFoundError(
            f"\nFinal model not found:\n{model_path}\n\n"
            "Please verify that Step 8 final model retraining "
            "completed successfully."
        )

    with open(
        model_path,
        "rb"
    ) as file:

        model = pickle.load(file)

    return model


def get_expected_features(model):
    """
    Retrieve the feature names recorded by the trained estimator/pipeline.

    This helps guarantee that the test features are supplied to the
    model in exactly the same order as during training.
    """

    if hasattr(
        model,
        "feature_names_in_"
    ):

        return list(
            model.feature_names_in_
        )

    return None


def align_test_features(
    model,
    model_name,
    X_test
):
    """
    Align test features to the feature order used during training.
    """

    expected_features = get_expected_features(
        model
    )

    # ---------------------------------------------------------------------
    # If the model stores feature names, perform strict feature checking.
    # ---------------------------------------------------------------------

    if expected_features is not None:

        missing_features = [
            feature
            for feature in expected_features
            if feature not in X_test.columns
        ]

        if missing_features:

            raise ValueError(
                f"\n{model_name}: Missing test features:\n"
                f"{missing_features}"
            )

        # Ignore any unexpected non-feature columns.
        extra_features = [
            feature
            for feature in X_test.columns
            if feature not in expected_features
        ]

        if extra_features:

            print(
                f"\nWARNING: {model_name} found extra test columns:"
            )

            for feature in extra_features:
                print(f"    {feature}")

        # Reorder exactly as training
        X_model = X_test[
            expected_features
        ].copy()

        return X_model

    # ---------------------------------------------------------------------
    # Fallback:
    # If the model does not store feature names, verify the number of
    # features and preserve the current dataset order.
    # ---------------------------------------------------------------------

    print(
        f"\nWARNING: {model_name} does not expose "
        "feature_names_in_."
    )

    if X_test.shape[1] != EXPECTED_FEATURE_COUNT:

        raise ValueError(
            f"\n{model_name}: Unexpected number of test features.\n"
            f"Expected: {EXPECTED_FEATURE_COUNT}\n"
            f"Found: {X_test.shape[1]}"
        )

    return X_test.copy()


def evaluate_model(
    model,
    model_name,
    X_test,
    y_test
):
    """
    Evaluate one final model on the independent test set.
    """

    print("\n" + "=" * 70)
    print(f"EVALUATING MODEL: {model_name}")
    print("=" * 70)

    # ---------------------------------------------------------------------
    # Align features
    # ---------------------------------------------------------------------

    X_model = align_test_features(
        model=model,
        model_name=model_name,
        X_test=X_test
    )

    print(
        f"\nFeatures supplied to {model_name}: "
        f"{X_model.shape[1]}"
    )

    # ---------------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------------

    y_pred = model.predict(
        X_model
    )

    # ---------------------------------------------------------------------
    # Probability prediction
    # ---------------------------------------------------------------------

    if not hasattr(
        model,
        "predict_proba"
    ):

        raise AttributeError(
            f"{model_name} does not provide predict_proba()."
        )

    y_proba = model.predict_proba(
        X_model
    )[:, 1]

    # ---------------------------------------------------------------------
    # Verify probability values
    # ---------------------------------------------------------------------

    if not np.isfinite(
        y_proba
    ).all():

        raise ValueError(
            f"{model_name} produced invalid probability values."
        )

    # ---------------------------------------------------------------------
    # Classification metrics
    # ---------------------------------------------------------------------

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

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_proba
    )

    pr_auc = average_precision_score(
        y_test,
        y_proba
    )

    # ---------------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # ---------------------------------------------------------------------
    # Specificity
    # ---------------------------------------------------------------------

    if (tn + fp) > 0:

        specificity = (
            tn / (tn + fp)
        )

    else:

        specificity = 0.0

    # ---------------------------------------------------------------------
    # Result dictionary
    # ---------------------------------------------------------------------

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "MCC": mcc,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    # ---------------------------------------------------------------------
    # Display
    # ---------------------------------------------------------------------

    print(
        f"\nAccuracy    : {accuracy:.4f}"
    )

    print(
        f"Precision   : {precision:.4f}"
    )

    print(
        f"Recall      : {recall:.4f}"
    )

    print(
        f"Specificity : {specificity:.4f}"
    )

    print(
        f"F1          : {f1:.4f}"
    )

    print(
        f"MCC         : {mcc:.4f}"
    )

    print(
        f"ROC-AUC     : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC      : {pr_auc:.4f}"
    )

    print("\nConfusion Matrix:")
    print(
        cm
    )

    print(
        "\nTN:", tn
    )
    print(
        "FP:", fp
    )
    print(
        "FN:", fn
    )
    print(
        "TP:", tp
    )

    return (
        metrics,
        y_pred,
        y_proba
    )


# =============================================================================
# 5. MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("JM1 INDEPENDENT TEST EVALUATION")
    print("=" * 80)

    print(
        "\nProject root:"
    )
    print(
        PROJECT_ROOT
    )

    print(
        "\nExperiment design:"
    )
    print(
        "Independent locked 20% test set"
    )
    print(
        "No SMOTE"
    )
    print(
        "No retraining"
    )
    print(
        "No hyperparameter tuning"
    )
    print(
        "No threshold tuning"
    )
    print(
        "Classification threshold = 0.50"
    )

    # =========================================================================
    # STEP 1 — LOCATE TEST DATASET
    # =========================================================================

    print("\n" + "-" * 80)
    print("1. LOCATING LOCKED TEST SET")
    print("-" * 80)

    test_file = locate_test_file()

    print(
        "\nUsing test dataset:"
    )
    print(
        test_file
    )

    # =========================================================================
    # STEP 2 — LOAD TEST DATASET
    # =========================================================================

    print("\n" + "-" * 80)
    print("2. LOADING LOCKED TEST SET")
    print("-" * 80)

    test_df = pd.read_csv(
        test_file
    )

    print(
        f"\nTest shape: {test_df.shape}"
    )

    # =========================================================================
    # STEP 3 — VALIDATE DATASET STRUCTURE
    # =========================================================================

    print("\n" + "-" * 80)
    print("3. VALIDATING TEST SET")
    print("-" * 80)

    # Target existence
    if TARGET_COLUMN not in test_df.columns:

        raise ValueError(
            f"\nTarget column '{TARGET_COLUMN}' "
            "was not found in the test dataset."
        )

    # Separate features and target
    X_test = test_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_test = test_df[
        TARGET_COLUMN
    ]

    print(
        f"\nTest samples : {len(test_df)}"
    )

    print(
        f"Test features: {X_test.shape[1]}"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        y_test.value_counts(
            dropna=False
        ).sort_index()
    )

    # -------------------------------------------------------------------------
    # Check target values
    # -------------------------------------------------------------------------

    unique_targets = sorted(
        y_test.dropna().unique().tolist()
    )

    print(
        "\nUnique target values:"
    )
    print(
        unique_targets
    )

    if unique_targets != [0, 1]:

        raise ValueError(
            "\nUnexpected target values. "
            "Expected binary labels [0, 1]."
        )

    # -------------------------------------------------------------------------
    # Check missing target values
    # -------------------------------------------------------------------------

    if y_test.isnull().any():

        raise ValueError(
            "\nTarget column contains missing values."
        )

    # -------------------------------------------------------------------------
    # Check missing feature values
    # -------------------------------------------------------------------------

    total_missing = int(
        X_test.isnull().sum().sum()
    )

    print(
        f"\nTotal missing feature values: "
        f"{total_missing}"
    )

    if total_missing != 0:

        raise ValueError(
            "\nTest set contains missing feature values."
        )

    # -------------------------------------------------------------------------
    # Check feature count
    # -------------------------------------------------------------------------

    if X_test.shape[1] != EXPECTED_FEATURE_COUNT:

        raise ValueError(
            "\nUnexpected number of features.\n"
            f"Expected: {EXPECTED_FEATURE_COUNT}\n"
            f"Found: {X_test.shape[1]}"
        )

    # -------------------------------------------------------------------------
    # Check locked test size
    # -------------------------------------------------------------------------

    if len(test_df) != EXPECTED_TEST_ROWS:

        raise ValueError(
            "\nUnexpected test-set size.\n"
            f"Expected: {EXPECTED_TEST_ROWS}\n"
            f"Found: {len(test_df)}"
        )

    # -------------------------------------------------------------------------
    # Check class distribution
    # -------------------------------------------------------------------------

    class_counts = (
        y_test
        .value_counts()
        .to_dict()
    )

    actual_class_0 = int(
        class_counts.get(
            0,
            0
        )
    )

    actual_class_1 = int(
        class_counts.get(
            1,
            0
        )
    )

    if actual_class_0 != EXPECTED_CLASS_0:

        raise ValueError(
            "\nUnexpected class-0 count.\n"
            f"Expected: {EXPECTED_CLASS_0}\n"
            f"Found: {actual_class_0}"
        )

    if actual_class_1 != EXPECTED_CLASS_1:

        raise ValueError(
            "\nUnexpected class-1 count.\n"
            f"Expected: {EXPECTED_CLASS_1}\n"
            f"Found: {actual_class_1}"
        )

    print(
        "\nTest-set integrity check: PASSED"
    )

    print(
        "\nVerified:"
    )

    print(
        f"    Samples   : {EXPECTED_TEST_ROWS}"
    )

    print(
        f"    Features  : {EXPECTED_FEATURE_COUNT}"
    )

    print(
        f"    Class 0   : {EXPECTED_CLASS_0}"
    )

    print(
        f"    Class 1   : {EXPECTED_CLASS_1}"
    )

    # =========================================================================
    # STEP 4 — LOAD FINAL MODELS
    # =========================================================================

    print("\n" + "-" * 80)
    print("4. LOADING FINAL TRAINED MODELS")
    print("-" * 80)

    models = {}

    for model_name, filename in MODEL_FILES.items():

        model_path = (
            MODEL_DIR
            / filename
        )

        print(
            f"\nLoading {model_name}:"
        )

        print(
            model_path
        )

        model = load_pickle_model(
            model_path
        )

        models[
            model_name
        ] = model

        print(
            f"{model_name} loaded successfully."
        )

    print(
        "\nAll four final models loaded successfully."
    )

    # =========================================================================
    # STEP 5 — EVALUATE ALL MODELS
    # =========================================================================

    print("\n" + "-" * 80)
    print("5. INDEPENDENT TEST EVALUATION")
    print("-" * 80)

    all_metrics = []

    # -------------------------------------------------------------------------
    # This dataframe will later be used by SHAP/statistical analysis.
    # -------------------------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "y_true": y_test.to_numpy()
        }
    )

    # -------------------------------------------------------------------------
    # ROC / PR curve data
    # -------------------------------------------------------------------------

    roc_curve_data = {}
    pr_curve_data = {}

    # -------------------------------------------------------------------------
    # Evaluate models
    # -------------------------------------------------------------------------

    for model_name, model in models.items():

        (
            metrics,
            y_pred,
            y_proba
        ) = evaluate_model(
            model=model,
            model_name=model_name,
            X_test=X_test,
            y_test=y_test
        )

        all_metrics.append(
            metrics
        )

        # -------------------------------------------------------------
        # Save prediction and probability
        # -------------------------------------------------------------

        prediction_df[
            f"{model_name}_pred"
        ] = y_pred

        prediction_df[
            f"{model_name}_proba"
        ] = y_proba

        # -------------------------------------------------------------
        # ROC curve
        # -------------------------------------------------------------

        fpr, tpr, _ = roc_curve(
            y_test,
            y_proba
        )

        roc_curve_data[
            model_name
        ] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": metrics["ROC_AUC"]
        }

        # -------------------------------------------------------------
        # Precision-recall curve
        # -------------------------------------------------------------

        precision_curve, recall_curve, _ = (
            precision_recall_curve(
                y_test,
                y_proba
            )
        )

        pr_curve_data[
            model_name
        ] = {
            "precision": precision_curve,
            "recall": recall_curve,
            "auc": metrics["PR_AUC"]
        }

    # =========================================================================
    # STEP 6 — PERFORMANCE RESULTS DATAFRAME
    # =========================================================================

    print("\n" + "-" * 80)
    print("6. CREATING PERFORMANCE RESULTS")
    print("-" * 80)

    results_df = pd.DataFrame(
        all_metrics
    )

    # -------------------------------------------------------------------------
    # Descriptive ranking by F1
    #
    # IMPORTANT:
    # This does NOT change the previously selected models.
    # It is only for reporting the independent-test results.
    # -------------------------------------------------------------------------

    results_df = (
        results_df
        .sort_values(
            by="F1",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    results_df.insert(
        0,
        "Rank_by_F1",
        np.arange(
            1,
            len(results_df) + 1
        )
    )

    # =========================================================================
    # STEP 7 — SAVE PERFORMANCE CSV
    # =========================================================================

    performance_file = (
        OUTPUT_DIR
        / "independent_test_model_performance.csv"
    )

    results_df.to_csv(
        performance_file,
        index=False
    )

    print(
        f"\nSaved:"
    )
    print(
        performance_file
    )

    # =========================================================================
    # STEP 8 — SAVE TEST PREDICTIONS
    # =========================================================================

    predictions_file = (
        OUTPUT_DIR
        / "independent_test_predictions.csv"
    )

    prediction_df.to_csv(
        predictions_file,
        index=False
    )

    print(
        f"\nSaved:"
    )
    print(
        predictions_file
    )

    # =========================================================================
    # STEP 9 — SAVE CONFUSION MATRICES
    # =========================================================================

    confusion_rows = []

    for row in all_metrics:

        confusion_rows.append(
            {
                "Model": row["Model"],
                "TN": row["TN"],
                "FP": row["FP"],
                "FN": row["FN"],
                "TP": row["TP"]
            }
        )

    confusion_df = pd.DataFrame(
        confusion_rows
    )

    confusion_file = (
        OUTPUT_DIR
        / "independent_test_confusion_matrices.csv"
    )

    confusion_df.to_csv(
        confusion_file,
        index=False
    )

    print(
        f"\nSaved:"
    )
    print(
        confusion_file
    )

    # =========================================================================
    # STEP 10 — SAVE TEXT REPORT
    # =========================================================================

    report_file = (
        OUTPUT_DIR
        / "independent_test_evaluation_report.txt"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "JM1 INDEPENDENT TEST EVALUATION REPORT\n"
        )

        file.write(
            "=" * 80 + "\n\n"
        )

        file.write(
            "Dataset: NASA JM1\n"
        )

        file.write(
            f"Test samples: {len(test_df)}\n"
        )

        file.write(
            f"Features: {X_test.shape[1]}\n"
        )

        file.write(
            f"Test dataset: {test_file}\n"
        )

        file.write(
            "Test set: Locked independent 20% test set\n"
        )

        file.write(
            "SMOTE applied to test set: NO\n"
        )

        file.write(
            "Retraining on test set: NO\n"
        )

        file.write(
            "Hyperparameter tuning on test set: NO\n"
        )

        file.write(
            "Threshold tuning: NO\n"
        )

        file.write(
            "Classification threshold: 0.50\n\n"
        )

        file.write(
            "PERFORMANCE RESULTS\n"
        )

        file.write(
            "-" * 80 + "\n"
        )

        for _, row in results_df.iterrows():

            file.write(
                f"\nModel: {row['Model']}\n"
            )

            file.write(
                f"Accuracy    : {row['Accuracy']:.6f}\n"
            )

            file.write(
                f"Precision   : {row['Precision']:.6f}\n"
            )

            file.write(
                f"Recall      : {row['Recall']:.6f}\n"
            )

            file.write(
                f"Specificity : {row['Specificity']:.6f}\n"
            )

            file.write(
                f"F1          : {row['F1']:.6f}\n"
            )

            file.write(
                f"MCC         : {row['MCC']:.6f}\n"
            )

            file.write(
                f"ROC-AUC     : {row['ROC_AUC']:.6f}\n"
            )

            file.write(
                f"PR-AUC      : {row['PR_AUC']:.6f}\n"
            )

            file.write(
                f"TN          : {int(row['TN'])}\n"
            )

            file.write(
                f"FP          : {int(row['FP'])}\n"
            )

            file.write(
                f"FN          : {int(row['FN'])}\n"
            )

            file.write(
                f"TP          : {int(row['TP'])}\n"
            )

        file.write(
            "\n\n"
        )

        file.write(
            "NOTE:\n"
        )

        file.write(
            "The F1 ranking shown in this report is descriptive only. "
            "It does not alter the final model selection performed "
            "during development.\n"
        )

    print(
        f"\nSaved:"
    )

    print(
        report_file
    )

    # =========================================================================
    # STEP 11 — SAVE METADATA
    # =========================================================================

    positive_prevalence = (
        EXPECTED_CLASS_1
        / EXPECTED_TEST_ROWS
    )

    metadata = {
        "dataset": "NASA JM1",
        "test_file": str(test_file),
        "test_samples": int(len(test_df)),
        "test_features": int(X_test.shape[1]),
        "target": TARGET_COLUMN,
        "class_0_count": int(
            (y_test == 0).sum()
        ),
        "class_1_count": int(
            (y_test == 1).sum()
        ),
        "positive_class_prevalence": float(
            positive_prevalence
        ),
        "smote_applied_to_test": False,
        "retrained_on_test": False,
        "hyperparameter_tuning_on_test": False,
        "threshold_tuning": False,
        "classification_threshold": 0.50,
        "models": list(
            MODEL_FILES.keys()
        ),
        "random_state": RANDOM_STATE
    }

    metadata_file = (
        OUTPUT_DIR
        / "independent_test_evaluation_metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        f"\nSaved:"
    )

    print(
        metadata_file
    )

    # =========================================================================
    # STEP 12 — PERFORMANCE BAR CHART
    # =========================================================================

    print("\n" + "-" * 80)
    print("7. GENERATING PERFORMANCE FIGURE")
    print("-" * 80)

    metrics_to_plot = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "MCC",
        "ROC_AUC",
        "PR_AUC"
    ]

    x = np.arange(
        len(results_df)
    )

    width = 0.10

    plt.figure(
        figsize=(14, 7)
    )

    for i, metric in enumerate(
        metrics_to_plot
    ):

        offset = (
            i
            - len(metrics_to_plot) / 2
        ) * width

        plt.bar(
            x + offset,
            results_df[metric].to_numpy(),
            width,
            label=metric
        )

    plt.xticks(
        x,
        results_df["Model"]
    )

    plt.ylabel(
        "Score"
    )

    plt.xlabel(
        "Model"
    )

    plt.title(
        "Independent Test Performance of Final JM1 Models"
    )

    plt.ylim(
        0,
        1.0
    )

    plt.legend(
        bbox_to_anchor=(
            1.02,
            1
        ),
        loc="upper left"
    )

    plt.tight_layout()

    performance_figure = (
        FIGURE_DIR
        / "independent_test_model_performance.png"
    )

    plt.savefig(
        performance_figure,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved:"
    )

    print(
        performance_figure
    )

    # =========================================================================
    # STEP 13 — ROC CURVES
    # =========================================================================

    plt.figure(
        figsize=(8, 7)
    )

    for model_name, curve_data in (
        roc_curve_data.items()
    ):

        plt.plot(
            curve_data["fpr"],
            curve_data["tpr"],
            label=(
                f"{model_name} "
                f"(AUC = "
                f"{curve_data['auc']:.3f})"
            )
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curves on Independent JM1 Test Set"
    )

    plt.legend()

    plt.tight_layout()

    roc_figure = (
        FIGURE_DIR
        / "independent_test_roc_curves.png"
    )

    plt.savefig(
        roc_figure,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved:"
    )

    print(
        roc_figure
    )

    # =========================================================================
    # STEP 14 — PRECISION-RECALL CURVES
    # =========================================================================

    plt.figure(
        figsize=(8, 7)
    )

    for model_name, curve_data in (
        pr_curve_data.items()
    ):

        plt.plot(
            curve_data["recall"],
            curve_data["precision"],
            label=(
                f"{model_name} "
                f"(AP = "
                f"{curve_data['auc']:.3f})"
            )
        )

    # Positive-class prevalence is the appropriate baseline
    # for the precision-recall plot.
    plt.axhline(
        positive_prevalence,
        linestyle="--",
        label=(
            f"Positive prevalence = "
            f"{positive_prevalence:.3f}"
        )
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision-Recall Curves on Independent JM1 Test Set"
    )

    plt.legend()

    plt.tight_layout()

    pr_figure = (
        FIGURE_DIR
        / "independent_test_precision_recall_curves.png"
    )

    plt.savefig(
        pr_figure,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved:"
    )

    print(
        pr_figure
    )

    # =========================================================================
    # STEP 15 — FINAL CONSOLE TABLE
    # =========================================================================

    print("\n" + "=" * 80)
    print("INDEPENDENT TEST RESULTS")
    print("=" * 80)

    display_columns = [
        "Rank_by_F1",
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "MCC",
        "ROC_AUC",
        "PR_AUC"
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value:
            f"{value:.4f}"
        )
    )

    # =========================================================================
    # STEP 16 — FINAL INTEGRITY MESSAGE
    # =========================================================================

    print("\n" + "=" * 80)
    print("STEP 9 COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print(
        "\nIndependent test set:"
    )

    print(
        "    Used only for final evaluation"
    )

    print(
        "    No SMOTE applied"
    )

    print(
        "    No retraining performed"
    )

    print(
        "    No hyperparameter tuning performed"
    )

    print(
        "    No threshold tuning performed"
    )

    print(
        "\nSaved test predictions will be used later "
        "for SHAP and explanation-consistency analysis."
    )

    print(
        "\nNext methodological stage:"
    )

    print(
        "    SHAP analysis of the final models"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
