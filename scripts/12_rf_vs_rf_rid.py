"""
STEP 11
RF vs RF-RID (Random Forest with Redundancy-Induced Dimensionality Reduction)

Research project:
Evaluating SHAP Explanation Consistency in Ensemble Models
for Software Defect Prediction

Dataset:
NASA JM1

Experimental principles:
- Development set only for feature selection
- Locked independent test set remains untouched
- Pearson correlation and VIF used to identify redundancy
- No manual feature deletion
- SMOTE applied only to development data
- RF and RF-RID use identical RF hyperparameters
- Same random state throughout
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
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

from statsmodels.stats.outliers_influence import variance_inflation_factor

from imblearn.over_sampling import SMOTE


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

PROJECT_ROOT = r"E:\Programming\Evaluating SHAP Explanation Consistency in Ensemble Models for Software Defect Prediction"

DEVELOPMENT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "JM1",
    "JM1_development.csv",
)

TEST_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "JM1",
    "JM1_test.csv",
)

FINAL_RF_MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "models",
    "final",
    "rf_final.pkl",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "results",
    "rf_vs_rf_rid",
)

FIGURE_DIR = os.path.join(
    PROJECT_ROOT,
    "figures",
    "rf_vs_rf_rid",
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "rf_rid",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

TARGET = "defects"

FEATURES = [
    "loc",
    "v(g)",
    "ev(g)",
    "iv(g)",
    "n",
    "v",
    "l",
    "d",
    "i",
    "e",
    "b",
    "t",
    "lOCode",
    "lOComment",
    "lOBlank",
    "locCodeAndComment",
    "uniq_Op",
    "uniq_Opnd",
    "total_Op",
    "total_Opnd",
    "branchCount",
]


# =============================================================================
# REDUNDANCY REDUCTION PARAMETERS
# =============================================================================

PEARSON_THRESHOLD = 0.80
VIF_THRESHOLD = 10.0

# Maximum number of iterations for safety.
MAX_ITERATIONS = 100


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_vif(df):
    """
    Calculate VIF for all columns.

    Returns:
        DataFrame with feature and VIF.
    """

    values = df.astype(float).values

    vif_values = []

    for i, feature in enumerate(df.columns):
        try:
            vif = variance_inflation_factor(values, i)
        except Exception:
            vif = np.inf

        vif_values.append(vif)

    result = pd.DataFrame({
        "feature": df.columns,
        "VIF": vif_values,
    })

    result = result.sort_values(
        by="VIF",
        ascending=False,
    ).reset_index(drop=True)

    return result


def calculate_mean_absolute_correlation(df):
    """
    Calculate mean absolute Pearson correlation
    of each feature with all other features.
    """

    corr = df.corr(method="pearson").abs()

    mean_abs_corr = {}

    for feature in corr.columns:
        values = corr.loc[feature].drop(labels=[feature])
        mean_abs_corr[feature] = values.mean()

    return pd.Series(mean_abs_corr)


def identify_high_correlation_pairs(df):
    """
    Identify feature pairs whose absolute Pearson
    correlation is >= PEARSON_THRESHOLD.
    """

    corr = df.corr(method="pearson")

    pairs = []

    columns = corr.columns.tolist()

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):

            r = corr.iloc[i, j]

            if abs(r) >= PEARSON_THRESHOLD:

                pairs.append({
                    "feature_1": columns[i],
                    "feature_2": columns[j],
                    "pearson_r": r,
                    "abs_pearson_r": abs(r),
                })

    result = pd.DataFrame(pairs)

    if not result.empty:
        result = result.sort_values(
            by="abs_pearson_r",
            ascending=False,
        ).reset_index(drop=True)

    return result


def choose_feature_to_remove(df):
    """
    Deterministic redundancy-removal rule.

    Priority:
    1. Highest VIF
    2. Highest mean absolute Pearson correlation
    3. Fixed feature order

    This ensures that feature selection is reproducible
    and does not use the test set.
    """

    vif_df = calculate_vif(df)

    mean_corr = calculate_mean_absolute_correlation(df)

    candidates = vif_df.copy()

    candidates["mean_abs_correlation"] = candidates["feature"].map(
        mean_corr
    )

    # Only consider features violating VIF threshold.
    violating = candidates[
        candidates["VIF"] >= VIF_THRESHOLD
    ].copy()

    if violating.empty:
        return None, candidates

    # Sort deterministically.
    #
    # Highest VIF first.
    # If tied, highest mean absolute correlation.
    # If still tied, original feature order.
    feature_order = {
        feature: index
        for index, feature in enumerate(FEATURES)
    }

    violating["feature_order"] = violating["feature"].map(
        feature_order
    )

    violating = violating.sort_values(
        by=[
            "VIF",
            "mean_abs_correlation",
            "feature_order",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )

    feature_to_remove = violating.iloc[0]["feature"]

    return feature_to_remove, candidates


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("STEP 11 — RF vs RF-RID")
    print("=" * 80)

    # =========================================================================
    # STEP 1 — LOAD DEVELOPMENT DATA
    # =========================================================================

    print("\n[1/10] Loading development data...")

    development = pd.read_csv(DEVELOPMENT_FILE)

    print(f"Development dataset: {DEVELOPMENT_FILE}")
    print(f"Shape: {development.shape}")

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in development.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing development features: {missing_features}"
        )

    if TARGET not in development.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    X_dev = development[FEATURES].copy()
    y_dev = development[TARGET].copy()

    print(f"Features: {X_dev.shape}")
    print(f"Target: {y_dev.shape}")

    # Ensure numeric.
    X_dev = X_dev.apply(pd.to_numeric, errors="coerce")

    if X_dev.isnull().sum().sum() > 0:
        raise ValueError(
            "Missing/non-numeric values detected in development features."
        )

    if np.isinf(X_dev.values).any():
        raise ValueError(
            "Infinite values detected in development features."
        )

    # =========================================================================
    # STEP 2 — ORIGINAL MULTICOLLINEARITY STATUS
    # =========================================================================

    print("\n[2/10] Calculating original VIF and Pearson redundancy...")

    original_vif = calculate_vif(X_dev)

    original_pairs = identify_high_correlation_pairs(X_dev)

    original_vif.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "rf_rid_original_vif.csv",
        ),
        index=False,
    )

    original_pairs.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "rf_rid_original_high_correlation_pairs.csv",
        ),
        index=False,
    )

    print(
        f"Original features: {len(FEATURES)}"
    )

    print(
        f"High-correlation pairs "
        f"(|r| >= {PEARSON_THRESHOLD}): "
        f"{len(original_pairs)}"
    )

    print(
        f"Features with VIF >= {VIF_THRESHOLD}: "
        f"{(original_vif['VIF'] >= VIF_THRESHOLD).sum()}"
    )

    # =========================================================================
    # STEP 3 — AUTOMATIC REDUNDANCY REDUCTION
    # =========================================================================

    print("\n[3/10] Performing deterministic redundancy reduction...")

    selected_features = FEATURES.copy()

    removal_log = []

    iteration = 0

    while iteration < MAX_ITERATIONS:

        iteration += 1

        current_X = X_dev[selected_features].copy()

        current_vif = calculate_vif(current_X)

        max_vif = current_vif["VIF"].max()

        # Check whether VIF criterion is satisfied.
        if max_vif < VIF_THRESHOLD:
            break

        feature_to_remove, diagnostic_vif = choose_feature_to_remove(
            current_X
        )

        if feature_to_remove is None:
            break

        # Get information about feature before removal.
        feature_vif = diagnostic_vif.loc[
            diagnostic_vif["feature"] == feature_to_remove,
            "VIF",
        ].iloc[0]

        mean_corr = calculate_mean_absolute_correlation(
            current_X
        )

        mean_abs_corr = mean_corr[feature_to_remove]

        # Highest correlation involving this feature.
        corr_matrix = current_X.corr(method="pearson").abs()

        corr_values = corr_matrix[
            feature_to_remove
        ].drop(
            labels=[feature_to_remove]
        )

        highest_partner = corr_values.idxmax()
        highest_pair_corr = corr_values.max()

        removal_log.append({
            "iteration": iteration,
            "removed_feature": feature_to_remove,
            "VIF_before_removal": feature_vif,
            "mean_abs_correlation": mean_abs_corr,
            "highest_correlated_partner": highest_partner,
            "highest_abs_correlation": highest_pair_corr,
            "features_before": len(selected_features),
        })

        print(
            f"  Iteration {iteration:02d}: "
            f"remove {feature_to_remove} "
            f"(VIF={feature_vif:.4f}, "
            f"mean |r|={mean_abs_corr:.4f})"
        )

        selected_features.remove(feature_to_remove)

        if len(selected_features) <= 2:
            print(
                "  Safety stop: only two features remain."
            )
            break

    removal_log_df = pd.DataFrame(removal_log)

    removal_log_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "rf_rid_feature_removal_log.csv",
        ),
        index=False,
    )

    # =========================================================================
    # STEP 4 — FINAL REDUCED FEATURE SET
    # =========================================================================

    print("\n[4/10] Final RF-RID feature set...")

    print(
        f"Original feature count: {len(FEATURES)}"
    )

    print(
        f"RF-RID feature count: {len(selected_features)}"
    )

    print(
        f"Features removed: "
        f"{len(FEATURES) - len(selected_features)}"
    )

    print("\nSelected features:")

    for i, feature in enumerate(selected_features, 1):
        print(
            f"  {i:02d}. {feature}"
        )

    removed_features = [
        feature
        for feature in FEATURES
        if feature not in selected_features
    ]

    print("\nRemoved features:")

    for feature in removed_features:
        print(
            f"  - {feature}"
        )

    # =========================================================================
    # STEP 5 — FINAL MULTICOLLINEARITY CHECK
    # =========================================================================

    print(
        "\n[5/10] Verifying multicollinearity after reduction..."
    )

    X_rid_dev = X_dev[selected_features].copy()

    final_vif = calculate_vif(X_rid_dev)

    final_pairs = identify_high_correlation_pairs(
        X_rid_dev
    )

    final_vif.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "rf_rid_final_vif.csv",
        ),
        index=False,
    )

    final_pairs.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "rf_rid_final_high_correlation_pairs.csv",
        ),
        index=False,
    )

    print("\nFinal VIF:")

    print(
        final_vif.to_string(
            index=False
        )
    )

    max_final_vif = final_vif["VIF"].max()

    max_final_corr = (
        final_pairs["abs_pearson_r"].max()
        if not final_pairs.empty
        else 0.0
    )

    print(
        f"\nMaximum final VIF: {max_final_vif:.4f}"
    )

    print(
        f"Maximum final |Pearson r|: "
        f"{max_final_corr:.4f}"
    )

    # =========================================================================
    # STEP 6 — SAVE FEATURE REGISTRY
    # =========================================================================

    print("\n[6/10] Saving RF-RID feature registry...")

    feature_registry = pd.DataFrame({
        "feature": FEATURES,
        "selected_for_rf_rid": [
            feature in selected_features
            for feature in FEATURES
        ],
    })

    # Add original VIF.
    original_vif_map = dict(
        zip(
            original_vif["feature"],
            original_vif["VIF"],
        )
    )

    final_vif_map = dict(
        zip(
            final_vif["feature"],
            final_vif["VIF"],
        )
    )

    feature_registry["original_VIF"] = (
        feature_registry["feature"]
        .map(original_vif_map)
    )

    feature_registry["final_VIF"] = (
        feature_registry["feature"]
        .map(final_vif_map)
    )

    feature_registry.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "rf_rid_feature_registry.csv",
        ),
        index=False,
    )

    with open(
        os.path.join(
            MODEL_DIR,
            "rf_rid_feature_list.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "method": (
                    "Deterministic VIF-based redundancy "
                    "reduction"
                ),
                "pearson_threshold": PEARSON_THRESHOLD,
                "vif_threshold": VIF_THRESHOLD,
                "random_state": RANDOM_STATE,
                "original_features": FEATURES,
                "selected_features": selected_features,
                "removed_features": removed_features,
            },
            f,
            indent=4,
        )

    # =========================================================================
    # STEP 7 — APPLY SMOTE TO DEVELOPMENT DATA
    # =========================================================================

    print("\n[7/10] Applying SMOTE to development data...")

    smote = SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=5,
    )

    X_dev_smote, y_dev_smote = smote.fit_resample(
        X_rid_dev,
        y_dev,
    )

    print(
        f"Before SMOTE: {X_rid_dev.shape}"
    )

    print(
        f"After SMOTE: {X_dev_smote.shape}"
    )

    print(
        "Class distribution after SMOTE:"
    )

    print(
        pd.Series(y_dev_smote).value_counts().sort_index()
    )

    # =========================================================================
    # STEP 8 — TRAIN RF-RID
    # =========================================================================

    print("\n[8/10] Training RF-RID...")

    # Same tuned RF hyperparameters as the final RF.
    rf_rid = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        max_features="sqrt",
        min_samples_leaf=1,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rf_rid.fit(
        X_dev_smote,
        y_dev_smote,
    )

    rf_rid_model_file = os.path.join(
        MODEL_DIR,
        "rf_rid_final.pkl",
    )

    import joblib

    joblib.dump(
        rf_rid,
        rf_rid_model_file,
    )

    print(
        f"RF-RID model saved:\n"
        f"{rf_rid_model_file}"
    )

    # =========================================================================
    # STEP 9 — LOAD LOCKED TEST DATA
    # =========================================================================

    print(
        "\n[9/10] Loading independent test set..."
    )

    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(
            f"Test file not found:\n{TEST_FILE}"
        )

    test = pd.read_csv(TEST_FILE)

    print(
        f"Test dataset: {TEST_FILE}"
    )

    print(
        f"Test shape: {test.shape}"
    )

    if TARGET not in test.columns:
        raise ValueError(
            f"Target '{TARGET}' not found in test set."
        )

    for feature in FEATURES:
        if feature not in test.columns:
            raise ValueError(
                f"Feature '{feature}' missing from test set."
            )

    X_test_full = test[FEATURES].copy()

    X_test_full = X_test_full.apply(
        pd.to_numeric,
        errors="coerce",
    )

    y_test = test[TARGET].copy()

    if X_test_full.isnull().sum().sum() > 0:
        raise ValueError(
            "Missing/non-numeric values detected in test features."
        )

    if np.isinf(X_test_full.values).any():
        raise ValueError(
            "Infinite values detected in test features."
        )

    # =========================================================================
    # LOAD EXISTING FULL RF
    # =========================================================================

    if not os.path.exists(FINAL_RF_MODEL_FILE):
        raise FileNotFoundError(
            f"Existing final RF model not found:\n"
            f"{FINAL_RF_MODEL_FILE}"
        )

    rf_full = joblib.load(
        FINAL_RF_MODEL_FILE
    )

    print(
        f"Full RF model loaded:\n"
        f"{FINAL_RF_MODEL_FILE}"
    )

    # =========================================================================
    # PREDICTIONS
    # =========================================================================

    X_test_rid = X_test_full[selected_features]

    # Full RF
    rf_full_probability = rf_full.predict_proba(
        X_test_full
    )[:, 1]

    rf_full_prediction = (
        rf_full_probability >= 0.50
    ).astype(int)

    # RF-RID
    rf_rid_probability = rf_rid.predict_proba(
        X_test_rid
    )[:, 1]

    rf_rid_prediction = (
        rf_rid_probability >= 0.50
    ).astype(int)

    # =========================================================================
    # METRIC FUNCTION
    # =========================================================================

    def calculate_metrics(
        y_true,
        y_pred,
        y_probability,
        model_name,
    ):

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )

        tn, fp, fn, tp = cm.ravel()

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        )

        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(
                y_true,
                y_pred,
            ),
            "Precision": precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "Specificity": specificity,
            "F1": f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "MCC": matthews_corrcoef(
                y_true,
                y_pred,
            ),
            "ROC_AUC": roc_auc_score(
                y_true,
                y_probability,
            ),
            "PR_AUC": average_precision_score(
                y_true,
                y_probability,
            ),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        }

        return metrics

    # =========================================================================
    # CALCULATE TEST METRICS
    # =========================================================================

    print(
        "\nCalculating independent test performance..."
    )

    rf_metrics = calculate_metrics(
        y_test,
        rf_full_prediction,
        rf_full_probability,
        "RF",
    )

    rid_metrics = calculate_metrics(
        y_test,
        rf_rid_prediction,
        rf_rid_probability,
        "RF-RID",
    )

    comparison = pd.DataFrame([
        rf_metrics,
        rid_metrics,
    ])

    comparison.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "rf_vs_rf_rid_test_performance.csv",
        ),
        index=False,
    )

    # =========================================================================
    # CONFUSION MATRICES
    # =========================================================================

    confusion_results = pd.DataFrame([
        {
            "Model": "RF",
            "TN": rf_metrics["TN"],
            "FP": rf_metrics["FP"],
            "FN": rf_metrics["FN"],
            "TP": rf_metrics["TP"],
        },
        {
            "Model": "RF-RID",
            "TN": rid_metrics["TN"],
            "FP": rid_metrics["FP"],
            "FN": rid_metrics["FN"],
            "TP": rid_metrics["TP"],
        },
    ])

    confusion_results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "rf_vs_rf_rid_confusion_matrices.csv",
        ),
        index=False,
    )

    # =========================================================================
    # PREDICTIONS
    # =========================================================================

    predictions = pd.DataFrame({
        "y_true": y_test.values,

        "RF_probability": rf_full_probability,
        "RF_prediction": rf_full_prediction,

        "RF_RID_probability": rf_rid_probability,
        "RF_RID_prediction": rf_rid_prediction,
    })

    predictions.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "rf_vs_rf_rid_test_predictions.csv",
        ),
        index=False,
    )

    # =========================================================================
    # PRINT RESULTS
    # =========================================================================

    print("\n" + "=" * 80)
    print("INDEPENDENT TEST PERFORMANCE")
    print("=" * 80)

    print(
        comparison[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "Specificity",
                "F1",
                "MCC",
                "ROC_AUC",
                "PR_AUC",
            ]
        ].to_string(index=False)
    )

    # =========================================================================
    # PERFORMANCE DIFFERENCE
    # =========================================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "RF-RID MINUS RF"
    )

    print(
        "=" * 80
    )

    difference = {
        "Metric": [],
        "RF": [],
        "RF_RID": [],
        "RF_RID_minus_RF": [],
    }

    metrics_to_compare = [
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "MCC",
        "ROC_AUC",
        "PR_AUC",
    ]

    for metric in metrics_to_compare:

        rf_value = rf_metrics[metric]
        rid_value = rid_metrics[metric]

        difference["Metric"].append(metric)
        difference["RF"].append(rf_value)
        difference["RF_RID"].append(rid_value)
        difference["RF_RID_minus_RF"].append(
            rid_value - rf_value
        )

        print(
            f"{metric:12s}: "
            f"RF={rf_value:.4f} | "
            f"RF-RID={rid_value:.4f} | "
            f"Difference={rid_value - rf_value:+.4f}"
        )

    difference_df = pd.DataFrame(
        difference
    )

    difference_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "rf_vs_rf_rid_metric_differences.csv",
        ),
        index=False,
    )

    # =========================================================================
    # ROC CURVE
    # =========================================================================

    print("\nCreating ROC curve...")

    rf_fpr, rf_tpr, _ = roc_curve(
        y_test,
        rf_full_probability,
    )

    rid_fpr, rid_tpr, _ = roc_curve(
        y_test,
        rf_rid_probability,
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        rf_fpr,
        rf_tpr,
        label=f"RF (AUC={rf_metrics['ROC_AUC']:.3f})",
    )

    plt.plot(
        rid_fpr,
        rid_tpr,
        label=f"RF-RID (AUC={rid_metrics['ROC_AUC']:.3f})",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        "RF vs RF-RID — Independent Test ROC Curves"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "rf_vs_rf_rid_roc_curves.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # =========================================================================
    # PRECISION-RECALL CURVE
    # =========================================================================

    print(
        "Creating Precision-Recall curve..."
    )

    rf_precision, rf_recall, _ = precision_recall_curve(
        y_test,
        rf_full_probability,
    )

    rid_precision, rid_recall, _ = precision_recall_curve(
        y_test,
        rf_rid_probability,
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        rf_recall,
        rf_precision,
        label=f"RF (AP={rf_metrics['PR_AUC']:.3f})",
    )

    plt.plot(
        rid_recall,
        rid_precision,
        label=f"RF-RID (AP={rid_metrics['PR_AUC']:.3f})",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(
        "RF vs RF-RID — Independent Test Precision-Recall Curves"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "rf_vs_rf_rid_precision_recall_curves.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # =========================================================================
    # FEATURE COUNT FIGURE
    # =========================================================================

    print(
        "Creating feature-count comparison..."
    )

    plt.figure(figsize=(7, 5))

    models = [
        "RF",
        "RF-RID",
    ]

    feature_counts = [
        len(FEATURES),
        len(selected_features),
    ]

    plt.bar(
        models,
        feature_counts,
    )

    plt.ylabel(
        "Number of Features"
    )

    plt.title(
        "Feature Dimensionality: RF vs RF-RID"
    )

    for i, value in enumerate(feature_counts):
        plt.text(
            i,
            value,
            str(value),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "rf_vs_rf_rid_feature_count.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # =========================================================================
    # FINAL REPORT
    # =========================================================================

    print(
        "\nGenerating report..."
    )

    report_file = os.path.join(
        OUTPUT_DIR,
        "rf_vs_rf_rid_report.txt",
    )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "STEP 11 — RF vs RF-RID\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            "Experimental design\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            "Feature selection source: Development set only\n"
        )

        f.write(
            f"Pearson threshold: |r| >= {PEARSON_THRESHOLD}\n"
        )

        f.write(
            f"VIF threshold: >= {VIF_THRESHOLD}\n"
        )

        f.write(
            f"Random state: {RANDOM_STATE}\n"
        )

        f.write(
            "SMOTE applied to development data only\n"
        )

        f.write(
            "Independent test set remained untouched during feature selection and training\n\n"
        )

        f.write(
            "Feature reduction\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            f"Original feature count: {len(FEATURES)}\n"
        )

        f.write(
            f"RF-RID feature count: {len(selected_features)}\n"
        )

        f.write(
            f"Features removed: {len(removed_features)}\n\n"
        )

        f.write(
            "Selected features:\n"
        )

        for feature in selected_features:
            f.write(
                f"  - {feature}\n"
            )

        f.write(
            "\nRemoved features:\n"
        )

        for feature in removed_features:
            f.write(
                f"  - {feature}\n"
            )

        f.write(
            "\nFinal multicollinearity status\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            f"Maximum final VIF: {max_final_vif:.6f}\n"
        )

        f.write(
            f"Maximum final |Pearson r|: {max_final_corr:.6f}\n\n"
        )

        f.write(
            "Independent test performance\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            comparison.to_string(
                index=False
            )
        )

        f.write(
            "\n\nRF-RID minus RF\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            difference_df.to_string(
                index=False
            )
        )

        f.write(
            "\n"
        )

    # =========================================================================
    # METADATA
    # =========================================================================

    metadata = {
        "step": "Step 11 — RF vs RF-RID",

        "development_file": DEVELOPMENT_FILE,
        "test_file": TEST_FILE,

        "random_state": RANDOM_STATE,

        "pearson_threshold": PEARSON_THRESHOLD,
        "vif_threshold": VIF_THRESHOLD,

        "original_feature_count": len(FEATURES),
        "rf_rid_feature_count": len(selected_features),
        "removed_feature_count": len(removed_features),

        "original_features": FEATURES,
        "selected_features": selected_features,
        "removed_features": removed_features,

        "original_high_correlation_pair_count": len(
            original_pairs
        ),

        "final_high_correlation_pair_count": len(
            final_pairs
        ),

        "original_max_vif": float(
            original_vif["VIF"].max()
        ),

        "final_max_vif": float(
            max_final_vif
        ),

        "original_max_abs_pearson": float(
            original_pairs["abs_pearson_r"].max()
            if not original_pairs.empty
            else 0.0
        ),

        "final_max_abs_pearson": float(
            max_final_corr
        ),

        "rf_hyperparameters": {
            "n_estimators": 300,
            "max_depth": 10,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "min_samples_split": 5,
            "random_state": RANDOM_STATE,
        },

        "smote": {
            "random_state": RANDOM_STATE,
            "k_neighbors": 5,
        },

        "test_threshold": 0.50,
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "rf_vs_rf_rid_metadata.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    # =========================================================================
    # COMPLETION
    # =========================================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 11 COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nOriginal features : {len(FEATURES)}"
    )

    print(
        f"RF-RID features   : {len(selected_features)}"
    )

    print(
        f"Features removed  : {len(removed_features)}"
    )

    print(
        f"\nMaximum final VIF : {max_final_vif:.4f}"
    )

    print(
        f"Maximum final |r| : {max_final_corr:.4f}"
    )

    print(
        "\nRF vs RF-RID test performance:"
    )

    print(
        comparison[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "MCC",
                "ROC_AUC",
                "PR_AUC",
            ]
        ].to_string(index=False)
    )

    print(
        "\nMain output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nModel directory:"
    )

    print(
        MODEL_DIR
    )

    print(
        "\nFigure directory:"
    )

    print(
        FIGURE_DIR
    )

    print(
        "\nNEXT STEP:"
    )

    print(
        "STEP 12 — SHAP comparison: RF vs RF-RID"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    main()
