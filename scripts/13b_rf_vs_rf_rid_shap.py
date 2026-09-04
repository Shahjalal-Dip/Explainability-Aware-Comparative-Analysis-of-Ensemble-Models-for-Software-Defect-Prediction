"""
STEP 14 — CORRECTED RF vs RF-RID SHAP EXPLANATION COMPARISON

Purpose
-------
Compare SHAP explanations of:
    1. Full Random Forest (RF)
    2. Random Forest with Redundancy-Informed Dimensionality reduction (RF-RID)

IMPORTANT
---------
RF-RID uses only the 12 features selected during the RF-RID experiment.

The full RF SHAP values are therefore restricted to the same 12 features
before explanation consistency is calculated.

This avoids the invalid previous comparison where RF-RID was incorrectly
interpreted as having all 21 features.

Comparison metrics
------------------
Global:
    - Spearman rank correlation
    - Cosine similarity

Local:
    - Per-instance Spearman correlation
    - Per-instance cosine similarity

Experimental controls
---------------------
- Same independent test set
- Same 300 test observations
- Same 100 background observations
- Same feature ordering
- Test set remains untouched during training

Outputs
-------
data/results/rf_vs_rf_rid_shap/
    rf_vs_rf_rid_global_comparison.csv
    rf_vs_rf_rid_local_comparison.csv
    rf_vs_rf_rid_summary.csv
    rf_vs_rf_rid_shap_metadata.json
    rf_vs_rf_rid_shap_report.txt

figures/rf_vs_rf_rid_shap/
    global_shap_comparison.png
    local_spearman_distribution.png
    local_cosine_distribution.png
"""

import matplotlib.pyplot as plt
import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import shap

from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "JM1",
    "JM1_test.csv"
)

RF_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "final",
    "rf_final.pkl"
)

RF_RID_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "rf_rid",
    "rf_rid_final.pkl"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "results",
    "rf_vs_rf_rid_shap"
)

FIGURE_DIR = os.path.join(
    PROJECT_ROOT,
    "figures",
    "rf_vs_rf_rid_shap"
)

RANDOM_STATE = 42

SHAP_SAMPLE_SIZE = 300
BACKGROUND_SIZE = 100


# =============================================================================
# ACTUAL RF-RID FEATURE SET
# =============================================================================
#
# These are the 12 features selected by Step 11 RF-RID.
#
# Removed:
#   loc
#   n
#   v
#   e
#   b
#   lOCode
#   total_Op
#   total_Opnd
#   branchCount
#
# Retained:
#   v(g)
#   ev(g)
#   iv(g)
#   l
#   d
#   i
#   t
#   lOComment
#   lOBlank
#   locCodeAndComment
#   uniq_Op
#   uniq_Opnd
#
# =============================================================================

RF_RID_FEATURES = [
    "v(g)",
    "ev(g)",
    "iv(g)",
    "l",
    "d",
    "i",
    "t",
    "lOComment",
    "lOBlank",
    "locCodeAndComment",
    "uniq_Op",
    "uniq_Opnd",
]

REMOVED_FEATURES = [
    "loc",
    "n",
    "v",
    "e",
    "b",
    "lOCode",
    "total_Op",
    "total_Opnd",
    "branchCount",
]

TARGET = "defects"


# =============================================================================
# CREATE DIRECTORIES
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_positive_class_shap(shap_values):
    """
    Convert SHAP output into a 2D array corresponding to the positive class.

    Handles common SHAP output formats:
        - ndarray: (n_samples, n_features)
        - ndarray: (n_samples, n_features, n_classes)
        - list of arrays: one array per class
    """

    if isinstance(shap_values, list):

        # Binary classification:
        # list[0] = class 0
        # list[1] = class 1
        if len(shap_values) == 2:
            return np.asarray(shap_values[1])

        return np.asarray(shap_values[-1])

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 2:
        return shap_values

    if shap_values.ndim == 3:

        # Usually:
        # (samples, features, classes)
        if shap_values.shape[-1] == 2:
            return shap_values[:, :, 1]

        # Less common:
        # (classes, samples, features)
        if shap_values.shape[0] == 2:
            return shap_values[1]

    raise ValueError(
        f"Unsupported SHAP value shape: {shap_values.shape}"
    )


def calculate_cosine(a, b):
    """
    Calculate cosine similarity between two SHAP vectors.
    """

    a = np.asarray(a).reshape(1, -1)
    b = np.asarray(b).reshape(1, -1)

    return float(cosine_similarity(a, b)[0, 0])


def calculate_spearman(a, b):
    """
    Calculate Spearman rank correlation.

    Returns NaN if either vector is constant.
    """

    a = np.asarray(a)
    b = np.asarray(b)

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    rho, _ = spearmanr(a, b)

    return float(rho)


def summarize_distribution(values):
    """
    Return descriptive statistics for an array.
    """

    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("STEP 14 — CORRECTED RF vs RF-RID SHAP EXPLANATION COMPARISON")
print("=" * 80)


# =============================================================================
# [1/9] LOAD TEST DATA
# =============================================================================

print("\n[1/9] Loading independent test data...")

test_df = pd.read_csv(TEST_PATH)

if TARGET not in test_df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found in test dataset."
    )

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print(f"Test shape: {test_df.shape}")
print(f"Feature count: {X_test.shape[1]}")
print(f"Target: {TARGET}")


# =============================================================================
# [2/9] VERIFY RF-RID FEATURE SET
# =============================================================================

print("\n[2/9] Verifying RF-RID feature configuration...")

missing_rid_features = [
    f for f in RF_RID_FEATURES
    if f not in X_test.columns
]

if missing_rid_features:
    raise ValueError(
        "The following RF-RID features are missing from the test dataset:\n"
        + "\n".join(missing_rid_features)
    )

print(f"Full RF feature count: {X_test.shape[1]}")
print(f"RF-RID feature count: {len(RF_RID_FEATURES)}")

print("\nRF-RID selected features:")
for i, feature in enumerate(RF_RID_FEATURES, start=1):
    print(f"  {i:2d}. {feature}")

print("\nRF-RID removed features:")
for feature in REMOVED_FEATURES:
    print(f"  - {feature}")


# =============================================================================
# [3/9] LOAD MODELS
# =============================================================================

print("\n[3/9] Loading trained models...")

if not os.path.exists(RF_MODEL_PATH):
    raise FileNotFoundError(
        f"RF model not found:\n{RF_MODEL_PATH}"
    )

if not os.path.exists(RF_RID_MODEL_PATH):
    raise FileNotFoundError(
        f"RF-RID model not found:\n{RF_RID_MODEL_PATH}"
    )

rf_model = joblib.load(RF_MODEL_PATH)
rf_rid_model = joblib.load(RF_RID_MODEL_PATH)

print("RF model loaded.")
print("RF-RID model loaded.")


# =============================================================================
# [4/9] VERIFY MODEL FEATURE COUNTS
# =============================================================================

print("\n[4/9] Verifying model feature counts...")

# RF
if hasattr(rf_model, "n_features_in_"):
    rf_model_features = rf_model.n_features_in_
else:
    rf_model_features = X_test.shape[1]

# RF-RID
if hasattr(rf_rid_model, "n_features_in_"):
    rf_rid_model_features = rf_rid_model.n_features_in_
else:
    rf_rid_model_features = len(RF_RID_FEATURES)

print(f"RF model expects: {rf_model_features} features")
print(f"RF-RID model expects: {rf_rid_model_features} features")

if rf_model_features != X_test.shape[1]:
    raise ValueError(
        "RF model feature count does not match the 21 test features."
    )

if rf_rid_model_features != len(RF_RID_FEATURES):
    raise ValueError(
        "RF-RID model does not appear to be trained on the expected "
        f"{len(RF_RID_FEATURES)} RF-RID features."
    )


# =============================================================================
# [5/9] FIXED TEST SAMPLE + BACKGROUND SAMPLE
# =============================================================================

print("\n[5/9] Creating fixed SHAP evaluation samples...")

rng = np.random.RandomState(RANDOM_STATE)

if len(X_test) < SHAP_SAMPLE_SIZE:
    raise ValueError(
        f"Test set contains only {len(X_test)} rows, "
        f"but SHAP_SAMPLE_SIZE={SHAP_SAMPLE_SIZE}."
    )

test_indices = rng.choice(
    len(X_test),
    size=SHAP_SAMPLE_SIZE,
    replace=False
)

test_indices = np.sort(test_indices)

X_shap_full = X_test.iloc[test_indices].copy()

# Background sample is taken from the independent test set exactly as in
# the existing SHAP workflow. It is used only as SHAP background data and
# does not affect model fitting.
if len(X_test) < BACKGROUND_SIZE:
    raise ValueError(
        f"Test set contains only {len(X_test)} rows, "
        f"but BACKGROUND_SIZE={BACKGROUND_SIZE}."
    )

background_indices = rng.choice(
    len(X_test),
    size=BACKGROUND_SIZE,
    replace=False
)

background_indices = np.sort(background_indices)

X_background_full = X_test.iloc[background_indices].copy()

print(f"SHAP evaluation observations: {len(X_shap_full)}")
print(f"Background observations: {len(X_background_full)}")

# Same 12-feature representation for both models
X_shap_rid = X_shap_full[RF_RID_FEATURES].copy()
X_background_rid = X_background_full[RF_RID_FEATURES].copy()


# =============================================================================
# [6/9] CALCULATE RF SHAP
# =============================================================================

print("\n[6/9] Calculating RF SHAP values...")

print("Using TreeExplainer for RF...")

rf_explainer = shap.TreeExplainer(
    rf_model
)

rf_shap_raw = rf_explainer.shap_values(
    X_shap_full
)

rf_shap_full = extract_positive_class_shap(
    rf_shap_raw
)

print(f"RF SHAP shape: {rf_shap_full.shape}")

if rf_shap_full.shape != (
    SHAP_SAMPLE_SIZE,
    X_test.shape[1]
):
    raise ValueError(
        "Unexpected RF SHAP shape: "
        f"{rf_shap_full.shape}"
    )

# Map feature -> SHAP column
rf_feature_to_index = {
    feature: idx
    for idx, feature in enumerate(X_test.columns)
}

rid_indices_in_rf = [
    rf_feature_to_index[feature]
    for feature in RF_RID_FEATURES
]

# IMPORTANT:
# Restrict full RF explanation to the exact 12 RF-RID features.
rf_shap_rid = rf_shap_full[:, rid_indices_in_rf]

print(
    "RF SHAP restricted to RF-RID features: "
    f"{rf_shap_rid.shape}"
)


# =============================================================================
# [7/9] CALCULATE RF-RID SHAP
# =============================================================================

print("\n[7/9] Calculating RF-RID SHAP values...")

print("Using TreeExplainer for RF-RID...")

rf_rid_explainer = shap.TreeExplainer(
    rf_rid_model
)

rf_rid_shap_raw = rf_rid_explainer.shap_values(
    X_shap_rid
)

rf_rid_shap = extract_positive_class_shap(
    rf_rid_shap_raw
)

print(f"RF-RID SHAP shape: {rf_rid_shap.shape}")

if rf_rid_shap.shape != (
    SHAP_SAMPLE_SIZE,
    len(RF_RID_FEATURES)
):
    raise ValueError(
        "Unexpected RF-RID SHAP shape: "
        f"{rf_rid_shap.shape}. "
        f"Expected ({SHAP_SAMPLE_SIZE}, "
        f"{len(RF_RID_FEATURES)})."
    )


# =============================================================================
# SHAP DATAFRAME CONSTRUCTION
# =============================================================================

rf_shap_df = pd.DataFrame(
    rf_shap_rid,
    columns=RF_RID_FEATURES
)

rf_rid_shap_df = pd.DataFrame(
    rf_rid_shap,
    columns=RF_RID_FEATURES
)


# =============================================================================
# [8/9] GLOBAL + LOCAL COMPARISON
# =============================================================================

print("\n[8/9] Calculating explanation consistency metrics...")


# -----------------------------------------------------------------------------
# GLOBAL FEATURE IMPORTANCE
# -----------------------------------------------------------------------------

rf_global_importance = (
    rf_shap_df.abs()
    .mean(axis=0)
)

rf_rid_global_importance = (
    rf_rid_shap_df.abs()
    .mean(axis=0)
)

global_spearman = calculate_spearman(
    rf_global_importance.values,
    rf_rid_global_importance.values
)

global_cosine = calculate_cosine(
    rf_global_importance.values,
    rf_rid_global_importance.values
)

print("\nGLOBAL EXPLANATION CONSISTENCY")
print("-" * 60)
print(f"Spearman correlation : {global_spearman:.6f}")
print(f"Cosine similarity    : {global_cosine:.6f}")


# -----------------------------------------------------------------------------
# FEATURE-LEVEL GLOBAL TABLE
# -----------------------------------------------------------------------------

global_comparison = pd.DataFrame({
    "Feature": RF_RID_FEATURES,
    "RF_Mean_Abs_SHAP": [
        rf_global_importance[f]
        for f in RF_RID_FEATURES
    ],
    "RF_RID_Mean_Abs_SHAP": [
        rf_rid_global_importance[f]
        for f in RF_RID_FEATURES
    ],
})

global_comparison["RF_Rank"] = (
    global_comparison["RF_Mean_Abs_SHAP"]
    .rank(
        method="average",
        ascending=False
    )
)

global_comparison["RF_RID_Rank"] = (
    global_comparison["RF_RID_Mean_Abs_SHAP"]
    .rank(
        method="average",
        ascending=False
    )
)

global_comparison["Absolute_Difference"] = (
    global_comparison["RF_Mean_Abs_SHAP"]
    -
    global_comparison["RF_RID_Mean_Abs_SHAP"]
).abs()

global_comparison["Rank_Difference"] = (
    global_comparison["RF_Rank"]
    -
    global_comparison["RF_RID_Rank"]
).abs()


# -----------------------------------------------------------------------------
# LOCAL EXPLANATION CONSISTENCY
# -----------------------------------------------------------------------------

print("\nCalculating local explanation consistency...")

local_results = []

for i in range(SHAP_SAMPLE_SIZE):

    rf_vector = rf_shap_rid[i]
    rf_rid_vector = rf_rid_shap[i]

    rho = calculate_spearman(
        rf_vector,
        rf_rid_vector
    )

    cosine = calculate_cosine(
        rf_vector,
        rf_rid_vector
    )

    local_results.append({
        "SHAP_Observation": i,
        "Original_Test_Index": int(test_indices[i]),
        "Local_Spearman": rho,
        "Local_Cosine": cosine,
    })


local_comparison = pd.DataFrame(local_results)


# -----------------------------------------------------------------------------
# LOCAL SUMMARY
# -----------------------------------------------------------------------------

local_spearman_summary = summarize_distribution(
    local_comparison["Local_Spearman"].values
)

local_cosine_summary = summarize_distribution(
    local_comparison["Local_Cosine"].values
)

print("\nLOCAL EXPLANATION CONSISTENCY")
print("-" * 60)

print(
    "Local Spearman:"
    f" mean={local_spearman_summary['mean']:.6f},"
    f" std={local_spearman_summary['std']:.6f},"
    f" median={local_spearman_summary['median']:.6f}"
)

print(
    "Local Cosine:"
    f" mean={local_cosine_summary['mean']:.6f},"
    f" std={local_cosine_summary['std']:.6f},"
    f" median={local_cosine_summary['median']:.6f}"
)


# =============================================================================
# SUMMARY TABLE
# =============================================================================

summary = pd.DataFrame([{
    "Model_Comparison": "RF_vs_RF_RID",
    "Full_RF_Features": X_test.shape[1],
    "RF_RID_Features": len(RF_RID_FEATURES),
    "Removed_Features": len(REMOVED_FEATURES),
    "SHAP_Sample_Size": SHAP_SAMPLE_SIZE,
    "Background_Size": BACKGROUND_SIZE,

    "Global_Spearman": global_spearman,
    "Global_Cosine": global_cosine,

    "Local_Spearman_Mean":
        local_spearman_summary["mean"],

    "Local_Spearman_STD":
        local_spearman_summary["std"],

    "Local_Spearman_Median":
        local_spearman_summary["median"],

    "Local_Cosine_Mean":
        local_cosine_summary["mean"],

    "Local_Cosine_STD":
        local_cosine_summary["std"],

    "Local_Cosine_Median":
        local_cosine_summary["median"],
}])


# =============================================================================
# [9/9] SAVE RESULTS + FIGURES
# =============================================================================

print("\n[9/9] Saving results and figures...")


# -----------------------------------------------------------------------------
# SAVE CSV FILES
# -----------------------------------------------------------------------------

global_path = os.path.join(
    OUTPUT_DIR,
    "rf_vs_rf_rid_global_comparison.csv"
)

local_path = os.path.join(
    OUTPUT_DIR,
    "rf_vs_rf_rid_local_comparison.csv"
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "rf_vs_rf_rid_summary.csv"
)

global_comparison.to_csv(
    global_path,
    index=False
)

local_comparison.to_csv(
    local_path,
    index=False
)

summary.to_csv(
    summary_path,
    index=False
)


# =============================================================================
# SAVE METADATA
# =============================================================================

metadata = {
    "experiment": "RF vs RF-RID SHAP Explanation Comparison",

    "random_state": RANDOM_STATE,

    "test_dataset": TEST_PATH,

    "test_rows": int(len(X_test)),

    "shap_sample_size": SHAP_SAMPLE_SIZE,

    "background_size": BACKGROUND_SIZE,

    "full_rf_feature_count": int(X_test.shape[1]),

    "rf_rid_feature_count": int(len(RF_RID_FEATURES)),

    "rf_rid_selected_features": RF_RID_FEATURES,

    "rf_rid_removed_features": REMOVED_FEATURES,

    "global_metrics": {
        "spearman": global_spearman,
        "cosine": global_cosine,
    },

    "local_metrics": {
        "spearman": local_spearman_summary,
        "cosine": local_cosine_summary,
    },

    "methodological_note": (
        "Full RF SHAP values were restricted to the 12 features retained "
        "by RF-RID before calculating explanation consistency. "
        "Therefore, both models were compared in the same 12-dimensional "
        "feature space."
    ),
}

metadata_path = os.path.join(
    OUTPUT_DIR,
    "rf_vs_rf_rid_shap_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# =============================================================================
# SAVE TEXT REPORT
# =============================================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "rf_vs_rf_rid_shap_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 80 + "\n"
    )

    f.write(
        "RF vs RF-RID SHAP EXPLANATION COMPARISON\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        "Experimental design\n"
    )

    f.write(
        "-" * 40 + "\n"
    )

    f.write(
        f"Test observations: {len(X_test)}\n"
    )

    f.write(
        f"SHAP observations: {SHAP_SAMPLE_SIZE}\n"
    )

    f.write(
        f"Background observations: {BACKGROUND_SIZE}\n"
    )

    f.write(
        f"Full RF features: {X_test.shape[1]}\n"
    )

    f.write(
        f"RF-RID features: {len(RF_RID_FEATURES)}\n"
    )

    f.write(
        f"RF-RID removed features: {len(REMOVED_FEATURES)}\n\n"
    )

    f.write(
        "RF-RID selected features\n"
    )

    f.write(
        "-" * 40 + "\n"
    )

    for feature in RF_RID_FEATURES:
        f.write(
            f"- {feature}\n"
        )

    f.write(
        "\nRF-RID removed features\n"
    )

    f.write(
        "-" * 40 + "\n"
    )

    for feature in REMOVED_FEATURES:
        f.write(
            f"- {feature}\n"
        )

    f.write(
        "\nGLOBAL EXPLANATION CONSISTENCY\n"
    )

    f.write(
        "-" * 40 + "\n"
    )

    f.write(
        f"Spearman correlation: {global_spearman:.6f}\n"
    )

    f.write(
        f"Cosine similarity:    {global_cosine:.6f}\n"
    )

    f.write(
        "\nLOCAL EXPLANATION CONSISTENCY\n"
    )

    f.write(
        "-" * 40 + "\n"
    )

    f.write(
        f"Local Spearman mean:   "
        f"{local_spearman_summary['mean']:.6f}\n"
    )

    f.write(
        f"Local Spearman std:    "
        f"{local_spearman_summary['std']:.6f}\n"
    )

    f.write(
        f"Local Spearman median: "
        f"{local_spearman_summary['median']:.6f}\n"
    )

    f.write(
        f"Local Cosine mean:     "
        f"{local_cosine_summary['mean']:.6f}\n"
    )

    f.write(
        f"Local Cosine std:      "
        f"{local_cosine_summary['std']:.6f}\n"
    )

    f.write(
        f"Local Cosine median:   "
        f"{local_cosine_summary['median']:.6f}\n"
    )

    f.write(
        "\nIMPORTANT METHODOLOGICAL NOTE\n"
    )

    f.write(
        "-" * 40 + "\n"
    )

    f.write(
        "The full RF SHAP explanations were restricted to the exact "
        "12 features retained by RF-RID. The previous analysis that "
        "treated RF-RID as a 21-feature model must not be used for "
        "final reporting.\n"
    )


# =============================================================================
# FIGURES
# =============================================================================


# -----------------------------------------------------------------------------
# FIGURE 1 — GLOBAL SHAP COMPARISON
# -----------------------------------------------------------------------------

plt.figure(figsize=(10, 7))

x = np.arange(len(RF_RID_FEATURES))
width = 0.38

plt.bar(
    x - width / 2,
    global_comparison["RF_Mean_Abs_SHAP"],
    width,
    label="RF"
)

plt.bar(
    x + width / 2,
    global_comparison["RF_RID_Mean_Abs_SHAP"],
    width,
    label="RF-RID"
)

plt.xticks(
    x,
    RF_RID_FEATURES,
    rotation=75,
    ha="right"
)

plt.ylabel("Mean Absolute SHAP Value")

plt.title(
    "Global SHAP Importance: RF vs RF-RID"
)

plt.legend()

plt.tight_layout()

global_fig_path = os.path.join(
    FIGURE_DIR,
    "global_shap_comparison.png"
)

plt.savefig(
    global_fig_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -----------------------------------------------------------------------------
# FIGURE 2 — LOCAL SPEARMAN
# -----------------------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    local_comparison["Local_Spearman"],
    bins=30
)

plt.axvline(
    local_spearman_summary["mean"],
    linestyle="--",
    label=(
        f"Mean = "
        f"{local_spearman_summary['mean']:.3f}"
    )
)

plt.xlabel(
    "Local Spearman Correlation"
)

plt.ylabel(
    "Number of Test Observations"
)

plt.title(
    "Local SHAP Spearman Similarity: RF vs RF-RID"
)

plt.legend()

plt.tight_layout()

spearman_fig_path = os.path.join(
    FIGURE_DIR,
    "local_spearman_distribution.png"
)

plt.savefig(
    spearman_fig_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -----------------------------------------------------------------------------
# FIGURE 3 — LOCAL COSINE
# -----------------------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    local_comparison["Local_Cosine"],
    bins=30
)

plt.axvline(
    local_cosine_summary["mean"],
    linestyle="--",
    label=(
        f"Mean = "
        f"{local_cosine_summary['mean']:.3f}"
    )
)

plt.xlabel(
    "Local Cosine Similarity"
)

plt.ylabel(
    "Number of Test Observations"
)

plt.title(
    "Local SHAP Cosine Similarity: RF vs RF-RID"
)

plt.legend()

plt.tight_layout()

cosine_fig_path = os.path.join(
    FIGURE_DIR,
    "local_cosine_distribution.png"
)

plt.savefig(
    cosine_fig_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# FINAL CONSOLE OUTPUT
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 14 COMPLETED")
print("=" * 80)

print("\nRF-RID FEATURE SPACE:")
print(f"  Full RF features : {X_test.shape[1]}")
print(f"  RF-RID features  : {len(RF_RID_FEATURES)}")
print(f"  Removed features : {len(REMOVED_FEATURES)}")

print("\nGLOBAL EXPLANATION CONSISTENCY:")
print(
    f"  Spearman : {global_spearman:.6f}"
)
print(
    f"  Cosine   : {global_cosine:.6f}"
)

print("\nLOCAL EXPLANATION CONSISTENCY:")

print(
    f"  Spearman mean   : "
    f"{local_spearman_summary['mean']:.6f}"
)

print(
    f"  Spearman std    : "
    f"{local_spearman_summary['std']:.6f}"
)

print(
    f"  Spearman median : "
    f"{local_spearman_summary['median']:.6f}"
)

print(
    f"  Cosine mean     : "
    f"{local_cosine_summary['mean']:.6f}"
)

print(
    f"  Cosine std      : "
    f"{local_cosine_summary['std']:.6f}"
)

print(
    f"  Cosine median   : "
    f"{local_cosine_summary['median']:.6f}"
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nFigure directory:")
print(FIGURE_DIR)

print("\nIMPORTANT:")
print(
    "These results replace the previous invalid RF vs RF-RID "
    "SHAP comparison."
)

print("\nNEXT STEP:")
print(
    "Run the script and send me the complete terminal output."
)

print("=" * 80)
