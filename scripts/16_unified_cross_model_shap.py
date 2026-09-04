"""
STEP 15 — UNIFIED CROSS-MODEL SHAP CONSISTENCY ANALYSIS

Purpose
-------
Recalculate cross-model SHAP explanation consistency using ONE common
SHAP methodology for all four models:

    RF
    XGB
    LGBM
    MLP

Why this analysis is required
-----------------------------
The previous cross-model SHAP analysis used:
    - TreeExplainer for RF
    - TreeExplainer for LGBM
    - fallback TreeExplainer for XGB
    - permutation SHAP for MLP

Because the models were not all explained using the same SHAP framework
and output scale, direct cross-model comparisons were not fully
methodologically consistent.

This script therefore uses:

    shap.Explainer(
        prediction_function,
        background,
        algorithm="permutation"
    )

for ALL FOUR MODELS.

The prediction function returns the positive-class probability.

Experimental controls
---------------------
- Same independent test set
- Same 300 SHAP observations
- Same 100 background observations
- Same feature ordering
- Same prediction output: P(defective)
- Same permutation SHAP methodology
- Test set is never used for model fitting

Global metrics
--------------
- Spearman correlation of mean absolute SHAP importance
- Cosine similarity of mean absolute SHAP importance

Local metrics
--------------
- Per-observation Spearman correlation
- Per-observation cosine similarity

Model pairs
-----------
RF-XGB
RF-LGBM
RF-MLP
XGB-LGBM
XGB-MLP
LGBM-MLP

Outputs
-------
data/results/unified_cross_model_shap/

    unified_global_shap_consistency.csv
    unified_local_shap_consistency.csv
    unified_cross_model_summary.csv
    unified_cross_model_metadata.json
    unified_cross_model_report.txt

figures/unified_cross_model_shap/

    unified_global_shap_importance.png
    unified_global_consistency_heatmap.png
    unified_local_spearman_boxplot.png
    unified_local_cosine_boxplot.png

IMPORTANT
---------
This script is for CROSS-MODEL explanation consistency.

The intra-model stability results from Step 12/13 remain separate.

Do NOT combine IMSI and cross-model consistency into a final SSI until
both analyses have been reviewed.
"""

import os
import json
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import joblib
import shap

from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "JM1",
    "JM1_test.csv"
)

MODEL_PATHS = {
    "RF": os.path.join(
        PROJECT_ROOT,
        "models",
        "final",
        "rf_final.pkl"
    ),

    "XGB": os.path.join(
        PROJECT_ROOT,
        "models",
        "final",
        "xgb_final.pkl"
    ),

    "LGBM": os.path.join(
        PROJECT_ROOT,
        "models",
        "final",
        "lgbm_final.pkl"
    ),

    "MLP": os.path.join(
        PROJECT_ROOT,
        "models",
        "final",
        "mlp_final.pkl"
    ),
}

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "results",
    "unified_cross_model_shap"
)

FIGURE_DIR = os.path.join(
    PROJECT_ROOT,
    "figures",
    "unified_cross_model_shap"
)

TARGET = "defects"

RANDOM_STATE = 42

SHAP_SAMPLE_SIZE = 300
BACKGROUND_SIZE = 100

# -------------------------------------------------------------------------
# Permutation SHAP evaluation budget.
#
# Minimum for N features is approximately:
#     2 * N + 1
#
# We use:
#     10 * N + 1
#
# This provides a more reliable approximation while keeping computation
# manageable.
# -------------------------------------------------------------------------

MAX_EVALS = 10 * 21 + 1


# =============================================================================
# CREATE OUTPUT DIRECTORIES
# =============================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_spearman(a, b):
    """
    Calculate Spearman rank correlation.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if (
        not np.all(np.isfinite(a))
        or not np.all(np.isfinite(b))
    ):
        return np.nan

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    rho, _ = spearmanr(
        a,
        b
    )

    return float(rho)


def calculate_cosine(a, b):
    """
    Calculate cosine similarity.
    """

    a = np.asarray(a, dtype=float).reshape(1, -1)
    b = np.asarray(b, dtype=float).reshape(1, -1)

    if (
        not np.all(np.isfinite(a))
        or not np.all(np.isfinite(b))
    ):
        return np.nan

    return float(
        cosine_similarity(
            a,
            b
        )[0, 0]
    )


def positive_probability(model, X):
    """
    Return positive-class probability.

    All four models are expected to expose predict_proba().
    """

    probabilities = model.predict_proba(X)

    probabilities = np.asarray(
        probabilities
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "Unexpected predict_proba output shape: "
            f"{probabilities.shape}"
        )

    if probabilities.shape[1] < 2:
        raise ValueError(
            "Binary positive-class probability could not be extracted."
        )

    return probabilities[:, 1]


def summarize(values):
    """
    Descriptive statistics.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "mean": float(
            np.mean(values)
        ),

        "std": float(
            np.std(
                values,
                ddof=1
            )
        ) if len(values) > 1 else 0.0,

        "median": float(
            np.median(values)
        ),

        "min": float(
            np.min(values)
        ),

        "max": float(
            np.max(values)
        ),
    }


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("STEP 15 — UNIFIED CROSS-MODEL SHAP CONSISTENCY ANALYSIS")
print("=" * 80)


# =============================================================================
# [1/10] LOAD TEST DATA
# =============================================================================

print("\n[1/10] Loading independent test data...")

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        f"Test dataset not found:\n{TEST_PATH}"
    )

test_df = pd.read_csv(
    TEST_PATH
)

if TARGET not in test_df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

X_test = test_df.drop(
    columns=[TARGET]
)

y_test = test_df[TARGET]

FEATURES = list(
    X_test.columns
)

N_FEATURES = len(FEATURES)

print(
    f"Test shape: {test_df.shape}"
)

print(
    f"Feature count: {N_FEATURES}"
)

print(
    f"Target: {TARGET}"
)

print("\nFeatures:")

for i, feature in enumerate(
    FEATURES,
    start=1
):
    print(
        f"  {i:2d}. {feature}"
    )


# =============================================================================
# [2/10] LOAD MODELS
# =============================================================================

print("\n[2/10] Loading final models...")

models = {}

for model_name, model_path in MODEL_PATHS.items():

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_name} model not found:\n{model_path}"
        )

    models[model_name] = joblib.load(
        model_path
    )

    print(
        f"{model_name}: loaded"
    )


# =============================================================================
# [3/10] VERIFY MODEL FEATURE COUNTS
# =============================================================================

print("\n[3/10] Verifying model feature counts...")

for model_name, model in models.items():

    if hasattr(
        model,
        "n_features_in_"
    ):

        expected_features = (
            model.n_features_in_
        )

        print(
            f"{model_name}: "
            f"{expected_features} features"
        )

        if expected_features != N_FEATURES:

            raise ValueError(
                f"{model_name} expects "
                f"{expected_features} features, "
                f"but test data has "
                f"{N_FEATURES}."
            )

    else:

        print(
            f"{model_name}: "
            "feature count attribute unavailable"
        )


# =============================================================================
# [4/10] CREATE FIXED SHAP SAMPLE
# =============================================================================

print(
    "\n[4/10] Creating fixed SHAP evaluation sample..."
)

rng = np.random.RandomState(
    RANDOM_STATE
)

if len(X_test) < SHAP_SAMPLE_SIZE:
    raise ValueError(
        "Test dataset is smaller than SHAP sample size."
    )

if len(X_test) < BACKGROUND_SIZE:
    raise ValueError(
        "Test dataset is smaller than background size."
    )


# -------------------------------------------------------------------------
# Fixed evaluation observations
# -------------------------------------------------------------------------

shap_indices = rng.choice(
    len(X_test),
    size=SHAP_SAMPLE_SIZE,
    replace=False
)

shap_indices = np.sort(
    shap_indices
)

X_shap = X_test.iloc[
    shap_indices
].copy()


# -------------------------------------------------------------------------
# Fixed background observations
# -------------------------------------------------------------------------

background_indices = rng.choice(
    len(X_test),
    size=BACKGROUND_SIZE,
    replace=False
)

background_indices = np.sort(
    background_indices
)

X_background = X_test.iloc[
    background_indices
].copy()


print(
    f"SHAP observations: {len(X_shap)}"
)

print(
    f"Background observations: {len(X_background)}"
)

print(
    f"Permutation max_evals: {MAX_EVALS}"
)


# =============================================================================
# [5/10] VERIFY POSITIVE-CLASS PROBABILITY
# =============================================================================

print(
    "\n[5/10] Verifying positive-class probability..."
)

for model_name, model in models.items():

    sample_probability = positive_probability(
        model,
        X_shap.iloc[:5]
    )

    print(
        f"{model_name}: "
        f"probability range = "
        f"{sample_probability.min():.6f} "
        f"to "
        f"{sample_probability.max():.6f}"
    )


# =============================================================================
# [6/10] CALCULATE UNIFIED SHAP
# =============================================================================

print(
    "\n[6/10] Calculating unified permutation SHAP..."
)

print(
    "All models use the same:"
)

print(
    "  - prediction output: positive-class probability"
)

print(
    "  - SHAP algorithm: permutation"
)

print(
    "  - evaluation observations: 300"
)

print(
    "  - background observations: 100"
)

print(
    "  - feature ordering: identical"
)


shap_values = {}

explainer_metadata = {}


for model_name, model in models.items():

    print("\n" + "-" * 70)

    print(
        f"Calculating SHAP for {model_name}..."
    )

    # ---------------------------------------------------------------------
    # Model-specific prediction wrapper
    # ---------------------------------------------------------------------

    def prediction_function(data):
        return positive_probability(
            model,
            data
        )

    # ---------------------------------------------------------------------
    # Common SHAP explainer
    # ---------------------------------------------------------------------

    explainer = shap.Explainer(
        prediction_function,
        X_background,
        algorithm="permutation"
    )

    # ---------------------------------------------------------------------
    # Calculate SHAP values
    # ---------------------------------------------------------------------

    explanation = explainer(
        X_shap,
        max_evals=MAX_EVALS
    )

    values = np.asarray(
        explanation.values
    )

    # Some SHAP versions may return
    # (samples, features, 1)
    if values.ndim == 3:

        if values.shape[-1] == 1:
            values = values[:, :, 0]

        else:
            raise ValueError(
                f"Unexpected SHAP output shape "
                f"for {model_name}: {values.shape}"
            )

    if values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP output shape "
            f"for {model_name}: {values.shape}"
        )

    if values.shape != (
        SHAP_SAMPLE_SIZE,
        N_FEATURES
    ):

        raise ValueError(
            f"{model_name} SHAP shape "
            f"{values.shape} does not match "
            f"expected "
            f"({SHAP_SAMPLE_SIZE}, {N_FEATURES})."
        )

    shap_values[
        model_name
    ] = values

    explainer_metadata[
        model_name
    ] = {
        "algorithm": "permutation",
        "prediction_output": "positive_class_probability",
        "max_evals": MAX_EVALS,
        "shap_shape": list(
            values.shape
        )
    }

    print(
        f"{model_name} SHAP shape: "
        f"{values.shape}"
    )


# =============================================================================
# [7/10] GLOBAL SHAP CONSISTENCY
# =============================================================================

print(
    "\n[7/10] Calculating global explanation consistency..."
)


global_importance = {}

for model_name in models:

    importance = np.mean(
        np.abs(
            shap_values[model_name]
        ),
        axis=0
    )

    global_importance[
        model_name
    ] = importance


# -------------------------------------------------------------------------
# Pairwise global comparisons
# -------------------------------------------------------------------------

model_names = list(
    models.keys()
)

pairwise_global = []

for model_a, model_b in combinations(
    model_names,
    2
):

    importance_a = (
        global_importance[model_a]
    )

    importance_b = (
        global_importance[model_b]
    )

    rho = calculate_spearman(
        importance_a,
        importance_b
    )

    cosine = calculate_cosine(
        importance_a,
        importance_b
    )

    pairwise_global.append({

        "Model_A": model_a,

        "Model_B": model_b,

        "Global_Spearman": rho,

        "Global_Cosine": cosine,

        "Number_of_Features": N_FEATURES
    })


global_df = pd.DataFrame(
    pairwise_global
)


print(
    "\nGLOBAL CROSS-MODEL CONSISTENCY"
)

print(
    "-" * 70
)

for _, row in global_df.iterrows():

    print(
        f"{row['Model_A']}-{row['Model_B']}: "
        f"Spearman={row['Global_Spearman']:.6f}, "
        f"Cosine={row['Global_Cosine']:.6f}"
    )


# =============================================================================
# GLOBAL FEATURE IMPORTANCE TABLE
# =============================================================================

global_importance_df = pd.DataFrame(
    global_importance,
    index=FEATURES
)

global_importance_df.index.name = (
    "Feature"
)


# =============================================================================
# [8/10] LOCAL SHAP CONSISTENCY
# =============================================================================

print(
    "\n[8/10] Calculating local explanation consistency..."
)


local_records = []


for model_a, model_b in combinations(
    model_names,
    2
):

    shap_a = shap_values[
        model_a
    ]

    shap_b = shap_values[
        model_b
    ]

    for i in range(
        SHAP_SAMPLE_SIZE
    ):

        vector_a = shap_a[i]
        vector_b = shap_b[i]

        rho = calculate_spearman(
            vector_a,
            vector_b
        )

        cosine = calculate_cosine(
            vector_a,
            vector_b
        )

        local_records.append({

            "Model_A": model_a,

            "Model_B": model_b,

            "SHAP_Observation": i,

            "Original_Test_Index":
                int(shap_indices[i]),

            "Local_Spearman": rho,

            "Local_Cosine": cosine
        })


local_df = pd.DataFrame(
    local_records
)


# =============================================================================
# LOCAL SUMMARY
# =============================================================================

local_summary_records = []


for (
    model_a,
    model_b
), group in local_df.groupby(
    ["Model_A", "Model_B"]
):

    spearman_summary = summarize(
        group[
            "Local_Spearman"
        ].values
    )

    cosine_summary = summarize(
        group[
            "Local_Cosine"
        ].values
    )

    local_summary_records.append({

        "Model_A": model_a,

        "Model_B": model_b,

        "Local_Spearman_Mean":
            spearman_summary["mean"],

        "Local_Spearman_STD":
            spearman_summary["std"],

        "Local_Spearman_Median":
            spearman_summary["median"],

        "Local_Spearman_Min":
            spearman_summary["min"],

        "Local_Spearman_Max":
            spearman_summary["max"],

        "Local_Cosine_Mean":
            cosine_summary["mean"],

        "Local_Cosine_STD":
            cosine_summary["std"],

        "Local_Cosine_Median":
            cosine_summary["median"],

        "Local_Cosine_Min":
            cosine_summary["min"],

        "Local_Cosine_Max":
            cosine_summary["max"],

        "N_Observations":
            len(group)
    })


local_summary_df = pd.DataFrame(
    local_summary_records
)


print(
    "\nLOCAL CROSS-MODEL CONSISTENCY"
)

print(
    "-" * 70
)

for _, row in local_summary_df.iterrows():

    print(
        f"{row['Model_A']}-{row['Model_B']}: "
        f"Spearman mean="
        f"{row['Local_Spearman_Mean']:.6f}, "
        f"Cosine mean="
        f"{row['Local_Cosine_Mean']:.6f}"
    )


# =============================================================================
# [9/10] SAVE RESULTS
# =============================================================================

print(
    "\n[9/10] Saving results..."
)


# -------------------------------------------------------------------------
# Global pairwise results
# -------------------------------------------------------------------------

global_path = os.path.join(
    OUTPUT_DIR,
    "unified_global_shap_consistency.csv"
)

global_df.to_csv(
    global_path,
    index=False
)


# -------------------------------------------------------------------------
# Local results
# -------------------------------------------------------------------------

local_path = os.path.join(
    OUTPUT_DIR,
    "unified_local_shap_consistency.csv"
)

local_df.to_csv(
    local_path,
    index=False
)


# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------

summary_path = os.path.join(
    OUTPUT_DIR,
    "unified_cross_model_summary.csv"
)

local_summary_df.to_csv(
    summary_path,
    index=False
)


# -------------------------------------------------------------------------
# Feature importance
# -------------------------------------------------------------------------

feature_importance_path = os.path.join(
    OUTPUT_DIR,
    "unified_global_feature_importance.csv"
)

global_importance_df.to_csv(
    feature_importance_path
)


# =============================================================================
# METADATA
# =============================================================================

metadata = {

    "experiment":
        "Unified Cross-Model SHAP Consistency",

    "random_state":
        RANDOM_STATE,

    "test_dataset":
        TEST_PATH,

    "test_rows":
        int(len(X_test)),

    "number_of_features":
        int(N_FEATURES),

    "shap_sample_size":
        int(SHAP_SAMPLE_SIZE),

    "background_size":
        int(BACKGROUND_SIZE),

    "max_evals":
        int(MAX_EVALS),

    "models":
        model_names,

    "features":
        FEATURES,

    "prediction_output":
        "positive_class_probability",

    "shap_algorithm":
        "permutation",

    "explainer_metadata":
        explainer_metadata,

    "methodological_note":
        (
            "All four models were explained using the same permutation "
            "SHAP framework and the same positive-class probability "
            "prediction output. The same 300 test observations, 100 "
            "background observations, feature ordering, and evaluation "
            "budget were used across models."
        )
}


metadata_path = os.path.join(
    OUTPUT_DIR,
    "unified_cross_model_metadata.json"
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
# TEXT REPORT
# =============================================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "unified_cross_model_report.txt"
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
        "UNIFIED CROSS-MODEL SHAP CONSISTENCY ANALYSIS\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        "Experimental design\n"
    )

    f.write(
        "-" * 50 + "\n"
    )

    f.write(
        f"Test rows: {len(X_test)}\n"
    )

    f.write(
        f"Features: {N_FEATURES}\n"
    )

    f.write(
        f"SHAP observations: {SHAP_SAMPLE_SIZE}\n"
    )

    f.write(
        f"Background observations: {BACKGROUND_SIZE}\n"
    )

    f.write(
        f"Prediction output: positive-class probability\n"
    )

    f.write(
        f"SHAP algorithm: permutation\n"
    )

    f.write(
        f"Maximum evaluations: {MAX_EVALS}\n\n"
    )

    f.write(
        "GLOBAL CONSISTENCY\n"
    )

    f.write(
        "-" * 50 + "\n"
    )

    for _, row in global_df.iterrows():

        f.write(
            f"{row['Model_A']}-{row['Model_B']}: "
            f"Spearman="
            f"{row['Global_Spearman']:.6f}, "
            f"Cosine="
            f"{row['Global_Cosine']:.6f}\n"
        )

    f.write(
        "\nLOCAL CONSISTENCY\n"
    )

    f.write(
        "-" * 50 + "\n"
    )

    for _, row in local_summary_df.iterrows():

        f.write(
            f"{row['Model_A']}-{row['Model_B']}: "
            f"Spearman mean="
            f"{row['Local_Spearman_Mean']:.6f}, "
            f"Cosine mean="
            f"{row['Local_Cosine_Mean']:.6f}\n"
        )

    f.write(
        "\nMETHODOLOGICAL NOTE\n"
    )

    f.write(
        "-" * 50 + "\n"
    )

    f.write(
        "This analysis replaces the previous cross-model SHAP "
        "comparison because all models are now explained using "
        "the same permutation SHAP methodology and the same "
        "positive-class probability output.\n"
    )


# =============================================================================
# [10/10] CREATE FIGURES
# =============================================================================

print(
    "\n[10/10] Creating figures..."
)


# =============================================================================
# FIGURE 1 — GLOBAL SHAP FEATURE IMPORTANCE
# =============================================================================

plt.figure(
    figsize=(12, 8)
)

x = np.arange(
    N_FEATURES
)

width = 0.18

for j, model_name in enumerate(
    model_names
):

    plt.bar(
        x + (
            j - (len(model_names) - 1) / 2
        ) * width,

        global_importance[
            model_name
        ],

        width,

        label=model_name
    )


plt.xticks(
    x,
    FEATURES,
    rotation=75,
    ha="right"
)

plt.xlabel(
    "Feature"
)

plt.ylabel(
    "Mean Absolute SHAP Value"
)

plt.title(
    "Unified Global SHAP Feature Importance Across Models"
)

plt.legend()

plt.tight_layout()

global_importance_fig = os.path.join(
    FIGURE_DIR,
    "unified_global_shap_importance.png"
)

plt.savefig(
    global_importance_fig,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# FIGURE 2 — GLOBAL CONSISTENCY HEATMAP
# =============================================================================

global_heatmap = pd.DataFrame(
    np.eye(len(model_names)),
    index=model_names,
    columns=model_names
)

for _, row in global_df.iterrows():

    a = row["Model_A"]
    b = row["Model_B"]

    value = row[
        "Global_Spearman"
    ]

    global_heatmap.loc[
        a,
        b
    ] = value

    global_heatmap.loc[
        b,
        a
    ] = value


plt.figure(
    figsize=(7, 6)
)

# Option A: Use 'coolwarm' - clean scientific colormap
im = plt.imshow(
    global_heatmap.values,
    vmin=-1,
    vmax=1,
    cmap='coolwarm'  # or 'RdBu', 'seismic'
)

plt.xticks(
    range(len(model_names)),
    model_names,
    fontsize=11
)

plt.yticks(
    range(len(model_names)),
    model_names,
    fontsize=11
)

cbar = plt.colorbar(
    im,
    label="Spearman Correlation",
    shrink=0.8
)
cbar.ax.tick_params(labelsize=10)

# Add text annotations with improved visibility
for i in range(len(model_names)):
    for j in range(len(model_names)):
        value = global_heatmap.iloc[i, j]
        # Determine text color based on background
        if abs(value) > 0.4:
            text_color = 'white'
        else:
            text_color = 'black'
        
        plt.text(
            j,
            i,
            f"{value:.3f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=10
        )

plt.tight_layout()

heatmap_fig = os.path.join(
    FIGURE_DIR,
    "unified_global_consistency_heatmap.png"
)

plt.savefig(
    heatmap_fig,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# FIGURE 3 — LOCAL SPEARMAN
# =============================================================================

plt.figure(
    figsize=(11, 7)
)

pair_labels = []

spearman_data = []

for _, row in local_summary_df.iterrows():

    pair = (
        f"{row['Model_A']}-"
        f"{row['Model_B']}"
    )

    pair_labels.append(
        pair
    )

    spearman_data.append(
        local_df[
            (
                local_df["Model_A"]
                == row["Model_A"]
            )
            &
            (
                local_df["Model_B"]
                == row["Model_B"]
            )
        ]["Local_Spearman"]
        .dropna()
        .values
    )


plt.boxplot(
    spearman_data,
    tick_labels=pair_labels
)

plt.ylabel(
    "Local Spearman Correlation"
)

plt.xlabel(
    "Model Pair"
)

plt.title(
    "Local SHAP Spearman Consistency Across Models"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

spearman_fig = os.path.join(
    FIGURE_DIR,
    "unified_local_spearman_boxplot.png"
)

plt.savefig(
    spearman_fig,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# FIGURE 4 — LOCAL COSINE
# =============================================================================

plt.figure(
    figsize=(11, 7)
)

cosine_data = []

for _, row in local_summary_df.iterrows():

    cosine_data.append(
        local_df[
            (
                local_df["Model_A"]
                == row["Model_A"]
            )
            &
            (
                local_df["Model_B"]
                == row["Model_B"]
            )
        ]["Local_Cosine"]
        .dropna()
        .values
    )


plt.boxplot(
    cosine_data,
    tick_labels=pair_labels
)

plt.ylabel(
    "Local Cosine Similarity"
)

plt.xlabel(
    "Model Pair"
)

plt.title(
    "Local SHAP Cosine Consistency Across Models"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

cosine_fig = os.path.join(
    FIGURE_DIR,
    "unified_local_cosine_boxplot.png"
)

plt.savefig(
    cosine_fig,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# FINAL OUTPUT
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 15 COMPLETED")
print("=" * 80)

print(
    "\nUNIFIED CROSS-MODEL SHAP ANALYSIS:"
)

print(
    "  Models: RF, XGB, LGBM, MLP"
)

print(
    "  SHAP method: permutation"
)

print(
    "  Prediction output: positive-class probability"
)

print(
    f"  SHAP observations: {SHAP_SAMPLE_SIZE}"
)

print(
    f"  Background observations: {BACKGROUND_SIZE}"
)

print(
    f"  Features: {N_FEATURES}"
)

print(
    "\nGLOBAL CONSISTENCY:"
)

for _, row in global_df.iterrows():

    print(
        f"  {row['Model_A']}-{row['Model_B']}: "
        f"Spearman="
        f"{row['Global_Spearman']:.6f}, "
        f"Cosine="
        f"{row['Global_Cosine']:.6f}"
    )


print(
    "\nLOCAL CONSISTENCY:"
)

for _, row in local_summary_df.iterrows():

    print(
        f"  {row['Model_A']}-{row['Model_B']}: "
        f"Spearman="
        f"{row['Local_Spearman_Mean']:.6f}, "
        f"Cosine="
        f"{row['Local_Cosine_Mean']:.6f}"
    )


print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nFigure directory:"
)

print(
    FIGURE_DIR
)

print(
    "\nIMPORTANT:"
)

print(
    "These unified results should replace the previous "
    "cross-model SHAP results if they complete successfully."
)

print(
    "\nNEXT STEP:"
)

print(
    "Send me the complete terminal output before we calculate "
    "the final cross-model consistency index / SSI."
)

print("=" * 80)
