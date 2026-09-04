"""
================================================================================
STEP 17 — CROSS-MODEL EXPLANATION CONSISTENCY INDEX (CECI)
================================================================================

Purpose:
    Calculate a transparent Cross-Model Explanation Consistency Index (CECI)
    from the unified permutation-SHAP analysis.

Input:
    Step 15:
        unified_global_shap_consistency.csv
        unified_local_shap_consistency.csv

    Step 16:
        cross_model_shap_statistical_validation.csv

Methodology:
    Six model-pair comparisons:

        RF-XGB
        RF-LGBM
        RF-MLP
        XGB-LGBM
        XGB-MLP
        LGBM-MLP

    Four consistency dimensions:

        1. Global Spearman
        2. Global Cosine
        3. Local Spearman
        4. Local Cosine

    All metrics are bounded in [-1, 1] and are normalized to [0, 1]:

        normalized_score = (score + 1) / 2

    Component scores:

        Global Spearman Component
        Global Cosine Component
        Local Spearman Component
        Local Cosine Component

    CECI:

        CECI =
            mean(
                Global Spearman,
                Global Cosine,
                Local Spearman,
                Local Cosine
            )

    Equal weighting is used for transparency and simplicity.

IMPORTANT:
    CECI measures cross-model explanation consistency.

    It is distinct from:
        - IMSI: Intra-Model Stability Index
        - RF-RID explanation preservation
        - Predictive performance

    CECI is NOT combined with IMSI in this script.

================================================================================
"""

import os
import json
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# -----------------------------------------------------------------------------
# Input directories
# -----------------------------------------------------------------------------

UNIFIED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "results",
    "unified_cross_model_shap"
)

STAT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "results",
    "statistical_validation_shap_consistency"
)


# -----------------------------------------------------------------------------
# Output directory
# -----------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "results",
    "cross_model_explanation_consistency_index"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# -----------------------------------------------------------------------------
# Input files
# -----------------------------------------------------------------------------

GLOBAL_FILE = os.path.join(
    UNIFIED_DIR,
    "unified_global_shap_consistency.csv"
)

LOCAL_FILE = os.path.join(
    UNIFIED_DIR,
    "unified_local_shap_consistency.csv"
)

STAT_FILE = os.path.join(
    STAT_DIR,
    "cross_model_shap_statistical_validation.csv"
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_similarity(value):
    """
    Normalize a similarity/correlation score from [-1, 1] to [0, 1].

        normalized = (value + 1) / 2
    """

    return (
        float(value) + 1.0
    ) / 2.0


def make_pair(model_a, model_b):
    """
    Construct deterministic model-pair label.
    """

    return (
        str(model_a)
        + "-"
        + str(model_b)
    )


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("STEP 17 — CROSS-MODEL EXPLANATION CONSISTENCY INDEX (CECI)")
print("=" * 80)

print()
print("Purpose:")
print(
    "  Calculate a transparent cross-model explanation "
    "consistency index."
)

print()
print("Consistency dimensions:")
print("  1. Global Spearman")
print("  2. Global Cosine")
print("  3. Local Spearman")
print("  4. Local Cosine")

print()
print("Normalization:")
print("  score_normalized = (score + 1) / 2")

print()
print("Weighting:")
print("  Equal weighting across four dimensions")

print()
print("Important:")
print(
    "  CECI is kept separate from IMSI and RF-RID."
)


# =============================================================================
# [1/8] VERIFY INPUT FILES
# =============================================================================

print()
print("[1/8] Verifying input files...")

required_files = [
    GLOBAL_FILE,
    LOCAL_FILE,
    STAT_FILE
]

for file_path in required_files:

    if not os.path.exists(
        file_path
    ):

        raise FileNotFoundError(
            f"Required input file not found:\n"
            f"{file_path}"
        )

    print(
        f"Found: {os.path.basename(file_path)}"
    )


# =============================================================================
# [2/8] LOAD STEP 15 RESULTS
# =============================================================================

print()
print("[2/8] Loading unified SHAP consistency results...")

global_df = pd.read_csv(
    GLOBAL_FILE
)

local_df = pd.read_csv(
    LOCAL_FILE
)

stat_df = pd.read_csv(
    STAT_FILE
)

print(
    f"Global results shape: {global_df.shape}"
)

print(
    f"Local results shape : {local_df.shape}"
)

print(
    f"Statistical results : {stat_df.shape}"
)


# =============================================================================
# [3/8] VALIDATE GLOBAL DATA
# =============================================================================

print()
print("[3/8] Validating global consistency results...")

print()
print("Global columns:")

for column in global_df.columns:

    print(
        f"  - {column}"
    )


required_global_columns = [
    "Model_A",
    "Model_B",
    "Global_Spearman",
    "Global_Cosine"
]

missing_global = [
    column
    for column in required_global_columns
    if column not in global_df.columns
]

if missing_global:

    raise ValueError(
        "Missing global columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_global
        )
    )


# Create pair identifier
global_df["Model_Pair"] = (
    global_df["Model_A"].astype(str)
    + "-"
    + global_df["Model_B"].astype(str)
)

# Numeric conversion
global_df["Global_Spearman"] = pd.to_numeric(
    global_df["Global_Spearman"],
    errors="coerce"
)

global_df["Global_Cosine"] = pd.to_numeric(
    global_df["Global_Cosine"],
    errors="coerce"
)

# Validate
if global_df[
    [
        "Global_Spearman",
        "Global_Cosine"
    ]
].isna().any().any():

    raise ValueError(
        "NaN values detected in global consistency results."
    )


print()
print(
    f"Global model pairs: "
    f"{len(global_df)}"
)


# =============================================================================
# [4/8] CALCULATE LOCAL SUMMARY
# =============================================================================

print()
print("[4/8] Calculating local consistency summaries...")

required_local_columns = [
    "Model_A",
    "Model_B",
    "Local_Spearman",
    "Local_Cosine"
]

missing_local = [
    column
    for column in required_local_columns
    if column not in local_df.columns
]

if missing_local:

    raise ValueError(
        "Missing local columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_local
        )
    )


local_df["Model_Pair"] = (
    local_df["Model_A"].astype(str)
    + "-"
    + local_df["Model_B"].astype(str)
)

local_df["Local_Spearman"] = pd.to_numeric(
    local_df["Local_Spearman"],
    errors="coerce"
)

local_df["Local_Cosine"] = pd.to_numeric(
    local_df["Local_Cosine"],
    errors="coerce"
)

local_df = local_df.dropna(
    subset=[
        "Local_Spearman",
        "Local_Cosine"
    ]
)


local_summary = (
    local_df
    .groupby("Model_Pair")
    .agg(
        Local_Spearman_Mean=(
            "Local_Spearman",
            "mean"
        ),

        Local_Spearman_Median=(
            "Local_Spearman",
            "median"
        ),

        Local_Cosine_Mean=(
            "Local_Cosine",
            "mean"
        ),

        Local_Cosine_Median=(
            "Local_Cosine",
            "median"
        ),

        n_observations=(
            "Local_Spearman",
            "count"
        )
    )
    .reset_index()
)


print()
print("Local summary:")

for _, row in local_summary.iterrows():

    print(
        f"  {row['Model_Pair']}: "
        f"n={int(row['n_observations'])}, "
        f"Spearman={row['Local_Spearman_Mean']:.6f}, "
        f"Cosine={row['Local_Cosine_Mean']:.6f}"
    )


# =============================================================================
# [5/8] MERGE GLOBAL + LOCAL RESULTS
# =============================================================================

print()
print("[5/8] Combining global and local consistency results...")

global_summary = global_df[
    [
        "Model_Pair",
        "Global_Spearman",
        "Global_Cosine"
    ]
].copy()


combined = pd.merge(
    global_summary,
    local_summary[
        [
            "Model_Pair",
            "Local_Spearman_Mean",
            "Local_Spearman_Median",
            "Local_Cosine_Mean",
            "Local_Cosine_Median",
            "n_observations"
        ]
    ],
    on="Model_Pair",
    how="inner"
)


if len(combined) != 6:

    raise ValueError(
        "Expected exactly 6 model-pair comparisons, "
        f"but found {len(combined)}."
    )


# =============================================================================
# [6/8] NORMALIZE METRICS + CALCULATE CECI
# =============================================================================

print()
print("[6/8] Calculating normalized components and CECI...")


# -----------------------------------------------------------------------------
# Normalize global metrics
# -----------------------------------------------------------------------------

combined["Global_Spearman_Normalized"] = (
    combined["Global_Spearman"]
    .apply(normalize_similarity)
)

combined["Global_Cosine_Normalized"] = (
    combined["Global_Cosine"]
    .apply(normalize_similarity)
)


# -----------------------------------------------------------------------------
# Normalize local metrics
# -----------------------------------------------------------------------------

combined["Local_Spearman_Normalized"] = (
    combined["Local_Spearman_Mean"]
    .apply(normalize_similarity)
)

combined["Local_Cosine_Normalized"] = (
    combined["Local_Cosine_Mean"]
    .apply(normalize_similarity)
)


# -----------------------------------------------------------------------------
# Component scores
# -----------------------------------------------------------------------------

combined["Global_CECI_Component"] = (
    combined[
        [
            "Global_Spearman_Normalized",
            "Global_Cosine_Normalized"
        ]
    ]
    .mean(axis=1)
)

combined["Local_CECI_Component"] = (
    combined[
        [
            "Local_Spearman_Normalized",
            "Local_Cosine_Normalized"
        ]
    ]
    .mean(axis=1)
)


# -----------------------------------------------------------------------------
# Pairwise CECI
# -----------------------------------------------------------------------------

combined["Pairwise_CECI"] = (
    combined[
        [
            "Global_Spearman_Normalized",
            "Global_Cosine_Normalized",
            "Local_Spearman_Normalized",
            "Local_Cosine_Normalized"
        ]
    ]
    .mean(axis=1)
)


# =============================================================================
# OVERALL CECI
# =============================================================================

global_spearman_mean = (
    combined["Global_Spearman_Normalized"]
    .mean()
)

global_cosine_mean = (
    combined["Global_Cosine_Normalized"]
    .mean()
)

local_spearman_mean = (
    combined["Local_Spearman_Normalized"]
    .mean()
)

local_cosine_mean = (
    combined["Local_Cosine_Normalized"]
    .mean()
)


global_ceci = (
    global_spearman_mean
    + global_cosine_mean
) / 2.0


local_ceci = (
    local_spearman_mean
    + local_cosine_mean
) / 2.0


ceci = (
    global_ceci
    + local_ceci
) / 2.0


# =============================================================================
# RANK MODEL PAIRS
# =============================================================================

combined = combined.sort_values(
    "Pairwise_CECI",
    ascending=False
).reset_index(
    drop=True
)

combined["CECI_Rank"] = (
    np.arange(
        1,
        len(combined) + 1
    )
)


# =============================================================================
# [7/8] DISPLAY RESULTS
# =============================================================================

print()
print("=" * 80)
print("PAIRWISE CROSS-MODEL CECI RESULTS")
print("=" * 80)

for _, row in combined.iterrows():

    print()
    print(
        f"{row['CECI_Rank']}. "
        f"{row['Model_Pair']}"
    )

    print(
        f"   Global Spearman = "
        f"{row['Global_Spearman']:.6f}"
    )

    print(
        f"   Global Cosine   = "
        f"{row['Global_Cosine']:.6f}"
    )

    print(
        f"   Local Spearman  = "
        f"{row['Local_Spearman_Mean']:.6f}"
    )

    print(
        f"   Local Cosine    = "
        f"{row['Local_Cosine_Mean']:.6f}"
    )

    print(
        f"   Global component = "
        f"{row['Global_CECI_Component']:.6f}"
    )

    print(
        f"   Local component  = "
        f"{row['Local_CECI_Component']:.6f}"
    )

    print(
        f"   Pairwise CECI    = "
        f"{row['Pairwise_CECI']:.6f}"
    )


# =============================================================================
# OVERALL CECI DISPLAY
# =============================================================================

print()
print("=" * 80)
print("OVERALL CROSS-MODEL EXPLANATION CONSISTENCY INDEX")
print("=" * 80)

print()
print(
    f"Global Spearman component = "
    f"{global_spearman_mean:.6f}"
)

print(
    f"Global Cosine component   = "
    f"{global_cosine_mean:.6f}"
)

print(
    f"Local Spearman component  = "
    f"{local_spearman_mean:.6f}"
)

print(
    f"Local Cosine component    = "
    f"{local_cosine_mean:.6f}"
)

print()
print(
    f"Global CECI = "
    f"{global_ceci:.6f}"
)

print(
    f"Local CECI  = "
    f"{local_ceci:.6f}"
)

print()
print(
    f"CECI = "
    f"{ceci:.6f}"
)

print()
print(
    f"CECI percentage = "
    f"{ceci * 100:.2f}%"
)


# =============================================================================
# [8/8] SAVE RESULTS
# =============================================================================

print()
print("[8/8] Saving CECI results...")


# -----------------------------------------------------------------------------
# Pairwise results
# -----------------------------------------------------------------------------

PAIRWISE_FILE = os.path.join(
    OUTPUT_DIR,
    "pairwise_ceci_results.csv"
)

combined.to_csv(
    PAIRWISE_FILE,
    index=False
)


# -----------------------------------------------------------------------------
# Component summary
# -----------------------------------------------------------------------------

component_summary = pd.DataFrame(
    [
        {
            "Component":
                "Global Spearman",

            "Normalized_Score":
                global_spearman_mean,

            "Weight":
                0.25
        },

        {
            "Component":
                "Global Cosine",

            "Normalized_Score":
                global_cosine_mean,

            "Weight":
                0.25
        },

        {
            "Component":
                "Local Spearman",

            "Normalized_Score":
                local_spearman_mean,

            "Weight":
                0.25
        },

        {
            "Component":
                "Local Cosine",

            "Normalized_Score":
                local_cosine_mean,

            "Weight":
                0.25
        }
    ]
)

COMPONENT_FILE = os.path.join(
    OUTPUT_DIR,
    "ceci_component_summary.csv"
)

component_summary.to_csv(
    COMPONENT_FILE,
    index=False
)


# -----------------------------------------------------------------------------
# Overall CECI
# -----------------------------------------------------------------------------

overall_summary = pd.DataFrame(
    [
        {
            "Global_CECI":
                global_ceci,

            "Local_CECI":
                local_ceci,

            "Overall_CECI":
                ceci,

            "CECI_Percentage":
                ceci * 100,

            "Weight_Global":
                0.50,

            "Weight_Local":
                0.50,

            "Weight_Each_Metric":
                0.25,

            "Number_Model_Pairs":
                len(combined)
        }
    ]
)

OVERALL_FILE = os.path.join(
    OUTPUT_DIR,
    "overall_ceci_summary.csv"
)

overall_summary.to_csv(
    OVERALL_FILE,
    index=False
)


# =============================================================================
# SAVE JSON METADATA
# =============================================================================

metadata = {
    "analysis":
        "Cross-Model Explanation Consistency Index",

    "dataset":
        "NASA JM1",

    "models":
        [
            "RF",
            "XGB",
            "LGBM",
            "MLP"
        ],

    "number_model_pairs":
        6,

    "shap_method":
        "Unified permutation SHAP",

    "prediction_output":
        "positive-class probability",

    "shap_observations":
        300,

    "background_observations":
        100,

    "metrics":
        [
            "Global Spearman",
            "Global Cosine",
            "Local Spearman",
            "Local Cosine"
        ],

    "normalization":
        "score_normalized = (score + 1) / 2",

    "weighting":
        {
            "global":
                0.50,

            "local":
                0.50,

            "each_metric":
                0.25
        },

    "global_ceci":
        global_ceci,

    "local_ceci":
        local_ceci,

    "overall_ceci":
        ceci,

    "overall_ceci_percentage":
        ceci * 100
}

METADATA_FILE = os.path.join(
    OUTPUT_DIR,
    "ceci_metadata.json"
)

with open(
    METADATA_FILE,
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

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "ceci_report.txt"
)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 17 — CROSS-MODEL EXPLANATION CONSISTENCY INDEX (CECI)\n"
    )

    f.write(
        "=" * 80
        + "\n\n"
    )

    f.write(
        "Methodology\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    f.write(
        "SHAP method: Unified permutation SHAP\n"
    )

    f.write(
        "Prediction output: Positive-class probability\n"
    )

    f.write(
        "SHAP observations: 300\n"
    )

    f.write(
        "Background observations: 100\n"
    )

    f.write(
        "Model pairs: 6\n"
    )

    f.write(
        "Metrics: Global Spearman, Global Cosine, "
        "Local Spearman, Local Cosine\n"
    )

    f.write(
        "Normalization: (score + 1) / 2\n"
    )

    f.write(
        "Weighting: Equal weighting\n\n"
    )

    f.write(
        "Overall CECI\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    f.write(
        f"Global CECI = "
        f"{global_ceci:.6f}\n"
    )

    f.write(
        f"Local CECI = "
        f"{local_ceci:.6f}\n"
    )

    f.write(
        f"Overall CECI = "
        f"{ceci:.6f}\n"
    )

    f.write(
        f"Overall CECI (%) = "
        f"{ceci * 100:.2f}%\n\n"
    )

    f.write(
        "Pairwise Results\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    for _, row in combined.iterrows():

        f.write(
            f"{int(row['CECI_Rank'])}. "
            f"{row['Model_Pair']}\n"
        )

        f.write(
            f"   Global Spearman: "
            f"{row['Global_Spearman']:.6f}\n"
        )

        f.write(
            f"   Global Cosine: "
            f"{row['Global_Cosine']:.6f}\n"
        )

        f.write(
            f"   Local Spearman: "
            f"{row['Local_Spearman_Mean']:.6f}\n"
        )

        f.write(
            f"   Local Cosine: "
            f"{row['Local_Cosine_Mean']:.6f}\n"
        )

        f.write(
            f"   Global component: "
            f"{row['Global_CECI_Component']:.6f}\n"
        )

        f.write(
            f"   Local component: "
            f"{row['Local_CECI_Component']:.6f}\n"
        )

        f.write(
            f"   Pairwise CECI: "
            f"{row['Pairwise_CECI']:.6f}\n\n"
        )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 80)
print("STEP 17 COMPLETED")
print("=" * 80)

print()
print(
    f"Overall CECI = {ceci:.6f}"
)

print(
    f"Overall CECI = {ceci * 100:.2f}%"
)

print()
print("Output directory:")
print(
    OUTPUT_DIR
)

print()
print("Files created:")

print(
    f"  {PAIRWISE_FILE}"
)

print(
    f"  {COMPONENT_FILE}"
)

print(
    f"  {OVERALL_FILE}"
)

print(
    f"  {METADATA_FILE}"
)

print(
    f"  {REPORT_FILE}"
)

print()
print("IMPORTANT:")
print(
    "  CECI is a cross-model explanation consistency measure."
)

print(
    "  It is intentionally kept separate from IMSI."
)

print(
    "  RF-RID explanation preservation is also kept separate."
)

print()
print("NEXT STEP:")
print(
    "  Review the CECI results together with IMSI and "
    "RF-RID explanation preservation before defining "
    "any overall SSI."
)

print("=" * 80)
