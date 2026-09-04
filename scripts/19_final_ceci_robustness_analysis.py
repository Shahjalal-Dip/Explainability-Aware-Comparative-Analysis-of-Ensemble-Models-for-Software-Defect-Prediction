"""
================================================================================
STEP 18 — FINAL CECI + ROBUSTNESS ANALYSIS
================================================================================

Purpose:
    Finalize the Cross-Model Explanation Consistency Index (CECI)
    and evaluate its robustness using bootstrap resampling of the
    local SHAP consistency observations.

Input:
    Step 15:
        data/results/unified_cross_model_shap/
            unified_global_shap_consistency.csv
            unified_local_shap_consistency.csv

    Step 17:
        data/results/cross_model_explanation_consistency_index/
            pairwise_ceci_results.csv
            ceci_component_summary.csv
            overall_ceci_summary.csv
            ceci_metadata.json
            ceci_report.txt

Method:
    1. Reconstruct pairwise CECI from the actual Step 17 methodology.
    2. Bootstrap local observations within each model pair.
    3. Recalculate pairwise CECI for every bootstrap sample.
    4. Recalculate overall CECI for every bootstrap sample.
    5. Calculate 95% bootstrap confidence intervals.
    6. Assess sensitivity to model-pair removal (leave-one-pair-out).
    7. Produce final robustness tables and report.

Important:
    - No model retraining.
    - No SHAP recomputation.
    - No SMOTE.
    - CECI remains separate from IMSI and RF-RID.
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
N_BOOTSTRAPS = 5000
CONFIDENCE_LEVEL = 0.95

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# =============================================================================
# DIRECTORIES
# =============================================================================

UNIFIED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "results",
    "unified_cross_model_shap"
)

CECI_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "results",
    "cross_model_explanation_consistency_index"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "results",
    "final_ceci_robustness"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =============================================================================
# INPUT FILES
# =============================================================================

GLOBAL_FILE = os.path.join(
    UNIFIED_DIR,
    "unified_global_shap_consistency.csv"
)

LOCAL_FILE = os.path.join(
    UNIFIED_DIR,
    "unified_local_shap_consistency.csv"
)

STEP17_PAIRWISE_FILE = os.path.join(
    CECI_DIR,
    "pairwise_ceci_results.csv"
)

STEP17_COMPONENT_FILE = os.path.join(
    CECI_DIR,
    "ceci_component_summary.csv"
)

STEP17_OVERALL_FILE = os.path.join(
    CECI_DIR,
    "overall_ceci_summary.csv"
)

STEP17_METADATA_FILE = os.path.join(
    CECI_DIR,
    "ceci_metadata.json"
)


# =============================================================================
# EXPECTED MODEL PAIRS
# =============================================================================

EXPECTED_PAIRS = [
    "RF-XGB",
    "RF-LGBM",
    "RF-MLP",
    "XGB-LGBM",
    "XGB-MLP",
    "LGBM-MLP"
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_similarity(value):
    """
    Normalize [-1, 1] similarity/correlation to [0, 1].
    """
    return (float(value) + 1.0) / 2.0


def bootstrap_mean(
    values,
    n_bootstraps=5000,
    confidence_level=0.95,
    random_state=42
):
    """
    Bootstrap the mean and return percentile confidence interval.
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
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan
        }

    rng = np.random.default_rng(
        random_state
    )

    bootstrap_means = np.empty(
        n_bootstraps
    )

    for i in range(n_bootstraps):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        bootstrap_means[i] = np.mean(
            sample
        )

    alpha = 1.0 - confidence_level

    lower = np.percentile(
        bootstrap_means,
        100 * (alpha / 2)
    )

    upper = np.percentile(
        bootstrap_means,
        100 * (1 - alpha / 2)
    )

    return {
        "n": len(values),
        "mean": np.mean(values),
        "std": (
            np.std(values, ddof=1)
            if len(values) > 1
            else 0.0
        ),
        "ci_lower": lower,
        "ci_upper": upper
    }


def calculate_pair_ceci(
    global_spearman,
    global_cosine,
    local_spearman,
    local_cosine
):
    """
    Calculate pairwise CECI using exactly the Step 17 formula.
    """

    global_spearman_norm = normalize_similarity(
        global_spearman
    )

    global_cosine_norm = normalize_similarity(
        global_cosine
    )

    local_spearman_norm = normalize_similarity(
        local_spearman
    )

    local_cosine_norm = normalize_similarity(
        local_cosine
    )

    global_component = (
        global_spearman_norm
        + global_cosine_norm
    ) / 2.0

    local_component = (
        local_spearman_norm
        + local_cosine_norm
    ) / 2.0

    pairwise_ceci = (
        global_component
        + local_component
    ) / 2.0

    return {
        "Global_Spearman_Normalized":
            global_spearman_norm,

        "Global_Cosine_Normalized":
            global_cosine_norm,

        "Local_Spearman_Normalized":
            local_spearman_norm,

        "Local_Cosine_Normalized":
            local_cosine_norm,

        "Global_CECI_Component":
            global_component,

        "Local_CECI_Component":
            local_component,

        "Pairwise_CECI":
            pairwise_ceci
    }


def build_model_pair(df):
    """
    Create model-pair labels using the same Model_A / Model_B
    representation used in Step 17.
    """

    df["Model_A"] = df["Model_A"].astype(str).str.strip()
    df["Model_B"] = df["Model_B"].astype(str).str.strip()

    df["Model_Pair"] = (
        df["Model_A"]
        + "-"
        + df["Model_B"]
    )

    return df


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("STEP 18 — FINAL CECI + ROBUSTNESS ANALYSIS")
print("=" * 80)

print()

print("Purpose:")
print(
    "  Finalize CECI and evaluate its robustness "
    "using bootstrap resampling."
)

print()

print("Bootstrap repetitions:")
print(
    f"  {N_BOOTSTRAPS}"
)

print()

print("Confidence level:")
print(
    f"  {CONFIDENCE_LEVEL * 100:.0f}%"
)

print()

print("Random state:")
print(
    f"  {RANDOM_STATE}"
)


# =============================================================================
# [1/9] VERIFY FILES
# =============================================================================

print()
print("[1/9] Verifying input files...")

required_files = [
    GLOBAL_FILE,
    LOCAL_FILE,
    STEP17_PAIRWISE_FILE,
    STEP17_COMPONENT_FILE,
    STEP17_OVERALL_FILE,
    STEP17_METADATA_FILE
]

for file_path in required_files:

    if not os.path.exists(
        file_path
    ):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )

    print(
        f"Found: {file_path}"
    )


# =============================================================================
# [2/9] LOAD RESULTS
# =============================================================================

print()
print("[2/9] Loading Step 15 and Step 17 results...")

global_df = pd.read_csv(
    GLOBAL_FILE
)

local_df = pd.read_csv(
    LOCAL_FILE
)

step17_pairwise_df = pd.read_csv(
    STEP17_PAIRWISE_FILE
)

step17_component_df = pd.read_csv(
    STEP17_COMPONENT_FILE
)

step17_overall_df = pd.read_csv(
    STEP17_OVERALL_FILE
)

with open(
    STEP17_METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


print()
print(
    f"Global SHAP consistency shape: "
    f"{global_df.shape}"
)

print(
    f"Local SHAP consistency shape : "
    f"{local_df.shape}"
)

print(
    f"Step 17 pairwise shape        : "
    f"{step17_pairwise_df.shape}"
)


# =============================================================================
# [3/9] VALIDATE INPUT STRUCTURE
# =============================================================================

print()
print("[3/9] Validating input structure...")

required_global_columns = [
    "Model_A",
    "Model_B",
    "Global_Spearman",
    "Global_Cosine"
]

required_local_columns = [
    "Model_A",
    "Model_B",
    "Local_Spearman",
    "Local_Cosine"
]

for column in required_global_columns:

    if column not in global_df.columns:

        raise ValueError(
            f"Missing global column: {column}"
        )


for column in required_local_columns:

    if column not in local_df.columns:

        raise ValueError(
            f"Missing local column: {column}"
        )


global_df = build_model_pair(
    global_df
)

local_df = build_model_pair(
    local_df
)


global_df["Global_Spearman"] = pd.to_numeric(
    global_df["Global_Spearman"],
    errors="coerce"
)

global_df["Global_Cosine"] = pd.to_numeric(
    global_df["Global_Cosine"],
    errors="coerce"
)

local_df["Local_Spearman"] = pd.to_numeric(
    local_df["Local_Spearman"],
    errors="coerce"
)

local_df["Local_Cosine"] = pd.to_numeric(
    local_df["Local_Cosine"],
    errors="coerce"
)


global_df = global_df.dropna(
    subset=[
        "Global_Spearman",
        "Global_Cosine"
    ]
)

local_df = local_df.dropna(
    subset=[
        "Local_Spearman",
        "Local_Cosine"
    ]
)


# =============================================================================
# [4/9] VERIFY SIX MODEL PAIRS
# =============================================================================

print()
print("[4/9] Verifying six model-pair comparisons...")

global_pairs = set(
    global_df["Model_Pair"].unique()
)

local_pairs = set(
    local_df["Model_Pair"].unique()
)

print()
print("Global pairs:")
for pair in sorted(global_pairs):
    print(
        f"  {pair}"
    )

print()
print("Local pairs:")
for pair in sorted(local_pairs):
    print(
        f"  {pair}"
    )


missing_global_pairs = [
    pair
    for pair in EXPECTED_PAIRS
    if pair not in global_pairs
]

missing_local_pairs = [
    pair
    for pair in EXPECTED_PAIRS
    if pair not in local_pairs
]

if missing_global_pairs:

    raise ValueError(
        "Missing global model pairs:\n"
        + "\n".join(
            missing_global_pairs
        )
    )

if missing_local_pairs:

    raise ValueError(
        "Missing local model pairs:\n"
        + "\n".join(
            missing_local_pairs
        )
    )


# =============================================================================
# [5/9] RECONSTRUCT STEP 17 CECI
# =============================================================================

print()
print("[5/9] Reconstructing Step 17 CECI...")

global_summary = (
    global_df
    .groupby("Model_Pair")
    .agg(
        Global_Spearman=(
            "Global_Spearman",
            "mean"
        ),
        Global_Cosine=(
            "Global_Cosine",
            "mean"
        )
    )
    .reset_index()
)


local_summary = (
    local_df
    .groupby("Model_Pair")
    .agg(
        Local_Spearman_Mean=(
            "Local_Spearman",
            "mean"
        ),
        Local_Cosine_Mean=(
            "Local_Cosine",
            "mean"
        ),
        Local_Spearman_Median=(
            "Local_Spearman",
            "median"
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


reconstructed = pd.merge(
    global_summary,
    local_summary,
    on="Model_Pair",
    how="inner"
)


if len(reconstructed) != 6:

    raise ValueError(
        "Expected six model pairs after merging, "
        f"found {len(reconstructed)}."
    )


rows = []

for _, row in reconstructed.iterrows():

    metrics = calculate_pair_ceci(
        row["Global_Spearman"],
        row["Global_Cosine"],
        row["Local_Spearman_Mean"],
        row["Local_Cosine_Mean"]
    )

    result = row.to_dict()

    result.update(
        metrics
    )

    rows.append(
        result
    )


reconstructed = pd.DataFrame(
    rows
)

reconstructed = reconstructed.sort_values(
    "Pairwise_CECI",
    ascending=False
).reset_index(
    drop=True
)

reconstructed["CECI_Rank"] = (
    np.arange(
        1,
        len(reconstructed) + 1
    )
)


# =============================================================================
# COMPARE AGAINST STEP 17
# =============================================================================

print()
print("Comparing reconstructed CECI with Step 17...")

if "Model_Pair" in step17_pairwise_df.columns:

    comparison = pd.merge(
        reconstructed[
            [
                "Model_Pair",
                "Pairwise_CECI"
            ]
        ],
        step17_pairwise_df[
            [
                "Model_Pair",
                "Pairwise_CECI"
            ]
        ],
        on="Model_Pair",
        suffixes=(
            "_Step18",
            "_Step17"
        )
    )

    comparison["Absolute_Difference"] = (
        comparison["Pairwise_CECI_Step18"]
        - comparison["Pairwise_CECI_Step17"]
    ).abs()

    print()

    print(
        comparison.to_string(
            index=False
        )
    )

    max_difference = (
        comparison["Absolute_Difference"]
        .max()
    )

    print()

    print(
        f"Maximum absolute difference: "
        f"{max_difference:.12f}"
    )

    if max_difference > 1e-10:

        print(
            "WARNING: Step 18 reconstruction "
            "differs from Step 17."
        )

    else:

        print(
            "PASS: Step 18 reproduces "
            "Step 17 CECI exactly."
        )


# =============================================================================
# [6/9] BOOTSTRAP PAIRWISE CECI
# =============================================================================

print()
print("[6/9] Running bootstrap robustness analysis...")


rng = np.random.default_rng(
    RANDOM_STATE
)


bootstrap_pair_results = []

bootstrap_overall_values = []

bootstrap_global_values = []
bootstrap_local_values = []

bootstrap_component_values = {
    "Global_Spearman": [],
    "Global_Cosine": [],
    "Local_Spearman": [],
    "Local_Cosine": []
}


# -----------------------------------------------------------------------------
# Pre-store global values
# -----------------------------------------------------------------------------

global_values_by_pair = {}

for pair in EXPECTED_PAIRS:

    pair_rows = global_df[
        global_df["Model_Pair"] == pair
    ]

    global_values_by_pair[pair] = {
        "spearman":
            pair_rows["Global_Spearman"]
            .to_numpy(),

        "cosine":
            pair_rows["Global_Cosine"]
            .to_numpy()
    }


# -----------------------------------------------------------------------------
# Pre-store local values
# -----------------------------------------------------------------------------

local_values_by_pair = {}

for pair in EXPECTED_PAIRS:

    pair_rows = local_df[
        local_df["Model_Pair"] == pair
    ]

    local_values_by_pair[pair] = {
        "spearman":
            pair_rows["Local_Spearman"]
            .to_numpy(),

        "cosine":
            pair_rows["Local_Cosine"]
            .to_numpy()
    }


# -----------------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------------

for bootstrap_index in range(
    N_BOOTSTRAPS
):

    pair_ceci_values = []

    pair_global_values = []
    pair_local_values = []

    pair_global_spearman = []
    pair_global_cosine = []
    pair_local_spearman = []
    pair_local_cosine = []

    for pair in EXPECTED_PAIRS:

        # -------------------------------------------------------------
        # Global bootstrap
        # -------------------------------------------------------------

        global_spearman_values = (
            global_values_by_pair[pair]["spearman"]
        )

        global_cosine_values = (
            global_values_by_pair[pair]["cosine"]
        )

        if len(global_spearman_values) > 1:

            sampled_global_spearman = rng.choice(
                global_spearman_values,
                size=len(global_spearman_values),
                replace=True
            )

            sampled_global_cosine = rng.choice(
                global_cosine_values,
                size=len(global_cosine_values),
                replace=True
            )

        else:

            sampled_global_spearman = (
                global_spearman_values
            )

            sampled_global_cosine = (
                global_cosine_values
            )

        global_spearman_mean = np.mean(
            sampled_global_spearman
        )

        global_cosine_mean = np.mean(
            sampled_global_cosine
        )

        # -------------------------------------------------------------
        # Local bootstrap
        # -------------------------------------------------------------

        local_spearman_values = (
            local_values_by_pair[pair]["spearman"]
        )

        local_cosine_values = (
            local_values_by_pair[pair]["cosine"]
        )

        sampled_local_spearman = rng.choice(
            local_spearman_values,
            size=len(local_spearman_values),
            replace=True
        )

        sampled_local_cosine = rng.choice(
            local_cosine_values,
            size=len(local_cosine_values),
            replace=True
        )

        local_spearman_mean = np.mean(
            sampled_local_spearman
        )

        local_cosine_mean = np.mean(
            sampled_local_cosine
        )

        # -------------------------------------------------------------
        # Calculate pairwise CECI
        # -------------------------------------------------------------

        metrics = calculate_pair_ceci(
            global_spearman_mean,
            global_cosine_mean,
            local_spearman_mean,
            local_cosine_mean
        )

        pair_ceci = metrics[
            "Pairwise_CECI"
        ]

        pair_ceci_values.append(
            pair_ceci
        )

        pair_global_values.append(
            metrics["Global_CECI_Component"]
        )

        pair_local_values.append(
            metrics["Local_CECI_Component"]
        )

        pair_global_spearman.append(
            metrics[
                "Global_Spearman_Normalized"
            ]
        )

        pair_global_cosine.append(
            metrics[
                "Global_Cosine_Normalized"
            ]
        )

        pair_local_spearman.append(
            metrics[
                "Local_Spearman_Normalized"
            ]
        )

        pair_local_cosine.append(
            metrics[
                "Local_Cosine_Normalized"
            ]
        )

    # -------------------------------------------------------------------------
    # Overall bootstrap CECI
    # -------------------------------------------------------------------------

    overall_bootstrap_ceci = np.mean(
        pair_ceci_values
    )

    overall_global_ceci = np.mean(
        pair_global_values
    )

    overall_local_ceci = np.mean(
        pair_local_values
    )

    bootstrap_overall_values.append(
        overall_bootstrap_ceci
    )

    bootstrap_global_values.append(
        overall_global_ceci
    )

    bootstrap_local_values.append(
        overall_local_ceci
    )

    bootstrap_component_values[
        "Global_Spearman"
    ].append(
        np.mean(
            pair_global_spearman
        )
    )

    bootstrap_component_values[
        "Global_Cosine"
    ].append(
        np.mean(
            pair_global_cosine
        )
    )

    bootstrap_component_values[
        "Local_Spearman"
    ].append(
        np.mean(
            pair_local_spearman
        )
    )

    bootstrap_component_values[
        "Local_Cosine"
    ].append(
        np.mean(
            pair_local_cosine
        )
    )

    # Store pairwise values
    if bootstrap_index < 10:
        bootstrap_pair_results.append(
            pair_ceci_values
        )

    if (
        (bootstrap_index + 1) % 500 == 0
    ):

        print(
            f"  Completed "
            f"{bootstrap_index + 1:,}/"
            f"{N_BOOTSTRAPS:,} bootstrap samples"
        )


# =============================================================================
# [7/9] CALCULATE BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

print()
print("[7/9] Calculating bootstrap confidence intervals...")


alpha = (
    1.0 -
    CONFIDENCE_LEVEL
)


def percentile_ci(
    values
):

    values = np.asarray(
        values,
        dtype=float
    )

    return (
        np.percentile(
            values,
            100 * alpha / 2
        ),
        np.percentile(
            values,
            100 * (1 - alpha / 2)
        )
    )


overall_lower, overall_upper = percentile_ci(
    bootstrap_overall_values
)

global_lower, global_upper = percentile_ci(
    bootstrap_global_values
)

local_lower, local_upper = percentile_ci(
    bootstrap_local_values
)


# =============================================================================
# OVERALL ROBUSTNESS TABLE
# =============================================================================

overall_robustness = pd.DataFrame(
    [
        {
            "Metric":
                "Overall CECI",

            "Observed_Value":
                reconstructed[
                    "Pairwise_CECI"
                ].mean(),

            "Bootstrap_Mean":
                np.mean(
                    bootstrap_overall_values
                ),

            "Bootstrap_SD":
                np.std(
                    bootstrap_overall_values,
                    ddof=1
                ),

            "CI_Lower":
                overall_lower,

            "CI_Upper":
                overall_upper,

            "CI_Width":
                overall_upper - overall_lower
        },

        {
            "Metric":
                "Global CECI",

            "Observed_Value":
                reconstructed[
                    "Global_CECI_Component"
                ].mean(),

            "Bootstrap_Mean":
                np.mean(
                    bootstrap_global_values
                ),

            "Bootstrap_SD":
                np.std(
                    bootstrap_global_values,
                    ddof=1
                ),

            "CI_Lower":
                global_lower,

            "CI_Upper":
                global_upper,

            "CI_Width":
                global_upper - global_lower
        },

        {
            "Metric":
                "Local CECI",

            "Observed_Value":
                reconstructed[
                    "Local_CECI_Component"
                ].mean(),

            "Bootstrap_Mean":
                np.mean(
                    bootstrap_local_values
                ),

            "Bootstrap_SD":
                np.std(
                    bootstrap_local_values,
                    ddof=1
                ),

            "CI_Lower":
                local_lower,

            "CI_Upper":
                local_upper,

            "CI_Width":
                local_upper - local_lower
        }
    ]
)


OVERALL_ROBUSTNESS_FILE = os.path.join(
    OUTPUT_DIR,
    "ceci_bootstrap_overall_robustness.csv"
)

overall_robustness.to_csv(
    OVERALL_ROBUSTNESS_FILE,
    index=False
)


# =============================================================================
# COMPONENT ROBUSTNESS
# =============================================================================

print()
print("Calculating component robustness...")

# -------------------------------------------------------------------------
# IMPORTANT:
# Observed component values are calculated directly from the successfully
# reconstructed Step 17 CECI results rather than searching for text labels
# inside ceci_component_summary.csv.
#
# This avoids dependence on the exact naming convention used by Step 17.
# -------------------------------------------------------------------------

observed_component_values = {
    "Global_Spearman": (
        reconstructed["Global_Spearman_Normalized"].mean()
    ),
    "Global_Cosine": (
        reconstructed["Global_Cosine_Normalized"].mean()
    ),
    "Local_Spearman": (
        reconstructed["Local_Spearman_Normalized"].mean()
    ),
    "Local_Cosine": (
        reconstructed["Local_Cosine_Normalized"].mean()
    )
}

component_rows = []

for metric_name, values in bootstrap_component_values.items():

    lower, upper = percentile_ci(values)

    observed_value = observed_component_values[metric_name]

    component_rows.append(
        {
            "Component":
                metric_name,

            "Observed_Value":
                observed_value,

            "Bootstrap_Mean":
                np.mean(values),

            "Bootstrap_SD":
                np.std(
                    values,
                    ddof=1
                ),

            "CI_Lower":
                lower,

            "CI_Upper":
                upper,

            "CI_Width":
                upper - lower
        }
    )

component_robustness_df = pd.DataFrame(
    component_rows
)

COMPONENT_ROBUSTNESS_FILE = os.path.join(
    OUTPUT_DIR,
    "ceci_bootstrap_component_robustness.csv"
)

component_robustness_df.to_csv(
    COMPONENT_ROBUSTNESS_FILE,
    index=False
)

print()
print("Component robustness calculated successfully.")

print()
print(
    component_robustness_df.to_string(
        index=False
    )
)


# =============================================================================
# [8/9] PAIRWISE BOOTSTRAP ROBUSTNESS
# =============================================================================

print()
print("[8/9] Calculating pairwise robustness...")


pairwise_bootstrap_rows = []


for pair in EXPECTED_PAIRS:

    global_spearman_values = (
        global_values_by_pair[pair]["spearman"]
    )

    global_cosine_values = (
        global_values_by_pair[pair]["cosine"]
    )

    local_spearman_values = (
        local_values_by_pair[pair]["spearman"]
    )

    local_cosine_values = (
        local_values_by_pair[pair]["cosine"]
    )

    pair_bootstrap_ceci = np.empty(
        N_BOOTSTRAPS
    )

    for i in range(
        N_BOOTSTRAPS
    ):

        # Global bootstrap
        sampled_gs = rng.choice(
            global_spearman_values,
            size=len(global_spearman_values),
            replace=True
        )

        sampled_gc = rng.choice(
            global_cosine_values,
            size=len(global_cosine_values),
            replace=True
        )

        # Local bootstrap
        sampled_ls = rng.choice(
            local_spearman_values,
            size=len(local_spearman_values),
            replace=True
        )

        sampled_lc = rng.choice(
            local_cosine_values,
            size=len(local_cosine_values),
            replace=True
        )

        pair_bootstrap_ceci[i] = (
            calculate_pair_ceci(
                np.mean(sampled_gs),
                np.mean(sampled_gc),
                np.mean(sampled_ls),
                np.mean(sampled_lc)
            )["Pairwise_CECI"]
        )

    lower, upper = percentile_ci(
        pair_bootstrap_ceci
    )

    observed = reconstructed.loc[
        reconstructed["Model_Pair"] == pair,
        "Pairwise_CECI"
    ].iloc[0]

    pairwise_bootstrap_rows.append(
        {
            "Model_Pair":
                pair,

            "Observed_Pairwise_CECI":
                observed,

            "Bootstrap_Mean":
                np.mean(
                    pair_bootstrap_ceci
                ),

            "Bootstrap_SD":
                np.std(
                    pair_bootstrap_ceci,
                    ddof=1
                ),

            "CI_Lower":
                lower,

            "CI_Upper":
                upper,

            "CI_Width":
                upper - lower,

            "N_Global_Observations":
                len(
                    global_spearman_values
                ),

            "N_Local_Observations":
                len(
                    local_spearman_values
                )
        }
    )


pairwise_bootstrap_df = pd.DataFrame(
    pairwise_bootstrap_rows
)

pairwise_bootstrap_df = pairwise_bootstrap_df.sort_values(
    "Observed_Pairwise_CECI",
    ascending=False
).reset_index(
    drop=True
)

pairwise_bootstrap_df["CECI_Rank"] = (
    np.arange(
        1,
        len(pairwise_bootstrap_df) + 1
    )
)


PAIRWISE_ROBUSTNESS_FILE = os.path.join(
    OUTPUT_DIR,
    "ceci_pairwise_bootstrap_robustness.csv"
)

pairwise_bootstrap_df.to_csv(
    PAIRWISE_ROBUSTNESS_FILE,
    index=False
)


# =============================================================================
# LEAVE-ONE-PAIR-OUT SENSITIVITY
# =============================================================================

print()
print("Calculating leave-one-pair-out sensitivity...")


leave_one_out_rows = []


for removed_pair in EXPECTED_PAIRS:

    remaining = reconstructed[
        reconstructed["Model_Pair"]
        != removed_pair
    ]

    loo_global = remaining[
        "Global_CECI_Component"
    ].mean()

    loo_local = remaining[
        "Local_CECI_Component"
    ].mean()

    loo_overall = remaining[
        "Pairwise_CECI"
    ].mean()

    full_ceci = reconstructed[
        "Pairwise_CECI"
    ].mean()

    difference = (
        loo_overall -
        full_ceci
    )

    leave_one_out_rows.append(
        {
            "Removed_Model_Pair":
                removed_pair,

            "Remaining_Model_Pairs":
                len(remaining),

            "Leave_One_Out_Global_CECI":
                loo_global,

            "Leave_One_Out_Local_CECI":
                loo_local,

            "Leave_One_Out_CECI":
                loo_overall,

            "Difference_From_Full_CECI":
                difference,

            "Absolute_Difference":
                abs(difference)
        }
    )


leave_one_out_df = pd.DataFrame(
    leave_one_out_rows
)

LEAVE_ONE_OUT_FILE = os.path.join(
    OUTPUT_DIR,
    "ceci_leave_one_pair_out_sensitivity.csv"
)

leave_one_out_df.to_csv(
    LEAVE_ONE_OUT_FILE,
    index=False
)


# =============================================================================
# ROBUSTNESS INTERPRETATION
# =============================================================================

full_ceci = reconstructed[
    "Pairwise_CECI"
].mean()

ci_width = (
    overall_upper -
    overall_lower
)

relative_ci_width = (
    ci_width /
    abs(full_ceci)
    if abs(full_ceci) > 1e-12
    else np.nan
)


max_loo_difference = (
    leave_one_out_df[
        "Absolute_Difference"
    ].max()
)


if relative_ci_width < 0.05:

    bootstrap_interpretation = (
        "Very high bootstrap stability"
    )

elif relative_ci_width < 0.10:

    bootstrap_interpretation = (
        "High bootstrap stability"
    )

elif relative_ci_width < 0.20:

    bootstrap_interpretation = (
        "Moderate bootstrap stability"
    )

else:

    bootstrap_interpretation = (
        "Lower bootstrap stability"
    )


# =============================================================================
# FINAL ROBUSTNESS SUMMARY
# =============================================================================

robustness_summary = pd.DataFrame(
    [
        {
            "Observed_CECI":
                full_ceci,

            "Bootstrap_Mean":
                np.mean(
                    bootstrap_overall_values
                ),

            "Bootstrap_SD":
                np.std(
                    bootstrap_overall_values,
                    ddof=1
                ),

            "95_CI_Lower":
                overall_lower,

            "95_CI_Upper":
                overall_upper,

            "95_CI_Width":
                ci_width,

            "Relative_CI_Width":
                relative_ci_width,

            "Maximum_Leave_One_Out_Difference":
                max_loo_difference,

            "Bootstrap_Interpretation":
                bootstrap_interpretation,

            "Bootstrap_Repetitions":
                N_BOOTSTRAPS,

            "Random_State":
                RANDOM_STATE
        }
    ]
)


ROBUSTNESS_SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "final_ceci_robustness_summary.csv"
)

robustness_summary.to_csv(
    ROBUSTNESS_SUMMARY_FILE,
    index=False
)


# =============================================================================
# [9/9] SAVE FINAL CECI TABLE
# =============================================================================

print()
print("[9/9] Saving final CECI tables...")


FINAL_CECI_FILE = os.path.join(
    OUTPUT_DIR,
    "final_ceci_pairwise_results.csv"
)

reconstructed.to_csv(
    FINAL_CECI_FILE,
    index=False
)


# =============================================================================
# FINAL REPORT
# =============================================================================

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "STEP_18_FINAL_CECI_ROBUSTNESS_REPORT.txt"
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 18 — FINAL CECI + ROBUSTNESS ANALYSIS\n"
    )

    f.write(
        "=" * 80
        + "\n\n"
    )

    f.write(
        "Research Objective\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    f.write(
        "Evaluate the robustness of the Cross-Model Explanation "
        "Consistency Index (CECI) using bootstrap resampling "
        "of the unified permutation-SHAP consistency results.\n\n"
    )

    f.write(
        "Methodology\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    f.write(
        "Dataset: NASA JM1\n"
    )

    f.write(
        "Models: RF, XGB, LGBM, MLP\n"
    )

    f.write(
        "Model pairs: 6\n"
    )

    f.write(
        "SHAP method: Unified permutation SHAP\n"
    )

    f.write(
        "Consistency dimensions: "
        "Global Spearman, Global Cosine, "
        "Local Spearman, Local Cosine\n"
    )

    f.write(
        "Normalization: (score + 1) / 2\n"
    )

    f.write(
        "Weighting: Equal weighting\n"
    )

    f.write(
        f"Bootstrap repetitions: {N_BOOTSTRAPS}\n"
    )

    f.write(
        f"Confidence level: "
        f"{CONFIDENCE_LEVEL * 100:.0f}%\n\n"
    )

    # -------------------------------------------------------------------------
    # Overall
    # -------------------------------------------------------------------------

    f.write(
        "FINAL CECI\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    f.write(
        f"Observed Overall CECI: "
        f"{full_ceci:.6f}\n"
    )

    f.write(
        f"Observed Overall CECI (%): "
        f"{full_ceci * 100:.2f}%\n"
    )

    f.write(
        f"Bootstrap Mean: "
        f"{np.mean(bootstrap_overall_values):.6f}\n"
    )

    f.write(
        f"Bootstrap SD: "
        f"{np.std(bootstrap_overall_values, ddof=1):.6f}\n"
    )

    f.write(
        f"95% Bootstrap CI: "
        f"[{overall_lower:.6f}, "
        f"{overall_upper:.6f}]\n"
    )

    f.write(
        f"CI Width: "
        f"{ci_width:.6f}\n"
    )

    f.write(
        f"Relative CI Width: "
        f"{relative_ci_width:.6f}\n"
    )

    f.write(
        f"Bootstrap Interpretation: "
        f"{bootstrap_interpretation}\n\n"
    )

    # -------------------------------------------------------------------------
    # Pairwise
    # -------------------------------------------------------------------------

    f.write(
        "PAIRWISE CECI RESULTS\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    for _, row in reconstructed.iterrows():

        f.write(
            f"{int(row['CECI_Rank'])}. "
            f"{row['Model_Pair']}\n"
        )

        f.write(
            f"   Pairwise CECI: "
            f"{row['Pairwise_CECI']:.6f}\n"
        )

        f.write(
            f"   Global CECI Component: "
            f"{row['Global_CECI_Component']:.6f}\n"
        )

        f.write(
            f"   Local CECI Component: "
            f"{row['Local_CECI_Component']:.6f}\n"
        )

        pair_row = pairwise_bootstrap_df[
            pairwise_bootstrap_df[
                "Model_Pair"
            ] == row["Model_Pair"]
        ].iloc[0]

        f.write(
            f"   Bootstrap Mean: "
            f"{pair_row['Bootstrap_Mean']:.6f}\n"
        )

        f.write(
            f"   95% CI: "
            f"[{pair_row['CI_Lower']:.6f}, "
            f"{pair_row['CI_Upper']:.6f}]\n\n"
        )

    # -------------------------------------------------------------------------
    # Leave-one-out
    # -------------------------------------------------------------------------

    f.write(
        "LEAVE-ONE-PAIR-OUT SENSITIVITY\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    for _, row in leave_one_out_df.iterrows():

        f.write(
            f"Removed: "
            f"{row['Removed_Model_Pair']}\n"
        )

        f.write(
            f"   Leave-one-out CECI: "
            f"{row['Leave_One_Out_CECI']:.6f}\n"
        )

        f.write(
            f"   Difference from full CECI: "
            f"{row['Difference_From_Full_CECI']:.6f}\n\n"
        )

    # -------------------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------------------

    f.write(
        "ROBUSTNESS INTERPRETATION\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    f.write(
        f"Maximum leave-one-pair-out change: "
        f"{max_loo_difference:.6f}\n"
    )

    f.write(
        f"Bootstrap stability: "
        f"{bootstrap_interpretation}\n\n"
    )

    f.write(
        "CECI remains a cross-model explanation consistency "
        "measure and is not combined with IMSI or RF-RID.\n"
    )


# =============================================================================
# CONSOLE OUTPUT
# =============================================================================

print()
print("=" * 80)
print("FINAL CECI")
print("=" * 80)

print()

print(
    f"Observed Overall CECI = "
    f"{full_ceci:.6f}"
)

print(
    f"Observed Overall CECI = "
    f"{full_ceci * 100:.2f}%"
)

print()

print(
    f"Bootstrap Mean = "
    f"{np.mean(bootstrap_overall_values):.6f}"
)

print(
    f"Bootstrap SD = "
    f"{np.std(bootstrap_overall_values, ddof=1):.6f}"
)

print()

print(
    f"95% Bootstrap CI = "
    f"[{overall_lower:.6f}, "
    f"{overall_upper:.6f}]"
)

print()

print(
    f"Bootstrap Interpretation = "
    f"{bootstrap_interpretation}"
)

print()

print("=" * 80)
print("PAIRWISE CECI")
print("=" * 80)

print()

for _, row in reconstructed.iterrows():

    print(
        f"{int(row['CECI_Rank'])}. "
        f"{row['Model_Pair']}: "
        f"{row['Pairwise_CECI']:.6f}"
    )

print()

print("=" * 80)
print("ROBUSTNESS ANALYSIS COMPLETE")
print("=" * 80)

print()

print("Output directory:")
print(
    OUTPUT_DIR
)

print()

print("Files created:")

print(
    f"  {FINAL_CECI_FILE}"
)

print(
    f"  {OVERALL_ROBUSTNESS_FILE}"
)

print(
    f"  {COMPONENT_ROBUSTNESS_FILE}"
)

print(
    f"  {PAIRWISE_ROBUSTNESS_FILE}"
)

print(
    f"  {LEAVE_ONE_OUT_FILE}"
)

print(
    f"  {ROBUSTNESS_SUMMARY_FILE}"
)

print(
    f"  {REPORT_FILE}"
)

print()

print("NEXT STEP:")
print(
    "  STEP 19 — FINAL EXPERIMENTAL RESULTS & RESEARCH FINDINGS"
)

print("=" * 80)
