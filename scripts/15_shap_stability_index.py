"""
STEP 13 — SHAP STABILITY INDEX (SSI)

Research project:
Evaluating SHAP Explanation Consistency in Ensemble Models
for Software Defect Prediction

Purpose
-------
Convert the intra-model SHAP stability measurements into a
transparent normalized stability index.

This step uses the results from:

    Step 12 — Intra-Model SHAP Stability

Metrics:
    1. Global Spearman correlation
    2. Global cosine similarity
    3. Local Spearman correlation
    4. Local cosine similarity

All metrics are normalized to [0, 1].

For Spearman:
    normalized = (rho + 1) / 2

For cosine:
    normalized = (cosine + 1) / 2

Intra-Model Stability Index (IMSI):

    IMSI =
        mean(
            normalized global Spearman,
            normalized global cosine,
            normalized local Spearman,
            normalized local cosine
        )

Higher IMSI = greater explanation stability.

IMPORTANT
---------
This script does NOT yet calculate the final cross-model
consistency component.

The final overall SSI will be calculated only after the
RF vs RF-RID explanation comparison is corrected and the
cross-model explanation results are finalized.
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = (
    r"E:\Programming\Evaluating SHAP Explanation Consistency "
    r"in Ensemble Models for Software Defect Prediction"
)

INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "results",
    "intra_model_shap_stability",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "results",
    "shap_stability_index",
)

FIGURE_DIR = os.path.join(
    PROJECT_ROOT,
    "figures",
    "shap_stability_index",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True,
)


# =============================================================================
# INPUT FILES
# =============================================================================

GLOBAL_FILE = os.path.join(
    INPUT_DIR,
    "global_intra_model_stability.csv",
)

LOCAL_FILE = os.path.join(
    INPUT_DIR,
    "local_intra_model_stability.csv",
)

SUMMARY_FILE = os.path.join(
    INPUT_DIR,
    "intra_model_shap_stability_summary.csv",
)


# =============================================================================
# HELPERS
# =============================================================================

def normalize_correlation(values):
    """
    Convert a correlation-like measure from [-1, 1] to [0, 1].
    """

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    return (
        values + 1.0
    ) / 2.0


def safe_mean(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    return float(
        values.mean()
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("STEP 13 — SHAP STABILITY INDEX (SSI)")
    print("=" * 80)

    # =========================================================================
    # LOAD RESULTS
    # =========================================================================

    print(
        "\n[1/7] Loading Step 12 results..."
    )

    if not os.path.exists(
        GLOBAL_FILE
    ):

        raise FileNotFoundError(
            "Global stability file not found:\n"
            + GLOBAL_FILE
        )

    if not os.path.exists(
        LOCAL_FILE
    ):

        raise FileNotFoundError(
            "Local stability file not found:\n"
            + LOCAL_FILE
        )

    global_df = pd.read_csv(
        GLOBAL_FILE
    )

    local_df = pd.read_csv(
        LOCAL_FILE
    )

    print(
        f"Global stability rows: "
        f"{len(global_df)}"
    )

    print(
        f"Local stability rows: "
        f"{len(local_df)}"
    )

    print(
        "\nModels:"
    )

    models = sorted(
        set(
            global_df[
                "Model"
            ].unique()
        )
    )

    for model in models:

        print(
            f"  - {model}"
        )

    # =========================================================================
    # GLOBAL METRICS
    # =========================================================================

    print(
        "\n[2/7] Calculating normalized global stability..."
    )

    global_summary_rows = []

    for model in models:

        model_global = global_df[
            global_df[
                "Model"
            ] == model
        ].copy()

        global_spearman = safe_mean(
            model_global[
                "Global_Spearman"
            ]
        )

        global_cosine = safe_mean(
            model_global[
                "Global_Cosine"
            ]
        )

        normalized_global_spearman = (
            global_spearman + 1.0
        ) / 2.0

        normalized_global_cosine = (
            global_cosine + 1.0
        ) / 2.0

        global_component = np.mean([
            normalized_global_spearman,
            normalized_global_cosine,
        ])

        global_summary_rows.append({
            "Model":
                model,

            "Global_Spearman":
                global_spearman,

            "Global_Cosine":
                global_cosine,

            "Normalized_Global_Spearman":
                normalized_global_spearman,

            "Normalized_Global_Cosine":
                normalized_global_cosine,

            "Global_Stability_Component":
                global_component,
        })

    global_summary = pd.DataFrame(
        global_summary_rows
    )

    # =========================================================================
    # LOCAL METRICS
    # =========================================================================

    print(
        "\n[3/7] Calculating normalized local stability..."
    )

    local_summary_rows = []

    for model in models:

        model_local = local_df[
            local_df[
                "Model"
            ] == model
        ].copy()

        local_spearman = safe_mean(
            model_local[
                "Local_Spearman"
            ]
        )

        local_cosine = safe_mean(
            model_local[
                "Local_Cosine"
            ]
        )

        normalized_local_spearman = (
            local_spearman + 1.0
        ) / 2.0

        normalized_local_cosine = (
            local_cosine + 1.0
        ) / 2.0

        local_component = np.mean([
            normalized_local_spearman,
            normalized_local_cosine,
        ])

        local_summary_rows.append({
            "Model":
                model,

            "Local_Spearman":
                local_spearman,

            "Local_Cosine":
                local_cosine,

            "Normalized_Local_Spearman":
                normalized_local_spearman,

            "Normalized_Local_Cosine":
                normalized_local_cosine,

            "Local_Stability_Component":
                local_component,
        })

    local_summary = pd.DataFrame(
        local_summary_rows
    )

    # =========================================================================
    # COMBINE
    # =========================================================================

    print(
        "\n[4/7] Calculating Intra-Model Stability Index..."
    )

    summary = global_summary.merge(
        local_summary,
        on="Model",
        how="inner",
    )

    summary[
        "IMSI"
    ] = summary[
        [
            "Normalized_Global_Spearman",
            "Normalized_Global_Cosine",
            "Normalized_Local_Spearman",
            "Normalized_Local_Cosine",
        ]
    ].mean(
        axis=1
    )

    # =========================================================================
    # RANKING
    # =========================================================================

    summary[
        "IMSI_Rank"
    ] = summary[
        "IMSI"
    ].rank(
        ascending=False,
        method="min",
    ).astype(int)

    summary = summary.sort_values(
        "IMSI",
        ascending=False,
    ).reset_index(
        drop=True
    )

    # =========================================================================
    # SAVE SUMMARY
    # =========================================================================

    print(
        "\n[5/7] Saving SSI results..."
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "intra_model_stability_index.csv",
    )

    summary.to_csv(
        output_file,
        index=False,
    )

    # =========================================================================
    # SAVE COMPONENTS
    # =========================================================================

    components = summary[
        [
            "Model",
            "Global_Stability_Component",
            "Local_Stability_Component",
            "IMSI",
            "IMSI_Rank",
        ]
    ].copy()

    components.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "shap_stability_index_components.csv",
        ),
        index=False,
    )

    # =========================================================================
    # FIGURE
    # =========================================================================

    print(
        "\n[6/7] Creating SSI figure..."
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        summary["Model"],
        summary["IMSI"],
    )

    plt.ylim(
        0,
        1,
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Intra-Model Stability Index"
    )

    plt.title(
        "Intra-Model SHAP Stability Index"
    )

    for index, value in enumerate(
        summary["IMSI"]
    ):

        plt.text(
            index,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "intra_model_stability_index.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # =========================================================================
    # REPORT
    # =========================================================================

    report_file = os.path.join(
        OUTPUT_DIR,
        "shap_stability_index_report.txt",
    )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "STEP 13 — SHAP STABILITY INDEX\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            "Definition\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            "Correlation normalization:\n"
        )

        f.write(
            "    normalized = (metric + 1) / 2\n\n"
        )

        f.write(
            "Intra-Model Stability Index:\n"
        )

        f.write(
            "    IMSI = mean(\n"
            "        normalized global Spearman,\n"
            "        normalized global cosine,\n"
            "        normalized local Spearman,\n"
            "        normalized local cosine\n"
            "    )\n\n"
        )

        f.write(
            "Higher IMSI indicates greater SHAP explanation stability.\n\n"
        )

        f.write(
            "RESULTS\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            summary.to_string(
                index=False
            )
        )

        f.write(
            "\n\nIMPORTANT NOTE\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            "This is the intra-model stability component only.\n"
        )

        f.write(
            "The final overall SSI should be calculated only after "
            "the cross-model SHAP consistency results and RF vs RF-RID "
            "explanation comparison have been finalized.\n"
        )

    # =========================================================================
    # METADATA
    # =========================================================================

    metadata = {
        "step":
            "Step 13 — SHAP Stability Index",

        "metric_normalization":
            "(metric + 1) / 2",

        "metrics":
            [
                "Global_Spearman",
                "Global_Cosine",
                "Local_Spearman",
                "Local_Cosine",
            ],

        "index":
            "IMSI = mean of four normalized stability metrics",

        "models":
            models,

        "note":
            "Intra-model component only; final overall SSI "
            "requires finalized cross-model consistency.",
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "shap_stability_index_metadata.json",
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
    # TERMINAL OUTPUT
    # =========================================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 13 COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        "\nINTRA-MODEL STABILITY INDEX:"
    )

    print(
        summary[
            [
                "Model",
                "Global_Stability_Component",
                "Local_Stability_Component",
                "IMSI",
                "IMSI_Rank",
            ]
        ].to_string(
            index=False
        )
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
        "\nNEXT STEP:"
    )

    print(
        "Correct and rerun RF vs RF-RID SHAP explanation comparison."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()
