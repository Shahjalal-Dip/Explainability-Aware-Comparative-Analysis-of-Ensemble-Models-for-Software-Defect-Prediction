"""
STEP 14
GLOBAL + LOCAL SHAP EXPLANATION ANALYSIS

Research project:
Evaluating SHAP Explanation Consistency in Ensemble Models
for Software Defect Prediction

Models:
RF, XGB, LGBM, MLP

Important:
- Independent test set is used only for final explanation evaluation.
- No feature selection or model tuning is performed here.
- Same test instances are used across models.
- Global SHAP importance is compared using Spearman rho
  and cosine similarity.
- Local SHAP consistency is evaluated on fixed test instances.
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

PROJECT_ROOT = (
    r"E:\Programming\Evaluating SHAP Explanation Consistency "
    r"in Ensemble Models for Software Defect Prediction"
)

TEST_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "JM1",
    "JM1_test.csv",
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "final",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "results",
    "shap_analysis",
)

FIGURE_DIR = os.path.join(
    PROJECT_ROOT,
    "figures",
    "shap_analysis",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# =============================================================================
# FEATURES
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


MODELS = {
    "RF": "rf_final.pkl",
    "XGB": "xgb_final.pkl",
    "LGBM": "lgbm_final.pkl",
    "MLP": "mlp_final.pkl",
}


# Number of test observations used for SHAP.
# Using a fixed subset makes the analysis reproducible
# and avoids unnecessarily expensive computation.
SHAP_TEST_SIZE = 300

# Background size for probability-space SHAP.
BACKGROUND_SIZE = 100


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_shap_importance(values):
    """
    Convert mean absolute SHAP values into relative contributions.
    """

    values = np.asarray(values, dtype=float)

    total = values.sum()

    if total == 0:
        return np.zeros_like(values)

    return values / total


def extract_positive_class_shap(shap_values):
    """
    Standardize SHAP output shape.

    Handles:
    - Explanation objects
    - arrays
    - binary-class outputs
    """

    if isinstance(shap_values, shap.Explanation):
        values = shap_values.values
    else:
        values = np.asarray(shap_values)

    # Some SHAP versions return:
    # samples x features x classes
    if values.ndim == 3:
        values = values[:, :, 1]

    return values


def rank_vector(values):
    """
    Convert values to ranks.
    Higher SHAP importance receives higher rank.
    """

    return pd.Series(values).rank(
        method="average",
        ascending=False,
    ).values


def spearman_similarity(a, b):
    """
    Spearman rank correlation.
    """

    rho, p_value = spearmanr(a, b)

    return float(rho), float(p_value)


def cosine_similarity_value(a, b):
    """
    Cosine similarity.
    """

    a = np.asarray(a).reshape(1, -1)
    b = np.asarray(b).reshape(1, -1)

    return float(
        cosine_similarity(a, b)[0, 0]
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("STEP 14 — GLOBAL + LOCAL SHAP ANALYSIS")
    print("=" * 80)

    # =========================================================================
    # LOAD TEST DATA
    # =========================================================================

    print("\n[1/8] Loading independent test data...")

    test = pd.read_csv(TEST_FILE)

    X_test = test[FEATURES].copy()
    y_test = test[TARGET].copy()

    print(
        f"Test shape: {test.shape}"
    )

    print(
        f"Feature matrix: {X_test.shape}"
    )

    # =========================================================================
    # FIXED TEST SUBSET
    # =========================================================================

    print(
        "\n[2/8] Creating fixed SHAP test subset..."
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    if len(X_test) > SHAP_TEST_SIZE:

        selected_indices = np.sort(
            rng.choice(
                len(X_test),
                size=SHAP_TEST_SIZE,
                replace=False,
            )
        )

    else:

        selected_indices = np.arange(
            len(X_test)
        )

    X_shap = X_test.iloc[
        selected_indices
    ].copy()

    y_shap = y_test.iloc[
        selected_indices
    ].copy()

    print(
        f"SHAP observations: {len(X_shap)}"
    )

    print(
        "The same observations will be used for every model."
    )

    # =========================================================================
    # BACKGROUND DATA
    # =========================================================================

    background_indices = np.sort(
        rng.choice(
            len(X_test),
            size=min(
                BACKGROUND_SIZE,
                len(X_test),
            ),
            replace=False,
        )
    )

    X_background = X_test.iloc[
        background_indices
    ].copy()

    # =========================================================================
    # LOAD MODELS
    # =========================================================================

    print(
        "\n[3/8] Loading final models..."
    )

    models = {}

    for model_name, filename in MODELS.items():

        path = os.path.join(
            MODEL_DIR,
            filename,
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Model not found:\n{path}"
            )

        models[model_name] = joblib.load(
            path
        )

        print(
            f"  {model_name}: loaded"
        )

    # =========================================================================
    # SHAP CALCULATION
    # =========================================================================

    print(
        "\n[4/8] Calculating SHAP values..."
    )

    shap_values_by_model = {}

    for model_name, model in models.items():

        print(
            f"\n  Calculating SHAP for {model_name}..."
        )

        # ---------------------------------------------------------------------
        # Tree models
        # ---------------------------------------------------------------------

        if model_name in [
            "RF",
            "XGB",
            "LGBM",
        ]:

            try:

                explainer = shap.TreeExplainer(
                    model,
                    data=X_background,
                    feature_perturbation="interventional",
                    model_output="probability",
                )

                values = explainer(
                    X_shap
                )

            except Exception as error:

                print(
                    "  Probability-space TreeExplainer "
                    "failed."
                )

                print(
                    f"  Reason: {error}"
                )

                print(
                    "  Falling back to default TreeExplainer."
                )

                explainer = shap.TreeExplainer(
                    model
                )

                values = explainer(
                    X_shap
                )

        # ---------------------------------------------------------------------
        # MLP
        # ---------------------------------------------------------------------

        else:

            # MLP is explained using a probability prediction function.
            def mlp_predict(data):

                data = np.asarray(data)

                return model.predict_proba(
                    data
                )[:, 1]

            explainer = shap.Explainer(
                mlp_predict,
                X_background,
                algorithm="permutation",
            )

            values = explainer(
                X_shap
            )

        values = extract_positive_class_shap(
            values
        )

        # Ensure correct dimensions.
        if values.shape != (
            len(X_shap),
            len(FEATURES),
        ):

            raise ValueError(
                f"Unexpected SHAP shape for {model_name}: "
                f"{values.shape}"
            )

        shap_values_by_model[
            model_name
        ] = values

        print(
            f"  SHAP shape: {values.shape}"
        )

    # =========================================================================
    # SAVE RAW SHAP VALUES
    # =========================================================================

    print(
        "\n[5/8] Saving SHAP values..."
    )

    for model_name, values in shap_values_by_model.items():

        shap_df = pd.DataFrame(
            values,
            columns=FEATURES,
        )

        shap_df.insert(
            0,
            "test_index",
            selected_indices,
        )

        shap_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                f"{model_name.lower()}_shap_values.csv",
            ),
            index=False,
        )

    # =========================================================================
    # GLOBAL SHAP IMPORTANCE
    # =========================================================================

    print(
        "\n[6/8] Calculating global SHAP importance..."
    )

    global_rows = []

    global_vectors = {}

    for model_name, values in shap_values_by_model.items():

        mean_abs = np.mean(
            np.abs(values),
            axis=0,
        )

        normalized = normalize_shap_importance(
            mean_abs
        )

        global_vectors[
            model_name
        ] = normalized

        for feature, raw, norm in zip(
            FEATURES,
            mean_abs,
            normalized,
        ):

            global_rows.append({
                "Model": model_name,
                "Feature": feature,
                "Mean_Absolute_SHAP": raw,
                "Normalized_SHAP": norm,
            })

    global_importance = pd.DataFrame(
        global_rows
    )

    global_importance.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "global_shap_feature_importance.csv",
        ),
        index=False,
    )

    # =========================================================================
    # GLOBAL RANKING
    # =========================================================================

    ranking_rows = []

    for model_name in MODELS:

        temp = global_importance[
            global_importance["Model"]
            == model_name
        ].copy()

        temp = temp.sort_values(
            "Mean_Absolute_SHAP",
            ascending=False,
        )

        temp["Rank"] = np.arange(
            1,
            len(temp) + 1,
        )

        for _, row in temp.iterrows():

            ranking_rows.append({
                "Model": model_name,
                "Feature": row["Feature"],
                "Rank": int(row["Rank"]),
                "Mean_Absolute_SHAP":
                    row["Mean_Absolute_SHAP"],
                "Normalized_SHAP":
                    row["Normalized_SHAP"],
            })

    ranking_df = pd.DataFrame(
        ranking_rows
    )

    ranking_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "global_shap_feature_ranking.csv",
        ),
        index=False,
    )

    # =========================================================================
    # GLOBAL SHAP CONSISTENCY
    # =========================================================================

    print(
        "\n[7/8] Calculating global SHAP consistency..."
    )

    model_names = list(
        MODELS.keys()
    )

    spearman_rows = []
    cosine_rows = []

    for i in range(
        len(model_names)
    ):

        for j in range(
            i + 1,
            len(model_names),
        ):

            model_a = model_names[i]
            model_b = model_names[j]

            vector_a = global_vectors[
                model_a
            ]

            vector_b = global_vectors[
                model_b
            ]

            rho, p_value = spearman_similarity(
                vector_a,
                vector_b,
            )

            cosine = cosine_similarity_value(
                vector_a,
                vector_b,
            )

            spearman_rows.append({
                "Model_A": model_a,
                "Model_B": model_b,
                "Spearman_rho": rho,
                "p_value": p_value,
            })

            cosine_rows.append({
                "Model_A": model_a,
                "Model_B": model_b,
                "Cosine_similarity": cosine,
            })

    global_spearman = pd.DataFrame(
        spearman_rows
    )

    global_cosine = pd.DataFrame(
        cosine_rows
    )

    global_spearman.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "global_shap_spearman.csv",
        ),
        index=False,
    )

    global_cosine.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "global_shap_cosine_similarity.csv",
        ),
        index=False,
    )

    # =========================================================================
    # LOCAL SHAP SPEARMAN
    # =========================================================================

    print(
        "\n[8/8] Calculating local SHAP consistency..."
    )

    local_rows = []

    # Pairwise local Spearman for every observation.
    for sample_position in range(
        len(X_shap)
    ):

        test_index = int(
            selected_indices[
                sample_position
            ]
        )

        for i in range(
            len(model_names)
        ):

            for j in range(
                i + 1,
                len(model_names),
            ):

                model_a = model_names[i]
                model_b = model_names[j]

                vector_a = shap_values_by_model[
                    model_a
                ][sample_position]

                vector_b = shap_values_by_model[
                    model_b
                ][sample_position]

                rho, p_value = spearman_similarity(
                    vector_a,
                    vector_b,
                )

                cosine = cosine_similarity_value(
                    vector_a,
                    vector_b,
                )

                local_rows.append({
                    "test_index": test_index,
                    "Model_A": model_a,
                    "Model_B": model_b,
                    "Local_Spearman_rho": rho,
                    "Local_Spearman_p_value": p_value,
                    "Local_Cosine_similarity": cosine,
                })

    local_consistency = pd.DataFrame(
        local_rows
    )

    local_consistency.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "local_shap_consistency.csv",
        ),
        index=False,
    )

    # =========================================================================
    # LOCAL SUMMARY
    # =========================================================================

    local_summary = (
        local_consistency
        .groupby(
            [
                "Model_A",
                "Model_B",
            ]
        )
        .agg(
            Local_Spearman_mean=(
                "Local_Spearman_rho",
                "mean",
            ),
            Local_Spearman_std=(
                "Local_Spearman_rho",
                "std",
            ),
            Local_Spearman_median=(
                "Local_Spearman_rho",
                "median",
            ),
            Local_Cosine_mean=(
                "Local_Cosine_similarity",
                "mean",
            ),
            Local_Cosine_std=(
                "Local_Cosine_similarity",
                "std",
            ),
        )
        .reset_index()
    )

    local_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "local_shap_consistency_summary.csv",
        ),
        index=False,
    )

    # =========================================================================
    # GLOBAL IMPORTANCE PLOT
    # =========================================================================

    print(
        "\nCreating global SHAP figure..."
    )

    global_plot_data = (
        global_importance
        .pivot(
            index="Feature",
            columns="Model",
            values="Normalized_SHAP",
        )
    )

    global_plot_data = global_plot_data.loc[
        global_plot_data.mean(axis=1)
        .sort_values(
            ascending=False
        ).index
    ]

    global_plot_data.plot(
        kind="bar",
        figsize=(14, 7),
    )

    plt.ylabel(
        "Normalized Mean |SHAP|"
    )

    plt.xlabel(
        "Feature"
    )

    plt.title(
        "Global SHAP Feature Importance Across Final Models"
    )

    plt.xticks(
        rotation=70,
        ha="right",
    )

    plt.legend(
        title="Model"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "global_shap_feature_importance.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # =========================================================================
    # TOP FEATURES TABLE
    # =========================================================================

    top_features = {}

    for model_name in model_names:

        temp = ranking_df[
            ranking_df["Model"]
            == model_name
        ].sort_values(
            "Rank"
        )

        top_features[
            model_name
        ] = temp.head(10)[
            [
                "Feature",
                "Rank",
                "Normalized_SHAP",
            ]
        ].to_dict(
            orient="records"
        )

    # =========================================================================
    # REPORT
    # =========================================================================

    report_file = os.path.join(
        OUTPUT_DIR,
        "shap_analysis_report.txt",
    )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "STEP 14 — GLOBAL + LOCAL SHAP ANALYSIS\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            f"Test observations: {len(X_test)}\n"
        )

        f.write(
            f"SHAP observations: {len(X_shap)}\n"
        )

        f.write(
            f"Background observations: {len(X_background)}\n\n"
        )

        f.write(
            "GLOBAL SHAP SPEARMAN\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            global_spearman.to_string(
                index=False
            )
        )

        f.write(
            "\n\nGLOBAL SHAP COSINE SIMILARITY\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            global_cosine.to_string(
                index=False
            )
        )

        f.write(
            "\n\nLOCAL SHAP CONSISTENCY SUMMARY\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            local_summary.to_string(
                index=False
            )
        )

        f.write(
            "\n\nTOP 10 GLOBAL FEATURES\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        for model_name in model_names:

            f.write(
                f"\n{model_name}\n"
            )

            temp = ranking_df[
                ranking_df["Model"]
                == model_name
            ].sort_values(
                "Rank"
            ).head(10)

            f.write(
                temp[
                    [
                        "Feature",
                        "Rank",
                        "Mean_Absolute_SHAP",
                        "Normalized_SHAP",
                    ]
                ].to_string(
                    index=False
                )
            )

            f.write("\n")

    # =========================================================================
    # METADATA
    # =========================================================================

    metadata = {
        "step": "Step 14 — Global + Local SHAP Analysis",
        "random_state": RANDOM_STATE,
        "test_file": TEST_FILE,
        "test_size": len(X_test),
        "shap_test_size": len(X_shap),
        "background_size": len(X_background),
        "features": FEATURES,
        "models": list(MODELS.keys()),
        "shap_test_indices": [
            int(x)
            for x in selected_indices
        ],
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "shap_analysis_metadata.json",
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
    # FINISH
    # =========================================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 14 COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        "\nGlobal SHAP Spearman:"
    )

    print(
        global_spearman.to_string(
            index=False
        )
    )

    print(
        "\nGlobal SHAP Cosine:"
    )

    print(
        global_cosine.to_string(
            index=False
        )
    )

    print(
        "\nLocal SHAP summary:"
    )

    print(
        local_summary.to_string(
            index=False
        )
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nFigures:"
    )

    print(
        FIGURE_DIR
    )

    print(
        "\nNEXT:"
    )

    print(
        "STEP 15 — Intra-model SHAP stability"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore"
    )

    main()
