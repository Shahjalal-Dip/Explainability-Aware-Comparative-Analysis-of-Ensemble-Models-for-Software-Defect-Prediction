"""
STEP 12 — INTRA-MODEL SHAP STABILITY

Research project:
Evaluating SHAP Explanation Consistency in Ensemble Models
for Software Defect Prediction

Purpose
-------
Evaluate whether SHAP explanations remain stable when the same
model is retrained under repeated perturbations of the
development data.

Experimental design
-------------------
- Development data only for model fitting.
- Independent test set remains untouched for training.
- 5 bootstrap repetitions.
- SMOTE is applied independently within each repetition.
- Same fixed 300 independent test observations are explained
  in every repetition.
- Same fixed background observations are used for every
  repetition.
- Four final model types:
      RF
      XGB
      LGBM
      MLP

Stability measures
------------------
Global:
    - Spearman rank correlation
    - Cosine similarity

Local:
    - Local Spearman correlation
    - Local cosine similarity

Important
---------
This measures INTRA-MODEL stability.

It is different from:
    - cross-model SHAP consistency
    - RF vs RF-RID explanation comparison
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
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

N_REPETITIONS = 5

SHAP_TEST_SIZE = 300

BACKGROUND_SIZE = 100

PROJECT_ROOT = (
    r"E:\Programming\Evaluating SHAP Explanation Consistency "
    r"in Ensemble Models for Software Defect Prediction"
)

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

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "results",
    "intra_model_shap_stability",
)

FIGURE_DIR = os.path.join(
    PROJECT_ROOT,
    "figures",
    "intra_model_shap_stability",
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "intra_model_shap_stability",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True,
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)


# =============================================================================
# DATA
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
# MODEL CONFIGURATIONS
# =============================================================================

MODEL_CONFIGS = {

    "RF": {
        "model": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            max_features="sqrt",
            min_samples_leaf=1,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    },

    "XGB": {
        "model": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
        )
    },

    "LGBM": {
        "model": LGBMClassifier(
            n_estimators=200,
            max_depth=20,
            learning_rate=0.01,
            num_leaves=63,
            min_child_samples=30,
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1,
        )
    },

    "MLP": {
        "model": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(50,),
                    activation="tanh",
                    alpha=0.01,
                    learning_rate_init=0.001,
                    max_iter=500,
                    random_state=RANDOM_STATE,
                )
            )
        ])
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def extract_shap_values(output):
    """
    Convert SHAP output into a 2-D matrix:
        observations x features
    """

    if isinstance(
        output,
        shap.Explanation,
    ):
        values = output.values
    else:
        values = np.asarray(output)

    if values.ndim == 3:

        values = values[:, :, 1]

    return np.asarray(
        values,
        dtype=float,
    )


def normalize(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    total = np.sum(values)

    if total == 0:

        return np.zeros_like(
            values
        )

    return values / total


def safe_spearman(a, b):

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    if (
        np.all(a == a[0])
        or np.all(b == b[0])
    ):

        return np.nan

    rho, _ = spearmanr(
        a,
        b,
    )

    return float(rho)


def safe_cosine(a, b):

    a = np.asarray(
        a,
        dtype=float,
    ).reshape(
        1,
        -1,
    )

    b = np.asarray(
        b,
        dtype=float,
    ).reshape(
        1,
        -1,
    )

    norm_a = np.linalg.norm(a)

    norm_b = np.linalg.norm(b)

    if (
        norm_a == 0
        or norm_b == 0
    ):

        return np.nan

    return float(
        cosine_similarity(
            a,
            b,
        )[0, 0]
    )


def calculate_tree_shap(
    model,
    X_background,
    X_explain,
):
    """
    Calculate Tree SHAP.

    First attempts probability-space interventional SHAP.
    If unsupported by the installed SHAP/model combination,
    falls back to the model's default TreeExplainer output.
    """

    try:

        explainer = shap.TreeExplainer(
            model,
            data=X_background,
            feature_perturbation="interventional",
            model_output="probability",
        )

        output = explainer(
            X_explain
        )

        return extract_shap_values(
            output
        )

    except Exception as error:

        print(
            "    Probability-space Tree SHAP failed."
        )

        print(
            f"    Reason: {error}"
        )

        print(
            "    Falling back to default TreeExplainer."
        )

        explainer = shap.TreeExplainer(
            model
        )

        output = explainer(
            X_explain
        )

        return extract_shap_values(
            output
        )


def calculate_mlp_shap(
    model,
    X_background,
    X_explain,
):
    """
    Permutation SHAP for MLP.

    The prediction function returns positive-class probability.
    """

    def predict_fn(X):

        return model.predict_proba(
            X
        )[:, 1]

    explainer = shap.Explainer(
        predict_fn,
        X_background,
        algorithm="permutation",
    )

    output = explainer(
        X_explain
    )

    return extract_shap_values(
        output
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("STEP 12 — INTRA-MODEL SHAP STABILITY")
    print("=" * 80)

    # =========================================================================
    # LOAD DEVELOPMENT DATA
    # =========================================================================

    print(
        "\n[1/8] Loading development data..."
    )

    development = pd.read_csv(
        DEVELOPMENT_FILE
    )

    X_dev = development[
        FEATURES
    ].copy()

    y_dev = development[
        TARGET
    ].astype(int)

    print(
        f"Development shape: {development.shape}"
    )

    print(
        f"Feature matrix: {X_dev.shape}"
    )

    print(
        "Class distribution:"
    )

    print(
        y_dev.value_counts().sort_index()
    )

    # =========================================================================
    # LOAD TEST DATA
    # =========================================================================

    print(
        "\n[2/8] Loading independent test data..."
    )

    test = pd.read_csv(
        TEST_FILE
    )

    X_test = test[
        FEATURES
    ].copy()

    print(
        f"Test shape: {test.shape}"
    )

    # =========================================================================
    # FIXED SHAP TEST SUBSET
    # =========================================================================

    print(
        "\n[3/8] Creating fixed SHAP evaluation subset..."
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    shap_size = min(
        SHAP_TEST_SIZE,
        len(X_test),
    )

    shap_indices = np.sort(
        rng.choice(
            len(X_test),
            size=shap_size,
            replace=False,
        )
    )

    background_size = min(
        BACKGROUND_SIZE,
        len(X_test),
    )

    background_indices = np.sort(
        rng.choice(
            len(X_test),
            size=background_size,
            replace=False,
        )
    )

    X_shap = X_test.iloc[
        shap_indices
    ].copy()

    X_background = X_test.iloc[
        background_indices
    ].copy()

    print(
        f"Fixed SHAP observations: {len(X_shap)}"
    )

    print(
        f"Fixed background observations: {len(X_background)}"
    )

    print(
        "These observations remain identical across all repetitions."
    )

    # =========================================================================
    # SAVE FIXED INDICES
    # =========================================================================

    pd.DataFrame({
        "test_position":
            shap_indices
    }).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "fixed_shap_test_indices.csv",
        ),
        index=False,
    )

    pd.DataFrame({
        "test_position":
            background_indices
    }).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "fixed_shap_background_indices.csv",
        ),
        index=False,
    )

    # =========================================================================
    # STORAGE
    # =========================================================================

    all_global_importance = []

    all_local_shap = []

    all_run_metadata = []

    # =========================================================================
    # REPETITIONS
    # =========================================================================

    print(
        "\n[4/8] Running bootstrap repetitions..."
    )

    for repetition in range(
        1,
        N_REPETITIONS + 1,
    ):

        repetition_seed = (
            RANDOM_STATE
            + repetition
        )

        print(
            "\n" + "-" * 80
        )

        print(
            f"REPETITION {repetition}/{N_REPETITIONS}"
        )

        print(
            f"Random seed: {repetition_seed}"
        )

        print(
            "-" * 80
        )

        # =====================================================================
        # BOOTSTRAP DEVELOPMENT DATA
        # =====================================================================

        rng_rep = np.random.RandomState(
            repetition_seed
        )

        bootstrap_indices = rng_rep.choice(
            len(X_dev),
            size=len(X_dev),
            replace=True,
        )

        X_bootstrap = X_dev.iloc[
            bootstrap_indices
        ].reset_index(
            drop=True
        )

        y_bootstrap = y_dev.iloc[
            bootstrap_indices
        ].reset_index(
            drop=True
        )

        # =====================================================================
        # SMOTE
        # =====================================================================

        smote = SMOTE(
            random_state=repetition_seed,
            k_neighbors=5,
        )

        X_resampled, y_resampled = smote.fit_resample(
            X_bootstrap,
            y_bootstrap,
        )

        X_resampled = pd.DataFrame(
            X_resampled,
            columns=FEATURES,
        )

        y_resampled = pd.Series(
            y_resampled,
            name=TARGET,
        )

        print(
            f"Bootstrap samples: {len(X_bootstrap)}"
        )

        print(
            f"After SMOTE: {len(X_resampled)}"
        )

        # =====================================================================
        # MODEL LOOP
        # =====================================================================

        for model_name, config in MODEL_CONFIGS.items():

            print(
                f"\n  Training {model_name}..."
            )

            model = clone(
                config["model"]
            )

            # Set repetition-specific seed
            if model_name == "RF":

                model.set_params(
                    random_state=repetition_seed
                )

            elif model_name == "XGB":

                model.set_params(
                    random_state=repetition_seed
                )

            elif model_name == "LGBM":

                model.set_params(
                    random_state=repetition_seed
                )

            elif model_name == "MLP":

                model.set_params(
                    model__random_state=repetition_seed
                )

            model.fit(
                X_resampled,
                y_resampled,
            )

            # =================================================================
            # SAVE MODEL
            # =================================================================

            model_path = os.path.join(
                MODEL_DIR,
                f"{model_name}_repetition_{repetition}.pkl",
            )

            joblib.dump(
                model,
                model_path,
            )

            print(
                f"  Model saved: {model_path}"
            )

            # =================================================================
            # SHAP
            # =================================================================

            print(
                f"  Calculating {model_name} SHAP..."
            )

            if model_name in [
                "RF",
                "XGB",
                "LGBM",
            ]:

                shap_values = calculate_tree_shap(
                    model,
                    X_background,
                    X_shap,
                )

            else:

                shap_values = calculate_mlp_shap(
                    model,
                    X_background,
                    X_shap,
                )

            expected_shape = (
                len(X_shap),
                len(FEATURES),
            )

            if shap_values.shape != expected_shape:

                raise ValueError(
                    f"{model_name} repetition "
                    f"{repetition} produced SHAP shape "
                    f"{shap_values.shape}; "
                    f"expected {expected_shape}"
                )

            print(
                f"  SHAP shape: {shap_values.shape}"
            )

            # =================================================================
            # GLOBAL IMPORTANCE
            # =================================================================

            mean_abs = np.mean(
                np.abs(shap_values),
                axis=0,
            )

            normalized = normalize(
                mean_abs
            )

            for feature, raw, norm in zip(
                FEATURES,
                mean_abs,
                normalized,
            ):

                all_global_importance.append({
                    "Model":
                        model_name,
                    "Repetition":
                        repetition,
                    "Feature":
                        feature,
                    "Mean_Absolute_SHAP":
                        raw,
                    "Normalized_SHAP":
                        norm,
                })

            # =================================================================
            # LOCAL SHAP
            # =================================================================

            for sample_position in range(
                len(X_shap)
            ):

                row = {
                    "Model":
                        model_name,
                    "Repetition":
                        repetition,
                    "test_position":
                        int(
                            shap_indices[
                                sample_position
                            ]
                        ),
                }

                for feature_position, feature in enumerate(
                    FEATURES
                ):

                    row[
                        feature
                    ] = shap_values[
                        sample_position,
                        feature_position,
                    ]

                all_local_shap.append(
                    row
                )

            all_run_metadata.append({
                "Model":
                    model_name,
                "Repetition":
                    repetition,
                "Random_Seed":
                    repetition_seed,
                "Bootstrap_Sample_Size":
                    len(X_bootstrap),
                "SMOTE_Sample_Size":
                    len(X_resampled),
                "SHAP_Test_Size":
                    len(X_shap),
                "Background_Size":
                    len(X_background),
            })

    # =========================================================================
    # SAVE RAW RESULTS
    # =========================================================================

    print(
        "\n[5/8] Saving raw SHAP results..."
    )

    global_df = pd.DataFrame(
        all_global_importance
    )

    global_file = os.path.join(
        OUTPUT_DIR,
        "intra_model_global_shap_importance.csv",
    )

    global_df.to_csv(
        global_file,
        index=False,
    )

    local_df = pd.DataFrame(
        all_local_shap
    )

    local_file = os.path.join(
        OUTPUT_DIR,
        "intra_model_local_shap_values.csv",
    )

    local_df.to_csv(
        local_file,
        index=False,
    )

    metadata_df = pd.DataFrame(
        all_run_metadata
    )

    metadata_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "intra_model_run_metadata.csv",
        ),
        index=False,
    )

    # =========================================================================
    # GLOBAL STABILITY
    # =========================================================================

    print(
        "\n[6/8] Calculating global intra-model stability..."
    )

    global_pair_rows = []

    for model_name in MODEL_CONFIGS.keys():

        model_data = global_df[
            global_df[
                "Model"
            ] == model_name
        ]

        repetitions = sorted(
            model_data[
                "Repetition"
            ].unique()
        )

        for i in range(
            len(repetitions)
        ):

            for j in range(
                i + 1,
                len(repetitions)
            ):

                rep_a = repetitions[i]

                rep_b = repetitions[j]

                a = model_data[
                    model_data[
                        "Repetition"
                    ] == rep_a
                ].sort_values(
                    "Feature"
                )

                b = model_data[
                    model_data[
                        "Repetition"
                    ] == rep_b
                ].sort_values(
                    "Feature"
                )

                # Align features explicitly
                merged = a[
                    [
                        "Feature",
                        "Normalized_SHAP",
                    ]
                ].merge(
                    b[
                        [
                            "Feature",
                            "Normalized_SHAP",
                        ]
                    ],
                    on="Feature",
                    suffixes=(
                        "_A",
                        "_B",
                    ),
                )

                rho = safe_spearman(
                    merged[
                        "Normalized_SHAP_A"
                    ],
                    merged[
                        "Normalized_SHAP_B"
                    ],
                )

                cosine = safe_cosine(
                    merged[
                        "Normalized_SHAP_A"
                    ],
                    merged[
                        "Normalized_SHAP_B"
                    ],
                )

                global_pair_rows.append({
                    "Model":
                        model_name,
                    "Repetition_A":
                        rep_a,
                    "Repetition_B":
                        rep_b,
                    "Global_Spearman":
                        rho,
                    "Global_Cosine":
                        cosine,
                })

    global_stability_df = pd.DataFrame(
        global_pair_rows
    )

    global_stability_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "global_intra_model_stability.csv",
        ),
        index=False,
    )

    global_summary = (
        global_stability_df
        .groupby(
            "Model"
        )[
            [
                "Global_Spearman",
                "Global_Cosine",
            ]
        ]
        .agg(
            [
                "mean",
                "std",
                "median",
                "min",
                "max",
            ]
        )
    )

    global_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "global_intra_model_stability_summary.csv",
        )
    )

    # =========================================================================
    # LOCAL STABILITY
    # =========================================================================

    print(
        "\n[7/8] Calculating local intra-model stability..."
    )

    local_pair_rows = []

    for model_name in MODEL_CONFIGS.keys():

        model_local = local_df[
            local_df[
                "Model"
            ] == model_name
        ]

        repetitions = sorted(
            model_local[
                "Repetition"
            ].unique()
        )

        for i in range(
            len(repetitions)
        ):

            for j in range(
                i + 1,
                len(repetitions)
            ):

                rep_a = repetitions[i]

                rep_b = repetitions[j]

                a = model_local[
                    model_local[
                        "Repetition"
                    ] == rep_a
                ].sort_values(
                    "test_position"
                )

                b = model_local[
                    model_local[
                        "Repetition"
                    ] == rep_b
                ].sort_values(
                    "test_position"
                )

                for position in range(
                    len(X_shap)
                ):

                    row_a = a.iloc[
                        position
                    ]

                    row_b = b.iloc[
                        position
                    ]

                    shap_a = np.array([
                        row_a[
                            feature
                        ]
                        for feature in FEATURES
                    ])

                    shap_b = np.array([
                        row_b[
                            feature
                        ]
                        for feature in FEATURES
                    ])

                    rho = safe_spearman(
                        shap_a,
                        shap_b,
                    )

                    cosine = safe_cosine(
                        shap_a,
                        shap_b,
                    )

                    local_pair_rows.append({
                        "Model":
                            model_name,
                        "Repetition_A":
                            rep_a,
                        "Repetition_B":
                            rep_b,
                        "test_position":
                            int(
                                row_a[
                                    "test_position"
                                ]
                            ),
                        "Local_Spearman":
                            rho,
                        "Local_Cosine":
                            cosine,
                    })

    local_stability_df = pd.DataFrame(
        local_pair_rows
    )

    local_stability_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "local_intra_model_stability.csv",
        ),
        index=False,
    )

    local_summary = (
        local_stability_df
        .groupby(
            "Model"
        )[
            [
                "Local_Spearman",
                "Local_Cosine",
            ]
        ]
        .agg(
            [
                "mean",
                "std",
                "median",
                "min",
                "max",
            ]
        )
    )

    local_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "local_intra_model_stability_summary.csv",
        )
    )

    # =========================================================================
    # COMBINED SUMMARY
    # =========================================================================

    combined_rows = []

    for model_name in MODEL_CONFIGS.keys():

        g = global_stability_df[
            global_stability_df[
                "Model"
            ] == model_name
        ]

        l = local_stability_df[
            local_stability_df[
                "Model"
            ] == model_name
        ]

        combined_rows.append({
            "Model":
                model_name,

            "Global_Spearman_Mean":
                g[
                    "Global_Spearman"
                ].mean(),

            "Global_Spearman_STD":
                g[
                    "Global_Spearman"
                ].std(),

            "Global_Cosine_Mean":
                g[
                    "Global_Cosine"
                ].mean(),

            "Global_Cosine_STD":
                g[
                    "Global_Cosine"
                ].std(),

            "Local_Spearman_Mean":
                l[
                    "Local_Spearman"
                ].mean(),

            "Local_Spearman_STD":
                l[
                    "Local_Spearman"
                ].std(),

            "Local_Cosine_Mean":
                l[
                    "Local_Cosine"
                ].mean(),

            "Local_Cosine_STD":
                l[
                    "Local_Cosine"
                ].std(),
        })

    combined_df = pd.DataFrame(
        combined_rows
    )

    combined_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "intra_model_shap_stability_summary.csv",
        ),
        index=False,
    )

    # =========================================================================
    # FIGURE 1 — GLOBAL SPEARMAN
    # =========================================================================

    print(
        "\n[8/8] Creating figures and report..."
    )

    plot_df = (
        global_stability_df
        .groupby(
            "Model"
        )[
            "Global_Spearman"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        plot_df.index,
        plot_df.values,
    )

    plt.ylabel(
        "Mean Global Spearman ρ"
    )

    plt.xlabel(
        "Model"
    )

    plt.title(
        "Intra-Model Global SHAP Stability"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "intra_model_global_spearman.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # =========================================================================
    # FIGURE 2 — GLOBAL COSINE
    # =========================================================================

    plot_df = (
        global_stability_df
        .groupby(
            "Model"
        )[
            "Global_Cosine"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        plot_df.index,
        plot_df.values,
    )

    plt.ylabel(
        "Mean Global Cosine Similarity"
    )

    plt.xlabel(
        "Model"
    )

    plt.title(
        "Intra-Model Global SHAP Cosine Stability"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "intra_model_global_cosine.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # =========================================================================
    # FIGURE 3 — LOCAL SPEARMAN
    # =========================================================================

    plot_df = (
        local_stability_df
        .groupby(
            "Model"
        )[
            "Local_Spearman"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        plot_df.index,
        plot_df.values,
    )

    plt.ylabel(
        "Mean Local Spearman ρ"
    )

    plt.xlabel(
        "Model"
    )

    plt.title(
        "Intra-Model Local SHAP Stability"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "intra_model_local_spearman.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # =========================================================================
    # FIGURE 4 — LOCAL COSINE
    # =========================================================================

    plot_df = (
        local_stability_df
        .groupby(
            "Model"
        )[
            "Local_Cosine"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        plot_df.index,
        plot_df.values,
    )

    plt.ylabel(
        "Mean Local Cosine Similarity"
    )

    plt.xlabel(
        "Model"
    )

    plt.title(
        "Intra-Model Local SHAP Cosine Stability"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "intra_model_local_cosine.png",
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
        "intra_model_shap_stability_report.txt",
    )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "STEP 12 — INTRA-MODEL SHAP STABILITY\n"
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
            f"Bootstrap repetitions: {N_REPETITIONS}\n"
        )

        f.write(
            f"Fixed SHAP test observations: {len(X_shap)}\n"
        )

        f.write(
            f"Fixed background observations: {len(X_background)}\n"
        )

        f.write(
            "SMOTE applied independently in every repetition.\n"
        )

        f.write(
            "Independent test observations were never used for model fitting.\n\n"
        )

        f.write(
            "GLOBAL STABILITY SUMMARY\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            global_summary.to_string()
        )

        f.write(
            "\n\nLOCAL STABILITY SUMMARY\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            local_summary.to_string()
        )

        f.write(
            "\n\nCOMBINED SUMMARY\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            combined_df.to_string(
                index=False
            )
        )

    # =========================================================================
    # METADATA
    # =========================================================================

    metadata = {
        "step":
            "Step 12 — Intra-Model SHAP Stability",

        "random_state":
            RANDOM_STATE,

        "n_repetitions":
            N_REPETITIONS,

        "shap_test_size":
            len(X_shap),

        "background_size":
            len(X_background),

        "development_size":
            len(X_dev),

        "test_size":
            len(X_test),

        "features":
            FEATURES,

        "models":
            list(
                MODEL_CONFIGS.keys()
            ),

        "smote":
            {
                "enabled":
                    True,
                "k_neighbors":
                    5,
            },

        "design":
            "Bootstrap development-data perturbation + "
            "SMOTE within each repetition + fixed independent "
            "test subset for explanation evaluation.",

        "shap_test_indices":
            [
                int(x)
                for x in shap_indices
            ],

        "background_indices":
            [
                int(x)
                for x in background_indices
            ],
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "intra_model_shap_stability_metadata.json",
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
    # FINAL TERMINAL OUTPUT
    # =========================================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 12 COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        "\nGLOBAL STABILITY:"
    )

    print(
        combined_df[
            [
                "Model",
                "Global_Spearman_Mean",
                "Global_Cosine_Mean",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nLOCAL STABILITY:"
    )

    print(
        combined_df[
            [
                "Model",
                "Local_Spearman_Mean",
                "Local_Cosine_Mean",
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
        "STEP 13 — SHAP STABILITY INDEX (SSI)"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore"
    )

    main()
