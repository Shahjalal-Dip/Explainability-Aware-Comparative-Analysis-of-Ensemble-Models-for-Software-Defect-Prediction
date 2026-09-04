# =============================================================================
# 11_multicollinearity_analysis.py
#
# STEP 10: MULTICOLLINEARITY ANALYSIS
#
# Research Project:
# Evaluating SHAP Explanation Consistency in Ensemble Models
# for Software Defect Prediction
#
# Dataset:
# NASA JM1
#
# Purpose:
#   1. Calculate Pearson correlation among the 21 software metrics
#   2. Identify highly correlated feature pairs
#   3. Calculate Variance Inflation Factor (VIF)
#   4. Identify potentially problematic multicollinearity
#   5. Generate research-ready CSV files, figures, and report
#
# IMPORTANT:
#   - Development set ONLY
#   - Locked independent test set is NOT used
#   - No feature is removed at this stage
#   - Results will be used in Step 11: RF vs RF-RID
#
# Random state is not required because no sampling/model training is performed.
# =============================================================================

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = r"E:\Programming\Evaluating SHAP Explanation Consistency in Ensemble Models for Software Defect Prediction"

# Primary development dataset
INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "JM1",
    "JM1_development.csv"
)

# Fallback path
FALLBACK_INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "JM1_development.csv"
)

TARGET_COLUMN = "defects"

# Correlation threshold for identifying highly correlated feature pairs
CORRELATION_THRESHOLD = 0.80

# Severe VIF threshold
VIF_SEVERE_THRESHOLD = 10.0

# Moderate/high VIF threshold
VIF_MODERATE_THRESHOLD = 5.0

# Output directories
RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "results",
    "multicollinearity"
)

FIGURES_DIR = os.path.join(
    PROJECT_ROOT,
    "figures",
    "multicollinearity"
)


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

EXPECTED_FEATURES = [
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
    "branchCount"
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def ensure_directories():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def find_input_dataset():
    """
    Locate the development dataset.
    """

    if os.path.exists(INPUT_PATH):
        return INPUT_PATH

    if os.path.exists(FALLBACK_INPUT_PATH):
        return FALLBACK_INPUT_PATH

    raise FileNotFoundError(
        "\nDevelopment dataset could not be found.\n\n"
        f"Checked:\n"
        f"1. {INPUT_PATH}\n"
        f"2. {FALLBACK_INPUT_PATH}\n"
    )


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():

    print_header("LOADING JM1 DEVELOPMENT DATASET")

    input_path = find_input_dataset()

    print(f"Input dataset:")
    print(input_path)

    df = pd.read_csv(input_path)

    print(f"\nDataset shape: {df.shape}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found."
        )

    missing_features = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "The following expected features are missing:\n"
            + "\n".join(missing_features)
        )

    print(f"Target column: {TARGET_COLUMN}")
    print(f"Number of features: {len(EXPECTED_FEATURES)}")

    return df, input_path


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_data(df):

    print_header("DATA VALIDATION")

    X = df[EXPECTED_FEATURES].copy()

    # Check numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()

    if non_numeric:
        raise ValueError(
            "Non-numeric feature columns detected:\n"
            + "\n".join(non_numeric)
        )

    print("All 21 features are numeric.")

    # Missing values
    missing_values = X.isnull().sum().sum()

    print(f"Total missing feature values: {missing_values}")

    if missing_values > 0:
        print("\nMissing values by feature:")
        print(X.isnull().sum()[X.isnull().sum() > 0])

        raise ValueError(
            "Missing values detected. Multicollinearity analysis "
            "requires handling them before continuing."
        )

    # Infinite values
    infinite_values = np.isinf(X.to_numpy()).sum()

    print(f"Total infinite feature values: {infinite_values}")

    if infinite_values > 0:
        raise ValueError(
            "Infinite values detected in the feature matrix."
        )

    print("\nData validation PASSED.")


# =============================================================================
# PEARSON CORRELATION
# =============================================================================

def calculate_pearson_correlation(df):

    print_header("PEARSON CORRELATION ANALYSIS")

    X = df[EXPECTED_FEATURES].copy()

    correlation_matrix = X.corr(method="pearson")

    output_path = os.path.join(
        RESULTS_DIR,
        "pearson_correlation_matrix.csv"
    )

    correlation_matrix.to_csv(output_path)

    print("Pearson correlation matrix calculated.")

    print(f"\nSaved:")
    print(output_path)

    # -------------------------------------------------------------------------
    # Identify high-correlation pairs
    # -------------------------------------------------------------------------

    high_corr_pairs = []

    for i in range(len(EXPECTED_FEATURES)):
        for j in range(i + 1, len(EXPECTED_FEATURES)):

            feature_1 = EXPECTED_FEATURES[i]
            feature_2 = EXPECTED_FEATURES[j]

            r = correlation_matrix.loc[feature_1, feature_2]

            if abs(r) >= CORRELATION_THRESHOLD:

                high_corr_pairs.append({
                    "Feature_1": feature_1,
                    "Feature_2": feature_2,
                    "Pearson_r": r,
                    "Absolute_r": abs(r)
                })

    high_corr_pairs_df = pd.DataFrame(high_corr_pairs)

    if not high_corr_pairs_df.empty:

        high_corr_pairs_df = high_corr_pairs_df.sort_values(
            by="Absolute_r",
            ascending=False
        ).reset_index(drop=True)

    else:

        high_corr_pairs_df = pd.DataFrame(
            columns=[
                "Feature_1",
                "Feature_2",
                "Pearson_r",
                "Absolute_r"
            ]
        )

    high_corr_path = os.path.join(
        RESULTS_DIR,
        "highly_correlated_feature_pairs.csv"
    )

    high_corr_pairs_df.to_csv(
        high_corr_path,
        index=False
    )

    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------

    print("\n" + "-" * 90)
    print(
        f"HIGH CORRELATION PAIRS | |r| >= {CORRELATION_THRESHOLD}"
    )
    print("-" * 90)

    if high_corr_pairs_df.empty:

        print("No highly correlated feature pairs found.")

    else:

        for _, row in high_corr_pairs_df.iterrows():

            print(
                f"{row['Feature_1']} <-> "
                f"{row['Feature_2']} : "
                f"r = {row['Pearson_r']:.4f}"
            )

    print(f"\nNumber of highly correlated pairs: {len(high_corr_pairs_df)}")

    print(f"\nSaved:")
    print(high_corr_path)

    return correlation_matrix, high_corr_pairs_df


# =============================================================================
# VIF CALCULATION
# =============================================================================

def calculate_vif(df):

    print_header("VARIANCE INFLATION FACTOR (VIF) ANALYSIS")

    X = df[EXPECTED_FEATURES].copy()

    # Convert to float explicitly
    X = X.astype(float)

    vif_results = []

    print("\nCalculating VIF for all 21 features...")
    print("This may take a short while.\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for i, feature in enumerate(EXPECTED_FEATURES):

            try:
                vif_value = variance_inflation_factor(
                    X.values,
                    i
                )

            except Exception as exc:

                print(
                    f"Warning: VIF calculation failed for "
                    f"{feature}: {exc}"
                )

                vif_value = np.inf

            # Handle numerical instability
            if np.isnan(vif_value):
                vif_value = np.inf

            # Classify VIF
            if np.isinf(vif_value):

                interpretation = "Severe / Infinite"

            elif vif_value >= VIF_SEVERE_THRESHOLD:

                interpretation = "Severe"

            elif vif_value >= VIF_MODERATE_THRESHOLD:

                interpretation = "Moderate / High"

            else:

                interpretation = "Low / Acceptable"

            vif_results.append({
                "Feature": feature,
                "VIF": vif_value,
                "Interpretation": interpretation
            })

    vif_df = pd.DataFrame(vif_results)

    # Sort from highest VIF to lowest
    vif_df = vif_df.sort_values(
        by="VIF",
        ascending=False
    ).reset_index(drop=True)

    output_path = os.path.join(
        RESULTS_DIR,
        "vif_results.csv"
    )

    vif_df.to_csv(
        output_path,
        index=False
    )

    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------

    print("-" * 90)
    print("VIF RESULTS")
    print("-" * 90)

    for _, row in vif_df.iterrows():

        vif_value = row["VIF"]

        if np.isinf(vif_value):

            vif_text = "INF"

        else:

            vif_text = f"{vif_value:.4f}"

        print(
            f"{row['Feature']:<20} "
            f"VIF = {vif_text:<12} "
            f"{row['Interpretation']}"
        )

    # Summary
    infinite_count = np.isinf(vif_df["VIF"]).sum()

    severe_count = (
        (vif_df["VIF"] >= VIF_SEVERE_THRESHOLD)
        & np.isfinite(vif_df["VIF"])
    ).sum()

    moderate_count = (
        (vif_df["VIF"] >= VIF_MODERATE_THRESHOLD)
        & (vif_df["VIF"] < VIF_SEVERE_THRESHOLD)
    ).sum()

    acceptable_count = (
        vif_df["VIF"] < VIF_MODERATE_THRESHOLD
    ).sum()

    print("\n" + "-" * 90)
    print("VIF SUMMARY")
    print("-" * 90)

    print(f"Infinite VIF:        {infinite_count}")
    print(f"Severe VIF (>=10):   {severe_count}")
    print(f"Moderate VIF (5-10): {moderate_count}")
    print(f"Acceptable VIF (<5): {acceptable_count}")

    print(f"\nSaved:")
    print(output_path)

    return vif_df


# =============================================================================
# PEARSON CORRELATION HEATMAP
# =============================================================================

def create_correlation_heatmap(correlation_matrix):

    print_header("GENERATING PEARSON CORRELATION HEATMAP")

    fig, ax = plt.subplots(
        figsize=(14, 12)
    )

    matrix = correlation_matrix.values

    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest"
    )

    # Axis labels
    ax.set_xticks(
        np.arange(len(correlation_matrix.columns))
    )

    ax.set_yticks(
        np.arange(len(correlation_matrix.index))
    )

    ax.set_xticklabels(
        correlation_matrix.columns,
        rotation=90
    )

    ax.set_yticklabels(
        correlation_matrix.index
    )

    ax.set_title(
        "Pearson Correlation Matrix of JM1 Software Metrics",
        fontsize=15,
        pad=15
    )

    # Add correlation values
    for i in range(matrix.shape[0]):

        for j in range(matrix.shape[1]):

            value = matrix[i, j]

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7
            )

    cbar = fig.colorbar(image, ax=ax)

    cbar.set_label(
        "Pearson correlation coefficient",
        rotation=270,
        labelpad=20
    )

    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "pearson_correlation_heatmap.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved:")
    print(output_path)


# =============================================================================
# VIF PLOT - MODERN AND CLEAN VERSION
# =============================================================================

def create_vif_plot(vif_df):

    print_header("GENERATING VIF PLOT")

    plot_df = vif_df.copy()

    # Handle infinite values
    finite_vifs = plot_df.loc[np.isfinite(plot_df["VIF"]), "VIF"]
    max_finite = finite_vifs.max() if len(finite_vifs) > 0 else 20
    display_cap = max(25, max_finite * 1.15)

    plot_df["VIF_plot"] = plot_df["VIF"].replace(np.inf, display_cap)
    plot_df = plot_df.sort_values(by="VIF_plot", ascending=True).reset_index(drop=True)

    # Create figure with clean proportions
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color mapping function with professional palette
    def get_vif_color(vif):
        if np.isinf(vif) or vif >= 10:
            return '#C0392B'  # Deep red - severe
        elif vif >= 5:
            return '#E67E22'  # Orange - moderate
        elif vif >= 2:
            return '#3498DB'  # Blue - elevated
        else:
            return '#2ECC71'  # Green - acceptable

    colors = [get_vif_color(vif) for vif in plot_df["VIF"]]

    # Create horizontal bar chart with clean styling
    bars = ax.barh(
        plot_df["Feature"],
        plot_df["VIF_plot"],
        color=colors,
        alpha=0.85,
        height=0.6,
        edgecolor='white',
        linewidth=0.5
    )

    # Add value labels with proper formatting
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        vif = row["VIF"]
        vif_plot = row["VIF_plot"]

        if np.isinf(vif):
            label = "∞"
            x_pos = vif_plot - 1.5
            ha = 'right'
            color = 'white'
            weight = 'bold'
        else:
            label = f"{vif:.1f}"
            x_pos = vif_plot + 0.5
            ha = 'left'
            color = '#2C3E50'
            weight = 'bold' if vif >= 10 else 'normal'

        ax.text(
            x_pos, i, label,
            va='center', ha=ha,
            fontsize=9, fontweight=weight,
            color=color
        )

    # Threshold lines with clear styling
    ax.axvline(
        VIF_MODERATE_THRESHOLD,
        linestyle='--',
        linewidth=1.5,
        color='#F39C12',
        alpha=0.7
    )

    ax.axvline(
        VIF_SEVERE_THRESHOLD,
        linestyle='--',
        linewidth=1.5,
        color='#E74C3C',
        alpha=0.7
    )

    # Customize axis labels and title
    ax.set_xlabel(
        'Variance Inflation Factor (VIF)',
        fontsize=12,
        fontweight='bold',
        color='#2C3E50'
    )

    ax.set_ylabel(
        '',
        fontsize=12
    )

    ax.set_title(
        'Multicollinearity Analysis: Variance Inflation Factor (VIF)',
        fontsize=14,
        fontweight='bold',
        pad=15,
        color='#2C3E50'
    )

    # Add subtle grid
    ax.grid(
        axis='x',
        linestyle=':',
        alpha=0.4,
        color='#95A5A6'
    )

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')

    # Customize tick labels
    ax.tick_params(
        axis='both',
        labelsize=10,
        colors='#2C3E50'
    )

    # Add legend with color categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#C0392B', label='Severe (VIF ≥ 10)'),
        Patch(facecolor='#E67E22', label='Moderate (5 ≤ VIF < 10)'),
        Patch(facecolor='#3498DB', label='Elevated (2 ≤ VIF < 5)'),
        Patch(facecolor='#2ECC71', label='Acceptable (VIF < 2)')
    ]

    ax.legend(
        handles=legend_elements,
        loc='lower right',
        fontsize=9,
        frameon=False
    )

    # Add summary statistics box
    infinite_count = np.isinf(plot_df["VIF"]).sum()
    severe_count = ((plot_df["VIF"] >= 10) & np.isfinite(plot_df["VIF"])).sum()
    moderate_count = ((plot_df["VIF"] >= 5) & (plot_df["VIF"] < 10)).sum()
    elevated_count = ((plot_df["VIF"] >= 2) & (plot_df["VIF"] < 5) & np.isfinite(plot_df["VIF"])).sum()
    acceptable_count = ((plot_df["VIF"] < 2) & np.isfinite(plot_df["VIF"])).sum()

    summary_text = (
        f"VIF Summary\n"
        f"{'─' * 20}\n"
        f"∞ Severe/Infinite: {infinite_count}\n"
        f"⚠ Severe (≥10): {severe_count}\n"
        f"⚡ Moderate (5-10): {moderate_count}\n"
        f"● Elevated (2-5): {elevated_count}\n"
        f"✓ Acceptable (<2): {acceptable_count}"
    )

    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=9,
        fontfamily='monospace',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            edgecolor='#BDC3C7',
            alpha=0.95
        ),
        color='#2C3E50'
    )

    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "vif_plot.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor='white'
    )

    plt.close()

    print(f"Saved:")
    print(output_path)



# =============================================================================
# GENERATE REPORT
# =============================================================================

def generate_report(
    df,
    input_path,
    correlation_matrix,
    high_corr_pairs_df,
    vif_df
):

    print_header("GENERATING MULTICOLLINEARITY REPORT")

    report_path = os.path.join(
        RESULTS_DIR,
        "multicollinearity_report.txt"
    )

    infinite_count = np.isinf(vif_df["VIF"]).sum()

    severe_count = (
        (vif_df["VIF"] >= VIF_SEVERE_THRESHOLD)
        & np.isfinite(vif_df["VIF"])
    ).sum()

    moderate_count = (
        (vif_df["VIF"] >= VIF_MODERATE_THRESHOLD)
        & (vif_df["VIF"] < VIF_SEVERE_THRESHOLD)
    ).sum()

    acceptable_count = (
        vif_df["VIF"] < VIF_MODERATE_THRESHOLD
    ).sum()

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as report:

        report.write(
            "=" * 90 + "\n"
        )

        report.write(
            "JM1 MULTICOLLINEARITY ANALYSIS REPORT\n"
        )

        report.write(
            "=" * 90 + "\n\n"
        )

        report.write(
            "RESEARCH PROJECT\n"
        )

        report.write(
            "Evaluating SHAP Explanation Consistency in Ensemble Models "
            "for Software Defect Prediction\n\n"
        )

        report.write(
            "DATASET INFORMATION\n"
        )

        report.write(
            "-" * 90 + "\n"
        )

        report.write(
            f"Input dataset: {input_path}\n"
        )

        report.write(
            f"Rows analyzed: {len(df):,}\n"
        )

        report.write(
            f"Columns: {len(df.columns)}\n"
        )

        report.write(
            f"Features analyzed: {len(EXPECTED_FEATURES)}\n"
        )

        report.write(
            f"Target column: {TARGET_COLUMN}\n"
        )

        report.write(
            "Dataset split: Development set only\n"
        )

        report.write(
            "Independent test set: NOT USED\n\n"
        )

        # ---------------------------------------------------------------------
        # Pearson
        # ---------------------------------------------------------------------

        report.write(
            "PEARSON CORRELATION ANALYSIS\n"
        )

        report.write(
            "-" * 90 + "\n"
        )

        report.write(
            f"High-correlation threshold: |r| >= "
            f"{CORRELATION_THRESHOLD}\n"
        )

        report.write(
            f"Number of high-correlation pairs: "
            f"{len(high_corr_pairs_df)}\n\n"
        )

        if high_corr_pairs_df.empty:

            report.write(
                "No feature pairs exceeded the correlation threshold.\n\n"
            )

        else:

            for _, row in high_corr_pairs_df.iterrows():

                report.write(
                    f"{row['Feature_1']} <-> "
                    f"{row['Feature_2']} : "
                    f"r = {row['Pearson_r']:.4f}\n"
                )

            report.write("\n")

        # ---------------------------------------------------------------------
        # VIF
        # ---------------------------------------------------------------------

        report.write(
            "VARIANCE INFLATION FACTOR (VIF)\n"
        )

        report.write(
            "-" * 90 + "\n"
        )

        report.write(
            "Interpretation thresholds:\n"
        )

        report.write(
            "  VIF < 5     : Low / generally acceptable\n"
        )

        report.write(
            "  5 <= VIF < 10 : Moderate / high\n"
        )

        report.write(
            "  VIF >= 10   : Severe multicollinearity\n"
        )

        report.write(
            "  Infinite VIF: Perfect or near-perfect linear dependence\n\n"
        )

        for _, row in vif_df.iterrows():

            vif_value = row["VIF"]

            if np.isinf(vif_value):

                vif_text = "INF"

            else:

                vif_text = f"{vif_value:.6f}"

            report.write(
                f"{row['Feature']:<20} "
                f"VIF = {vif_text:<15} "
                f"{row['Interpretation']}\n"
            )

        report.write("\n")

        # ---------------------------------------------------------------------
        # Summary
        # ---------------------------------------------------------------------

        report.write(
            "SUMMARY\n"
        )

        report.write(
            "-" * 90 + "\n"
        )

        report.write(
            f"Features with infinite VIF: {infinite_count}\n"
        )

        report.write(
            f"Features with severe VIF (>=10): {severe_count}\n"
        )

        report.write(
            f"Features with moderate/high VIF (5-10): "
            f"{moderate_count}\n"
        )

        report.write(
            f"Features with VIF <5: {acceptable_count}\n\n"
        )

        # ---------------------------------------------------------------------
        # Methodological interpretation
        # ---------------------------------------------------------------------

        report.write(
            "METHODOLOGICAL INTERPRETATION\n"
        )

        report.write(
            "-" * 90 + "\n"
        )

        report.write(
            "This analysis was conducted exclusively on the development set. "
            "The independent test set remained completely untouched.\n\n"
        )

        report.write(
            "Pearson correlation measures pairwise linear association between "
            "software metrics, while VIF evaluates the extent to which each "
            "feature can be explained by the remaining features collectively.\n\n"
        )

        report.write(
            "The presence of highly correlated features does not by itself "
            "justify removing features at this stage. Instead, these results "
            "provide the empirical basis for the subsequent RF-RID experiment, "
            "where redundant features will be reduced and the effects on both "
            "predictive performance and SHAP explanations will be evaluated.\n\n"
        )

        report.write(
            "This distinction is important because the research objective is "
            "not simply to maximize predictive performance, but to investigate "
            "whether feature redundancy influences explanation consistency "
            "across models.\n\n"
        )

        report.write(
            "NEXT ANALYSIS STAGE\n"
        )

        report.write(
            "-" * 90 + "\n"
        )

        report.write(
            "Step 11: RF vs RF-RID\n"
        )

        report.write(
            "The multicollinearity findings will be used to construct the "
            "redundancy-reduced feature set for the RF-RID experiment.\n"
        )

    print(f"Saved:")
    print(report_path)

    return report_path


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("\n")
    print("=" * 90)
    print("JM1 MULTICOLLINEARITY ANALYSIS")
    print("STEP 10: PEARSON CORRELATION + VIF")
    print("=" * 90)

    print("\nIMPORTANT:")
    print("- Development set ONLY")
    print("- Independent test set remains LOCKED")
    print("- No features are removed at this stage")
    print("- Results will support the RF-RID experiment")

    # -------------------------------------------------------------------------
    # Create output directories
    # -------------------------------------------------------------------------

    ensure_directories()

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    df, input_path = load_data()

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    validate_data(df)

    # -------------------------------------------------------------------------
    # Pearson correlation
    # -------------------------------------------------------------------------

    correlation_matrix, high_corr_pairs_df = (
        calculate_pearson_correlation(df)
    )

    # -------------------------------------------------------------------------
    # VIF
    # -------------------------------------------------------------------------

    vif_df = calculate_vif(df)

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------

    create_correlation_heatmap(
        correlation_matrix
    )

    create_vif_plot(
        vif_df
    )

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    report_path = generate_report(
        df=df,
        input_path=input_path,
        correlation_matrix=correlation_matrix,
        high_corr_pairs_df=high_corr_pairs_df,
        vif_df=vif_df
    )

    # -------------------------------------------------------------------------
    # Final output summary
    # -------------------------------------------------------------------------

    print_header("STEP 10 COMPLETED")

    print("Generated files:")
    print()
    print(
        os.path.join(
            RESULTS_DIR,
            "pearson_correlation_matrix.csv"
        )
    )

    print(
        os.path.join(
            RESULTS_DIR,
            "highly_correlated_feature_pairs.csv"
        )
    )

    print(
        os.path.join(
            RESULTS_DIR,
            "vif_results.csv"
        )
    )

    print(
        os.path.join(
            RESULTS_DIR,
            "multicollinearity_report.txt"
        )
    )

    print()
    print("Generated figures:")
    print()

    print(
        os.path.join(
            FIGURES_DIR,
            "pearson_correlation_heatmap.png"
        )
    )

    print(
        os.path.join(
            FIGURES_DIR,
            "vif_plot.png"
        )
    )

    print("\n" + "=" * 90)
    print("NEXT STEP: STEP 11 — RF vs RF-RID")
    print("=" * 90)


if __name__ == "__main__":
    main()
