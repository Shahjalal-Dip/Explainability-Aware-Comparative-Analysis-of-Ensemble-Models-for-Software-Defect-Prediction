"""
================================================================================
SOFTWARE DEFECT DATASET
CONFLICT / DUPLICATE ANALYSIS
================================================================================

Purpose:
    Analyze duplicate feature vectors and conflicting defect labels.

Dataset:
    Software defect prediction metrics

Main objectives:
    1. Load conflict-analysis files safely.
    2. Identify exact duplicates.
    3. Identify feature duplicates regardless of label.
    4. Identify conflicting feature vectors.
    5. Measure defect/non-defect label distribution.
    6. Calculate conflict severity.
    7. Calculate label disagreement / ambiguity.
    8. Produce publication-ready CSV and TXT outputs.

Important:
    This script uses paths relative to its own directory, so it can be
    executed from the project root:

        python data\results\cleaning\analyze_conflicts.py

================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np


# =============================================================================
# 1. PATH CONFIGURATION
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SUMMARY_FILE = os.path.join(
    SCRIPT_DIR,
    "conflicting_feature_groups_summary.csv"
)

ROWS_FILE = os.path.join(
    SCRIPT_DIR,
    "conflicting_rows.csv"
)

EXISTING_REPORT = os.path.join(
    SCRIPT_DIR,
    "duplicate_conflict_report.txt"
)


# Output files
DETAILED_GROUP_FILE = os.path.join(
    SCRIPT_DIR,
    "conflict_group_detailed_analysis.csv"
)

LABEL_DISTRIBUTION_FILE = os.path.join(
    SCRIPT_DIR,
    "conflict_label_distribution.csv"
)

ANALYSIS_REPORT = os.path.join(
    SCRIPT_DIR,
    "conflict_analysis_report.txt"
)


# =============================================================================
# 2. PRINT HELPERS
# =============================================================================

def print_section(title):
    """Print a formatted section heading."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_subsection(title):
    """Print a formatted subsection heading."""
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


# =============================================================================
# 3. CHECK INPUT FILES
# =============================================================================

print_section("SOFTWARE DEFECT DATASET - CONFLICT ANALYSIS")

print("\nScript directory:")
print(SCRIPT_DIR)

print_section("CHECKING INPUT FILES")

print("\nSummary file:")
print(SUMMARY_FILE)

print("\nConflicting rows file:")
print(ROWS_FILE)

print("\nExisting conflict report:")
print(EXISTING_REPORT)


missing_files = []

if not os.path.isfile(SUMMARY_FILE):
    missing_files.append(SUMMARY_FILE)

if not os.path.isfile(ROWS_FILE):
    missing_files.append(ROWS_FILE)

if missing_files:
    print("\nERROR: The following required file(s) were not found:")

    for file_path in missing_files:
        print("  -", file_path)

    print("\nMake sure the files exist in:")
    print(SCRIPT_DIR)

    sys.exit(1)


# =============================================================================
# 4. LOAD DATA
# =============================================================================

print_section("LOADING DATA")

try:
    summary_df = pd.read_csv(SUMMARY_FILE)
    rows_df = pd.read_csv(ROWS_FILE)
except Exception as e:
    print("\nERROR while loading CSV files:")
    print(e)
    sys.exit(1)


print("\nFiles loaded successfully.")

print(
    f"\nFeature-group summary shape: {summary_df.shape}"
)

print(
    f"Conflicting rows shape:      {rows_df.shape}"
)


# =============================================================================
# 5. COLUMN INFORMATION
# =============================================================================

print_section("COLUMN INFORMATION")

print("\nSummary columns:")

for col in summary_df.columns:
    print(f"  - {col}")


print("\nConflicting-row columns:")

for col in rows_df.columns:
    print(f"  - {col}")


# =============================================================================
# 6. BASIC DATA VALIDATION
# =============================================================================

print_section("DATA VALIDATION")

required_columns = [
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
    "defects"
]


missing_columns = [
    col for col in required_columns
    if col not in rows_df.columns
]

if missing_columns:
    print("\nWARNING: Missing expected columns:")

    for col in missing_columns:
        print("  -", col)

    print("\nThe analysis cannot continue safely.")
    sys.exit(1)


print("\nAll required columns are present.")


# =============================================================================
# 7. NORMALIZE DEFECT LABEL
# =============================================================================

print_section("NORMALIZING DEFECT LABEL")

print("\nOriginal defect values:")

print(rows_df["defects"].value_counts(dropna=False))


def normalize_defect(value):
    """
    Convert common representations of True/False into real booleans.

    This prevents pandas from treating True/False as row-selection masks
    during later groupby/reindex operations.
    """

    if pd.isna(value):
        return np.nan

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, np.integer)):
        if value == 1:
            return True

        if value == 0:
            return False

    if isinstance(value, (float, np.floating)):
        if value == 1.0:
            return True

        if value == 0.0:
            return False

    text = str(value).strip().lower()

    if text in ["true", "1", "yes", "defective", "defect"]:
        return True

    if text in ["false", "0", "no", "non-defective", "non_defective",
                "nondefective", "clean"]:
        return False

    return np.nan


rows_df["defects"] = rows_df["defects"].apply(normalize_defect)


print("\nNormalized defect values:")

print(rows_df["defects"].value_counts(dropna=False))


# =============================================================================
# 8. IDENTIFY FEATURE COLUMNS
# =============================================================================

print_section("IDENTIFYING FEATURE COLUMNS")

feature_columns = [
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


print(f"\nNumber of feature columns: {len(feature_columns)}")

print("\nFeatures used for conflict comparison:")

for feature in feature_columns:
    print(f"  - {feature}")


# =============================================================================
# 9. BASIC CONFLICT STATISTICS
# =============================================================================

print_section("BASIC CONFLICT STATISTICS")

number_of_conflict_groups = len(summary_df)

number_of_conflicting_rows = len(rows_df)

print(
    f"\nNumber of conflict groups : {number_of_conflict_groups}"
)

print(
    f"Number of conflicting rows: {number_of_conflicting_rows}"
)


# =============================================================================
# 10. DEFECT LABEL DISTRIBUTION
# =============================================================================

print_section("DEFECT LABEL DISTRIBUTION")

label_counts = rows_df["defects"].value_counts(dropna=False)

print("\nCounts:")
print(label_counts)


label_percentages = (
    rows_df["defects"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(2)
)

print("\nPercentages:")
print(label_percentages)


# Save label distribution
label_distribution_rows = []

for label in [False, True]:

    count = int(
        (rows_df["defects"] == label).sum()
    )

    percentage = (
        count / len(rows_df) * 100
        if len(rows_df) > 0
        else 0
    )

    label_distribution_rows.append(
        {
            "defect_label": label,
            "count": count,
            "percentage": round(percentage, 2)
        }
    )


label_distribution_df = pd.DataFrame(
    label_distribution_rows
)

label_distribution_df.to_csv(
    LABEL_DISTRIBUTION_FILE,
    index=False
)


# =============================================================================
# 11. EXACT DUPLICATE ANALYSIS
# =============================================================================

print_section("EXACT DUPLICATE ANALYSIS")

# Same features AND same label
exact_duplicate_mask = rows_df.duplicated(
    subset=feature_columns + ["defects"],
    keep=False
)

exact_duplicate_rows = int(
    exact_duplicate_mask.sum()
)


# Same features regardless of label
feature_duplicate_mask = rows_df.duplicated(
    subset=feature_columns,
    keep=False
)

feature_duplicate_rows = int(
    feature_duplicate_mask.sum()
)


print(
    f"\nExact duplicate rows "
    f"(same features + same defect label): "
    f"{exact_duplicate_rows}"
)

print(
    f"\nFeature duplicate rows "
    f"(same features, regardless of label): "
    f"{feature_duplicate_rows}"
)


# =============================================================================
# 12. FEATURE-LEVEL CONFLICT ANALYSIS
# =============================================================================

print_section("FEATURE-LEVEL CONFLICT ANALYSIS")

print(
    f"\nUnique conflicting feature vectors: "
    f"{number_of_conflict_groups}"
)

print(
    f"\nRows belonging to conflicting feature vectors: "
    f"{number_of_conflicting_rows}"
)


# =============================================================================
# 13. GROUP DATA BY FEATURE VECTOR
# =============================================================================

print_section("BUILDING CONFLICT GROUP ANALYSIS")

grouped = rows_df.groupby(
    feature_columns,
    dropna=False
)


detailed_results = []


for group_number, (feature_values, group) in enumerate(
    grouped,
    start=1
):

    # -------------------------------------------------------------------------
    # Convert tuple into dictionary
    # -------------------------------------------------------------------------

    if not isinstance(feature_values, tuple):
        feature_values = (feature_values,)

    feature_dict = dict(
        zip(feature_columns, feature_values)
    )

    # -------------------------------------------------------------------------
    # Label counts
    # -------------------------------------------------------------------------

    defective_count = int(
        (group["defects"] == True).sum()
    )

    non_defective_count = int(
        (group["defects"] == False).sum()
    )

    total_count = defective_count + non_defective_count

    # -------------------------------------------------------------------------
    # Check whether both labels exist
    # -------------------------------------------------------------------------

    is_conflict = (
        defective_count > 0
        and non_defective_count > 0
    )

    # -------------------------------------------------------------------------
    # Majority label
    # -------------------------------------------------------------------------

    if defective_count > non_defective_count:
        majority_label = True

    elif non_defective_count > defective_count:
        majority_label = False

    else:
        majority_label = "Tie"

    # -------------------------------------------------------------------------
    # Minority count
    # -------------------------------------------------------------------------

    majority_count = max(
        defective_count,
        non_defective_count
    )

    minority_count = min(
        defective_count,
        non_defective_count
    )

    # -------------------------------------------------------------------------
    # Conflict rate
    #
    # Percentage of observations belonging to the minority label.
    # -------------------------------------------------------------------------

    if total_count > 0:
        ambiguity_rate = (
            minority_count / total_count
        ) * 100
    else:
        ambiguity_rate = 0.0

    # -------------------------------------------------------------------------
    # Label disagreement
    #
    # 0 = no disagreement
    # >0 = conflicting labels
    # -------------------------------------------------------------------------

    if total_count > 0:
        disagreement_rate = (
            1 -
            majority_count / total_count
        ) * 100
    else:
        disagreement_rate = 0.0

    # -------------------------------------------------------------------------
    # Number of unique labels
    # -------------------------------------------------------------------------

    unique_labels = (
        group["defects"]
        .dropna()
        .nunique()
    )

    # -------------------------------------------------------------------------
    # Store result
    # -------------------------------------------------------------------------

    result = {}

    result.update(feature_dict)

    result.update(
        {
            "group_id": group_number,
            "total_rows": total_count,
            "defective_count": defective_count,
            "non_defective_count": non_defective_count,
            "unique_labels": unique_labels,
            "is_conflict": is_conflict,
            "majority_label": majority_label,
            "majority_count": majority_count,
            "minority_count": minority_count,
            "ambiguity_rate_percent": round(
                ambiguity_rate,
                2
            ),
            "disagreement_rate_percent": round(
                disagreement_rate,
                2
            )
        }
    )

    detailed_results.append(result)


# =============================================================================
# 14. CREATE DETAILED DATAFRAME
# =============================================================================

detailed_df = pd.DataFrame(
    detailed_results
)


# Sort by ambiguity / disagreement
detailed_df = detailed_df.sort_values(
    by=[
        "disagreement_rate_percent",
        "total_rows"
    ],
    ascending=[
        False,
        False
    ]
).reset_index(drop=True)


# =============================================================================
# 15. SAVE DETAILED GROUP ANALYSIS
# =============================================================================

detailed_df.to_csv(
    DETAILED_GROUP_FILE,
    index=False
)


print(
    f"\nDetailed conflict-group analysis saved to:"
)

print(
    DETAILED_GROUP_FILE
)


# =============================================================================
# 16. CONFLICT SEVERITY ANALYSIS
# =============================================================================

print_section("CONFLICT SEVERITY ANALYSIS")

conflict_only_df = detailed_df[
    detailed_df["is_conflict"] == True
].copy()


print(
    f"\nNumber of conflicting groups: "
    f"{len(conflict_only_df)}"
)


if len(conflict_only_df) > 0:

    # -------------------------------------------------------------------------
    # Perfect 50/50 conflicts
    # -------------------------------------------------------------------------

    perfect_tie_count = int(
        (
            conflict_only_df[
                "defective_count"
            ]
            ==
            conflict_only_df[
                "non_defective_count"
            ]
        ).sum()
    )

    print(
        f"\nPerfect 50/50 conflict groups: "
        f"{perfect_tie_count}"
    )

    # -------------------------------------------------------------------------
    # High ambiguity
    # -------------------------------------------------------------------------

    high_ambiguity_count = int(
        (
            conflict_only_df[
                "disagreement_rate_percent"
            ] >= 40
        ).sum()
    )

    print(
        f"\nHigh-ambiguity groups "
        f"(disagreement >= 40%): "
        f"{high_ambiguity_count}"
    )

    # -------------------------------------------------------------------------
    # Moderate ambiguity
    # -------------------------------------------------------------------------

    moderate_ambiguity_count = int(
        (
            (
                conflict_only_df[
                    "disagreement_rate_percent"
                ] >= 20
            )
            &
            (
                conflict_only_df[
                    "disagreement_rate_percent"
                ] < 40
            )
        ).sum()
    )

    print(
        f"\nModerate-ambiguity groups "
        f"(20% <= disagreement < 40%): "
        f"{moderate_ambiguity_count}"
    )

    # -------------------------------------------------------------------------
    # Low ambiguity
    # -------------------------------------------------------------------------

    low_ambiguity_count = int(
        (
            conflict_only_df[
                "disagreement_rate_percent"
            ] < 20
        ).sum()
    )

    print(
        f"\nLow-ambiguity groups "
        f"(disagreement < 20%): "
        f"{low_ambiguity_count}"
    )


# =============================================================================
# 17. TOP CONFLICT GROUPS
# =============================================================================

print_section("TOP CONFLICT GROUPS")

if len(conflict_only_df) > 0:

    top_columns = [
        "group_id",
        "total_rows",
        "defective_count",
        "non_defective_count",
        "majority_label",
        "minority_count",
        "ambiguity_rate_percent",
        "disagreement_rate_percent"
    ]

    top_conflicts = conflict_only_df[
        top_columns
    ].head(20)

    print(
        "\nTop 20 conflict groups:"
    )

    print(
        top_conflicts.to_string(
            index=False
        )
    )

else:

    print(
        "\nNo conflicting feature groups were found."
    )


# =============================================================================
# 18. OVERALL LABEL AMBIGUITY
# =============================================================================

print_section("OVERALL LABEL AMBIGUITY")

if len(conflict_only_df) > 0:

    total_conflict_observations = int(
        conflict_only_df[
            "total_rows"
        ].sum()
    )

    total_minority_observations = int(
        conflict_only_df[
            "minority_count"
        ].sum()
    )

    if total_conflict_observations > 0:

        overall_ambiguity_rate = (
            total_minority_observations
            /
            total_conflict_observations
        ) * 100

    else:

        overall_ambiguity_rate = 0.0

    print(
        f"\nTotal observations in conflict groups: "
        f"{total_conflict_observations}"
    )

    print(
        f"Minority-label observations in conflicts: "
        f"{total_minority_observations}"
    )

    print(
        f"Overall conflict ambiguity rate: "
        f"{overall_ambiguity_rate:.2f}%"
    )

else:

    overall_ambiguity_rate = 0.0

    print(
        "\nNo conflict groups available."
    )


# =============================================================================
# 19. CONFLICT GROUP SIZE ANALYSIS
# =============================================================================

print_section("CONFLICT GROUP SIZE ANALYSIS")

if len(conflict_only_df) > 0:

    size_distribution = (
        conflict_only_df[
            "total_rows"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nConflict groups by number of observations:"
    )

    print(
        size_distribution.to_string()
    )

    largest_group_size = int(
        conflict_only_df[
            "total_rows"
        ].max()
    )

    print(
        f"\nLargest conflict group size: "
        f"{largest_group_size}"
    )


# =============================================================================
# 20. LABEL FLIP ANALYSIS
# =============================================================================

print_section("LABEL FLIP ANALYSIS")

if len(conflict_only_df) > 0:

    label_flip_groups = conflict_only_df[
        [
            "group_id",
            "total_rows",
            "defective_count",
            "non_defective_count",
            "disagreement_rate_percent"
        ]
    ].copy()

    label_flip_groups = label_flip_groups.sort_values(
        by="disagreement_rate_percent",
        ascending=False
    )

    print(
        "\nGroups with the strongest label disagreement:"
    )

    print(
        label_flip_groups.head(10).to_string(
            index=False
        )
    )

else:

    print(
        "\nNo label-flip groups found."
    )


# =============================================================================
# 21. COMPARE WITH PROVIDED SUMMARY FILE
# =============================================================================

print_section("COMPARING WITH ORIGINAL SUMMARY")

summary_required = [
    "total_rows",
    "unique_labels",
    "defective_count",
    "non_defective_count"
]


available_summary_columns = [
    col
    for col in summary_required
    if col in summary_df.columns
]


print(
    "\nSummary columns available for comparison:"
)

for col in available_summary_columns:
    print(f"  - {col}")


if (
    "defective_count" in summary_df.columns
    and
    "non_defective_count" in summary_df.columns
):

    summary_conflict_mask = (
        (
            summary_df["defective_count"] > 0
        )
        &
        (
            summary_df["non_defective_count"] > 0
        )
    )

    summary_conflict_count = int(
        summary_conflict_mask.sum()
    )

    print(
        f"\nConflicting groups according to summary file: "
        f"{summary_conflict_count}"
    )


# =============================================================================
# 22. GENERATE PUBLICATION-READY REPORT
# =============================================================================

print_section("GENERATING ANALYSIS REPORT")


report_lines = []


report_lines.append(
    "=" * 80
)

report_lines.append(
    "SOFTWARE DEFECT DATASET - CONFLICT ANALYSIS REPORT"
)

report_lines.append(
    "=" * 80
)

report_lines.append("")


# Basic information
report_lines.append(
    "1. DATASET / ANALYSIS SIZE"
)

report_lines.append(
    f"Number of conflict groups: {number_of_conflict_groups}"
)

report_lines.append(
    f"Number of conflicting rows: {number_of_conflicting_rows}"
)

report_lines.append("")


# Labels
report_lines.append(
    "2. DEFECT LABEL DISTRIBUTION"
)

defective_total = int(
    (rows_df["defects"] == True).sum()
)

non_defective_total = int(
    (rows_df["defects"] == False).sum()
)

report_lines.append(
    f"Defective rows: {defective_total}"
)

report_lines.append(
    f"Non-defective rows: {non_defective_total}"
)

if len(rows_df) > 0:

    report_lines.append(
        f"Defective percentage: "
        f"{defective_total / len(rows_df) * 100:.2f}%"
    )

    report_lines.append(
        f"Non-defective percentage: "
        f"{non_defective_total / len(rows_df) * 100:.2f}%"
    )

report_lines.append("")


# Duplicate analysis
report_lines.append(
    "3. DUPLICATE ANALYSIS"
)

report_lines.append(
    f"Exact duplicate rows: {exact_duplicate_rows}"
)

report_lines.append(
    f"Feature duplicate rows: {feature_duplicate_rows}"
)

report_lines.append("")


# Conflict analysis
report_lines.append(
    "4. FEATURE-LEVEL CONFLICT ANALYSIS"
)

report_lines.append(
    f"Unique conflicting feature vectors: "
    f"{number_of_conflict_groups}"
)

report_lines.append(
    f"Rows belonging to conflict groups: "
    f"{number_of_conflicting_rows}"
)

report_lines.append("")


# Ambiguity
report_lines.append(
    "5. LABEL AMBIGUITY"
)

report_lines.append(
    f"Number of conflict groups: "
    f"{len(conflict_only_df)}"
)

report_lines.append(
    f"Overall ambiguity rate: "
    f"{overall_ambiguity_rate:.2f}%"
)

if len(conflict_only_df) > 0:

    report_lines.append(
        f"Perfect 50/50 conflict groups: "
        f"{perfect_tie_count}"
    )

    report_lines.append(
        f"High-ambiguity groups: "
        f"{high_ambiguity_count}"
    )

    report_lines.append(
        f"Moderate-ambiguity groups: "
        f"{moderate_ambiguity_count}"
    )

    report_lines.append(
        f"Low-ambiguity groups: "
        f"{low_ambiguity_count}"
    )

report_lines.append("")


# Interpretation
report_lines.append(
    "6. INTERPRETATION"
)

report_lines.append(
    "The conflict analysis identifies feature vectors that occur with "
    "both defective and non-defective labels."
)

report_lines.append(
    "Such observations represent label ambiguity at the feature level "
    "and are relevant when evaluating model behavior and explanation "
    "consistency."
)

report_lines.append(
    "These conflict groups should be considered when interpreting "
    "prediction performance and SHAP explanation stability."
)

report_lines.append("")


# Generated files
report_lines.append(
    "7. GENERATED OUTPUT FILES"
)

report_lines.append(
    f"Detailed conflict analysis: {DETAILED_GROUP_FILE}"
)

report_lines.append(
    f"Label distribution: {LABEL_DISTRIBUTION_FILE}"
)

report_lines.append("")


report_lines.append(
    "=" * 80
)


with open(
    ANALYSIS_REPORT,
    "w",
    encoding="utf-8"
) as report_file:

    report_file.write(
        "\n".join(report_lines)
    )


# =============================================================================
# 23. FINAL SUMMARY
# =============================================================================

print_section("ANALYSIS COMPLETED")

print(
    "\nInput files successfully analyzed."
)

print(
    f"\nConflict groups: {number_of_conflict_groups}"
)

print(
    f"Conflicting rows: {number_of_conflicting_rows}"
)

print(
    f"Exact duplicate rows: {exact_duplicate_rows}"
)

print(
    f"Feature duplicate rows: {feature_duplicate_rows}"
)

print(
    f"Defective rows: {defective_total}"
)

print(
    f"Non-defective rows: {non_defective_total}"
)

print(
    f"Overall conflict ambiguity rate: "
    f"{overall_ambiguity_rate:.2f}%"
)


print_section("OUTPUT FILES CREATED")

print(
    "\n1. Detailed conflict-group analysis:"
)

print(
    DETAILED_GROUP_FILE
)

print(
    "\n2. Label distribution:"
)

print(
    LABEL_DISTRIBUTION_FILE
)

print(
    "\n3. Publication-oriented analysis report:"
)

print(
    ANALYSIS_REPORT
)


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
