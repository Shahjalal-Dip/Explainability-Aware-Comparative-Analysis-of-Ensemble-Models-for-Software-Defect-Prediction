from pathlib import Path
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "cleaning"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD JM1
# ============================================================

csv_files = [
    file for file in RAW_DIR.glob("*.csv")
    if "jm1" in file.stem.lower()
]

if len(csv_files) != 1:
    raise FileNotFoundError(
        "Exactly one JM1 CSV file should exist in data/raw/"
    )

dataset_path = csv_files[0]

df = pd.read_csv(dataset_path)

target = "defects"

feature_columns = [
    column
    for column in df.columns
    if column != target
]


# ============================================================
# 3. IDENTIFY CONFLICTING FEATURE GROUPS
# ============================================================

grouped = (
    df.groupby(feature_columns, dropna=False)
    [target]
    .agg(
        total_rows="size",
        unique_labels="nunique",
        defective_count=lambda x: int(x.sum()),
        non_defective_count=lambda x: int((~x).sum())
    )
    .reset_index()
)


conflicts = grouped[
    grouped["unique_labels"] > 1
].copy()


# ============================================================
# 4. PRINT SUMMARY
# ============================================================

print("=" * 70)
print("JM1 DUPLICATE / LABEL-CONFLICT INVESTIGATION")
print("=" * 70)

print(f"\nOriginal number of rows: {len(df)}")

print(
    f"Number of conflicting feature groups: "
    f"{len(conflicts)}"
)


# Number of observations involved
conflict_feature_rows = (
    df.merge(
        conflicts[feature_columns],
        on=feature_columns,
        how="inner"
    )
)

print(
    f"Rows involved in conflicting groups: "
    f"{len(conflict_feature_rows)}"
)


# ============================================================
# 5. DISPLAY CONFLICT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CONFLICT SUMMARY")
print("=" * 70)

print(
    conflicts[
        [
            "total_rows",
            "defective_count",
            "non_defective_count"
        ]
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# 6. SAVE CONFLICT SUMMARY
# ============================================================

summary_path = (
    RESULTS_DIR /
    "conflicting_feature_groups_summary.csv"
)

conflicts.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 7. EXTRACT ALL CONFLICTING ROWS
# ============================================================

conflicting_rows = (
    df.merge(
        conflicts[feature_columns],
        on=feature_columns,
        how="inner"
    )
)


conflicting_rows = conflicting_rows.sort_values(
    by=feature_columns
)


rows_path = (
    RESULTS_DIR /
    "conflicting_rows.csv"
)

conflicting_rows.to_csv(
    rows_path,
    index=False
)


# ============================================================
# 8. CHECK WHETHER CONFLICTS ARE EXACT DUPLICATES
# ============================================================

conflict_exact_duplicates = (
    conflicting_rows
    .duplicated(keep=False)
)


print("\n" + "=" * 70)
print("EXACT DUPLICATE CHECK")
print("=" * 70)

print(
    f"\nRows in conflicting groups that have "
    f"an exact duplicate: "
    f"{conflict_exact_duplicates.sum()}"
)


# ============================================================
# 9. SHOW FIRST 10 CONFLICTING GROUPS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 10 CONFLICTING FEATURE GROUPS")
print("=" * 70)


for index, (_, group) in enumerate(
    conflicting_rows.groupby(
        feature_columns,
        dropna=False
    ),
    start=1
):

    if index > 10:
        break

    print(f"\n--- Conflict group {index} ---")

    print(
        group[
            feature_columns + [target]
        ].to_string(index=False)
    )


# ============================================================
# 10. SAVE JSON-LIKE TEXT REPORT
# ============================================================

report_path = (
    RESULTS_DIR /
    "duplicate_conflict_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "JM1 DUPLICATE / LABEL-CONFLICT INVESTIGATION\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write(
        f"Original rows: {len(df)}\n"
    )

    file.write(
        f"Conflicting feature groups: "
        f"{len(conflicts)}\n"
    )

    file.write(
        f"Rows involved in conflicts: "
        f"{len(conflicting_rows)}\n\n"
    )

    file.write(
        conflicts.to_string(index=False)
    )


# ============================================================
# 11. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("INVESTIGATION COMPLETED")
print("=" * 70)

print("\nSaved files:")

print(
    f"- {summary_path.name}"
)

print(
    f"- {rows_path.name}"
)

print(
    f"- {report_path.name}"
)

print("\nThe original JM1 dataset was NOT modified.")
