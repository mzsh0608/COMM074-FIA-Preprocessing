"""Merge extracted FIA TREE, PLOT, and COND tables for Notebook 02."""

from pathlib import Path

import pandas as pd

from src.paths import INTERIM_DIR, OUTPUTS_DIR, PARQUET_DIR, create_project_dirs


TREE_REQUIRED_COLUMNS = [
    "CN",
    "PLT_CN",
    "CONDID",
    "TREE",
    "STATUSCD",
    "SPCD",
    "DIA",
    "HT",
    "ACTUALHT",
    "CR",
    "CARBON_AG",
]

PLOT_REQUIRED_COLUMNS = [
    "CN",
    "STATECD",
    "COUNTYCD",
    "INVYR",
]

COND_REQUIRED_COLUMNS = [
    "PLT_CN",
    "CONDID",
    "FORTYPCD",
    "OWNGRPCD",
    "SITECLCD",
    "STDAGE",
    "STDSZCD",
]


def load_extracted_tables():
    """Load the extracted TREE, PLOT, and COND Parquet files."""
    tree = pd.read_parquet(PARQUET_DIR / "tree_selected.parquet")
    plot = pd.read_parquet(PARQUET_DIR / "plot_selected.parquet")
    cond = pd.read_parquet(PARQUET_DIR / "cond_selected.parquet")

    return tree, plot, cond


def validate_required_columns(df, required_cols, table_name):
    """Print and return required columns that are missing from a dataframe."""
    missing_cols = [column for column in required_cols if column not in df.columns]

    if missing_cols:
        print(f"{table_name} missing required columns: {', '.join(missing_cols)}")
    else:
        print(f"{table_name} has all required columns.")

    return missing_cols


def _safe_missing_rate(df, column_name):
    """Return the missing rate for a column, or None if the column is absent."""
    if column_name not in df.columns or len(df) == 0:
        return None

    return df[column_name].isna().mean()


def _make_column_summary(df):
    """Create a simple column-level summary for immediate validation."""
    rows = []
    for column in df.columns:
        rows.append(
            {
                "column_name": column,
                "dtype": str(df[column].dtype),
                "missing_count": int(df[column].isna().sum()),
                "missing_rate": df[column].isna().mean(),
                "n_unique": df[column].nunique(dropna=True),
            }
        )

    return pd.DataFrame(rows)


def merge_tree_plot_cond():
    """Merge TREE to PLOT and COND, then save validation outputs."""
    create_project_dirs()
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    tree, plot, cond = load_extracted_tables()

    # Validate the extracted tables before doing any joins.
    tree_missing = validate_required_columns(tree, TREE_REQUIRED_COLUMNS, "TREE")
    plot_missing = validate_required_columns(plot, PLOT_REQUIRED_COLUMNS, "PLOT")
    cond_missing = validate_required_columns(cond, COND_REQUIRED_COLUMNS, "COND")

    if tree_missing or plot_missing or cond_missing:
        raise ValueError("Required columns are missing. Re-run extraction first.")

    original_tree_rows = len(tree)
    plot_rows = len(plot)
    cond_rows = len(cond)

    print(f"TREE rows before merge: {original_tree_rows:,}")
    print(f"PLOT rows before merge: {plot_rows:,}")
    print(f"COND rows before merge: {cond_rows:,}")

    # Rename PLOT CN so it does not conflict with TREE CN.
    plot_for_merge = plot.rename(columns={"CN": "PLOT_CN_KEY"})

    merged = tree.merge(
        plot_for_merge,
        how="left",
        left_on="PLT_CN",
        right_on="PLOT_CN_KEY",
        suffixes=("", "_PLOT"),
    )
    rows_after_plot_merge = len(merged)
    print(f"Rows after TREE-PLOT merge: {rows_after_plot_merge:,}")

    merged["has_plot_match"] = merged["PLOT_CN_KEY"].notna()

    merged = merged.merge(
        cond,
        how="left",
        on=["PLT_CN", "CONDID"],
        suffixes=("", "_COND"),
    )
    rows_after_cond_merge = len(merged)
    print(f"Rows after COND merge: {rows_after_cond_merge:,}")

    # FORTYPCD comes from COND, so it is a useful direct match check.
    merged["has_cond_match"] = merged["FORTYPCD"].notna()

    if rows_after_cond_merge != original_tree_rows:
        print(
            "Warning: final row count differs from original TREE row count "
            f"({rows_after_cond_merge:,} vs {original_tree_rows:,})."
        )

    plot_missing_match_rate = 1 - merged["has_plot_match"].mean()
    cond_missing_match_rate = 1 - merged["has_cond_match"].mean()

    summary_rows = [
        {"metric": "tree_rows_before_merge", "value": original_tree_rows},
        {"metric": "plot_rows_before_merge", "value": plot_rows},
        {"metric": "cond_rows_before_merge", "value": cond_rows},
        {"metric": "rows_after_tree_plot_merge", "value": rows_after_plot_merge},
        {"metric": "rows_after_cond_merge", "value": rows_after_cond_merge},
        {
            "metric": "final_row_count_matches_tree",
            "value": rows_after_cond_merge == original_tree_rows,
        },
        {"metric": "missing_plot_match_rate", "value": plot_missing_match_rate},
        {"metric": "missing_cond_match_rate", "value": cond_missing_match_rate},
        {
            "metric": "plot_statecd_missing_rate",
            "value": _safe_missing_rate(merged, "STATECD"),
        },
        {
            "metric": "cond_fortypcd_missing_rate",
            "value": _safe_missing_rate(merged, "FORTYPCD"),
        },
        {
            "metric": "cond_owngrpcd_missing_rate",
            "value": _safe_missing_rate(merged, "OWNGRPCD"),
        },
    ]
    merge_summary = pd.DataFrame(summary_rows)

    merged_output_path = INTERIM_DIR / "merged_tree_plot_cond.parquet"
    summary_output_path = OUTPUTS_DIR / "merge_summary.csv"
    column_summary_output_path = OUTPUTS_DIR / "merged_column_summary.csv"

    merged.to_parquet(merged_output_path, index=False)
    merge_summary.to_csv(summary_output_path, index=False)
    _make_column_summary(merged).to_csv(column_summary_output_path, index=False)

    print(f"Merged dataset saved to {merged_output_path}")
    print(f"Merge summary saved to {summary_output_path}")
    print(f"Merged column summary saved to {column_summary_output_path}")

    return merged, merge_summary


if __name__ == "__main__":
    merge_tree_plot_cond()
