"""Clean the merged FIA tree-level dataset and summarise missing values."""

from pathlib import Path

import pandas as pd

from src.paths import INTERIM_DIR, OUTPUTS_DIR, create_project_dirs


MERGED_DATA_PATH = INTERIM_DIR / "merged_tree_plot_cond.parquet"
CLEANED_DATA_PATH = INTERIM_DIR / "fia_tree_carbon_clean_unencoded.parquet"

KEY_VARIABLES = [
    "CARBON_AG",
    "DIA",
    "HT",
    "ACTUALHT",
    "CR",
    "SPCD",
    "STATECD",
    "FORTYPCD",
    "OWNGRPCD",
    "SITECLCD",
    "STDAGE",
    "STDSZCD",
]

NUMERIC_COLUMNS_TO_CONVERT = [
    "CARBON_AG",
    "CARBON_BG",
    "DRYBIO_AG",
    "DRYBIO_BG",
    "DIA",
    "HT",
    "ACTUALHT",
    "CR",
    "STDAGE",
    "BALIVE",
    "ALSTK",
    "GSSTK",
    "TPA_UNADJ",
]


def load_merged_dataset():
    """Load the merged TREE, PLOT, and COND dataset."""
    return pd.read_parquet(MERGED_DATA_PATH)


def convert_key_columns_to_numeric(df):
    """Convert key FIA measurement columns to numeric values when present."""
    converted_df = df.copy()

    for column in NUMERIC_COLUMNS_TO_CONVERT:
        if column in converted_df.columns:
            converted_df[column] = pd.to_numeric(
                converted_df[column],
                errors="coerce",
            )

    return converted_df


def summarise_missing_values(df, output_path=None):
    """Summarise missing values for every column in a dataframe."""
    missing_summary = pd.DataFrame(
        {
            "variable": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_percentage": (df.isna().mean().values * 100),
            "dtype": [str(dtype) for dtype in df.dtypes],
        }
    )
    missing_summary = missing_summary.sort_values(
        "missing_percentage",
        ascending=False,
    ).reset_index(drop=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        missing_summary.to_csv(output_path, index=False)

    return missing_summary


def _record_cleaning_step(summary_rows, step, rows_before, rows_after, reason):
    """Add one row to the cleaning summary."""
    rows_removed = rows_before - rows_after
    percentage_removed = 0 if rows_before == 0 else (rows_removed / rows_before) * 100

    summary_rows.append(
        {
            "step": step,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": rows_removed,
            "percentage_removed": percentage_removed,
            "reason": reason,
        }
    )


def _filter_rows(df, summary_rows, step, mask, reason):
    """Apply a row filter and record how many rows were removed."""
    rows_before = len(df)
    filtered_df = df.loc[mask].copy()
    rows_after = len(filtered_df)
    _record_cleaning_step(summary_rows, step, rows_before, rows_after, reason)

    return filtered_df


def clean_fia_tree_data(df, target_col="CARBON_AG", keep_live_only=True):
    """Apply simple cleaning rules for tree-level carbon analysis."""
    cleaned_df = convert_key_columns_to_numeric(df)
    summary_rows = []

    _record_cleaning_step(
        summary_rows,
        "Start",
        len(cleaned_df),
        len(cleaned_df),
        "Start with merged dataset.",
    )

    if target_col not in cleaned_df.columns:
        raise ValueError(f"Target column is missing: {target_col}")

    cleaned_df = _filter_rows(
        cleaned_df,
        summary_rows,
        "Remove missing target",
        cleaned_df[target_col].notna(),
        f"Remove rows where {target_col} is missing.",
    )

    cleaned_df = _filter_rows(
        cleaned_df,
        summary_rows,
        "Remove negative target",
        cleaned_df[target_col] >= 0,
        f"Remove rows where {target_col} is negative.",
    )

    if "DIA" not in cleaned_df.columns:
        raise ValueError("Required column is missing: DIA")

    cleaned_df = _filter_rows(
        cleaned_df,
        summary_rows,
        "Remove missing DIA",
        cleaned_df["DIA"].notna(),
        "Remove rows where tree diameter is missing.",
    )

    cleaned_df = _filter_rows(
        cleaned_df,
        summary_rows,
        "Remove DIA <= 0",
        cleaned_df["DIA"] > 0,
        "Remove rows where tree diameter is zero or negative.",
    )

    if "SPCD" not in cleaned_df.columns:
        raise ValueError("Required column is missing: SPCD")

    cleaned_df = _filter_rows(
        cleaned_df,
        summary_rows,
        "Remove missing SPCD",
        cleaned_df["SPCD"].notna(),
        "Remove rows where species code is missing.",
    )

    if keep_live_only and "STATUSCD" in cleaned_df.columns:
        cleaned_df = _filter_rows(
            cleaned_df,
            summary_rows,
            "Keep live trees only",
            cleaned_df["STATUSCD"] == 1,
            "Keep only rows where STATUSCD equals 1.",
        )
    elif keep_live_only:
        print("STATUSCD is missing, so live-tree filtering was skipped.")

    rows_before = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates().copy()
    rows_after = len(cleaned_df)
    _record_cleaning_step(
        summary_rows,
        "Remove exact duplicate rows",
        rows_before,
        rows_after,
        "Remove rows that are exact duplicates across all columns.",
    )

    cleaning_summary = pd.DataFrame(summary_rows)

    return cleaned_df, cleaning_summary


def summarise_key_variables(df, target_col="CARBON_AG"):
    """Summarise important variables without changing the data."""
    variables = [target_col]
    variables.extend(column for column in KEY_VARIABLES if column != target_col)

    rows = []
    for variable in variables:
        if variable not in df.columns:
            continue

        series = df[variable]
        missing_percentage = series.isna().mean() * 100

        if pd.api.types.is_numeric_dtype(series):
            rows.append(
                {
                    "variable": variable,
                    "summary_type": "numeric",
                    "count": int(series.notna().sum()),
                    "missing_percentage": missing_percentage,
                    "mean": series.mean(),
                    "median": series.median(),
                    "std": series.std(),
                    "min": series.min(),
                    "max": series.max(),
                    "n_unique": None,
                    "top_value": None,
                    "top_value_frequency": None,
                }
            )
        else:
            value_counts = series.value_counts(dropna=True)
            top_value = None if value_counts.empty else value_counts.index[0]
            top_frequency = None if value_counts.empty else int(value_counts.iloc[0])

            rows.append(
                {
                    "variable": variable,
                    "summary_type": "categorical",
                    "count": int(series.notna().sum()),
                    "missing_percentage": missing_percentage,
                    "mean": None,
                    "median": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "n_unique": series.nunique(dropna=True),
                    "top_value": top_value,
                    "top_value_frequency": top_frequency,
                }
            )

    return pd.DataFrame(rows)


def save_cleaning_outputs(cleaned_df, cleaning_summary, missing_summary, key_summary):
    """Save the cleaned dataset and all cleaning validation outputs."""
    create_project_dirs()
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_parquet(CLEANED_DATA_PATH, index=False)
    cleaning_summary.to_csv(OUTPUTS_DIR / "cleaning_summary.csv", index=False)
    missing_summary.to_csv(OUTPUTS_DIR / "missing_value_summary.csv", index=False)
    key_summary.to_csv(OUTPUTS_DIR / "key_variable_summary.csv", index=False)

    print(f"Cleaned dataset saved to {CLEANED_DATA_PATH}")
    print(f"Cleaning summary saved to {OUTPUTS_DIR / 'cleaning_summary.csv'}")
    print(f"Missing value summary saved to {OUTPUTS_DIR / 'missing_value_summary.csv'}")
    print(f"Key variable summary saved to {OUTPUTS_DIR / 'key_variable_summary.csv'}")


if __name__ == "__main__":
    merged_df = load_merged_dataset()
    missing_values = summarise_missing_values(merged_df)
    cleaned, summary = clean_fia_tree_data(merged_df)
    key_variables = summarise_key_variables(cleaned)
    save_cleaning_outputs(cleaned, summary, missing_values, key_variables)
