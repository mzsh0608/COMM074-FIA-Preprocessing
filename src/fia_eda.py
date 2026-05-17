"""Exploratory data analysis helpers for the cleaned FIA tree dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.paths import FIGURES_DIR, INTERIM_DIR, OUTPUTS_DIR, create_project_dirs


CLEANED_DATA_PATH = INTERIM_DIR / "fia_tree_carbon_clean_unencoded.parquet"
TARGET_COLUMN = "CARBON_AG"


def load_cleaned_dataset():
    """Load the cleaned live-tree FIA dataset."""
    return pd.read_parquet(CLEANED_DATA_PATH)


def create_eda_sample(df, max_rows=100000, random_state=42):
    """Return a random sample when the dataframe is larger than max_rows."""
    if len(df) <= max_rows:
        return df.copy()

    return df.sample(n=max_rows, random_state=random_state).copy()


def save_dataset_summary(df, output_path):
    """Save a one-row dataset summary for Notebook 04."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        [
            {
                "row_count": len(df),
                "column_count": len(df.columns),
                "target_column": TARGET_COLUMN,
                "number_of_states": _nunique_if_exists(df, "STATECD"),
                "number_of_species": _nunique_if_exists(df, "SPCD"),
                "min_year": _min_if_exists(df, "INVYR"),
                "max_year": _max_if_exists(df, "INVYR"),
            }
        ]
    )
    summary.to_csv(output_path, index=False)

    return summary


def _nunique_if_exists(df, column):
    """Return unique count for a column, or None if it is missing."""
    if column not in df.columns:
        return None

    return df[column].nunique(dropna=True)


def _min_if_exists(df, column):
    """Return minimum for a column, or None if it is missing."""
    if column not in df.columns:
        return None

    return df[column].min()


def _max_if_exists(df, column):
    """Return maximum for a column, or None if it is missing."""
    if column not in df.columns:
        return None

    return df[column].max()


def _save_current_figure(file_name):
    """Save the current matplotlib figure in the project figures folder."""
    create_project_dirs()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / file_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def _require_columns(df, columns, plot_name):
    """Check that required columns are present before making a plot."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        print(f"Skipping {plot_name}; missing columns: {', '.join(missing)}")
        return False

    return True


def plot_missing_values(df):
    """Plot the variables with the highest missing percentages."""
    missing = (
        df.isna().mean().mul(100).sort_values(ascending=False).head(25).reset_index()
    )
    missing.columns = ["variable", "missing_percentage"]

    plt.figure(figsize=(10, 7))
    sns.barplot(data=missing, x="missing_percentage", y="variable", color="#4C78A8")
    plt.title("Top Missing Value Percentages")
    plt.xlabel("Missing values (%)")
    plt.ylabel("Variable")

    return _save_current_figure("missing_values.png")


def plot_carbon_distribution(df):
    """Plot the distribution of aboveground tree carbon."""
    if not _require_columns(df, [TARGET_COLUMN], "carbon distribution"):
        return None

    plt.figure(figsize=(9, 6))
    sns.histplot(df[TARGET_COLUMN].dropna(), bins=60, color="#4C78A8")
    plt.title("Distribution of Aboveground Tree Carbon")
    plt.xlabel("Aboveground carbon")
    plt.ylabel("Tree count")

    return _save_current_figure("carbon_distribution.png")


def plot_log_carbon_distribution(df):
    """Plot log-transformed aboveground carbon using log1p."""
    if not _require_columns(df, [TARGET_COLUMN], "log carbon distribution"):
        return None

    log_carbon = np.log1p(df[TARGET_COLUMN].dropna())

    plt.figure(figsize=(9, 6))
    sns.histplot(log_carbon, bins=60, color="#72B7B2")
    plt.title("Distribution of Log Aboveground Tree Carbon")
    plt.xlabel("log1p(aboveground carbon)")
    plt.ylabel("Tree count")

    return _save_current_figure("log_carbon_distribution.png")


def plot_dia_vs_carbon(df_sample):
    """Plot tree diameter against aboveground carbon."""
    if not _require_columns(df_sample, ["DIA", TARGET_COLUMN], "DIA vs carbon"):
        return None

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_sample, x="DIA", y=TARGET_COLUMN, alpha=0.25, s=12)
    plt.title("Tree Diameter vs Aboveground Carbon")
    plt.xlabel("Diameter at breast height")
    plt.ylabel("Aboveground carbon")

    return _save_current_figure("dia_vs_carbon.png")


def plot_ht_vs_carbon(df_sample):
    """Plot tree height against aboveground carbon."""
    if not _require_columns(df_sample, ["HT", TARGET_COLUMN], "HT vs carbon"):
        return None

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_sample, x="HT", y=TARGET_COLUMN, alpha=0.25, s=12)
    plt.title("Tree Height vs Aboveground Carbon")
    plt.xlabel("Tree height")
    plt.ylabel("Aboveground carbon")

    return _save_current_figure("ht_vs_carbon.png")


def plot_cr_vs_carbon(df_sample):
    """Plot crown ratio against aboveground carbon."""
    if not _require_columns(df_sample, ["CR", TARGET_COLUMN], "CR vs carbon"):
        return None

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_sample, x="CR", y=TARGET_COLUMN, alpha=0.25, s=12)
    plt.title("Crown Ratio vs Aboveground Carbon")
    plt.xlabel("Crown ratio")
    plt.ylabel("Aboveground carbon")

    return _save_current_figure("cr_vs_carbon.png")


def plot_carbon_by_state(df):
    """Plot median aboveground carbon by state."""
    if not _require_columns(df, ["STATECD", TARGET_COLUMN], "carbon by state"):
        return None

    summary = (
        df.groupby("STATECD", dropna=False)[TARGET_COLUMN]
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(11, 6))
    sns.barplot(data=summary, x="STATECD", y=TARGET_COLUMN, color="#59A14F")
    plt.title("Median Aboveground Carbon by State")
    plt.xlabel("State code")
    plt.ylabel("Median aboveground carbon")
    plt.xticks(rotation=90)

    return _save_current_figure("carbon_by_state.png")


def plot_carbon_by_top_species(df, top_n=15):
    """Plot median aboveground carbon for the most common species."""
    if not _require_columns(df, ["SPCD", TARGET_COLUMN], "carbon by species"):
        return None

    top_species = df["SPCD"].value_counts().head(top_n).index
    summary = (
        df[df["SPCD"].isin(top_species)]
        .groupby("SPCD")[TARGET_COLUMN]
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary, x="SPCD", y=TARGET_COLUMN, color="#F28E2B")
    plt.title(f"Median Aboveground Carbon by Top {top_n} Species")
    plt.xlabel("Species code")
    plt.ylabel("Median aboveground carbon")
    plt.xticks(rotation=45)

    return _save_current_figure("carbon_by_top_species.png")


def plot_carbon_by_forest_type(df, top_n=15):
    """Plot median aboveground carbon for the most common forest types."""
    if not _require_columns(df, ["FORTYPCD", TARGET_COLUMN], "carbon by forest type"):
        return None

    top_types = df["FORTYPCD"].value_counts().head(top_n).index
    summary = (
        df[df["FORTYPCD"].isin(top_types)]
        .groupby("FORTYPCD")[TARGET_COLUMN]
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary, x="FORTYPCD", y=TARGET_COLUMN, color="#E15759")
    plt.title(f"Median Aboveground Carbon by Top {top_n} Forest Types")
    plt.xlabel("Forest type code")
    plt.ylabel("Median aboveground carbon")
    plt.xticks(rotation=45)

    return _save_current_figure("carbon_by_forest_type.png")


def plot_carbon_by_diameter_class(df):
    """Plot median carbon by diameter class."""
    if not _require_columns(df, ["DIA", TARGET_COLUMN], "carbon by diameter class"):
        return None

    plot_df = df[["DIA", TARGET_COLUMN]].dropna().copy()
    bins = [0, 5, 10, 15, 20, 30, 40, 60, np.inf]
    labels = ["0-5", "5-10", "10-15", "15-20", "20-30", "30-40", "40-60", "60+"]
    plot_df["diameter_class"] = pd.cut(
        plot_df["DIA"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    summary = plot_df.groupby("diameter_class", observed=True)[TARGET_COLUMN].median()
    summary = summary.reset_index()

    plt.figure(figsize=(9, 6))
    sns.barplot(data=summary, x="diameter_class", y=TARGET_COLUMN, color="#B07AA1")
    plt.title("Median Aboveground Carbon by Diameter Class")
    plt.xlabel("Diameter class")
    plt.ylabel("Median aboveground carbon")

    return _save_current_figure("carbon_by_diameter_class.png")


def plot_correlation_heatmap(df):
    """Plot correlations between key numeric variables."""
    numeric_columns = [
        TARGET_COLUMN,
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
    available_columns = [column for column in numeric_columns if column in df.columns]

    if len(available_columns) < 2:
        print("Skipping correlation heatmap; fewer than two numeric columns found.")
        return None

    corr = df[available_columns].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
    )
    plt.title("Correlation Heatmap for Key Numeric Variables")

    return _save_current_figure("correlation_heatmap.png")


def create_numeric_summary(df, output_path):
    """Save summary statistics for numeric columns."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_df = df.select_dtypes(include="number")
    summary = numeric_df.describe().transpose().reset_index()
    summary = summary.rename(columns={"index": "variable"})
    summary["missing_count"] = numeric_df.isna().sum().values
    summary["missing_percentage"] = numeric_df.isna().mean().values * 100
    summary.to_csv(output_path, index=False)

    return summary


def create_categorical_summary(df, output_path):
    """Save basic summaries for non-numeric columns."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categorical_df = df.select_dtypes(exclude="number")
    rows = []
    for column in categorical_df.columns:
        value_counts = categorical_df[column].value_counts(dropna=True)
        rows.append(
            {
                "variable": column,
                "count": int(categorical_df[column].notna().sum()),
                "missing_count": int(categorical_df[column].isna().sum()),
                "missing_percentage": categorical_df[column].isna().mean() * 100,
                "n_unique": categorical_df[column].nunique(dropna=True),
                "top_value": None if value_counts.empty else value_counts.index[0],
                "top_value_frequency": None
                if value_counts.empty
                else int(value_counts.iloc[0]),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)

    return summary


def create_state_summary(df, output_path):
    """Save tree counts and carbon summary statistics by state."""
    return _create_group_summary(df, "STATECD", output_path)


def create_species_summary(df, output_path):
    """Save tree counts and carbon summary statistics by species."""
    return _create_group_summary(df, "SPCD", output_path)


def create_forest_type_summary(df, output_path):
    """Save tree counts and carbon summary statistics by forest type."""
    return _create_group_summary(df, "FORTYPCD", output_path)


def _create_group_summary(df, group_col, output_path):
    """Create a count and target summary for one grouping variable."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not _require_columns(df, [group_col, TARGET_COLUMN], f"{group_col} summary"):
        summary = pd.DataFrame()
        summary.to_csv(output_path, index=False)
        return summary

    summary = (
        df.groupby(group_col, dropna=False)[TARGET_COLUMN]
        .agg(
            tree_count="count",
            mean_carbon="mean",
            median_carbon="median",
            min_carbon="min",
            max_carbon="max",
        )
        .reset_index()
        .sort_values("tree_count", ascending=False)
    )
    summary.to_csv(output_path, index=False)

    return summary


if __name__ == "__main__":
    create_project_dirs()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    fia_df = load_cleaned_dataset()
    fia_sample = create_eda_sample(fia_df)

    save_dataset_summary(fia_df, OUTPUTS_DIR / "eda_dataset_summary.csv")
    create_numeric_summary(fia_df, OUTPUTS_DIR / "eda_numeric_summary.csv")
    create_categorical_summary(fia_df, OUTPUTS_DIR / "eda_categorical_summary.csv")
    create_state_summary(fia_df, OUTPUTS_DIR / "eda_state_summary.csv")
    create_species_summary(fia_df, OUTPUTS_DIR / "eda_species_summary.csv")
    create_forest_type_summary(fia_df, OUTPUTS_DIR / "eda_forest_type_summary.csv")

    plot_missing_values(fia_df)
    plot_carbon_distribution(fia_df)
    plot_log_carbon_distribution(fia_df)
    plot_dia_vs_carbon(fia_sample)
    plot_ht_vs_carbon(fia_sample)
    plot_cr_vs_carbon(fia_sample)
    plot_carbon_by_state(fia_df)
    plot_carbon_by_top_species(fia_df)
    plot_carbon_by_forest_type(fia_df)
    plot_carbon_by_diameter_class(fia_df)
    plot_correlation_heatmap(fia_df)
