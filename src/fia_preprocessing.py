"""Final preprocessing and dataset splitting for FIA carbon modelling."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.paths import INTERIM_DIR, OUTPUTS_DIR, PROCESSED_DIR, create_project_dirs


CLEANED_DATA_PATH = INTERIM_DIR / "fia_tree_carbon_clean_unencoded.parquet"
TARGET_COLUMNS = ["CARBON_AG", "CARBON_AG_log"]

ID_COLUMNS = ["CN", "PLT_CN", "CONDID", "TREE"]

NUMERIC_PREDICTORS = [
    "DIA",
    "HT",
    "ACTUALHT",
    "CR",
    "STDAGE",
    "BALIVE",
    "ALSTK",
    "GSSTK",
    "DIA_squared",
    "DIA_HT_interaction",
]

CATEGORICAL_PREDICTORS = [
    "SPCD",
    "STATECD",
    "COUNTYCD",
    "FORTYPCD",
    "OWNGRPCD",
    "SITECLCD",
    "STDSZCD",
    "diameter_class",
]

LEAKAGE_RISK_COLUMNS = [
    "DRYBIO_AG",
    "DRYBIO_BG",
    "CARBON_BG",
    "VOLCFNET",
    "VOLCSNET",
    "VOLBFNET",
    "TPA_UNADJ",
]

DIAMETER_BINS = [0, 5, 10, 15, 20, 30, 40, 60, np.inf]
DIAMETER_LABELS = ["0-5", "5-10", "10-15", "15-20", "20-30", "30-40", "40-60", "60+"]


def load_cleaned_dataset():
    """Load the cleaned, unencoded FIA tree-level dataset."""
    return pd.read_parquet(CLEANED_DATA_PATH)


def create_engineered_features(df):
    """Create simple numeric and categorical features without fitting anything."""
    engineered_df = df.copy()

    if "DIA" not in engineered_df.columns:
        raise ValueError("DIA is required to create engineered features.")
    if "CARBON_AG" not in engineered_df.columns:
        raise ValueError("CARBON_AG is required to create the log target.")

    engineered_df["DIA_squared"] = engineered_df["DIA"] ** 2

    if "HT" in engineered_df.columns:
        engineered_df["DIA_HT_interaction"] = engineered_df["DIA"] * engineered_df["HT"]

    engineered_df["CARBON_AG_log"] = np.log1p(engineered_df["CARBON_AG"])
    engineered_df["diameter_class"] = pd.cut(
        engineered_df["DIA"],
        bins=DIAMETER_BINS,
        labels=DIAMETER_LABELS,
        include_lowest=True,
    )

    print("Engineered features created.")

    return engineered_df


def select_final_model_columns(df):
    """Select modelling columns and remove leakage-risk variables."""
    leakage_columns = _find_leakage_columns(df)
    print(f"Leakage-risk columns excluded: {len(leakage_columns)}")

    target_columns = [column for column in TARGET_COLUMNS if column in df.columns]
    id_columns = [column for column in ID_COLUMNS if column in df.columns]
    numeric_features = [
        column
        for column in NUMERIC_PREDICTORS
        if column in df.columns and column not in target_columns
    ]
    categorical_features = [
        column
        for column in CATEGORICAL_PREDICTORS
        if column in df.columns and column not in target_columns
    ]

    selected_columns = target_columns + id_columns + numeric_features + categorical_features
    selected_columns = [
        column for column in selected_columns if column not in leakage_columns
    ]

    model_df = df[selected_columns].copy()

    print(f"Selected modelling dataset shape: {model_df.shape}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    return (
        model_df,
        numeric_features,
        categorical_features,
        id_columns,
        target_columns,
        leakage_columns,
    )


def _find_leakage_columns(df):
    """Find explicitly listed and pattern-based leakage-risk columns."""
    leakage_columns = []
    allowed_target_columns = set(TARGET_COLUMNS)

    for column in df.columns:
        if column in allowed_target_columns:
            continue

        column_upper = column.upper()
        is_explicit_leakage = column in LEAKAGE_RISK_COLUMNS
        is_drybio_column = column_upper.startswith("DRYBIO")
        is_volume_column = column_upper.startswith("VOL")
        is_non_target_carbon = column_upper.startswith("CARBON")

        if (
            is_explicit_leakage
            or is_drybio_column
            or is_volume_column
            or is_non_target_carbon
        ):
            leakage_columns.append(column)

    return sorted(set(leakage_columns))


def create_stratified_regression_split(df, target_col="CARBON_AG", random_state=42):
    """Create train, validation, and test splits using binned target stratification."""
    if target_col not in df.columns:
        raise ValueError(f"Target column is missing: {target_col}")

    split_df = df.copy()

    try:
        split_df["_target_bin"] = pd.qcut(
            np.log1p(split_df[target_col]),
            q=10,
            duplicates="drop",
        )
        _validate_stratification_bins(split_df["_target_bin"], "train/temp")

        train_df, temp_df = train_test_split(
            split_df,
            test_size=0.30,
            random_state=random_state,
            stratify=split_df["_target_bin"],
        )

        _validate_stratification_bins(temp_df["_target_bin"], "validation/test")
        validation_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=random_state,
            stratify=temp_df["_target_bin"],
        )

        print("Created stratified train/validation/test split.")
    except ValueError as error:
        print(f"Warning: stratified split failed ({error}). Using random split instead.")

        split_df = split_df.drop(columns=["_target_bin"], errors="ignore")
        train_df, temp_df = train_test_split(
            split_df,
            test_size=0.30,
            random_state=random_state,
        )
        validation_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=random_state,
        )

    train_df = train_df.drop(columns=["_target_bin"], errors="ignore").copy()
    validation_df = validation_df.drop(columns=["_target_bin"], errors="ignore").copy()
    test_df = test_df.drop(columns=["_target_bin"], errors="ignore").copy()

    print(f"Train rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"Test rows: {len(test_df):,}")

    split_summary = _create_split_summary(
        df,
        train_df,
        validation_df,
        test_df,
        target_col=target_col,
    )

    return train_df, validation_df, test_df, split_summary


def _validate_stratification_bins(target_bins, split_name):
    """Raise a clear error if bins are too sparse for stratified splitting."""
    bin_counts = target_bins.value_counts()

    if len(bin_counts) < 2:
        raise ValueError(f"not enough target bins for {split_name} split")
    if bin_counts.min() < 2:
        raise ValueError(f"at least one target bin has fewer than 2 rows for {split_name}")


def create_final_data_dictionary(
    model_df,
    numeric_features,
    categorical_features,
    id_columns,
    target_columns,
):
    """Create a small data dictionary for the modelling handoff dataset."""
    rows = []

    for column in model_df.columns:
        if column in target_columns:
            role = "target"
        elif column in id_columns:
            role = "identifier/context"
        elif column in numeric_features:
            role = "numeric_predictor"
        elif column in categorical_features:
            role = "categorical_predictor"
        else:
            role = "other"

        rows.append(
            {
                "variable_name": column,
                "role": role,
                "type": str(model_df[column].dtype),
                "description": _describe_variable(column),
            }
        )

    return pd.DataFrame(rows)


def _describe_variable(column):
    """Return a short plain-English description for key FIA variables."""
    descriptions = {
        "CARBON_AG": "Aboveground tree carbon target.",
        "CARBON_AG_log": "Log-transformed aboveground carbon target using log1p.",
        "CN": "Unique tree record identifier.",
        "PLT_CN": "Plot identifier used to link TREE records to PLOT and COND.",
        "CONDID": "Condition identifier within a plot.",
        "TREE": "Tree number within a plot.",
        "DIA": "Tree diameter at breast height.",
        "HT": "Tree height.",
        "ACTUALHT": "Actual measured tree height when available.",
        "CR": "Crown ratio.",
        "STDAGE": "Stand age.",
        "BALIVE": "Live basal area for the condition.",
        "ALSTK": "All-live stocking.",
        "GSSTK": "Growing-stock stocking.",
        "DIA_squared": "Squared tree diameter feature.",
        "DIA_HT_interaction": "Interaction between tree diameter and height.",
        "SPCD": "Species code.",
        "STATECD": "State code.",
        "COUNTYCD": "County code.",
        "FORTYPCD": "Forest type code.",
        "OWNGRPCD": "Ownership group code.",
        "SITECLCD": "Site productivity class code.",
        "STDSZCD": "Stand size class code.",
        "diameter_class": "Binned tree diameter class.",
    }

    return descriptions.get(column, "Selected FIA modelling variable.")


def save_preprocessing_outputs(
    model_df,
    train_df,
    validation_df,
    test_df,
    data_dictionary,
    numeric_features,
    categorical_features,
    id_columns,
    target_columns,
    dropped_leakage_columns,
):
    """Save final preprocessing datasets and handoff documentation."""
    create_project_dirs()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = PROCESSED_DIR / "fia_model_data.parquet"
    train_path = PROCESSED_DIR / "fia_train.parquet"
    validation_path = PROCESSED_DIR / "fia_validation.parquet"
    test_path = PROCESSED_DIR / "fia_test.parquet"

    model_df.to_parquet(model_path, index=False)
    train_df.to_parquet(train_path, index=False)
    validation_df.to_parquet(validation_path, index=False)
    test_df.to_parquet(test_path, index=False)

    feature_list = _create_feature_list(
        numeric_features,
        categorical_features,
        id_columns,
        target_columns,
    )
    split_summary = _create_split_summary(model_df, train_df, validation_df, test_df)
    preprocessing_summary = _create_preprocessing_summary(
        model_df,
        numeric_features,
        categorical_features,
        id_columns,
        target_columns,
        dropped_leakage_columns,
    )

    feature_list.to_csv(OUTPUTS_DIR / "final_feature_list.csv", index=False)
    data_dictionary.to_csv(OUTPUTS_DIR / "final_data_dictionary.csv", index=False)
    split_summary.to_csv(OUTPUTS_DIR / "final_split_summary.csv", index=False)
    preprocessing_summary.to_csv(
        OUTPUTS_DIR / "final_preprocessing_summary.csv",
        index=False,
    )
    _save_handoff_notes(
        model_path,
        train_path,
        validation_path,
        test_path,
        numeric_features,
        categorical_features,
        dropped_leakage_columns,
    )

    print(f"Full modelling dataset saved to {model_path}")
    print(f"Train split saved to {train_path}")
    print(f"Validation split saved to {validation_path}")
    print(f"Test split saved to {test_path}")


def _create_feature_list(
    numeric_features,
    categorical_features,
    id_columns,
    target_columns,
):
    """Create a tidy list of selected variables and their modelling roles."""
    rows = []

    for role, columns in [
        ("target", target_columns),
        ("identifier/context", id_columns),
        ("numeric_predictor", numeric_features),
        ("categorical_predictor", categorical_features),
    ]:
        for column in columns:
            rows.append({"variable_name": column, "role": role})

    return pd.DataFrame(rows)


def _create_split_summary(
    model_df,
    train_df,
    validation_df,
    test_df,
    target_col="CARBON_AG",
):
    """Summarise row counts and percentages for each split."""
    total_rows = len(model_df)
    rows = []

    for split_name, split_df in [
        ("full", model_df),
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        rows.append(
            {
                "split": split_name,
                "row_count": len(split_df),
                "percentage": 0 if total_rows == 0 else len(split_df) / total_rows * 100,
                "target_mean": split_df[target_col].mean(),
                "target_median": split_df[target_col].median(),
                "target_min": split_df[target_col].min(),
                "target_max": split_df[target_col].max(),
            }
        )

    return pd.DataFrame(rows)


def _create_preprocessing_summary(
    model_df,
    numeric_features,
    categorical_features,
    id_columns,
    target_columns,
    dropped_leakage_columns,
):
    """Create a one-row summary of the final preprocessing output."""
    summary = {
        "row_count": len(model_df),
        "column_count": len(model_df.columns),
        "target_columns": json.dumps(target_columns),
        "id_columns": json.dumps(id_columns),
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "dropped_leakage_column_count": len(dropped_leakage_columns),
        "dropped_leakage_columns": json.dumps(dropped_leakage_columns),
    }

    return pd.DataFrame([summary])


def _save_handoff_notes(
    model_path,
    train_path,
    validation_path,
    test_path,
    numeric_features,
    categorical_features,
    dropped_leakage_columns,
):
    """Write short modelling handoff notes for the next notebook."""
    notes_path = OUTPUTS_DIR / "modelling_handoff_notes.txt"
    feature_metadata = {
        "target": "CARBON_AG",
        "log_target": "CARBON_AG_log",
        "task_type": "supervised regression",
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }

    notes = f"""FIA modelling handoff notes

Target:
- CARBON_AG is the primary target for aboveground tree carbon prediction.
- CARBON_AG_log is included as a transformed target option created with log1p.

Task type:
- Supervised regression.

Split files:
- Full modelling data: {model_path}
- Train split: {train_path}
- Validation split: {validation_path}
- Test split: {test_path}

Leakage warning:
- Do not use biomass, carbon component, volume, or TPA-derived variables as predictors.
- These columns were excluded before saving the modelling files:
  {json.dumps(dropped_leakage_columns)}

Pipeline warning:
- One-hot encoding, scaling, and imputation should be fitted only inside later
  modelling pipelines after the train/validation/test split.

Feature metadata:
{json.dumps(feature_metadata, indent=2)}
"""

    notes_path.write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    cleaned_df = load_cleaned_dataset()
    engineered_df = create_engineered_features(cleaned_df)
    (
        final_model_df,
        final_numeric_features,
        final_categorical_features,
        final_id_columns,
        final_target_columns,
        final_dropped_leakage_columns,
    ) = select_final_model_columns(engineered_df)
    train, validation, test, split_summary = create_stratified_regression_split(
        final_model_df
    )
    final_data_dictionary = create_final_data_dictionary(
        final_model_df,
        final_numeric_features,
        final_categorical_features,
        final_id_columns,
        final_target_columns,
    )
    save_preprocessing_outputs(
        final_model_df,
        train,
        validation,
        test,
        final_data_dictionary,
        final_numeric_features,
        final_categorical_features,
        final_id_columns,
        final_target_columns,
        final_dropped_leakage_columns,
    )
