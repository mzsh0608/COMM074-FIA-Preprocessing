"""Extract selected FIA DataMart tables into Parquet files."""

from pathlib import Path

import duckdb
import pandas as pd

from src.paths import OUTPUTS_DIR, PARQUET_DIR, RAW_DIR, create_project_dirs


# Keep the core extracts narrow enough for project work and easy inspection.
PLOT_COLUMNS = [
    "CN",
    "STATECD",
    "UNITCD",
    "COUNTYCD",
    "PLOT",
    "PLOT_STATUS_CD",
    "INVYR",
    "MEASYEAR",
    "MEASMON",
    "MEASDAY",
    "LAT",
    "LON",
    "ELEV",
]

TREE_COLUMNS = [
    "CN",
    "PLT_CN",
    "PREV_TRE_CN",
    "INVYR",
    "STATECD",
    "UNITCD",
    "COUNTYCD",
    "PLOT",
    "SUBP",
    "TREE",
    "CONDID",
    "STATUSCD",
    "SPCD",
    "DIA",
    "DIAHTCD",
    "HT",
    "HTCD",
    "ACTUALHT",
    "CR",
    "CCLCD",
    "CARBON_AG",
    "CARBON_BG",
    "DRYBIO_AG",
    "DRYBIO_BG",
    "VOLCFNET",
    "VOLCSNET",
    "VOLBFNET",
    "TPA_UNADJ",
]

COND_COLUMNS = [
    "CN",
    "PLT_CN",
    "CONDID",
    "COND_STATUS_CD",
    "FORTYPCD",
    "FLDTYPCD",
    "OWNGRPCD",
    "RESERVCD",
    "SITECLCD",
    "STDAGE",
    "STDSZCD",
    "BALIVE",
    "ALSTK",
    "GSSTK",
]

REFERENCE_TABLES = {
    "ENTIRE_REF_SPECIES.csv": [
        "SPCD",
        "COMMON_NAME",
        "SCIENTIFIC_NAME",
        "GENUS",
        "SPECIES",
        "SPECIES_SYMBOL",
    ],
    "ENTIRE_REF_SPECIES_GROUP.csv": [
        "SPGRPCD",
        "NAME",
        "REGION",
        "CLASS",
    ],
    "ENTIRE_REF_FOREST_TYPE.csv": [
        "VALUE",
        "MEANING",
        "TYPGRPCD",
    ],
    "ENTIRE_REF_FOREST_TYPE_GROUP.csv": [
        "VALUE",
        "MEANING",
        "ABBR",
    ],
    "ENTIRE_COUNTY.csv": [
        "CN",
        "STATECD",
        "UNITCD",
        "COUNTYCD",
        "COUNTYNM",
    ],
    "ENTIRE_REF_OWNGRPCD.csv": [
        "OWNGRPCD",
        "MEANING",
    ],
}


def get_available_columns(csv_path):
    """Read only the CSV header and return the available column names."""
    csv_path = Path(csv_path)
    return list(pd.read_csv(csv_path, nrows=0).columns)


def keep_existing_columns(requested_cols, available_cols):
    """Return requested columns that exist, and print any missing columns."""
    available_set = set(available_cols)
    existing_cols = [col for col in requested_cols if col in available_set]
    missing_cols = [col for col in requested_cols if col not in available_set]

    if missing_cols:
        print("Missing columns:", ", ".join(missing_cols))

    return existing_cols


def _sql_path(path):
    """Format a pathlib path for use inside a DuckDB SQL string."""
    return str(Path(path)).replace("\\", "/").replace("'", "''")


def _quote_column(column_name):
    """Quote a column name for DuckDB SQL."""
    escaped_name = column_name.replace('"', '""')
    return f'"{escaped_name}"'


def _select_columns(columns):
    """Build a SQL SELECT list from column names."""
    return ", ".join(_quote_column(column) for column in columns)


def _read_csv_sql(csv_path):
    """Build the DuckDB read_csv_auto call used by all extracts."""
    return f"read_csv_auto('{_sql_path(csv_path)}', header=true)"


def _count_parquet_rows(connection, parquet_path):
    """Count rows in a Parquet file without loading it into pandas."""
    query = f"SELECT COUNT(*) FROM read_parquet('{_sql_path(parquet_path)}')"
    return connection.execute(query).fetchone()[0]


def _extract_with_duckdb(connection, query, output_path, table_name, summary_rows):
    """Run a DuckDB extraction query, save Parquet, print and store row count."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    connection.execute(
        f"COPY ({query}) TO '{_sql_path(output_path)}' (FORMAT PARQUET)"
    )

    row_count = _count_parquet_rows(connection, output_path)
    print(f"{table_name}: {row_count:,} rows")
    summary_rows.append(
        {
            "table_name": table_name,
            "output_file": output_path.name,
            "row_count": row_count,
        }
    )


def _columns_for_file(csv_path, requested_columns):
    """Return requested columns that are present in a CSV file."""
    print(f"Checking columns in {Path(csv_path).name}")
    available_columns = get_available_columns(csv_path)
    columns = keep_existing_columns(requested_columns, available_columns)

    if not columns:
        raise ValueError(f"No requested columns were found in {csv_path}")

    return columns


def extract_core_tables(selected_state_codes):
    """Extract core FIA tables for selected state codes into Parquet files."""
    create_project_dirs()
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    state_codes = [int(code) for code in selected_state_codes]
    if not state_codes:
        raise ValueError("selected_state_codes must contain at least one state code.")

    plot_csv = RAW_DIR / "ENTIRE_PLOT.csv"
    tree_csv = RAW_DIR / "ENTIRE_TREE.csv"
    cond_csv = RAW_DIR / "ENTIRE_COND.csv"
    plot_parquet = PARQUET_DIR / "plot_selected.parquet"
    tree_parquet = PARQUET_DIR / "tree_selected.parquet"
    cond_parquet = PARQUET_DIR / "cond_selected.parquet"

    summary_rows = []

    with duckdb.connect() as connection:
        # Register selected state codes as a tiny in-memory table.
        state_codes_df = pd.DataFrame({"STATECD": state_codes})
        connection.register("selected_states", state_codes_df)

        plot_columns = _columns_for_file(plot_csv, PLOT_COLUMNS)
        if "CN" not in plot_columns or "STATECD" not in plot_columns:
            raise ValueError("ENTIRE_PLOT.csv must contain CN and STATECD.")

        plot_query = f"""
            SELECT {_select_columns(plot_columns)}
            FROM {_read_csv_sql(plot_csv)}
            WHERE STATECD IN (SELECT STATECD FROM selected_states)
        """
        _extract_with_duckdb(
            connection,
            plot_query,
            plot_parquet,
            "plot_selected",
            summary_rows,
        )

        tree_columns = _columns_for_file(tree_csv, TREE_COLUMNS)
        tree_available_columns = get_available_columns(tree_csv)
        if "CR" in tree_available_columns and "CR" not in tree_columns:
            tree_columns.append("CR")

        if "PLT_CN" not in tree_columns:
            raise ValueError("ENTIRE_TREE.csv must contain PLT_CN.")

        tree_query = f"""
            SELECT {_select_columns(tree_columns)}
            FROM {_read_csv_sql(tree_csv)}
            WHERE PLT_CN IN (
                SELECT CN FROM read_parquet('{_sql_path(plot_parquet)}')
            )
        """
        _extract_with_duckdb(
            connection,
            tree_query,
            tree_parquet,
            "tree_selected",
            summary_rows,
        )

        cond_columns = _columns_for_file(cond_csv, COND_COLUMNS)
        cond_available_columns = get_available_columns(cond_csv)
        for column in ["STDSZCD", "OWNGRPCD"]:
            if column in cond_available_columns and column not in cond_columns:
                cond_columns.append(column)

        if "PLT_CN" not in cond_columns:
            raise ValueError("ENTIRE_COND.csv must contain PLT_CN.")

        cond_query = f"""
            SELECT {_select_columns(cond_columns)}
            FROM {_read_csv_sql(cond_csv)}
            WHERE PLT_CN IN (
                SELECT CN FROM read_parquet('{_sql_path(plot_parquet)}')
            )
        """
        _extract_with_duckdb(
            connection,
            cond_query,
            cond_parquet,
            "cond_selected",
            summary_rows,
        )

        for file_name, requested_columns in REFERENCE_TABLES.items():
            csv_path = RAW_DIR / file_name
            table_name = file_name.removeprefix("ENTIRE_").removesuffix(".csv").lower()
            output_path = PARQUET_DIR / f"{table_name}.parquet"
            columns = _columns_for_file(csv_path, requested_columns)

            query = f"""
                SELECT {_select_columns(columns)}
                FROM {_read_csv_sql(csv_path)}
            """
            _extract_with_duckdb(
                connection,
                query,
                output_path,
                table_name,
                summary_rows,
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUTS_DIR / "extraction_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Extraction summary saved to {summary_path}")

    return summary
