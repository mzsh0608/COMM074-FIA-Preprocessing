"""Create lightweight inventories for FIA DataMart CSV extracts."""

from pathlib import Path

import pandas as pd

from src.paths import INVENTORY_DIR, RAW_DIR


def _read_csv_header(csv_path: Path) -> list[str]:
    """Read only the header row from a CSV file."""
    return list(pd.read_csv(csv_path, nrows=0).columns)


def create_file_inventory(raw_dir, output_path):
    """Scan ENTIRE_*.csv files and save a file-level inventory."""
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)

    records = []
    for csv_path in sorted(raw_dir.glob("ENTIRE_*.csv")):
        columns = _read_csv_header(csv_path)

        records.append(
            {
                "file_name": csv_path.name,
                "file_size_mb": csv_path.stat().st_size / (1024 * 1024),
                "n_columns": len(columns),
                "columns_preview": ", ".join(columns[:15]),
            }
        )

    inventory = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_path, index=False)

    return inventory


def create_table_columns_file(raw_dir, output_path):
    """Create a long-format file listing every column in each FIA CSV."""
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)

    records = []
    for csv_path in sorted(raw_dir.glob("ENTIRE_*.csv")):
        columns = _read_csv_header(csv_path)

        for position, column_name in enumerate(columns, start=1):
            records.append(
                {
                    "file_name": csv_path.name,
                    "column_position": position,
                    "column_name": column_name,
                }
            )

    table_columns = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_columns.to_csv(output_path, index=False)

    return table_columns


if __name__ == "__main__":
    create_file_inventory(RAW_DIR, INVENTORY_DIR / "file_inventory.csv")
    create_table_columns_file(RAW_DIR, INVENTORY_DIR / "table_columns.csv")
