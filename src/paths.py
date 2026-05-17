"""Project paths for the COMM074 FIA preprocessing workflow."""

from pathlib import Path


# Project root is one level above this src/ directory.
ROOT_DIR = Path(__file__).resolve().parents[1]

# Top-level project folders.
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"
OUTPUTS_DIR = ROOT_DIR / "outputs"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
SRC_DIR = ROOT_DIR / "src"

# Data subfolders used throughout preprocessing.
RAW_DIR = DATA_DIR / "raw_entire_fia"
INVENTORY_DIR = DATA_DIR / "inventory"
PARQUET_DIR = DATA_DIR / "parquet"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"


def create_project_dirs() -> None:
    """Create project folders that are generated or managed by the project."""
    dirs_to_create = [
        DATA_DIR,
        INVENTORY_DIR,
        PARQUET_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        FIGURES_DIR,
        OUTPUTS_DIR,
        NOTEBOOKS_DIR,
        SRC_DIR,
    ]

    # RAW_DIR is intentionally skipped because it should contain the original
    # raw FIA download and should be managed manually.
    for path in dirs_to_create:
        path.mkdir(parents=True, exist_ok=True)
