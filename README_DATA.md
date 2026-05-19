# Data README: COMM074 FIA Tree Carbon Storage Project

## Original raw data source

The raw dataset used in this project is the USDA Forest Service Forest Inventory and Analysis (FIA) DataMart ENTIRE CSV database.

The full raw FIA CSV archive can be downloaded from:

https://apps.fs.usda.gov/fia/datamart/CSV/CSV_FIADB_ENTIRE.zip

The FIA DataMart page is available at:

https://research.fs.usda.gov/products/dataandtools/fia-datamart

## Why the raw dataset is not included

The full raw FIA CSV database is very large. Following clarification from the module lecturer, the submission includes the processed datasets used for modelling, together with this README file explaining where the original raw data can be obtained.

## Raw files used

The main raw FIA files used in this project were:

- `ENTIRE_TREE.csv`
- `ENTIRE_PLOT.csv`
- `ENTIRE_COND.csv`
- `ENTIRE_REF_SPECIES.csv`
- `ENTIRE_REF_SPECIES_GROUP.csv`
- `ENTIRE_REF_FOREST_TYPE.csv`
- `ENTIRE_REF_FOREST_TYPE_GROUP.csv`
- `ENTIRE_COUNTY.csv`
- `ENTIRE_REF_OWNGRPCD.csv`

These files were originally stored locally in:

`data/raw_entire_fia/`

## Project scope

The project uses a selected-state proof-of-concept subset:

- Alabama (`STATECD = 1`)
- California (`STATECD = 6`)
- Georgia (`STATECD = 13`)
- Maine (`STATECD = 23`)
- Oregon (`STATECD = 41`)
- Washington (`STATECD = 53`)

The unit of analysis is one live individual tree.

## Target variable

The main supervised regression target is:

`CARBON_AG`

This represents individual-tree above-ground carbon storage.

An optional transformed target is also provided:

`CARBON_AG_log = log1p(CARBON_AG)`

## Processed modelling files included

The processed datasets included in the submission are:

- `data/processed/fia_model_data.parquet`
- `data/processed/fia_train.parquet`
- `data/processed/fia_validation.parquet`
- `data/processed/fia_test.parquet`

The train, validation and test files use a 70/15/15 split.

## Important modelling notes

`CARBON_AG` and `CARBON_AG_log` are target variables only and should not be used as predictors.

Biomass, volume and non-target carbon variables such as `DRYBIO_AG`, `DRYBIO_BG`, `CARBON_BG`, `VOLCFNET`, `VOLCSNET`, `VOLBFNET` and `TPA_UNADJ` were removed from the predictor set because they may introduce data leakage.

Some optional predictor columns contain missing values. These should be handled inside each modelling notebook using preprocessing fitted on the training data only.