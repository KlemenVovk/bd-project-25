import dask.dataframe as dd
import os
from paths import ROOT, TASK1_SCHEMA, SELECTED_COLUMNS, CONSISTENCY_COL_MAPPING, YEARS, TASK1_PARQUET_OUT_ROOT, get_files
# We checked, all 2020-2024 files have the same columns (albeit, some are named differently -lower/upper case, etc.)
# Go through all parquet files, rername columns according to consistency mapping.
# Based on those, we defined a SCHEMA for all tables in TASK1_SCHEMA
files = sorted(sum((get_files(ROOT, year) for year in YEARS), start=[]))
dfs = [dd.read_parquet(file) for file in files]
for i, file in enumerate(files):
    # display first and last 5 rows
    for old_col, new_col in CONSISTENCY_COL_MAPPING.items():
        if old_col in dfs[i].columns:
            dfs[i] = dfs[i].rename(columns={old_col: new_col})
    found_cols = sorted(dfs[i].columns.tolist())
    expected_cols = sorted(SELECTED_COLUMNS)
    assert found_cols == expected_cols, f"Columns in {file} do not match expected columns. Found: {found_cols}, Expected: {expected_cols}"

# Add a year column to each dataframe for partitioning (based on filename)
for i, file in enumerate(files):
    dfs[i]["year_from_filename"] = int(os.path.basename(file).split("_")[2][:4])

# Concatenate, partition and save
df = dd.concat(dfs, axis=0)

# to csv

# df.to_parquet(
#     TASK1_PARQUET_OUT_ROOT,
#     engine="pyarrow",
#     compression="snappy",
#     schema=TASK1_SCHEMA,
#     partition_on=["year_from_filename"],
# )