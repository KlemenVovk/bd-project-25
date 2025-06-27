from utils.constants import TASK1_OUT_ROOT, TASK2_OUT_ROOT, TASK1_FHV_SCHEMA, RESULTS_ROOT
import dask.dataframe as dd
import numpy as np
import os
from glob import glob
from tqdm import tqdm

REASONABLE_MIN_YEAR = 2014
REASONABLE_MAX_YEAR = 2025

# Load FHV parquet files
parquet_files = sorted(glob(os.path.join(TASK1_OUT_ROOT, "fhv_tripdata", "**", "*.parquet")))
print(f"Found {len(parquet_files)} parquet files in {TASK1_OUT_ROOT}/fhv_tripdata")

dfs = [dd.read_parquet(file) for file in parquet_files]
stats_ddfs = []

for i in tqdm(range(len(dfs))):
    df = dfs[i]
    df['year'] = df['pickup_datetime'].dt.year.astype(np.int16)

    # Timestamp cleaning
    valid_pickup = df['pickup_datetime'].notnull()
    reasonable_year = df['pickup_datetime'].dt.year.between(2014, 2025)
    valid_time = valid_pickup & reasonable_year

    # Boolean issue flags
    bad_timestamps = ~valid_time
    clean = valid_time

    # Attach issue flags to minimal frame
    df_year = df[['year']].copy()
    df_year['bad_timestamps'] = bad_timestamps
    df_year['clean'] = clean

    # Dask-native grouped stats (lazy)
    stats_ddfs.append(
        df_year.groupby("year").agg(
            total_trips=("year", "count"),
            bad_timestamps=("bad_timestamps", "sum"),
            clean_trips=("clean", "sum"),
        )
    )

    # Apply final cleaning mask
    clean_mask = valid_time
    dfs[i] = df[clean_mask].reset_index(drop=True)

# Final map-reduce over all files
print("Computing stats...")
combined_stats_ddf = dd.concat(stats_ddfs)
final_stats_ddf = combined_stats_ddf.groupby("year").sum()
final_stats = final_stats_ddf.compute().reset_index().sort_values("year")

# Save stats to CSV
os.makedirs(os.path.join(RESULTS_ROOT), exist_ok=True)
final_stats.to_csv(os.path.join(RESULTS_ROOT, "fhv_tripdata_quality_stats.csv"), index=False)

# Save cleaned parquet files
print("Saving cleaned data...")
os.makedirs(os.path.join(TASK2_OUT_ROOT, "fhv_tripdata"), exist_ok=True)
for i, ddf in enumerate(tqdm(dfs)):
    assert sorted(set(TASK1_FHV_SCHEMA.keys())) == sorted(set(ddf.columns.tolist())), \
        f"Schema mismatch: found {sorted(ddf.columns.tolist())}, expected {sorted(TASK1_FHV_SCHEMA.keys())}"
    ddf = ddf.astype(TASK1_FHV_SCHEMA)
    ddf.to_parquet(
        os.path.join(TASK2_OUT_ROOT, "fhv_tripdata"),
        engine='pyarrow',
        compression='snappy',
        partition_on=['year'],
        write_index=False,
        append=i > 0,
        overwrite=False,
        row_group_size=2_000_000,
    )
